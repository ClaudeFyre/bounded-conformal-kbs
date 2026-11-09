"""Temporal extrapolation check on sea ice (review response E). PREREGISTRATION: everything in this
docstring, and the code below, is committed before the first run. Results are reported whatever they are.

Question. Calibration and test units in the main protocol are drawn i.i.d. from one evaluation pool
(pool blocks, 2010-2019), which makes split conformal valid by construction. Does validity, and does
the comparison between latent families, survive when the test period lies after the calibration period?

Why only coverage is at stake. Interval width = 2 x calibration quantile x mean predicted scale of the
test units; it never uses test labels. Moving the test period therefore changes widths only through the
test inputs, and the width results below are NOT a temporal test of the paper's width claims; they are
reported as a replication with a different calibration sample. What the temporal split can change is
coverage, and the one outcome that would revise the paper's main line is a DIFFERENTIAL loss of
coverage between the Gaussian latent and a bounded latent or the post-hoc truncation: that would mean
bounds matter for validity under drift, a dimension the paper makes no claim about.

Design (no training; existing checkpoints of the main five-dataset experiment).
  Data: NOAA/NSIDC CDR v6 Hudson Bay, as in scrl/data_seaice.py. The spatial train/pool block split is
    reproduced exactly from data_seed = 0 (the split the checkpoints were trained on); this is asserted
    against the cached main split. Units are drawn from POOL blocks only, windows lying entirely in the
    stated years, with a fresh generator per seed (numpy default_rng(1000 + seed)).
  Arms, per seed s = 0..9 (checkpoint seed s):
    calibration: n_cal = 2000 units, years 2010-2014                         (shared by both arms)
    control  C : n_test = 2000 units, years 2010-2014 (exchangeable with calibration)
    temporal T : n_test = 2000 units, years 2015-2019 (strictly after calibration)
  Methods: Gaussian latent (G); bounded latents Kumaraswamy (K), truncated Gaussian, logit-normal, Beta;
    post-hoc truncation of G to the support (q*, same decoder; as combination (e) of counterfactual.py).
    All use 128 posterior samples with torch seeds fixed per seed and split, the tn_hetero decoder, and
    Mondrian split conformal with observation grouping, eps = 0.05, alpha = 0.1 (scrl/conformal.py).

Primary criterion (P1, differential validity under drift). For method M in {K, q*}, per seed:
    DiD_M = [cov_b(M,T) - cov_b(M,C)] - [cov_b(G,T) - cov_b(G,C)],
  cov_b = coverage on the boundary group. Differential failure for M is declared if |mean DiD_M| > 0.02
  AND the two-sided Wilcoxon signed-rank test over the 10 seeds gives p < 0.05.
  Consequence fixed in advance: if P1 holds for K or q*, the Discussion must state that the support
  choice affected validity under temporal drift on sea ice, and the main-line summary is revised; if it
  holds for neither, the paper states that the comparison was not differentially affected.
Secondary (descriptive): the same DiD for the other three bounded families; marginal and boundary
  coverage of every method in each arm, and whether it lies in the validity band 0.90 +- 0.02 (mean over
  seeds); if all methods leave the band in T by similar amounts, this is reported as a scope limitation of
  split conformal under drift, not as a result about bounds.
Not temporal (labelled as such): in arm T, the sign of rho_W (K against G, raw boundary width, mean over
  seeds) and the number of seeds in which q* is wider than G.
Drift descriptor: mean target and boundary-group share in each period (to show whether there is drift).

    python scripts/temporal_check.py     # writes results/temporal_check.json
"""
import argparse
import datetime as dt
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scrl import data_seaice, tn  # noqa: E402
from scrl.conformal import mondrian_eval  # noqa: E402
from counterfactual import load_model, evaluate  # noqa: E402

FAMILIES = ["kumaraswamy", "trunc_gaussian", "logit_normal", "beta"]
SEEDS = range(10)
EPS, ALPHA, S = 0.05, 0.1, 128
N_CAL, N_TEST = 2000, 2000
CAL_YEARS, C_YEARS, T_YEARS = (2010, 2014), (2010, 2014), (2015, 2019)
BAND, DID_MIN, P_MAX = 0.02, 0.02, 0.05
OUT = ROOT / "results" / "temporal_check.json"


def build_pool():
    """Reproduce data_seaice.generate up to the block split (data_seed = 0)."""
    cfg = data_seaice.SeaIceConfig(data_seed=0)
    dates, conc, sd = data_seaice.load_raw(ROOT / "raw_data" / "seaice")
    N = conc.shape[0]
    valid = np.isfinite(conc).mean(0) > 0.98
    cells = np.argwhere(valid)
    gen = np.random.default_rng(cfg.data_seed)
    blocks = sorted({(i // cfg.block, j // cfg.block) for i, j in cells})
    gen.shuffle(blocks)
    n_tr = int(round(cfg.train_block_frac * len(blocks)))
    train_blocks = set(blocks[:n_tr])
    in_train = np.array([(i // cfg.block, j // cfg.block) in train_blocks for i, j in cells])
    years = np.array([d.year for d in dates])
    season = torch.tensor([data_seaice._season(d.month) for d in dates])
    series = np.nan_to_num(conc[:, cells[:, 0], cells[:, 1]], nan=np.nan)
    for k in range(series.shape[1]):
        col = series[:, k]
        mask = np.isnan(col)
        if mask.any():
            idx = np.where(~mask, np.arange(N), 0)
            np.maximum.accumulate(idx, out=idx)
            series[:, k] = col[idx]
    series = torch.from_numpy(np.clip(series, 0, 1))
    obs_noise = float(np.nanmedian(sd[:, cells[:, 0], cells[:, 1]]))
    return cfg, series, years, season, in_train, obs_noise, N


def make(cfg, series, years, season, N, cell_mask, yr, n, gen):
    cand_cells = np.where(cell_mask)[0]
    L = cfg.T + cfg.horizon
    starts_ok = np.where((years >= yr[0]) & (years <= yr[1]))[0]
    starts_ok = starts_ok[starts_ok + L <= N]
    starts_ok = starts_ok[years[starts_ok + L - 1] <= yr[1]]
    ci = gen.choice(cand_cells, n)
    stt = gen.choice(starts_ok, n)
    idx = torch.from_numpy(stt)[:, None] + torch.arange(cfg.T)[None]
    x = series[idx, torch.from_numpy(ci)[:, None]].unsqueeze(-1).float()
    y = series[torch.from_numpy(stt + L - 1), torch.from_numpy(ci)].unsqueeze(-1).float()
    return {"x": x, "s": season[idx], "y": y, "cell": torch.from_numpy(ci)}


@torch.no_grad()
def predict(model, split, kind, torch_seed):
    """kind: 'own' (model's own posterior) or 'star' (Gaussian posterior truncated to [0,1]^d)."""
    p = model.encode(split["x"], split["s"])
    torch.manual_seed(torch_seed)
    if kind == "own":
        hs = model.family.rsample(p, S)
    else:
        mu, sg = p["mu"], p["sigma"]
        hs = tn.rsample(mu.expand(S, *mu.shape), sg.expand(S, *sg.shape), torch.rand(S, *mu.shape))
    return evaluate(split, hs, model)


def main():
    cfg, series, years, season, in_train, obs_noise, N = build_pool()
    # check: the reproduced pool contains every cell of the cached main calibration/test split
    import run_synthetic
    a = argparse.Namespace(dataset="seaice", data_seed=0, smoke=False, seed=0, pb=0.4, post_std=0.05)
    main_data, name, d, K, main_noise = run_synthetic.load_data(a)
    pool_idx = set(np.where(~in_train)[0].tolist())
    assert set(main_data["cal"]["cell"].tolist()) <= pool_idx and set(main_data["test"]["cell"].tolist()) <= pool_idx
    assert abs(obs_noise - main_noise) < 1e-9
    n_obs = main_data["cal"]["x"].shape[-1]

    rows, drift = [], {}
    for seed in SEEDS:
        gen = np.random.default_rng(1000 + seed)
        cal = make(cfg, series, years, season, N, ~in_train, CAL_YEARS, N_CAL, gen)
        tests = {"C": make(cfg, series, years, season, N, ~in_train, C_YEARS, N_TEST, gen),
                 "T": make(cfg, series, years, season, N, ~in_train, T_YEARS, N_TEST, gen)}
        if seed == 0:
            for arm, D in (("cal", cal), *tests.items()):
                xl = D["x"][:, -1, 0]
                drift[arm] = {"mean_y": float(D["y"].mean()),
                              "boundary_share": float(((xl < EPS) | (xl > 1 - EPS)).float().mean())}
        G = load_model(name, "gaussian", d, n_obs, K, obs_noise, seed)
        models = {"gaussian": (G, "own"), "trunc_star": (G, "star")}
        for fam in FAMILIES:
            models[fam] = (load_model(name, fam, d, n_obs, K, obs_noise, seed), "own")
        row = {"seed": seed}
        for mname, (m, kind) in models.items():
            ev_cal = predict(m, cal, kind, 10_000 * seed + 0)
            for arm, D in tests.items():
                r = mondrian_eval(ev_cal, predict(m, D, kind, 10_000 * seed + {"C": 1, "T": 2}[arm]),
                                  EPS, ALPHA, "x_last")
                row[f"{mname}_{arm}"] = {k: r[k] for k in ("coverage", "coverage_boundary", "biw",
                                                           "n_test_boundary", "q_boundary")}
        rows.append(row)
        print(f"s{seed}: " + " ".join(f"{m}:C={row[m + '_C']['coverage_boundary']:.3f}/T={row[m + '_T']['coverage_boundary']:.3f}"
                                      for m in models), flush=True)

    methods = ["gaussian", "trunc_star", *FAMILIES]
    cov = {m: {arm: {k: st.mean(r[f"{m}_{arm}"][k] for r in rows) for k in ("coverage", "coverage_boundary")}
               for arm in ("C", "T")} for m in methods}
    for m in methods:
        for arm in ("C", "T"):
            cov[m][arm]["in_band_marginal"] = bool(abs(cov[m][arm]["coverage"] - 0.9) <= BAND)
            cov[m][arm]["in_band_boundary"] = bool(abs(cov[m][arm]["coverage_boundary"] - 0.9) <= BAND)

    def did(m):
        return [(r[f"{m}_T"]["coverage_boundary"] - r[f"{m}_C"]["coverage_boundary"])
                - (r["gaussian_T"]["coverage_boundary"] - r["gaussian_C"]["coverage_boundary"]) for r in rows]

    dids = {}
    for m in ["kumaraswamy", "trunc_star", "trunc_gaussian", "logit_normal", "beta"]:
        v = did(m)
        p = float(wilcoxon(v).pvalue) if any(abs(x) > 0 for x in v) else 1.0
        dids[m] = {"did": v, "mean": float(st.mean(v)), "p": p,
                   "differential_failure": bool(abs(st.mean(v)) > DID_MIN and p < P_MAX),
                   "role": "primary" if m in ("kumaraswamy", "trunc_star") else "secondary"}
    rho = [100 * (r["gaussian_T"]["biw"] - r["kumaraswamy_T"]["biw"]) / r["gaussian_T"]["biw"] for r in rows]
    nontemporal = {"rho_W_kum_T_mean": float(st.mean(rho)), "rho_W_kum_T": rho,
                   "trunc_wider_T": int(sum(r["trunc_star_T"]["biw"] > r["gaussian_T"]["biw"] for r in rows)),
                   "n": len(rows)}
    out = {"design": {"cal_years": CAL_YEARS, "control_years": C_YEARS, "temporal_years": T_YEARS,
                      "n_cal": N_CAL, "n_test": N_TEST, "S": S, "eps": EPS, "alpha": ALPHA,
                      "did_min": DID_MIN, "p_max": P_MAX, "band": BAND},
           "drift": drift, "coverage": cov, "did": dids, "nontemporal": nontemporal,
           "P1_any_differential_failure": bool(dids["kumaraswamy"]["differential_failure"]
                                               or dids["trunc_star"]["differential_failure"]),
           "rows": rows}
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("drift", "coverage", "nontemporal", "P1_any_differential_failure")}, indent=1))
    print({m: (round(v["mean"], 4), round(v["p"], 4), v["differential_failure"]) for m, v in dids.items()})


if __name__ == "__main__":
    main()
