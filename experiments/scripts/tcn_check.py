"""Review response A (pre-registered in PLAN.md, 2026-10-23): analysis of the TCN backbone runs.

Primary: rho_W = 100 (W_G - W_K) / W_G per seed (raw width, observation grouping, eps = 0.05), mean over seeds,
paired two-sided Wilcoxon. A dataset reproduces the GRU result iff the mean has the GRU sign and p < 0.05
(GRU signs: sea ice +, Bucket -, ResOpsUS -). "The reversal does not depend on the GRU" iff Bucket and ResOpsUS both
reproduce. Secondary: post-hoc truncation of the TCN Gaussian models (same procedure as the counterfactual swap,
S = 128): number of (dataset, seed) pairs that widen. Comparisons with marginal coverage outside 0.90 +/- 0.02 are
flagged invalid. Writes results/tcn_check.json and prints a report.
"""
import argparse
import json
import statistics as st
import sys
from pathlib import Path

import torch
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scrl import tn  # noqa: E402
from scrl.conformal import mondrian_eval  # noqa: E402
from scrl.models import Model  # noqa: E402

DATASETS = {"bucket": -1, "resops": -1, "seaice": +1}      # GRU sign of rho_W
SEEDS = range(10)
EPS, ALPHA, S = 0.05, 0.1, 128


def res(ds, fam, seed):
    r = json.loads((ROOT / "results" / f"{ds}_tcn" / f"{fam}__tn_hetero__raw__s{seed}.json").read_text())
    c = next(c for c in r["conformal"] if abs(c["obs"]["eps"] - EPS) < 1e-9)["obs"]
    return c["biw"], c["coverage"]


@torch.no_grad()
def truncation(ds, seed):
    import run_synthetic
    a = argparse.Namespace(dataset=ds, data_seed=0, smoke=False, seed=0, pb=0.4, post_std=0.05, level=0)
    data, name, d, K, obs_noise = run_synthetic.load_data(a)
    cal, test = data["cal"], data["test"]
    G = Model("gaussian", d=d, n_obs=cal["x"].shape[-1], K=K, decoder="tn_hetero", obs_noise=obs_noise,
              backbone="tcn")
    G.load_state_dict(torch.load(ROOT / "checkpoints" / f"{name}_tcn" / f"gaussian__tn_hetero__raw__s{seed}.pt",
                                 map_location="cpu", weights_only=True))
    G.eval()
    W = {}
    for combo in ("a", "e"):
        ev = {}
        for sp, D in (("cal", cal), ("test", test)):
            p = G.encode(D["x"], D["s"])
            torch.manual_seed(1000 * seed + {"cal": 0, "test": 1}[sp])
            if combo == "a":
                hs = G.family.rsample(p, S)
            else:
                mu, sg = p["mu"], p["sigma"]
                hs = tn.rsample(mu.expand(S, *mu.shape), sg.expand(S, *sg.shape), torch.rand(S, *mu.shape))
            pred = G.predict_from_samples(hs)
            ev[sp] = {"y": D["y"], "y_hat": pred["y_hat"], "sigma": pred["sigma"], "x_last": D["x"][:, -1, :d]}
        r = mondrian_eval(ev["cal"], ev["test"], EPS, ALPHA, "x_last")
        W[combo] = (r["biw"], r["coverage"])
    return W


def main():
    out = {}
    for ds, gru_sign in DATASETS.items():
        rho, covs = [], []
        for s in SEEDS:
            (wg, cg), (wk, ck) = res(ds, "gaussian", s), res(ds, "kumaraswamy", s)
            rho.append(100 * (wg - wk) / wg)
            covs += [cg, ck]
        p = wilcoxon(rho).pvalue
        mean = st.mean(rho)
        valid = all(abs(c - 0.90) <= 0.02 for c in covs)
        reproduced = bool(valid and (mean > 0) == (gru_sign > 0) and p < 0.05)
        trunc = [truncation(ds, s) for s in SEEDS]
        wider = sum(t["e"][0] > t["a"][0] for t in trunc)
        out[ds] = {"rho": rho, "rho_mean": mean, "rho_sd": st.stdev(rho), "p": float(p), "coverage_min": min(covs),
                   "coverage_max": max(covs), "valid": valid, "reproduced": reproduced,
                   "trunc_wider": wider, "trunc_n": len(trunc),
                   "trunc_cov": [t["e"][1] for t in trunc] + [t["a"][1] for t in trunc]}
        print(f"{ds:7s} rho_W {mean:+.1f}% (sd {st.stdev(rho):.1f}), p = {p:.4f}, coverage {min(covs):.3f}-{max(covs):.3f}"
              f" -> {'REPRODUCED' if reproduced else 'not reproduced'}; truncation wider {wider}/{len(trunc)}", flush=True)
    both = out["bucket"]["reproduced"] and out["resops"]["reproduced"]
    out["reversal_independent_of_gru"] = both
    print(f"reversal does not depend on the GRU (Bucket and ResOpsUS both reproduced): {both}")
    (ROOT / "results" / "tcn_check.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
