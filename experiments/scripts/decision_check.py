"""Review response C (PLAN.md, 2026-10-23; exploratory): what the choice of prediction set does to a decision.

Sea ice, ice-edge threshold 0.15. From a prediction set S (restricted to [0,1]): "ice-free" if S lies in [0, 0.15),
"ice" if S lies in [0.15, 1], otherwise "undetermined". On boundary test units (observation grouping, eps = 0.05):
decisive fraction, and error rate among decisive units (the outcome lies on the other side of the threshold).
  (i)  Gaussian vs Kumaraswamy latent, TN-hetero, seeds 0-9: Mondrian residual intervals (S = 128 posterior samples,
       as in the counterfactual swap), clipped to [0,1].
  (ii) Sets A, B, C, E4 of the atom-aware experiment on its sea-ice checkpoints, seeds 0-7, with the selection rules
       of o1.py unchanged (tau for C chosen with the preregistered atom cost).
Writes results/decision_check.json.
"""
import argparse
import json
import statistics as st
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import counterfactual as cf  # noqa: E402
import m0_one  # noqa: E402
import o1  # noqa: E402
from scrl import o1_eval as oe  # noqa: E402
from scrl.conformal import conformal_quantile  # noqa: E402
from scrl.data_synthetic import boundary_distance  # noqa: E402
from scrl.misco_eval import _interior_masses, group_by_obs  # noqa: E402

THR, EPS, ALPHA = 0.15, 0.05, 0.1


def decide(low, high, y):
    """low/high: set meets [0, THR) / [THR, 1]. Returns decisive fraction and error rate among decisive units."""
    decisive = low ^ high
    say_free = low & ~high
    wrong = (say_free & (y >= THR)) | (~low & high & (y < THR))
    n_dec = int(decisive.sum())
    return {"decisive": n_dec / len(y), "error": (int(wrong.sum()) / n_dec) if n_dec else float("nan")}


@torch.no_grad()
def latent_part():
    a = argparse.Namespace(dataset="seaice", data_seed=0, smoke=False, seed=0, pb=0.4, post_std=0.05, level=0)
    import run_synthetic
    data, name, d, K, obs_noise = run_synthetic.load_data(a)
    cal, test = data["cal"], data["test"]
    out = {"gaussian": [], "kumaraswamy": []}
    for seed in range(10):
        for fam in out:
            M = cf.load_model(name, fam, d, cal["x"].shape[-1], K, obs_noise, seed)
            ev = {}
            for sp, D in (("cal", cal), ("test", test)):
                torch.manual_seed(1000 * seed + {"cal": 0, "test": 1}[sp])
                pred = M.predict_from_samples(M.family.rsample(M.encode(D["x"], D["s"]), cf.S))
                ev[sp] = (pred["y_hat"].view(-1), pred["sigma"].view(-1), D["y"].view(-1),
                          boundary_distance(D["x"][:, -1, :d]) < EPS)
            yh, sg, y, g = ev["cal"]
            q = {b: conformal_quantile(((y - yh).abs() / sg)[g == b], ALPHA) for b in (True, False)}
            yh, sg, y, g = ev["test"]
            qt = torch.where(g, torch.tensor(q[True]), torch.tensor(q[False]))
            lo, hi = (yh - qt * sg).clamp(0, 1), (yh + qt * sg).clamp(0, 1)
            r = decide((lo < THR)[g], (hi >= THR)[g], y[g])
            r["seed"] = seed
            out[fam].append(r)
    return out


def hpd_sides(pred, head, q, tau=None):
    """Whether the HPD set (atoms + grid cells with score <= q) meets [0, THR) and [THR, 1]."""
    G = oe.G
    cm, a0, a1 = _interior_masses(pred, head, G)
    if head != "atom_tn":
        a0, a1 = torch.zeros_like(a0), torch.zeros_like(a1)
    act0 = a0 > 0 if tau is None else a0 >= tau
    act1 = a1 > 0 if tau is None else a1 >= tau
    if head != "atom_tn":
        act0 = act1 = torch.zeros_like(a0, dtype=torch.bool)
    m0, m1 = a0 * act0, a1 * act1
    first0 = m0 >= m1
    inf = torch.full_like(a0, float("inf"))
    s0 = torch.where(act0, torch.where(first0, torch.zeros_like(a0), m1), inf)
    s1 = torch.where(act1, torch.where(first0 & act0, m0, torch.zeros_like(a1)), inf)
    order = torch.argsort(cm, -1, descending=True)
    sorted_m = cm.gather(-1, order)
    s_sorted = (m0 + m1).unsqueeze(-1) + torch.cumsum(sorted_m, -1) - sorted_m
    s_cell = torch.empty_like(s_sorted).scatter_(-1, order, s_sorted)          # same formula as o1_eval.hpd
    inc = s_cell <= q.unsqueeze(-1)
    centers = (torch.arange(G, device=inc.device) + 0.5) / G
    low = (inc & (centers < THR)).any(-1) | (s0 <= q)
    high = (inc & (centers >= THR)).any(-1) | (s1 <= q)
    return low, high, inc.sum(-1).float() / G


@torch.no_grad()
def atom_part():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = {m: [] for m in ("A", "B", "C", "E4")}
    for seed in o1.SEEDS:
        data, d, K, obs_noise = m0_one.load("seaice", seed)
        n_tr = data["train"]["x"].shape[0]
        perm = torch.randperm(n_tr, generator=torch.Generator().manual_seed(seed))
        val_idx = perm[: n_tr // 10]
        val = {k: v[val_idx].to(dev) for k, v in data["train"].items() if torch.is_tensor(v)}
        cal = {k: v.to(dev) for k, v in data["cal"].items() if torch.is_tensor(v)}
        test = {k: v.to(dev) for k, v in data["test"].items() if torch.is_tensor(v)}
        gv, gc, gt = (group_by_obs(D["x"], d, 0.05) for D in (val, cal, test))
        n_obs = cal["x"].shape[-1]
        M1 = o1.load_model("seaice", 1, seed, d, n_obs, K, obs_noise, n_tr, dev)
        M3 = o1.load_model("seaice", 3, seed, d, n_obs, K, obs_noise, n_tr, dev)
        P1 = {sp: o1.predictive(M1, D, seed) for sp, D in (("cal", cal), ("test", test))}
        P3 = {sp: o1.predictive(M3, D, seed) for sp, D in (("val", val), ("cal", cal), ("test", test))}
        yc, yt = cal["y"], test["y"]
        b = gt == 1
        y = yt.view(-1)
        tau = o1.select_tau(P3["val"], val["y"], gv, seed, o1.LAM_PRIMARY)
        for m, P, head, t in (("A", P1, "tn", None), ("B", P3, "atom_tn", None), ("C", P3, "atom_tn", tau)):
            s_c, _ = oe.hpd(P["cal"], head, yc, t)
            s_t, _ = oe.hpd(P["test"], head, yt, t)
            q = o1.mondrian(s_c, gc, s_t, gt)
            low, high, mu = hpd_sides(P["test"], head, q, t)
            r = decide(low[b].cpu(), high[b].cpu(), y[b].cpu())
            r["seed"] = seed
            r["mu_check"] = mu[b].mean().item()       # must equal the recorded mu of o1.json
            out[m].append(r)
        # E4: CPCI per observation group, exactly as in o1.py
        pv, fv = oe.cpci_parts(P3["val"])
        pc, fc = oe.cpci_parts(P3["cal"])
        pt, ft = oe.cpci_parts(P3["test"])
        cperm = torch.randperm(yc.shape[0], generator=torch.Generator().manual_seed(200 + seed)).to(dev)
        c1 = torch.zeros(yc.shape[0], dtype=torch.bool, device=dev)
        c1[cperm[: yc.shape[0] // 2]] = True
        low = torch.zeros_like(ft, dtype=torch.bool)
        high = torch.zeros_like(low)
        for g in (0, 1):
            vm, cm1, cm2, tm = gv == g, (gc == g) & c1, (gc == g) & ~c1, gt == g
            a_r, qq, *_ = oe.cpci_fit(pv[vm].cpu(), fv[vm].cpu(), val["y"].view(-1)[vm].cpu(),
                                      pc[cm1].cpu(), pc[cm2].cpu(), fc[cm2].cpu(), yc.view(-1)[cm2].cpu())
            zero = pt[tm] <= a_r
            lo, hi = (ft[tm] - qq).clamp(0, 1), (ft[tm] + qq).clamp(0, 1)
            low[tm] = zero | (lo < THR)
            high[tm] = ~zero & (hi >= THR)
        r = decide(low[b].cpu(), high[b].cpu(), y[b].cpu())
        r["seed"] = seed
        out["E4"].append(r)
        print(f"seed {seed}: " + " ".join(f"{m}={v[-1]['decisive']:.3f}/{v[-1]['error']:.3f}" for m, v in out.items()),
              flush=True)
    return out


def main():
    res = {"latent": latent_part(), "atoms": atom_part()}
    rec = {(r["seed"]): r["methods"] for r in json.loads((ROOT / "results" / "o1.json").read_text())
           if r["dataset"] == "seaice"}
    worst = max(abs(r["mu_check"] - rec[r["seed"]][m]["mu"]) for m in ("A", "B", "C") for r in res["atoms"][m])
    print(f"set reconstruction check: largest |mu - recorded mu| = {worst:.2e}")
    res["reconstruction_max_abs_diff"] = worst
    for part in ("latent", "atoms"):
        d = res[part]
        for m, rows in d.items():
            print(f"{part:7s} {m:12s} decisive {st.mean(r['decisive'] for r in rows):.3f}  "
                  f"error {st.mean(r['error'] for r in rows):.3f}")
    (ROOT / "results" / "decision_check.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
