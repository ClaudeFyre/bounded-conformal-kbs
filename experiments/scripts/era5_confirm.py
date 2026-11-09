"""ERA5 cloud cover: held-out confirmation (protocol pre-registered in PLAN.md; run once).

Trains models 1 (TN), 2 (CQR) and 3 (atom head) for 8 seeds, evaluates A, B, C, E1, E2, E4, E5 on the
same checkpoints, and reports the pre-registered comparisons with means and medians.
"""
import itertools
import json
import statistics as st
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import m0_one  # noqa: E402
import o1  # noqa: E402
from e5_dev import e5_grouped  # noqa: E402
from scrl import o1_eval as oe  # noqa: E402
from scrl.misco_eval import group_by_obs  # noqa: E402

DS = "era5_cloud"
SEEDS = range(8)
OUT = ROOT / "results" / "era5_confirm.json"
LAMS = oe.LAMBDA_GRID


def train():
    from run_phase import run_one
    for s in SEEDS:
        m0_one.load(DS, s)                          # sequential cache warm-up
    cmds = [[sys.executable, str(ROOT / "scripts" / "m0_one.py"), "--dataset", DS, "--model", str(m), "--seed", str(s),
             "--out_tag", o1.TAG, "--save_ckpt"]
            for m, s in itertools.product([1, 2, 3], SEEDS)
            if not (ROOT / "results" / o1.TAG / DS / f"model{m}__s{s}.json").exists()]
    with ThreadPoolExecutor(4) as pool:
        list(pool.map(run_one, cmds))


def evaluate():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for seed in SEEDS:
        data, d, K, obs_noise = m0_one.load(DS, seed)
        n_tr = data["train"]["x"].shape[0]
        perm = torch.randperm(n_tr, generator=torch.Generator().manual_seed(seed))
        val = {k: v[perm[: n_tr // 10]].to(dev) for k, v in data["train"].items() if torch.is_tensor(v)}
        cal = {k: v.to(dev) for k, v in data["cal"].items() if torch.is_tensor(v)}
        test = {k: v.to(dev) for k, v in data["test"].items() if torch.is_tensor(v)}
        gv, gc, gt = (group_by_obs(D["x"], d, 0.05) for D in (val, cal, test))
        n_obs = cal["x"].shape[-1]
        M1 = o1.load_model(DS, 1, seed, d, n_obs, K, obs_noise, n_tr, dev)
        M3 = o1.load_model(DS, 3, seed, d, n_obs, K, obs_noise, n_tr, dev)
        P1 = {sp: o1.predictive(M1, D, seed) for sp, D in (("cal", cal), ("test", test))}
        P3 = {sp: o1.predictive(M3, D, seed) for sp, D in (("val", val), ("cal", cal), ("test", test))}
        yc, yt = cal["y"], test["y"]
        M = {}
        M["A"] = o1.hpd_method(P1["cal"], P1["test"], "tn", yc, yt, gc, gt)
        M["B"] = o1.hpd_method(P3["cal"], P3["test"], "atom_tn", yc, yt, gc, gt)
        tau = o1.select_tau(P3["val"], val["y"], gv, seed, o1.LAM_PRIMARY)
        M["C"] = o1.hpd_method(P3["cal"], P3["test"], "atom_tn", yc, yt, gc, gt, tau)
        M["C"]["tau"] = tau
        s_c, _ = oe.residual(P1["cal"], yc)
        s_t, aux = oe.residual(P1["test"], yt)
        q = o1.mondrian(s_c, gc, s_t, gt)
        mu, i0, i1 = oe.residual_clipped_measure(aux, q)
        M["E1"] = o1.summarise(mu, i0, i1, s_t <= q, yt, gt)
        # E4 (CPCI), per observation group
        pv, fv = oe.cpci_parts(P3["val"]); pc, fc = oe.cpci_parts(P3["cal"]); pt, ft = oe.cpci_parts(P3["test"])
        cperm = torch.randperm(yc.shape[0], generator=torch.Generator().manual_seed(200 + seed)).to(dev)
        c1 = torch.zeros(yc.shape[0], dtype=torch.bool, device=dev)
        c1[cperm[: yc.shape[0] // 2]] = True
        mu = torch.zeros_like(ft); cov = torch.zeros_like(ft, dtype=torch.bool); zero = torch.zeros_like(cov)
        for g in (0, 1):
            vm, cm1, cm2, tm = gv == g, (gc == g) & c1, (gc == g) & ~c1, gt == g
            a_r, qq, r, gamma, beta = oe.cpci_fit(pv[vm].cpu(), fv[vm].cpu(), val["y"].view(-1)[vm].cpu(),
                                                  pc[cm1].cpu(), pc[cm2].cpu(), fc[cm2].cpu(), yc.view(-1)[cm2].cpu())
            m_, _, z_, c_ = oe.cpci_sets(pt[tm], ft[tm], yt[tm], a_r, qq)
            mu[tm], zero[tm], cov[tm] = m_, z_, c_
        M["E4"] = o1.summarise(mu, zero, torch.zeros_like(zero), cov, yt, gt)
        # E5 (frozen spec with feasibility revision)
        mu, i0, i1, cov, fits = e5_grouped(P3["val"], P3["cal"], P3["test"], val["y"], yc, yt, gv, gc, gt, seed)
        M["E5"] = o1.summarise(mu.to(dev), i0.to(dev), i1.to(dev), cov.to(dev), yt, gt)
        M["E5"]["fits"] = fits
        # E2: CQR, feasible (clipped) width from its own run
        rr = json.loads((ROOT / "results" / o1.TAG / DS / f"model2__s{seed}.json").read_text())
        s2 = rr["scores"]["2_clipped"][0]
        M["E2"] = {"mu": s2["width_boundary"], "n_atom": 0.0, "precision": float("nan"), "coverage": s2["coverage"],
                   "coverage_boundary": s2["coverage_boundary"], **{f"A_{l}": s2["width_boundary"] for l in o1.LAMBDAS}}
        for m in M.values():
            m["curve"] = [m["mu"] + lam * m["n_atom"] for lam in LAMS]
        rows.append({"seed": seed, "methods": M})
        print(f"seed {seed}: " + " ".join(f"{k}={v['mu']:.4f}/{v['n_atom']:.2f}/{v['coverage']:.3f}" for k, v in M.items()), flush=True)
    OUT.write_text(json.dumps(rows, indent=1, default=float))


def report():
    rows = json.loads(OUT.read_text())
    meths = ["A", "B", "C", "E1", "E2", "E4", "E5"]
    print("## ERA5 cloud cover: held-out confirmation (boundary group, obs eps = 0.05, 8 seeds)\n")
    print("| method | mu mean (median) | atoms/set | precision | coverage | boundary cov | A_0.05 mean (median) |")
    print("|---|---|---|---|---|---|---|")
    for m in meths:
        v = lambda k: [r["methods"][m][k] for r in rows]
        prec = [x for x in v("precision") if x == x]
        print(f"| {m} | {st.mean(v('mu')):.4f} ({st.median(v('mu')):.4f}) | {st.mean(v('n_atom')):.2f} | "
              f"{(st.mean(prec) if prec else float('nan')):.3f} | {st.mean(v('coverage')):.3f} | {st.mean(v('coverage_boundary')):.3f} | "
              f"{st.mean(v('A_0.05')):.4f} ({st.median(v('A_0.05')):.4f}) |")
    print("\nA_lambda curves (mean over seeds), lambda = " + ", ".join(f"{l:g}" for l in LAMS))
    for m in meths:
        c = [st.mean(r["methods"][m]["curve"][i] for r in rows) for i in range(len(LAMS))]
        print(f"  {m}: " + " ".join(f"{x:.4f}" for x in c))
    print("\nlambda* relative to A (per seed; defined when the method includes atoms):")
    for m in ["B", "C", "E4", "E5"]:
        ls = []
        for r in rows:
            a, b = r["methods"]["A"], r["methods"][m]
            if b["n_atom"] > 0:
                ls.append((a["mu"] - b["mu"]) / b["n_atom"])
        if ls:
            q = sorted(ls)
            print(f"  {m}: median {st.median(ls):.4f}, IQR [{q[len(q) // 4]:.4f}, {q[(3 * len(q)) // 4]:.4f}], n = {len(ls)}")
        else:
            print(f"  {m}: no atoms included in any seed")

    def cmp(a, b, key):
        xa = [r["methods"][a][key] for r in rows]
        xb = [r["methods"][b][key] for r in rows]
        diff = [p - q for p, q in zip(xa, xb)]
        p = wilcoxon(diff).pvalue if any(abs(x) > 1e-12 for x in diff) else 1.0
        return st.mean(xa), st.mean(xb), st.median(diff), p

    print("\n## Pre-registered comparisons (paired Wilcoxon, alpha = 0.05; negative = first method better)\n")
    for a, b, key in [("E5", "A", "mu"), ("E5", "A", "A_0.05"), ("E5", "E4", "A_0.05"),
                      ("B", "A", "mu"), ("B", "A", "A_0.05"), ("C", "A", "mu"), ("C", "A", "A_0.05")]:
        ma, mb, md, p = cmp(a, b, key)
        print(f"- {a} vs {b} on {key}: {ma:.4f} vs {mb:.4f}, median diff {md:+.4f}, p = {p:.3f} "
              f"-> {'significant' if p < 0.05 else 'n.s.'}{' (first better)' if p < 0.05 and ma < mb else (' (first worse)' if p < 0.05 else '')}")
    bad = [m for m in meths if abs(st.mean(r["methods"][m]["coverage"] for r in rows) - 0.9) > 0.02]
    print(f"\ncoverage outside 0.90 +/- 0.02 (width results invalid): {bad or 'none'}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.train:
        train()
    if a.evaluate:
        evaluate()
    if a.report:
        report()


if __name__ == "__main__":
    main()
