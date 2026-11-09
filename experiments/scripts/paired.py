"""Pre-registered paired comparison across seeds (Wilcoxon signed-rank), plus theory conditions C1-C3.

    python scripts/paired.py --dataset syn_full --eps 0.05 --grouping obs
"""
import argparse
import json
from pathlib import Path

from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
PAIRS = [
    ("main (TN-hetero)", "gaussian__tn_hetero__raw", "kumaraswamy__tn_hetero__raw"),
    ("fixed decoder (Gauss-homo)", "gaussian__gauss_homo__raw", "kumaraswamy__gauss_homo__raw"),
    ("saturating decoder (projected)", "gaussian__tn_hetero__projected", "kumaraswamy__tn_hetero__projected"),
]


def load(ds, cfg, eps, grouping):
    out = {}
    for fn in (ROOT / "results" / ds).glob(f"{cfg}__s*.json"):
        r = json.loads(fn.read_text())
        c = next(c for c in r["conformal"] if abs(c["obs"]["eps"] - eps) < 1e-9)
        out[int(fn.stem.rsplit("__s", 1)[1])] = (c[grouping], r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="syn_full")
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--grouping", default="obs")
    ap.add_argument("--metric", default="biw", choices=["biw", "biw_clipped"])
    a = ap.parse_args()
    print(f"## {a.dataset}  eps={a.eps}  grouping={a.grouping}\n")
    print("| comparison | seeds | Gaussian width | Kumaraswamy width | reduction | Wilcoxon p | cov G | cov K |")
    print("|---|---|---|---|---|---|---|---|")
    for label, g, k in PAIRS:
        G, K = load(a.dataset, g, a.eps, a.grouping), load(a.dataset, k, a.eps, a.grouping)
        seeds = sorted(set(G) & set(K))
        if not seeds:
            continue
        wg = [G[s][0][a.metric] for s in seeds]
        wk = [K[s][0][a.metric] for s in seeds]
        diff = [x - y for x, y in zip(wg, wk)]
        p = wilcoxon(diff).pvalue if len(seeds) >= 6 and any(diff) else float("nan")
        mg, mk = sum(wg) / len(wg), sum(wk) / len(wk)
        cg = sum(G[s][0]["coverage_boundary"] for s in seeds) / len(seeds)
        ck = sum(K[s][0]["coverage_boundary"] for s in seeds) / len(seeds)
        print(f"| {label} | {len(seeds)} | {mg:.4f} | {mk:.4f} | {100 * (1 - mk / mg):+.1f}% | {p:.4f} | {cg:.3f} | {ck:.3f} |")

    # theory conditions on the Gaussian main model (C1-C3 of the pre-registration)
    G = load(a.dataset, "gaussian__tn_hetero__raw", a.eps, a.grouping)
    rows = []
    for s, (_, r) in sorted(G.items()):
        t = next((t for t in r.get("theory", []) if abs(t["eps"] - a.eps) < 1e-9), None)
        if t and "q_gauss" in t:
            rows.append((r["mean_leakage_test"], t.get("L_emp_median"), t.get("m_tau0.1"), t.get("m_minus"),
                         t["q_gauss"], t["q_star"], t["cal_fn_vs_star"], t["cal_fp_vs_star"]))
    if rows:
        def m(i):
            v = [x[i] for x in rows if x[i] is not None]
            return sum(v) / len(v)
        print("\n### Theory conditions (Gaussian, TN-hetero; mean over seeds)\n")
        print(f"- C1 leakage (all test units) = {m(0):.3f}  (threshold 0.05)")
        print(f"- C2 ||Delta||/rho median = {m(1):.3f}  (threshold 0.1)")
        print(f"- C3 m(tau=0.1) = {m(2):.4f} vs m_- = {m(3):.4f}  -> {'holds' if m(2) > m(3) else 'FAILS'}")
        print(f"- quantiles on the latent boundary group: q^G = {m(4):.3f}, q* = {m(5):.3f}")
        print(f"- Prop. 3: FN vs h* = {m(6):.4f} (theory: 0), FP vs h* = {m(7):.3f}")


if __name__ == "__main__":
    main()
