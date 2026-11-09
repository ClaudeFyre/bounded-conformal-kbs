"""Aggregate result JSONs into mean ± std tables (markdown) with seed counts.

    python scripts/aggregate.py --dataset syn_full --eps 0.05
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEYS = ["biw", "biw_clipped", "coverage", "coverage_boundary", "coverage_true_boundary",
        "width_interior", "n_cal_boundary", "fp_vs_true", "fn_vs_true"]
THEORY_KEYS = ["cal_fp_vs_star", "cal_fn_vs_star", "q_gauss", "q_star", "q_star_shared_norm",
               "frac_score_increase", "inc_median", "m_minus", "m_tau0.1", "L_emp_median"]


def stats(v):
    v = [x for x in v if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not v:
        return "—"
    m = sum(v) / len(v)
    sd = (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5 if len(v) > 1 else 0.0
    return f"{m:.4f} +/- {sd:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="syn_full")
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--grouping", default="obs", choices=["latent", "obs", "oracle"])
    a = ap.parse_args()
    rows, theory, probe, leak = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list)), defaultdict(list)
    for fn in sorted((ROOT / "results" / a.dataset).glob("*.json")):
        r = json.loads(fn.read_text())
        cfg = fn.stem.rsplit("__s", 1)[0]
        c = next(c for c in r["conformal"] if abs(c["obs"]["eps"] - a.eps) < 1e-9)[a.grouping]
        for k in KEYS:
            rows[cfg][k].append(c[k])
        for k, v in r["probe"].items():
            probe[cfg][k].append(v)
        leak[cfg].append(r["mean_leakage_test"])
        if "theory" in r:
            t = next(t for t in r["theory"] if abs(t["eps"] - a.eps) < 1e-9)
            for k in THEORY_KEYS:
                theory[cfg][k].append(t.get(k))

    print(f"## {a.dataset}, eps = {a.eps}, grouping = {a.grouping}\n")
    print("| config | seeds | leakage | " + " | ".join(KEYS) + " |")
    print("|" + "---|" * (len(KEYS) + 3))
    for cfg in sorted(rows):
        print(f"| {cfg} | {len(leak[cfg])} | {stats(leak[cfg])} | " + " | ".join(stats(rows[cfg][k]) for k in KEYS) + " |")
    print("\n### Decoder extrapolation probe\n")
    print("| config | L_p05 | L_median | L_max |\n|---|---|---|---|")
    for cfg in sorted(probe):
        print(f"| {cfg} | " + " | ".join(stats(probe[cfg][k]) for k in ("L_p05", "L_median", "L_max")) + " |")
    if theory:
        print("\n### Theory checks (Gaussian latent)\n")
        print("| config | " + " | ".join(THEORY_KEYS) + " |")
        print("|" + "---|" * (len(THEORY_KEYS) + 1))
        for cfg in sorted(theory):
            print(f"| {cfg} | " + " | ".join(stats(theory[cfg][k]) for k in THEORY_KEYS) + " |")


if __name__ == "__main__":
    main()
