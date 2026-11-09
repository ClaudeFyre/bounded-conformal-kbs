"""Gate A2: analyse the bucket ablation ladder exactly as pre-registered in PLAN.md.

Per level B0..B5 and bounded family (logit_normal, kumaraswamy) vs Gaussian:
  dW = (W_G - W_B)/W_G per seed (observation grouping, eps = 0.05, raw width; oracle as secondary),
  sign consistency (>= 4/5 seeds), log R_q, log R_sigma, fit gaps (val y-log-lik, reconstruction),
  Gaussian leakage L, test RMSE, and increments Delta_k = dW(B_k) - dW(B_{k-1}).
Criterion 1 (sign reversal at a single level, both families) and criterion 2 (a mediator changes
at the same level) are evaluated mechanically; the swap-interaction mediator needs a separate run.

    python scripts/analyze_ladder.py
"""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAMS = ["logit_normal", "kumaraswamy"]
LEVELS = range(6)
SEEDS = range(5)


def load(level, fam, seed, grouping):
    r = json.loads((ROOT / "results" / f"ladder_B{level}" / f"{fam}__tn_hetero__raw__s{seed}.json").read_text())
    c = next(x for x in r["conformal"] if abs(x["obs"]["eps"] - 0.05) < 1e-9)[grouping]
    best = min(r["log"], key=lambda e: e["val_loss"])
    return {"W": c["biw"], "q": c["q_boundary"], "cov": c["coverage"], "covb": c["coverage_boundary"],
            "ylp": best["ylp"], "rec": best["rec"], "L": r["mean_leakage_test"], "rmse": r.get("rmse_test")}


def mean(v):
    v = [x for x in v if x is not None and math.isfinite(x)]
    return sum(v) / len(v) if v else float("nan")


def level_stats(level, fam, grouping):
    rows = []
    for s in SEEDS:
        g, b = load(level, "gaussian", s, grouping), load(level, fam, s, grouping)
        rows.append({
            "dW": (g["W"] - b["W"]) / g["W"],
            "logRq": math.log(g["q"] / b["q"]),
            "logRs": math.log((g["W"] / g["q"]) / (b["W"] / b["q"])),
            "ylp_gap": g["ylp"] - b["ylp"], "rec_gap": g["rec"] - b["rec"],
            "L": g["L"], "rmse_G": g["rmse"], "rmse_B": b["rmse"], "cov_G": g["cov"], "cov_B": b["cov"],
        })
    pos = sum(r["dW"] > 0 for r in rows)
    sign = 1 if pos >= 4 else (-1 if pos <= 1 else 0)
    out = {k: mean([r[k] for r in rows]) for k in rows[0]}
    out["sign"], out["pos"] = sign, pos
    return out


def mediator_changed(prev, cur):
    changes = []
    for k in ("logRq", "logRs"):
        a, b = prev[k], cur[k]
        if a * b < 0 or (abs(a) > 1e-9 and abs(b - a) > 0.5 * abs(a)):
            changes.append(k)
    for k in ("ylp_gap", "rec_gap"):
        if prev[k] * cur[k] < 0:
            changes.append(k)
    return changes


def main():
    for grouping in ("obs", "oracle"):
        print(f"\n## grouping = {grouping}{' (primary)' if grouping == 'obs' else ' (secondary)'}\n")
        crit1 = {}
        for fam in FAMS:
            print(f"### {fam} vs gaussian\n")
            print("| level | dW (B narrower +) | seeds + | Delta_k | log R_q | log R_sigma | y-loglik gap G-B | rec gap G-B | L | RMSE G / B | cov G / B | mediators changed |")
            print("|---|---|---|---|---|---|---|---|---|---|---|---|")
            prev = None
            flips = []
            for lv in LEVELS:
                st = level_stats(lv, fam, grouping)
                dk = "" if prev is None else f"{100 * (st['dW'] - prev['dW']):+.1f}"
                med = "" if prev is None else ", ".join(mediator_changed(prev, st)) or "none"
                flip = prev is not None and prev["sign"] != 0 and st["sign"] == -prev["sign"]
                if flip:
                    flips.append((lv, med))
                print(f"| B{lv}{' FLIP' if flip else ''} | {100 * st['dW']:+.1f}% | {st['pos']}/5 | {dk} | {st['logRq']:+.3f} | "
                      f"{st['logRs']:+.3f} | {st['ylp_gap']:+.3f} | {st['rec_gap']:+.3f} | {st['L']:.3f} | "
                      f"{st['rmse_G']:.4f} / {st['rmse_B']:.4f} | {st['cov_G']:.3f} / {st['cov_B']:.3f} | {med} |")
                prev = st
            crit1[fam] = flips
            print()
        common = set(l for l, _ in crit1[FAMS[0]]) & set(l for l, _ in crit1[FAMS[1]])
        print(f"criterion 1 (single-level stable reversal in both families): levels {sorted(common) or 'none'}")
        for lv in sorted(common):
            meds = {fam: dict(crit1[fam])[lv] for fam in FAMS}
            print(f"  B{lv}: mediators changed -> {meds}")


if __name__ == "__main__":
    main()
