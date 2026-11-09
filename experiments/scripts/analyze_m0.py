"""Gate M0 verdict, exactly as pre-registered in PLAN.md (paired Wilcoxon over 8 seeds, alpha 0.05).

Primary width: boundary width, observation grouping, eps = 0.05.
"""
import json
import math
from pathlib import Path

from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1] / "results" / "m0"
DATASETS = ["syn_full", "bucket", "seaice", "resops", "syn_atom"]
# score name -> (model file id, score key)
SCORES = {"1": (1, "1"), "7": (1, "7"), "1b": (1, "1b"), "2": (2, "2"), "2c": (2, "2_clipped"),
          "3": (3, "3"), "4": (4, "4"), "5": (5, "5"), "6": (6, "6")}


def load(ds, name, key="width_boundary"):
    mid, sk = SCORES[name]
    out = {}
    for fn in (ROOT / ds).glob(f"model{mid}__s*.json"):
        r = json.loads(fn.read_text())
        seed = int(fn.stem.split("__s")[1])
        row = r["scores"][sk][0]                       # eps = 0.05
        out[seed] = {"w": row[key], "cov": row["coverage"], "covb": row["coverage_boundary"],
                     "crps": r["crps_test"], "nll": r["nll_test"],
                     "ece_out": r.get("ece_output_atom"), "ece_lat": r.get("ece_latent_atom")}
    return out


def compare(ds, a, b, metric="w"):
    """Mean of a, mean of b, relative change (a/b - 1), two-sided Wilcoxon p on paired seeds."""
    A, B = load(ds, a), load(ds, b)
    seeds = sorted(set(A) & set(B))
    xa = [A[s][metric] for s in seeds]
    xb = [B[s][metric] for s in seeds]
    if len(seeds) < 6 or any(v is None or not math.isfinite(v) for v in xa + xb):
        return None
    diff = [x - y for x, y in zip(xa, xb)]
    p = wilcoxon(diff).pvalue if any(diff) else 1.0
    ma, mb = sum(xa) / len(xa), sum(xb) / len(xb)
    return {"a": ma, "b": mb, "rel": ma / mb - 1, "p": p, "n": len(seeds)}


def fmt(c):
    return "NA" if c is None else f"{c['a']:.4f} vs {c['b']:.4f} ({100 * c['rel']:+.1f}%, p={c['p']:.3f})"


def not_worse(c):
    return c is not None and not (c["rel"] > 0 and c["p"] < 0.05)


def better(c):
    return c is not None and c["rel"] < 0 and c["p"] < 0.05


def main():
    print("## Descriptive table (obs grouping, eps = 0.05; mean over seeds)\n")
    print("| dataset | score | boundary width | coverage | boundary coverage | CRPS | NLL | ECE out-atom | ECE latent-atom |")
    print("|---|---|---|---|---|---|---|---|---|")
    for ds in DATASETS:
        for name in SCORES:
            L = load(ds, name)
            if not L:
                continue
            m = lambda k: sum(v[k] for v in L.values() if v[k] is not None) / max(1, sum(v[k] is not None for v in L.values()))
            has = lambda k: any(v[k] is not None for v in L.values())
            print(f"| {ds} | {name} | {m('w'):.4f} | {m('cov'):.3f} | {m('covb'):.3f} | "
                  f"{m('crps'):.4f} | " + (f"{m('nll'):.3f}" if has('nll') else "NA") + " | "
                  + (f"{m('ece_out'):.3f}" if has('ece_out') else "") + " | "
                  + (f"{m('ece_lat'):.3f}" if has('ece_lat') else "") + " |")

    print("\n## Go conditions\n")
    res = {}
    c1 = {m: compare("bucket", m, "1") for m in ("4", "5")}
    res[1] = any(not_worse(c) for c in c1.values())
    print(f"1. Bucket, width of 4/5 not significantly worse than 1: 4: {fmt(c1['4'])}; 5: {fmt(c1['5'])} -> {res[1]}")
    c2 = {m: compare("seaice", m, "7") for m in ("4", "5")}
    res[2] = any(not_worse(c) for c in c2.values())
    print(f"2. Sea ice, feasible width of 4/5 not significantly worse than 7 (clipped 1): 4: {fmt(c2['4'])}; 5: {fmt(c2['5'])} -> {res[2]}")
    ok3 = []
    for m in ("4", "5"):
        w, c = compare("resops", m, "1"), compare("resops", m, "1", "crps")
        good = w is not None and c is not None and w["rel"] <= 0.15 and c["rel"] <= 0.15
        ok3.append(good)
        print(f"3. ResOpsUS model {m}: width {fmt(w)}; CRPS {fmt(c)} -> {good}")
    res[3] = any(ok3)
    ok4 = []
    for ds in ("bucket", "syn_atom"):
        for m in ("4", "5"):
            for metric in ("w", "crps"):
                c = compare(ds, m, "3", metric)
                if better(c):
                    ok4.append(f"{ds}/{m}/{metric}")
                print(f"4. {ds}: model {m} vs 3 on {metric}: {fmt(c)}")
    res[4] = bool(ok4)
    print(f"   -> satisfied by: {ok4 or 'none'}")
    c5 = compare("syn_atom", "5", "4", "crps")
    res[5] = better(c5)
    print(f"5. Syn-Atom, CRPS of 5 significantly below 4: {fmt(c5)} -> {res[5]}")
    bad = []
    for ds in DATASETS:
        for name in ("1", "7", "1b", "3", "4", "5"):
            L = load(ds, name)
            if L:
                cov = sum(v["cov"] for v in L.values()) / len(L)
                if abs(cov - 0.90) > 0.02:
                    bad.append(f"{ds}/{name}: {cov:.3f}")
    res[6] = not bad
    print(f"6. Marginal coverage within 0.90 +/- 0.02 for all decision models: violations {bad or 'none'} -> {res[6]}")
    print(f"\n**Gate M0: {'GO' if all(res.values()) else 'NO-GO'}** ({', '.join(f'{k}:{v}' for k, v in res.items())})")

    print("\n## Side analyses (not Go conditions)\n")
    for ds in DATASETS:
        print(f"- {ds}: 4 vs 1b {fmt(compare(ds, '4', '1b'))}; 5 vs 1b {fmt(compare(ds, '5', '1b'))}; "
              f"1b vs 1 {fmt(compare(ds, '1b', '1'))}; 3 vs 1b {fmt(compare(ds, '3', '1b'))}; "
              f"6 vs 1 {fmt(compare(ds, '6', '1'))}; 2 vs 1 {fmt(compare(ds, '2', '1'))}")


if __name__ == "__main__":
    main()
