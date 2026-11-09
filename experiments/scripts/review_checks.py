"""Post hoc checks raised in a simulated review (plan written and committed before this script was run).

    python scripts/review_checks.py      -> results/review_checks.json, results/review_checks.txt

Evidence role: EXPLORATORY (post hoc). Disclosure: while locating the stored fields, the means over seeds
of the validation-loss components of the main configuration were printed before this plan was written
(they show a much larger validation loss for the Kumaraswamy latent on ResOpsUS than for the Gaussian).
The signs of the bounded-latent width effect for all four families are already published (C1, Fig. 1).
Nothing below is a preregistered test; criteria are fixed here so that the reading of the output is not
chosen after seeing it.

R1  Fit quality of the bounded-latent comparison (review point: support confounded with fit)
    Data: results/{syn_full,syn_extreme,bucket,seaice,resops}/{family}__tn_hetero__raw__s{0..9}.json, the
    configuration of the main comparison; families gaussian, kumaraswamy, beta, logit_normal, trunc_gaussian.
    Per model: the validation loss of the restored checkpoint (best_val_loss = negative ELBO on held-back
    training units) and the target log-likelihood ylp of the validation log entry with the smallest val_loss.
    Per dataset x bounded family, paired over seeds with the Gaussian model of the same seed:
      dval = val_B - val_G   (positive: bounded fits worse)        mean, two-sided Wilcoxon p
      dylp = ylp_B - ylp_G   (negative: bounded predicts y worse)  mean, two-sided Wilcoxon p
      rho_raw, rho_feas      mean over seeds of (W_G - W_B)/W_G in percent, boundary group of the observation
                             grouping, eps = 0.05, from biw and biw_clipped; Wilcoxon p on raw width.
    Reading, fixed now:
      (a) a bounded family FITS WORSE on a dataset if mean dval > 0 with Wilcoxon p < 0.05;
      (b) worse fit can explain a widening (rho_raw < 0) but not a narrowing; a significant widening
          (rho_raw < 0, p < 0.05) is FIT-CONFOUNDED if the family fits worse by (a);
      (c) the claim that the sign of the bounded-latent effect depends on the system SURVIVES the fit check if
          among the non-synthetic datasets (bucket, seaice, resops) at least one shows a significant widening
          by a family that does not fit worse, and at least one dataset shows a significant narrowing;
          otherwise the paper must say that the widening may reflect fit rather than support;
      (d) for every dataset with a significant Kumaraswamy widening, list the families that widen significantly
          without fitting worse;
      (e) descriptive: Spearman correlation over the 20 dataset x family cells between mean dval and rho_raw.

R2  Interval estimates for the held-out ERA5 comparison (review point: absence of a difference at 8 seeds)
    Data: results/era5_confirm.json (mu, A_0.05 per seed and method) and results/endpoint_cost.json rows of
    era5_cloud (nu = endpoint-inclusive size at the recorded lambda).
    For B, C, E4, E5 against A: per-seed differences method - A; mean, 95% t-interval over seeds (df = n-1),
    and both expressed relative to the mean of A in percent.
    Reading, fixed now: the lower end of the relative interval is reported as the largest advantage over the
    continuous HPD set that the held-out data are compatible with at 95%; no equivalence margin is claimed.

R3  Scale of the effects (review point: effect sizes without reference)
    Main Gaussian model (gaussian__tn_hetero__raw), observation grouping, eps = 0.05, mean over seeds:
    boundary share of test units n_test_boundary / n_test, boundary width biw, interior width width_interior.
    ERA5 runs do not store group sizes; no ERA5 value is reported.
"""
import json
import statistics as st
from pathlib import Path

from scipy.stats import spearmanr, t as tdist, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
DATASETS = ["syn_full", "syn_extreme", "bucket", "seaice", "resops"]
NON_SYNTHETIC = ["bucket", "seaice", "resops"]
BOUNDED = ["kumaraswamy", "beta", "logit_normal", "trunc_gaussian"]
EPS = 0.05


def load(ds, fam):
    out = {}
    for fn in (RES / ds).glob(f"{fam}__tn_hetero__raw__s*.json"):
        r = json.loads(fn.read_text())
        c = next(c for c in r["conformal"] if abs(c["obs"]["eps"] - EPS) < 1e-9)["obs"]
        best = min(r["log"], key=lambda e: e["val_loss"])
        out[int(fn.stem.rsplit("__s", 1)[1])] = {
            "val": r["best_val_loss"], "ylp": best["ylp"], "biw": c["biw"], "biw_clipped": c["biw_clipped"],
            "n_test": c["n_test"], "n_test_boundary": c["n_test_boundary"], "width_interior": c["width_interior"]}
    return out


def wil(x):
    return float(wilcoxon(x).pvalue) if len(x) >= 6 and any(v != 0 for v in x) else float("nan")


def ci95(x):
    n, m = len(x), st.mean(x)
    h = tdist.ppf(0.975, n - 1) * st.stdev(x) / n ** 0.5
    return m, m - h, m + h


def r1():
    cells = []
    for ds in DATASETS:
        G = load(ds, "gaussian")
        for fam in BOUNDED:
            B = load(ds, fam)
            s = sorted(set(G) & set(B))
            dval = [B[i]["val"] - G[i]["val"] for i in s]
            dylp = [B[i]["ylp"] - G[i]["ylp"] for i in s]
            rraw = [100 * (G[i]["biw"] - B[i]["biw"]) / G[i]["biw"] for i in s]
            rfeas = [100 * (G[i]["biw_clipped"] - B[i]["biw_clipped"]) / G[i]["biw_clipped"] for i in s]
            p_w = wil([G[i]["biw"] - B[i]["biw"] for i in s])
            c = {"dataset": ds, "family": fam, "seeds": len(s),
                 "dval_mean": st.mean(dval), "dval_p": wil(dval), "dylp_mean": st.mean(dylp), "dylp_p": wil(dylp),
                 "rho_raw": st.mean(rraw), "rho_raw_p": p_w, "rho_feas": st.mean(rfeas)}
            c["fits_worse"] = c["dval_mean"] > 0 and c["dval_p"] < 0.05
            c["sig_widen"] = c["rho_raw"] < 0 and p_w < 0.05
            c["sig_narrow"] = c["rho_raw"] > 0 and p_w < 0.05
            c["fit_confounded"] = c["sig_widen"] and c["fits_worse"]
            cells.append(c)
    clean_widen = [c for c in cells if c["dataset"] in NON_SYNTHETIC and c["sig_widen"] and not c["fits_worse"]]
    any_narrow = [c for c in cells if c["sig_narrow"]]
    kum_widen = [c["dataset"] for c in cells if c["family"] == "kumaraswamy" and c["sig_widen"]]
    d = {ds: [c["family"] for c in cells if c["dataset"] == ds and c["sig_widen"] and not c["fits_worse"]]
         for ds in kum_widen}
    rho, p = spearmanr([c["dval_mean"] for c in cells], [c["rho_raw"] for c in cells])
    return {"cells": cells,
            "c_claim_survives": bool(clean_widen) and bool(any_narrow),
            "c_clean_widening_cells": [(c["dataset"], c["family"]) for c in clean_widen],
            "d_kumaraswamy_widening_datasets": d,
            "e_spearman_dval_rho": {"rho": float(rho), "p": float(p), "cells": len(cells)}}


def r2():
    era = json.loads((RES / "era5_confirm.json").read_text())
    end = [r for r in json.loads((RES / "endpoint_cost.json").read_text())["rows"] if r["dataset"] == "era5_cloud"]
    out = {}
    for m in ["B", "C", "E4", "E5"]:
        row = {}
        for metric, src, key in (("mu", era, "mu"), ("A_0.05", era, "A_0.05"), ("nu", end, "nu")):
            a = [r["methods"]["A"][key] for r in sorted(src, key=lambda r: r["seed"])]
            b = [r["methods"][m][key] for r in sorted(src, key=lambda r: r["seed"])]
            diff = [y - x for x, y in zip(a, b)]
            mean, lo, hi = ci95(diff)
            ma = st.mean(a)
            row[metric] = {"seeds": len(diff), "mean_diff": mean, "ci_lo": lo, "ci_hi": hi,
                           "rel_mean_pct": 100 * mean / ma, "rel_ci_lo_pct": 100 * lo / ma,
                           "rel_ci_hi_pct": 100 * hi / ma, "largest_compatible_advantage_pct": max(0.0, -100 * lo / ma)}
        out[m] = row
    return out


def r3():
    out = {}
    for ds in DATASETS:
        G = load(ds, "gaussian")
        v = list(G.values())
        out[ds] = {"seeds": len(v),
                   "boundary_share": st.mean(x["n_test_boundary"] / x["n_test"] for x in v),
                   "boundary_width": st.mean(x["biw"] for x in v),
                   "interior_width": st.mean(x["width_interior"] for x in v)}
    return out


def main():
    res = {"R1_fit_quality": r1(), "R2_era5_intervals": r2(), "R3_scale": r3()}
    (RES / "review_checks.json").write_text(json.dumps(res, indent=2))
    L = ["# Post hoc review checks (plan in scripts/review_checks.py)", "", "## R1 fit quality", "",
         "| dataset | family | dval | p | dylp | p | rho_raw | p | rho_feas | fits worse | widen | fit-confounded |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in res["R1_fit_quality"]["cells"]:
        L.append(f"| {c['dataset']} | {c['family']} | {c['dval_mean']:+.3f} | {c['dval_p']:.3f} | {c['dylp_mean']:+.3f} | "
                 f"{c['dylp_p']:.3f} | {c['rho_raw']:+.1f}% | {c['rho_raw_p']:.3f} | {c['rho_feas']:+.1f}% | "
                 f"{c['fits_worse']} | {c['sig_widen']} | {c['fit_confounded']} |")
    r = res["R1_fit_quality"]
    L += ["", f"(c) claim survives: {r['c_claim_survives']}; clean widening cells: {r['c_clean_widening_cells']}",
          f"(d) Kumaraswamy widening datasets -> families widening without worse fit: {r['d_kumaraswamy_widening_datasets']}",
          f"(e) Spearman(dval, rho_raw) = {r['e_spearman_dval_rho']['rho']:.3f}, p = {r['e_spearman_dval_rho']['p']:.3f}",
          "", "## R2 ERA5 differences to A (95% t-interval, relative to mean of A)", ""]
    for m, row in res["R2_era5_intervals"].items():
        for metric, v in row.items():
            L.append(f"{m} {metric}: {v['rel_mean_pct']:+.1f}% [{v['rel_ci_lo_pct']:+.1f}%, {v['rel_ci_hi_pct']:+.1f}%]"
                     f"  largest compatible advantage {v['largest_compatible_advantage_pct']:.1f}%  (n={v['seeds']})")
    L += ["", "## R3 scale (Gaussian main model)", ""]
    for ds, v in res["R3_scale"].items():
        L.append(f"{ds}: boundary share {v['boundary_share']:.3f}, boundary width {v['boundary_width']:.4f}, "
                 f"interior width {v['interior_width']:.4f} (n={v['seeds']})")
    (RES / "review_checks.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
