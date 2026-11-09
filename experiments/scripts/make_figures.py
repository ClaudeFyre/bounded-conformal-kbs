"""Generate all paper figures directly from results/ (no hand-typed numbers except where noted).

    python scripts/make_figures.py        -> figures/paper/fig*.pdf|png and figures/paper/figure_data.json

Every number drawn is also written to figure_data.json so RESULTS_MANIFEST.md can be checked against it.
"""
import argparse
import json
import math
import re
import statistics as st
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
RES = ROOT / "results"
OUT = ROOT / "figures" / "paper"

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight",
                     "pdf.fonttype": 42, "ps.fonttype": 42})   # TrueType, no Type 3 fonts (IEEE PDF checks)
DS5 = ["syn_full", "syn_extreme", "seaice", "bucket", "resops"]
LABEL = {"syn_full": "Syn-Full", "syn_extreme": "Syn-Extreme", "seaice": "Sea ice", "bucket": "Bucket",
         "resops": "ResOpsUS", "syn_outatom": "Syn-OutAtom", "era5_cloud": "ERA5 cloud"}
FAMS = ["trunc_gaussian", "logit_normal", "beta", "kumaraswamy"]
FAM_LABEL = {"trunc_gaussian": "Trunc. Gaussian", "logit_normal": "Logit-normal", "beta": "Beta", "kumaraswamy": "Kumaraswamy"}
COL = {"syn_full": "#4C72B0", "syn_extreme": "#55A868", "seaice": "#8172B2", "bucket": "#C44E52", "resops": "#CCB974",
       "syn_outatom": "#64B5CD", "era5_cloud": "#DD8452"}
DATA = {}


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)


def width(ds, fam, seed, key):
    r = json.loads((RES / ds / f"{fam}__tn_hetero__raw__s{seed}.json").read_text())
    return next(c for c in r["conformal"] if abs(c["obs"]["eps"] - 0.05) < 1e-9)["obs"][key]


# ---------------------------------------------------------------------------
def fig1_bounded_effect():
    """Relative width change of each bounded family vs Gaussian (TN-hetero, obs grouping, eps 0.05)."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    d = {}
    for ax, key, title in zip(axes, ["biw", "biw_clipped"], ["raw width", "feasible width (clipped to [0,1])"]):
        for i, ds in enumerate(DS5):
            for j, fam in enumerate(FAMS):
                rel = [100 * (1 - width(ds, fam, s, key) / width(ds, "gaussian", s, key)) for s in range(10)]
                x = i + (j - 1.5) * 0.17
                ax.scatter([x] * len(rel), rel, s=5, color=f"C{j}", alpha=0.35, lw=0)
                ax.errorbar(x, np.mean(rel), yerr=np.std(rel, ddof=1), fmt="o", ms=3.5, color=f"C{j}",
                            label=FAM_LABEL[fam] if (i == 0 and key == "biw") else None)
                d[f"{ds}/{fam}/{key}"] = {"mean": float(np.mean(rel)), "sd": float(np.std(rel, ddof=1))}
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(range(5), [LABEL[x] for x in DS5], rotation=25)
        ax.set_title(title)
    axes[0].set_ylabel("width reduction vs Gaussian (%)\n(positive = bounded narrower)")
    axes[0].legend(fontsize=7, frameon=False, loc="lower left")
    DATA["fig1"] = d
    save(fig, "fig1_bounded_effect")


def fig2_posthoc_correction():
    """Gate B: width with the Gaussian posterior (a) vs its truncation to M (e), same decoder; 200 pairs."""
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    d = {}
    for ds in DS5:
        rows = json.loads((RES / "counterfactual" / f"{ds}.json").read_text())
        # (a) and (e) use only the Gaussian checkpoint, so the 4 bounded-family rows per seed repeat the same
        # pair: keep one pair per seed (10 independent pairs per dataset, 50 in total)
        per_seed = {r["seed"]: (r["W_a"], r["W_e"]) for r in rows}
        wa = [v[0] for v in per_seed.values()]
        we = [v[1] for v in per_seed.values()]
        ax.scatter(wa, we, s=9, color=COL[ds], label=LABEL[ds], alpha=0.8, lw=0)
        d[ds] = {"n": len(wa), "n_wider": sum(e > a for a, e in zip(wa, we)),
                 "median_log_ratio": float(np.median(np.log(np.array(we) / np.array(wa))))}
    lim = [0.03, 1.0]
    ax.plot(lim, lim, "k--", lw=0.8)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("boundary width, Gaussian posterior")
    ax.set_ylabel("boundary width, truncated to support")
    tot = sum(v["n_wider"] for v in d.values()), sum(v["n"] for v in d.values())
    ax.set_title(f"post-hoc truncation widens {tot[0]}/{tot[1]} (seed, dataset) pairs", fontsize=8.5)
    ax.legend(fontsize=7, frameon=False)
    DATA["fig2"] = d
    save(fig, "fig2_posthoc_correction")


def fig3_decomposition():
    """log R_W = log R_q + log R_sigma: (left) truncation e vs a; (right) Kumaraswamy vs Gaussian main comparison."""
    tc = json.loads((RES / "theory_check.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=False)
    d = {"truncation": {}, "kumaraswamy_vs_gaussian": {}}
    for i, ds in enumerate(DS5):
        rq = [r["log_Rq"] for r in tc[ds]]; rs = [r["log_Rsigma"] for r in tc[ds]]
        axes[0].bar(i - 0.18, np.median(rq), 0.34, color="#4C72B0", label="log $R_q$" if i == 0 else None)
        axes[0].bar(i + 0.18, np.median(rs), 0.34, color="#DD8452", label="log $R_\\sigma$" if i == 0 else None)
        axes[0].scatter([i - 0.18] * len(rq), rq, s=4, color="k", zorder=3)
        axes[0].scatter([i + 0.18] * len(rs), rs, s=4, color="k", zorder=3)
        d["truncation"][ds] = {"median_log_Rq": float(np.median(rq)), "median_log_Rsigma": float(np.median(rs))}
        lq, ls = [], []
        for s in range(10):
            g = json.loads((RES / ds / f"gaussian__tn_hetero__raw__s{s}.json").read_text())
            k = json.loads((RES / ds / f"kumaraswamy__tn_hetero__raw__s{s}.json").read_text())
            cg = next(c for c in g["conformal"] if abs(c["obs"]["eps"] - 0.05) < 1e-9)["obs"]
            ck = next(c for c in k["conformal"] if abs(c["obs"]["eps"] - 0.05) < 1e-9)["obs"]
            lq.append(math.log(cg["q_boundary"] / ck["q_boundary"]))
            ls.append(math.log(cg["biw"] / ck["biw"]) - lq[-1])
        axes[1].bar(i - 0.18, np.median(lq), 0.34, color="#4C72B0")
        axes[1].bar(i + 0.18, np.median(ls), 0.34, color="#DD8452")
        axes[1].scatter([i - 0.18] * len(lq), lq, s=4, color="k", zorder=3)
        axes[1].scatter([i + 0.18] * len(ls), ls, s=4, color="k", zorder=3)
        d["kumaraswamy_vs_gaussian"][ds] = {"median_log_Rq": float(np.median(lq)), "median_log_Rsigma": float(np.median(ls))}
    for ax, t in zip(axes, ["truncated vs Gaussian posterior (same decoder)", "Gaussian vs Kumaraswamy (separately trained)"]):
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(range(5), [LABEL[x] for x in DS5], rotation=25)
        ax.set_title(t, fontsize=8.5)
    axes[0].set_ylabel("log ratio")
    axes[0].legend(fontsize=7, frameon=False)
    DATA["fig3"] = d
    save(fig, "fig3_quantile_scale")


def fig4_theorem3_conditions():
    tc = json.loads((RES / "theory_check.json").read_text())
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    d = {}
    for ds in DS5:
        sk = [r["skew"] for r in tc[ds]]; cv = [r["ratio_cv"] for r in tc[ds]]
        ax.scatter(sk, cv, s=12, color=COL[ds], label=LABEL[ds], lw=0)
        d[ds] = {"median_skew": float(np.median(sk)), "median_ratio_cv": float(np.median(cv)),
                 "wider": sum(r["log_Wratio"] > 0 for r in tc[ds]), "n": len(tc[ds])}
    ax.set_xlabel("skewness of normalized residual under $q^G$\n(Theorem 1 requires symmetry: 0)")
    ax.set_ylabel("CV of scale ratio $\\tilde\\sigma/\\hat\\sigma$\n(Theorem 1 requires a common ratio: 0)")
    ax.set_xscale("symlog", linthresh=0.1)
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("Theorem 1 conditions: approx. met only on synthetic data", fontsize=8.5)
    DATA["fig4"] = d
    save(fig, "fig4_theorem3_conditions")


def fig5_lambda_curves():
    lams = [0.01 * i for i in range(11)]
    o1 = json.loads((RES / "o1.json").read_text())
    era = json.loads((RES / "era5_confirm.json").read_text())
    panels = [("seaice", [r for r in o1 if r["dataset"] == "seaice"], ["A", "B", "C", "E4"]),
              ("syn_outatom", [r for r in o1 if r["dataset"] == "syn_outatom"], ["A", "B", "C", "E4"]),
              ("era5_cloud", era, ["A", "B", "C", "E4", "E5"])]
    names = {"A": "continuous HPD", "B": "free-atom HPD", "C": "gated-atom HPD", "E4": "CPCI", "E5": "two-boundary (E5)"}
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.6))
    d = {}
    for ax, (ds, rows, meths) in zip(axes, panels):
        d[ds] = {}
        for j, m in enumerate(meths):
            mu = st.mean(r["methods"][m]["mu"] for r in rows)
            na = st.mean(r["methods"][m]["n_atom"] for r in rows)
            ax.plot(lams, [mu + l * na for l in lams], color=f"C{j}", label=names[m])
            ls = [(r["methods"]["A"]["mu"] - r["methods"][m]["mu"]) / r["methods"][m]["n_atom"]
                  for r in rows if m != "A" and r["methods"][m]["n_atom"] > 0]
            d[ds][m] = {"mu": mu, "n_atom": na, "lambda_star_median": float(np.median(ls)) if ls else None}
            if ls and 0 < np.median(ls) < 0.1:
                ax.axvline(np.median(ls), color=f"C{j}", ls=":", lw=0.9)
        ax.set_title(LABEL[ds]); ax.set_xlabel("atom cost $\\lambda$")
    axes[0].set_ylabel("ambiguity $A_\\lambda=\\mu+\\lambda N_{atom}$")
    axes[2].legend(fontsize=6.5, frameon=False)
    DATA["fig5"] = d
    save(fig, "fig5_lambda_curves")


def fig_phase():
    """Phase diagram, paper version: sign category per (C, p_b) cell and family (from phase_summary.json, written by
    analyze_phase.py), with the discovery datasets placed on the C axis by their measured posterior scale
    (phase_placement.json; post hoc) and marked by the observed sign of that family's raw-width effect (fig1)."""
    from matplotlib.patches import Rectangle
    ps = json.loads((RES / "phase_summary.json").read_text())
    pl = json.loads((RES / "phase_placement.json").read_text())
    PB = [0.1, 0.2, 0.4, 0.6, 0.8]            # p_b = 0 column has no primary metric (design deviation 1)
    CS = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
    BLUE, RED, GRAY, INK, MUTED = "#2a78d6", "#e34948", "#f0efec", "#222222", "#6b6b6b"
    TINT = {1: "#d4e4f7", -1: "#f9d6d6", 0: GRAY}
    SYM = {1: "+", -1: "−", 0: "~"}
    placement = {ds: float(np.median([r["median_sigma"] for r in pl[ds]])) for ds in DS5}
    yof = lambda c: np.interp(math.log(c), [math.log(x) for x in CS], range(len(CS)))
    order = sorted(DS5, key=lambda s: placement[s])
    xdodge, prev, k = {}, None, 0
    for ds in order:
        k = k + 1 if prev is not None and yof(placement[ds]) - yof(placement[prev]) < 0.3 else 0
        xdodge[ds] = len(PB) - 0.5 + 0.22 + 0.22 * k
        prev = ds
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.9), sharey=True)
    d = {"placement_C": placement, "cells": {}}
    for k, (ax, fam) in enumerate(zip(axes, FAMS)):
        d["cells"][fam] = []
        for i in range(len(CS)):
            for j, pb in enumerate(PB):
                cell = ps[fam][i][PB.index(pb) + 1]
                sg = cell["sign"]
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=TINT[sg], edgecolor="white", lw=2))
                ax.text(j, i, SYM[sg], ha="center", va="center", fontsize=9, color=INK if sg else MUTED)
                d["cells"][fam].append({"C": CS[i], "p_b": pb, "sign": sg, "rel": cell["rel"]})
        # discovery datasets on the C axis (right edge), marked by the observed sign for this family;
        # datasets closer than 0.3 rows are dodged horizontally
        for ds in DS5:
            eff = DATA["fig1"][f"{ds}/{fam}/biw"]["mean"]
            up = eff > 0
            ax.plot(xdodge[ds], yof(placement[ds]), marker="^" if up else "v", ms=6, clip_on=False,
                    color=BLUE if up else RED, mec="white", mew=0.8, zorder=5)
        ax.set_xlim(-0.5, len(PB) - 0.5)
        ax.set_ylim(-0.5, len(CS) - 0.5)
        ax.set_xticks(range(len(PB)), [f"{p:g}" for p in PB])
        ax.set_yticks(range(len(CS)), [f"{c:g}" for c in CS])
        ax.tick_params(length=0, colors=MUTED, labelsize=6.5)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(FAM_LABEL[fam], fontsize=8.5, color=INK)
        ax.set_xlabel("boundary fraction $p_b$", color=MUTED)
    axes[0].set_ylabel("posterior scale $C$", color=MUTED)
    # label the dataset markers once, on the last panel
    ax = axes[-1]
    ty, last = {}, -9
    for ds in order:                                   # spread label positions to a minimum gap of 0.42 rows
        ty[ds] = max(yof(placement[ds]), last + 0.42)
        last = ty[ds]
    for ds in order:
        ax.annotate(LABEL[ds], xy=(xdodge[ds] + 0.08, yof(placement[ds])), xytext=(len(PB) + 0.45, ty[ds]),
                    fontsize=6.5, color=INK, va="center", annotation_clip=False,
                    arrowprops={"arrowstyle": "-", "lw": 0.5, "color": MUTED, "shrinkA": 1, "shrinkB": 3})
    from matplotlib.lines import Line2D
    handles = [Rectangle((0, 0), 1, 1, facecolor=TINT[1], label="+  bounded narrower (≥4/5 seeds)"),
               Rectangle((0, 0), 1, 1, facecolor=TINT[-1], label="−  Gaussian narrower (≥4/5 seeds)"),
               Rectangle((0, 0), 1, 1, facecolor=GRAY, label="~  sign unstable"),
               Line2D([], [], ls="", marker="^", color=BLUE, label="dataset: bounded narrower (observed)"),
               Line2D([], [], ls="", marker="v", color=RED, label="dataset: bounded wider (observed)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, -0.13))
    fig.subplots_adjust(wspace=0.42, right=0.86)
    DATA["fig_phase"] = d
    save(fig, "fig_phase_diagram")


def _parse_o15(fn):
    t = (RES / fn).read_text()
    p0, p1 = map(float, re.search(r"P\(Y=0\)\s*=?\s*([\d.]+),\s*P\(Y=1\)\s*=?\s*([\d.]+)", t).groups())
    inter = float(re.search(r"interior mass\s*=?\s*([\d.]+)", t).group(1))
    return p0, p1, inter


def fig6_structure_map():
    import m0_one
    pts = {}
    for ds in ("seaice", "resops"):                   # target structure from the seed-0 test split
        y = m0_one.load(ds, 0)[0]["test"]["y"].view(-1).numpy()
        pts[LABEL[ds]] = ((y <= 0).mean(), (y >= 1).mean(), ((y > 0) & (y < 1)).mean())
    pts["Snow_cci CryoClim"] = _parse_o15("o15_snow_check.txt")
    pts["MODIS SCFG"] = _parse_o15("o15_modis_scfg_check.txt")
    pts["MODIS SCFV"] = _parse_o15("o15_modis_scfv_check.txt")
    pts["ERA5 cloud"] = _parse_o15("o15_cloud_check.txt")
    # Diagnostic landscape. Region boundaries are the preregistered O1.5 thresholds (criterion 1: both atoms
    # >= 0.05; criterion 3a: interior mass >= 0.2); the width criterion 3b is not shown. Identity by direct label,
    # role by marker shape (no categorical colour: six points exceed the scatter palette cap).
    INK, MUTED = "#222222", "#6b6b6b"
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.fill_between([-0.01, 0.5], 0, 0.2, color="#e6e5e1", lw=0, zorder=0)
    ax.fill_between([-0.01, 0.05], 0.2, 1.05, color="#f3f2ef", lw=0, zorder=0)
    ax.fill_between([0.05, 0.5], 0.2, 1.05, color="#d4e4f7", lw=0, zorder=0)
    ax.text(0.28, 0.172, "atom-dominated (near-binary): classification + conditional regression",
            fontsize=6.5, ha="center", va="center", color=INK)
    ax.text(0.003, 0.5, "at most one\nweak atom:\ncontinuous\nset", fontsize=6.5, ha="left", va="center", color=INK)
    ax.text(0.49, 1.0, "mixed: atom-aware evaluation\n($A_\\lambda$, $\\lambda^\\star$)", fontsize=6.5, ha="right",
            va="top", color=INK)
    role = {"Sea ice": "discovery", "ResOpsUS": "discovery", "Snow_cci CryoClim": "candidate",
            "MODIS SCFG": "candidate", "MODIS SCFV": "candidate", "ERA5 cloud": "held-out"}
    style = {"discovery": dict(marker="o", s=34, facecolor=INK, edgecolor="white"),
             "candidate": dict(marker="s", s=30, facecolor="white", edgecolor=INK),
             "held-out": dict(marker="*", s=110, facecolor=INK, edgecolor="white")}
    off = {"Sea ice": (-6, 7, "right"), "ResOpsUS": (6, -2, "left"), "Snow_cci CryoClim": (-6, 6, "right"),
           "MODIS SCFG": (6, 5, "left"), "MODIS SCFV": (6, 5, "left"), "ERA5 cloud": (7, -3, "left")}
    for k, (p0, p1, inter) in pts.items():
        x = min(p0, p1)
        ax.scatter(x, inter, zorder=3, linewidths=0.9, **style[role[k]])
        dx, dy, ha = off[k]
        ax.annotate(k, (x, inter), xytext=(dx, dy), textcoords="offset points", fontsize=6.5, ha=ha, color=INK)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], ls="", marker=style[r]["marker"], markerfacecolor=style[r]["facecolor"],
                      markeredgecolor=INK, markersize=6 if r != "held-out" else 9, label=lab)
               for r, lab in (("discovery", "discovery data"), ("candidate", "O1.5 candidate (failed)"),
                              ("held-out", "held-out confirmation"))]
    ax.legend(handles=handles, fontsize=6.5, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.set_xlim(-0.01, 0.5); ax.set_ylim(0, 1.05)
    ax.tick_params(colors=MUTED, labelsize=7)
    ax.set_xlabel("weaker atom: min(P(Y=0), P(Y=1))", color=MUTED)
    ax.set_ylabel("interior mass P(0<Y<1)", color=MUTED)
    DATA["fig6"] = {k: {"p0": float(v[0]), "p1": float(v[1]), "interior": float(v[2])} for k, v in pts.items()}
    save(fig, "fig6_structure_map")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", help="regenerate only these figure functions; other entries of "
                                                "figure_data.json are kept")
    a = ap.parse_args()
    funcs = [fig1_bounded_effect, fig2_posthoc_correction, fig3_decomposition, fig4_theorem3_conditions,
             fig5_lambda_curves, fig6_structure_map, fig_phase]
    if a.only:
        DATA.update(json.loads((OUT / "figure_data.json").read_text()))
        funcs = [f for f in funcs if f.__name__ in a.only]
    for f in funcs:
        f()
        print(f"done {f.__name__}", flush=True)
    (OUT / "figure_data.json").write_text(json.dumps(DATA, indent=1, default=float))


if __name__ == "__main__":
    # TNNLS presentation refresh: preserve all existing scientific summaries.
    from redraw_tnnls_figures import main as publication_main
    publication_main()
