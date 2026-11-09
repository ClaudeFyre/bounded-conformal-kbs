"""Gate A: analyse the Stage 2 phase diagram exactly as pre-registered in PLAN.md.

Primary: oracle grouping, eps = 0.05, raw width; dW = W_G - W_B per seed, one panel per
bounded family. A cell is 'consistent' if sign(dW) agrees in >= 4/5 seeds. p_b = 0 is NA
(empty oracle boundary group; deviation 1). Also reports realised Gaussian leakage M.

    python scripts/analyze_phase.py
"""
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PB = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8]
C = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
BOUNDED = ["trunc_gaussian", "logit_normal", "beta", "kumaraswamy"]


def load(pb, c, fam, grouping="oracle", eps=0.05, key="biw"):
    vals, leak = {}, {}
    for s in range(5):
        fn = ROOT / "results" / "phase" / f"pb{pb:g}_c{c:g}" / f"{fam}__tn_hetero__raw__s{s}.json"
        r = json.loads(fn.read_text())
        cc = next(x for x in r["conformal"] if abs(x["obs"]["eps"] - eps) < 1e-9)
        vals[s] = cc[grouping][key]
        leak[s] = r["mean_leakage_test"]
    return vals, leak


def cell(pb, c, fam, grouping="oracle"):
    G, leakG = load(pb, c, "gaussian", grouping)
    B, _ = load(pb, c, fam, grouping)
    d = [G[s] - B[s] for s in range(5) if np.isfinite(G[s]) and np.isfinite(B[s])]
    if len(d) < 5:
        return None
    rel = 100 * np.mean(d) / np.mean([G[s] for s in range(5)])
    pos = sum(x > 0 for x in d)
    sign = 1 if pos >= 4 else (-1 if pos <= 1 else 0)
    return {"rel": rel, "sign": sign, "M": float(np.mean(list(leakG.values())))}


def main():
    out = {}
    print("Cell = relative dW = (W_G - W_B)/W_G in %, then sign consistency: + (B narrower in >=4/5 seeds), "
          "- (G narrower in >=4/5), ~ (inconsistent). Rows: C (posterior std); columns: p_b.\n")
    Mrow = [np.mean([cell(pb, c, "logit_normal")["M"] if cell(pb, c, "logit_normal") else np.nan for c in C])
            for pb in PB]
    for fam in BOUNDED:
        print(f"### {fam}\n")
        print("| C \\ p_b | " + " | ".join(f"{pb:g}" for pb in PB) + " |")
        print("|---" * (len(PB) + 1) + "|")
        grid = []
        for c in C:
            row = []
            for pb in PB:
                if pb == 0:
                    row.append(None)
                    continue
                row.append(cell(pb, c, fam))
            grid.append(row)
            cells = ["NA" if x is None else f"{x['rel']:+.1f}% {'+' if x['sign'] > 0 else ('-' if x['sign'] < 0 else '~')}"
                     for x in row]
            print(f"| {c:g} | " + " | ".join(cells) + " |")
        out[fam] = grid
        signs = np.array([[np.nan if x is None else x["sign"] for x in row] for row in grid])
        valid = signs[:, 1:]
        n_cons = int(np.sum(valid != 0))
        print(f"\nconsistent cells: {n_cons}/{valid.size}; "
              f"B-better {int(np.sum(valid > 0))}, G-better {int(np.sum(valid < 0))}\n")
    print("realised Gaussian leakage M by p_b (mean over C): " + ", ".join(f"p_b={pb:g}: {m:.3f}" for pb, m in zip(PB, Mrow)))
    print("\nrealised M per cell (rows C, columns p_b):")
    for c in C:
        print(f"  C={c:<6g} " + " ".join(f"{cell(pb, c, 'logit_normal')['M']:.3f}" if cell(pb, c, 'logit_normal') else "  NA " for pb in PB))
    (ROOT / "results" / "phase_summary.json").write_text(json.dumps(out, indent=1, default=float))
    plot(out)


def plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), sharey=True)
    for ax, fam in zip(axes, BOUNDED):
        grid = out[fam]
        Z = np.array([[np.nan if x is None else x["rel"] for x in row] for row in grid])
        lim = np.nanmax(np.abs(Z))
        im = ax.imshow(Z, cmap="RdBu", vmin=-lim, vmax=lim, origin="lower", aspect="auto")
        for i, row in enumerate(grid):
            for j, x in enumerate(row):
                txt = "NA" if x is None else f"{x['rel']:+.0f}{'' if x['sign'] else '~'}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=7)
        ax.set_xticks(range(len(PB)), [f"{p:g}" for p in PB])
        ax.set_yticks(range(len(C)), [f"{c:g}" for c in C])
        ax.set_xlabel("boundary fraction p_b")
        ax.set_title(f"{fam}: (W_G - W_B)/W_G %")
        fig.colorbar(im, ax=ax, fraction=0.046)
    axes[0].set_ylabel("posterior std C")
    fig.suptitle("Phase diagram (oracle grouping, eps=0.05; blue = bounded narrower; '~' = sign inconsistent over seeds)")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "phase_diagram.png", dpi=150)
    print("saved figures/phase_diagram.png")


if __name__ == "__main__":
    main()
