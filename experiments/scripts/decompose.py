"""Split the Gaussian/Kumaraswamy boundary-width ratio into quantile and scale factors.

Within a Mondrian group, mean width = 2 * q_g * mean(sigma_hat), so
    W_G / W_K = (q_G / q_K) * (sigmabar_G / sigmabar_K)   exactly.
Theorem 5 attributes the gap to the quantile factor; this checks which factor carries it.

    python scripts/decompose.py --dataset syn_full --eps 0.05 --grouping obs
"""
import argparse
import math

from paired import PAIRS, load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="syn_full")
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--grouping", default="obs")
    a = ap.parse_args()
    print(f"## {a.dataset}  eps={a.eps}  grouping={a.grouping}\n")
    print("| comparison | width ratio G/K | quantile ratio q_G/q_K | scale ratio | share of log-gap from quantile |")
    print("|---|---|---|---|---|")
    for label, g, k in PAIRS:
        G, K = load(a.dataset, g, a.eps, a.grouping), load(a.dataset, k, a.eps, a.grouping)
        seeds = sorted(set(G) & set(K))
        if not seeds:
            continue
        lw, lq = [], []
        for s in seeds:
            cg, ck = G[s][0], K[s][0]
            lw.append(math.log(cg["biw"] / ck["biw"]))
            lq.append(math.log(cg["q_boundary"] / ck["q_boundary"]))
        mw, mq = sum(lw) / len(lw), sum(lq) / len(lq)
        share = mq / mw if abs(mw) > 1e-9 else float("nan")
        print(f"| {label} | {math.exp(mw):.3f} | {math.exp(mq):.3f} | {math.exp(mw - mq):.3f} | {100 * share:.0f}% |")


if __name__ == "__main__":
    main()
