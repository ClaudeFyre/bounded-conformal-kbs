"""Review response B (PLAN.md, 2026-10-23, with amendment B-1): exploratory check of Theorem 2 on the post-hoc
truncation of the trained Gaussian models. Same setting as theory_check.py (TN-hetero Gaussian checkpoints, seeds
0-9, observation grouping, eps = 0.05, S = 128, same sampling seeds). Per seed, on the boundary calibration units:
  eps_i      normalized residual under q^G;                 b*  center of the shortest interval holding the
  b_i        normalized center movement (yhat* - yhat)/sigma      conformal number of eps_i
  shift      log[Q(|eps - b_i|) / Q(|eps|)]  (shift only, scale held fixed)
  remainder  log R_q - shift                 (scale heterogeneity)
  pred       log[h(median b) / h(0)]         (Theorem 2(ii) under a common-shift approximation)
  eta        h(b*) / h(0)                    (Theorem 2(i): no center correction can narrow more than this)
Quantiles are the split-conformal order statistics used throughout. Writes results/shift_check.json.
"""
import argparse
import json
import math
import statistics as st
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import counterfactual as cf  # noqa: E402
from scrl import tn  # noqa: E402
from scrl.conformal import conformal_quantile  # noqa: E402
from scrl.data_synthetic import boundary_distance  # noqa: E402

S, ALPHA = 128, 0.1


def shortest_center(e):
    """Center and half-length of the shortest interval containing the conformal number of sorted values."""
    v, _ = torch.sort(e)
    n = len(v)
    k = math.ceil((1 - ALPHA) * (n + 1))
    if k > n:
        return float("nan"), float("inf")
    lengths = v[k - 1:] - v[: n - k + 1]
    i = int(torch.argmin(lengths))
    return float((v[i] + v[i + k - 1]) / 2), float(lengths[i] / 2)


@torch.no_grad()
def main():
    import run_synthetic
    out = {}
    for ds in cf.DATASETS:
        a = argparse.Namespace(dataset=ds, data_seed=0, smoke=False, seed=0, pb=0.4, post_std=0.05, level=0)
        data, name, d, K, obs_noise = run_synthetic.load_data(a)
        cal = data["cal"]
        n_obs = cal["x"].shape[-1]
        rows = []
        for seed in cf.SEEDS:
            G = cf.load_model(name, "gaussian", d, n_obs, K, obs_noise, seed)
            p = G.encode(cal["x"], cal["s"])
            torch.manual_seed(1000 * seed)                     # same draw as theory_check.py (calibration split)
            hs = G.family.rsample(p, S)
            mu, sg = p["mu"], p["sigma"]
            hs_star = tn.rsample(mu.expand(S, *mu.shape), sg.expand(S, *sg.shape), torch.rand(S, *mu.shape))
            A = G.predict_from_samples(hs)
            E = G.predict_from_samples(hs_star)
            grp = boundary_distance(cal["x"][:, -1, :d]) < 0.05
            y = cal["y"].view(-1)
            ya, sa = A["y_hat"].view(-1)[grp], A["sigma"].view(-1)[grp]
            ye, se = E["y_hat"].view(-1)[grp], E["sigma"].view(-1)[grp]
            yg = y[grp]
            eps = (yg - ya) / sa
            b = (ye - ya) / sa
            q0 = conformal_quantile(eps.abs(), ALPHA)
            q_shift = conformal_quantile((eps - b).abs(), ALPHA)
            q_e = conformal_quantile((yg - ye).abs() / se, ALPHA)
            b_star, half = shortest_center(eps)
            b_med = float(b.median())
            rows.append({
                "seed": seed, "n": int(grp.sum()), "b_star": b_star, "b_median": b_med,
                "frac_b_pos": float((b > 0).float().mean()),
                "log_Rq": math.log(q_e / q0), "log_shift": math.log(q_shift / q0),
                "log_remainder": math.log(q_e / q0) - math.log(q_shift / q0),
                "log_pred_common": math.log(conformal_quantile((eps - b_med).abs(), ALPHA) / q0),
                "eta": half / q0,
                "away": bool(b_med * b_star <= 0),
            })
        out[ds] = rows
        m = lambda k: st.median(r[k] for r in rows)
        print(f"{ds:12s} b* {m('b_star'):+.3f}  median b {m('b_median'):+.3f}  eta {m('eta'):.4f}  "
              f"log R_q {m('log_Rq'):+.3f} = shift {m('log_shift'):+.3f} + remainder {m('log_remainder'):+.3f}  "
              f"pred(common) {m('log_pred_common'):+.3f}", flush=True)
    (ROOT / "results" / "shift_check.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
