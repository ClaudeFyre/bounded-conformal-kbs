"""Empirical check of Theorem 3's assumptions on the Gate B setting (Gaussian model vs its truncation q*).

For each discovery dataset and seed (TN-hetero Gaussian checkpoints), on the observation-grouped boundary
group: (C) symmetry of the normalised residual under q^G (skewness, median); (R) dispersion of the unit-wise
scale ratio sigma*/sigma^G; and the decomposition log(W*/W^G) = log R_q + log R_sigma. Exploratory.
"""
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

S = 128


@torch.no_grad()
def main():
    import argparse
    import run_synthetic
    out = {}
    for ds in cf.DATASETS:
        a = argparse.Namespace(dataset=ds, data_seed=0, smoke=False, seed=0, pb=0.4, post_std=0.05, level=0)
        data, name, d, K, obs_noise = run_synthetic.load_data(a)
        cal, test = data["cal"], data["test"]
        n_obs = cal["x"].shape[-1]
        rows = []
        for seed in cf.SEEDS:
            G = cf.load_model(name, "gaussian", d, n_obs, K, obs_noise, seed)
            res = {}
            for sp, D in (("cal", cal), ("test", test)):
                p = G.encode(D["x"], D["s"])
                torch.manual_seed(1000 * seed + (sp == "test"))
                hs = G.family.rsample(p, S)
                mu, sg = p["mu"], p["sigma"]
                hs_star = tn.rsample(mu.expand(S, *mu.shape), sg.expand(S, *sg.shape), torch.rand(S, *mu.shape))
                a_ = G.predict_from_samples(hs)
                e_ = G.predict_from_samples(hs_star)
                grp = boundary_distance(D["x"][:, -1, :d]) < 0.05
                res[sp] = (a_, e_, D["y"].view(-1), grp)
            (ac, ec, yc, gc), (at, et, yt, gt) = res["cal"], res["test"]
            eps = ((yc - ac["y_hat"].view(-1)) / ac["sigma"])[gc]
            m, sd = eps.mean(), eps.std()
            skew = (((eps - m) / sd) ** 3).mean().item()
            ratio = (et["sigma"] / at["sigma"])[gt]
            qa = conformal_quantile(((yc - ac["y_hat"].view(-1)).abs() / ac["sigma"])[gc], 0.1)
            qe = conformal_quantile(((yc - ec["y_hat"].view(-1)).abs() / ec["sigma"])[gc], 0.1)
            Wa, We = 2 * qa * at["sigma"][gt].mean().item(), 2 * qe * et["sigma"][gt].mean().item()
            rows.append({"seed": seed, "skew": skew, "median_eps": eps.median().item(),
                         "ratio_cv": (ratio.std() / ratio.mean()).item(), "ratio_mean": ratio.mean().item(),
                         "log_Wratio": math.log(We / Wa), "log_Rq": math.log(qe / qa),
                         "log_Rsigma": math.log(et["sigma"][gt].mean().item() / at["sigma"][gt].mean().item())})
        out[ds] = rows
        m = lambda k: st.median(r[k] for r in rows)
        print(f"{ds:12s} (C) skew {m('skew'):+.2f}, median eps {m('median_eps'):+.3f} | (R) ratio mean {m('ratio_mean'):.3f}, CV {m('ratio_cv'):.3f} | "
              f"log W*/W {m('log_Wratio'):+.3f} = log R_q {m('log_Rq'):+.3f} + log R_sigma {m('log_Rsigma'):+.3f} | "
              f"W* > W in {sum(r['log_Wratio'] > 0 for r in rows)}/{len(rows)}", flush=True)
    (ROOT / "results" / "theory_check.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
