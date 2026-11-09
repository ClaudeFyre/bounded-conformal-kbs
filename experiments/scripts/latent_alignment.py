"""Latent alignment check for the counterfactual swap (post hoc diagnostic; plan fixed in this docstring
before the first run, committed before running).

Question. The swap (counterfactual.py) combines the encoder of one model with the decoder of another.
This is interpretable only if the two models use the same latent coordinates. In the model, every
latent coordinate h_j is tied to the measured state variable x_T,j by the fixed observation model
x_T,j ~ TN(h_j, obs_noise^2) on [0,1] (scrl/models.py), which has no learned parameters, so the
coordinates are identified up to that noise by construction. This script measures how well the
Gaussian and bounded encoders actually agree.

Metrics, per dataset x seed x bounded family, on the test units of the swap (same checkpoints, same
data split as counterfactual.py):
  corr_min   minimum over latent coordinates of the Pearson correlation between the posterior means
             of the Gaussian and the bounded encoder;
  mad_GB     mean absolute difference between the two posterior means, over units and coordinates;
  mad_G_obs, mad_B_obs   mean absolute difference between each posterior mean and x_T (reference).

Criterion fixed before running: alignment is called adequate on a dataset if the median over
seeds x families of corr_min is at least 0.9 and the median of mad_GB is at most obs_noise.
The result is reported whatever it is; it does not change any swap number.

    python scripts/latent_alignment.py     # writes results/latent_alignment.json
"""
import argparse
import json
import statistics as st
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from counterfactual import DATASETS, FAMILIES, SEEDS, load_model  # noqa: E402

OUT = ROOT / "results" / "latent_alignment.json"
CORR_MIN, MAD_FACTOR = 0.9, 1.0


def corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    den = (a.norm() * b.norm()).item()
    return float((a * b).sum().item() / den) if den > 0 else float("nan")


@torch.no_grad()
def main():
    import run_synthetic
    rows, summ = [], {}
    for ds in DATASETS:
        a = argparse.Namespace(dataset=ds, data_seed=0, smoke=False, seed=0, pb=0.4, post_std=0.05)
        data, name, d, K, obs_noise = run_synthetic.load_data(a)
        test = data["test"]
        n_obs = test["x"].shape[-1]
        xT = test["x"][:, -1, :d]
        for seed in SEEDS:
            G = load_model(name, "gaussian", d, n_obs, K, obs_noise, seed)
            mG = G.family.mean(G.encode(test["x"], test["s"]))
            for fam in FAMILIES:
                B = load_model(name, fam, d, n_obs, K, obs_noise, seed)
                mB = B.family.mean(B.encode(test["x"], test["s"]))
                rows.append({
                    "dataset": ds, "seed": seed, "family": fam, "d": d, "obs_noise": obs_noise,
                    "corr_min": min(corr(mG[:, j], mB[:, j]) for j in range(d)),
                    "mad_GB": float((mG - mB).abs().mean()),
                    "mad_G_obs": float((mG - xT).abs().mean()),
                    "mad_B_obs": float((mB - xT).abs().mean()),
                })
        r = [x for x in rows if x["dataset"] == ds]
        med = {k: st.median(x[k] for x in r) for k in ("corr_min", "mad_GB", "mad_G_obs", "mad_B_obs")}
        med["min_corr_min"] = min(x["corr_min"] for x in r)
        med["max_mad_GB"] = max(x["mad_GB"] for x in r)
        med["obs_noise"] = obs_noise
        med["adequate"] = bool(med["corr_min"] >= CORR_MIN and med["mad_GB"] <= MAD_FACTOR * obs_noise)
        summ[ds] = med
        print(ds, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in med.items()}, flush=True)
    OUT.write_text(json.dumps({"criterion": {"corr_min": CORR_MIN, "mad_GB_over_obs_noise": MAD_FACTOR},
                               "summary": summ, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
