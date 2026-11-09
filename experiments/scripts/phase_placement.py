"""Where do the discovery datasets sit on the phase diagram's posterior-scale axis C? (exploratory, post hoc)

The phase generator fixes the nominal Gaussian posterior standard deviation C. For each discovery dataset we measure
the analogous quantity on the trained Gaussian models (TN-hetero checkpoints, seeds 0-9, test split): the
posterior standard deviation sigma(x), averaged over latent coordinates and test units. No training; read-only.

    python scripts/phase_placement.py   -> results/phase_placement.json
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

import counterfactual as cf  # noqa: E402


@torch.no_grad()
def main():
    import run_synthetic
    out = {}
    for ds in cf.DATASETS:
        a = argparse.Namespace(dataset=ds, data_seed=0, smoke=False, seed=0, pb=0.4, post_std=0.05)
        data, name, d, K, obs_noise = run_synthetic.load_data(a)
        test = data["test"]
        rows = []
        for seed in cf.SEEDS:
            G = cf.load_model(name, "gaussian", d, test["x"].shape[-1], K, obs_noise, seed)
            p = G.encode(test["x"], test["s"])
            rows.append({"seed": seed, "mean_sigma": p["sigma"].mean().item(),
                         "median_sigma": p["sigma"].median().item(),
                         "leakage": G.family.leakage(p).mean().item()})
        out[ds] = rows
        m = lambda k: st.median(r[k] for r in rows)
        print(f"{ds:12s} C (mean sigma) {m('mean_sigma'):.4f}  median sigma {m('median_sigma'):.4f}  "
              f"leakage {m('leakage'):.3f}", flush=True)
    (ROOT / "results" / "phase_placement.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
