"""Representation geometry: attainable (mean, std) region of each latent family vs data operating points.

Part 1: for each family, under the exact parameterisation and caps used in scrl/latents.py,
        the minimum attainable posterior std at each mean.
Part 2: operating points (mu, sigma) required by each diagnostic dataset, read off the trained
        Gaussian encoders (seed 0, test units, per latent coordinate).
Output: figures/geometry.png and a table of the fraction of operating points that each
        family cannot reach (sigma_required < sigma_min_family(mu)).
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scrl.latents import CONC_MAX  # noqa: E402
from scrl.models import Model  # noqa: E402

MEANS = np.linspace(0.005, 0.995, 199)
LOG_SHAPE = np.linspace(-6.0, math.log(CONC_MAX), 700)
DATASETS = ["syn_full", "syn_extreme", "bucket", "seaice", "resops"]


def _lbeta(x, y):
    return math.lgamma(x) + math.lgamma(y) - math.lgamma(x + y)


def kumaraswamy_min_std():
    """Grid over (log a, log b); moments E[h^k] = b B(1 + k/a, b)."""
    la, lb = np.meshgrid(LOG_SHAPE, LOG_SHAPE, indexing="ij")
    a, b = np.exp(la).ravel(), np.exp(lb).ravel()
    from scipy.special import betaln
    m1 = np.exp(np.log(b) + betaln(1 + 1 / a, b))
    m2 = np.exp(np.log(b) + betaln(1 + 2 / a, b))
    sd = np.sqrt(np.clip(m2 - m1**2, 0, None))
    out = np.full(len(MEANS), np.nan)
    idx = np.digitize(m1, (MEANS[1:] + MEANS[:-1]) / 2)
    for i in range(len(MEANS)):
        sel = idx == i
        if sel.any():
            out[i] = sd[sel].min()
    return out


def beta_min_std():
    n = CONC_MAX / np.maximum(MEANS, 1 - MEANS)           # a = m n <= CAP and b = (1-m) n <= CAP
    return np.sqrt(MEANS * (1 - MEANS) / (n + 1))


def min_std_curves():
    # Gaussian, truncated Gaussian and logit-normal reach std ~1e-3 or below at every mean
    # (scale floor 1e-3; logit-normal std ~ m(1-m) s), so only Beta and Kumaraswamy are binding.
    return {"beta": beta_min_std(), "kumaraswamy": kumaraswamy_min_std()}


@torch.no_grad()
def operating_points(dataset):
    import run_synthetic
    a = argparse.Namespace(dataset=dataset, data_seed=0, smoke=False, seed=0, pb=0.4, post_std=0.05)
    data, name, d, K, obs_noise = run_synthetic.load_data(a)
    test = data["test"]
    ck = ROOT / "checkpoints" / name / "gaussian__tn_hetero__raw__s0.pt"
    model = Model("gaussian", d=d, n_obs=test["x"].shape[-1], K=K, decoder="tn_hetero", obs_noise=obs_noise)
    model.load_state_dict(torch.load(ck, map_location="cpu", weights_only=True))
    model.eval()
    p = model.encode(test["x"], test["s"])
    return p["mu"].reshape(-1).numpy(), p["sigma"].reshape(-1).numpy()


def main():
    curves = min_std_curves()
    rows = []
    points = {}
    for ds in DATASETS:
        mu, sd = operating_points(ds)
        points[ds] = (mu, sd)
        inside = (mu > 0) & (mu < 1)
        row = {"dataset": ds, "n": int(len(mu)), "mean_outside_[0,1]": float(1 - inside.mean()),
               "median_sigma": float(np.median(sd))}
        for fam, curve in curves.items():
            need = np.interp(np.clip(mu, MEANS[0], MEANS[-1]), MEANS, np.nan_to_num(curve, nan=1.0))
            row[f"unreachable_{fam}"] = float(((sd < need) & inside).sum() / max(inside.sum(), 1))
        rows.append(row)

    print("| dataset | points | mean outside [0,1] | median sigma | unreachable by Beta | unreachable by Kumaraswamy |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['dataset']} | {r['n']} | {r['mean_outside_[0,1]']:.3f} | {r['median_sigma']:.4f} | "
              f"{r['unreachable_beta']:.3f} | {r['unreachable_kumaraswamy']:.3f} |")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    colors = {"syn_full": "#4C72B0", "syn_extreme": "#55A868", "bucket": "#C44E52", "seaice": "#8172B2", "resops": "#CCB974"}
    rng = np.random.default_rng(0)
    for ds, (mu, sd) in points.items():
        keep = rng.choice(len(mu), min(1500, len(mu)), replace=False)
        ax.scatter(np.clip(mu[keep], 0, 1), sd[keep], s=4, alpha=0.35, color=colors[ds], label=f"{ds} (Gaussian posterior)")
    ax.plot(MEANS, curves["kumaraswamy"], color="black", lw=2, label="Kumaraswamy min std (a, b ≤ 1e4)")
    ax.plot(MEANS, curves["beta"], color="black", lw=1.5, ls="--", label="Beta min std (a, b ≤ 1e4)")
    ax.set_yscale("log")
    ax.set_xlabel("posterior mean (clipped to [0, 1] for display)")
    ax.set_ylabel("posterior std")
    ax.set_title("Attainable region vs required operating points")
    ax.legend(fontsize=7, loc="lower center", ncol=2, markerscale=3)
    out = ROOT / "figures"
    out.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(out / "geometry.png", dpi=160)
    print(f"saved {out / 'geometry.png'}")


if __name__ == "__main__":
    main()
