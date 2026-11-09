"""Split Mondrian conformal prediction with normalised scores."""
import math

import torch

from .data_synthetic import boundary_distance


def scores(y, y_hat, sigma):
    return torch.linalg.vector_norm(y - y_hat, dim=-1) / sigma


def conformal_quantile(s, alpha):
    """The ceil((1-alpha)(n+1))-th smallest score; +inf if the group is too small."""
    n = s.numel()
    k = math.ceil((1 - alpha) * (n + 1))
    if n == 0 or k > n:
        return float("inf")
    return torch.sort(s).values[k - 1].item()


def groups_from_mean(post_mean, eps):
    return (boundary_distance(post_mean) < eps).long()  # 1 = boundary, 0 = interior


GROUPINGS = {
    # own posterior mean: the deployed pipeline, but groups differ across models
    "latent": "post_mean",
    # last observation: model-independent and available on real data
    "obs": "x_last",
    # data-generating state: model-independent oracle, synthetic data only
    "oracle": "h_true",
}


def mondrian_eval_all(cal, test, eps, alpha=0.1):
    return {name: mondrian_eval(cal, test, eps, alpha, key) for name, key in GROUPINGS.items() if key in cal}


def mondrian_eval(cal, test, eps, alpha=0.1, group_key="post_mean"):
    """cal/test: dicts with y, y_hat, sigma, post_mean, x_last, h_true (tensors on CPU)."""
    g_cal = groups_from_mean(cal[group_key], eps)
    g_test = groups_from_mean(test[group_key], eps)
    s_cal = scores(cal["y"], cal["y_hat"], cal["sigma"])
    s_test = scores(test["y"], test["y_hat"], test["sigma"])
    q = torch.tensor([conformal_quantile(s_cal[g_cal == g], alpha) for g in (0, 1)])

    radius = q[g_test] * test["sigma"]
    covered = s_test <= q[g_test]
    width = 2 * radius
    y_hat = test["y_hat"].squeeze(-1)
    clipped = (y_hat + radius).clamp(max=1) - (y_hat - radius).clamp(min=0)

    b = g_test == 1
    # real data has no ground-truth latent: true-boundary metrics are then undefined
    true_bnd = boundary_distance(test["h_true"]) < eps if "h_true" in test else torch.zeros_like(b)

    def mean(v, mask):
        return v[mask].float().mean().item() if mask.any() else float("nan")

    return {
        "eps": eps, "grouping": group_key,
        "q_interior": q[0].item(), "q_boundary": q[1].item(),
        "n_cal_boundary": int((g_cal == 1).sum()), "n_test_boundary": int(b.sum()),
        "n_test": int(g_test.numel()),
        "coverage": covered.float().mean().item(),
        "coverage_boundary": mean(covered, b),
        "coverage_true_boundary": mean(covered, true_bnd),
        "biw": mean(width, b),
        "biw_clipped": mean(clipped, b),
        "width_interior": mean(width, ~b),
        "width_true_boundary": mean(width, true_bnd),
        # grouping vs the data-generating latent h°
        "fp_vs_true": mean((b & ~true_bnd), torch.ones_like(b)),
        "fn_vs_true": mean((~b & true_bnd), torch.ones_like(b)),
    }
