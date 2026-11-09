"""Sanity checks for the closed forms used in the paper and the code.

    python -m pytest tests -q
"""
import sys
from pathlib import Path

import torch
from torch.special import ndtr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scrl import tn  # noqa: E402
from scrl.data_synthetic import boundary_distance  # noqa: E402
from scrl.latents import FAMILIES, get_family  # noqa: E402

torch.manual_seed(0)
S = 200_000


def test_kumaraswamy_kl_matches_monte_carlo():
    fam = get_family("kumaraswamy")
    for a, b in [(0.5, 0.7), (1.0, 1.0), (2.0, 5.0), (8.0, 1.5)]:
        p = {"a": torch.tensor([[a]]), "b": torch.tensor([[b]])}
        h = fam.rsample(p, S).double()
        a_, b_ = torch.tensor(a, dtype=torch.double), torch.tensor(b, dtype=torch.double)
        logq = torch.log(a_ * b_) + (a_ - 1) * torch.log(h) + (b_ - 1) * torch.log1p(-h**a_)
        assert abs(fam.kl(p).item() - logq.mean().item()) < 0.02, (a, b)


def test_posterior_means_match_samples():
    for name in FAMILIES:
        fam = get_family(name)
        raw = torch.randn(3, 4)
        p = fam.params(raw)
        hs = fam.rsample(p, S)
        tol = 5 * hs.std(0) / S**0.5 + 1e-4       # 5 Monte Carlo standard errors
        assert ((fam.mean(p) - hs.mean(0)).abs() <= tol).all(), name


def test_tn_moments_and_entropy():
    loc = torch.tensor([-0.3, 0.05, 0.5, 0.9, 1.4])
    scale = torch.tensor([0.2, 0.1, 0.3, 0.05, 0.4])
    x = tn.rsample(loc, scale, torch.rand(S, 5)).double()
    assert torch.allclose(tn.mean(loc, scale).double(), x.mean(0), atol=3e-3)
    assert torch.allclose(tn.variance(loc, scale).double(), x.var(0), atol=3e-3)
    ent_mc = -tn.log_prob(x.float(), loc, scale).double().mean(0)
    assert torch.allclose(tn.entropy(loc, scale).double(), ent_mc, atol=2e-2)


def test_gaussian_leakage_formula():
    fam = get_family("gaussian")
    p = {"mu": torch.tensor([[0.05, 0.5, 0.9]]), "sigma": torch.tensor([[0.1, 0.2, 0.15]])}
    h = fam.rsample(p, S)
    mc = 1 - ((h >= 0) & (h <= 1)).all(-1).float().mean()
    assert abs(fam.leakage(p).item() - mc.item()) < 5e-3


def test_prop3_truncation_moves_mean_inward_and_errors_are_one_sided():
    mu = torch.rand(100_000, 4) * 1.6 - 0.3          # includes means outside [0,1]
    sigma = torch.rand(100_000, 4) * 0.4 + 0.01
    mu, sigma = mu.double(), sigma.double()
    h_star = tn.mean(mu, sigma)
    D = boundary_distance(h_star) - boundary_distance(mu)
    assert (D >= 0).all()
    # Per coordinate, truncation strictly increases the distance to the nearer bound wherever the shift is
    # numerically visible (theory: whenever mu_j != 1/2; for tiny sigma it is exp(-large) and underflows).
    gain = torch.minimum(h_star, 1 - h_star) - torch.minimum(mu, 1 - mu)
    leak_j = 1 - (ndtr((1 - mu) / sigma) - ndtr(-mu / sigma))
    assert (gain >= 0).all()
    assert (gain[(leak_j > 1e-6) & ((mu - 0.5).abs() > 1e-3)] > 0).all()
    for eps in (0.025, 0.05, 0.1, 0.15):
        g_hat = boundary_distance(mu) < eps
        g_star = boundary_distance(h_star) < eps
        assert not (~g_hat & g_star).any()           # no false negatives relative to h*
