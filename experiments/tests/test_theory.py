"""Numerical checks of the Stage 1 propositions (theory/ASCL_theory.md), on a random decoder.

    python -m pytest tests -q
"""
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scrl import tn  # noqa: E402

torch.manual_seed(0)
S = 400_000
P = 2


def decoder(h):
    """Random smooth decoder R^1 -> [0,1]^2 (mean) and a positive variance."""
    m = torch.sigmoid(torch.cat([3 * h - 1, torch.sin(4 * h)], -1))
    v = 0.01 + 0.02 * torch.sigmoid(h).expand(*h.shape[:-1], P)
    return m, v


def moments(h):
    m, v = decoder(h)
    y = m.mean(0)
    W = v.mean()
    V = ((m - y) ** 2).sum(-1).mean() / P
    return y, W, V


def split(mu=0.03, sigma=0.12):
    hG = (mu + sigma * torch.randn(S, 1)).double()
    inside = (hG >= 0) & (hG <= 1)
    delta = inside.logical_not().double().mean()
    return hG, hG[inside.squeeze(-1)], hG[~inside.squeeze(-1)], delta


def test_prop_A_prediction_displacement():
    hG, hs, ho, delta = split()
    yG, _, _ = moments(hG)
    ys, _, _ = moments(hs)
    yo, _, _ = moments(ho)
    assert torch.allclose(yG - ys, delta * (yo - ys), atol=1e-9)


def test_prop_B_scale_path_exact():
    hG, hs, ho, delta = split()
    yG, WG, VG = moments(hG)
    ys, Ws, Vs = moments(hs)
    yo, Wo, Vo = moments(ho)
    Delta = yG - ys
    assert abs((WG - Ws) - delta * (Wo - Ws)) < 1e-9
    rhs = delta * (Vo - Vs) + (1 - delta) / delta * (Delta ** 2).sum() / P
    assert abs((VG - Vs) - rhs) < 1e-9


def test_prop_C_scores_shrink_when_scale_absorbs():
    # construct units where kappa exceeds the pointwise threshold and check scores do not increase
    r_star = torch.rand(1000, P) * 0.5 + 0.1
    Delta = torch.full((P,), 0.01)
    ratio = Delta.norm() / r_star.norm(dim=-1)
    kappa = (2 * ratio + ratio ** 2).max() + 1e-3
    sigma_star = torch.rand(1000) + 0.5
    sigma_G = sigma_star * math.sqrt(1 + kappa)
    sG = (r_star - Delta).norm(dim=-1) / sigma_G
    ss = r_star.norm(dim=-1) / sigma_star
    assert (sG <= ss + 1e-12).all()


def test_prop_D_pinsker_bound():
    # q* = truncated normal; a slightly different truncated normal plays the bounded approximation q^B
    loc, sc = torch.tensor([0.05]).double(), torch.tensor([0.1]).double()
    loc2, sc2 = torch.tensor([0.08]).double(), torch.tensor([0.09]).double()
    hs = tn.rsample(loc, sc, torch.rand(S, 1).double()).unsqueeze(-1)
    hb = tn.rsample(loc2, sc2, torch.rand(S, 1).double()).unsqueeze(-1)
    kl = (tn.log_prob(hs, loc, sc) - tn.log_prob(hs, loc2, sc2)).mean()     # KL(q* || q^B)
    ys, _, _ = moments(hs)
    yb, _, _ = moments(hb)
    R = math.sqrt(P)
    assert (yb - ys).norm() <= R * math.sqrt(max(kl.item(), 0) / 2) + 3e-3   # Monte Carlo slack
