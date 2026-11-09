"""Syn-Atom (Gate M0, pre-registered): latent atoms + bimodal/skewed interior, continuous target.

Each of d = 2 latent coordinates is exactly 0 with prob 0.15, exactly 1 with prob 0.10,
and otherwise drawn from a skewed bimodal mixture 0.6 Beta(2, 8) + 0.4 Beta(9, 3).
T = 50 bounded observations x_t ~ TN(h, 0.2^2) give a posterior std of about 0.03.
The target y = g(h) + TN noise 0.03 with g mapping into (0.05, 0.95), so y has no atoms:
this is a "latent atoms, no output atoms" scenario, like Bucket.
"""
from dataclasses import asdict, dataclass

import torch

from . import tn
from .data_synthetic import target_mean


@dataclass
class AtomConfig:
    name: str = "syn_atom"
    d: int = 2
    T: int = 50
    n_train: int = 10_000
    n_cal: int = 2_000
    n_test: int = 2_000
    p0: float = 0.15
    p1: float = 0.10
    obs_noise: float = 0.2
    y_noise: float = 0.03
    data_seed: int = 0
    map_seed: int = 777


def _latent(cfg, n, gen):
    u = torch.rand(n, cfg.d, generator=gen)
    b1 = torch.distributions.Beta(2.0, 8.0).sample((n, cfg.d))
    b2 = torch.distributions.Beta(9.0, 3.0).sample((n, cfg.d))
    interior = torch.where(torch.rand(n, cfg.d, generator=gen) < 0.6, b1, b2)
    return torch.where(u < cfg.p0, torch.zeros_like(u), torch.where(u < cfg.p0 + cfg.p1, torch.ones_like(u), interior))


def generate(cfg: AtomConfig):
    torch.manual_seed(cfg.data_seed)          # Beta sampling uses the global generator
    g_map = torch.Generator().manual_seed(cfg.map_seed)
    W1 = 3.0 * torch.randn(cfg.d, 16, generator=g_map)
    b1 = torch.randn(16, generator=g_map)
    w2 = torch.randn(16, generator=g_map) / 4.0
    gen = torch.Generator().manual_seed(cfg.data_seed)
    out = {"config": asdict(cfg)}
    for split, n in (("train", cfg.n_train), ("cal", cfg.n_cal), ("test", cfg.n_test)):
        h = _latent(cfg, n, gen)
        loc = h.unsqueeze(1).expand(n, cfg.T, cfg.d)
        x = tn.rsample(loc, torch.full_like(loc, cfg.obs_noise), torch.rand(loc.shape, generator=gen))
        mu = target_mean(h, W1, b1, w2)
        y = tn.rsample(mu, torch.full_like(mu, cfg.y_noise), torch.rand(n, generator=gen)).unsqueeze(-1)
        out[split] = {"x": x, "s": torch.zeros(n, cfg.T, dtype=torch.long), "y": y, "h_true": h}
    return out
