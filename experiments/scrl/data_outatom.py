"""Syn-OutAtom (Gate O1, pre-registered): known, x-dependent output atoms at both bounds.

Latent h ~ U(0,1)^2, observed through T = 50 bounded measurements x_t ~ TN(h, 0.2^2)
(the Syn-Atom observation model). The target is
    y ~ pi0(h) d0 + pi1(h) d1 + pic(h) TN(g(h), 0.05^2; [0,1]),
    pi0 = 0.35 * sigmoid(4 (0.3 - h1)),  pi1 = 0.25 * sigmoid(4 (h2 - 0.7)),
so atom probabilities are known and vary with x (ground truth for atom precision/ECE).
"""
from dataclasses import asdict, dataclass

import torch

from . import tn
from .data_synthetic import target_mean


@dataclass
class OutAtomConfig:
    name: str = "syn_outatom"
    d: int = 2
    T: int = 50
    n_train: int = 10_000
    n_cal: int = 2_000
    n_test: int = 2_000
    obs_noise: float = 0.2
    y_noise: float = 0.05
    data_seed: int = 0
    map_seed: int = 778


def atom_probs(h):
    pi0 = 0.35 * torch.sigmoid(4 * (0.3 - h[:, 0]))
    pi1 = 0.25 * torch.sigmoid(4 * (h[:, 1] - 0.7))
    return pi0, pi1


def generate(cfg: OutAtomConfig):
    g_map = torch.Generator().manual_seed(cfg.map_seed)
    W1 = 3.0 * torch.randn(cfg.d, 16, generator=g_map)
    b1 = torch.randn(16, generator=g_map)
    w2 = torch.randn(16, generator=g_map) / 4.0
    gen = torch.Generator().manual_seed(cfg.data_seed)
    out = {"config": asdict(cfg)}
    for split, n in (("train", cfg.n_train), ("cal", cfg.n_cal), ("test", cfg.n_test)):
        h = torch.rand(n, cfg.d, generator=gen)
        loc = h.unsqueeze(1).expand(n, cfg.T, cfg.d)
        x = tn.rsample(loc, torch.full_like(loc, cfg.obs_noise), torch.rand(loc.shape, generator=gen))
        pi0, pi1 = atom_probs(h)
        mu = target_mean(h, W1, b1, w2)
        yc = tn.rsample(mu, torch.full_like(mu, cfg.y_noise), torch.rand(n, generator=gen)).clamp(1e-4, 1 - 1e-4)
        u = torch.rand(n, generator=gen)
        y = torch.where(u < pi0, torch.zeros(n), torch.where(u < pi0 + pi1, torch.ones(n), yc)).unsqueeze(-1)
        out[split] = {"x": x, "s": torch.zeros(n, cfg.T, dtype=torch.long), "y": y, "h_true": h,
                      "pi0_true": pi0, "pi1_true": pi1}
    return out
