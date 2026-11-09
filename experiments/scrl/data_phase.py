"""Controlled synthetic benchmark for the support-expressiveness phase diagram.

Two knobs, fixed a priori (PLAN.md, Stage 2):
  * C   -- nominal posterior std: T noisy bounded observations of a static latent with
           obs noise C*sqrt(T), so the Gaussian posterior std is about C;
  * p_b -- fraction of boundary units, whose latent lies within one nominal posterior
           std of a bound; the rest sit in the interior [0.3, 0.7].
Large p_b drives Gaussian leakage up; small C demands concentrated posteriors, which
some bounded families struggle to represent.
"""
from dataclasses import asdict, dataclass

import torch

from . import tn
from .data_synthetic import target_mean


@dataclass
class PhaseConfig:
    post_std: float = 0.05
    p_boundary: float = 0.4
    T: int = 50
    n_train: int = 5_000
    n_cal: int = 2_000
    n_test: int = 2_000
    y_noise: float = 0.03
    data_seed: int = 0
    map_seed: int = 12345          # target map g is fixed across all cells and seeds

    @property
    def name(self):
        return f"phase/pb{self.p_boundary:g}_c{self.post_std:g}"

    @property
    def obs_noise(self):
        return self.post_std * self.T ** 0.5


def _latents(cfg, n, gen):
    boundary = torch.rand(n, generator=gen) < cfg.p_boundary
    near = cfg.post_std * torch.rand(n, generator=gen)
    upper = torch.rand(n, generator=gen) < 0.5
    h_b = torch.where(upper, 1 - near, near)
    h_i = 0.3 + 0.4 * torch.rand(n, generator=gen)
    return torch.where(boundary, h_b, h_i).unsqueeze(-1)


def generate(cfg: PhaseConfig):
    g_map = torch.Generator().manual_seed(cfg.map_seed)
    W1 = 3.0 * torch.randn(1, 16, generator=g_map)
    b1 = torch.randn(16, generator=g_map)
    w2 = torch.randn(16, generator=g_map) / 4.0
    gen = torch.Generator().manual_seed(cfg.data_seed)
    out = {"config": asdict(cfg) | {"name": cfg.name, "obs_noise": cfg.obs_noise}}
    for split, n in (("train", cfg.n_train), ("cal", cfg.n_cal), ("test", cfg.n_test)):
        h = _latents(cfg, n, gen)                                     # [n, 1]
        loc = h.unsqueeze(1).expand(n, cfg.T, 1)
        x = tn.rsample(loc, torch.full_like(loc, cfg.obs_noise), torch.rand(loc.shape, generator=gen))
        mu_y = target_mean(h, W1, b1, w2)
        y = tn.rsample(mu_y, torch.full_like(mu_y, cfg.y_noise), torch.rand(n, generator=gen)).unsqueeze(-1)
        out[split] = {"x": x, "s": torch.zeros(n, cfg.T, dtype=torch.long), "y": y, "h_true": h}
    return out
