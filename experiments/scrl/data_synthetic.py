"""Synthetic regime-switching data with known bounded latent states.

Each unit is one trajectory. The conformal target is y = g(h_T) observed
with bounded noise, so every unit contributes one prediction, one score
and one group label (trajectory-level exchangeability by construction).
"""
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from . import tn


@dataclass
class SynConfig:
    name: str = "syn_full"
    d: int = 4                 # latent = observation dimension
    K: int = 4                 # number of regimes
    T: int = 200               # trajectory length
    n_train: int = 10_000
    n_cal: int = 2_000
    n_test: int = 2_000
    weibull_shape: float = 1.5
    weibull_scale: float = 50.0
    rho_min: float = 0.5       # spectral radius range of the regime dynamics
    rho_max: float = 0.95
    proc_noise: float = 0.03
    obs_noise: float = 0.05
    y_noise: float = 0.03
    extreme: bool = False      # boundary-heavy regime equilibria
    data_seed: int = 0         # fixes regimes and target map (the "physics")


def syn_extreme_config(**kw):
    base = dict(name="syn_extreme", extreme=True, proc_noise=0.02)
    base.update(kw)
    return SynConfig(**base)


def _random_dynamics(cfg, gen):
    d, K = cfg.d, cfg.K
    A = torch.empty(K, d, d)
    for k in range(K):
        q, _ = torch.linalg.qr(torch.randn(d, d, generator=gen))
        rho = cfg.rho_min + (cfg.rho_max - cfg.rho_min) * torch.rand(d, generator=gen)
        A[k] = q @ torch.diag(rho) @ q.T  # symmetric => spectral radius = max rho
    if cfg.extreme:
        c = 0.2 + 0.6 * torch.rand(K, d, generator=gen)
        for k in range(K):
            n_edge = 1 + int(torch.randint(0, d, (1,), generator=gen))
            dims = torch.randperm(d, generator=gen)[:n_edge]
            side = torch.randint(0, 2, (n_edge,), generator=gen).float()
            c[k, dims] = side * (0.97 + 0.03 * torch.rand(n_edge, generator=gen)) + (1 - side) * (
                0.03 * torch.rand(n_edge, generator=gen)
            )
    else:
        c = torch.rand(K, d, generator=gen)
    return A, c


def _target_map(cfg, gen, hidden=16):
    W1 = 3.0 * torch.randn(cfg.d, hidden, generator=gen)
    b1 = torch.randn(hidden, generator=gen)
    w2 = torch.randn(hidden, generator=gen) / hidden**0.5
    return W1, b1, w2


def target_mean(h, W1, b1, w2):
    """g: [0,1]^d -> (0.05, 0.95); 2-layer MLP with sinusoidal activation."""
    return 0.05 + 0.9 * torch.sigmoid(2.0 * torch.sin(h @ W1 + b1) @ w2)


def _simulate(cfg, n, A, c, gen):
    d, K, T = cfg.d, cfg.K, cfg.T
    h = torch.empty(n, T, d)
    s = torch.empty(n, T, dtype=torch.long)
    reg = torch.randint(0, K, (n,), generator=gen)

    # Weibull regime durations via inverse CDF, so the generator controls them.
    def draw_duration(m):
        u = torch.rand(m, generator=gen).clamp(1e-7, 1 - 1e-7)
        return torch.ceil(cfg.weibull_scale * (-torch.log1p(-u)) ** (1 / cfg.weibull_shape)).long()

    remaining = draw_duration(n)
    cur = c[reg] + 0.05 * torch.randn(n, d, generator=gen)
    cur = cur.clamp(0, 1)
    for t in range(T):
        switch = remaining <= 0
        if switch.any():
            m = int(switch.sum())
            shift = torch.randint(1, K, (m,), generator=gen)
            reg[switch] = (reg[switch] + shift) % K
            remaining[switch] = draw_duration(m)
        ck = c[reg]
        cur = ck + torch.einsum("nij,nj->ni", A[reg], cur - ck) + cfg.proc_noise * torch.randn(n, d, generator=gen)
        cur = cur.clamp(0, 1)  # hard physical limit (e.g. zero flow): genuine point mass
        h[:, t] = cur
        s[:, t] = reg
        remaining -= 1
    u = torch.rand(n, T, d, generator=gen)
    x = tn.rsample(h, torch.full_like(h, cfg.obs_noise), u)
    return h, x, s


def generate(cfg: SynConfig, cache_dir=None):
    cache = None
    if cache_dir is not None:
        cache = Path(cache_dir) / f"{cfg.name}_seed{cfg.data_seed}.pt"
        if cache.exists():
            return torch.load(cache, weights_only=True)
    gen = torch.Generator().manual_seed(cfg.data_seed)
    A, c = _random_dynamics(cfg, gen)
    W1, b1, w2 = _target_map(cfg, gen)
    out = {"config": asdict(cfg), "A": A, "c": c, "g": (W1, b1, w2)}
    for i, (split, n) in enumerate([("train", cfg.n_train), ("cal", cfg.n_cal), ("test", cfg.n_test)]):
        g_split = torch.Generator().manual_seed(10_000 * (cfg.data_seed + 1) + i)
        h, x, s = _simulate(cfg, n, A, c, g_split)
        mu_y = target_mean(h[:, -1], W1, b1, w2)
        u = torch.rand(n, generator=g_split)
        y = tn.rsample(mu_y, torch.full_like(mu_y, cfg.y_noise), u)
        out[split] = {"x": x, "s": s, "y": y.unsqueeze(-1), "h_true": h[:, -1]}
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(out, cache)
    return out


def boundary_distance(h):
    return torch.minimum(h, 1 - h).min(dim=-1).values
