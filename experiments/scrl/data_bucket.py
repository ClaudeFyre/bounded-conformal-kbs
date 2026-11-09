"""Semi-synthetic data: a two-layer soil-water bucket model driven by real weather.

Real meteorological forcing (daily precipitation and temperature, e.g. Daymet
at CAMELS basin locations) drives a bucket model with known capacities, so
  * the latent state h = (S1/S1max, S2/S2max) lies in [0,1]^2 by physics
    (empty = wilting point, full = saturation), and both bounds are hit by
    real dry spells and storms;
  * the true state is known exactly, as in the fully synthetic data.

Unit = one window of `T` days at one site with its own bucket parameters.
Sites are split disjointly into train / pool; calibration and test units are
drawn i.i.d. from the same pool, so they are exchangeable by construction.
The target is the next-day upper-layer moisture, observed with bounded noise.
"""
import math
from dataclasses import asdict, dataclass

import torch

from . import tn


@dataclass
class BucketConfig:
    name: str = "bucket"
    T: int = 200
    n_train: int = 10_000
    n_cal: int = 2_000
    n_test: int = 2_000
    train_site_frac: float = 0.6
    obs_noise: float = 0.05
    y_noise: float = 0.03
    data_seed: int = 0
    # parameter ranges (sampled per unit): capacities in mm, rates per day
    s1max: tuple = (20.0, 60.0)
    s2max: tuple = (80.0, 250.0)
    perc: tuple = (0.05, 0.3)       # fraction of relative upper storage percolating per day (scaled by S1max)
    base: tuple = (0.005, 0.03)     # baseflow fraction of lower storage per day
    et_split: tuple = (0.4, 0.8)    # share of PET drawn from the upper layer
    # ablation-ladder switches (defaults reproduce the original bucket dataset exactly)
    site_split: bool = True         # train sites disjoint from calibration/test sites
    param_hetero: bool = True       # bucket parameters sampled per unit (else range midpoints)
    real_forcing: bool = True       # real Daymet drivers, also observed (else i.i.d. rain, constant PET, unobserved)
    n_layers: int = 2               # 2 = upper + lower layer; 1 = upper layer only
    target: str = "next_state"      # "next_state" or "nowcast_map" (y = g(h_T), as in Syn-Full)
    map_seed: int = 12345


LADDER = {
    0: {},
    1: {"site_split": False},
    2: {"site_split": False, "param_hetero": False},
    3: {"site_split": False, "param_hetero": False, "real_forcing": False},
    4: {"site_split": False, "param_hetero": False, "real_forcing": False, "n_layers": 1},
    5: {"site_split": False, "param_hetero": False, "real_forcing": False, "n_layers": 1, "target": "nowcast_map"},
}


def ladder_config(level, data_seed):
    """Pre-registered ablation ladder (PLAN.md): each level adds one change to the previous one."""
    return BucketConfig(name=f"ladder_B{level}", data_seed=data_seed, **LADDER[level])


def hargreaves_pet(tmin, tmax, lat_deg, doy):
    """Daily PET [mm/day] by Hargreaves-Samani; extraterrestrial radiation from latitude and day of year."""
    phi = math.radians(lat_deg)
    dr = 1 + 0.033 * torch.cos(2 * math.pi * doy / 365)
    dec = 0.409 * torch.sin(2 * math.pi * doy / 365 - 1.39)
    ws = torch.arccos((-math.tan(phi) * torch.tan(dec)).clamp(-1, 1))
    ra = (24 * 60 / math.pi) * 0.0820 * dr * (
        ws * math.sin(phi) * torch.sin(dec) + math.cos(phi) * torch.cos(dec) * torch.sin(ws)
    )  # MJ m^-2 day^-1
    tmean = (tmin + tmax) / 2
    pet = 0.0023 * 0.408 * ra * (tmean + 17.8) * (tmax - tmin).clamp_min(0).sqrt()
    return pet.clamp_min(0)


def water_stress(rel, crit=0.5):
    """ET limitation factor: 1 above `crit` relative moisture, sqrt decline below it.

    A linear decline makes the dry bound unreachable (exponential decay); the sqrt form
    lets soil dry out to the wilting point in finite time, as real soils do.
    """
    return (rel.clamp(0, 1) / crit).clamp(max=1).sqrt()


def simulate_bucket(P, PET, p, n_layers=2):
    """P, PET: [n, L] forcing; p: dict of [n] parameters. Returns relative storages [n, L, n_layers].

    With n_layers = 1 only the upper layer is kept and percolation leaves the system.
    """
    n, L = P.shape
    s1 = 0.5 * p["s1max"]
    s2 = 0.5 * p["s2max"]
    out = torch.empty(n, L, 2)
    for t in range(L):
        s1 = s1 + P[:, t]
        et1 = p["et_split"] * PET[:, t] * water_stress(s1 / p["s1max"])
        s1 = s1 - et1
        perc = p["perc"] * p["s1max"] * (s1 / p["s1max"]).clamp(0, 1) ** 2
        s1 = s1 - perc
        s1 = torch.minimum(s1.clamp_min(0.0), p["s1max"])   # excess above capacity = surface runoff
        s2 = s2 + perc
        et2 = (1 - p["et_split"]) * PET[:, t] * water_stress(s2 / p["s2max"])
        s2 = torch.minimum((s2 - et2 - p["base"] * s2).clamp_min(0.0), p["s2max"])
        out[:, t, 0] = s1 / p["s1max"]
        out[:, t, 1] = s2 / p["s2max"]
    return out[..., :n_layers]


def _sample_params(cfg, n, gen):
    def u(rng):
        lo, hi = rng
        if not cfg.param_hetero:
            return torch.full((n,), (lo + hi) / 2)
        return lo + (hi - lo) * torch.rand(n, generator=gen)
    return {"s1max": u(cfg.s1max), "s2max": u(cfg.s2max), "perc": u(cfg.perc),
            "base": u(cfg.base), "et_split": u(cfg.et_split)}


def _make_units(cfg, forcing, sites, n, gen, spinup=365, pooled=None):
    """forcing: dict site -> (P [days], PET [days]) tensors. Returns one split."""
    L = spinup + cfg.T + 1
    site_idx = torch.randint(0, len(sites), (n,), generator=gen)
    P = torch.empty(n, L)
    PET = torch.empty(n, L)
    if cfg.real_forcing:
        for i, si in enumerate(site_idx.tolist()):
            p_s, pet_s = forcing[sites[si]]
            start = int(torch.randint(0, p_s.numel() - L, (1,), generator=gen))
            P[i], PET[i] = p_s[start:start + L], pet_s[start:start + L]
    else:
        # autonomous dynamics: i.i.d. daily rain from the pooled marginal, constant mean PET
        P_all, pet_mean = pooled
        P = P_all[torch.randint(0, P_all.numel(), (n, L), generator=gen)]
        PET = torch.full((n, L), pet_mean)
    params = _sample_params(cfg, n, gen)
    h = simulate_bucket(P, PET, params, cfg.n_layers)[:, spinup:]    # [n, T+1, d]
    h_hist, h_next = h[:, :-1], h[:, -1]
    x_state = tn.rsample(h_hist, torch.full_like(h_hist, cfg.obs_noise), torch.rand(h_hist.shape, generator=gen))
    if cfg.real_forcing:
        # forcing is observed too (normalised); it is exogenous input, not part of the bounded latent
        Pn = (P[:, spinup:-1] / 50.0).clamp(0, 1).unsqueeze(-1)
        PETn = (PET[:, spinup:-1] / 10.0).clamp(0, 1).unsqueeze(-1)
        x = torch.cat([x_state, Pn, PETn], dim=-1)
    else:
        x = x_state
    if cfg.target == "nowcast_map":
        from .data_synthetic import target_mean
        g_map = torch.Generator().manual_seed(cfg.map_seed)
        W1 = 3.0 * torch.randn(h.shape[-1], 16, generator=g_map)
        b1 = torch.randn(16, generator=g_map)
        w2 = torch.randn(16, generator=g_map) / 4.0
        mu_y = target_mean(h[:, -2], W1, b1, w2).unsqueeze(-1)
    else:
        mu_y = h_next[:, :1]
    y = tn.rsample(mu_y, torch.full((n, 1), cfg.y_noise), torch.rand(n, 1, generator=gen))
    # a single "regime" (season) indicator: quarter of the year is not tracked here, use one regime
    s = torch.zeros(n, cfg.T, dtype=torch.long)
    return {"x": x, "s": s, "y": y, "h_true": h[:, -2], "site": site_idx}


def load_daymet(folder):
    """Read Daymet single-pixel CSVs + sites.csv -> dict site -> (P, PET) daily tensors [mm/day]."""
    import csv
    from pathlib import Path

    folder = Path(folder)
    with open(folder / "sites.csv") as f:
        sites = {r["site"]: float(r["lat"]) for r in csv.DictReader(f)}
    forcing = {}
    for site, lat in sites.items():
        fn = folder / f"{site}.csv"
        if not fn.exists():
            continue
        lines = fn.read_text().splitlines()
        start = next(i for i, l in enumerate(lines) if l.startswith("year,yday"))
        rows = [list(map(float, l.split(","))) for l in lines[start + 1:] if l.strip()]
        # Daymet uses a 365-day year: 1990-2019 must give exactly 30 * 365 rows ending in 2019.
        # Anything else is a truncated download and is skipped (and reported by the caller).
        if len(rows) != 30 * 365 or rows[-1][0] != 2019 or any(len(r) != 5 for r in rows):
            print(f"daymet: skipping incomplete file {fn.name} ({len(rows)} rows)")
            continue
        v = torch.tensor(rows)
        doy, P, tmax, tmin = v[:, 1], v[:, 2], v[:, 3], v[:, 4]
        forcing[site] = (P, hargreaves_pet(tmin, tmax, lat, doy))
    return forcing


def generate(cfg: BucketConfig, forcing):
    """forcing: dict site_id -> (P, PET) daily tensors (mm/day), >= 2 years each."""
    gen = torch.Generator().manual_seed(cfg.data_seed)
    sites = sorted(forcing)
    perm = torch.randperm(len(sites), generator=gen).tolist()
    n_tr = int(round(cfg.train_site_frac * len(sites)))
    if cfg.site_split:
        train_sites = [sites[i] for i in perm[:n_tr]]
        pool_sites = [sites[i] for i in perm[n_tr:]]
    else:
        train_sites = pool_sites = sites
    pooled = (torch.cat([forcing[s][0] for s in sites]), float(torch.cat([forcing[s][1] for s in sites]).mean()))
    out = {"config": asdict(cfg), "train_sites": train_sites, "pool_sites": pool_sites}
    out["train"] = _make_units(cfg, forcing, train_sites, cfg.n_train, gen, pooled=pooled)
    out["cal"] = _make_units(cfg, forcing, pool_sites, cfg.n_cal, gen, pooled=pooled)
    out["test"] = _make_units(cfg, forcing, pool_sites, cfg.n_test, gen, pooled=pooled)
    return out
