"""Real data: daily sea-ice concentration (NOAA/NSIDC CDR v6), Hudson Bay, 1990-2019.

Concentration is a fraction of area, so [0,1] is a definitional bound, and the
seasonal ice zone reaches both ends every year. There is no ground-truth latent:
the 'oracle' grouping is unavailable and only model-independent observation
grouping is used.

Unit = one grid cell over a window of T days; target = concentration `horizon`
days after the window. Regime = meteorological season (K = 4).
Split: cells are grouped into spatial blocks; blocks are divided into train and
pool blocks. Training units come from train blocks in 1990-2009; calibration and
test units are both drawn i.i.d. from pool blocks in 2010-2019, so they are
exchangeable by construction (with respect to that pool).
"""
import datetime as dt
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.io import netcdf_file


@dataclass
class SeaIceConfig:
    name: str = "seaice"
    T: int = 120
    horizon: int = 7
    n_train: int = 10_000
    n_cal: int = 2_000
    n_test: int = 2_000
    block: int = 4              # spatial block size in (strided) grid cells
    train_block_frac: float = 0.6
    train_years: tuple = (1990, 2009)
    eval_years: tuple = (2010, 2019)
    data_seed: int = 0


def load_raw(folder):
    """Concatenate the yearly chunks -> (dates [N], conc [N, ny, nx] float32 with NaN off-ocean, stdev)."""
    files = sorted(Path(folder).glob("hudson_*.nc"))
    dates, conc, sd = [], [], []
    for fn in files:
        with netcdf_file(fn, "r", mmap=False) as f:
            t = f.variables["time"][:].copy()
            c = f.variables["cdr_seaice_conc"][:].astype(np.float32)
            s = f.variables["cdr_seaice_conc_stdev"][:].astype(np.float32)
            fill_c = getattr(f.variables["cdr_seaice_conc"], "_FillValue", None)
        c[(c < 0) | (c > 1) | ~np.isfinite(c)] = np.nan
        if fill_c is not None:
            c[c == fill_c] = np.nan
        s[~np.isfinite(s) | (s < 0) | (s > 1)] = np.nan
        dates.extend(dt.datetime(1970, 1, 1) + dt.timedelta(seconds=float(v)) for v in t)
        conc.append(c)
        sd.append(s)
    return dates, np.concatenate(conc), np.concatenate(sd)


def _season(month):
    return {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}[month]


def generate(cfg: SeaIceConfig, folder):
    dates, conc, sd = load_raw(folder)
    N, ny, nx = conc.shape
    # ocean cells with (nearly) complete records
    valid = np.isfinite(conc).mean(0) > 0.98
    cells = np.argwhere(valid)
    gen = np.random.default_rng(cfg.data_seed)
    blocks = sorted({(i // cfg.block, j // cfg.block) for i, j in cells})
    gen.shuffle(blocks)
    n_tr = int(round(cfg.train_block_frac * len(blocks)))
    train_blocks = set(blocks[:n_tr])
    in_train = np.array([(i // cfg.block, j // cfg.block) in train_blocks for i, j in cells])
    years = np.array([d.year for d in dates])
    season = torch.tensor([_season(d.month) for d in dates])
    # fill rare gaps by carrying the last valid value forward (per cell)
    series = np.nan_to_num(conc[:, cells[:, 0], cells[:, 1]], nan=np.nan)
    for k in range(series.shape[1]):
        col = series[:, k]
        mask = np.isnan(col)
        if mask.any():
            idx = np.where(~mask, np.arange(N), 0)
            np.maximum.accumulate(idx, out=idx)
            series[:, k] = col[idx]
    series = torch.from_numpy(np.clip(series, 0, 1))
    obs_noise = float(np.nanmedian(sd[:, cells[:, 0], cells[:, 1]]))

    def make(n, cell_mask, yr):
        cand_cells = np.where(cell_mask)[0]
        L = cfg.T + cfg.horizon
        starts_ok = np.where((years >= yr[0]) & (years <= yr[1]))[0]
        starts_ok = starts_ok[starts_ok + L <= N]
        starts_ok = starts_ok[years[starts_ok + L - 1] <= yr[1]]
        ci = gen.choice(cand_cells, n)
        st = gen.choice(starts_ok, n)
        idx = torch.from_numpy(st)[:, None] + torch.arange(cfg.T)[None]
        x = series[idx, torch.from_numpy(ci)[:, None]].unsqueeze(-1).float()
        y = series[torch.from_numpy(st + L - 1), torch.from_numpy(ci)].unsqueeze(-1).float()
        s = season[idx]
        return {"x": x, "s": s, "y": y, "cell": torch.from_numpy(ci)}

    out = {"config": asdict(cfg), "obs_noise": obs_noise, "n_cells": int(len(cells)),
           "n_train_cells": int(in_train.sum())}
    out["train"] = make(cfg.n_train, in_train, cfg.train_years)
    out["cal"] = make(cfg.n_cal, ~in_train, cfg.eval_years)
    out["test"] = make(cfg.n_test, ~in_train, cfg.eval_years)
    return out
