"""ERA5 total cloud cover, held-out confirmation data (protocol pre-registered in PLAN.md).

Unit = 48-hour window at one site; target = total cloud cover 3 h after the window (fraction,
exact 0 and 1 included). Sites are split per seed: 30 training sites (2010-2016) and 20 evaluation
sites; calibration and test units are both drawn i.i.d. from the evaluation sites in 2017-2019.
Input columns: cloud cover (first column: observed state for the d = 1 latent), relative humidity/100,
(surface pressure - 1000)/50, temperature (standardised on training sites), wind/20, min(precip,10)/10,
sin/cos of hour and month.
"""
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class ERA5Config:
    name: str = "era5_cloud"
    T: int = 48
    horizon: int = 3
    n_train: int = 10_000
    n_cal: int = 2_000
    n_test: int = 2_000
    n_train_sites: int = 30
    obs_noise: float = 0.02
    data_seed: int = 0


def _load_site(folder, i):
    h = json.loads((Path(folder) / f"site{i:02d}.json").read_text())["hourly"]
    f = lambda v: np.array([np.nan if x is None else x for x in h[v]], dtype=np.float32)
    t = np.array(h["time"], dtype="datetime64[h]")
    return {"cloud": f("cloud_cover") / 100, "rh": f("relative_humidity_2m") / 100,
            "p": (f("surface_pressure") - 1000) / 50, "temp": f("temperature_2m"),
            "wind": f("wind_speed_10m") / 20, "precip": np.minimum(f("precipitation"), 10) / 10,
            "hour": (t.astype(np.int64) % 24).astype(np.float32),
            "month": (t.astype("datetime64[M]").astype(np.int64) % 12).astype(np.float32),
            "year": t.astype("datetime64[Y]").astype(np.int64) + 1970}


def generate(cfg: ERA5Config, folder):
    rng = np.random.default_rng(cfg.data_seed)
    sites = rng.permutation(50)
    train_sites, eval_sites = sites[: cfg.n_train_sites], sites[cfg.n_train_sites:]
    S = {i: _load_site(folder, i) for i in range(50)}
    t_mu = np.nanmean(np.concatenate([S[i]["temp"] for i in train_sites]))
    t_sd = np.nanstd(np.concatenate([S[i]["temp"] for i in train_sites])) + 1e-6
    L = cfg.T + cfg.horizon

    def feats(s):
        return np.stack([s["cloud"], s["rh"], s["p"], (s["temp"] - t_mu) / t_sd, s["wind"], s["precip"],
                         np.sin(2 * np.pi * s["hour"] / 24), np.cos(2 * np.pi * s["hour"] / 24),
                         np.sin(2 * np.pi * s["month"] / 12), np.cos(2 * np.pi * s["month"] / 12)], -1)

    F = {i: feats(S[i]) for i in range(50)}

    def starts(i, y0, y1):
        yrs = S[i]["year"]
        ok = np.isfinite(F[i]).all(-1)
        bad = np.convolve(~ok, np.ones(L, dtype=int), mode="valid") > 0
        st = np.arange(len(bad))[~bad]
        return st[(yrs[st] >= y0) & (yrs[np.minimum(st + L - 1, len(yrs) - 1)] <= y1)]

    def make(n, site_pool, y0, y1):
        cand = [(i, starts(i, y0, y1)) for i in site_pool]
        w = np.array([len(st) for _, st in cand], dtype=float)
        pick = rng.choice(len(cand), n, p=w / w.sum())
        xs, ys = [], []
        for k in pick:
            i, st = cand[k]
            s0 = int(rng.choice(st))
            xs.append(F[i][s0:s0 + cfg.T])
            ys.append([S[i]["cloud"][s0 + L - 1]])
        return {"x": torch.tensor(np.array(xs), dtype=torch.float32),
                "s": torch.zeros(n, cfg.T, dtype=torch.long),
                "y": torch.tensor(np.array(ys), dtype=torch.float32).clamp(0, 1)}

    out = {"config": asdict(cfg), "train_sites": train_sites.tolist(), "eval_sites": eval_sites.tolist()}
    out["train"] = make(cfg.n_train, train_sites, 2010, 2016)
    out["cal"] = make(cfg.n_cal, eval_sites, 2017, 2019)
    out["test"] = make(cfg.n_test, eval_sites, 2017, 2019)
    return out
