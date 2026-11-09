"""Real data: daily reservoir storage (ResOpsUS) bounded by engineering capacity (NID).

Bounds (pre-registered, PLAN.md revision 5): lower = 0, upper = NID "Max Storage"
(storage at top of dam). ResOpsUS dams are matched to NID dams by the nearest NID
dam within 3 km whose name shares the first word. Dams whose observed storage
exceeds NID Max Storage by more than 2% are excluded as metadata-inconsistent.

Unit = one dam over a window of T days; target = relative storage `horizon` days
after the window. Inputs: relative storage, relative inflow (NaN -> 0) and an
inflow-missing flag. Regime = season (K = 4). Dams are split into train / pool;
training uses 1980-2009, and calibration and test units are both drawn i.i.d.
from pool dams in 2010-2020.
"""
import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

ACRE_FT_TO_MCM = 0.0012335


@dataclass
class ResOpsConfig:
    name: str = "resops"
    T: int = 180
    horizon: int = 7
    n_train: int = 10_000
    n_cal: int = 2_000
    n_test: int = 2_000
    train_dam_frac: float = 0.6
    train_years: tuple = (1980, 2009)
    eval_years: tuple = (2010, 2020)
    match_km: float = 3.0
    exceed_tol: float = 0.02
    obs_noise: float = 0.01      # storage records: 1% of capacity (fixed a priori)
    max_gap: int = 7             # forward-fill storage gaps up to this many days
    data_seed: int = 0


def _haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def _first_word(name):
    w = "".join(ch if ch.isalnum() or ch == " " else " " for ch in (name or "").lower()).split()
    return w[0] if w else ""


def load_nid(path):
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        next(f)  # "Data Last Updated" line
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        try:
            lat, lon = float(r["Latitude"]), float(r["Longitude"])
            cap = float(r["Max Storage (Acre-Ft)"])
        except (ValueError, KeyError, TypeError):
            continue
        if cap > 0:
            out.append((lat, lon, cap * ACRE_FT_TO_MCM, r["Dam Name"], r["NID ID"]))
    return out


def match_capacities(resops_dir, nid_path, cfg):
    nid = load_nid(nid_path)
    nlat = np.array([r[0] for r in nid])
    nlon = np.array([r[1] for r in nid])
    with open(Path(resops_dir) / "attributes" / "reservoir_attributes.csv", encoding="utf-8-sig") as f:
        dams = list(csv.DictReader(f))
    matched, report = {}, {"n_resops": len(dams), "no_match": 0}
    for d in dams:
        try:
            lat, lon = float(d["LAT"]), float(d["LONG"])
        except ValueError:
            report["no_match"] += 1
            continue
        dist = _haversine_km(lat, lon, nlat, nlon)
        ok = np.where(dist <= cfg.match_km)[0]
        ok = [i for i in ok[np.argsort(dist[ok])] if _first_word(nid[i][3]) == _first_word(d["DAM_NAME"])]
        if not ok:
            report["no_match"] += 1
            continue
        i = ok[0]
        matched[d["DAM_ID"]] = {"cap_mcm": nid[i][2], "nid_id": nid[i][4], "dist_km": float(dist[i])}
    report["matched"] = len(matched)
    return matched, report


def _read_series(fn):
    dates, stor, inflow = [], [], []
    with open(fn, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if not r["date"] or not r["date"][0].isdigit():   # some files repeat the header mid-file
                continue
            dates.append(np.datetime64(r["date"]))
            stor.append(float(r["storage"]) if r["storage"] not in ("NA", "") else np.nan)
            inflow.append(float(r["inflow"]) if r["inflow"] not in ("NA", "") else np.nan)
    return np.array(dates), np.array(stor), np.array(inflow)


def _ffill(v, max_gap):
    out = v.copy()
    last, gap = np.nan, 0
    for i in range(len(out)):
        if np.isnan(out[i]):
            gap += 1
            if gap <= max_gap:
                out[i] = last
        else:
            last, gap = out[i], 0
    return out


def _season(month):
    return (month % 12) // 3


def generate(cfg: ResOpsConfig, resops_dir, nid_path):
    matched, report = match_capacities(resops_dir, nid_path, cfg)
    series = {}
    report["excluded_exceed"] = 0
    report["excluded_no_storage"] = 0
    for dam_id, m in matched.items():
        fn = Path(resops_dir) / "time_series_all" / f"ResOpsUS_{dam_id}.csv"
        if not fn.exists():
            report["excluded_no_storage"] += 1
            continue
        dates, stor, inflow = _read_series(fn)
        if np.isfinite(stor).sum() < 3 * 365:
            report["excluded_no_storage"] += 1
            continue
        cap = m["cap_mcm"]
        if np.nanmax(stor) > (1 + cfg.exceed_tol) * cap:
            report["excluded_exceed"] += 1
            continue
        rel = np.clip(_ffill(stor, cfg.max_gap) / cap, 0, 1)
        inflow_rel = inflow * 86400 / 1e6 / cap               # fraction of capacity per day
        series[dam_id] = {
            "dates": dates, "rel": rel,
            "inflow": np.nan_to_num(np.clip(inflow_rel / 0.05, 0, 1), nan=0.0),   # 5%/day -> 1
            "inflow_missing": np.isnan(inflow_rel).astype(np.float32),
        }
    report["used"] = len(series)
    gen = np.random.default_rng(cfg.data_seed)
    dam_ids = sorted(series)
    gen.shuffle(dam_ids)
    n_tr = int(round(cfg.train_dam_frac * len(dam_ids)))
    groups = {"train": dam_ids[:n_tr], "pool": dam_ids[n_tr:]}
    L = cfg.T + cfg.horizon

    def valid_starts(dam, yr):
        s = series[dam]
        years = s["dates"].astype("datetime64[Y]").astype(int) + 1970
        ok = np.isfinite(s["rel"])
        # window [st, st+L) fully observed and inside the year range
        bad = np.convolve(~ok, np.ones(L, dtype=int), mode="valid") > 0
        st = np.arange(len(bad))[~bad]
        return st[(years[st] >= yr[0]) & (years[np.minimum(st + L - 1, len(years) - 1)] <= yr[1])]

    def make(n, dams, yr):
        cand = [(d, valid_starts(d, yr)) for d in dams]
        cand = [(d, st) for d, st in cand if len(st)]
        weights = np.array([len(st) for _, st in cand], dtype=float)
        pick = gen.choice(len(cand), n, p=weights / weights.sum())   # uniform over (dam, window) pairs
        xs, ys, ss, cells = [], [], [], []
        for k in pick:
            d, st = cand[k]
            s0 = int(gen.choice(st))
            sr = series[d]
            sl = slice(s0, s0 + cfg.T)
            xs.append(np.stack([sr["rel"][sl], sr["inflow"][sl], sr["inflow_missing"][sl]], -1))
            ys.append([sr["rel"][s0 + L - 1]])
            months = sr["dates"][sl].astype("datetime64[M]").astype(int) % 12 + 1
            ss.append([_season(int(mo)) for mo in months])
            cells.append(int(d))
        return {"x": torch.tensor(np.array(xs), dtype=torch.float32),
                "s": torch.tensor(ss, dtype=torch.long),
                "y": torch.tensor(ys, dtype=torch.float32),
                "dam": torch.tensor(cells)}

    out = {"config": asdict(cfg), "report": report, "obs_noise": cfg.obs_noise}
    out["train"] = make(cfg.n_train, groups["train"], cfg.train_years)
    out["cal"] = make(cfg.n_cal, groups["pool"], cfg.eval_years)
    out["test"] = make(cfg.n_test, groups["pool"], cfg.eval_years)
    return out
