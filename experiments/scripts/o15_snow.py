"""Gate O1.5 candidate 2: ESA Snow_cci CryoClim FSC v1.0 (pre-registered in PLAN.md).

    python scripts/o15_snow.py --fetch   # OPeNDAP ASCII subsets, 30-70N every 1 deg, 2005-2009 daily
    python scripts/o15_snow.py --check   # the six pre-registered data-property criteria (no neural models)
"""
import argparse
import datetime as dt
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
START, END = dt.date(2005, 1, 1), dt.date(2009, 12, 31)
DAP = "https://dap.ceda.ac.uk/thredds/dodsC/neodc/esacci/snow/data"
# product -> (raw folder, variable, url builder, lat indices, lon indices, lat0, dlat, lon0, dlon)
PRODUCTS = {
    "cryoclim": ("snow_cci", "scfg", lambda d: f"{DAP}/scfg/CryoClim/v1.0/{d:%Y/%m}/{d:%Y%m%d}-ESACCI-L3S_SNOW-SCFG-MERGED-CRYOCLIM-fv1.0.nc",
                 range(400, 1181, 20), range(0, 7181, 20), 89.975, -0.05, -179.975, 0.05),
    "modis_scfg": ("snow_modis_scfg", "scfg", lambda d: f"{DAP}/scfg/MODIS/v4.0/{d:%Y/%m}/{d:%Y%m%d}-ESACCI-L3C_SNOW-SCFG-MODIS_TERRA-fv4.0.nc",
                   range(12000, 15901, 100), range(0, 35901, 100), -89.995, 0.01, -179.995, 0.01),
    "modis_scfv": ("snow_modis_scfv", "scfv", lambda d: f"{DAP}/scfv/MODIS/v4.0/{d:%Y/%m}/{d:%Y%m%d}-ESACCI-L3C_SNOW-SCFV-MODIS_TERRA-fv4.0.nc",
                   range(12000, 15901, 100), range(0, 35901, 100), -89.995, 0.01, -179.995, 0.01),
}
PRODUCT = "cryoclim"


def cfg():
    return PRODUCTS[PRODUCT]


def raw():
    return ROOT / "raw_data" / cfg()[0]


def url(day):
    folder, var, build, li, lo, *_ = cfg()
    sl = lambda r: f"%5B{r.start}:{r.step}:{r.stop - 1}%5D"
    return f"{build(day)}.ascii?{var}%5B0:0%5D{sl(li)}{sl(lo)}"


def parse(text):
    """ASCII grid rows '[0][i], v, v, ...' after the 'scfg.scfg[1][40][360]' header -> int16 codes [40, 360]."""
    rows = []
    var = cfg()[1]
    body = text.split(f"{var}.{var}[1]", 1)[1].split(f"{var}.time", 1)[0]
    for line in body.splitlines():
        if line.startswith("[0]["):
            vals = [int(v) for v in line.split(",")[1:]]
            rows.append([v + 256 if v < 0 else v for v in vals])       # signed byte -> unsigned code
    arr = np.array(rows, dtype=np.int16)
    assert arr.shape == (len(cfg()[3]), len(cfg()[4])), arr.shape
    return arr


def fetch():
    RAW = raw()
    RAW.mkdir(parents=True, exist_ok=True)
    day = START
    while day <= END:
        fn = RAW / f"{day:%Y%m%d}.npy"
        if not fn.exists():
            for attempt in range(4):
                try:
                    with urllib.request.urlopen(url(day), timeout=120) as r:
                        np.save(fn, parse(r.read().decode()))
                    break
                except urllib.error.HTTPError as e:
                    if e.code == 404:          # day missing from the archive: record as all-missing
                        np.save(fn, np.full((len(cfg()[3]), len(cfg()[4])), 255, dtype=np.int16))
                        break
                    time.sleep(10 * (attempt + 1))
                except Exception:
                    time.sleep(10 * (attempt + 1))
            time.sleep(0.5)
        if day.day == 1:
            print(f"{day}", flush=True)
        day += dt.timedelta(days=1)


def check():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.neighbors import NearestNeighbors

    days = []
    d = START
    while d <= END:
        days.append(d)
        d += dt.timedelta(days=1)
    RAW = raw()
    _, _, _, LAT_IDX, LON_IDX, lat0, dlat, lon0, dlon = cfg()
    cube = np.stack([np.load(RAW / f"{x:%Y%m%d}.npy") for x in days])          # [T, 40, 360]
    T = cube.shape[0]
    lat = lat0 + dlat * np.array(LAT_IDX)
    lon = lon0 + dlon * np.array(LON_IDX)
    # extra report (not a criterion): valid-observation rate by month x latitude band
    months = np.array([x.month for x in days])
    print("valid-observation rate by month (rows) x latitude band 30-40/40-50/50-60/60-70N:")
    for m in range(1, 13):
        vm = cube[months == m] <= 100
        rates = [vm[:, (lat >= b) & (lat < b + 10)].mean() for b in (30, 40, 50, 60)]
        print(f"  month {m:2d}: " + " ".join(f"{r:.2f}" for r in rates))
    valid = cube <= 100
    doy = np.array([x.timetuple().tm_yday for x in days], dtype=float)
    t = np.arange(7, T - 1)
    ok = valid[t + 1] & valid[t] & valid[t - 1] & valid[t - 7]                    # [len(t), 40, 360]
    ti, yi, xi = np.nonzero(ok)
    tt = t[ti]
    Y = cube[tt + 1, yi, xi] / 100.0
    X = np.stack([cube[tt, yi, xi] / 100, cube[tt - 1, yi, xi] / 100, cube[tt - 7, yi, xi] / 100, lat[yi],
                  np.sin(np.radians(lon[xi])), np.cos(np.radians(lon[xi])),
                  np.sin(2 * np.pi * doy[tt] / 365.25), np.cos(2 * np.pi * doy[tt] / 365.25)], -1)
    site = yi * len(LON_IDX) + xi
    sites = np.unique(site)
    print(f"days {T}, valid-unit fraction among land/sea grid-days {ok.mean():.3f}; units {len(Y)} over {len(sites)} grid points")
    rng = np.random.default_rng(0)
    te = np.isin(site, rng.choice(sites, int(round(0.3 * len(sites))), replace=False))
    mu, sd = X[~te].mean(0), X[~te].std(0) + 1e-9
    Z = (X - mu) / sd
    p0, p1 = (Y <= 0).mean(), (Y >= 1).mean()
    interior = (Y > 0) & (Y < 1)
    c1 = p0 >= 0.05 and p1 >= 0.05
    print(f"criterion 1: P(Y=0) {p0:.4f}, P(Y=1) {p1:.4f} (both >= 0.05) -> {c1}")
    tr_int, te_int = np.where(~te & interior)[0], np.where(te & interior)[0]
    sub_tr = rng.choice(tr_int, min(200_000, len(tr_int)), replace=False)
    q_pts = rng.choice(te_int, min(2000, len(te_int)), replace=False)
    _, idx = NearestNeighbors(n_neighbors=50).fit(Z[sub_tr]).kneighbors(Z[q_pts])
    nb = Y[sub_tr][idx]
    med_w = float(np.median(np.quantile(nb, 0.95, axis=1) - np.quantile(nb, 0.05, axis=1)))
    c3 = interior.mean() >= 0.2 and med_w >= 0.1
    print(f"criterion 3: interior mass {interior.mean():.3f} (>= 0.2), median kNN 90% width {med_w:.3f} (>= 0.1) -> {c3}")
    sub = rng.choice(np.where(~te)[0], min(300_000, (~te).sum()), replace=False)
    aucs = {}
    for name, ev in (("Y=0", Y <= 0), ("Y=1", Y >= 1)):
        clf = LogisticRegression(max_iter=1000).fit(Z[sub], ev[sub])
        aucs[name] = roc_auc_score(ev[te], clf.predict_proba(Z[te])[:, 1])
    c4 = all(v >= 0.7 for v in aucs.values())
    print(f"criterion 4: grid-point-held-out AUC {aucs} (>= 0.7) -> {c4}")
    c5 = len(sites) >= 30 and len(Y) >= 100_000
    print(f"criterion 5: {len(Y)} units over {len(sites)} grid points -> {c5}")
    print("criterion 2: semantic snow-free / fully snow-covered states (selection caveat recorded in PLAN) -> True")
    print("criterion 6: ESA CCI Snow terms, citation required, anonymous access -> True")
    ok_all = c1 and c3 and c4 and c5
    print(f"\n**Snow_cci product {PRODUCT}: {'MEETS' if ok_all else 'FAILS'} the O1.5 criteria**")
    print(f"(interior value resolution: first distinct interior values {np.unique(Y[interior])[:5]})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--product", default="cryoclim", choices=list(PRODUCTS))
    a = ap.parse_args()
    global PRODUCT
    PRODUCT = a.product
    if a.fetch:
        fetch()
    if a.check:
        check()


if __name__ == "__main__":
    main()
