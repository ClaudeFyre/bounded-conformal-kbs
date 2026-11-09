"""Gate O1.5 candidate 1: ERA5 total cloud cover via Open-Meteo (pre-registered in PLAN.md).

    python scripts/o15_cloud.py --fetch   # 50 random sites, 2010-2019 hourly (raw_data/era5_cloud/)
    python scripts/o15_cloud.py --check   # the six pre-registered data-property criteria (no neural models)
"""
import argparse
import json
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw_data" / "era5_cloud"
VARS = ["cloud_cover", "relative_humidity_2m", "surface_pressure", "temperature_2m", "wind_speed_10m", "precipitation"]
URL = ("https://archive-api.open-meteo.com/v1/archive?latitude={lat:.4f}&longitude={lon:.4f}"
       "&start_date=2010-01-01&end_date=2019-12-31&hourly=" + ",".join(VARS) + "&models=era5&timezone=GMT")


def sites():
    rng = np.random.default_rng(0)
    lat = np.degrees(np.arcsin(rng.uniform(np.sin(np.radians(-55)), np.sin(np.radians(70)), 50)))  # area-uniform
    lon = rng.uniform(-180, 180, 50)
    return list(zip(lat, lon))


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    for i, (lat, lon) in enumerate(sites()):
        fn = RAW / f"site{i:02d}.json"
        if fn.exists() and fn.stat().st_size > 1_000_000:
            continue
        for attempt in range(4):
            try:
                with urllib.request.urlopen(URL.format(lat=lat, lon=lon), timeout=300) as r:
                    fn.write_bytes(r.read())
                break
            except Exception as e:
                print(f"retry site {i}: {e}", flush=True)
                time.sleep(30 * (attempt + 1))
        print(f"{i + 1}/50 site ({lat:.2f}, {lon:.2f}) {fn.stat().st_size if fn.exists() else 0} bytes", flush=True)
        time.sleep(5)


def load_site(i):
    h = json.loads((RAW / f"site{i:02d}.json").read_text())["hourly"]
    arr = {v: np.array([np.nan if x is None else x for x in h[v]], dtype=float) for v in VARS}
    t = np.array(h["time"], dtype="datetime64[h]")
    return arr, t


def check():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.neighbors import NearestNeighbors

    Xs, Ys, S = [], [], []
    for i in range(50):
        if not (RAW / f"site{i:02d}.json").exists():
            continue
        a, t = load_site(i)
        hour = (t.astype(int) % 24).astype(float)
        month = (t.astype("datetime64[M]").astype(int) % 12).astype(float)
        X = np.stack([a["cloud_cover"] / 100, a["relative_humidity_2m"], a["surface_pressure"], a["temperature_2m"],
                      a["wind_speed_10m"], a["precipitation"], np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
                      np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12)], -1)[:-3]
        Y = a["cloud_cover"][3:] / 100
        ok = np.isfinite(X).all(-1) & np.isfinite(Y)
        Xs.append(X[ok]); Ys.append(Y[ok]); S.append(np.full(ok.sum(), i))
    X, Y, S = np.concatenate(Xs), np.concatenate(Ys), np.concatenate(S)
    n_sites = len(np.unique(S))
    rng = np.random.default_rng(0)
    test_sites = rng.choice(np.unique(S), int(round(0.3 * n_sites)), replace=False)
    te = np.isin(S, test_sites)
    mu, sd = X[~te].mean(0), X[~te].std(0) + 1e-9
    Z = (X - mu) / sd

    p0, p1 = (Y <= 0).mean(), (Y >= 1).mean()
    interior = (Y > 0) & (Y < 1)
    print(f"sites {n_sites}, units {len(Y)}; P(Y=0) = {p0:.3f}, P(Y=1) = {p1:.3f}, interior mass = {interior.mean():.3f}")
    c1 = p0 >= 0.05 and p1 >= 0.05
    # criterion 3: kNN conditional 90% width for interior points
    tr_int = np.where(~te & interior)[0]
    te_int = np.where(te & interior)[0]
    sub_tr = rng.choice(tr_int, min(200_000, len(tr_int)), replace=False)
    q_pts = rng.choice(te_int, min(2000, len(te_int)), replace=False)
    nn = NearestNeighbors(n_neighbors=50).fit(Z[sub_tr])
    _, idx = nn.kneighbors(Z[q_pts])
    nb = Y[sub_tr][idx]
    width = np.quantile(nb, 0.95, axis=1) - np.quantile(nb, 0.05, axis=1)
    med_w = float(np.median(width))
    c3 = interior.mean() >= 0.2 and med_w >= 0.1
    print(f"criterion 3: interior mass {interior.mean():.3f} (>= 0.2), median kNN 90% width {med_w:.3f} (>= 0.1) -> {c3}")
    # criterion 4: atom predictability from X (site-held-out AUC)
    sub = rng.choice(np.where(~te)[0], min(300_000, (~te).sum()), replace=False)
    aucs = {}
    for name, ev in (("Y=0", Y <= 0), ("Y=1", Y >= 1)):
        clf = LogisticRegression(max_iter=1000).fit(Z[sub], ev[sub])
        aucs[name] = roc_auc_score(ev[te], clf.predict_proba(Z[te])[:, 1])
    c4 = all(v >= 0.7 for v in aucs.values())
    print(f"criterion 4: site-held-out AUC {aucs} (>= 0.7) -> {c4}")
    print(f"criterion 1: P(Y=0) {p0:.3f}, P(Y=1) {p1:.3f} (both >= 0.05) -> {c1}")
    print(f"criterion 5: {len(Y)} units over {n_sites} sites -> {n_sites >= 30}")
    print("criterion 2: semantic clear / overcast states (reanalysis caveat recorded in PLAN) -> True")
    print("criterion 6: Open-Meteo non-commercial, attribution to Open-Meteo and Copernicus ERA5 -> True")
    ok = c1 and c3 and c4 and n_sites >= 30
    print(f"\n**Candidate 1 (ERA5 cloud cover): {'MEETS' if ok else 'FAILS'} the O1.5 criteria**")
    grid = np.round(np.unique(Y[interior])[:5], 4)
    print(f"(interior value resolution check: first distinct interior values {grid})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.fetch:
        fetch()
    if a.check:
        check()


if __name__ == "__main__":
    main()
