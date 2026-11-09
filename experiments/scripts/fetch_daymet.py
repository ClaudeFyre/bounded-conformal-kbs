"""Fetch Daymet v4 daily weather (prcp, tmax, tmin; 1990-2019) at a fixed random sample of sites.

Sites: ResOpsUS dam coordinates (spread across CONUS climates). The sample is
fixed by --seed and written to raw_data/daymet/sites.csv before any download.

    python scripts/fetch_daymet.py --n_sites 120
"""
import argparse
import csv
import random
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ATTR = ROOT / "raw_data" / "resopsus" / "ResOpsUS" / "attributes" / "reservoir_attributes.csv"
OUT = ROOT / "raw_data" / "daymet"
URL = ("https://daymet.ornl.gov/single-pixel/api/data?lat={lat}&lon={lon}"
       "&vars=prcp,tmax,tmin&start=1990-01-01&end=2019-12-31&format=csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_sites", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with open(ATTR, encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r["LAT"] not in ("", "NA") and r["LONG"] not in ("", "NA")]
    rows.sort(key=lambda r: int(r["DAM_ID"]))
    sites = random.Random(a.seed).sample(rows, a.n_sites)
    with open(OUT / "sites.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["site", "lat", "lon", "state"])
        for r in sites:
            w.writerow([r["DAM_ID"], r["LAT"], r["LONG"], r["STATE"]])
    for i, r in enumerate(sites):
        fn = OUT / f"{r['DAM_ID']}.csv"
        if fn.exists() and fn.stat().st_size > 100_000:
            continue
        for attempt in range(3):
            try:
                urllib.request.urlretrieve(URL.format(lat=r["LAT"], lon=r["LONG"]), fn)
                break
            except Exception as e:  # network hiccup: retry politely
                print(f"retry {r['DAM_ID']}: {e}", flush=True)
                time.sleep(5)
        print(f"{i + 1}/{len(sites)} {r['DAM_ID']} {fn.stat().st_size if fn.exists() else 0} bytes", flush=True)
        time.sleep(1.0)


if __name__ == "__main__":
    main()
