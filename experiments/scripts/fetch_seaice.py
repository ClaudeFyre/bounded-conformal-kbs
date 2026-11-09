"""Fetch daily NOAA/NSIDC sea-ice concentration CDR (G02202 v6) for Hudson Bay, 1990-2019, via NOAA ERDDAP.

Region is fixed a priori on physical grounds (a fully seasonal ice zone that reaches
both 0% and ~100% cover every year): lat 56-63N, lon 94-79W, every 2nd 25-km cell.
Output: raw_data/seaice/hudson_<years>.nc (netCDF-3, readable by scipy).
"""
import math
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "raw_data" / "seaice"
BASE = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/nsidcG02202v6nh1day.nc"
LAT, LON = (56.0, 63.0), (-94.0, -79.0)
STRIDE = 2
CHUNKS = [(y, y + 4) for y in range(1990, 2020, 5)]


def polar_stereo_xy(lat, lon, lat_ts=70.0, lon0=-45.0, R=6371000.0):
    """Spherical approximation of the NSIDC north polar stereographic grid (enough to pick a box)."""
    k0 = (1 + math.sin(math.radians(lat_ts))) / 2
    rho = 2 * R * k0 * math.tan(math.pi / 4 - math.radians(lat) / 2)
    lam = math.radians(lon - lon0)
    return rho * math.sin(lam), -rho * math.cos(lam)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    corners = [polar_stereo_xy(a, o) for a in LAT for o in LON]
    x0, x1 = min(c[0] for c in corners), max(c[0] for c in corners)
    y0, y1 = min(c[1] for c in corners), max(c[1] for c in corners)
    print(f"box x [{x0:.0f}, {x1:.0f}] y [{y0:.0f}, {y1:.0f}]", flush=True)
    for a, b in CHUNKS:
        fn = OUT / f"hudson_{a}_{b}.nc"
        if fn.exists() and fn.stat().st_size > 1_000_000:
            continue
        sel = f"[({a}-01-01T00:00:00Z):1:({b}-12-31T00:00:00Z)][({y1:.0f}):{STRIDE}:({y0:.0f})][({x0:.0f}):{STRIDE}:({x1:.0f})]"
        url = f"{BASE}?cdr_seaice_conc{sel},cdr_seaice_conc_stdev{sel}"
        for attempt in range(3):
            try:
                urllib.request.urlretrieve(url, fn)
                break
            except Exception as e:
                print(f"retry {a}-{b}: {e}", flush=True)
                time.sleep(10)
        print(f"{a}-{b}: {fn.stat().st_size if fn.exists() else 0} bytes", flush=True)


if __name__ == "__main__":
    main()
