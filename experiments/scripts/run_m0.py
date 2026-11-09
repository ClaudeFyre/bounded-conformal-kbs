"""Gate M0 grid: 5 datasets x 6 models x 8 seeds (pre-registered in PLAN.md).

Data caches are generated sequentially first, so parallel workers never race on a cache file.
"""
import argparse
import itertools
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from run_phase import run_one

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ["syn_full", "bucket", "seaice", "resops", "syn_atom"]
MODELS = [1, 2, 3, 4, 5, 6]
SEEDS = range(8)


def warm_caches():
    import m0_one
    for ds, seed in itertools.product(DATASETS, SEEDS):
        if ds != "syn_atom":
            m0_one.load(ds, seed)
            print(f"cache ready: {ds} seed {seed}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    warm_caches()
    cmds = []
    for ds, seed, m in itertools.product(DATASETS, SEEDS, MODELS):
        if not (ROOT / "results" / "m0" / ds / f"model{m}__s{seed}.json").exists():
            cmds.append([sys.executable, str(ROOT / "scripts" / "m0_one.py"), "--dataset", ds,
                         "--model", str(m), "--seed", str(seed)])
    print(f"{len(cmds)} runs to do with {a.workers} workers", flush=True)
    with ThreadPoolExecutor(a.workers) as pool:
        list(pool.map(run_one, cmds))


if __name__ == "__main__":
    main()
