"""Bucket ablation ladder (pre-registered in PLAN.md): 6 levels x 3 families x 5 seeds, TN-hetero."""
import argparse
import itertools
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from run_phase import run_one

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ["gaussian", "logit_normal", "kumaraswamy"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    cmds = []
    for level, seed, fam in itertools.product(range(6), range(5), FAMILIES):
        out = ROOT / "results" / f"ladder_B{level}" / f"{fam}__tn_hetero__raw__s{seed}.json"
        if not out.exists():
            cmds.append([sys.executable, str(ROOT / "scripts" / "run_synthetic.py"), "--dataset", "ladder",
                         "--level", str(level), "--family", fam, "--seed", str(seed)])
    print(f"{len(cmds)} runs to do with {a.workers} workers", flush=True)
    with ThreadPoolExecutor(a.workers) as pool:
        list(pool.map(run_one, cmds))


if __name__ == "__main__":
    main()
