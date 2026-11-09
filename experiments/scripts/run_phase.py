"""Stage 2 phase-diagram grid (design fixed in PLAN.md before running).

36 cells (p_b x C) x 5 latent families x 5 seeds, TN-hetero decoder.
Skips runs whose JSON already exists; retries occasional native crashes.
"""
import argparse
import itertools
import subprocess
from concurrent.futures import ThreadPoolExecutor
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PB = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8]
C = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
FAMILIES = ["gaussian", "trunc_gaussian", "logit_normal", "beta", "kumaraswamy"]
SEEDS = range(5)


def run_one(cmd):
    for attempt in range(3):
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0:
            print(r.stdout.strip().splitlines()[-1], flush=True)
            return
        print(f"retry {attempt + 1}: {' '.join(cmd[2:])}", flush=True)
    print(f"FAILED after 3 attempts: {' '.join(cmd[2:])}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    cmds = []
    for pb, c, seed, fam in itertools.product(PB, C, SEEDS, FAMILIES):
        out = ROOT / "results" / "phase" / f"pb{pb:g}_c{c:g}" / f"{fam}__tn_hetero__raw__s{seed}.json"
        if not out.exists():
            cmds.append([sys.executable, str(ROOT / "scripts" / "run_synthetic.py"), "--dataset", "phase",
                         "--pb", str(pb), "--post_std", str(c), "--family", fam, "--seed", str(seed)])
    print(f"{len(cmds)} runs to do with {a.workers} workers", flush=True)
    # startup (torch import + CUDA init) dominates each run, so a few parallel workers help a lot
    with ThreadPoolExecutor(a.workers) as pool:
        list(pool.map(run_one, cmds))


if __name__ == "__main__":
    main()
