"""Review response A (pre-registered in PLAN.md, 2026-10-23): TCN backbone robustness runs.

Bucket, ResOpsUS and sea ice; Gaussian and Kumaraswamy latents; TN-hetero decoder; seeds 0-9; data_seed 0, as in
the original five-dataset comparison. Everything except the backbone is unchanged. Results go to
results/<dataset>_tcn/ and checkpoints/<dataset>_tcn/ (never overwriting the GRU results). A failed run is retried
once with the same settings, and the retry is logged.

    python scripts/run_tcn.py            # trains what is missing, 3 processes in parallel
"""
import itertools
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ["bucket", "resops", "seaice"]
FAMILIES = ["gaussian", "kumaraswamy"]
SEEDS = range(10)
LOG = ROOT / "results" / "tcn_train.txt"


def out_json(ds, fam, seed):
    return ROOT / "results" / f"{ds}_tcn" / f"{fam}__tn_hetero__raw__s{seed}.json"


def run(job):
    ds, fam, seed = job
    cmd = [sys.executable, str(ROOT / "scripts" / "run_synthetic.py"), "--dataset", ds, "--family", fam,
           "--backbone", "tcn", "--seed", str(seed), "--data_seed", "0"]
    for attempt in (1, 2):
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        ok = r.returncode == 0 and out_json(ds, fam, seed).exists()
        line = (r.stdout.strip().splitlines() or [""])[-1]
        with open(LOG, "a", encoding="utf8") as f:
            f.write(f"{ds} {fam} seed {seed} attempt {attempt} {'ok' if ok else 'FAILED'}: {line}\n")
            if not ok:
                f.write(r.stderr[-2000:] + "\n")
        if ok:
            return True
    return False


def main():
    jobs = [j for j in itertools.product(DATASETS, FAMILIES, SEEDS) if not out_json(*j).exists()]
    print(f"{len(jobs)} runs to do", flush=True)
    with ThreadPoolExecutor(3) as pool:
        done = list(pool.map(run, jobs))
    print(f"finished: {sum(done)}/{len(jobs)} ok", flush=True)


if __name__ == "__main__":
    main()
