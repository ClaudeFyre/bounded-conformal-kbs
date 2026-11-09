"""Run the full synthetic grid sequentially (skips configs whose JSON already exists).

Grid (per dataset, per seed):
  main:        5 latent families x TN-hetero decoder
  fixed-dec:   5 latent families x Gaussian-homo decoder (representation-only control)
  saturating:  gaussian, kumaraswamy x TN-hetero with projected decoder input (L = 0 outside M)
"""
import argparse
import itertools
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ["gaussian", "trunc_gaussian", "beta", "logit_normal", "kumaraswamy"]


def configs():
    for f in FAMILIES:
        yield f, "tn_hetero", "raw"
        yield f, "gauss_homo", "raw"
    for f in ("gaussian", "kumaraswamy"):
        yield f, "tn_hetero", "projected"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["syn_full", "syn_extreme"])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    a = ap.parse_args()
    for ds, seed, (fam, dec, inp) in itertools.product(a.datasets, range(a.seeds), configs()):
        out = ROOT / "results" / ds / f"{fam}__{dec}__{inp}__s{seed}.json"
        if out.exists():
            continue
        cmd = [sys.executable, str(ROOT / "scripts" / "run_synthetic.py"), "--dataset", ds, "--family", fam,
               "--decoder", dec, "--dec_input", inp, "--seed", str(seed), *a.extra]
        # Retry: occasional native crashes (exit 0xC0000409) on Windows/CUDA are not reproducible.
        for attempt in range(3):
            if subprocess.run(cmd).returncode == 0:
                break
            print(f"retry {attempt + 1} after failure: {' '.join(cmd[2:])}", flush=True)
        else:
            print(f"FAILED after 3 attempts: {' '.join(cmd[2:])}", flush=True)


if __name__ == "__main__":
    main()
