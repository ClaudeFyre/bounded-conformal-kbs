"""Gate B: counterfactual encoder/decoder swap experiment (pre-registered in PLAN.md).

For each discovery dataset, seed and bounded family B, evaluate five combinations
with conformal recalibration (observation grouping, eps = 0.05, raw width):
  (a) enc_G + dec_G     (b) enc_B + dec_B     (c) enc_B + dec_G
  (d) enc_G + dec_B     (e) q* (q^G restricted to M) + dec_G
and the 2x2 decomposition of the observed gap (a) - (b) into posterior and decoder main effects.

    python scripts/counterfactual.py            # runs everything, writes results/counterfactual/*.json
    python scripts/counterfactual.py --summary  # prints the Gate B table
"""
import argparse
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scrl import tn  # noqa: E402
from scrl.conformal import mondrian_eval  # noqa: E402
from scrl.models import Model  # noqa: E402

DATASETS = ["syn_full", "syn_extreme", "seaice", "bucket", "resops"]
FAMILIES = ["trunc_gaussian", "logit_normal", "beta", "kumaraswamy"]
SEEDS = range(10)
EPS, ALPHA, S = 0.05, 0.1, 128   # 128 samples: MC noise at S=32 was ~1.4% of width
OUT = ROOT / "results" / "counterfactual"


def load_model(name, family, d, n_obs, K, obs_noise, seed):
    m = Model(family, d=d, n_obs=n_obs, K=K, decoder="tn_hetero", obs_noise=obs_noise)
    ck = ROOT / "checkpoints" / name / f"{family}__tn_hetero__raw__s{seed}.pt"
    m.load_state_dict(torch.load(ck, map_location="cpu", weights_only=True))
    return m.eval()


@torch.no_grad()
def evaluate(split, samples, decoder_model):
    pred = decoder_model.predict_from_samples(samples)
    return {"y": split["y"], "y_hat": pred["y_hat"], "sigma": pred["sigma"],
            "x_last": split["x"][:, -1, : decoder_model.d]}


@torch.no_grad()
def run_dataset(ds):
    import run_synthetic
    a = argparse.Namespace(dataset=ds, data_seed=0, smoke=False, seed=0, pb=0.4, post_std=0.05)
    data, name, d, K, obs_noise = run_synthetic.load_data(a)
    cal, test = data["cal"], data["test"]
    n_obs = cal["x"].shape[-1]
    rows = []
    for seed in SEEDS:
        G = load_model(name, "gaussian", d, n_obs, K, obs_noise, seed)
        # encode once per split; samples drawn with a fixed generator state per combination
        pG = {sp: G.encode(D["x"], D["s"]) for sp, D in (("cal", cal), ("test", test))}
        for fam in FAMILIES:
            B = load_model(name, fam, d, n_obs, K, obs_noise, seed)
            pB = {sp: B.encode(D["x"], D["s"]) for sp, D in (("cal", cal), ("test", test))}
            row = {"dataset": ds, "seed": seed, "family": fam}
            for combo, enc, dec in (("a", "G", G), ("b", "B", B), ("c", "B", G), ("d", "G", B), ("e", "star", G)):
                ev = {}
                for sp, D in (("cal", cal), ("test", test)):
                    torch.manual_seed(1000 * seed + {"cal": 0, "test": 1}[sp])
                    if enc == "G":
                        hs = G.family.rsample(pG[sp], S)
                    elif enc == "B":
                        hs = B.family.rsample(pB[sp], S)
                    else:   # q*: the Gaussian posterior restricted to [0,1]^d
                        mu, sg = pG[sp]["mu"], pG[sp]["sigma"]
                        hs = tn.rsample(mu.expand(S, *mu.shape), sg.expand(S, *sg.shape), torch.rand(S, *mu.shape))
                    ev[sp] = evaluate(D, hs, dec)
                r = mondrian_eval(ev["cal"], ev["test"], EPS, ALPHA, "x_last")
                row[f"W_{combo}"] = r["biw"]
                row[f"cov_{combo}"] = r["coverage"]
                row[f"covb_{combo}"] = r["coverage_boundary"]
            rows.append(row)
            print(f"{ds} s{seed} {fam}: " + " ".join(f"{c}={row['W_' + c]:.4f}" for c in "abcde"), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{ds}.json").write_text(json.dumps(rows, indent=1))


def summary():
    print("| family | dataset | observed gap a-b | posterior main | decoder main | interaction | posterior share | theory a-e same sign |")
    print("|---|---|---|---|---|---|---|---|")
    passes = {}
    for fam in FAMILIES:
        n_ok = 0
        for ds in DATASETS:
            fn = OUT / f"{ds}.json"
            if not fn.exists():
                continue
            rows = [r for r in json.loads(fn.read_text()) if r["family"] == fam]
            m = {k: sum(r[k] for r in rows) / len(rows) for k in ("W_a", "W_b", "W_c", "W_d", "W_e")}
            gap = m["W_a"] - m["W_b"]
            post = 0.5 * ((m["W_a"] - m["W_c"]) + (m["W_d"] - m["W_b"]))
            dec = 0.5 * ((m["W_a"] - m["W_d"]) + (m["W_c"] - m["W_b"]))
            inter = 0.5 * ((m["W_a"] - m["W_c"]) - (m["W_d"] - m["W_b"]))
            share = post / gap if gap else float("nan")
            nonneg = abs(math.log(m["W_a"] / m["W_b"])) > 0.02
            ok = nonneg and post * gap > 0 and abs(post) >= 0.5 * abs(gap)
            n_ok += ok
            theory_sign = (m["W_a"] - m["W_e"]) * gap > 0
            print(f"| {fam} | {ds} | {gap:+.4f}{'' if nonneg else ' (negligible)'} | {post:+.4f} | {dec:+.4f} | "
                  f"{inter:+.4f} | {100 * share:.0f}% {'PASS' if ok else ''} | {'yes' if theory_sign else 'no'} |")
        passes[fam] = n_ok
    print()
    for fam, n in passes.items():
        print(f"- {fam}: {n}/5 datasets pass -> {'family passes' if n >= 3 else 'family fails'}")
    n_fam = sum(n >= 3 for n in passes.values())
    print(f"\n**Gate B: {n_fam}/4 families pass -> {'PASS' if n_fam >= 3 else 'FAIL'}**")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    a = ap.parse_args()
    if a.summary:
        summary()
        return
    torch.set_num_threads(4)   # leave CPU for the phase grid
    for ds in a.datasets:
        run_dataset(ds)
    summary()


if __name__ == "__main__":
    main()
