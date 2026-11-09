"""ResOpsUS training-parameterisation effect: confirmation on NEW seeds 8-15 (pre-registered in PLAN.md).

Trains models 1 and 3 (checkpoints saved), evaluates A (continuous HPD, model 1), B (free-atom HPD,
model 3) and D (model 3 with atoms dropped) exactly as in Gate O1, then tests D vs A on mu with a
paired Wilcoxon over seeds 8-15 only; seeds 0-15 combined is reported as secondary.
"""
import itertools
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import m0_one  # noqa: E402
import o1  # noqa: E402
from scrl import o1_eval as oe  # noqa: E402
from scrl.misco_eval import group_by_obs  # noqa: E402

NEW = range(8, 16)
OUT = ROOT / "results" / "resops_ext.json"


def main():
    from run_phase import run_one
    for s in NEW:                                   # sequential cache warm-up (no write races)
        m0_one.load("resops", s)
    cmds = [[sys.executable, str(ROOT / "scripts" / "m0_one.py"), "--dataset", "resops", "--model", str(m),
             "--seed", str(s), "--out_tag", o1.TAG, "--save_ckpt"]
            for m, s in itertools.product([1, 3], NEW)
            if not (ROOT / "checkpoints" / o1.TAG / "resops" / f"model{m}__s{s}.pt").exists()]
    with ThreadPoolExecutor(4) as pool:
        list(pool.map(run_one, cmds))

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for s in NEW:
        data, d, K, obs_noise = m0_one.load("resops", s)
        n_tr = data["train"]["x"].shape[0]
        cal = {k: v.to(dev) for k, v in data["cal"].items() if torch.is_tensor(v)}
        test = {k: v.to(dev) for k, v in data["test"].items() if torch.is_tensor(v)}
        gc, gt = group_by_obs(cal["x"], d, 0.05), group_by_obs(test["x"], d, 0.05)
        M1 = o1.load_model("resops", 1, s, d, cal["x"].shape[-1], K, obs_noise, n_tr, dev)
        M3 = o1.load_model("resops", 3, s, d, cal["x"].shape[-1], K, obs_noise, n_tr, dev)
        P1 = {sp: o1.predictive(M1, D, s) for sp, D in (("cal", cal), ("test", test))}
        P3 = {sp: o1.predictive(M3, D, s) for sp, D in (("cal", cal), ("test", test))}
        yc, yt = cal["y"], test["y"]
        rows.append({"seed": s,
                     "A": o1.hpd_method(P1["cal"], P1["test"], "tn", yc, yt, gc, gt),
                     "B": o1.hpd_method(P3["cal"], P3["test"], "atom_tn", yc, yt, gc, gt),
                     "D": o1.hpd_method(oe.drop_atoms(P3["cal"]), oe.drop_atoms(P3["test"]), "tn", yc, yt, gc, gt)})
        print(f"seed {s}: " + " ".join(f"{k}={rows[-1][k]['mu']:.4f}/{rows[-1][k]['coverage']:.3f}" for k in "ABD"), flush=True)
    OUT.write_text(json.dumps(rows, indent=1, default=float))

    def test(pairs, label):
        d_mu = [r["D"]["mu"] for r in pairs]
        a_mu = [r["A"]["mu"] for r in pairs]
        diff = [x - y for x, y in zip(d_mu, a_mu)]
        p = wilcoxon(diff).pvalue
        rel = sum(d_mu) / sum(a_mu) - 1
        print(f"{label}: D {sum(d_mu) / len(d_mu):.4f} vs A {sum(a_mu) / len(a_mu):.4f} ({100 * rel:+.1f}%), "
              f"Wilcoxon p = {p:.4f}, n = {len(pairs)}")
        return rel, p

    rel, p = test(rows, "CONFIRMATORY (new seeds 8-15)")
    old = [{"seed": r["seed"], **{k: r["methods"][k] for k in "ABD"}}
           for r in json.loads((ROOT / "results" / "o1.json").read_text()) if r["dataset"] == "resops"]
    test(old + rows, "secondary (seeds 0-15 combined)")
    print(f"\n**{'CONFIRMED' if (p < 0.05 and rel < 0) else 'NOT CONFIRMED -> close this branch'}**")


if __name__ == "__main__":
    main()
