"""HPD grid-convergence check (pre-registered in PLAN.md).

Step 1 (training): python scripts/conv_check.py --train   -> models 1 and 3 on seaice/resops, 8 seeds, checkpoints saved
Step 2 (analysis): python scripts/conv_check.py            -> re-evaluate the same checkpoints for G in 200/500/1000/2000
"""
import argparse
import itertools
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import m0_one  # noqa: E402
from scrl import misco_eval as ev  # noqa: E402
from scrl.misco import MiscoModel  # noqa: E402

DATASETS = ["seaice", "resops"]
MODELS = [1, 3]
SEEDS = range(8)
GRIDS = [200, 500, 1000, 2000]
TAG = "m0_rerun"


def train_all(workers):
    from run_phase import run_one
    cmds = [[sys.executable, str(ROOT / "scripts" / "m0_one.py"), "--dataset", ds, "--model", str(m), "--seed", str(s),
             "--out_tag", TAG, "--save_ckpt"]
            for ds, m, s in itertools.product(DATASETS, MODELS, SEEDS)
            if not (ROOT / "checkpoints" / TAG / ds / f"model{m}__s{s}.pt").exists()]
    print(f"{len(cmds)} training runs", flush=True)
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(run_one, cmds))


@torch.no_grad()
def preds(model, split, bs=200):
    out = []
    for i in range(0, split["x"].shape[0], bs):
        out.append(model.predictive(split["x"][i:i + bs], split["s"][i:i + bs], S=16))
    return out


def analyse():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for ds, seed in itertools.product(DATASETS, SEEDS):
        data, d, K, obs_noise = m0_one.load(ds, seed)
        cal = {k: v.to(dev) for k, v in data["cal"].items() if torch.is_tensor(v)}
        test = {k: v.to(dev) for k, v in data["test"].items() if torch.is_tensor(v)}
        gc = ev.group_by_obs(data["cal"]["x"], d, 0.05)
        gt = ev.group_by_obs(data["test"]["x"], d, 0.05)
        for m in MODELS:
            latent, head = m0_one.SPEC[m]
            model = MiscoModel(latent, head, d, cal["x"].shape[-1], K, obs_noise,
                               n_train=data["train"]["x"].shape[0]).to(dev)
            model.load_state_dict(torch.load(ROOT / "checkpoints" / TAG / ds / f"model{m}__s{seed}.pt",
                                             map_location=dev, weights_only=True))
            model.eval()
            torch.manual_seed(1000 + seed)
            pc, pt = preds(model, cal), preds(model, test)
            if m == 1:   # grid-independent residual scores: models 1 and 7
                yh_c, sg_c = map(torch.cat, zip(*[ev._mix_moments(p) for p in pc]))
                yh_t, sg_t = map(torch.cat, zip(*[ev._mix_moments(p) for p in pt]))
                s_c = (cal["y"].view(-1) - yh_c).abs() / sg_c
                s_t = (test["y"].view(-1) - yh_t).abs() / sg_t
                for name, meas in (("1", lambda q: (2 * q * sg_t.cpu(), torch.zeros_like(q))),
                                   ("7", lambda q: (((yh_t.cpu() + q * sg_t.cpu()).clamp(max=1)
                                                     - (yh_t.cpu() - q * sg_t.cpu()).clamp(min=0)).clamp_min(0),
                                                    torch.zeros_like(q)))):
                    r = ev.evaluate(name, {"s": s_c.cpu(), "grp": gc}, {"s": s_t.cpu(), "grp": gt, "measure": meas}, 0.05)
                    rows.append({"dataset": ds, "seed": seed, "score": name, "G": None, **r})
            for G in GRIDS:
                def hpd(plist, y):
                    ss, auxs = [], []
                    for j, p in enumerate(plist):
                        s, aux = ev.hpd_scores(p, head, y[j * 200:(j + 1) * 200], G)
                        ss.append(s.cpu())
                        auxs.append({k: (v.cpu() if torch.is_tensor(v) else v) for k, v in aux.items()})
                    cat = {k: (torch.cat([a[k] for a in auxs]) if torch.is_tensor(auxs[0][k]) else auxs[0][k])
                           for k in auxs[0]}
                    return torch.cat(ss), cat
                s_c, _ = hpd(pc, cal["y"])
                s_t, aux_t = hpd(pt, test["y"])
                name = "1b" if m == 1 else "3"
                r = ev.evaluate(name, {"s": s_c, "grp": gc},
                                {"s": s_t, "grp": gt, "measure": lambda q: ev.hpd_measure(aux_t, q)}, 0.05)
                qb = torch.tensor([r["q_interior"], r["q_boundary"]])[gt]
                span, pieces = ev.hpd_shape(aux_t, qb)
                b = gt == 1
                r["span_boundary"] = span[b].mean().item()
                r["pieces_boundary"] = pieces[b].mean().item()
                rows.append({"dataset": ds, "seed": seed, "score": name, "G": G, **r})
            print(f"{ds} seed {seed} model {m} done", flush=True)
    out = ROOT / "results" / "conv_check.json"
    out.write_text(json.dumps(rows, indent=1))
    report(rows)


def report(rows):
    def mean(ds, score, G, key):
        v = [r[key] for r in rows if r["dataset"] == ds and r["score"] == score and r["G"] == G]
        return sum(v) / len(v)
    ok = True
    for ds in DATASETS:
        print(f"\n## {ds} (obs grouping, eps = 0.05, mean over 8 seeds)\n")
        print("| score | G | boundary width (Lebesgue) | span | pieces | atom rate | coverage | boundary cov |")
        print("|---|---|---|---|---|---|---|---|")
        for sc in ("1", "7"):
            print(f"| {sc} | - | {mean(ds, sc, None, 'width_boundary'):.5f} | | | | {mean(ds, sc, None, 'coverage'):.4f} | "
                  f"{mean(ds, sc, None, 'coverage_boundary'):.4f} |")
        for sc in ("1b", "3"):
            for G in GRIDS:
                print(f"| {sc} | {G} | {mean(ds, sc, G, 'width_boundary'):.5f} | {mean(ds, sc, G, 'span_boundary'):.5f} | "
                      f"{mean(ds, sc, G, 'pieces_boundary'):.2f} | {mean(ds, sc, G, 'atom_rate_boundary'):.3f} | "
                      f"{mean(ds, sc, G, 'coverage'):.4f} | {mean(ds, sc, G, 'coverage_boundary'):.4f} |")
            w1, w2 = mean(ds, sc, 1000, "width_boundary"), mean(ds, sc, 2000, "width_boundary")
            c1, c2 = mean(ds, sc, 1000, "coverage"), mean(ds, sc, 2000, "coverage")
            crit1, crit2 = abs(w2 - w1) / w2 < 0.02, abs(c2 - c1) < 0.002
            ok &= crit1 and crit2
            print(f"\n{sc}: |W2000-W1000|/W2000 = {100 * abs(w2 - w1) / w2:.2f}% ({'pass' if crit1 else 'FAIL'}); "
                  f"|cov2000-cov1000| = {abs(c2 - c1):.4f} ({'pass' if crit2 else 'FAIL'})\n")
        for G in (1000, 2000):
            order = sorted([("3", mean(ds, "3", G, "width_boundary")), ("1b", mean(ds, "1b", G, "width_boundary")),
                            ("7", mean(ds, "7", None, "width_boundary")), ("1", mean(ds, "1", None, "width_boundary"))],
                           key=lambda t: t[1])
            print(f"ranking at G={G}: " + " < ".join(f"{n} ({w:.4f})" for n, w in order))
        r3 = {G: mean(ds, "3", G, "width_boundary") / mean(ds, "1b", G, "width_boundary") - 1 for G in GRIDS}
        r7 = {G: mean(ds, "3", G, "width_boundary") / mean(ds, "7", None, "width_boundary") - 1 for G in GRIDS}
        print("model 3 vs 1b: " + ", ".join(f"G={G}: {100 * v:+.1f}%" for G, v in r3.items()))
        print("model 3 vs 7 : " + ", ".join(f"G={G}: {100 * v:+.1f}%" for G, v in r7.items()))
        rk = [tuple(n for n, _ in sorted([("3", mean(ds, "3", G, "width_boundary")),
                                          ("1b", mean(ds, "1b", G, "width_boundary")),
                                          ("7", mean(ds, "7", None, "width_boundary")),
                                          ("1", mean(ds, "1", None, "width_boundary"))], key=lambda t: t[1]))
              for G in (1000, 2000)]
        ok &= rk[0] == rk[1]
        print(f"ranking stable for G >= 1000: {rk[0] == rk[1]}")
    print(f"\n**Convergence check: {'PASS' if ok else 'FAIL'}**")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.train:
        train_all(a.workers)
    else:
        analyse()


if __name__ == "__main__":
    main()
