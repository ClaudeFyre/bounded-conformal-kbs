"""E5 development self-check on DISCOVERY data only (seaice, resops, syn_outatom; existing checkpoints).

Checks coverage and implementation sanity for A, E4 and E5 on the O1 checkpoints. Not used to change any
design choice (PLAN.md, ERA5 protocol). ERA5 is not touched here.
"""
import itertools
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import m0_one  # noqa: E402
import o1  # noqa: E402
from scrl import o1_eval as oe  # noqa: E402
from scrl.misco_eval import group_by_obs  # noqa: E402


def e5_grouped(P_val, P_cal, P_test, yv, yc, yt, gv, gc, gt, seed):
    """E5 run separately in each observation group (Mondrian adaptation, as E4)."""
    dev = yt.device
    parts = {}
    for name, P, y in (("val", P_val, yv), ("cal", P_cal, yc), ("test", P_test, yt)):
        p0, p1, f = oe.e5_parts(P)
        parts[name] = {"p0": p0.cpu(), "p1": p1.cpu(), "f": f.cpu(), "y": y.view(-1).cpu()}
    n = yc.shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(200 + seed))
    c1 = torch.zeros(n, dtype=torch.bool)
    c1[perm[: n // 2]] = True
    mu = torch.zeros(yt.shape[0]); i0 = torch.zeros_like(mu, dtype=torch.bool); i1 = torch.zeros_like(i0)
    cov = torch.zeros_like(i0)
    fits = {}
    for g in (0, 1):
        sel = lambda D, m: {k: v[m] for k, v in D.items()}
        gvm, gcm, gtm = (gv == g).cpu(), (gc == g).cpu(), (gt == g).cpu()
        fit, rep = oe.e5_fit(sel(parts["val"], gvm), sel(parts["cal"], gcm & c1), sel(parts["cal"], gcm & ~c1))
        m_, a_, b_, c_ = oe.e5_sets(sel(parts["test"], gtm), fit)
        mu[gtm], i0[gtm], i1[gtm], cov[gtm] = m_, a_, b_, c_
        fits[g] = {k: fit[k] for k in ("r0", "r1", "gamma", "q", "n_c", "regret")}
        fits[g]["n_feasible"] = sum(r["feasible"] for r in rep)
    return mu, i0, i1, cov, fits


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = []
    for ds, seed in itertools.product(["seaice", "resops", "syn_outatom"], range(8)):
        data, d, K, obs_noise = m0_one.load(ds, seed)
        n_tr = data["train"]["x"].shape[0]
        perm = torch.randperm(n_tr, generator=torch.Generator().manual_seed(seed))
        val = {k: v[perm[: n_tr // 10]].to(dev) for k, v in data["train"].items() if torch.is_tensor(v)}
        cal = {k: v.to(dev) for k, v in data["cal"].items() if torch.is_tensor(v)}
        test = {k: v.to(dev) for k, v in data["test"].items() if torch.is_tensor(v)}
        gv, gc, gt = (group_by_obs(D["x"], d, 0.05) for D in (val, cal, test))
        M3 = o1.load_model(ds, 3, seed, d, cal["x"].shape[-1], K, obs_noise, n_tr, dev)
        P = {sp: o1.predictive(M3, D, seed) for sp, D in (("val", val), ("cal", cal), ("test", test))}
        mu, i0, i1, cov, fits = e5_grouped(P["val"], P["cal"], P["test"], val["y"], cal["y"], test["y"], gv, gc, gt, seed)
        row = o1.summarise(mu.to(dev), i0.to(dev), i1.to(dev), cov.to(dev), test["y"], gt)
        row.update({"dataset": ds, "seed": seed, "fits": fits})
        out.append(row)
        print(f"{ds} s{seed}: E5 mu={row['mu']:.4f} atoms={row['n_atom']:.2f} cov={row['coverage']:.3f} "
              f"covb={row['coverage_boundary']:.3f} fits={ {g: (f['r0'], f['r1'], f['n_feasible']) for g, f in fits.items()} }", flush=True)
    (ROOT / "results" / "e5_dev.json").write_text(json.dumps(out, indent=1, default=float))
    print("\n## E5 development self-check (discovery data; mean over 8 seeds)\n")
    o1rows = json.loads((ROOT / "results" / "o1.json").read_text())
    for ds in ["seaice", "resops", "syn_outatom"]:
        R = [r for r in out if r["dataset"] == ds]
        m = lambda k: sum(r[k] for r in R) / len(R)
        base = [r for r in o1rows if r["dataset"] == ds]
        ma = lambda meth, k: sum(r["methods"][meth][k] for r in base) / len(base)
        print(f"- {ds}: E5 mu {m('mu'):.4f}, atoms {m('n_atom'):.2f}, A_0.05 {m('A_0.05'):.4f}, coverage {m('coverage'):.3f} | "
              f"A mu {ma('A', 'mu'):.4f} | E4 mu {ma('E4', 'mu'):.4f}, A_0.05 {ma('E4', 'A_0.05'):.4f}")


if __name__ == "__main__":
    main()
