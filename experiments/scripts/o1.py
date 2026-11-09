"""Gate O1: output-layer ablation (pre-registered in PLAN.md).

    python scripts/o1.py --train      # Syn-OutAtom: models 1, 2, 3, 6 x 8 seeds (checkpoints saved)
    python scripts/o1.py --evaluate   # all methods on seaice / resops / syn_outatom -> results/o1.json
    python scripts/o1.py --verdict    # Gate O1 conditions
"""
import argparse
import itertools
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import m0_one  # noqa: E402
from scrl import o1_eval as oe  # noqa: E402
from scrl.misco import MiscoModel  # noqa: E402
from scrl.misco_eval import conformal_quantile, ece, group_by_obs  # noqa: E402

DATASETS = ["seaice", "resops", "syn_outatom"]
SEEDS = range(8)
TAUS = [0.01, 0.05, 0.10]
LAMBDAS = [0.01, 0.05, 0.10]
LAM_PRIMARY = 0.05
TAG = "m0_rerun"
OUT = ROOT / "results" / "o1.json"


# ---------------------------------------------------------------------------
def train(workers):
    from run_phase import run_one
    cmds = [[sys.executable, str(ROOT / "scripts" / "m0_one.py"), "--dataset", "syn_outatom", "--model", str(m),
             "--seed", str(s), "--out_tag", TAG, "--save_ckpt"]
            for m, s in itertools.product([1, 2, 3, 6], SEEDS)
            if not (ROOT / "results" / TAG / "syn_outatom" / f"model{m}__s{s}.json").exists()]
    print(f"{len(cmds)} training runs", flush=True)
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(run_one, cmds))


# ---------------------------------------------------------------------------
def load_model(ds, m, seed, d, n_obs, K, obs_noise, n_train, dev):
    latent, head = m0_one.SPEC[m]
    model = MiscoModel(latent, head, d, n_obs, K, obs_noise, n_train=n_train).to(dev)
    model.load_state_dict(torch.load(ROOT / "checkpoints" / TAG / ds / f"model{m}__s{seed}.pt",
                                     map_location=dev, weights_only=True))
    return model.eval()


@torch.no_grad()
def predictive(model, split, seed, bs=200):
    torch.manual_seed(5000 + seed)
    parts = [model.predictive(split["x"][i:i + bs], split["s"][i:i + bs], S=16) for i in range(0, split["x"].shape[0], bs)]
    cat1 = lambda key: torch.cat([p[key] for p in parts], 1)
    return {"w": cat1("w"), "o": {k: torch.cat([p["o"][k] for p in parts], 1) for k in parts[0]["o"]}}


def subset(pred, idx):
    return {"w": pred["w"][:, idx], "o": {k: v[:, idx] for k, v in pred["o"].items()}}


def mondrian(s_cal, g_cal, s_test, g_test, alpha=0.1):
    q = torch.tensor([conformal_quantile(s_cal[g_cal == g], alpha) for g in (0, 1)], device=s_test.device)
    return q[g_test]


def summarise(mu, inc0, inc1, cov, y, g):
    b = g == 1
    yv = y.view(-1)
    n_atom = inc0.float() + inc1.float()
    hits = (inc0 & (yv <= 0)).float() + (inc1 & (yv >= 1)).float()
    row = {"mu": mu[b].mean().item(), "n_atom": n_atom[b].mean().item(),
           "precision": (hits[b].sum() / n_atom[b].sum()).item() if n_atom[b].sum() > 0 else float("nan"),
           "coverage": cov.float().mean().item(), "coverage_boundary": cov[b].float().mean().item()}
    for lam in LAMBDAS:
        row[f"A_{lam}"] = (mu[b] + lam * n_atom[b]).mean().item()
    return row


def hpd_method(pred_cal, pred_test, head, y_cal, y_test, g_cal, g_test, tau=None):
    s_c, _ = oe.hpd(pred_cal, head, y_cal, tau)
    s_t, aux = oe.hpd(pred_test, head, y_test, tau)
    q = mondrian(s_c, g_cal, s_t, g_test)
    mu, i0, i1 = oe.hpd_measure(aux, q)
    return summarise(mu, i0, i1, s_t <= q, y_test, g_test)


def select_tau(pred_val, y_val, g_val, seed, lam):
    """Pre-registered: calibrate on validation half 1, score J on half 2; coverage >= 0.88 required."""
    n = y_val.shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(100 + seed)).to(y_val.device)
    h1, h2 = perm[: n // 2], perm[n // 2:]
    best = None
    for tau in TAUS:
        r = hpd_method(subset(pred_val, h1), subset(pred_val, h2), "atom_tn", y_val[h1], y_val[h2],
                       g_val[h1], g_val[h2], tau)
        if r["coverage"] < 0.88:
            continue
        J = r[f"A_{lam}"]
        if best is None or J < best[0]:
            best = (J, tau)
    return best[1] if best else max(TAUS)


def evaluate():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for ds, seed in itertools.product(DATASETS, SEEDS):
        data, d, K, obs_noise = m0_one.load(ds, seed)
        n_tr = data["train"]["x"].shape[0]
        perm = torch.randperm(n_tr, generator=torch.Generator().manual_seed(seed))
        val_idx = perm[: n_tr // 10]                       # identical to the split used in m0_one
        val = {k: v[val_idx].to(dev) for k, v in data["train"].items() if torch.is_tensor(v)}
        cal = {k: v.to(dev) for k, v in data["cal"].items() if torch.is_tensor(v)}
        test = {k: v.to(dev) for k, v in data["test"].items() if torch.is_tensor(v)}
        gv, gc, gt = (group_by_obs(D["x"], d, 0.05) for D in (val, cal, test))
        n_obs = cal["x"].shape[-1]
        M1 = load_model(ds, 1, seed, d, n_obs, K, obs_noise, n_tr, dev)
        M3 = load_model(ds, 3, seed, d, n_obs, K, obs_noise, n_tr, dev)
        P1 = {sp: predictive(M1, D, seed) for sp, D in (("cal", cal), ("test", test))}
        P3 = {sp: predictive(M3, D, seed) for sp, D in (("val", val), ("cal", cal), ("test", test))}
        yc, yt = cal["y"], test["y"]
        res = {"dataset": ds, "seed": seed, "methods": {}}
        res["methods"]["A"] = hpd_method(P1["cal"], P1["test"], "tn", yc, yt, gc, gt)
        res["methods"]["B"] = hpd_method(P3["cal"], P3["test"], "atom_tn", yc, yt, gc, gt)
        for lam in LAMBDAS:
            tau = select_tau(P3["val"], val["y"], gv, seed, lam)
            r = hpd_method(P3["cal"], P3["test"], "atom_tn", yc, yt, gc, gt, tau)
            r["tau"] = tau
            res["methods"]["C" if lam == LAM_PRIMARY else f"C_lam{lam}"] = r
        res["methods"]["D"] = hpd_method(oe.drop_atoms(P3["cal"]), oe.drop_atoms(P3["test"]), "tn", yc, yt, gc, gt)
        # E1: residual interval of model 1, clipped to [0,1]
        s_c, _ = oe.residual(P1["cal"], yc)
        s_t, aux = oe.residual(P1["test"], yt)
        q = mondrian(s_c, gc, s_t, gt)
        mu, i0, i1 = oe.residual_clipped_measure(aux, q)
        res["methods"]["E1"] = summarise(mu, i0, i1, s_t <= q, yt, gt)
        # E4: CPCI, run separately inside each observation group (Mondrian adaptation)
        pv, fv = oe.cpci_parts(P3["val"])
        pc, fc = oe.cpci_parts(P3["cal"])
        pt, ft = oe.cpci_parts(P3["test"])
        cperm = torch.randperm(yc.shape[0], generator=torch.Generator().manual_seed(200 + seed)).to(dev)
        c1 = torch.zeros(yc.shape[0], dtype=torch.bool, device=dev)
        c1[cperm[: yc.shape[0] // 2]] = True
        mu = torch.zeros_like(ft); cov = torch.zeros_like(ft, dtype=torch.bool); zero = torch.zeros_like(cov)
        fits = {}
        for g in (0, 1):
            vm, cm1, cm2, tm = gv == g, (gc == g) & c1, (gc == g) & ~c1, gt == g
            a_r, qq, r, gamma, beta = oe.cpci_fit(pv[vm].cpu(), fv[vm].cpu(), val["y"].view(-1)[vm].cpu(),
                                                  pc[cm1].cpu(), pc[cm2].cpu(), fc[cm2].cpu(), yc.view(-1)[cm2].cpu())
            m_, _, z_, c_ = oe.cpci_sets(pt[tm], ft[tm], yt[tm], a_r, qq)
            mu[tm], zero[tm], cov[tm] = m_, z_, c_
            fits[g] = {"r": r, "alpha_r": a_r, "q": qq, "gamma": gamma, "beta": beta}
        res["methods"]["E4"] = summarise(mu, zero, torch.zeros_like(zero), cov, yt, gt)
        res["methods"]["E4"]["fits"] = fits
        # E2 / E3: CQR and Beta+Wu, reused from Gate M0 (seaice/resops) or trained here (syn_outatom)
        src = TAG if ds == "syn_outatom" else "m0"
        for name, mid, key in (("E2", 2, "2_clipped"), ("E3", 6, "6")):
            rr = json.loads((ROOT / "results" / src / ds / f"model{mid}__s{seed}.json").read_text())
            s = rr["scores"][key][0]
            res["methods"][name] = {"mu": s["width_boundary"], "n_atom": 0.0, "precision": float("nan"),
                                    "coverage": s["coverage"], "coverage_boundary": s["coverage_boundary"],
                                    **{f"A_{lam}": s["width_boundary"] for lam in LAMBDAS}}
        if ds == "syn_outatom":
            pi = P3["test"]["o"]["logpi"].exp()
            w = P3["test"]["w"]
            a0, a1 = (w * pi[..., 0]).sum(0).cpu(), (w * pi[..., 1]).sum(0).cpu()
            res["atom_mae"] = ((a0 - test["pi0_true"].cpu()).abs().mean().item() + (a1 - test["pi1_true"].cpu()).abs().mean().item()) / 2
            yv = test["y"].view(-1).cpu()
            res["atom_ece"] = (ece(a0, yv <= 0) + ece(a1, yv >= 1)) / 2
        rows.append(res)
        print(f"{ds} seed {seed}: " + " ".join(f"{k}={v['mu']:.4f}/{v['n_atom']:.2f}/{v['coverage']:.3f}"
                                              for k, v in res["methods"].items() if not k.startswith("C_")), flush=True)
    OUT.write_text(json.dumps(rows, indent=1, default=float))


# ---------------------------------------------------------------------------
def verdict():
    rows = json.loads(OUT.read_text())

    def vals(ds, meth, key):
        return {r["seed"]: r["methods"][meth][key] for r in rows if r["dataset"] == ds}

    def cmp(ds, a, b, key):
        A, B = vals(ds, a, key), vals(ds, b, key)
        s = sorted(set(A) & set(B))
        xa, xb = [A[i] for i in s], [B[i] for i in s]
        diff = [p - q for p, q in zip(xa, xb)]
        p = wilcoxon(diff).pvalue if any(abs(x) > 1e-12 for x in diff) else 1.0
        ma, mb = sum(xa) / len(xa), sum(xb) / len(xb)
        return ma, mb, (ma / mb - 1) if mb else float("nan"), p

    def mean(ds, meth, key):
        v = [x for x in vals(ds, meth, key).values() if x == x]
        return sum(v) / len(v) if v else float("nan")

    for ds in DATASETS:
        print(f"\n## {ds} (boundary group, obs eps = 0.05, mean over 8 seeds)\n")
        print("| method | mu | atoms/set | precision | A_0.01 | A_0.05 | A_0.10 | coverage | boundary cov |")
        print("|---|---|---|---|---|---|---|---|---|")
        for m in ["A", "B", "C", "D", "E1", "E2", "E3", "E4"]:
            print(f"| {m} | {mean(ds, m, 'mu'):.4f} | {mean(ds, m, 'n_atom'):.2f} | {mean(ds, m, 'precision'):.3f} | "
                  f"{mean(ds, m, 'A_0.01'):.4f} | {mean(ds, m, 'A_0.05'):.4f} | {mean(ds, m, 'A_0.1'):.4f} | "
                  f"{mean(ds, m, 'coverage'):.3f} | {mean(ds, m, 'coverage_boundary'):.3f} |")
        taus = [r["methods"]["C"]["tau"] for r in rows if r["dataset"] == ds]
        print(f"\nselected tau (lambda_A = 0.05) per seed: {taus}")
        for lam in (0.01, 0.1):
            print(f"  sensitivity lambda_A = {lam}: taus {[r['methods'][f'C_lam{lam}']['tau'] for r in rows if r['dataset'] == ds]}")
        if ds == "syn_outatom":
            ma = [r["atom_mae"] for r in rows if r["dataset"] == ds]
            ec = [r["atom_ece"] for r in rows if r["dataset"] == ds]
            print(f"  atom probability MAE vs truth {sum(ma) / len(ma):.4f}; atom ECE {sum(ec) / len(ec):.4f}")

    print("\n## Gate O1 conditions\n")
    ok1 = []
    for m in ("B", "C", "D"):
        mu = cmp("seaice", m, "A", "mu")
        amb = cmp("seaice", m, "A", "A_0.05")
        cov = mean("seaice", m, "coverage")
        good = mu[2] < 0 and mu[3] < 0.05 and abs(cov - 0.9) <= 0.02 and not (amb[2] > 0 and amb[3] < 0.05)
        ok1.append(good)
        print(f"1. seaice {m} vs A: mu {mu[0]:.4f} vs {mu[1]:.4f} ({100 * mu[2]:+.1f}%, p={mu[3]:.3f}); "
              f"A_0.05 {100 * amb[2]:+.1f}% (p={amb[3]:.3f}); coverage {cov:.3f} -> {good}")
    print(f"   condition 1: {any(ok1)}")
    gB = mean("resops", "A", "mu") - mean("resops", "B", "mu")
    gD = mean("resops", "A", "mu") - mean("resops", "D", "mu")
    share = gD / gB if gB else float("nan")
    print(f"2. resops: B gain over A {gB:.4f}, D gain over A {gD:.4f} -> D retains {100 * share:.0f}% "
          f"-> {'NOT attributable to atoms' if share >= 0.5 else 'atoms may contribute'}")
    print("3. valid efficiency gains (mu down significantly, A_0.05 not significantly up, precision not down vs B):")
    for ds in DATASETS:
        for m in ("B", "C", "D", "E4"):
            mu = cmp(ds, m, "A", "mu")
            amb = cmp(ds, m, "A", "A_0.05")
            valid = mu[2] < 0 and mu[3] < 0.05 and not (amb[2] > 0 and amb[3] < 0.05)
            prec = ""
            if m in ("C", "E4"):
                pr = cmp(ds, m, "B", "precision")
                if pr[0] == pr[0] and pr[1] == pr[1]:
                    valid = valid and not (pr[2] < 0 and pr[3] < 0.05)
                    prec = f"; precision vs B {pr[0]:.3f} vs {pr[1]:.3f} (p={pr[3]:.3f})"
            print(f"   {ds} {m}: mu {100 * mu[2]:+.1f}% (p={mu[3]:.3f}), A_0.05 {100 * amb[2]:+.1f}% (p={amb[3]:.3f}){prec} -> {valid}")
    print("4. E5 vs E4: pending (E5 not implemented, per pre-registration)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--verdict", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.train:
        train(a.workers)
    if a.evaluate:
        evaluate()
    if a.verdict:
        verdict()


if __name__ == "__main__":
    main()
