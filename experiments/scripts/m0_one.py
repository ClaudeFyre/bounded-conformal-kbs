"""Gate M0: train one (dataset, model, seed) and evaluate all scores attached to that model.

Model ids (PLAN.md, Gate M0):
  1   gaussian latent + TN head        -> scores: residual (1), residual clipped (7), HPD (1b)
  2   gaussian latent + quantile head  -> CQR (raw and clipped)
  3   gaussian latent + atom_tn head   -> HPD
  4   mixed_ln latent + TN head        -> HPD
  5   mixed_flow latent + TN head      -> HPD
  6   point encoder + Beta head        -> Wu-style quantile-residual score
data_seed = seed for every dataset.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scrl import misco_eval as ev  # noqa: E402
from scrl.misco import MiscoModel  # noqa: E402

SPEC = {1: ("gaussian", "tn"), 2: ("gaussian", "quantile"), 3: ("gaussian", "atom_tn"),
        4: ("mixed_ln", "tn"), 5: ("mixed_flow", "tn"), 6: ("point", "beta")}
EPS_GRID = (0.05, 0.1)


def load(dataset, seed):
    if dataset == "era5_cloud":
        from scrl import data_era5
        cfg = data_era5.ERA5Config(data_seed=seed)
        cache = ROOT / "data_cache" / f"era5_cloud_seed{seed}.pt"
        if cache.exists():
            data = torch.load(cache, weights_only=True)
        else:
            data = data_era5.generate(cfg, ROOT / "raw_data" / "era5_cloud")
            cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save({k: v for k, v in data.items() if k in ("train", "cal", "test")}, cache)
        return data, 1, 1, cfg.obs_noise
    if dataset == "syn_outatom":
        from scrl import data_outatom
        cfg = data_outatom.OutAtomConfig(data_seed=seed)
        return data_outatom.generate(cfg), cfg.d, 1, cfg.obs_noise
    if dataset == "syn_atom":
        from scrl import data_atom
        cfg = data_atom.AtomConfig(data_seed=seed)
        return data_atom.generate(cfg), cfg.d, 1, cfg.obs_noise
    import run_synthetic
    a = argparse.Namespace(dataset=dataset, data_seed=seed, smoke=False, seed=seed, pb=0.4, post_std=0.05, level=0)
    data, name, d, K, obs_noise = run_synthetic.load_data(a)
    return data, d, K, obs_noise


@torch.no_grad()
def split_outputs(model, split, head, model_id, bs=200):
    S = 8 if model.lat.mixed else 16
    out = {k: [] for k in ("s_res", "yhat", "sig", "s_hpd", "hpd", "s_cqr", "lo", "hi", "s_wu", "wu",
                           "crps", "nll", "a0", "a1", "lat_pi")}
    for i in range(0, split["x"].shape[0], bs):
        x, s, y = split["x"][i:i + bs], split["s"][i:i + bs], split["y"][i:i + bs]
        pred = model.predictive(x, s, S=S)
        yv = y.view(-1)
        if head == "tn":
            yhat, sig = ev._mix_moments(pred)
            out["yhat"].append(yhat); out["sig"].append(sig); out["s_res"].append((yv - yhat).abs() / sig)
        if head in ("tn", "atom_tn"):
            sh, aux = ev.hpd_scores(pred, head, y)
            out["s_hpd"].append(sh); out["hpd"].append(aux)
        if head == "atom_tn":
            pi = pred["o"]["logpi"].exp()
            out["a0"].append((pred["w"] * pi[..., 0]).sum(0)); out["a1"].append((pred["w"] * pi[..., 1]).sum(0))
        if head == "quantile":
            lo, hi = (pred["w"] * pred["o"]["lo"]).sum(0), (pred["w"] * pred["o"]["hi"]).sum(0)
            out["lo"].append(lo); out["hi"].append(hi); out["s_cqr"].append(torch.maximum(lo - yv, yv - hi))
        if head == "beta":
            sw, aux = ev.wu_scores(pred, model, y)
            out["s_wu"].append(sw); out["wu"].append(aux)
        out["crps"].append(ev.crps(pred, head, y))
        if head != "quantile":
            out["nll"].append(ev.nll(pred, model, y))
        if "latent_pi" in pred:
            out["lat_pi"].append(pred["latent_pi"])
    cat = {}
    for k, v in out.items():
        if not v:
            continue
        if isinstance(v[0], dict):
            cat[k] = {kk: (torch.cat([t[kk] for t in v]) if torch.is_tensor(v[0][kk]) else v[0][kk]) for kk in v[0]}
        else:
            cat[k] = torch.cat(v)
    return cat


def score_sets(o, head, model_id):
    """Map model outputs to {score_name: (s, measure)} with measure(q) -> (width, atoms)."""
    zeros = lambda q: torch.zeros_like(q)
    sets = {}
    if head == "tn" and model_id == 1:
        yhat, sig = o["yhat"], o["sig"]
        sets["1"] = (o["s_res"], lambda q: (2 * q * sig, zeros(q)))
        sets["7"] = (o["s_res"], lambda q: (((yhat + q * sig).clamp(max=1) - (yhat - q * sig).clamp(min=0)).clamp_min(0), zeros(q)))
    if head in ("tn", "atom_tn"):
        name = "1b" if model_id == 1 else str(model_id)
        aux = o["hpd"]
        sets[name] = (o["s_hpd"], lambda q, aux=aux: ev.hpd_measure(aux, q))
    if head == "quantile":
        lo, hi = o["lo"], o["hi"]
        sets["2"] = (o["s_cqr"], lambda q: ((hi - lo + 2 * q).clamp_min(0), zeros(q)))
        sets["2_clipped"] = (o["s_cqr"], lambda q: (((hi + q).clamp(max=1) - (lo - q).clamp(min=0)).clamp_min(0), zeros(q)))
    if head == "beta":
        aux = o["wu"]
        sets["6"] = (o["s_wu"], lambda q, aux=aux: (ev.wu_measure(aux, q), zeros(q)))
    return sets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out_tag", default="m0", help="results/<out_tag>/ and checkpoints/<out_tag>/")
    ap.add_argument("--save_ckpt", action="store_true")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data, d, K, obs_noise = load(a.dataset, a.seed)
    if a.smoke:
        a.epochs = 2
        data = {k: ({kk: vv[:400] for kk, vv in v.items()} if isinstance(v, dict) and "x" in v else v) for k, v in data.items()}
    latent, head = SPEC[a.model]
    n_tr = data["train"]["x"].shape[0]
    perm = torch.randperm(n_tr, generator=torch.Generator().manual_seed(a.seed))
    nv = n_tr // 10
    to = lambda D, idx=None: {k: (v[idx] if idx is not None else v).to(dev) for k, v in D.items() if torch.is_tensor(v)}
    tr, va = to(data["train"], perm[nv:]), to(data["train"], perm[:nv])
    cal, test = to(data["cal"]), to(data["test"])
    model = MiscoModel(latent, head, d, tr["x"].shape[-1], K, obs_noise, n_train=n_tr).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    best, best_state, bad, log = float("inf"), None, 0, []
    t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        order = torch.randperm(tr["x"].shape[0], device=dev)
        for i in range(0, len(order), a.batch):
            idx = order[i:i + a.batch]
            loss, _ = model.loss(tr["x"][idx], tr["s"][idx], tr["y"][idx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            vl = []
            for i in range(0, va["x"].shape[0], 500):
                l, parts = model.loss(va["x"][i:i + 500], va["s"][i:i + 500], va["y"][i:i + 500])
                vl.append(l.item())
            vl = sum(vl) / len(vl)
        log.append({"epoch": ep, "val_loss": vl, **parts})
        if vl < best - 1e-4:
            best, bad, best_state = vl, 0, {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= a.patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    train_time = time.time() - t0

    oc = split_outputs(model, cal, head, a.model)
    ot = split_outputs(model, test, head, a.model)
    res = {"args": vars(a), "train_time_s": train_time, "epochs_run": len(log), "best_val": best, "log": log,
           "crps_test": ot["crps"].mean().item() if head != "quantile" else None,
           "nll_test": ot["nll"].mean().item() if "nll" in ot else None, "scores": {}}
    yt = test["y"].view(-1).cpu()
    if "a0" in ot:
        res["ece_output_atom"] = (ev.ece(ot["a0"].cpu(), yt <= 0) + ev.ece(ot["a1"].cpu(), yt >= 1)) / 2
    if "lat_pi" in ot and "h_true" in test:
        pi, h = ot["lat_pi"].cpu(), test["h_true"].cpu()
        res["ece_latent_atom"] = sum(ev.ece(pi[:, j, 0], h[:, j] == 0) + ev.ece(pi[:, j, 1], h[:, j] == 1)
                                     for j in range(d)) / (2 * d)
    ot_cpu = {k: (v.cpu() if torch.is_tensor(v) else ({kk: (vv.cpu() if torch.is_tensor(vv) else vv) for kk, vv in v.items()} if isinstance(v, dict) else v)) for k, v in ot.items()}
    oc_cpu = {k: (v.cpu() if torch.is_tensor(v) else ({kk: (vv.cpu() if torch.is_tensor(vv) else vv) for kk, vv in v.items()} if isinstance(v, dict) else v)) for k, v in oc.items()}
    sc, st = score_sets(oc_cpu, head, a.model), score_sets(ot_cpu, head, a.model)
    for name in sc:
        res["scores"][name] = []
        for eps in EPS_GRID:
            gc = ev.group_by_obs(cal["x"].cpu(), d, eps)
            gt = ev.group_by_obs(test["x"].cpu(), d, eps)
            r = ev.evaluate(name, {"s": sc[name][0], "grp": gc}, {"s": st[name][0], "grp": gt, "measure": st[name][1]}, eps)
            res["scores"][name].append(r)
    if a.save_ckpt:
        ck = ROOT / "checkpoints" / a.out_tag / a.dataset
        ck.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ck / f"model{a.model}__s{a.seed}.pt")
    out = ROOT / "results" / a.out_tag / a.dataset
    out.mkdir(parents=True, exist_ok=True)
    (out / f"model{a.model}__s{a.seed}.json").write_text(json.dumps(res, indent=1))
    first = {n: v[0] for n, v in res["scores"].items()}
    print(f"[m0 {a.dataset}] model {a.model} seed {a.seed} time {train_time:.0f}s crps {res['crps_test']} " +
          " ".join(f"{n}: w_b={v['width_boundary']:.4f} cov={v['coverage']:.3f}" for n, v in first.items()), flush=True)


if __name__ == "__main__":
    main()
