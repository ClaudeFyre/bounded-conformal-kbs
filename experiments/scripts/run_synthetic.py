"""Train one (dataset, family, decoder, dec_input, seed) configuration and save metrics as JSON.

Example:
    python scripts/run_synthetic.py --dataset syn_full --family kumaraswamy --decoder tn_hetero --seed 0
    python scripts/run_synthetic.py --smoke     # tiny end-to-end check
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scrl import diagnostics  # noqa: E402
from scrl.conformal import mondrian_eval_all  # noqa: E402
from scrl.data_synthetic import SynConfig, generate, syn_extreme_config  # noqa: E402
from scrl.models import Model  # noqa: E402

EPS_GRID = (0.025, 0.05, 0.1, 0.15)


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="syn_full", choices=["syn_full", "syn_extreme", "bucket", "seaice", "resops", "phase", "ladder"])
    ap.add_argument("--level", type=int, default=0, help="ladder: ablation level B0-B5")
    ap.add_argument("--pb", type=float, default=0.4, help="phase: fraction of boundary units")
    ap.add_argument("--post_std", type=float, default=0.05, help="phase: nominal posterior std")
    ap.add_argument("--family", default="kumaraswamy")
    ap.add_argument("--decoder", default="tn_hetero")
    ap.add_argument("--dec_input", default="raw")
    ap.add_argument("--backbone", default="gru", choices=["gru", "tcn"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--S_train", type=int, default=4)
    ap.add_argument("--S_eval", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--smoke", action="store_true")
    return ap.parse_args()


def to(dev, split):
    return {k: v.to(dev) for k, v in split.items()}


@torch.no_grad()
def predict_split(model, split, S, bs=1000):
    outs = []
    for i in range(0, split["x"].shape[0], bs):
        outs.append(model.predict(split["x"][i:i + bs], split["s"][i:i + bs], S=S))
    cat = {k: torch.cat([o[k] for o in outs]).cpu() for k in ("y_hat", "sigma", "post_mean", "leakage")}
    cat["params"] = {k: torch.cat([o["params"][k] for o in outs]) for k in outs[0]["params"]}
    cat["y"] = split["y"].cpu()
    if "h_true" in split:
        cat["h_true"] = split["h_true"].cpu()
    cat["x_last"] = split["x"][:, -1, : model.d].cpu()
    return cat


def load_data(a):
    """Returns (data, name, d, K, obs_noise). Data splits hold x, s, y, h_true."""
    size = dict(n_train=600, n_cal=300, n_test=300, T=50) if a.smoke else {}
    if a.dataset == "ladder":
        from scrl import data_bucket
        # data varies with the seed, as in the phase grid
        cfg = data_bucket.ladder_config(a.level, data_seed=a.seed)
        cache = ROOT / "data_cache" / f"{cfg.name}_seed{cfg.data_seed}.pt"
        if cache.exists():
            data = torch.load(cache, weights_only=True)
        else:
            data = data_bucket.generate(cfg, data_bucket.load_daymet(ROOT / "raw_data" / "daymet"))
            cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save(data, cache)
        return data, cfg.name, data["test"]["h_true"].shape[-1], 1, cfg.obs_noise
    if a.dataset == "phase":
        from scrl import data_phase
        # data varies with the seed so that coverage is averaged over calibration/test draws
        cfg = data_phase.PhaseConfig(post_std=a.post_std, p_boundary=a.pb, data_seed=a.seed)
        if a.smoke:
            cfg.n_train, cfg.n_cal, cfg.n_test, cfg.T = 600, 300, 300, 20
        return data_phase.generate(cfg), cfg.name + ("_smoke" if a.smoke else ""), 1, 1, cfg.obs_noise
    if a.dataset == "resops":
        from scrl import data_resops
        cfg = data_resops.ResOpsConfig(data_seed=a.data_seed, **size)
        if a.smoke:
            cfg.name += "_smoke"
        cache = ROOT / "data_cache" / f"{cfg.name}_seed{cfg.data_seed}.pt"
        if cache.exists():
            data = torch.load(cache, weights_only=True)
        else:
            data = data_resops.generate(cfg, ROOT / "raw_data" / "resopsus" / "ResOpsUS",
                                        ROOT / "raw_data" / "nid" / "nation.csv")
            cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save(data, cache)
        return data, cfg.name, 1, 4, data["obs_noise"]
    if a.dataset == "seaice":
        from scrl import data_seaice
        cfg = data_seaice.SeaIceConfig(data_seed=a.data_seed, **size)
        if a.smoke:
            cfg.name += "_smoke"
        cache = ROOT / "data_cache" / f"{cfg.name}_seed{cfg.data_seed}.pt"
        if cache.exists():
            data = torch.load(cache, weights_only=True)
        else:
            data = data_seaice.generate(cfg, ROOT / "raw_data" / "seaice")
            cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save(data, cache)
        return data, cfg.name, 1, 4, data["obs_noise"]
    if a.dataset == "bucket":
        from scrl import data_bucket
        cfg = data_bucket.BucketConfig(data_seed=a.data_seed, **size)
        if a.smoke:
            cfg.name += "_smoke"
        cache = ROOT / "data_cache" / f"{cfg.name}_seed{cfg.data_seed}.pt"
        if cache.exists():
            data = torch.load(cache, weights_only=True)
        else:
            data = data_bucket.generate(cfg, data_bucket.load_daymet(ROOT / "raw_data" / "daymet"))
            cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save(data, cache)
        return data, cfg.name, 2, 1, cfg.obs_noise
    cfg = (syn_extreme_config if a.dataset == "syn_extreme" else SynConfig)(data_seed=a.data_seed, **size)
    if a.smoke:
        cfg.name += "_smoke"
    return generate(cfg, cache_dir=ROOT / "data_cache"), cfg.name, cfg.d, cfg.K, cfg.obs_noise


def main():
    a = parse()
    torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    if a.smoke:
        a.epochs = 3
    data, name, d, K, obs_noise = load_data(a)

    # train/val split of the training trajectories (val used only for early stopping)
    n_tr = data["train"]["x"].shape[0]
    perm = torch.randperm(n_tr, generator=torch.Generator().manual_seed(a.seed))
    n_val = n_tr // 10
    tr = to(dev, {k: v[perm[n_val:]] for k, v in data["train"].items()})
    va = to(dev, {k: v[perm[:n_val]] for k, v in data["train"].items()})
    cal, test = to(dev, data["cal"]), to(dev, data["test"])

    model = Model(a.family, d=d, n_obs=tr["x"].shape[-1], K=K, decoder=a.decoder, dec_input=a.dec_input,
                  obs_noise=obs_noise, backbone=a.backbone).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)

    best, best_state, bad, log = float("inf"), None, 0, []
    t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        order = torch.randperm(tr["x"].shape[0], device=dev)
        for i in range(0, len(order), a.batch):
            idx = order[i:i + a.batch]
            loss, _ = model.loss(tr["x"][idx], tr["s"][idx], tr["y"][idx], S=a.S_train, beta=a.beta)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            vl, parts = model.loss(va["x"], va["s"], va["y"], S=a.S_train, beta=a.beta)
        log.append({"epoch": ep, "val_loss": vl.item(), **parts})
        if vl.item() < best - 1e-4:
            best, bad = vl.item(), 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= a.patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    train_time = time.time() - t0

    P_cal = predict_split(model, cal, a.S_eval)
    P_test = predict_split(model, test, a.S_eval)
    result = {
        "args": vars(a), "data_config": data["config"], "device": dev,
        "train_time_s": train_time, "epochs_run": len(log), "best_val_loss": best, "log": log,
        "mean_leakage_test": P_test["leakage"].mean().item(),
        "rmse_test": (P_test["y"] - P_test["y_hat"]).pow(2).mean().sqrt().item(),
        "conformal": [mondrian_eval_all(P_cal, P_test, e, a.alpha) for e in EPS_GRID],
        "probe": diagnostics.extrapolation_probe(model, d, device=dev),
        "true_boundary_frac_test": (
            (torch.minimum(P_test["h_true"], 1 - P_test["h_true"]).min(-1).values < 0.05).float().mean().item()
            if "h_true" in P_test else None
        ),
    }
    if a.family == "gaussian":
        R_cal = diagnostics.reference_predictions(model, {k: v.to(dev) for k, v in P_cal["params"].items()}, a.S_eval)
        R_test = diagnostics.reference_predictions(model, {k: v.to(dev) for k, v in P_test["params"].items()}, a.S_eval)
        R_cal = {k: v.cpu() for k, v in R_cal.items()}
        R_test = {k: v.cpu() for k, v in R_test.items()}
        rho = diagnostics.leakage_displacement({k: v.to(dev) for k, v in P_cal["params"].items()}).cpu()
        result["theory"] = [diagnostics.theory_check(P_cal, P_test, R_cal, R_test, rho, e, a.alpha) for e in EPS_GRID]

    if a.backbone != "gru":                  # robustness runs never overwrite the GRU results
        name = f"{name}_{a.backbone}"
    out_dir = Path(a.out) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{a.family}__{a.decoder}__{a.dec_input}__s{a.seed}"
    fn = out_dir / f"{stem}.json"
    fn.write_text(json.dumps(result, indent=1))
    ckpt_dir = ROOT / "checkpoints" / name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / f"{stem}.pt")
    c = result["conformal"][1]
    print(f"[{name}] {a.family}/{a.decoder}/{a.dec_input} seed={a.seed} "
          f"BIW(latent)={c['latent']['biw']:.4f} BIW(obs)={c['obs']['biw']:.4f} BIW(oracle)={c.get('oracle', {}).get('biw', float('nan')):.4f} "
          f"cov={c['obs']['coverage']:.3f} n_cal_b(latent)={c['latent']['n_cal_boundary']} "
          f"leak={result['mean_leakage_test']:.3f} time={train_time:.0f}s")


if __name__ == "__main__":
    main()
