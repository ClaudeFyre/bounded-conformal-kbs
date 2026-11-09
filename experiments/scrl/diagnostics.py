"""Direct checks of the theory on a trained Gaussian-latent model.

For each unit we build the support-restricted reference q* (the same
Gaussian conditioned on [0,1]^d) and compare, on the same units:
  * Prop. 3: boundary grouping by mu vs by h* = E_{q*}[h] (only false positives expected);
  * Assumption 4 / Thm. 5: decoder displacement Delta, leakage displacement rho,
    score increments S^G - S*, quantile-critical mass m and crossing mass m_-,
    and the empirical group quantiles q^G vs q*.
Also: the decoder extrapolation probe (percentiles, not just the max).
"""
import torch

from . import tn
from .conformal import conformal_quantile, scores
from .data_synthetic import boundary_distance


@torch.no_grad()
def reference_predictions(model, params, S=32):
    mu, sigma = params["mu"], params["sigma"]
    u = torch.rand((S,) + mu.shape, device=mu.device)
    hs_star = tn.rsample(mu, sigma, u)
    pred = model.predict_from_samples(hs_star)
    pred["h_star"] = tn.mean(mu, sigma)
    return pred


@torch.no_grad()
def leakage_displacement(params, S=256):
    mu, sigma = params["mu"], params["sigma"]
    hs = mu + sigma * torch.randn((S,) + mu.shape, device=mu.device)
    excess = (hs - hs.clamp(0, 1)).norm(dim=-1)   # dist(h, M) for the box
    return excess.mean(0)


def theory_check(cal, test, cal_ref, test_ref, rho_cal, eps, alpha=0.1, taus=(0.05, 0.1, 0.2, 0.5)):
    """All inputs on CPU. cal/test: Gaussian-pipeline outputs; *_ref: q* outputs."""
    out = {"eps": eps}
    # --- Prop. 3: one-sided partition error relative to h*
    for name, P, R in (("cal", cal, cal_ref), ("test", test, test_ref)):
        g_hat = boundary_distance(P["post_mean"]) < eps
        g_star = boundary_distance(R["h_star"]) < eps
        out[f"{name}_fp_vs_star"] = (g_hat & ~g_star).float().mean().item()
        out[f"{name}_fn_vs_star"] = (~g_hat & g_star).float().mean().item()   # theory: exactly 0
        out[f"{name}_D_min"] = (boundary_distance(R["h_star"]) - boundary_distance(P["post_mean"])).min().item()

    # --- Thm. 5 on the deployed boundary calibration group G = Ghat_1^G
    G = boundary_distance(cal["post_mean"]) < eps
    out["n_group"] = int(G.sum())
    if G.sum() < 20:
        return out
    sG = scores(cal["y"], cal["y_hat"], cal["sigma"])[G]
    s_star = scores(cal["y"], cal_ref["y_hat"], cal_ref["sigma"])[G]
    s_star_shared = scores(cal["y"], cal_ref["y_hat"], cal["sigma"])[G]   # shared normaliser
    delta = (cal["y_hat"] - cal_ref["y_hat"]).norm(dim=-1)[G]
    rho = rho_cal[G]
    inc = sG - s_star
    q_g = conformal_quantile(sG, alpha)
    q_s = conformal_quantile(s_star, alpha)
    q_ss = conformal_quantile(s_star_shared, alpha)
    out.update({
        "q_gauss": q_g, "q_star": q_s, "q_star_shared_norm": q_ss,
        "frac_score_increase": (inc > 0).float().mean().item(),
        "inc_median": inc.median().item(),
        "inc_q10": inc.quantile(0.1).item(), "inc_q90": inc.quantile(0.9).item(),
        "delta_median": delta.median().item(),
        "rho_median": rho.median().item(),
        "L_emp_median": (delta / rho.clamp_min(1e-8))[rho > 1e-4].median().item() if (rho > 1e-4).any() else float("nan"),
        "m_minus": ((s_star > q_s) & (sG <= q_s)).float().mean().item(),
    })
    for tau in taus:
        affected = inc >= tau
        out[f"m_tau{tau}"] = (affected & (s_star > q_s - tau) & (s_star <= q_s)).float().mean().item()
    return out


@torch.no_grad()
def extrapolation_probe(model, d, n_rays=200, steps=(0.01, 0.05, 0.1, 0.2), device="cpu", seed=0):
    """Ratio ||mu_dec(h) - mu_dec(Pi(h))|| / dist(h, M) along outward rays from boundary anchors."""
    gen = torch.Generator().manual_seed(seed)
    anchors = torch.rand(n_rays, d, generator=gen)
    face = torch.randint(0, d, (n_rays,), generator=gen)
    side = torch.randint(0, 2, (n_rays,), generator=gen).float()
    anchors[torch.arange(n_rays), face] = side
    normal = torch.zeros(n_rays, d)
    normal[torch.arange(n_rays), face] = 2 * side - 1
    ratios = []
    for t in steps:
        h = (anchors + t * normal).to(device)
        proj = h.clamp(0, 1)
        num = (model.dec_mean_fn(h) - model.dec_mean_fn(proj)).norm(dim=-1)
        ratios.append((num / t).cpu())
    r = torch.cat(ratios)
    return {"L_p05": r.quantile(0.05).item(), "L_median": r.median().item(), "L_max": r.max().item()}
