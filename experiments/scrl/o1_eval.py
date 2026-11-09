"""Gate O1 set constructions on a fixed checkpoint (pre-registered in PLAN.md).

A  continuous HPD (TN head of model 1)
B  free-atom HPD (atom_tn head)            C  gated-atom HPD: atom j excluded when pi_j(x) < tau
D  atom_tn head with pi0 = pi1 = 0 at prediction time (renormalised continuous part)
E1 residual interval clipped to [0,1]      E4 CPCI (arXiv 2605.04219), Mondrian-adapted

Every construction returns per-unit (score of the true y) and a measure(q) callable giving
(Lebesgue measure of the continuous part, included atom 0, included atom 1).
"""
import math

import torch

from . import tn
from .misco_eval import _interior_masses, _mix_moments, conformal_quantile

G = 1000   # grid size for HPD sets (the convergence check showed < 1% change beyond G = 1000)


def hpd(pred, head, y, tau=None, G=G):
    """Atom-aware HPD with optional hard gate. head 'tn' ignores atoms (used for A and D)."""
    cm, a0, a1 = _interior_masses(pred, head, G)
    if head != "atom_tn":
        a0, a1 = torch.zeros_like(a0), torch.zeros_like(a1)
    act0 = a0 > 0 if tau is None else a0 >= tau
    act1 = a1 > 0 if tau is None else a1 >= tau
    if head != "atom_tn":
        act0 = act1 = torch.zeros_like(a0, dtype=torch.bool)
    m0, m1 = a0 * act0, a1 * act1
    first0 = m0 >= m1
    inf = torch.full_like(a0, float("inf"))
    s0 = torch.where(act0, torch.where(first0, torch.zeros_like(a0), m1), inf)
    s1 = torch.where(act1, torch.where(first0 & act0, m0, torch.zeros_like(a1)), inf)
    A = m0 + m1
    order = torch.argsort(cm, -1, descending=True)
    sorted_m = cm.gather(-1, order)
    s_sorted = A.unsqueeze(-1) + torch.cumsum(sorted_m, -1) - sorted_m
    s_cell = torch.empty_like(s_sorted).scatter_(-1, order, s_sorted)
    yv = y.view(-1)
    k = (yv * G).long().clamp(0, G - 1)
    s_y = s_cell.gather(-1, k.unsqueeze(-1)).squeeze(-1)
    s_y = torch.where(yv <= 0, s0 if head == "atom_tn" else s_y, torch.where(yv >= 1, s1 if head == "atom_tn" else s_y, s_y))
    aux = {"s_sorted": s_sorted, "s0": s0, "s1": s1}
    return s_y, aux


def hpd_measure(aux, q):
    mu = (aux["s_sorted"] <= q.unsqueeze(-1)).sum(-1).float() / aux["s_sorted"].shape[-1]
    return mu, aux["s0"] <= q, aux["s1"] <= q


def residual(pred, y):
    yhat, sig = _mix_moments(pred)
    return (y.view(-1) - yhat).abs() / sig, {"yhat": yhat, "sig": sig}


def residual_clipped_measure(aux, q):
    lo, hi = aux["yhat"] - q * aux["sig"], aux["yhat"] + q * aux["sig"]
    mu = (hi.clamp(max=1) - lo.clamp(min=0)).clamp_min(0)
    z = torch.zeros_like(q, dtype=torch.bool)
    return mu, z, z


def drop_atoms(pred):
    """Model-3 predictive with pi0 = pi1 = 0: only the (renormalised) continuous TN part."""
    return {"w": pred["w"], "o": {"loc": pred["o"]["loc"], "scale": pred["o"]["scale"]}}


# ---------------------------------------------------------------------------
# CPCI (E4)
# ---------------------------------------------------------------------------
R_GRID = [round(0.05 * i, 2) for i in range(20)]       # 0, 0.05, ..., 0.95


def cpci_parts(pred):
    """p_hat = P(Y != 0 | x) and f_hat = E[Y | Y > 0, x] from an atom_tn predictive."""
    w, o = pred["w"], pred["o"]
    pi = o["logpi"].exp()
    m = tn.mean(o["loc"], o["scale"])
    p_pos = (w * (pi[..., 1] + pi[..., 2])).sum(0)
    ey_pos = (w * (pi[..., 1] + pi[..., 2] * m)).sum(0)
    return p_pos, ey_pos / p_pos.clamp_min(1e-12)


def _kth(values, level):
    """ceil(level (n+1))-th smallest of values (+inf if it does not exist); level <= 0 -> -inf."""
    n = values.numel()
    if level <= 0:
        return -math.inf
    k = math.ceil(level * (n + 1))
    return math.inf if k > n or n == 0 else torch.sort(values).values[k - 1].item()


def cpci_fit(p_val, f_val, y_val, p_c1, p_c2, f_c2, y_c2, alpha=0.1):
    """Choose r on D_val, alpha_r on D_cal1, beta on D_val, q_r on D_cal2. Returns (alpha_r, q_r, r)."""
    target = 1 - alpha
    best = None
    for r in R_GRID:
        a_r = _kth(torch.cat([p_c1, torch.ones(1)]), r) if r > 0 else -math.inf
        zero_val = p_val <= a_r
        beta = (y_val[zero_val] <= 0).float().mean().item() if zero_val.any() else 0.0
        gamma = max(0.0, min(1.0, (target - r * beta) / (1 - r)))
        s_val = (y_val - f_val).abs()[~zero_val]
        q_val = conformal_quantile(s_val, 1 - gamma) if gamma > 0 else 0.0
        length = 2 * (1 - r) * q_val
        if best is None or length < best[0]:
            best = (length, r, a_r, gamma, beta)
    _, r, a_r, gamma, beta = best
    s_c2 = (y_c2 - f_c2).abs()[p_c2 > a_r]
    q = _kth(s_c2, gamma) if gamma > 0 else 0.0
    return a_r, q, r, gamma, beta


# ---------------------------------------------------------------------------
# E5: dual-boundary extension (frozen specification in PLAN.md; not the CPCI paper)
# ---------------------------------------------------------------------------
E5_GRID = [0.0, 0.025, 0.05, 0.10, 0.20, 0.30]
E5_PAIRS = [(a, b) for a in E5_GRID for b in E5_GRID if a + b <= 0.5 + 1e-12]
LAMBDA_GRID = [round(0.01 * i, 2) for i in range(11)]


def e5_parts(pred):
    """Atom probabilities p0, p1 and centre f = E[Y | x] of an atom_tn predictive mixture."""
    w, o = pred["w"], pred["o"]
    pi = o["logpi"].exp()
    m = tn.mean(o["loc"], o["scale"])
    p0, p1 = (w * pi[..., 0]).sum(0), (w * pi[..., 1]).sum(0)
    f = (w * (pi[..., 1] + pi[..., 2] * m)).sum(0)
    return p0, p1, f


def _threshold(p_cal1, r):
    """Routing threshold: the (1 - r) quantile of p on D_cal,1 (r = 0 -> +inf, nobody routed)."""
    if r <= 0 or p_cal1.numel() == 0:
        return math.inf
    return torch.quantile(p_cal1.float(), 1 - r, interpolation="higher").item()


def e5_route(p0, p1, t0, t1):
    """0 = continuous branch, 1 = atom {0}, 2 = atom {1}; conflicts by normalised margin, ties -> continuous."""
    r0, r1 = p0 >= t0, p1 >= t1
    m0 = torch.where(r0, (p0 - t0) / max(1 - t0, 1e-12), torch.full_like(p0, -math.inf))
    m1 = torch.where(r1, (p1 - t1) / max(1 - t1, 1e-12), torch.full_like(p1, -math.inf))
    route = torch.zeros_like(p0, dtype=torch.long)
    route[r0 & (~r1 | (m0 > m1))] = 1
    route[r1 & (~r0 | (m1 > m0))] = 2
    return route


def _q_conformal(scores, gamma):
    """Frozen finite-sample convention: k = ceil((n_c + 1) gamma); k > n_c -> +inf; gamma <= 0 -> 0."""
    n = scores.numel()
    if gamma <= 0:
        return 0.0
    k = math.ceil((n + 1) * gamma)
    return math.inf if k > n or n == 0 else torch.sort(scores).values[k - 1].item()


def _gamma_raw(r0, r1, b0, b1, target=0.9):
    return (target - r0 * b0 - r1 * b1) / (1 - r0 - r1)


def _gamma(r0, r1, b0, b1, target=0.9):
    return max(0.0, min(1.0, _gamma_raw(r0, r1, b0, b1, target)))


def e5_measure(route, f, q):
    lo, hi = f - q, f + q
    mu = torch.where(route == 0, (hi.clamp(max=1) - lo.clamp(min=0)).clamp_min(0), torch.zeros_like(f))
    return mu, route == 1, route == 2


def e5_fit(val, cal1, cal2):
    """val / cal1 / cal2: dicts with p0, p1, f, y (1-D tensors). Returns the fitted configuration and a report."""
    report = []
    rows = {}
    for r0, r1 in E5_PAIRS:
        t0, t1 = _threshold(cal1["p0"], r0), _threshold(cal1["p1"], r1)
        rv = e5_route(val["p0"], val["p1"], t0, t1)
        yv = val["y"]
        b0 = (yv[rv == 1] <= 0).float().mean().item() if (rv == 1).any() else 0.0
        b1 = (yv[rv == 2] >= 1).float().mean().item() if (rv == 2).any() else 0.0
        g = _gamma(r0, r1, b0, b1)
        s_val = (yv - val["f"]).abs()[rv == 0]
        q_val = _q_conformal(s_val, g)
        mu, i0, i1 = e5_measure(rv, val["f"], q_val)
        rc2 = e5_route(cal2["p0"], cal2["p1"], t0, t1)
        n_c = int((rc2 == 0).sum())
        # feasibility (M0/E5 revision, found in the development self-check): the coverage budget must be
        # attainable, i.e. the unclipped gamma <= 1 and the validation quantile finite. Otherwise clipping
        # gamma to 1 silently breaks coverage while the clipped measure hides q = +inf (mu = 1 < inf).
        feasible = _gamma_raw(r0, r1, b0, b1) <= 1 and math.isfinite(q_val)
        rows[(r0, r1)] = {"t0": t0, "t1": t1, "beta0": b0, "beta1": b1, "gamma": g, "feasible": feasible,
                          "mu": mu.mean().item(), "n_atom": (i0.float() + i1.float()).mean().item(), "n_c": n_c}
    # minimax regret over the lambda grid among feasible candidates
    # (a linear lambda-average would collapse to A_0.05; (0, 0) = plain conformal is always feasible)
    A = {c: [v["mu"] + lam * v["n_atom"] for lam in LAMBDA_GRID] for c, v in rows.items() if v["feasible"]}
    best_per_lam = [min(A[c][i] for c in A) for i in range(len(LAMBDA_GRID))]
    regret = {c: max(A[c][i] - best_per_lam[i] for i in range(len(LAMBDA_GRID))) for c in A}
    c_star = min(regret, key=lambda c: (regret[c], c))
    fit = dict(rows[c_star])
    fit["r0"], fit["r1"], fit["regret"] = c_star[0], c_star[1], regret[c_star]
    rc2 = e5_route(cal2["p0"], cal2["p1"], fit["t0"], fit["t1"])
    fit["q"] = _q_conformal((cal2["y"] - cal2["f"]).abs()[rc2 == 0], fit["gamma"])
    report = [{"r0": c[0], "r1": c[1], **rows[c], "regret": regret.get(c, float("nan"))} for c in rows]
    return fit, report


def e5_sets(test, fit):
    route = e5_route(test["p0"], test["p1"], fit["t0"], fit["t1"])
    mu, i0, i1 = e5_measure(route, test["f"], fit["q"])
    y = test["y"]
    lo, hi = test["f"] - fit["q"], test["f"] + fit["q"]
    cov = torch.where(route == 1, y <= 0, torch.where(route == 2, y >= 1, (y >= lo) & (y <= hi)))
    return mu, i0, i1, cov


def cpci_sets(p, f, y, a_r, q):
    zero = p <= a_r
    lo, hi = f - q, f + q
    mu = torch.where(zero, torch.zeros_like(f), (hi.clamp(max=1) - lo.clamp(min=0)).clamp_min(0))
    raw = torch.where(zero, torch.zeros_like(f), torch.full_like(f, 2 * q))
    yv = y.view(-1)
    cov = torch.where(zero, yv <= 0, (yv >= lo) & (yv <= hi))
    return mu, raw, zero, cov
