"""Gate M0 evaluation: conformal sets for each score type, CRPS, NLL and atom calibration.

Scores
  residual  |y - yhat| / sigma              (model 1; clipped width = model 7)
  hpd       atom-aware HPD: s(y) = predictive mass strictly more plausible than y;
            atoms (infinite density w.r.t. Lebesgue) come first, the interior is
            ranked by density on a G-cell grid. Sets = atoms U interior HPD region.
  cqr       max(lo - y, y - hi)
  wu        |Phi^{-1}(F_beta(y_squeezed))|  (quantile-residual score, Wu et al. style)
All sets are calibrated by (Mondrian) split conformal on the calibration split.
"""
import math

import torch
from torch.special import ndtri

from . import tn
from .data_synthetic import boundary_distance

G = 200   # default grid size (Gate M0); the convergence check passes other values explicitly


def _mix_moments(pred):
    o, w = pred["o"], pred["w"]
    m, v = tn.mean(o["loc"], o["scale"]), tn.variance(o["loc"], o["scale"])
    yhat = (w * m).sum(0)
    var = (w * (v + m ** 2)).sum(0) - yhat ** 2
    return yhat, var.clamp_min(1e-10).sqrt()


def _interior_masses(pred, head, G=G):
    """[B, G] interior cell masses and [B] atom masses a0, a1 of the predictive mixture."""
    o, w = pred["o"], pred["w"]
    cells = ((torch.arange(G, device=w.device) + 0.5) / G).view(G, 1, 1)
    dens = tn.log_prob(cells, o["loc"].unsqueeze(0), o["scale"].unsqueeze(0)).exp()        # [G, N, B]
    if head == "atom_tn":
        pi = o["logpi"].exp()
        dens = dens * pi[..., 2].unsqueeze(0)
        a0, a1 = (w * pi[..., 0]).sum(0), (w * pi[..., 1]).sum(0)
    else:
        a0 = a1 = torch.zeros(w.shape[1], device=w.device)
    cm = (dens * w.unsqueeze(0)).sum(1).T / G                                            # [B, G]
    cm = cm / cm.sum(-1, keepdim=True).clamp_min(1e-12) * (1 - a0 - a1).unsqueeze(-1)    # renormalise the grid
    return cm, a0, a1


def hpd_scores(pred, head, y, G=G):
    """Returns (score of the true y, per-unit data needed to measure sets at any threshold)."""
    cm, a0, a1 = _interior_masses(pred, head, G)
    B = cm.shape[0]
    # atom ranks: the heavier atom first
    first0 = a0 >= a1
    s_atom0 = torch.where(first0, torch.zeros_like(a0), a1)
    s_atom1 = torch.where(first0, a0, torch.zeros_like(a1))
    A = a0 + a1
    order = torch.argsort(cm, -1, descending=True)
    sorted_m = cm.gather(-1, order)
    excl = torch.cumsum(sorted_m, -1) - sorted_m                                        # mass strictly before
    s_cell_sorted = A.unsqueeze(-1) + excl
    s_cell = torch.empty_like(s_cell_sorted).scatter_(-1, order, s_cell_sorted)         # [B, G] by cell
    yv = y.view(-1)
    k = (yv * G).long().clamp(0, G - 1)
    s_y = s_cell.gather(-1, k.unsqueeze(-1)).squeeze(-1)
    if head == "atom_tn":
        s_y = torch.where(yv <= 0, s_atom0, torch.where(yv >= 1, s_atom1, s_y))
    return s_y, {"s_cell_sorted": s_cell_sorted, "s_atom0": s_atom0, "s_atom1": s_atom1, "atoms": head == "atom_tn",
                 "s_cell": s_cell}


def hpd_measure(aux, q):
    """Lebesgue measure of the interior part and atom inclusion for per-unit thresholds q [B]."""
    # Lebesgue measure = (number of included cells) * cell length: the sum of |I_k| over all
    # disjoint pieces, never the span from the leftmost to the rightmost point; atoms add 0.
    Gc = aux["s_cell_sorted"].shape[-1]
    n_cells = (aux["s_cell_sorted"] <= q.unsqueeze(-1)).sum(-1).float()
    atoms = ((aux["s_atom0"] <= q).float() + (aux["s_atom1"] <= q).float()) if aux["atoms"] else torch.zeros_like(q)
    return n_cells / Gc, atoms


def hpd_shape(aux, q):
    """Span (leftmost to rightmost included cell) and number of disjoint pieces, for comparison only."""
    inc = aux["s_cell"] <= q.unsqueeze(-1)                      # [B, G] in cell order
    Gc = inc.shape[-1]
    idx = torch.arange(Gc, device=inc.device).expand_as(inc)
    any_ = inc.any(-1)
    left = torch.where(inc, idx, Gc).min(-1).values
    right = torch.where(inc, idx, -1).max(-1).values
    span = torch.where(any_, (right - left + 1).float() / Gc, torch.zeros_like(q))
    starts = inc & ~torch.nn.functional.pad(inc[..., :-1], (1, 0))
    return span, starts.sum(-1).float()


def wu_scores(pred, model, y, grid=2000):
    o = pred["o"]
    a, b = o["a"][0], o["b"][0]                              # point latent: a single sample
    ys = (torch.arange(grid, device=a.device) + 0.5) / grid
    logpdf = torch.distributions.Beta(a.unsqueeze(-1), b.unsqueeze(-1)).log_prob(ys)      # [B, grid]
    cdf = torch.cumsum(logpdf.exp(), -1)
    cdf = cdf / cdf[..., -1:].clamp_min(1e-12)
    ysq = model.squeeze_y(y.view(-1))
    k = (ysq * grid).long().clamp(0, grid - 1)
    Fy = cdf.gather(-1, k.unsqueeze(-1)).squeeze(-1).clamp(1e-6, 1 - 1e-6)
    s = ndtri(Fy).abs()
    return s, {"grid_score": ndtri(cdf.clamp(1e-6, 1 - 1e-6)).abs()}


def wu_measure(aux, q):
    return (aux["grid_score"] <= q.unsqueeze(-1)).float().mean(-1)


def crps(pred, head, y, M=64):
    """Sample-based CRPS of the predictive mixture (NaN for the quantile head)."""
    if head == "quantile":
        return torch.full_like(y.view(-1), float("nan"))
    w = pred["w"]
    N, B = w.shape
    idx = torch.multinomial(w.T.clamp_min(0) + 1e-12, M, replacement=True).T               # [M, B]
    o = {k: v.gather(0, idx) if v.dim() == 2 else v.gather(0, idx.unsqueeze(-1).expand(M, B, v.shape[-1]))
         for k, v in pred["o"].items()}
    if head == "beta":
        ys = torch.distributions.Beta(o["a"], o["b"]).sample()
    else:
        ys = tn.rsample(o["loc"], o["scale"], torch.rand_like(o["loc"]))
        if head == "atom_tn":
            c = torch.distributions.Categorical(logits=o["logpi"]).sample()
            ys = torch.where(c == 0, torch.zeros_like(ys), torch.where(c == 1, torch.ones_like(ys), ys))
    yv = y.view(1, -1)
    t1 = (ys - yv).abs().mean(0)
    t2 = (ys.unsqueeze(0) - ys.unsqueeze(1)).abs().mean((0, 1))
    return t1 - 0.5 * t2


def nll(pred, model, y):
    o, w = pred["o"], pred["w"]
    lp = model.logp(o, y.view(1, -1).expand(w.shape))
    return -torch.logsumexp(torch.log(w.clamp_min(1e-30)) + lp, 0)


def ece(p, event, bins=10):
    idx = (p * bins).long().clamp(0, bins - 1)
    out = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            out += m.float().mean() * (p[m].mean() - event[m].float().mean()).abs()
    return float(out)


def conformal_quantile(s, alpha):
    n = s.numel()
    k = math.ceil((1 - alpha) * (n + 1))
    return float("inf") if n == 0 or k > n else torch.sort(s).values[k - 1].item()


def evaluate(score, cal, test, eps, alpha=0.1):
    """cal/test: dicts with s (scores of true y), grp (0/1), measure (callable q -> (width, atoms))."""
    q = torch.tensor([conformal_quantile(cal["s"][cal["grp"] == g], alpha) for g in (0, 1)])
    qt = q[test["grp"]]
    width, atoms = test["measure"](qt)
    cov = test["s"] <= qt
    b = test["grp"] == 1

    def m(v, mask):
        return v[mask].float().mean().item() if mask.any() else float("nan")

    return {"eps": eps, "q_interior": q[0].item(), "q_boundary": q[1].item(),
            "coverage": cov.float().mean().item(), "coverage_boundary": m(cov, b),
            "width_boundary": m(width, b), "width_interior": m(width, ~b), "width_all": width.mean().item(),
            "atom_rate_boundary": m(atoms, b), "n_cal_boundary": int((cal["grp"] == 1).sum()),
            "n_test_boundary": int(b.sum())}


def group_by_obs(x, d, eps):
    return (boundary_distance(x[:, -1, :d]) < eps).long()
