"""Checks for the MiSCo components: spline flow, exact mixed marginalisation, KL, HPD sets."""
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scrl import misco_eval as ev  # noqa: E402
from scrl.misco import MixedLatent, rq_spline  # noqa: E402

torch.manual_seed(0)


def test_rq_spline_monotone_bijective_and_logdet():
    raw = torch.randn(5, 25)
    u = torch.linspace(0.001, 0.999, 2001).view(-1, 1).expand(2001, 5)
    z, logdet = rq_spline(u, raw.expand(2001, 5, 25))
    assert (z[1:] >= z[:-1] - 1e-7).all()                                   # monotone
    assert (z[0] < 0.01).all() and (z[-1] > 0.99).all()                      # onto [0,1]
    # exact derivative by autograd (finite differences are too coarse where the spline is steep)
    uu = u.clone().double().requires_grad_(True)
    zz, ld = rq_spline(uu, raw.double().expand(2001, 5, 25))
    (grad,) = torch.autograd.grad(zz.sum(), uu)
    inside = (zz > 1e-5) & (zz < 1 - 1e-5)                                   # away from the output clamp
    assert torch.allclose(grad[inside], ld.exp()[inside], rtol=1e-6, atol=1e-8)


def test_mixed_weights_marginalisation_and_kl():
    torch.manual_seed(1)                     # independent of test order
    for interior in ("logit_normal", "flow"):
        lat = MixedLatent(2, interior)
        raw = torch.randn(3, 2, lat.n_per_dim)
        p = lat.params(raw)
        h, w = lat.samples(p, 40000)
        assert torch.allclose(w.sum(0), torch.ones(3), atol=1e-5)
        pi = p["logpi"].exp()
        # exact atom probabilities: P(h_j = 0) = pi_j0
        p0 = (w.unsqueeze(-1) * (h == 0).float()).sum(0)
        assert torch.allclose(p0, pi[..., 0], atol=1e-5)
        # E[h] from marginalisation matches the analytic posterior mean
        assert torch.allclose((w.unsqueeze(-1) * h).sum(0), lat.mean(p), atol=1e-2)
        kl = lat.kl(p, S=40000)
        assert torch.isfinite(kl).all() and (kl > -1e-2).all()      # KL >= 0 up to Monte Carlo error


def test_hpd_set_covers_at_nominal_rate_with_atoms():
    # a model whose predictive is exactly the data distribution must reach ~90% coverage
    B = 20000
    loc, scale = torch.full((1, B), 0.6), torch.full((1, B), 0.15)
    logpi = torch.log(torch.tensor([0.3, 0.1, 0.6])).expand(1, B, 3)
    pred = {"w": torch.ones(1, B), "o": {"loc": loc, "scale": scale, "logpi": logpi}}
    c = torch.distributions.Categorical(logits=logpi[0]).sample()
    yc = __import__("scrl.tn", fromlist=["x"]).rsample(loc[0], scale[0], torch.rand(B))
    y = torch.where(c == 0, torch.zeros(B), torch.where(c == 1, torch.ones(B), yc))
    s, aux = ev.hpd_scores(pred, "atom_tn", y)
    q = ev.conformal_quantile(s[:10000], 0.1)
    cov = (s[10000:] <= q).float().mean().item()
    assert abs(cov - 0.9) < 0.02
    width, atoms = ev.hpd_measure({k: (v[10000:] if torch.is_tensor(v) else v) for k, v in aux.items()},
                                  torch.full((10000,), q))
    assert (atoms >= 1).all()                          # the 30% atom at 0 is always included
    assert width.mean() < 0.9                          # interior part is a strict subset of (0,1)
