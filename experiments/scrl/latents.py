"""Latent distribution families compared under the representation-only protocol.

Every family maps the same raw encoder output (2*d numbers per unit) to a
posterior over h in R^d, and exposes: rsample, posterior mean (used for
Mondrian grouping), KL to its prior, and support leakage outside [0,1]^d.

Priors: bounded families use Uniform[0,1]^d. The Gaussian uses
N(0.5, 1/12), which matches the uniform prior's mean and variance.
"""
import math

import torch
import torch.nn.functional as F
from numpy.polynomial.hermite import hermgauss
from torch.distributions import Beta, kl_divergence
from torch.special import digamma, ndtr

from . import tn

EPS = 1e-6
PRIOR_STD = 1 / math.sqrt(12)
EULER_GAMMA = 0.5772156649015329
CONC_MAX = 1e4  # cap on Kumaraswamy/Beta shape parameters; 50 (old spec) cannot fit tight interior posteriors


class Family:
    name = "base"
    bounded = True

    def params(self, raw):
        raise NotImplementedError

    def rsample(self, p, S):
        raise NotImplementedError

    def mean(self, p):
        raise NotImplementedError

    def kl(self, p):
        """KL(q || prior), summed over latent dimensions -> [B]."""
        raise NotImplementedError

    def leakage(self, p):
        """q(h not in [0,1]^d) -> [B]."""
        any_param = next(iter(p.values()))
        return torch.zeros(any_param.shape[0], device=any_param.device)

    def init_bias(self, d):
        return torch.zeros(2 * d)


def _shape(r):
    """Log-space shape parameter in [e^-6, CONC_MAX]; reaches tight posteriors without huge raw outputs."""
    return torch.exp(r.clamp(-6.0, math.log(CONC_MAX)))


def _split(raw):
    d = raw.shape[-1] // 2
    return raw[..., :d], raw[..., d:]


class Gaussian(Family):
    name = "gaussian"
    bounded = False

    def params(self, raw):
        r1, r2 = _split(raw)
        return {"mu": 0.5 + r1, "sigma": F.softplus(r2) + 1e-3}

    def rsample(self, p, S):
        eps = torch.randn((S,) + p["mu"].shape, device=p["mu"].device)
        return p["mu"] + p["sigma"] * eps

    def mean(self, p):
        return p["mu"]

    def kl(self, p):
        mu, s = p["mu"], p["sigma"]
        kl = torch.log(PRIOR_STD / s) + (s**2 + (mu - 0.5) ** 2) / (2 * PRIOR_STD**2) - 0.5
        return kl.sum(-1)

    def leakage(self, p):
        mu, s = p["mu"], p["sigma"]
        inside = ndtr((1 - mu) / s) - ndtr(-mu / s)
        return 1 - inside.prod(-1)

    def init_bias(self, d):
        return torch.cat([torch.zeros(d), torch.full((d,), -2.0)])


class TruncatedGaussian(Family):
    name = "trunc_gaussian"

    def params(self, raw):
        r1, r2 = _split(raw)
        return {"loc": (0.5 + r1).clamp(-1.0, 2.0), "scale": F.softplus(r2) + 1e-3}

    def rsample(self, p, S):
        u = torch.rand((S,) + p["loc"].shape, device=p["loc"].device)
        return tn.rsample(p["loc"], p["scale"], u)

    def mean(self, p):
        return tn.mean(p["loc"], p["scale"])

    def kl(self, p):
        return -tn.entropy(p["loc"], p["scale"]).sum(-1)

    def init_bias(self, d):
        return torch.cat([torch.zeros(d), torch.full((d,), -2.0)])


class Kumaraswamy(Family):
    name = "kumaraswamy"

    def params(self, raw):
        r1, r2 = _split(raw)
        return {"a": _shape(r1), "b": _shape(r2)}

    def rsample(self, p, S):
        a, b = p["a"], p["b"]
        u = torch.rand((S,) + a.shape, device=a.device).clamp(EPS, 1 - EPS)
        # h = (1 - (1-u)^{1/b})^{1/a}, computed in log space
        inner = -torch.expm1(torch.log1p(-u) / b)
        return torch.exp(torch.log(inner.clamp_min(1e-12)) / a).clamp(EPS, 1 - EPS)

    def mean(self, p):
        # b * B(1 + 1/a, b), in float64: lgamma(b) - lgamma(b + c) cancels badly in float32 for large b
        a, b = p["a"].double(), p["b"].double()
        m = torch.exp(torch.log(b) + torch.lgamma(1 + 1 / a) + torch.lgamma(b) - torch.lgamma(1 + 1 / a + b))
        return m.to(p["a"].dtype)

    def kl(self, p):
        # Exact KL(Kum(a,b) || Beta(1,1)); see paper eq. (kl_exact).
        a, b = p["a"], p["b"]
        kl = torch.log(a * b) + (a - 1) / a * (-EULER_GAMMA - digamma(b) - 1 / b) - (b - 1) / b
        return kl.sum(-1)

    def init_bias(self, d):
        return torch.zeros(2 * d)  # a = b = 1: uniform


class BetaLatent(Family):
    name = "beta"

    def params(self, raw):
        r1, r2 = _split(raw)
        return {"a": _shape(r1), "b": _shape(r2)}

    def rsample(self, p, S):
        return Beta(p["a"], p["b"]).rsample((S,)).clamp(EPS, 1 - EPS)

    def mean(self, p):
        return p["a"] / (p["a"] + p["b"])

    def kl(self, p):
        prior = Beta(torch.ones_like(p["a"]), torch.ones_like(p["b"]))
        return kl_divergence(Beta(p["a"], p["b"]), prior).sum(-1)

    def init_bias(self, d):
        return torch.zeros(2 * d)  # a = b = 1: uniform


class LogitNormal(Family):
    name = "logit_normal"
    # Gauss-Hermite nodes for E[sigmoid(m + s z)]: deterministic and accurate,
    # so Mondrian grouping does not depend on Monte Carlo noise.
    _gh_x, _gh_w = (torch.tensor(v, dtype=torch.float32) for v in hermgauss(64))

    def params(self, raw):
        r1, r2 = _split(raw)
        return {"m": r1, "s": F.softplus(r2) + 1e-3}

    def _z(self, p, S):
        eps = torch.randn((S,) + p["m"].shape, device=p["m"].device)
        return p["m"] + p["s"] * eps

    def rsample(self, p, S):
        return torch.sigmoid(self._z(p, S)).clamp(EPS, 1 - EPS)

    @torch.no_grad()
    def mean(self, p):
        x = self._gh_x.to(p["m"].device).view(-1, *([1] * p["m"].dim()))
        w = self._gh_w.to(p["m"].device).view(-1, *([1] * p["m"].dim()))
        vals = torch.sigmoid(p["m"] + math.sqrt(2) * p["s"] * x)
        return (w * vals).sum(0) / math.sqrt(math.pi)

    def kl(self, p, S=16):
        # KL to Uniform = E_q[log q(h)] = -H(N(m,s)) - E[log h + log(1-h)]
        z = self._z(p, S)
        neg_h_normal = -(0.5 * math.log(2 * math.pi * math.e) + torch.log(p["s"]))
        log_jac = F.logsigmoid(z) + F.logsigmoid(-z)
        return (neg_h_normal - log_jac.mean(0)).sum(-1)

    def init_bias(self, d):
        return torch.cat([torch.zeros(d), torch.full((d,), -1.0)])


FAMILIES = {f.name: f for f in (Gaussian, TruncatedGaussian, Kumaraswamy, BetaLatent, LogitNormal)}


def get_family(name):
    return FAMILIES[name]()
