"""Numerically stable truncated-normal utilities on [lo, hi].

Used by the synthetic generator (bounded observations and targets), the
truncated-normal latent family, the truncated-normal decoder, and the
support-restricted reference q* in the diagnostics.
"""
import math

import torch
from torch.special import log_ndtr, ndtr, ndtri

LOG_SQRT_2PI = 0.5 * math.log(2 * math.pi)


def log_phi(z):
    return -0.5 * z * z - LOG_SQRT_2PI


def log_diff_ndtr(a, b):
    """log(Phi(b) - Phi(a)) for a < b, stable in both tails."""
    # Right tail (a > 0): use Phi(b) - Phi(a) = Phi(-a) - Phi(-b).
    right = a > 0
    hi = torch.where(right, -a, b)
    lo = torch.where(right, -b, a)
    lhi, llo = log_ndtr(hi), log_ndtr(lo)
    return lhi + torch.log1p(-torch.exp(llo - lhi).clamp(max=1 - 1e-12))


def standardize(loc, scale, lo=0.0, hi=1.0):
    alpha = (lo - loc) / scale
    beta = (hi - loc) / scale
    return alpha, beta, log_diff_ndtr(alpha, beta)


def log_prob(x, loc, scale, lo=0.0, hi=1.0):
    _, _, log_z = standardize(loc, scale, lo, hi)
    return log_phi((x - loc) / scale) - torch.log(scale) - log_z


def mean(loc, scale, lo=0.0, hi=1.0):
    alpha, beta, log_z = standardize(loc, scale, lo, hi)
    ratio = torch.exp(log_phi(alpha) - log_z) - torch.exp(log_phi(beta) - log_z)
    return (loc + scale * ratio).clamp(lo, hi)


def variance(loc, scale, lo=0.0, hi=1.0):
    alpha, beta, log_z = standardize(loc, scale, lo, hi)
    pa = torch.exp(log_phi(alpha) - log_z)
    pb = torch.exp(log_phi(beta) - log_z)
    # alpha * phi(alpha) with alpha = -inf is 0; guard the product.
    term = torch.nan_to_num(alpha * pa) - torch.nan_to_num(beta * pb)
    var = scale**2 * (1 + term - (pa - pb) ** 2)
    return var.clamp(min=1e-10, max=(hi - lo) ** 2 / 4)


def entropy(loc, scale, lo=0.0, hi=1.0):
    alpha, beta, log_z = standardize(loc, scale, lo, hi)
    pa = torch.exp(log_phi(alpha) - log_z)
    pb = torch.exp(log_phi(beta) - log_z)
    term = torch.nan_to_num(alpha * pa) - torch.nan_to_num(beta * pb)
    return LOG_SQRT_2PI + 0.5 + torch.log(scale) + log_z + 0.5 * term


def rsample(loc, scale, u, lo=0.0, hi=1.0):
    """Inverse-CDF reparameterised sample; u ~ Uniform(0,1) with loc's shape."""
    alpha = (lo - loc) / scale
    beta = (hi - loc) / scale
    pa, pb = ndtr(alpha), ndtr(beta)
    p = (pa + u * (pb - pa)).clamp(1e-7, 1 - 1e-7)
    x = loc + scale * ndtri(p)
    return x.clamp(lo, hi)
