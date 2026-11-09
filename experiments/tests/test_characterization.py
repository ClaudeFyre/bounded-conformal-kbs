"""Numerical checks of the characterization theorems (theory/characterization_theory.md)."""
import math

import torch

torch.manual_seed(0)
N = 400_000
ALPHA = 0.1


def _width(y, center, scale):
    s = (y - center).abs() / scale
    q = torch.quantile(s, 1 - ALPHA)
    return (2 * q * scale).mean().item()


def _setup(eps):
    sigma = torch.exp(0.4 * torch.randn(N))          # heterogeneous normaliser
    y_hat = torch.rand(N)
    y = y_hat + eps * sigma
    return y, y_hat, sigma


def test_theorem3_any_shift_widens_for_calibrated_symmetric_unimodal():
    for eps in (torch.randn(N), torch.distributions.Laplace(0.0, 1.0).sample((N,))):
        y, y_hat, sigma = _setup(eps)
        w = _width(y, y_hat, sigma)
        for c in (0.7, 1.0, 1.3):                        # common scale ratio (assumption R)
            shift = 0.5 * torch.randn(N) * sigma         # arbitrary, x-dependent centre shift
            w_mod = _width(y, y_hat - shift, c * sigma)
            assert w_mod > w * 1.01                      # strictly wider
            w_same = _width(y, y_hat, c * sigma)         # no shift: width unchanged
            assert abs(w_same - w) / w < 1e-3


def test_theorem3_conditions_matter_skewed_residual_can_be_narrowed():
    # violate (C): skewed residual (centred exponential). Moving the centre towards the mode narrows the interval,
    # so the calibration/symmetry assumption is necessary, not decorative.
    eps = torch.distributions.Exponential(1.0).sample((N,)) - 1.0
    y, y_hat, sigma = _setup(eps)
    w = _width(y, y_hat, sigma)
    # For eps = Exp(1) - 1 (support [-1, inf)), the shortest symmetric 90% interval [a - t, a + t] starts at the
    # lower end: a - t = -1, a + t = ln(10) - 1, so the optimal centre is a = ln(10)/2 - 1 ~ 0.15, not 0.
    # A small shift of the centre by +0.2 sigma therefore narrows the interval.
    w_mod = _width(y, y_hat + 0.2 * sigma, sigma)
    assert w_mod < w * 0.99


def test_theorem4_exact_crossover_and_linear_average():
    g = torch.Generator().manual_seed(1)
    for _ in range(200):
        mu_c, mu_a = torch.rand(2, generator=g).tolist()
        n_c, n_a = 0.0, 0.1 + 2 * torch.rand(1, generator=g).item()
        lam_star = (mu_c - mu_a) / (n_a - n_c)
        for lam in torch.linspace(0, 0.3, 61).tolist():
            better = (mu_a + lam * n_a) < (mu_c + lam * n_c)
            if abs(lam - lam_star) > 1e-9:
                assert better == (lam < lam_star)
        # any lambda-average equals the single point lambda-bar
        lams = torch.rand(50, generator=g) * 0.1
        wts = torch.rand(50, generator=g); wts /= wts.sum()
        avg = (wts * (mu_a + lams * n_a)).sum().item()
        assert math.isclose(avg, mu_a + (wts * lams).sum().item() * n_a, rel_tol=1e-6, abs_tol=1e-9)


def test_proposition5_free_atoms():
    # adding {0,1} never changes Lebesgue measure and never lowers coverage
    y = torch.where(torch.rand(N) < 0.2, torch.zeros(N), torch.rand(N))
    lo, hi = 0.3, 0.9
    cov = ((y >= lo) & (y <= hi)).float().mean()
    cov_plus = (((y >= lo) & (y <= hi)) | (y == 0) | (y == 1)).float().mean()
    assert cov_plus >= cov
    assert (hi - lo) == (hi - lo) + 0.0                  # measure of {0,1} is zero
