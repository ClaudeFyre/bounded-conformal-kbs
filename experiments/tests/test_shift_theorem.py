"""Numerical checks of Theorem 2 (post-hoc center shift under unimodal, possibly skewed residuals).

For a residual eps with unimodal density and a constant normalized center movement b, the half-width needed for
coverage 1 - alpha is h(b) = inf{t : F(b + t) - F(b - t) >= 1 - alpha}. Claims checked:
  (1) h is quasiconvex in b (nonincreasing, then nondecreasing);
  (2) its minimum is at b*, the center of the shortest interval of probability 1 - alpha, with h(b*) = half its length;
  (3) if b and b* lie on opposite sides of 0 (or b = 0), then h(b) >= h(0): moving away from b* cannot narrow;
  (4) for 0 < b <= b* (same side, no overshoot), h(b) <= h(0).
Run: python tests/test_shift_theorem.py
"""
import numpy as np
from scipy import optimize, stats

ALPHA = 0.1


def centered(dist):
    m = dist.mean()
    return lambda x: dist.cdf(np.asarray(x) + m), lambda p: dist.ppf(p) - m


DISTS = {
    "exp-1": centered(stats.expon()),
    "gamma2": centered(stats.gamma(2.0)),
    "skewnorm5": centered(stats.skewnorm(5.0)),
    "lognorm0.5": centered(stats.lognorm(0.5)),
    "triang0.2": centered(stats.triang(0.2)),
    "normal": centered(stats.norm()),
}


def h(cdf, b, hi=60.0):
    g = lambda t: cdf(b + t) - cdf(b - t) - (1 - ALPHA)
    return optimize.brentq(g, 0.0, hi, xtol=1e-12)


def shortest(ppf):
    res = optimize.minimize_scalar(lambda p: ppf(p + 1 - ALPHA) - ppf(p), bounds=(1e-12, ALPHA - 1e-12),
                                   method="bounded", options={"xatol": 1e-12})
    lo, up = ppf(res.x), ppf(res.x + 1 - ALPHA)
    return (lo + up) / 2, (up - lo) / 2


def check(name, cdf, ppf):
    b_star, half = shortest(ppf)
    grid = np.linspace(b_star - 3, b_star + 3, 1201)
    hv = np.array([h(cdf, b) for b in grid])
    k = int(np.argmin(hv))
    tol = 1e-7
    assert np.all(np.diff(hv[: k + 1]) <= tol), f"{name}: not nonincreasing left of the minimum"
    assert np.all(np.diff(hv[k:]) >= -tol), f"{name}: not nondecreasing right of the minimum"
    assert abs(grid[k] - b_star) <= 6 / 1200 + 1e-6, f"{name}: argmin {grid[k]:.4f} != b* {b_star:.4f}"
    assert abs(h(cdf, b_star) - half) < 1e-6, f"{name}: h(b*) != half-length of shortest interval"
    h0 = h(cdf, 0.0)
    away = grid[(grid * b_star <= 0)]
    assert all(h(cdf, b) >= h0 - tol for b in away), f"{name}: a shift away from b* narrowed"
    toward = grid[(grid * b_star > 0) & (np.abs(grid) <= abs(b_star))]
    assert all(h(cdf, b) <= h0 + tol for b in toward), f"{name}: a shift toward b* widened"
    return b_star, h0, half


def check_bound(name, sampler, eta, n=400_000, seed=0):
    """Part (i): for unit-varying shifts b(x) and a common scale, the empirical width ratio is >= eta."""
    rng = np.random.default_rng(seed)
    eps = sampler(rng, n)
    q0 = np.quantile(np.abs(eps), 1 - ALPHA)
    for spread in (0.0, 0.2, 1.0):
        for loc in (-0.5, 0.0, 0.15, 0.5):
            b = loc + spread * rng.standard_normal(n)
            ratio = np.quantile(np.abs(eps - b), 1 - ALPHA) / q0
            assert ratio >= eta - 2e-3, f"{name}: ratio {ratio:.4f} below bound {eta:.4f}"


if __name__ == "__main__":
    etas = {}
    for name, (cdf, ppf) in DISTS.items():
        b_star, h0, half = check(name, cdf, ppf)
        etas[name] = half / h0
        print(f"{name:11s} b* = {b_star:+.4f}  h(0) = {h0:.4f}  h(b*) = {half:.4f}  eta = {half / h0:.4f}")
    check_bound("exp-1", lambda r, n: r.exponential(1.0, n) - 1.0, etas["exp-1"])
    check_bound("gamma2", lambda r, n: r.gamma(2.0, 1.0, n) - 2.0, etas["gamma2"])
    check_bound("normal", lambda r, n: r.standard_normal(n), etas["normal"])
    print("all checks passed")
