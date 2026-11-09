"""Checks for the Gate O1 constructions: gated HPD and CPCI reach nominal coverage."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scrl import o1_eval as oe  # noqa: E402
from scrl import tn  # noqa: E402
from scrl.misco_eval import conformal_quantile  # noqa: E402


def _truth(B, seed):
    g = torch.Generator().manual_seed(seed)
    p0 = 0.4 * torch.rand(B, generator=g)
    p1 = 0.2 * torch.rand(B, generator=g)
    loc = 0.3 + 0.4 * torch.rand(B, generator=g)
    scale = torch.full((B,), 0.1)
    u = torch.rand(B, generator=g)
    yc = tn.rsample(loc, scale, torch.rand(B, generator=g)).clamp(1e-4, 1 - 1e-4)
    y = torch.where(u < p0, torch.zeros(B), torch.where(u < p0 + p1, torch.ones(B), yc))
    logpi = torch.log(torch.stack([p0, p1, 1 - p0 - p1], -1)).unsqueeze(0)
    pred = {"w": torch.ones(1, B), "o": {"loc": loc.unsqueeze(0), "scale": scale.unsqueeze(0), "logpi": logpi}}
    return pred, y


def test_gated_hpd_coverage_and_gate():
    pc, yc = _truth(6000, 1)
    pt, yt = _truth(6000, 2)
    for tau in (None, 0.05, 0.3):
        sc, _ = oe.hpd(pc, "atom_tn", yc, tau, G=500)
        st, aux = oe.hpd(pt, "atom_tn", yt, tau, G=500)
        q = conformal_quantile(sc, 0.1)
        if tau == 0.3:
            # over-aggressive gate: > 10% of calibration targets sit on gated atoms (score +inf),
            # so no finite threshold reaches 90% coverage and the set degenerates to everything
            assert q == float("inf")
            continue
        assert abs((st <= q).float().mean().item() - 0.9) < 0.02
        _, i0, i1 = oe.hpd_measure(aux, torch.full((6000,), q))
        if tau is not None:
            a0 = pt["o"]["logpi"][0, :, 0].exp()
            assert not i0[a0 < tau].any()          # gated atoms never enter the set


def test_e5_coverage_shapes_and_conflicts():
    parts = {}
    for name, (n, seed) in {"val": (3000, 6), "cal": (4000, 7), "test": (6000, 8)}.items():
        pred, y = _truth(n, seed)
        p0, p1, f = oe.e5_parts(pred)
        parts[name] = {"p0": p0, "p1": p1, "f": f, "y": y}
    c1 = {k: v[:2000] for k, v in parts["cal"].items()}
    c2 = {k: v[2000:] for k, v in parts["cal"].items()}
    fit, report = oe.e5_fit(parts["val"], c1, c2)
    assert len(report) == len(oe.E5_PAIRS) and all(r["r0"] + r["r1"] <= 0.5 + 1e-9 for r in report)
    mu, i0, i1, cov = oe.e5_sets(parts["test"], fit)
    assert not (i0 & i1).any()                                # never {0, 1}
    assert ((mu > 0) & (i0 | i1)).sum() == 0                  # singletons carry no interval
    assert abs(cov.float().mean().item() - 0.9) < 0.03        # approximate (beta estimated on D_val)


def test_e5_route_tie_goes_to_continuous():
    p = torch.tensor([0.5])
    assert oe.e5_route(p, p, 0.0, 0.0).item() == 0            # equal margins -> continuous branch
    assert oe.e5_route(torch.tensor([0.9]), torch.tensor([0.5]), 0.0, 0.0).item() == 1


def test_cpci_coverage():
    pv, yv = _truth(3000, 3)
    pc, yc = _truth(4000, 4)
    pt, yt = _truth(6000, 5)
    p_v, f_v = oe.cpci_parts(pv)
    p_c, f_c = oe.cpci_parts(pc)
    p_t, f_t = oe.cpci_parts(pt)
    a_r, q, r, gamma, beta = oe.cpci_fit(p_v, f_v, yv, p_c[:2000], p_c[2000:], f_c[2000:], yc[2000:])
    _, _, zero, cov = oe.cpci_sets(p_t, f_t, yt, a_r, q)
    assert abs(cov.float().mean().item() - 0.9) < 0.03       # approximate: beta is estimated on D_val
