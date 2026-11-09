"""MiSCo Gate M0 models: mixed-support latents, atom-aware output heads, and baselines.

Kept separate from scrl/models.py so the frozen SCRL results stay reproducible.

Latent families (per coordinate, mean-field):
  gaussian     N(mu, s^2)                                  (unbounded baseline)
  point        h = mu, no sampling                          (deterministic encoder for model 6)
  mixed_ln     pi0 d0 + pi1 d1 + pic * logit-normal
  mixed_flow   pi0 d0 + pi1 d1 + pic * RQ-spline flow on [0,1]
Mixed families are marginalised exactly over the 3^d atom/interior branches, so no
Gumbel or REINFORCE estimator is needed; their prior is mixed as well
(rho0 = rho1 = 0.1, rho_c = 0.8, uniform interior), which keeps the KL finite:
  KL = KL(Cat(pi) || Cat(rho)) + pi_c * KL(q_c || U).

Output heads: tn (heteroscedastic truncated normal), atom_tn (pi0 d0 + pi1 d1 + pic TN),
quantile (CQR lower/upper), beta (Beta regression, Smithson-Verkuilen squeezed target).
"""
import itertools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.polynomial.hermite import hermgauss

from . import tn

LOG_RHO = torch.log(torch.tensor([0.1, 0.1, 0.8]))
EPS = 1e-6


# ---------------------------------------------------------------------------
# rational-quadratic spline on [0, 1] (Durkan et al., 2019)
# ---------------------------------------------------------------------------
def rq_spline(u, raw, K=8, min_w=1e-3, min_d=1e-3):
    """Monotone RQ spline [0,1] -> [0,1]. raw: [..., 3K+1]. Returns z, log dz/du."""
    w = F.softmax(raw[..., :K], -1) * (1 - K * min_w) + min_w
    h = F.softmax(raw[..., K:2 * K], -1) * (1 - K * min_w) + min_w
    d = F.softplus(raw[..., 2 * K:]) + min_d
    cw = F.pad(torch.cumsum(w, -1), (1, 0))
    ch = F.pad(torch.cumsum(h, -1), (1, 0))
    cw[..., -1] = 1.0
    ch[..., -1] = 1.0
    idx = torch.searchsorted(cw[..., 1:-1].contiguous(), u.unsqueeze(-1).contiguous()).squeeze(-1)
    idx = idx.clamp(0, K - 1)
    g = lambda t: t.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
    xk, wk, yk, hk = g(cw[..., :-1]), g(w), g(ch[..., :-1]), g(h)
    dk = g(d[..., :-1])
    dk1 = d[..., 1:].gather(-1, idx.unsqueeze(-1)).squeeze(-1)
    sk = hk / wk
    th = ((u - xk) / wk).clamp(0, 1)
    tt = th * (1 - th)
    den = sk + (dk1 + dk - 2 * sk) * tt
    z = yk + hk * (sk * th ** 2 + dk * tt) / den
    deriv = sk ** 2 * (dk1 * th ** 2 + 2 * sk * tt + dk * (1 - th) ** 2) / den ** 2
    return z.clamp(EPS, 1 - EPS), torch.log(deriv.clamp_min(1e-12))


# ---------------------------------------------------------------------------
# latent families
# ---------------------------------------------------------------------------
class Latent:
    mixed = False
    n_per_dim = 2

    def __init__(self, d):
        self.d = d


class GaussianLatent(Latent):
    def params(self, raw):
        return {"mu": 0.5 + raw[..., 0], "s": F.softplus(raw[..., 1] - 2.0) + 1e-3}

    def samples(self, p, S):
        h = p["mu"] + p["s"] * torch.randn((S,) + p["mu"].shape, device=p["mu"].device)
        return h, torch.full(h.shape[:2], 1.0 / S, device=h.device)

    def kl(self, p):
        s0 = 1 / math.sqrt(12)
        return (torch.log(s0 / p["s"]) + (p["s"] ** 2 + (p["mu"] - 0.5) ** 2) / (2 * s0 ** 2) - 0.5).sum(-1)

    def mean(self, p):
        return p["mu"]


class PointLatent(Latent):
    def params(self, raw):
        return {"mu": 0.5 + raw[..., 0]}

    def samples(self, p, S):
        h = p["mu"].unsqueeze(0)
        return h, torch.ones(h.shape[:2], device=h.device)

    def kl(self, p):
        return torch.zeros(p["mu"].shape[0], device=p["mu"].device)

    def mean(self, p):
        return p["mu"]


class MixedLatent(Latent):
    """pi0 d0 + pi1 d1 + pic q_c per coordinate; exact marginalisation over atom branches."""
    mixed = True
    _gh_x, _gh_w = (torch.tensor(v, dtype=torch.float32) for v in hermgauss(64))

    def __init__(self, d, interior, K=8):
        super().__init__(d)
        self.interior, self.K = interior, K
        self.n_int = 2 if interior == "logit_normal" else 3 * K + 1
        self.n_per_dim = 3 + self.n_int
        self.combos = torch.tensor(list(itertools.product(range(3), repeat=d)))   # [3^d, d]; 0/1 atom, 2 interior

    def params(self, raw):
        logits = raw[..., :3] + torch.tensor([-2.0, -2.0, 0.0], device=raw.device)
        return {"logpi": F.log_softmax(logits, -1), "int": raw[..., 3:]}

    def _interior_samples(self, p, u):
        if self.interior == "logit_normal":
            m, s = p["int"][..., 0], F.softplus(p["int"][..., 1] - 1.0) + 1e-3
            eps = torch.special.ndtri(u.clamp(1e-6, 1 - 1e-6))
            z = torch.sigmoid(m + s * eps).clamp(EPS, 1 - EPS)
            # log q(z) = log N(eps) - log s - log z(1-z)
            logq = -0.5 * eps ** 2 - 0.5 * math.log(2 * math.pi) - torch.log(s) - torch.log(z * (1 - z))
            return z, logq
        z, logdet = rq_spline(u, p["int"].expand(*u.shape, self.n_int), self.K)
        return z, -logdet                                   # base Uniform(0,1): log q(z) = -log dz/du

    def samples(self, p, S):
        B, d = p["logpi"].shape[:2]
        dev = p["logpi"].device
        u = torch.rand(S, B, d, device=dev)
        zc, _ = self._interior_samples(p, u)                # [S, B, d]
        combos = self.combos.to(dev)                         # [Nb, d]
        Nb = combos.shape[0]
        vals = torch.stack([torch.zeros_like(zc), torch.ones_like(zc), zc], -1)   # [S, B, d, 3]
        idx = combos.view(Nb, 1, 1, d, 1).expand(Nb, S, B, d, 1)
        h = vals.unsqueeze(0).expand(Nb, S, B, d, 3).gather(-1, idx).squeeze(-1)  # [Nb, S, B, d]
        logw = p["logpi"].unsqueeze(0).expand(Nb, B, d, 3).gather(
            -1, combos.view(Nb, 1, d, 1).expand(Nb, B, d, 1)).squeeze(-1).sum(-1)   # [Nb, B]
        w = (logw.exp() / S).unsqueeze(1).expand(Nb, S, B)
        return h.reshape(Nb * S, B, d), w.reshape(Nb * S, B)

    def kl(self, p, S=8):
        pi = p["logpi"].exp()
        kl_cat = (pi * (p["logpi"] - LOG_RHO.to(pi.device))).sum(-1)       # [B, d]
        u = torch.rand((S,) + pi.shape[:2], device=pi.device)
        _, logq = self._interior_samples(p, u)
        kl_int = logq.mean(0)                                               # KL(q_c || U) = E[log q_c]
        return (kl_cat + pi[..., 2] * kl_int).sum(-1)

    @torch.no_grad()
    def mean(self, p):
        pi = p["logpi"].exp()
        if self.interior == "logit_normal":
            m, s = p["int"][..., 0], F.softplus(p["int"][..., 1] - 1.0) + 1e-3
            x = self._gh_x.to(m.device).view(-1, 1, 1)
            w = self._gh_w.to(m.device).view(-1, 1, 1)
            ez = (w * torch.sigmoid(m + math.sqrt(2) * s * x)).sum(0) / math.sqrt(math.pi)
        else:
            u = ((torch.arange(256, device=pi.device) + 0.5) / 256).view(-1, 1, 1).expand(256, *pi.shape[:2])
            z, _ = self._interior_samples(p, u)
            ez = z.mean(0)
        return pi[..., 1] + pi[..., 2] * ez


def make_latent(name, d):
    return {"gaussian": lambda: GaussianLatent(d), "point": lambda: PointLatent(d),
            "mixed_ln": lambda: MixedLatent(d, "logit_normal"),
            "mixed_flow": lambda: MixedLatent(d, "flow")}[name]()


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
HEAD_SIZE = {"tn": 2, "atom_tn": 5, "quantile": 2, "beta": 2}


class MiscoModel(nn.Module):
    def __init__(self, latent, head, d, n_obs, K, obs_noise, alpha=0.1, hidden=64, dec_hidden=64, n_train=10_000):
        super().__init__()
        self.lat = make_latent(latent, d)
        self.head, self.d, self.K, self.obs_noise, self.alpha = head, d, K, obs_noise, alpha
        self.n_train = n_train
        self.gru = nn.GRU(n_obs + K, hidden, batch_first=True)
        self.enc = nn.Linear(hidden, d * self.lat.n_per_dim)
        with torch.no_grad():
            self.enc.weight.mul_(0.1)
            self.enc.bias.zero_()
        self.dec = nn.Sequential(nn.Linear(d, dec_hidden), nn.SiLU(), nn.Linear(dec_hidden, dec_hidden), nn.SiLU(),
                                 nn.Linear(dec_hidden, HEAD_SIZE[head]))

    def encode(self, x, s):
        _, hN = self.gru(torch.cat([x, F.one_hot(s, self.K).float()], -1))
        return self.lat.params(self.enc(hN[-1]).view(x.shape[0], self.d, self.lat.n_per_dim))

    # ---- output head -------------------------------------------------------
    def out(self, h):
        o = self.dec(h)
        if self.head == "tn":
            return {"loc": o[..., 0], "scale": F.softplus(o[..., 1]) + 1e-3}
        if self.head == "atom_tn":
            return {"logpi": F.log_softmax(o[..., :3] + torch.tensor([-2.0, -2.0, 0.0], device=o.device), -1),
                    "loc": o[..., 3], "scale": F.softplus(o[..., 4]) + 1e-3}
        if self.head == "quantile":
            lo = o[..., 0]
            return {"lo": lo, "hi": lo + F.softplus(o[..., 1])}
        return {"a": F.softplus(o[..., 0]) + 1e-2, "b": F.softplus(o[..., 1]) + 1e-2}

    def squeeze_y(self, y):
        n = self.n_train
        return (y * (n - 1) + 0.5) / n

    def logp(self, o, y):
        """log p(y | h) per sample; y broadcast to o's shape. Masses for atoms, densities inside."""
        if self.head == "tn":
            return tn.log_prob(y, o["loc"], o["scale"])
        if self.head == "atom_tn":
            at0, at1 = y <= 0.0, y >= 1.0
            inner = o["logpi"][..., 2] + tn.log_prob(y.clamp(EPS, 1 - EPS), o["loc"], o["scale"])
            return torch.where(at0, o["logpi"][..., 0], torch.where(at1, o["logpi"][..., 1], inner))
        if self.head == "beta":
            ys = self.squeeze_y(y)
            return torch.distributions.Beta(o["a"], o["b"]).log_prob(ys)
        # quantile head: negative pinball loss acts as the "log-likelihood"
        a = self.alpha / 2
        pin = lambda q, t: torch.maximum(t * (y - q), (t - 1) * (y - q))
        return -(pin(o["lo"], a) + pin(o["hi"], 1 - a))

    def loss(self, x, s, y, S=4, beta=1.0):
        p = self.encode(x, s)
        h, w = self.lat.samples(p, S)                                   # [N, B, d], [N, B]
        xT = x[:, -1, : self.d].unsqueeze(0).expand_as(h)
        rec = (w * tn.log_prob(xT, h, torch.full_like(h, self.obs_noise)).sum(-1)).sum(0)
        o = self.out(h)
        yl = (w * self.logp(o, y.squeeze(-1).unsqueeze(0).expand(h.shape[:2]))).sum(0)
        kl = self.lat.kl(p)
        return (-(rec + yl) + beta * kl).mean(), {"rec": rec.mean().item(), "ylp": yl.mean().item(),
                                                  "kl": kl.mean().item()}

    # ---- predictive distribution ------------------------------------------
    @torch.no_grad()
    def predictive(self, x, s, S=16):
        p = self.encode(x, s)
        h, w = self.lat.samples(p, S)
        o = self.out(h)
        pred = {"w": w, "o": o, "post_mean": self.lat.mean(p)}
        if self.lat.mixed:
            pred["latent_pi"] = p["logpi"].exp()                         # [B, d, 3]
        return pred
