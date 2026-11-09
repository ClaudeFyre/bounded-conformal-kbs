"""Shared backbone: GRU encoder -> latent family -> target decoder.

Only two things vary across models: the latent family, and (in the
decoder ablations) the target decoder. The observation model is the known
bounded measurement x_T ~ TN(h_T, obs_noise^2) on [0,1], which anchors the
latent coordinates to physical coordinates so that boundary distance in
latent space is meaningful.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import tn
from .latents import get_family

class TCN(nn.Module):
    """Causal dilated 1-D convolution stack (robustness backbone; PLAN.md, review response A).
    4 residual blocks, 64 channels, kernel 3, dilations 1, 2, 4, 8; returns the features of the last time step."""

    def __init__(self, n_in, hidden=64, kernel=3, dilations=(1, 2, 4, 8)):
        super().__init__()
        self.kernel, self.dilations = kernel, dilations
        self.convs = nn.ModuleList()
        self.skips = nn.ModuleList()
        c = n_in
        for d in dilations:
            self.convs.append(nn.Conv1d(c, hidden, kernel, dilation=d))
            self.skips.append(nn.Conv1d(c, hidden, 1) if c != hidden else nn.Identity())
            c = hidden

    def forward(self, x):                     # x: [B, T, C]
        h = x.transpose(1, 2)
        for conv, skip, d in zip(self.convs, self.skips, self.dilations):
            out = conv(F.pad(h, ((self.kernel - 1) * d, 0)))   # left padding only: causal
            h = F.relu(out) + skip(h)
        return h[:, :, -1]


DECODERS = ("gauss_homo", "tn_hetero")
DEC_INPUTS = ("raw", "projected")  # projected = clamp to [0,1]: flat outside M (L = 0)


class Model(nn.Module):
    def __init__(self, family, d, n_obs, K, p_out=1, decoder="tn_hetero", dec_input="raw",
                 hidden=64, dec_hidden=64, obs_noise=0.05, backbone="gru"):
        super().__init__()
        assert decoder in DECODERS and dec_input in DEC_INPUTS and backbone in ("gru", "tcn")
        self.backbone = backbone
        self.family = get_family(family)
        self.d, self.K, self.p_out = d, K, p_out
        self.decoder, self.dec_input = decoder, dec_input
        self.obs_noise = obs_noise
        if backbone == "gru":
            self.gru = nn.GRU(n_obs + K, hidden, batch_first=True)
        else:
            self.tcn = TCN(n_obs + K, hidden)
        self.head = nn.Linear(hidden, 2 * d)
        with torch.no_grad():
            self.head.weight.mul_(0.1)
            self.head.bias.copy_(self.family.init_bias(d))
        n_head = p_out if decoder == "gauss_homo" else 2 * p_out
        self.dec = nn.Sequential(
            nn.Linear(d, dec_hidden), nn.SiLU(),
            nn.Linear(dec_hidden, dec_hidden), nn.SiLU(),
            nn.Linear(dec_hidden, n_head),
        )
        self.log_sigma_y = nn.Parameter(torch.tensor(math.log(0.1)))

    # ---- encoder -------------------------------------------------------
    def encode(self, x, s):
        inp = torch.cat([x, F.one_hot(s, self.K).float()], dim=-1)
        if self.backbone == "tcn":
            return self.family.params(self.head(self.tcn(inp)))
        _, h_n = self.gru(inp)
        return self.family.params(self.head(h_n[-1]))

    # ---- decoder -------------------------------------------------------
    def _dec_raw(self, h):
        if self.dec_input == "projected":
            h = h.clamp(0.0, 1.0)
        return self.dec(h)

    def dec_mean_fn(self, h):
        """Deterministic decoder mean mu_dec(h); used by the extrapolation probe."""
        out = self._dec_raw(h)
        if self.decoder == "gauss_homo":
            return out
        loc, scale = self._tn_params(out)
        return tn.mean(loc, scale)

    def _tn_params(self, out):
        loc = out[..., : self.p_out]
        scale = F.softplus(out[..., self.p_out:]) + 1e-3
        return loc, scale

    def dec_stats(self, h):
        """Per-sample predictive mean and variance of y given h."""
        out = self._dec_raw(h)
        if self.decoder == "gauss_homo":
            var = torch.exp(2 * self.log_sigma_y).expand_as(out)
            return out, var
        loc, scale = self._tn_params(out)
        return tn.mean(loc, scale), tn.variance(loc, scale)

    def dec_logp(self, h, y):
        out = self._dec_raw(h)
        if self.decoder == "gauss_homo":
            sig = torch.exp(self.log_sigma_y)
            return (-0.5 * ((y - out) / sig) ** 2 - torch.log(sig) - 0.5 * math.log(2 * math.pi)).sum(-1)
        loc, scale = self._tn_params(out)
        return tn.log_prob(y, loc, scale).sum(-1)

    # ---- objective -----------------------------------------------------
    def loss(self, x, s, y, S=4, beta=1.0):
        p = self.encode(x, s)
        hs = self.family.rsample(p, S)                                   # [S,B,d]
        x_T = x[:, -1, : self.d].unsqueeze(0).expand_as(hs)   # first d columns observe the state
        obs_scale = torch.full_like(hs, self.obs_noise)
        rec = tn.log_prob(x_T, hs, obs_scale).sum(-1).mean(0)            # [B]
        ylp = self.dec_logp(hs, y.unsqueeze(0).expand(S, *y.shape)).mean(0)
        kl = self.family.kl(p)
        loss = -(rec + ylp) + beta * kl
        return loss.mean(), {"rec": rec.mean().item(), "ylp": ylp.mean().item(), "kl": kl.mean().item()}

    # ---- prediction ----------------------------------------------------
    @torch.no_grad()
    def predict(self, x, s, S=32):
        p = self.encode(x, s)
        hs = self.family.rsample(p, S)
        return self.predict_from_samples(hs) | {
            "post_mean": self.family.mean(p),
            "leakage": self.family.leakage(p),
            "params": {k: v for k, v in p.items()},
        }

    @torch.no_grad()
    def predict_from_samples(self, hs):
        """Law-of-total-variance prediction from latent samples hs [S,B,d]."""
        m, v = self.dec_stats(hs)                                        # [S,B,p]
        y_hat = m.mean(0)
        within = v.mean(0).mean(-1)
        between = ((m - y_hat) ** 2).sum(-1).mean(0) / self.p_out
        return {"y_hat": y_hat, "sigma": torch.sqrt(within + between)}
