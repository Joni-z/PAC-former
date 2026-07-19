"""Masked-reconstruction pretraining for the tri-axial backbone (AGENT.md sec. 14).

Thesis: cross-frequency structure (phase->amplitude coupling) does not survive as
a competing *layer* under supervised training (pac_scale->0, sec. 9.17). It has to
be forced by the *objective*. This module makes the objective a masked
reconstruction of per-token log band-amplitude with two masking modes:

  * "random"    -- standard MAE: mask a random fraction of grid tokens. Safety net;
                   this is the proven paradigm (LaBraM/CBraMod/REVE all mask).
  * "crossfreq" -- OURS: mask every HIGH-band token and reconstruct its amplitude
                   from the visible LOW bands. The only signal that solves this is
                   low-phase -> high-amplitude coupling, so the model is forced to
                   learn PAC. Uses freq_mixer="attention" so no coupling matrix is
                   fed (the true coupling is computed from the masked bands and
                   would leak the target); the model must route low->high itself.

Target = frontend log mean amplitude per (electrode, band, patch): deterministic,
so no collapse and no target-encoder needed. The encoder never sees a masked
token's own embedding (replaced by a learned mask token + positional encodings),
only its neighbours.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .frontend.triaxial import TriAxialFrontend
from .triaxial import TriAxialEncoder, BandPE, SpatialPE


class MAEPretrain(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        d = cfg["d_model"]
        self.mask_mode = cfg.get("mask_mode", "random")
        self.mask_ratio = cfg.get("mask_ratio", 0.5)
        self.frontend = TriAxialFrontend(
            n_bands=cfg["n_bands"], hidden_dim=d, sample_rate=cfg["sample_rate"],
            kernel_size=cfg.get("kernel_size", 201), patch_len=cfg.get("patch_len", 200),
        )
        self.band_pe = BandPE(d)
        self.spatial_pe = SpatialPE(cfg["n_channels"], d)
        self.encoder = TriAxialEncoder(
            depth=cfg["depth"], d_model=d,
            freq_mixer=cfg.get("freq_mixer", "attention"),
            n_heads=cfg.get("n_heads", 4), dropout=cfg.get("dropout", 0.1),
        )
        self.mask_token = nn.Parameter(torch.zeros(d))
        self.recon = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def _mask(self, B, C, nb, P, device):
        """Return a boolean (B, C, nb, P) mask, True = hidden/reconstruct."""
        if self.mask_mode == "crossfreq":
            m = torch.zeros(B, C, nb, P, dtype=torch.bool, device=device)
            m[:, :, nb // 2:, :] = True                 # hide the high-frequency half
            return m
        # random: independent Bernoulli per token
        return torch.rand(B, C, nb, P, device=device) < self.mask_ratio

    def encode(self, x):
        """Frontend + PEs + encoder with NO masking -- for probing/finetuning."""
        tokens, coupling, band_hz = self.frontend(x)
        B, C, nb, P, D = tokens.shape
        tokens = tokens + self.band_pe(band_hz).view(1, 1, nb, 1, D)
        tokens = tokens + self.spatial_pe(C, tokens.device).view(1, C, 1, 1, D)
        return self.encoder(tokens, coupling)          # (B, C, nb, P, D)

    def forward(self, x):
        tokens, coupling, band_hz, amp_target = self.frontend(x, return_amp_target=True)
        B, C, nb, P, D = tokens.shape
        mask = self._mask(B, C, nb, P, x.device)                        # (B,C,nb,P)

        # replace hidden tokens with the learned mask token, THEN add positional
        # encodings so the encoder still knows where the hidden tokens live.
        tok = torch.where(mask.unsqueeze(-1), self.mask_token.view(1, 1, 1, 1, D), tokens)
        tok = tok + self.band_pe(band_hz).view(1, 1, nb, 1, D)
        tok = tok + self.spatial_pe(C, x.device).view(1, C, 1, 1, D)

        # crossfreq: don't hand the encoder the true coupling (built from the very
        # high bands we're hiding) -- that would leak the answer.
        cpl = torch.zeros_like(coupling) if self.mask_mode == "crossfreq" else coupling
        h = self.encoder(tok, cpl)                                      # (B,C,nb,P,D)

        pred = self.recon(h).squeeze(-1)                               # (B,C,nb,P)
        loss = F.mse_loss(pred[mask], amp_target.detach()[mask])
        return loss
