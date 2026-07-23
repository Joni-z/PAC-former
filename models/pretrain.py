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
from .build import _spatial_coords


class MAEPretrain(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        d = cfg["d_model"]
        self.mask_mode = cfg.get("mask_mode", "random")
        self.mask_ratio = cfg.get("mask_ratio", 0.5)
        # crossfreq shape knobs (AGENT.md sec. 13.16). Defaults reproduce the
        # original all-of-the-top-half mask exactly, so existing configs are
        # unaffected.
        #   crossfreq_frac    -- fraction of the band axis, counted from the top,
        #                        that forms the "high" region (0.5 = top half).
        #   crossfreq_density -- probability a token inside that region is actually
        #                        hidden (1.0 = hide all of it, the original).
        #   mixed_p           -- for mask_mode="mixed", per-batch probability of
        #                        drawing the crossfreq mask instead of random.
        self.crossfreq_frac = cfg.get("crossfreq_frac", 0.5)
        self.crossfreq_density = cfg.get("crossfreq_density", 1.0)
        self.mixed_p = cfg.get("mixed_p", 0.5)
        self.pretrain_task = cfg.get("pretrain_task", "mae")
        self.freq_mixer = cfg.get("freq_mixer", "attention")
        self.needs_pac_vector = (
            self.freq_mixer == "phase" or self.pretrain_task == "phase_align"
        )
        self.frontend = TriAxialFrontend(
            n_bands=cfg["n_bands"], hidden_dim=d, sample_rate=cfg["sample_rate"],
            kernel_size=cfg.get("kernel_size", 201), patch_len=cfg.get("patch_len", 200),
            return_pac_vector=self.needs_pac_vector,
        )
        self.band_pe = BandPE(d)
        self.spatial_pe = SpatialPE(cfg["n_channels"], d, coords=_spatial_coords(cfg))
        self.encoder = TriAxialEncoder(
            depth=cfg["depth"], d_model=d,
            freq_mixer=self.freq_mixer,
            n_heads=cfg.get("n_heads", 4), dropout=cfg.get("dropout", 0.1),
        )
        self.mask_token = nn.Parameter(torch.zeros(d))
        self.recon = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.align_head = nn.Linear(d, 1)

    def _mask(self, B, C, nb, P, device):
        """Return a boolean (B, C, nb, P) mask, True = hidden/reconstruct."""
        mode = self.mask_mode
        if mode == "mixed":
            # Per-batch coin flip between the two objectives: keep crossfreq's
            # low->high forcing while still getting standard MAE's broad-coverage
            # signal, which is what the multi-class tasks appear to need (sec. 13.10b).
            mode = "crossfreq" if torch.rand(1).item() < self.mixed_p else "random"
        if mode == "crossfreq":
            m = torch.zeros(B, C, nb, P, dtype=torch.bool, device=device)
            n_high = max(1, int(round(nb * self.crossfreq_frac)))
            m[:, :, nb - n_high:, :] = True             # hide the high-frequency region
            if self.crossfreq_density < 1.0:
                # Reveal some of the high region so the pretext is less destructive
                # while low->high reconstruction is still the only route for what
                # stays hidden.
                reveal = torch.rand(B, C, nb, P, device=device) >= self.crossfreq_density
                m = m & ~reveal
            return m
        # random: independent Bernoulli per token
        return torch.rand(B, C, nb, P, device=device) < self.mask_ratio

    def encode(self, x):
        """Frontend + PEs + encoder with NO masking -- for probing/finetuning."""
        frontend_out = self.frontend(x)
        if self.needs_pac_vector:
            tokens, coupling, band_hz, pac_vector = frontend_out
        else:
            tokens, coupling, band_hz = frontend_out
            pac_vector = None
        B, C, nb, P, D = tokens.shape
        tokens = tokens + self.band_pe(band_hz).view(1, 1, nb, 1, D)
        tokens = tokens + self.spatial_pe(C, tokens.device).view(1, C, 1, 1, D)
        return self.encoder(tokens, coupling, pac_vector)  # (B, C, nb, P, D)

    def forward(self, x):
        if self.pretrain_task == "phase_align":
            return self._phase_alignment_loss(x)

        frontend_out = self.frontend(x, return_amp_target=True)
        if self.needs_pac_vector:
            tokens, coupling, band_hz, amp_target, pac_vector = frontend_out
        else:
            tokens, coupling, band_hz, amp_target = frontend_out
            pac_vector = None
        B, C, nb, P, D = tokens.shape
        mask = self._mask(B, C, nb, P, x.device)                        # (B,C,nb,P)

        # replace hidden tokens with the learned mask token, THEN add positional
        # encodings so the encoder still knows where the hidden tokens live.
        tok = torch.where(mask.unsqueeze(-1), self.mask_token.view(1, 1, 1, 1, D), tokens)
        tok = tok + self.band_pe(band_hz).view(1, 1, nb, 1, D)
        tok = tok + self.spatial_pe(C, x.device).view(1, C, 1, 1, D)

        # Leakage control (applies to any freq_mixer that USES coupling, i.e.
        # "coupling"; attention/cotar ignore it). coupling[.., i, j] = mean_t(
        # phase_i * amp_j) within a patch, so an entry touching a hidden band leaks
        # that band's own amplitude/phase -- exactly the reconstruction target. Keep
        # coupling ONLY between band-tokens that are BOTH visible at each (channel,
        # patch); zero every entry whose driving band i or driven band j is masked.
        # For crossfreq this leaves the low->low block (the operator must still LEARN
        # low->high routing through its Q/K/V -- the coupling prior can't hand it the
        # answer); for random it leaves the visible-visible pairs. Same policy in both
        # objective columns so the 2x2 doesn't confound objective with leakage policy.
        vis = (~mask).permute(0, 1, 3, 2)                              # (B,C,P,nb) True=visible
        keep = (vis.unsqueeze(-1) & vis.unsqueeze(-2)).to(coupling.dtype)  # (B,C,P,nb,nb)
        cpl = coupling * keep
        pac = None if pac_vector is None else pac_vector * keep
        h = self.encoder(tok, cpl, pac)                                 # (B,C,nb,P,D)

        pred = self.recon(h).squeeze(-1)                               # (B,C,nb,P)
        loss = F.mse_loss(pred[mask], amp_target.detach()[mask])
        return loss

    def _phase_alignment_loss(self, x):
        """Discriminate measured PAC geometry from magnitude-matched phase scrambles.

        Positive and negative examples share *identical tokens and coupling
        magnitudes*.  The negative changes only the complex preferred phase of
        every PAC edge, so power, amplitude, and ordinary spectral shortcuts are
        unavailable.  With ``freq_mixer=phase`` the encoder must learn whether
        the phase-steered cross-band messages are consistent with the EEG token
        content.  This directly trains the mechanism that mean-amplitude MAE only
        encouraged indirectly.
        """
        tokens, coupling, band_hz, pac_vector = self.frontend(x)
        B, C, nb, P, D = tokens.shape
        tok = tokens + self.band_pe(band_hz).view(1, 1, nb, 1, D)
        tok = tok + self.spatial_pe(C, x.device).view(1, C, 1, 1, D)

        # Keep the *entire real preferred-phase distribution* and every local
        # |Z|, but break their correspondence to the token grid by permuting
        # phase angles across (electrode, patch) locations within each sample.
        # Detach both geometries so the frontend cannot manufacture an easy
        # positive/negative separation by moving its filter cutoffs.
        pac_reference = pac_vector.detach()
        mag = pac_reference.abs()
        unit = pac_reference / mag.clamp_min(1e-8)
        flat = unit.reshape(B, C * P, nb, nb)
        order = torch.rand(B, C * P, device=x.device).argsort(dim=1)
        gather = order[:, :, None, None].expand_as(flat)
        shuffled_unit = flat.gather(1, gather).reshape_as(unit)
        pac_negative = mag * shuffled_unit

        h_pos = self.encoder(tok, coupling, pac_reference)
        h_neg = self.encoder(tok, coupling, pac_negative)
        pooled = torch.cat(
            [h_pos.mean(dim=(1, 2, 3)), h_neg.mean(dim=(1, 2, 3))], dim=0
        )
        logits = self.align_head(pooled).squeeze(-1)
        labels = torch.cat(
            [torch.ones(B, device=x.device), torch.zeros(B, device=x.device)]
        )
        return F.binary_cross_entropy_with_logits(logits, labels)
