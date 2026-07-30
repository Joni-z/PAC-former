"""Masked-reconstruction pretraining for the tri-axial backbone (AGENT.md sec. 14).

Thesis: cross-frequency structure (phase->amplitude coupling) does not survive as
a competing *layer* under supervised training (pac_scale->0, sec. 9.17). It has to
be forced by the *objective*. This module makes the objective a masked
reconstruction of per-token log band-amplitude with two masking modes:

  * "random"    -- standard MAE: mask a random fraction of grid tokens. Safety net;
                   this is the proven paradigm (LaBraM/CBraMod/REVE all mask).
  * "crossfreq" -- OURS: mask every HIGH-band token and reconstruct its amplitude
                   from visible LOW bands plus spatio-temporal context.  This is a
                   PAC-inspired asymmetric mask distribution: it encourages the
                   representation to retain low-to-high dependencies, but does not
                   by itself prove that the network learned biological PAC.

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
from .montage import coords_for


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
        self.structured_mask_mode = cfg.get(
            "structured_mask_mode", "crossfreq"
        )
        if self.structured_mask_mode not in ("crossfreq", "lowfreq", "bandrand"):
            raise ValueError(
                "structured_mask_mode must be crossfreq/lowfreq/bandrand"
            )
        self.pretrain_task = cfg.get("pretrain_task", "mae")
        self.freq_mixer = cfg.get("freq_mixer", "attention")
        self.needs_pac_vector = (
            self.freq_mixer == "phase" or self.pretrain_task == "phase_align"
        )
        self.frontend = TriAxialFrontend(
            n_bands=cfg["n_bands"], hidden_dim=d, sample_rate=cfg["sample_rate"],
            kernel_size=cfg.get("kernel_size", 201), patch_len=cfg.get("patch_len", 200),
            return_pac_vector=self.needs_pac_vector,
            tokenizer_mode=cfg.get("tokenizer_mode", "raw"),
            pac_token_mode=cfg.get("pac_token_mode", "measured"),
            interaction_mode=cfg.get("interaction_mode", "product"),
        )
        self.band_pe = BandPE(d, n_bands=cfg["n_bands"], mode=cfg.get("band_pe", "hz"))
        self.spatial_pe = SpatialPE(cfg["n_channels"], d, coords=_spatial_coords(cfg))
        self.encoder = TriAxialEncoder(
            depth=cfg["depth"], d_model=d,
            freq_mixer=self.freq_mixer,
            n_heads=cfg.get("n_heads", 4), dropout=cfg.get("dropout", 0.1),
            mi_k=cfg.get("mi_k", 3),   # see build.py: inert for every other mixer
        )
        self.mask_token = nn.Parameter(torch.zeros(d))
        self.recon = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.align_head = nn.Linear(d, 1)
        self.recon_loss = cfg.get("recon_loss", "mse")

        # Pooled batches carry a member id.  Keep the corresponding coordinate
        # tables so xyz SpatialPE can be selected at runtime.  The current four
        # corpora share a montage, but this removes that assumption from the
        # model/pretrainer boundary and is required when the big-cluster corpus
        # grows to additional channel layouts.
        self.pool_coords = []
        for spec in cfg.get("pretrain_pool", []):
            coords = coords_for(spec["name"])
            if cfg.get("spatial_pe") == "xyz" and coords is None:
                raise ValueError(
                    f"no xyz montage coordinates registered for pooled dataset "
                    f"{spec['name']!r}"
                )
            self.pool_coords.append(coords)

    def _spatial_encoding(self, C, device, dataset_idx=None):
        coords = None
        if self.pool_coords and dataset_idx is not None:
            if torch.is_tensor(dataset_idx):
                unique = torch.unique(dataset_idx.detach().cpu())
                if unique.numel() != 1:
                    raise ValueError(
                        "pooled pretrain batch mixes datasets; enable the "
                        "dataset-homogeneous mixture batch sampler"
                    )
                dataset_idx = int(unique.item())
            coords = self.pool_coords[int(dataset_idx)]
        return self.spatial_pe(C, device, coords=coords)

    def _mask(self, B, C, nb, P, device):
        """Return a boolean (B, C, nb, P) mask, True = hidden/reconstruct."""
        mode = self.mask_mode
        if mode == "mixed":
            # Per-batch coin flip between the two objectives: keep crossfreq's
            # low->high forcing while still getting standard MAE's broad-coverage
            # signal, which is what the multi-class tasks appear to need (sec. 13.10b).
            mode = (
                self.structured_mask_mode
                if torch.rand(1).item() < self.mixed_p
                else "random"
            )
        n_hide = max(1, int(round(nb * self.crossfreq_frac)))
        # Mechanism controls (AGENT.md 13.40-D): all hide the SAME NUMBER of
        # whole bands as crossfreq, differing only in WHICH bands, to isolate whether
        # the benefit is about the high bands specifically ("PAC / low->high routing")
        # or merely about hiding entire bands.
        if mode in ("crossfreq", "lowfreq", "bandrand"):
            m = torch.zeros(B, C, nb, P, dtype=torch.bool, device=device)
            if mode == "crossfreq":
                idx = torch.arange(nb - n_hide, nb, device=device)   # top n_hide bands
            elif mode == "lowfreq":
                idx = torch.arange(0, n_hide, device=device)         # bottom n_hide bands
            else:  # bandrand: n_hide random whole bands (shared across B,C,P this batch)
                idx = torch.randperm(nb, device=device)[:n_hide]
            m[:, :, idx, :] = True
            if mode == "crossfreq" and self.crossfreq_density < 1.0:
                reveal = torch.rand(B, C, nb, P, device=device) >= self.crossfreq_density
                m = m & ~reveal
            return m
        # random: independent Bernoulli per token
        return torch.rand(B, C, nb, P, device=device) < self.mask_ratio

    def encode(self, x, dataset_idx=None):
        """Frontend + PEs + encoder with NO masking -- for probing/finetuning."""
        frontend_out = self.frontend(x)
        if self.needs_pac_vector:
            tokens, coupling, band_hz, pac_vector = frontend_out
        else:
            tokens, coupling, band_hz = frontend_out
            pac_vector = None
        B, C, nb, P, D = tokens.shape
        tokens = tokens + self.band_pe(band_hz).view(1, 1, nb, 1, D)
        tokens = tokens + self._spatial_encoding(
            C, tokens.device, dataset_idx
        ).view(1, C, 1, 1, D)
        return self.encoder(tokens, coupling, pac_vector)  # (B, C, nb, P, D)

    def forward(self, x, dataset_idx=None):
        if self.pretrain_task == "phase_align":
            return self._phase_alignment_loss(x, dataset_idx)

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
        tok = tok + self._spatial_encoding(
            C, x.device, dataset_idx
        ).view(1, C, 1, 1, D)

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
        return self._reconstruction_loss(pred, amp_target.detach(), mask)

    def _reconstruction_loss(self, pred, target, mask):
        """Masked amplitude loss, optionally balanced over frequency bands.

        Standard elementwise MSE is retained as the default for exact backward
        compatibility.  The foundation recipe uses band-balanced Smooth-L1:
        every reconstructed band contributes one mean loss regardless of its
        absolute envelope scale or how many tokens happened to be masked.  This
        directly addresses the low-frequency energy dominance highlighted by
        BandVQ/TFM without discarding absolute log-amplitude information.
        """
        if self.recon_loss == "mse":
            return F.mse_loss(pred[mask], target[mask])
        if self.recon_loss not in ("band_balanced_mse", "band_balanced_smooth_l1"):
            raise ValueError(f"unknown recon_loss={self.recon_loss!r}")
        if self.recon_loss == "band_balanced_mse":
            error = F.mse_loss(pred, target, reduction="none")
        else:
            error = F.smooth_l1_loss(pred, target, reduction="none")
        band_losses = []
        for band in range(pred.shape[2]):
            selected = mask[:, :, band, :]
            if selected.any():
                band_losses.append(error[:, :, band, :][selected].mean())
        if not band_losses:
            raise RuntimeError("masked reconstruction produced an empty mask")
        return torch.stack(band_losses).mean()

    def _phase_alignment_loss(self, x, dataset_idx=None):
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
        tok = tok + self._spatial_encoding(
            C, x.device, dataset_idx
        ).view(1, C, 1, 1, D)

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
