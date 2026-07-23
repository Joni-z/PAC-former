"""config -> assembled PAC-Former model.

Two architectures, selected by ``cfg['arch']``:

  * "flat" (default, v1): Frontend (band tokens, channels collapsed) -> Encoder
    (single swappable token mixer) -> Head. The original mixer-swap ablation.
  * "triaxial" (v2, AGENT.md sec. 13): TriAxialFrontend (electrode x band x
    time-patch GRID, channels kept) + physics positional encodings ->
    TriAxialEncoder (time/space/freq axis mixers, only the freq mixer is
    swapped: cfg['freq_mixer']) -> Head. The foundation-model backbone.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .frontend import Frontend
from .frontend.conv import ConvFrontend
from .frontend.triaxial import TriAxialFrontend
from .encoder import Encoder
from .triaxial import TriAxialEncoder, BandPE, SpatialPE
from .head import ClassificationHead
from .augment import RandomAugment
from .montage import coords_for


def _spatial_coords(cfg: dict):
    """xyz electrode coords for SpatialPE when cfg opts in, else None (learned
    index embedding). `spatial_pe: xyz` uses the dataset's montage coordinates
    (models/montage.py); anything else keeps the original index embedding so
    existing configs are unchanged (AGENT.md sec. 13.23 A)."""
    if cfg.get("spatial_pe", "index") != "xyz":
        return None
    coords = coords_for(cfg.get("dataset"))
    if coords is not None and coords.shape[0] != cfg["n_channels"]:
        raise ValueError(
            f"spatial_pe=xyz: montage for {cfg.get('dataset')!r} has "
            f"{coords.shape[0]} channels but cfg n_channels={cfg['n_channels']}"
        )
    return coords


class PACFormer(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        d = cfg["d_model"]
        self.augment = RandomAugment(cfg.get("augmentations", []))
        if cfg.get("frontend", "sinc") == "conv":
            self.frontend = ConvFrontend(
                n_channels=cfg["n_channels"], hidden_dim=d,
                patch_len=cfg.get("patch_len", 100),
            )
        else:
            self.frontend = Frontend(
                n_bands=cfg["n_bands"], hidden_dim=d, seq_len=cfg["seq_len"],
                sample_rate=cfg["sample_rate"], kernel_size=cfg.get("kernel_size", 101),
                n_channels=cfg["n_channels"], patch_len=cfg.get("patch_len", 200),
            )
        self.encoder = Encoder(
            depth=cfg["depth"], d_model=d, mixer=cfg["mixer"],
            dropout=cfg.get("dropout", 0.1), **cfg.get("mixer_kwargs", {}),
        )
        self.head = ClassificationHead(d, cfg["num_classes"])

    def forward(self, x: torch.Tensor, phase_mode: str = "normal") -> torch.Tensor:
        x = self.augment(x)
        token, phase_unit, amplitude = self.frontend(x)
        h = self.encoder(token, phase_unit=phase_unit, amplitude=amplitude)
        return self.head(h)


class TriAxialPACFormer(nn.Module):
    """v2 foundation-model backbone (AGENT.md sec. 13)."""

    def __init__(self, cfg: dict):
        super().__init__()
        d = cfg["d_model"]
        self.freq_mixer = cfg.get("freq_mixer", "coupling")
        self.augment = RandomAugment(cfg.get("augmentations", []))
        self.frontend = TriAxialFrontend(
            n_bands=cfg["n_bands"], hidden_dim=d, sample_rate=cfg["sample_rate"],
            kernel_size=cfg.get("kernel_size", 201), patch_len=cfg.get("patch_len", 200),
            return_pac_vector=self.freq_mixer == "phase",
        )
        self.band_pe = BandPE(d)
        self.spatial_pe = SpatialPE(cfg["n_channels"], d, coords=_spatial_coords(cfg))
        self.encoder = TriAxialEncoder(
            depth=cfg["depth"], d_model=d,
            freq_mixer=self.freq_mixer,
            n_heads=cfg.get("n_heads", 4), dropout=cfg.get("dropout", 0.1),
        )
        self.head = ClassificationHead(d, cfg["num_classes"])

        # Optional crossfreq-reconstruction auxiliary head (AGENT.md sec. 13.15).
        # When aux_recon_weight > 0, supervised training adds a masked-amplitude
        # reconstruction loss (mask the high-band half, rebuild from visible low
        # bands) alongside the classification loss. Purpose: give the objective an
        # incentive to keep low->high coupling info that plain supervised CE has no
        # reason to preserve (sec. 9.17 Finding 1), and watch whether it stops
        # pac_scale from collapsing. Zero/absent weight => model behaves exactly as
        # before (no new params allocated, no new forward path taken).
        self.aux_recon_weight = cfg.get("aux_recon_weight", 0.0)
        if self.aux_recon_weight > 0:
            self.aux_mask_mode = cfg.get("aux_mask_mode", "crossfreq")
            self.aux_mask_ratio = cfg.get("aux_mask_ratio", 0.5)
            self.mask_token = nn.Parameter(torch.zeros(d))
            self.recon = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def crossfreq_aux_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Masked band-amplitude reconstruction as a supervised auxiliary.

        Mirrors models/pretrain.py's MAE crossfreq path (mask token + PEs +
        visible-visible coupling leakage control), but shares the classifier's
        frontend/encoder so the auxiliary gradient shapes the same representation
        the classifier uses. No augmentation here -- the reconstruction target
        must stay clean. Only called during training when aux_recon_weight > 0.
        """
        frontend_out = self.frontend(x, return_amp_target=True)
        if self.freq_mixer == "phase":
            tokens, coupling, band_hz, amp_target, pac_vector = frontend_out
        else:
            tokens, coupling, band_hz, amp_target = frontend_out
            pac_vector = None
        B, C, nb, P, D = tokens.shape

        if self.aux_mask_mode == "crossfreq":
            mask = torch.zeros(B, C, nb, P, dtype=torch.bool, device=x.device)
            mask[:, :, nb // 2:, :] = True                  # hide high-frequency half
        else:                                               # "random"
            mask = torch.rand(B, C, nb, P, device=x.device) < self.aux_mask_ratio

        tok = torch.where(mask.unsqueeze(-1), self.mask_token.view(1, 1, 1, 1, D), tokens)
        tok = tok + self.band_pe(band_hz).view(1, 1, nb, 1, D)
        tok = tok + self.spatial_pe(C, x.device).view(1, C, 1, 1, D)

        # leakage control: keep coupling only between bands BOTH visible at each
        # (channel, patch); zero every entry touching a masked band (else the
        # reconstruction target leaks through the coupling matrix).
        vis = (~mask).permute(0, 1, 3, 2)                   # (B,C,P,nb) True=visible
        keep = (vis.unsqueeze(-1) & vis.unsqueeze(-2)).to(coupling.dtype)
        cpl = coupling * keep
        pac = None if pac_vector is None else pac_vector * keep
        h = self.encoder(tok, cpl, pac)

        pred = self.recon(h).squeeze(-1)                    # (B,C,nb,P)
        return F.mse_loss(pred[mask], amp_target.detach()[mask])

    def forward(self, x: torch.Tensor, phase_mode: str = "normal") -> torch.Tensor:
        x = self.augment(x)
        frontend_out = self.frontend(x)
        if self.freq_mixer == "phase":
            tokens, coupling, band_hz, pac_vector = frontend_out
            if phase_mode == "magnitude":
                # Preserve every PAC edge magnitude but remove preferred phase.
                pac_vector = torch.complex(pac_vector.abs(), torch.zeros_like(pac_vector.real))
            elif phase_mode == "scramble":
                # Preserve magnitude exactly while independently randomising the
                # measured preferred phase. This is the decisive mechanism test.
                theta = 2.0 * torch.pi * torch.rand_like(pac_vector.real)
                pac_vector = pac_vector * torch.complex(theta.cos(), theta.sin())
            elif phase_mode != "normal":
                raise ValueError(f"unknown phase_mode={phase_mode!r}")
        else:
            tokens, coupling, band_hz = frontend_out
            pac_vector = None
        B, C, nb, P, D = tokens.shape
        # physics positional encodings: band by center-freq, electrode by position
        tokens = tokens + self.band_pe(band_hz).view(1, 1, nb, 1, D)
        tokens = tokens + self.spatial_pe(C, tokens.device).view(1, C, 1, 1, D)
        h = self.encoder(tokens, coupling, pac_vector)   # (B,C,nb,P,D)
        return self.head(h.reshape(B, C * nb * P, D))    # head mean-pools dim=1


def build_model(cfg: dict) -> nn.Module:
    if cfg.get("arch", "flat") == "triaxial":
        return TriAxialPACFormer(cfg)
    return PACFormer(cfg)
