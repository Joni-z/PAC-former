"""config -> assembled PAC-Former model.

The full model is Frontend -> Encoder -> Head. The frontend produces the band
tokens AND the phase/amplitude; the latter are threaded to the encoder as
kwargs so the MI mixer can use them while the attention/CoTAR mixers ignore
them. Switching mixer is purely ``cfg['mixer']``.
"""

import torch
import torch.nn as nn

from .frontend import Frontend
from .encoder import Encoder
from .head import ClassificationHead


class PACFormer(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        d = cfg["d_model"]
        self.frontend = Frontend(
            n_bands=cfg["n_bands"],
            hidden_dim=d,
            seq_len=cfg["seq_len"],
            sample_rate=cfg["sample_rate"],
            kernel_size=cfg.get("kernel_size", 101),
        )
        self.encoder = Encoder(
            depth=cfg["depth"],
            d_model=d,
            mixer=cfg["mixer"],
            dropout=cfg.get("dropout", 0.1),
            **cfg.get("mixer_kwargs", {}),
        )
        self.head = ClassificationHead(d, cfg["num_classes"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: (B, C, T) raw EEG -> logits (B, num_classes)."""
        token, phase_unit, amplitude = self.frontend(x)
        h = self.encoder(token, phase_unit=phase_unit, amplitude=amplitude)
        return self.head(h)


def build_model(cfg: dict) -> PACFormer:
    return PACFormer(cfg)
