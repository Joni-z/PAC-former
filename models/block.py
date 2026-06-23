"""Pre-mixer-norm transformer block: the mixer is injected, nothing else changes.

Structure follows TeCh's EncoderLayer (norm -> mixer -> residual; norm -> FFN ->
residual). The mixer is whichever ``TokenMixer`` the config chose; ``**kwargs``
(phase/amplitude) are forwarded to it untouched so the MI operator gets what it
needs and the other two ignore them.
"""

import torch
import torch.nn as nn

from .mixers import build_mixer


class Block(nn.Module):
    def __init__(self, d_model: int, mixer: str, dropout: float = 0.1, **mixer_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.mixer = build_mixer(mixer, d_model, **mixer_kwargs)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.norm1(x + self.mixer(x, **kwargs))
        x = self.norm2(x + self.ffn(x))
        return x
