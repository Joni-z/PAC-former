"""A stack of identical blocks. Phase/amplitude kwargs flow through every block."""

import torch
import torch.nn as nn

from .block import Block


class Encoder(nn.Module):
    def __init__(self, depth: int, d_model: int, mixer: str, dropout: float = 0.1, **mixer_kwargs):
        super().__init__()
        self.blocks = nn.ModuleList(
            [Block(d_model, mixer, dropout=dropout, **mixer_kwargs) for _ in range(depth)]
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, **kwargs)
        return x
