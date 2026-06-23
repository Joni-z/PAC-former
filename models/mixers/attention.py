"""Baseline mixer: vanilla multi-head self-attention.

Ported almost verbatim from TeCh (``layers/Transformer_EncDec.py``), wrapped in
the TokenMixer interface so it can be swapped one-for-one with CoTAR / MI.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import TokenMixer


class SelfAttention(TokenMixer):
    def __init__(self, d_model: int, n_heads: int = 4, **_):
        super().__init__()
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.scale = (d_model // n_heads) ** -0.5

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        b, n, dim = x.shape
        h, hd = self.n_heads, dim // self.n_heads

        qkv = self.qkv(x).reshape(b, n, 3, h, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(b, n, dim)
        return self.out_proj(out)
