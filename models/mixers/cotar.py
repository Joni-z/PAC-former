"""Baseline mixer: CoTAR (Core Token Aggregation-Redistribution).

Ported from TeCh (``layers/Transformer_EncDec.py``, ICLR 2026). This is the
symmetric aggregate -> redistribute operator that our MI operator generalises
into a *directional* one; keeping the port faithful keeps that comparison fair.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import TokenMixer


class CoTAR(TokenMixer):
    def __init__(self, d_model: int, d_core: int | None = None, **_):
        super().__init__()
        d_core = d_core or d_model // 4
        self.lin1 = nn.Linear(d_model, d_model)
        self.lin2 = nn.Linear(d_model, d_core)
        self.lin3 = nn.Linear(d_model + d_core, d_model)
        self.lin4 = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        B, N, D = x.shape

        # aggregate: tokens -> a single global core vector
        core = self.lin2(F.gelu(self.lin1(x)))
        weight = F.softmax(core, dim=1)
        core = torch.sum(core * weight, dim=1, keepdim=True).repeat(1, N, 1)

        # redistribute: concat core back onto each token, project
        core_cat = torch.cat([x, core], dim=-1)
        return self.lin4(F.gelu(self.lin3(core_cat)))
