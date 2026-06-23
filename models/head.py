"""Mean-pool over band tokens, then a linear classifier."""

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    def __init__(self, d_model: int, num_classes: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(x.mean(dim=1)))
