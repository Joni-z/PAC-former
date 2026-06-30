"""Training-time data augmentation, ported from TeCh (``layers/Augmentation.py``).

TeCh leans on augmentation heavily -- its scripts always pass a list like
``flip0.2,frequency0.2,jitter0.,mask0.,channel0.,drop0.4`` and apply ONE randomly
chosen augmentation per forward pass (training only). We replicate that so the
CoTAR baseline runs under the conditions it was actually validated in, and the
same augmentation is applied to all three mixers (the ablation only swaps the
mixer, never the training regime).

Augmentations act on the raw EEG ``(B, C, T)`` before the frontend.
"""

import random

import torch
import torch.nn as nn


class Jitter(nn.Module):
    def __init__(self, scale=0.0):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        if self.training and self.scale > 0:
            x = x + torch.randn_like(x) * self.scale
        return x


class Flip(nn.Module):
    """Left-right (time-reversal) flip with probability ``prob``."""

    def __init__(self, prob=0.0):
        super().__init__()
        self.prob = prob

    def forward(self, x):
        if self.training and torch.rand(1).item() < self.prob:
            return torch.flip(x, [-1])
        return x


class TemporalMask(nn.Module):
    def __init__(self, ratio=0.0):
        super().__init__()
        self.ratio = ratio

    def forward(self, x):
        if self.training and self.ratio > 0:
            T = x.shape[-1]
            idx = torch.randperm(T)[: int(T * self.ratio)]
            x = x.clone()
            x[:, :, idx] = 0
        return x


class ChannelMask(nn.Module):
    def __init__(self, ratio=0.0):
        super().__init__()
        self.ratio = ratio

    def forward(self, x):
        if self.training and self.ratio > 0:
            C = x.shape[1]
            idx = torch.randperm(C)[: int(C * self.ratio)]
            x = x.clone()
            x[:, idx, :] = 0
        return x


class FrequencyMask(nn.Module):
    def __init__(self, ratio=0.0):
        super().__init__()
        self.ratio = ratio

    def forward(self, x):
        if self.training and self.ratio > 0:
            T = x.shape[-1]
            xf = torch.fft.rfft(x, dim=-1)
            xf = xf * (torch.rand(xf.shape, device=x.device) > self.ratio)
            x = torch.fft.irfft(xf, n=T, dim=-1)
        return x


def _make(spec: str) -> nn.Module:
    """Parse a spec like ``"jitter0.1"`` / ``"drop0.4"`` / ``"none"``."""
    table = {"jitter": Jitter, "flip": Flip, "mask": TemporalMask,
             "channel": ChannelMask, "frequency": FrequencyMask}
    if spec == "none":
        return nn.Identity()
    if spec.startswith("drop"):
        return nn.Dropout(float(spec[4:] or 0.1))
    for name, cls in table.items():
        if spec.startswith(name):
            return cls(float(spec[len(name):] or 0.0))
    raise ValueError(f"unknown augmentation '{spec}'")


class RandomAugment(nn.Module):
    """Hold several augmentations; apply one at random per forward (train only)."""

    def __init__(self, specs):
        super().__init__()
        self.augs = nn.ModuleList([_make(s) for s in specs])

    def forward(self, x):
        if not self.training or len(self.augs) == 0:
            return x
        return self.augs[random.randint(0, len(self.augs) - 1)](x)
