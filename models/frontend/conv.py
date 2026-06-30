"""Diagnostic frontend: a plain conv patch tokenizer on the raw signal.

This exists only for the frontend-bottleneck diagnostic (does our sinc+analytic
band-token frontend cap performance, or is it the dataset's from-scratch
ceiling?). It does NO frequency decomposition -- it just patchifies the raw
(B, C, T) signal into tokens, the standard strong tokenizer used by BIOT-style
EEG transformers. Same output contract as Frontend so it drops into the same
encoder, but it has no phase/amplitude to expose (returns None for both), so it
can only be paired with the attention / CoTAR mixers, not the MI operator.
"""

import torch
import torch.nn as nn


class ConvFrontend(nn.Module):
    def __init__(self, n_channels: int, hidden_dim: int, patch_len: int = 100):
        super().__init__()
        # Conv over time, mixing all channels into hidden_dim; stride=patch_len
        # gives non-overlapping patch tokens (like a 1D ViT tokenizer).
        self.tokenizer = nn.Conv1d(n_channels, hidden_dim,
                                   kernel_size=patch_len, stride=patch_len)

    def forward(self, x: torch.Tensor):
        """``x``: (B, C, T) raw EEG -> (token (B, n_patches, hidden), None, None)."""
        tokens = self.tokenizer(x).transpose(1, 2)  # (B, n_patches, hidden)
        return tokens, None, None
