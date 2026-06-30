"""Frontend: raw EEG -> (band x time-patch) tokens + per-band phase/amplitude.

Redesigned after the frontend-bottleneck diagnostic (a plain conv tokenizer hit
0.88 AUROC on TUAB while the old band frontend capped at 0.79). The diagnostic
pinned the loss on two things the old frontend threw away:

  1. it mean-pooled over the 16 channels  -> spatial info gone (abnormality is
     often localized to a few electrodes);
  2. it collapsed each band's whole time course into ONE token via a
     Linear(seq_len -> hidden) -> temporal structure gone.

This version keeps both, the same way the winning conv tokenizer did, but stays
band-structured so the MI operator still has its frequency-band identity:

  * sinc bandpass per channel                       -> (B, C, n_bands, T)
  * per band, a Conv1d(C -> hidden) patch tokenizer mixes channels (learned, not
    averaged) and patchifies time -> (B, n_bands, P, hidden)
  * tokens are flattened to (B, n_bands * P, hidden) for the generic mixer
    interface; the (n_bands, P) layout is recoverable (P = N // n_bands) so the
    MI operator can still build a band x band coupling matrix.

Phase/amplitude (for the MI operator) are still exposed per band at full time
resolution; channel handling there is left as a channel-mean for now (only the
MI mixer consumes it, and that path is revisited when MI is tested).
"""

import torch
import torch.nn as nn

from .sinc import SincBandpass
from .analytic import hilbert, phase_amplitude


class Frontend(nn.Module):
    def __init__(
        self,
        n_bands: int,
        hidden_dim: int,
        seq_len: int,
        sample_rate: int,
        kernel_size: int = 101,
        n_channels: int = 16,
        patch_len: int = 200,
    ):
        super().__init__()
        self.n_bands = n_bands
        self.sinc = SincBandpass(n_bands, sample_rate, kernel_size=kernel_size)
        # per-band patch tokenizer: mixes channels (Conv in_channels=C) and
        # patchifies time (stride=patch_len -> non-overlapping patches). Shared
        # across bands. This is the conv tokenizer that won the diagnostic,
        # applied per frequency band instead of on the raw signal.
        self.tokenizer = nn.Conv1d(n_channels, hidden_dim,
                                   kernel_size=patch_len, stride=patch_len)

    def forward(self, x: torch.Tensor):
        """``x``: (B, C, T) raw EEG -> (tokens, phase_unit, amplitude).

        tokens: ``(B, n_bands * P, hidden)``  (P = number of time patches)
        phase_unit / amplitude: ``(B, n_bands, T)``  (per band, full time)
        """
        B, C, T = x.shape
        filtered = self.sinc(x.reshape(B * C, 1, T))            # (B*C, n_bands, T)
        filtered = filtered.reshape(B, C, self.n_bands, T)

        # (a) tokens: per-band conv tokenizer over (channels, time)
        f = filtered.permute(0, 2, 1, 3).reshape(B * self.n_bands, C, T)
        feat = self.tokenizer(f)                                # (B*n_bands, hidden, P)
        tokens = feat.transpose(1, 2).reshape(B, self.n_bands, feat.shape[-1], -1)
        tokens = tokens.reshape(B, -1, feat.shape[1])           # (B, n_bands*P, hidden)

        # (b) phase/amplitude per band (channel-averaged analytic signal) for MI
        z = hilbert(filtered).mean(dim=1)                       # (B, n_bands, T)
        phase_unit, amplitude = phase_amplitude(z)
        return tokens, phase_unit, amplitude
