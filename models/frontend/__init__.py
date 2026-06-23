"""Frontend: raw EEG -> band tokens + per-band phase/amplitude.

This wrapper enforces the frontend contract (AGENT.md sec. 4). It exposes BOTH
of the things downstream needs, so the MI operator never has to reach back into
internals:

  (a) ``x``          : pooled per-band token, ``(B, n_bands, hidden_dim)``
                       -- the generic input every TokenMixer consumes.
  (b) ``phase_unit`` : unit complex phase per band, ``(B, n_bands, T)``
      ``amplitude``  : amplitude per band,        ``(B, n_bands, T)``
                       -- passed as explicit kwargs to the MI operator.

Multi-channel handling: the sinc bank runs per channel; band tokens are mean-
pooled over channels, and the analytic signal is averaged over channels before
the phase/amplitude split. PAC is conventionally computed per channel then
aggregated, so a channel mean is a defensible first-version choice.
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
    ):
        super().__init__()
        self.n_bands = n_bands
        self.sinc = SincBandpass(n_bands, sample_rate, kernel_size=kernel_size)
        # per-band token: project the band's time course to hidden_dim
        self.token_proj = nn.Linear(seq_len, hidden_dim)

    def forward(self, x: torch.Tensor):
        """``x``: (B, C, T) raw EEG -> (token, phase_unit, amplitude)."""
        B, C, T = x.shape
        filtered = self.sinc(x.reshape(B * C, 1, T))        # (B*C, n_bands, T)
        filtered = filtered.reshape(B, C, self.n_bands, T)

        # (a) band tokens: project time -> hidden, mean over channels
        token = self.token_proj(filtered).mean(dim=1)       # (B, n_bands, hidden)

        # (b) phase/amplitude from the channel-averaged analytic signal
        z = hilbert(filtered).mean(dim=1)                   # (B, n_bands, T) complex
        phase_unit, amplitude = phase_amplitude(z)
        return token, phase_unit, amplitude
