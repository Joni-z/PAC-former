"""OURS: a learnable SincNet-style bandpass filter bank.

Each band is a band-pass FIR filter parameterised by exactly two learned scalars
(low cutoff, band width), following Ravanelli & Bengio's SincNet. We adapt the
reference (``mravanelli/SincNet``, ``SincConv_fast``) to EEG: linear (not mel)
band initialisation, EEG sample rates, and a band count we choose.

The classic SincNet t=0 NaN is avoided exactly as in the reference -- the filter
is built from a half window plus an analytic centre tap ``2*band`` rather than
evaluating sinc at zero.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SincBandpass(nn.Module):
    def __init__(
        self,
        n_bands: int,
        sample_rate: int,
        kernel_size: int = 101,
        min_low_hz: float = 1.0,
        min_band_hz: float = 1.0,
        f_min: float = 2.0,
        f_max: float | None = None,
        spacing: str = "log",
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1  # force odd so the filter has a centre tap
        self.n_bands = n_bands
        self.sample_rate = sample_rate
        self.kernel_size = kernel_size
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz
        f_max = f_max if f_max is not None else sample_rate / 2 - (min_low_hz + min_band_hz)

        # Log (constant-Q) spacing by default: low bands are narrow (good phase
        # estimates) and high bands are wide. The width matters for PAC -- a band
        # carrying a modulated amplitude needs to span the f_amp +/- f_phase
        # sidebands, so the amplitude side must be wide; constant-Q gives that
        # for free at high frequency. Linear spacing is available for ablation.
        if spacing == "log":
            edges = torch.logspace(torch.log10(torch.tensor(f_min)),
                                   torch.log10(torch.tensor(f_max)), n_bands + 1)
        else:
            edges = torch.linspace(f_min, f_max, n_bands + 1)
        self.low_hz_ = nn.Parameter(edges[:-1].view(-1, 1))
        self.band_hz_ = nn.Parameter(edges.diff().view(-1, 1))

        # half Hamming window (the other half is mirrored), and the half time axis
        n_lin = torch.linspace(0, kernel_size // 2 - 1, steps=kernel_size // 2)
        self.register_buffer("window_", 0.54 - 0.46 * torch.cos(2 * math.pi * n_lin / kernel_size))
        n = (kernel_size - 1) / 2.0
        self.register_buffer("n_", 2 * math.pi * torch.arange(-n, 0).view(1, -1) / sample_rate)

    def filters(self) -> torch.Tensor:
        """Build the (n_bands, 1, kernel_size) filter bank from the 2 params/band."""
        low = self.min_low_hz + self.low_hz_.abs()
        high = torch.clamp(
            low + self.min_band_hz + self.band_hz_.abs(), self.min_low_hz, self.sample_rate / 2
        )
        band = (high - low)[:, 0]

        f_low = low @ self.n_
        f_high = high @ self.n_
        left = ((torch.sin(f_high) - torch.sin(f_low)) / (self.n_ / 2)) * self.window_
        center = 2 * band.view(-1, 1)
        right = torch.flip(left, dims=[1])
        band_pass = torch.cat([left, center, right], dim=1) / (2 * band[:, None])
        return band_pass.view(self.n_bands, 1, self.kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: (B, 1, T) single channel -> filtered (B, n_bands, T)."""
        return F.conv1d(x, self.filters(), padding=self.kernel_size // 2)
