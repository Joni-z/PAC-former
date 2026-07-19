"""v2 tri-axial frontend: raw EEG -> (electrode x band x time-patch) token GRID.

Unlike the v1 frontend (models/frontend/__init__.py), this one does NOT collapse
the channel axis -- electrodes stay an explicit token dimension so the encoder
can model space and so variable montages are possible (AGENT.md sec. 13.3).

Outputs, for x = (B, C, T):
  * tokens   : (B, C, n_bands, P, d_model)   -- the 3D token grid
  * coupling : (B, C, P, n_bands, n_bands)   -- time-resolved, per-channel MVL
               coupling (AGENT.md sec. 13.6 / 9.17 Finding 2: computed WITHIN
               each patch and per channel, never averaged over time/channels)
  * band_hz  : (n_bands, 2)  center-freq + bandwidth per band, for the band PE

The analytic-signal math (unit complex phase vector, mean-centred amplitude
debiasing, no atan2) is identical to v1 and still validated by
scripts/synth_pac_test.py; only the reduction axes change.
"""

import torch
import torch.nn as nn

from .sinc import SincBandpass
from .analytic import hilbert, phase_amplitude

# Fixed divisor for the MVL normalisation (same rationale as v1's 4D path:
# dividing by a per-patch amplitude std blows up to NaN on flat/dead channels).
NORM_CONST = 100.0


def patch_coupling(phase_unit, amplitude, P, normalize=True):
    """Time-resolved directional MVL coupling, per channel, per patch.

    phase_unit, amplitude: (B, C, n_bands, T), T divisible by P.
    Returns (B, C, P, n_bands, n_bands) with [.., i, j] = band i (phase)
    driving band j (amplitude), within that patch.
    """
    B, C, nb, T = phase_unit.shape
    L = T // P
    ph = phase_unit[..., : P * L].reshape(B, C, nb, P, L)
    am = amplitude[..., : P * L].reshape(B, C, nb, P, L)
    am = am - am.mean(dim=-1, keepdim=True)                      # dPAC debiasing
    # Z[b,c,p,i,j] = mean_t phase_i * amp_j   (within patch p)
    Z = torch.einsum("bcipl,bcjpl->bcpij", ph, am.to(ph.dtype)) / L
    coupling = Z.abs()
    if normalize:
        coupling = coupling / NORM_CONST
    return coupling


class TriAxialFrontend(nn.Module):
    def __init__(
        self,
        n_bands: int,
        hidden_dim: int,
        sample_rate: int,
        kernel_size: int = 201,
        patch_len: int = 200,
        normalize: bool = True,
        **_,
    ):
        super().__init__()
        self.n_bands = n_bands
        self.patch_len = patch_len
        self.normalize = normalize
        self.sinc = SincBandpass(n_bands, sample_rate, kernel_size=kernel_size)
        # per-(channel, band) conv patch tokenizer: one input channel (the
        # filtered signal for that electrode+band), patchify time. Shared across
        # all (channel, band) pairs -- channels are NOT mixed here.
        self.tokenizer = nn.Conv1d(1, hidden_dim, kernel_size=patch_len, stride=patch_len)

    def band_hz(self) -> torch.Tensor:
        """(n_bands, 2): [center_freq, bandwidth] in Hz, from the sinc params."""
        low = self.sinc.min_low_hz + self.sinc.low_hz_.abs()
        high = low + self.sinc.min_band_hz + self.sinc.band_hz_.abs()
        center = (low + high) / 2
        width = high - low
        return torch.cat([center, width], dim=1)                # (n_bands, 2)

    def forward(self, x: torch.Tensor, return_amp_target: bool = False):
        B, C, T = x.shape
        filtered = self.sinc(x.reshape(B * C, 1, T)).reshape(B, C, self.n_bands, T)

        # tokens: patchify each (channel, band) signal independently
        f = filtered.reshape(B * C * self.n_bands, 1, T)
        feat = self.tokenizer(f)                                 # (B*C*nb, D, P)
        P = feat.shape[-1]
        tokens = feat.transpose(1, 2).reshape(B, C, self.n_bands, P, -1)

        # phase / amplitude -> time-resolved per-channel coupling
        z = hilbert(filtered)                                    # (B, C, nb, T)
        phase_unit, amplitude = phase_amplitude(z)
        coupling = patch_coupling(phase_unit, amplitude, P, self.normalize)

        if return_amp_target:
            # Per-token (electrode, band, patch) log mean amplitude -- a fixed,
            # deterministic regression target for masked-reconstruction pretraining
            # (models/pretrain.py). Deterministic => no representation collapse, no
            # target encoder needed. Predicting a HIGH band's amplitude from a
            # masked grid whose only visible cue is the LOW bands' phase is exactly
            # the phase->amplitude coupling the objective is meant to force.
            L = T // P
            am = amplitude[..., : P * L].reshape(B, C, self.n_bands, P, L)
            amp_target = torch.log(am.mean(dim=-1) + 1e-6)      # (B, C, nb, P)
            return tokens, coupling, self.band_hz(), amp_target

        return tokens, coupling, self.band_hz()
