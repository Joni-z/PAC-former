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


def patch_pac_vector(phase_unit, amplitude, P, normalize=True):
    """Complex, time-resolved directional PAC vector per channel and patch.

    phase_unit, amplitude: (B, C, n_bands, T), T divisible by P.
    Returns complex ``Z`` with shape (B, C, P, n_bands, n_bands), where
    ``Z[..., i, j]`` is low-band-i phase driving band-j amplitude.  Keeping Z
    complex preserves the preferred PAC phase; taking ``abs`` too early was
    exactly what prevented the old mixer from defining a phase geometry.
    """
    B, C, nb, T = phase_unit.shape
    L = T // P
    ph = phase_unit[..., : P * L].reshape(B, C, nb, P, L)
    am = amplitude[..., : P * L].reshape(B, C, nb, P, L)
    am = am - am.mean(dim=-1, keepdim=True)                      # dPAC debiasing
    # Z[b,c,p,i,j] = mean_t phase_i * amp_j   (within patch p)
    Z = torch.einsum("bcipl,bcjpl->bcpij", ph, am.to(ph.dtype)) / L
    if normalize:
        Z = Z / NORM_CONST
    return Z


def patch_coupling(phase_unit, amplitude, P, normalize=True):
    """Backward-compatible MVL magnitude used by the older mixers."""
    return patch_pac_vector(phase_unit, amplitude, P, normalize).abs()


class TriAxialFrontend(nn.Module):
    def __init__(
        self,
        n_bands: int,
        hidden_dim: int,
        sample_rate: int,
        kernel_size: int = 201,
        patch_len: int = 200,
        normalize: bool = True,
        return_pac_vector: bool = False,
        tokenizer_mode: str = "raw",
        pac_token_mode: str = "measured",
        interaction_mode: str = "product",
        **_,
    ):
        super().__init__()
        self.n_bands = n_bands
        self.patch_len = patch_len
        self.normalize = normalize
        self.return_pac_vector = return_pac_vector
        if tokenizer_mode not in ("raw", "pac_interaction"):
            raise ValueError(
                f"tokenizer_mode must be raw/pac_interaction, got {tokenizer_mode!r}"
            )
        if pac_token_mode not in ("measured", "uniform", "scramble", "magnitude"):
            raise ValueError(
                "pac_token_mode must be measured/uniform/scramble/magnitude, got "
                f"{pac_token_mode!r}"
            )
        if interaction_mode not in ("product", "concat"):
            raise ValueError(
                f"interaction_mode must be product/concat, got {interaction_mode!r}"
            )
        self.tokenizer_mode = tokenizer_mode
        self.pac_token_mode = pac_token_mode
        self.interaction_mode = interaction_mode
        self.sinc = SincBandpass(n_bands, sample_rate, kernel_size=kernel_size)
        if tokenizer_mode == "raw":
            # Per-(channel, band) raw-waveform patch tokenizer. Shared across
            # all channel/band pairs; retained as the exact legacy baseline.
            self.tokenizer = nn.Conv1d(
                1, hidden_dim, kernel_size=patch_len, stride=patch_len
            )
        else:
            if hidden_dim % 2:
                raise ValueError("pac_interaction tokenizer needs an even hidden_dim")
            complex_dim = hidden_dim // 2
            # A single real linear map is shared by real/imaginary unit phase.
            # With no bias it is exactly phase-equivariant:
            # L(e^{i delta} p) = e^{i delta} L(p).
            self.phase_tokenizer = nn.Conv1d(
                1, complex_dim, kernel_size=patch_len,
                stride=patch_len, bias=False,
            )
            self.amplitude_tokenizer = nn.Conv1d(
                1, complex_dim, kernel_size=patch_len,
                stride=patch_len,
            )
            # Diagonal amplitude calibration keeps the PAC tokenizer exactly
            # parameter-matched to the legacy Conv1d tokenizer (whose output
            # bias has hidden_dim rather than complex_dim entries). It lies on
            # the sole token path and cannot bypass the interaction.
            self.amplitude_scale = nn.Parameter(torch.ones(complex_dim))
            if interaction_mode == "concat":
                # SleepPACNet-style fusion control (AGENT.md 13.43-G, "most
                # important missing baseline"): expose the SAME invariant
                # ingredients (a_j, Re/Im of the aligned phase) but let a
                # learned projection combine them instead of forcing a
                # multiplicative interaction. This is the free path §13.18
                # warns about: the network COULD learn to reduce to
                # amplitude-only and ignore phase; the product arms cannot.
                # Adds ~25K params vs the product arms (concat_proj below) --
                # documented, not hidden -- so a product win is conservative
                # (product wins with LESS capacity); a concat win would need
                # this margin controlled before being taken at face value.
                self.concat_proj = nn.Linear(3 * complex_dim, hidden_dim)

    def band_hz(self) -> torch.Tensor:
        """(n_bands, 2): [center_freq, bandwidth] in Hz, from the sinc params."""
        low = self.sinc.min_low_hz + self.sinc.low_hz_.abs()
        high = low + self.sinc.min_band_hz + self.sinc.band_hz_.abs()
        center = (low + high) / 2
        width = high - low
        return torch.cat([center, width], dim=1)                # (n_bands, 2)

    def _pac_interaction(self, phase_feat, amplitude_feat, pac_vector):
        """Mandatory gauge-invariant phase-amplitude token interaction.

        ``phase_feat`` is complex (B,C,P,I,K), ``amplitude_feat`` is real
        (B,C,P,J,K), and ``pac_vector[...,I,J]`` is

            Z_ij = E[(A_j - mean A_j) exp(i phi_i)].

        Both fusion modes start from the SAME gauge-invariant ingredient,
        ``aligned_phase`` (the alpha-weighted, preferred-phase-aligned sum of
        slower-band phase features):

            aligned_phase_j = sum_{i<j} alpha_ij exp(-i angle Z_ij) p_i   (j>0)
            aligned_phase_0 = p_0                                        (root)

        where alpha is the row-normalised |Z|. Under an arbitrary phase
        reference shift delta_i, p_i -> exp(i delta_i)p_i and
        Z_ij -> exp(i delta_i)Z_ij, so the two factors cancel exactly and
        aligned_phase_{j>0} is gauge-invariant -- this holds regardless of how
        it is later combined with amplitude, so BOTH fusion modes below
        inherit it. This is the physical gauge invariance the old
        phase-steered mixer lacked.

        ``interaction_mode="product"`` (OURS, mandatory): h_j = a_j *
        aligned_phase_j. There is no raw high-band token beside this
        interaction, so the amplitude and the aligned phase cannot be pulled
        apart again downstream -- the interaction is forced.

        ``interaction_mode="concat"`` (SleepPACNet-style control): h_j =
        Linear([a_j, Re(aligned_phase_j), Im(aligned_phase_j)]). The same
        invariant ingredients are exposed, but nothing forces them to
        interact -- the projection could in principle learn to ignore the
        phase columns and pass amplitude through, exactly the free path
        §13.18 says gets optimised away when a prior has one.
        """
        B, C, P, nb, K = phase_feat.shape
        edge = pac_vector.transpose(-2, -1)               # (B,C,P,target,source)
        valid = torch.tril(
            torch.ones(nb, nb, dtype=torch.bool, device=edge.device),
            diagonal=-1,
        )
        mag = edge.abs() * valid
        unit = edge / edge.abs().clamp_min(1e-8)

        if self.pac_token_mode == "uniform":
            count = valid.sum(dim=-1, keepdim=True).clamp_min(1)
            coeff = (valid.to(edge.dtype) / count).view(
                1, 1, 1, nb, nb
            ).expand(B, C, P, -1, -1)
        else:
            if self.pac_token_mode == "scramble":
                # Preserve every |Z| and the batch's exact preferred-phase
                # distribution while breaking which edge owns which phase.
                valid_flat = valid.reshape(nb * nb)
                values = unit.reshape(B, C, P, nb * nb)[..., valid_flat]
                order = torch.rand_like(values.real).argsort(-1)
                shuffled = values.gather(-1, order)
                flat = torch.zeros(
                    B, C, P, nb * nb, dtype=unit.dtype, device=unit.device
                )
                flat[..., valid_flat] = shuffled
                unit = flat.reshape_as(unit)
            if self.pac_token_mode == "magnitude":
                # Deterministic phase-destruction control (AGENT.md 13.43-G6).
                # Keeps the measured magnitude weighting, drops preferred-phase
                # alignment entirely. Unlike `scramble` it injects NO randomness,
                # so `measured - magnitude` isolates the preferred phase without
                # scramble's per-forward permutation noise confound.
                # Like `uniform`/`scramble`, it is NOT gauge-invariant -- only
                # `measured` is, because only there does exp(-i angle Z) cancel
                # the rotation of p_i.
                phase_factor = torch.ones_like(unit)
            else:
                phase_factor = unit.conj()
            denom = mag.sum(dim=-1, keepdim=True)
            measured = (mag / denom.clamp_min(1e-8)) * phase_factor
            count = valid.sum(dim=-1, keepdim=True).clamp_min(1)
            fallback = valid.to(edge.dtype) / count
            coeff = torch.where(denom > 1e-8, measured, fallback)

        aligned_phase = torch.einsum(
            "bcpji,bcpik->bcpjk", coeff, phase_feat
        )
        # The slowest band has no lower-frequency driver. Preserve its own
        # analytic token as the root of the directed hierarchy.
        aligned_phase[:, :, :, 0, :] = phase_feat[:, :, :, 0, :]

        if self.interaction_mode == "product":
            return amplitude_feat.to(aligned_phase.dtype) * aligned_phase
        # concat: expose the same ingredients, let a learned projection combine
        # them. Real, already at the token width (hidden_dim), no view_as_real
        # needed downstream.
        feat = torch.cat(
            [amplitude_feat, aligned_phase.real, aligned_phase.imag], dim=-1
        )
        return self.concat_proj(feat)

    def _interaction_tokens(self, phase_unit, amplitude, pac_vector):
        """Analytic phase/amplitude -> real interleaved PAC interaction tokens."""
        B, C, nb, T = phase_unit.shape
        flat_shape = (B * C * nb, 1, T)
        pr = self.phase_tokenizer(phase_unit.real.reshape(flat_shape))
        pi = self.phase_tokenizer(phase_unit.imag.reshape(flat_shape))
        amp = self.amplitude_tokenizer(
            torch.log1p(amplitude).reshape(flat_shape)
        )
        amp = amp * self.amplitude_scale.view(1, -1, 1)
        P, K = pr.shape[-1], pr.shape[1]
        phase_feat = torch.complex(pr, pi).transpose(1, 2).reshape(
            B, C, nb, P, K
        ).permute(0, 1, 3, 2, 4)
        amplitude_feat = amp.transpose(1, 2).reshape(
            B, C, nb, P, K
        ).permute(0, 1, 3, 2, 4)
        interaction = self._pac_interaction(
            phase_feat, amplitude_feat, pac_vector
        )
        if self.interaction_mode == "product":
            tokens = torch.view_as_real(interaction).flatten(-2)  # (B,C,P,nb,D)
        else:
            tokens = interaction                          # concat_proj already (B,C,P,nb,D) real
        return tokens.permute(0, 1, 3, 2, 4).contiguous()   # (B,C,nb,P,D)

    def forward(self, x: torch.Tensor, return_amp_target: bool = False):
        B, C, T = x.shape
        filtered = self.sinc(x.reshape(B * C, 1, T)).reshape(B, C, self.n_bands, T)

        # phase / amplitude -> time-resolved per-channel coupling
        z = hilbert(filtered)                                    # (B, C, nb, T)
        phase_unit, amplitude = phase_amplitude(z)
        if self.tokenizer_mode == "raw":
            f = filtered.reshape(B * C * self.n_bands, 1, T)
            feat = self.tokenizer(f)                             # (B*C*nb, D, P)
            P = feat.shape[-1]
            tokens = feat.transpose(1, 2).reshape(
                B, C, self.n_bands, P, -1
            )
        else:
            P = T // self.patch_len
        pac_vector = patch_pac_vector(phase_unit, amplitude, P, self.normalize)
        if self.tokenizer_mode == "pac_interaction":
            tokens = self._interaction_tokens(
                phase_unit, amplitude, pac_vector
            )
        coupling = pac_vector.abs()

        if return_amp_target:
            # Per-token (electrode, band, patch) log mean amplitude -- a fixed,
            # deterministic regression target for masked-reconstruction pretraining
            # (models/pretrain.py). Deterministic => no representation collapse, no
            # target encoder needed. Predicting a HIGH band's amplitude from a
            # masked grid.  The asymmetric high-band mask is PAC-inspired, but the
            # target remains a statistical amplitude target rather than a claim
            # that biological coupling is uniquely identified.
            L = T // P
            am = amplitude[..., : P * L].reshape(B, C, self.n_bands, P, L)
            amp_target = torch.log(am.mean(dim=-1) + 1e-6)      # (B, C, nb, P)
            if self.return_pac_vector:
                return tokens, coupling, self.band_hz(), amp_target, pac_vector
            return tokens, coupling, self.band_hz(), amp_target

        if self.return_pac_vector:
            return tokens, coupling, self.band_hz(), pac_vector
        return tokens, coupling, self.band_hz()
