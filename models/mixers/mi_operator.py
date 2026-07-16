"""OURS: the differentiable Modulation-Index (MI) token mixer.

Design (v5): **adaptive gated PAC branch on top of a full-strength attention
floor.**

Motivation (AGENT.md sec. 9.15). The previous design (v3/v4) added the PAC
coupling score as an *additive bias on the QK attention logits*, with a single
learned per-layer scalar `pac_scale`:

    attn[j, i] = Q_j . K_i / sqrt(d_k)  +  pac_scale * coupling[i, j]

That had two problems. (1) On tasks where PAC is not the driving signal,
training simply pushes `pac_scale -> 0` and the operator degenerates into a
*single-head, band-level* attention built on `band_repr` (the per-band mean
over patches) -- strictly weaker than the multi-head, token-level attention
baseline, so MI's *floor* sat below baseline instead of matching it. (2) The
whole thing reads as "attention + a bias term", i.e. a trivial self-attention
variant.

v5 fixes both at once by splitting the mixer into two parallel branches and a
learned gate:

    core_attn = MultiHeadSelfAttention(x)          # floor: full token-level attention
    core_pac  = DirectionalCouplingAggregation(x)  # ceiling: PAC-weighted cross-band mixing
    g         = sigmoid(MLP(coupling_statistics))   # per-(sample, band) detector in (0, 1)
    out       = core_attn  +  g * redistribute(core_pac)

Properties:
  * **Floor >= baseline.** When g -> 0 the PAC branch is switched off entirely
    and `out == core_attn`, which is *exactly* the vanilla-attention baseline
    mixer (same multi-head computation) -- so on non-PAC data MI can, by
    construction, fall back to the attention baseline rather than to a
    crippled single-head one.
  * **Ceiling preserved.** When g is high the directional PAC aggregation is
    injected, giving the physiological prior on PAC-strong tasks.
  * **Input-conditioned adaptivity.** `g` is a function of the coupling
    matrix's own (scale-invariant) statistics -- the model *detects* whether a
    genuine cross-frequency coupling structure is present in this input and
    dials the prior up/down accordingly, per band, per sample. It is no longer
    a frozen scalar.
  * **Distinct from attention.** PAC is now a separate, gated structural
    operator (coupling-weighted aggregation), not a bias folded into the QK
    logits.

Complexity: the attention floor makes this O(N^2) in the token count
N = n_bands*P (a deliberate choice -- see AGENT.md sec. 5 complexity note --
trading the old sub-quadratic property for the accuracy of patch-level
attention). The PAC branch stays O(n_bands^2). The coupling matrix depends
only on the frontend's phase/amplitude, so it is computed once and reused
across all layers (see `Encoder`); each block only re-does the cheap gate +
aggregation.

Coupling matrix follows Canolty (2006) MVL with Ozkurt normalisation and
mean-centred amplitude debiasing (van Driel dPAC style) -- unchanged from v3,
still validated by scripts/synth_pac_test.py.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import TokenMixer

# Fixed divisor for the MVL-coupling normalisation on the real (per-channel)
# path. Replaces the numerically unstable per-channel amplitude-std division
# that produced NaN on flat/dead channels in 16-channel clinical EEG.
NORM_CONST = 100.0


class MIOperator(TokenMixer):
    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        normalize: bool = True,
        gate_hidden: int = 16,
        **_,
    ):
        super().__init__()
        self.normalize = normalize
        self.n_heads = n_heads
        self.attn_scale = (d_model // n_heads) ** -0.5

        # --- Branch A: full-strength multi-head self-attention (the "floor").
        # Identical computation to models/mixers/attention.py::SelfAttention, so
        # that g -> 0 recovers the attention baseline exactly.
        self.attn_qkv = nn.Linear(d_model, d_model * 3)
        self.attn_out = nn.Linear(d_model, d_model)

        # --- Branch B: directional PAC-coupling aggregation (the "ceiling").
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        # temperature on the coupling logits before softmax (learned, init 1.0)
        self.coupling_scale = nn.Parameter(torch.ones(1))
        # redistribute MLP: [token ; PAC-aggregated core] -> token (CoTAR-style)
        self.lin_out1 = nn.Linear(2 * d_model, d_model)
        self.lin_out2 = nn.Linear(d_model, d_model)

        # --- Gate: reads 3 scale-invariant statistics of the coupling column
        # for each target band and emits a value in (0, 1). Small by design --
        # this is a detector, not a capacity add (cf. the failed v3 multi-head
        # experiment, sec. 9.9).
        self.gate_mlp = nn.Sequential(
            nn.Linear(3, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, 1),
        )

        # last mean gate value (per forward, detached) -- purely a training
        # diagnostic, read by train.py to see how much PAC is being used. Not a
        # parameter/buffer so it never enters state_dict.
        self.last_gate = 0.0

    def coupling_matrix(
        self, phase_unit: torch.Tensor, amplitude: torch.Tensor
    ) -> torch.Tensor:
        """Directional MVL coupling |Z|, shape (B, n_bands, n_bands) (row=phase, col=amp).

        Mean-centred amplitude debiasing removes the spurious low-frequency term
        that arises because raw amplitude is strictly positive (van Driel dPAC).
        Ozkurt normalisation by amplitude std makes the score a coupling measure,
        not a power measure.

        Accepts either ``(B, n_bands, T)`` (no channel dim -- used by the
        synthetic/unit-test single-channel inputs) or ``(B, C, n_bands, T)``
        (the real frontend, one analytic signal per channel). In the latter
        case the coupling is computed per channel and *then* averaged -- never
        the raw analytic signal -- since averaging phase/amplitude across
        channels first can have channels cancel or dilute each other's
        coupling (v3 fix, AGENT.md sec. 9.7).
        """
        if phase_unit.dim() == 4:
            amp_c = amplitude - amplitude.mean(dim=-1, keepdim=True)
            Z = torch.einsum("bcit,bcjt->bcij", phase_unit, amp_c.to(phase_unit.dtype))
            Z = Z / amplitude.shape[-1]
            coupling = Z.abs()
            if self.normalize:
                # Fixed-scale normalisation instead of dividing by the per-channel
                # amplitude std. On 16-channel clinical EEG (TUEV/TUEP/TUSZ) some
                # channels are flat/dead in parts, so the per-channel std -> 0 and
                # the division blew up to NaN (only 2-channel Sleep-EDF was safe).
                # A fixed constant removes the division-by-near-zero entirely; the
                # gate absorbs the overall magnitude anyway.
                coupling = coupling / NORM_CONST
            return coupling.mean(dim=1)

        T = amplitude.shape[-1]
        amp_c = amplitude - amplitude.mean(dim=-1, keepdim=True)
        Z = torch.einsum("bit,bjt->bij", phase_unit, amp_c.to(phase_unit.dtype)) / T
        coupling = Z.abs()
        if self.normalize:
            denom = torch.sqrt((amp_c ** 2).mean(dim=-1)).clamp_min(1e-6)
            coupling = coupling / denom.unsqueeze(1)
        return coupling

    def _self_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Multi-head self-attention over the N tokens -- verbatim SelfAttention."""
        b, n, dim = x.shape
        h, hd = self.n_heads, dim // self.n_heads
        qkv = self.attn_qkv(x).reshape(b, n, 3, h, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.attn_scale
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, n, dim)
        return self.attn_out(out)

    def _gate(self, coupling: torch.Tensor) -> torch.Tensor:
        """Per-(sample, target-band) gate in (0, 1) from coupling statistics.

        ``coupling[b, i, j]`` is band i (phase) -> band j (amplitude). For a
        target band j the relevant evidence is the column ``coupling[:, :, j]``
        (everything driving j's amplitude). We feed three *scale-invariant*
        summaries so the gate keys on coupling *structure*, not raw magnitude
        (magnitude varies a lot across datasets / the NORM_CONST path):

          * mean drive into j
          * peak drive into j
          * peakedness (peak - mean): is one source band dominating?

        Scale-invariance comes from dividing by the matrix's own mean first.
        """
        eps = 1e-8
        denom = coupling.mean(dim=(1, 2), keepdim=True).clamp_min(eps)
        c_rel = coupling / denom                        # (B, nb, nb), relative
        col = c_rel                                     # [:, i, j], j = target band
        f_mean = col.mean(dim=1)                        # (B, nb)
        f_max = col.max(dim=1).values                   # (B, nb)
        f_peak = f_max - f_mean                         # (B, nb)
        feat = torch.stack([f_mean, f_max, f_peak], dim=-1)  # (B, nb, 3)
        g = torch.sigmoid(self.gate_mlp(feat))          # (B, nb, 1)
        self.last_gate = g.mean().item()
        return g

    def forward(
        self,
        x: torch.Tensor,
        phase_unit: torch.Tensor | None = None,
        amplitude: torch.Tensor | None = None,
        coupling: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if coupling is None:
            if phase_unit is None or amplitude is None:
                raise ValueError(
                    "MIOperator needs `coupling` or (`phase_unit`, `amplitude`)."
                )
            # standalone / test path -- the Encoder normally precomputes and
            # passes `coupling` once so it is not redone every layer.
            coupling = self.coupling_matrix(phase_unit, amplitude)  # (B, nb, nb)[i,j]

        B, N, D = x.shape
        n_bands = coupling.shape[-1]
        P = N // n_bands
        xb = x.view(B, n_bands, P, D)

        # --- Branch A: full-strength attention (floor) ---
        core_attn = self._self_attention(x)             # (B, N, D)

        # --- Branch B: directional PAC-coupling aggregation (ceiling) ---
        band_repr = xb.mean(dim=2)                      # (B, nb, D)
        v = self.v_proj(band_repr)                      # (B, nb, D)
        # target band j aggregates source bands i by coupling[i -> j]
        logits = self.coupling_scale * coupling.transpose(1, 2)  # (B, nb, nb)[j,i]
        weight = F.softmax(logits, dim=-1)
        core_pac = torch.bmm(weight, v)                 # (B, nb, D)
        core_pac = core_pac.unsqueeze(2).expand(-1, -1, P, -1)   # (B, nb, P, D)

        # redistribute (CoTAR-style concat + MLP), then gate the whole branch
        core_cat = torch.cat([xb, core_pac], dim=-1)
        pac_delta = self.lin_out2(F.gelu(self.lin_out1(core_cat)))  # (B, nb, P, D)
        pac_delta = pac_delta.reshape(B, N, D)

        g = self._gate(coupling)                        # (B, nb, 1)
        g_tokens = g.unsqueeze(2).expand(-1, -1, P, -1).reshape(B, N, 1)

        return core_attn + g_tokens * pac_delta
