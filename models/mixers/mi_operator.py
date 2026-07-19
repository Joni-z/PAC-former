"""OURS: the differentiable Modulation-Index (MI) token mixer.

Design (v6): **gated convex mix between token-level attention and the
band-level PAC-biased operator.**

    core_attn = MultiHeadSelfAttention(x)              # token-level  (g -> 0 end)
    core_band = v3 band-level op(x, coupling)          # band-level   (g -> 1 end)
    g         = sigmoid(MLP(coupling_statistics))      # per-(sample, band) in (0, 1)
    out       = (1 - g) * core_attn  +  g * core_band

Why this shape (AGENT.md sec. 9.17). v5 tried "attention floor + additive
gated PAC side-branch" on the theory that when PAC is useless the operator
should fall back to the attention baseline, because v3's `pac_scale -> 0`
degenerate form (a *single-head, band-level* attention on `band_repr`) was
assumed to be strictly weaker than the multi-head token-level baseline. **The
logs say that assumption is false.** On CHB-MIT, v3's `pac_scale` collapsed to
~0 by epoch 4 -- yet that run still scored 0.735 test AUROC against the
attention baseline's 0.642. The band-level bottleneck (mean-pool the P patches
per band, mix across the n_bands summaries, broadcast back) is not a
degeneracy, it is *the asset* -- a strong structural prior on noisy, heavily
imbalanced EEG. v5 demoted exactly that to an optional side branch and welded
the always-on token-level attention into the main path, which cost the ceiling
(Sleep-EDF kappa 0.5199 -> 0.5101; CHB-MIT val AUROC 0.853 -> 0.720 at epoch 0).

So the axis the data actually wants to adapt along is **band-level bottleneck
vs. token-level attention**, not "attention plus optional PAC". v6 spans both
endpoints with a convex mix:

  * `g -> 0`: `out == core_attn`, *exactly* the vanilla-attention baseline
    mixer -- the floor that TUAB/TUEV want (v5 did prove this fix works there:
    TUAB val bacc 0.8045 >= attention's 0.7959).
  * `g -> 1`: `out == core_band`, *exactly* the v3 operator, including its own
    learned per-layer `pac_scale` on the coupling bias -- the ceiling that
    Sleep-EDF / CHB-MIT want. Note `pac_scale` is kept *inside* branch B
    precisely because the two datasets use it differently (Sleep-EDF settles
    ~0.2-0.3, CHB-MIT ~0), so the operator must be able to express both.
  * in between: `g` is input-conditioned on the coupling matrix's own
    scale-invariant statistics, so the model *detects* whether real
    cross-frequency structure is present and picks the regime per band, per
    sample, per layer.

Complexity: O(N^2) in the token count N = n_bands*P via branch A (deliberate --
see AGENT.md sec. 5). Branch B stays O(n_bands^2). The coupling matrix depends
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
        d_k: int | None = None,
        gate_hidden: int = 16,
        **_,
    ):
        super().__init__()
        self.normalize = normalize
        self.n_heads = n_heads
        self.d_k = d_k or max(d_model // 4, 16)
        self.attn_scale = (d_model // n_heads) ** -0.5

        # --- Branch A: token-level multi-head self-attention (the g->0 end).
        # Identical computation to models/mixers/attention.py::SelfAttention, so
        # that g -> 0 recovers the attention baseline exactly.
        self.attn_qkv = nn.Linear(d_model, d_model * 3)
        self.attn_out = nn.Linear(d_model, d_model)

        # --- Branch B: the band-level PAC-biased operator (the g->1 end).
        # This is the *whole v3 operator*: learned cross-band QK on band_repr
        # plus the MVL coupling as an additive logit bias with its own learned
        # per-layer pac_scale. Restored deliberately in v6 -- it is what the
        # v3 numbers actually came from (sec. 9.17).
        self.q_proj = nn.Linear(d_model, self.d_k, bias=False)
        self.k_proj = nn.Linear(d_model, self.d_k, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.pac_scale = nn.Parameter(torch.ones(1))
        # redistribute MLP: [token ; band-aggregated core] -> token (CoTAR-style)
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

        # --- Branch A (g -> 0 end): token-level multi-head attention ---
        core_attn = self._self_attention(x)             # (B, N, D)

        # --- Branch B (g -> 1 end): band-level PAC-biased operator (= v3) ---
        band_repr = xb.mean(dim=2)                      # (B, nb, D)
        q = self.q_proj(band_repr)                      # (B, nb, d_k)
        k = self.k_proj(band_repr)                      # (B, nb, d_k)
        v = self.v_proj(band_repr)                      # (B, nb, D)
        # learned cross-band QK logits + the MVL coupling as an additive prior;
        # pac_scale lets each layer set how much of the prior it wants (on
        # Sleep-EDF it settles ~0.2-0.3, on CHB-MIT it goes to ~0 and the band
        # structure alone does the work -- sec. 9.17).
        attn_logits = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.d_k)
        pac_bias = self.pac_scale * coupling.transpose(1, 2)     # (B, nb, nb)[j,i]
        weight = F.softmax(attn_logits + pac_bias, dim=-1)
        core_band = torch.bmm(weight, v)                # (B, nb, D)
        core_band = core_band.unsqueeze(2).expand(-1, -1, P, -1)  # (B, nb, P, D)

        # redistribute (CoTAR-style concat + MLP)
        core_cat = torch.cat([xb, core_band], dim=-1)
        band_out = self.lin_out2(F.gelu(self.lin_out1(core_cat)))  # (B, nb, P, D)
        band_out = band_out.reshape(B, N, D)

        # --- convex mix: spans BOTH endpoints (sec. 9.17) ---
        # g=0 -> exactly the attention baseline; g=1 -> exactly the v3 operator.
        g = self._gate(coupling)                        # (B, nb, 1)
        g_tokens = g.unsqueeze(2).expand(-1, -1, P, -1).reshape(B, N, 1)

        return (1.0 - g_tokens) * core_attn + g_tokens * band_out
