"""v2 tri-axial backbone: positional encodings, axis mixers, block, encoder.

The token grid is (B, C, n_bands, P, D) -- electrode x band x time-patch. Each
block mixes ONE axis at a time (AGENT.md sec. 13.5):

  time  : RoPE self-attention over P patches         (per electrode+band fiber)
  space : self-attention over C electrodes           (per band+patch)
  freq  : directional coupling operator over n_bands  (per electrode+patch)  <-- ours

Factorising the mixing is what keeps compute cheap: instead of one attention
over all C*n_bands*P tokens, each axis is O(axis_len^2) with the other two axes
folded into the batch. The frequency axis stays O(n_bands^2), constant in
sequence length.

Only the FREQUENCY-axis mixer is swapped in the ablation (coupling / attention /
cotar); time and space are always attention. base.py's "swap only the mixer"
contract now means "swap only the frequency-axis mixer" (sec. 13.8).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Positional encodings (AGENT.md sec. 13.4)
# --------------------------------------------------------------------------- #
class BandPE(nn.Module):
    """Encode each band by its (center_freq, bandwidth) in Hz, NOT its index --
    so a different filter bank at finetune time still lands in the same space."""

    def __init__(self, d_model: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )

    def forward(self, band_hz: torch.Tensor) -> torch.Tensor:
        # normalise Hz to ~O(1) so the MLP sees a stable scale across sample rates
        return self.mlp(band_hz / 100.0)                        # (n_bands, D)


class SpatialPE(nn.Module):
    """Per-electrode positional encoding. Learned index embedding for now; the
    montage-agnostic version (MLP over electrode xyz coords) drops in here
    without touching callers once datasets ship coordinates (sec. 13.4)."""

    def __init__(self, n_channels: int, d_model: int):
        super().__init__()
        self.emb = nn.Embedding(n_channels, d_model)

    def forward(self, C: int, device) -> torch.Tensor:
        return self.emb(torch.arange(C, device=device))         # (C, D)


def rope(x: torch.Tensor) -> torch.Tensor:
    """Rotary position embedding over the sequence axis of (..., L, head_dim)."""
    *_, L, hd = x.shape
    half = hd // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(0, half, device=x.device).float() / half
    )
    pos = torch.arange(L, device=x.device).float()
    ang = torch.outer(pos, freqs)                               # (L, half)
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


# --------------------------------------------------------------------------- #
# Axis mixers -- each takes (M, L, D) [L = the axis being mixed] -> (M, L, D)
# --------------------------------------------------------------------------- #
class _MHA(nn.Module):
    """Plain multi-head self-attention over the L axis, optional RoPE."""

    def __init__(self, d_model: int, n_heads: int = 4, use_rope: bool = False):
        super().__init__()
        self.h = n_heads
        self.use_rope = use_rope
        self.scale = (d_model // n_heads) ** -0.5
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        M, L, D = x.shape
        hd = D // self.h
        qkv = self.qkv(x).reshape(M, L, 3, self.h, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.use_rope:
            q, k = rope(q), rope(k)
        a = F.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)
        o = (a @ v).transpose(1, 2).reshape(M, L, D)
        return self.out(o)


class FreqCoupling(nn.Module):
    """OURS: directional PAC-coupling mixer over the n_bands axis.

    For each (electrode, patch) the band tokens attend to each other with logits
    = learned cross-band QK  +  pac_scale * coupling[i->j]. `coupling` is the
    time-resolved MVL matrix for THIS (electrode, patch) (sec. 13.6). Always on:
    this is the only channel through which bands exchange information -- no
    attention fallback path (contrast v5, sec. 9.15/9.17).
    """

    def __init__(self, d_model: int, d_k: int | None = None, **_):
        super().__init__()
        self.d_k = d_k or max(d_model // 4, 16)
        self.q_proj = nn.Linear(d_model, self.d_k, bias=False)
        self.k_proj = nn.Linear(d_model, self.d_k, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.pac_scale = nn.Parameter(torch.ones(1))
        self.lin_out1 = nn.Linear(2 * d_model, d_model)
        self.lin_out2 = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, coupling: torch.Tensor,
                pac_vector: torch.Tensor | None = None) -> torch.Tensor:
        # x: (M, nb, D) with M = B*C*P ; coupling: (M, nb, nb) [i, j]
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        logits = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.d_k)
        logits = logits + self.pac_scale * coupling.transpose(1, 2)   # [j, i]
        w = F.softmax(logits, dim=-1)
        core = torch.bmm(w, v)
        out = self.lin_out2(F.gelu(self.lin_out1(torch.cat([x, core], dim=-1))))
        return out


class FreqAttention(nn.Module):
    """Ablation baseline: plain attention over bands, ignores coupling."""

    def __init__(self, d_model: int, n_heads: int = 4, **_):
        super().__init__()
        self.mha = _MHA(d_model, n_heads)

    def forward(self, x, coupling=None, pac_vector=None):
        return self.mha(x)


class FreqCoTAR(nn.Module):
    """Ablation baseline: CoTAR aggregate-redistribute over bands."""

    def __init__(self, d_model: int, d_core: int | None = None, **_):
        super().__init__()
        d_core = d_core or d_model // 4
        self.lin1 = nn.Linear(d_model, d_model)
        self.lin2 = nn.Linear(d_model, d_core)
        self.lin3 = nn.Linear(d_model + d_core, d_model)
        self.lin4 = nn.Linear(d_model, d_model)

    def forward(self, x, coupling=None, pac_vector=None):
        B, N, D = x.shape
        core = self.lin2(F.gelu(self.lin1(x)))
        core = torch.sum(core * F.softmax(core, dim=1), dim=1, keepdim=True).repeat(1, N, 1)
        return self.lin4(F.gelu(self.lin3(torch.cat([x, core], dim=-1))))


class FreqCoherenceGate(nn.Module):
    """OURS (new primitive): multiplicative coherence gate on plain band attention.

    Motivation — the communication-through-coherence hypothesis (Fries): bands
    should exchange information preferentially when they are phase-coupled. Unlike
    FreqCoupling, which ADDS `pac_scale * coupling` into the attention logits (a
    bias the model learned to zero out -> pac_scale->0, AGENT.md 9.17), this
    MULTIPLIES the softmax attention probabilities by a coupling-derived gate and
    renormalises. A multiplicative gate can *veto* a high-QK-similarity band pair
    that is not coupled -- something an additive logit bias cannot do once the QK
    term dominates.

    Graceful degradation: gate_w initialised to 0 makes the gate uniform, which
    cancels under renormalisation -> at init this is EXACTLY plain attention, so
    the model never starts worse than the FreqAttention baseline and can only
    switch coherence-gating on if it helps. `last_gate` is logged by train.py.
    """

    def __init__(self, d_model: int, n_heads: int = 4, **_):
        super().__init__()
        self.h = n_heads
        self.scale = (d_model // n_heads) ** -0.5
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)
        self.gate_w = nn.Parameter(torch.zeros(1))   # 0 -> uniform gate -> plain attn
        self.gate_b = nn.Parameter(torch.zeros(1))
        self.last_gate = 0.0

    def forward(self, x: torch.Tensor, coupling: torch.Tensor | None = None,
                pac_vector: torch.Tensor | None = None) -> torch.Tensor:
        M, L, D = x.shape
        hd = D // self.h
        qkv = self.qkv(x).reshape(M, L, 3, self.h, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        w = F.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)   # (M, h, L, L)
        if coupling is not None:
            # coupling[.., i, j] = band i (phase) drives band j (amplitude); align
            # to attention's [query j, key i] by transposing, broadcast over heads.
            c = coupling.transpose(1, 2).unsqueeze(1)                   # (M, 1, L, L)
            g = torch.sigmoid(self.gate_w * c + self.gate_b)            # (M, 1, L, L)
            w = w * g
            w = w / (w.sum(dim=-1, keepdim=True) + 1e-8)
            self.last_gate = float(g.mean().item())
        o = (w @ v).transpose(1, 2).reshape(M, L, D)
        return self.out(o)


class FreqPhaseSteered(nn.Module):
    """Parameter-free, directional cross-band communication through complex PAC.

    ``pac_vector[i, j] = mean_t A_j(t) exp(i phi_i(t))`` retains both the
    coupling magnitude and its preferred physical phase.  For every target
    band j, messages may arrive only from slower bands i < j.  Each source
    token is rotated in paired feature planes by angle(pac_vector[i, j]) before
    magnitude-normalised aggregation.

    There is deliberately no QK path, learned PAC scale, gate, or value/output
    projection in this mixer.  Consequently the only way information crosses
    the frequency axis is the measured phase-amplitude geometry itself.  The
    surrounding block still supplies the ordinary within-token residual and
    FFN; those cannot create cross-band communication.
    """

    def __init__(self, d_model: int, **_):
        super().__init__()
        if d_model % 2:
            raise ValueError("FreqPhaseSteered requires an even d_model")

    def forward(self, x: torch.Tensor, coupling: torch.Tensor | None = None,
                pac_vector: torch.Tensor | None = None) -> torch.Tensor:
        if pac_vector is None:
            raise ValueError("FreqPhaseSteered requires the complex pac_vector")

        M, nb, D = x.shape
        if pac_vector.shape != (M, nb, nb):
            raise ValueError(
                f"pac_vector shape {tuple(pac_vector.shape)} != {(M, nb, nb)}"
            )

        # Convert [source phase i, target amplitude j] to [target j, source i].
        z = pac_vector.transpose(1, 2)
        valid = torch.tril(
            torch.ones(nb, nb, dtype=torch.bool, device=x.device), diagonal=-1
        )  # row=target j, col=source i; only i < j
        z = z * valid

        mag = z.abs()
        denom = mag.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        weight = mag / denom
        unit = z / mag.clamp_min(1e-8)
        c, s = unit.real, unit.imag

        # Adjacent feature pairs form 2-D planes.  Complex batched matrix
        # multiplication performs the per-edge rotation and source aggregation
        # without materialising an (M, target, source, D/2) tensor.  That tensor
        # was the dominant cost on 16-electrode TUSZ/CHB-MIT batches.
        value = torch.view_as_complex(x.reshape(M, nb, D // 2, 2).contiguous())
        coeff = weight * torch.complex(c, s)               # (M, target, source)
        out = torch.bmm(coeff, value)                       # (M, target, D/2), complex
        return torch.view_as_real(out).reshape(M, nb, D)


FREQ_MIXERS = {
    "coupling": FreqCoupling,
    "attention": FreqAttention,
    "cotar": FreqCoTAR,
    "coherence": FreqCoherenceGate,
    "phase": FreqPhaseSteered,
}


# --------------------------------------------------------------------------- #
# Tri-axial block + encoder
# --------------------------------------------------------------------------- #
class TriAxialBlock(nn.Module):
    """Pre-norm, one sub-layer per axis, then an FFN. Grid in, grid out."""

    def __init__(self, d_model, freq_mixer="coupling", n_heads=4, dropout=0.1, **mk):
        super().__init__()
        self.n_time = nn.LayerNorm(d_model)
        self.time = _MHA(d_model, n_heads, use_rope=True)
        self.n_space = nn.LayerNorm(d_model)
        self.space = _MHA(d_model, n_heads)
        self.n_freq = nn.LayerNorm(d_model)
        self.freq = FREQ_MIXERS[freq_mixer](d_model, n_heads=n_heads, **mk)
        self.n_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 2 * d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model), nn.Dropout(dropout),
        )

    def forward(self, x, coupling, pac_vector=None):
        # x: (B, C, nb, P, D) ; coupling: (B, C, P, nb, nb)
        B, C, nb, P, D = x.shape

        # time: mix over P, per (B, C, nb)
        h = self.n_time(x).reshape(B * C * nb, P, D)
        x = x + self.time(h).reshape(B, C, nb, P, D)

        # space: mix over C, per (B, nb, P)
        h = self.n_space(x).permute(0, 2, 3, 1, 4).reshape(B * nb * P, C, D)
        h = self.space(h).reshape(B, nb, P, C, D).permute(0, 3, 1, 2, 4)
        x = x + h

        # freq: mix over nb, per (B, C, P), using this (C,P)'s coupling matrix
        h = self.n_freq(x).permute(0, 1, 3, 2, 4).reshape(B * C * P, nb, D)
        cpl = coupling.reshape(B * C * P, nb, nb)
        pac = None if pac_vector is None else pac_vector.reshape(B * C * P, nb, nb)
        h = self.freq(h, cpl, pac).reshape(B, C, P, nb, D).permute(0, 1, 3, 2, 4)
        x = x + h

        x = x + self.ffn(self.n_ffn(x))
        return x


class TriAxialEncoder(nn.Module):
    def __init__(self, depth, d_model, freq_mixer="coupling", n_heads=4, dropout=0.1, **mk):
        super().__init__()
        self.blocks = nn.ModuleList([
            TriAxialBlock(d_model, freq_mixer, n_heads, dropout, **mk) for _ in range(depth)
        ])

    def forward(self, x, coupling, pac_vector=None):
        for blk in self.blocks:
            x = blk(x, coupling, pac_vector)
        return x
