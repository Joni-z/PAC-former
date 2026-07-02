"""OURS: the differentiable Modulation-Index (MI) token mixer.

Design: PAC-biased cross-band attention.

For each target band j, we want to aggregate information from source bands i
weighted by how strongly i (phase) drives j (amplitude). We do this with
cross-band attention whose logits are the sum of a learned QK term AND the MVL
coupling score -- so the operator has both the flexibility of learned attention
(per-layer, data-driven) and the PAC prior (physiologically motivated).

  attn[j, i] = Q_j · K_i / sqrt(d_k)  +  pac_scale * coupling[i, j]
  weight[j, i] = softmax_i(attn[j, i])
  core_j = sum_i weight[j, i] * V_i

This strictly generalises the old pure-PAC mixer (set pac_scale >> 0 and freeze
Q/K) and pure attention (set pac_scale = 0). The model can learn where on that
spectrum is best for each layer.

Coupling matrix follows Canolty (2006) MVL with Ozkurt normalisation and
mean-centred amplitude debiasing (van Driel dPAC style).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import TokenMixer


class MIOperator(TokenMixer):
    def __init__(self, d_model: int, normalize: bool = True, d_k: int | None = None, **_):
        super().__init__()
        self.normalize = normalize
        self.d_k = d_k or max(d_model // 4, 16)

        # Per-layer learned Q / K / V projections for cross-band attention
        self.q_proj = nn.Linear(d_model, self.d_k, bias=False)
        self.k_proj = nn.Linear(d_model, self.d_k, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        # Learnable scale for the PAC bias term (starts at 1.0)
        self.pac_scale = nn.Parameter(torch.ones(1))

        # Redistribute MLP: [token ; aggregated core] -> token
        self.lin_out1 = nn.Linear(2 * d_model, d_model)
        self.lin_out2 = nn.Linear(d_model, d_model)

    def coupling_matrix(
        self, phase_unit: torch.Tensor, amplitude: torch.Tensor
    ) -> torch.Tensor:
        """Directional MVL coupling |Z|, shape (B, n_bands, n_bands) (row=phase, col=amp).

        Mean-centred amplitude debiasing removes the spurious low-frequency term
        that arises because raw amplitude is strictly positive (van Driel dPAC).
        Ozkurt normalisation by amplitude std makes the score a coupling measure,
        not a power measure.
        """
        T = amplitude.shape[-1]
        amp_c = amplitude - amplitude.mean(dim=-1, keepdim=True)
        Z = torch.einsum("bit,bjt->bij", phase_unit, amp_c.to(phase_unit.dtype)) / T
        coupling = Z.abs()
        if self.normalize:
            denom = torch.sqrt((amp_c ** 2).mean(dim=-1)).clamp_min(1e-6)
            coupling = coupling / denom.unsqueeze(1)
        return coupling

    def forward(
        self,
        x: torch.Tensor,
        phase_unit: torch.Tensor | None = None,
        amplitude: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if phase_unit is None or amplitude is None:
            raise ValueError(
                "MIOperator needs `phase_unit` and `amplitude` from the frontend."
            )

        B, N, D = x.shape
        n_bands = phase_unit.shape[1]
        P = N // n_bands
        xb = x.view(B, n_bands, P, D)

        # Band-level representations (mean over patches)
        band_repr = xb.mean(dim=2)   # (B, n_bands, D)

        # --- PAC coupling prior ---
        # coupling[b, i, j]: band i (phase) -> band j (amplitude)
        # For attention query j attending to key i, pac_bias[b, j, i] = coupling[b, i, j]
        coupling = self.coupling_matrix(phase_unit, amplitude)   # (B, nb, nb) [i, j]
        pac_bias = self.pac_scale * coupling.transpose(1, 2)     # (B, nb, nb) [j, i]

        # --- Learned cross-band QK attention ---
        q = self.q_proj(band_repr)   # (B, nb, d_k)
        k = self.k_proj(band_repr)   # (B, nb, d_k)
        v = self.v_proj(band_repr)   # (B, nb, D)

        attn_logits = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.d_k)  # (B, nb, nb) [j, i]

        # PAC-biased attention: learned logits + PAC prior
        weight = F.softmax(attn_logits + pac_bias, dim=-1)   # (B, nb, nb), softmax over i

        # Aggregate: for each band j, gather from source bands i
        core = torch.bmm(weight, v)                          # (B, nb, D)
        core = core.unsqueeze(2).expand(-1, -1, P, -1)      # (B, nb, P, D)

        # Redistribute: concat PAC-aggregated core onto each token, project
        core_cat = torch.cat([xb, core], dim=-1)
        out = self.lin_out2(F.gelu(self.lin_out1(core_cat)))
        return out.reshape(B, N, D)
