"""OURS: the differentiable Modulation-Index (MI) token mixer.

This is the contribution. It replaces self-attention with a directional
phase-amplitude-coupling operator built on the same aggregate -> redistribute
skeleton as CoTAR, so it is a legitimate attention replacement rather than a
pooling layer (see README, "MI 算子与 CoTAR 的同构对应").

Construction, given per-band unit phase vectors and amplitudes over time:

  1. Aggregate (the Mean-Vector-Length form, Canolty 2006):
         Z[i, j] = (1/T) sum_t  A_j(t) * e^{i phi_i(t)}
     row i = low-frequency *modulator* (supplies phase),
     col j = high-frequency *modulated* band (supplies amplitude).
     |Z| is the directional N x N coupling matrix -- the part we add on top of
     CoTAR's symmetric core. Ozkurt-normalised so the model learns coupling,
     not which band has the most power.

  2. Redistribute (concat + MLP, matching CoTAR; AGENT.md section 9 default):
     for each modulated band j, pull together the modulator tokens weighted by
     how strongly they couple into j, concat onto x_j, project.

Numerical-stability rule (AGENT.md section 4): we never take an angle. Phase
enters only as a precomputed unit complex vector ``z / |z|``; the operator stays
in complex arithmetic and the only guarded singularity (|z| -> 0) is handled in
the frontend.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import TokenMixer


class MIOperator(TokenMixer):
    def __init__(self, d_model: int, normalize: bool = True, **_):
        super().__init__()
        self.normalize = normalize
        # redistribute MLP: [token ; directional-core] -> token  (mirrors CoTAR)
        self.lin_out1 = nn.Linear(2 * d_model, d_model)
        self.lin_out2 = nn.Linear(d_model, d_model)

    def coupling_matrix(
        self, phase_unit: torch.Tensor, amplitude: torch.Tensor
    ) -> torch.Tensor:
        """Directional MVL coupling |Z|, shape ``(B, N, N)`` (row=phase, col=amp).

        ``phase_unit`` : complex ``(B, N, T)``, unit modulus per sample.
        ``amplitude``  : real ``(B, N, T)``.

        We mean-centre the amplitude envelope before the MVL sum (debiasing, cf.
        van Driel's dPAC). Raw MVL is ``mean_t A_j e^{i phi_i}``; because A_j is
        strictly positive it carries a ``mean(A_j) * mean(e^{i phi_i})`` term
        that is large whenever the phase distribution is non-uniform (low-
        frequency modulators, finite windows), producing spurious coupling at
        the lowest bands. Centring A_j removes exactly that term, leaving the
        genuine phase->amplitude covariance.
        """
        T = amplitude.shape[-1]
        amp_c = amplitude - amplitude.mean(dim=-1, keepdim=True)
        Z = torch.einsum("bit,bjt->bij", phase_unit, amp_c.to(phase_unit.dtype)) / T
        coupling = Z.abs()
        if self.normalize:
            # normalise by the amplitude std per band j (Ozkurt-style), so the
            # operator scores coupling, not which band carries the most power
            denom = torch.sqrt((amp_c ** 2).mean(dim=-1)).clamp_min(1e-6)
            coupling = coupling / denom.unsqueeze(1)  # broadcast over modulators i
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
                "MIOperator needs `phase_unit` and `amplitude` from the frontend; "
                "pass them as keyword arguments (see frontend contract)."
            )

        coupling = self.coupling_matrix(phase_unit, amplitude)  # (B, N, N)

        # aggregate: for each modulated band j, attend over its modulators i
        weight = F.softmax(coupling, dim=1)              # normalise over i (rows)
        core = torch.einsum("bij,bid->bjd", weight, x)   # (B, N, D)

        # redistribute: concat core onto each token, project
        core_cat = torch.cat([x, core], dim=-1)
        return self.lin_out2(F.gelu(self.lin_out1(core_cat)))
