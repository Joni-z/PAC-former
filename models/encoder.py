"""A stack of identical blocks. Phase/amplitude kwargs flow through every block.

For the MI mixer, the band x band coupling matrix depends only on the
frontend's phase/amplitude (identical across layers), so it is computed **once**
here and threaded to every block as ``coupling=`` -- the blocks then skip the
per-layer coupling einsum (AGENT.md sec. 3.2 / 9.15). attention / CoTAR have no
``coupling_matrix`` method, so this is a no-op for them and the swap stays clean.
"""

import torch
import torch.nn as nn

from .block import Block


class Encoder(nn.Module):
    def __init__(self, depth: int, d_model: int, mixer: str, dropout: float = 0.1, **mixer_kwargs):
        super().__init__()
        self.blocks = nn.ModuleList(
            [Block(d_model, mixer, dropout=dropout, **mixer_kwargs) for _ in range(depth)]
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # Precompute the coupling matrix once and reuse across layers. Every
        # block's mixer is the same class with the same `normalize`, so any
        # block can produce it; blocks[0] is convenient. Only done when the
        # mixer actually consumes it (has `coupling_matrix`) and the frontend
        # supplied phase/amplitude.
        mixer0 = self.blocks[0].mixer
        if (
            "coupling" not in kwargs
            and hasattr(mixer0, "coupling_matrix")
            and kwargs.get("phase_unit") is not None
            and kwargs.get("amplitude") is not None
        ):
            kwargs["coupling"] = mixer0.coupling_matrix(
                kwargs["phase_unit"], kwargs["amplitude"]
            )

        for block in self.blocks:
            x = block(x, **kwargs)
        return x
