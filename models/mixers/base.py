"""The load-bearing contract of the repo: every token mixer is interchangeable.

The whole point of PAC-Former is one controlled ablation -- same backbone, same
training, swap only the token mixer between (a) self-attention, (b) CoTAR, and
(c) our MI operator. That swap stays clean only if all three obey this exact
interface (shape and dtype in == out).
"""

import torch
import torch.nn as nn


class TokenMixer(nn.Module):
    """Mixes information across the ``n_bands`` frequency-band tokens.

    Input  : ``x`` of shape ``(batch, n_bands, hidden_dim)``.
    Output : same shape ``(batch, n_bands, hidden_dim)``.

    A mixer may not change ``n_bands`` or ``hidden_dim``, and may not reach back
    into the frontend. If a mixer needs per-band phase/amplitude (the MI
    operator does), those are passed explicitly through ``**kwargs``; mixers
    that do not need them simply ignore ``**kwargs``.
    """

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:  # noqa: D401
        raise NotImplementedError
