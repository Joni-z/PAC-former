"""Token mixers and the registry that lets a config pick one by name.

Swapping the mixer is a single config string -- this registry is the only place
that maps that string to a class. All three satisfy ``TokenMixer``.
"""

from .base import TokenMixer
from .attention import SelfAttention
from .cotar import CoTAR
from .mi_operator import MIOperator

MIXERS = {
    "attention": SelfAttention,
    "cotar": CoTAR,
    "mi": MIOperator,
}


def build_mixer(name: str, d_model: int, **kwargs) -> TokenMixer:
    if name not in MIXERS:
        raise KeyError(f"unknown mixer '{name}', choose from {list(MIXERS)}")
    return MIXERS[name](d_model, **kwargs)
