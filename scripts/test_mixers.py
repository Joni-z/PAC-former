"""Mixer-interface acceptance check (AGENT.md sec. 3).

Instantiate all three mixers with the same dims, feed them the same tensors, and
assert identical output shape/dtype and finite gradients on every one. This must
pass before any mixer is trusted in the encoder.

    python scripts/test_mixers.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.mixers import MIXERS  # noqa: E402


def main():
    B, N, D, T = 4, 16, 64, 500
    x = torch.randn(B, N, D, requires_grad=True)
    # auxiliary phase/amplitude (only the MI mixer reads them; others ignore)
    z = torch.randn(B, N, T) + 1j * torch.randn(B, N, T)
    aux = dict(phase_unit=z / z.abs().clamp_min(1e-6), amplitude=z.abs())

    ref_shape, ref_dtype = None, None
    for name, cls in MIXERS.items():
        mixer = cls(D)
        out = mixer(x, **aux)
        assert out.shape == x.shape, f"{name}: shape {out.shape} != {x.shape}"
        if ref_shape is None:
            ref_shape, ref_dtype = out.shape, out.dtype
        assert out.shape == ref_shape and out.dtype == ref_dtype, f"{name}: mismatch"

        out.sum().backward()
        for p in mixer.parameters():
            assert p.grad is None or torch.isfinite(p.grad).all(), f"{name}: non-finite grad"
        assert torch.isfinite(x.grad).all(), f"{name}: non-finite input grad"
        x.grad = None
        print(f"  {name:10s} OK  out={tuple(out.shape)} {out.dtype}")

    print("All mixers satisfy the interface.")


if __name__ == "__main__":
    main()
