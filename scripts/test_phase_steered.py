"""Fast acceptance tests for the non-bypassable phase-steered frequency mixer."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.build import build_model  # noqa: E402
from models.triaxial import FreqPhaseSteered  # noqa: E402


def operator_checks():
    torch.manual_seed(0)
    M, nb, D = 2, 4, 8
    x = torch.randn(M, nb, D, requires_grad=True)
    mixer = FreqPhaseSteered(D)

    # [source, target]. Only low -> high entries should communicate.
    z = torch.zeros(M, nb, nb, dtype=torch.complex64)
    z[:, 0, 2] = 1.0 + 0.0j
    z[:, 0, 3] = 0.0 + 1.0j
    z[:, 1, 3] = 1.0 + 0.0j
    z.requires_grad_()
    y = mixer(x, pac_vector=z)

    assert y.shape == x.shape and torch.isfinite(y).all()
    assert torch.equal(y[:, 0], torch.zeros_like(y[:, 0]))
    assert torch.equal(y[:, 1], torch.zeros_like(y[:, 1]))
    assert torch.allclose(y[:, 2], x[:, 0], atol=1e-6)

    # Coupling magnitude is normalised, while preferred physical phase changes
    # the representation geometry.
    assert torch.allclose(y, mixer(x, pac_vector=5.0 * z), atol=1e-6)
    z_shift = z.detach() * 1j
    assert not torch.allclose(y[:, 2:], mixer(x, pac_vector=z_shift)[:, 2:])

    # Reverse (high -> low) edges are forbidden by construction.
    reverse = torch.zeros_like(z.detach())
    reverse[:, 3, 0] = 1.0
    assert torch.equal(mixer(x, pac_vector=reverse), torch.zeros_like(x))

    y.square().mean().backward()
    assert torch.isfinite(x.grad).all()
    assert z.grad is not None and torch.isfinite(z.grad).all()


def full_model_check():
    cfg = {
        "arch": "triaxial",
        "freq_mixer": "phase",
        "n_channels": 2,
        "seq_len": 400,
        "sample_rate": 100,
        "num_classes": 3,
        "n_bands": 4,
        "d_model": 16,
        "depth": 2,
        "n_heads": 4,
        "kernel_size": 101,
        "patch_len": 100,
        "dropout": 0.0,
    }
    model = build_model(cfg)
    x = torch.randn(2, 2, 400)
    logits = model(x)
    assert logits.shape == (2, 3) and torch.isfinite(logits).all()
    logits.square().mean().backward()
    for name, p in model.named_parameters():
        assert p.grad is None or torch.isfinite(p.grad).all(), name


if __name__ == "__main__":
    operator_checks()
    full_model_check()
    print("PASS: phase-steered directionality, phase sensitivity, and gradients")
