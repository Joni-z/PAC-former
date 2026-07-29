"""Mathematical and gradient checks for the mandatory PAC tokenizer."""

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.build import build_model
from models.frontend.triaxial import TriAxialFrontend


def frontend(mode="measured"):
    return TriAxialFrontend(
        n_bands=4, hidden_dim=8, sample_rate=200,
        kernel_size=31, patch_len=100,
        tokenizer_mode="pac_interaction", pac_token_mode=mode,
    )


def main():
    torch.manual_seed(7)
    measured = frontend("measured")
    B, C, P, nb, K = 2, 1, 3, 4, 4
    phase = torch.randn(B, C, P, nb, K, dtype=torch.complex64)
    amplitude = torch.randn(B, C, P, nb, K)
    pac = torch.randn(B, C, P, nb, nb, dtype=torch.complex64)

    # Gauge invariance: independently shift every source band's phase reference.
    delta = torch.randn(nb)
    rotation = torch.polar(torch.ones_like(delta), delta)
    phase_rot = phase * rotation.view(1, 1, 1, nb, 1)
    pac_rot = pac * rotation.view(1, 1, 1, nb, 1)
    y = measured._pac_interaction(phase, amplitude, pac)
    y_rot = measured._pac_interaction(phase_rot, amplitude, pac_rot)
    assert torch.allclose(y[:, :, :, 1:], y_rot[:, :, :, 1:],
                          atol=2e-5, rtol=2e-5)
    print("[1] measured cross-frequency tokens are invariant to phase gauge shifts")

    # The non-PAC matched control lacks the cancelling preferred-phase factor.
    uniform = frontend("uniform")
    u = uniform._pac_interaction(phase, amplitude, pac)
    u_rot = uniform._pac_interaction(phase_rot, amplitude, pac_rot)
    assert not torch.allclose(u[:, :, :, 1:], u_rot[:, :, :, 1:],
                              atol=1e-4, rtol=1e-4)
    print("[2] uniform topology control correctly lacks gauge invariance")

    # High-band tokens have no raw-token path: both their amplitude and a slower
    # phase are load-bearing inputs to the sole construction.
    amp_changed = amplitude.clone()
    amp_changed[:, :, :, 3, :] += 1.0
    phase_changed = phase.clone()
    phase_changed[:, :, :, 0, :] *= 1j
    assert not torch.allclose(
        y[:, :, :, 3], measured._pac_interaction(phase, amp_changed, pac)[:, :, :, 3]
    )
    assert not torch.allclose(
        y[:, :, :, 3], measured._pac_interaction(phase_changed, amplitude, pac)[:, :, :, 3]
    )
    assert not hasattr(measured, "tokenizer")
    print("[3] target amplitude and slower phase are mandatory; no raw-token branch")

    # Full supervised model: forward/backward reaches sinc, phase and amplitude
    # tokenizers through the one main path.
    cfg = {
        "arch": "triaxial", "freq_mixer": "attention",
        "tokenizer_mode": "pac_interaction", "pac_token_mode": "measured",
        "spatial_pe": "xyz", "band_pe": "index", "dataset": "tuab",
        "n_channels": 16, "n_bands": 4, "d_model": 16, "depth": 1,
        "n_heads": 4, "dropout": 0.0, "sample_rate": 200,
        "kernel_size": 31, "patch_len": 100, "num_classes": 2,
        "augmentations": [],
    }
    model = build_model(cfg)
    x = torch.randn(2, 16, 400)
    labels = torch.tensor([0, 1])
    loss = nn.CrossEntropyLoss()(model(x), labels)
    loss.backward()
    for name, parameter in (
        ("sinc", model.frontend.sinc.low_hz_),
        ("phase", model.frontend.phase_tokenizer.weight),
        ("amplitude", model.frontend.amplitude_tokenizer.weight),
    ):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    print(f"[4] full supervised forward/backward finite, loss={loss.item():.4f}")
    print("ALL GREEN")


if __name__ == "__main__":
    main()
