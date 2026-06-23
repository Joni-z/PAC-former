"""OURS: differentiable analytic signal via an FFT-based Hilbert transform.

Given a real band-limited signal we return its analytic signal ``z`` (complex),
from which the MI operator reads two things, *never an angle* (AGENT.md sec. 4):

  * unit phase vector :  ``z / |z|``   (clamped away from |z| = 0)
  * amplitude         :  ``|z|``

The Hilbert transform is the textbook frequency-domain construction: FFT, zero
the negative frequencies and double the positive ones (a step function in
frequency), inverse FFT. It is fully differentiable -- no ``atan2``, no ``arg``.
"""

import torch


def hilbert(x: torch.Tensor) -> torch.Tensor:
    """Analytic signal of real ``x`` along the last axis. Returns complex tensor."""
    T = x.shape[-1]
    Xf = torch.fft.fft(x, dim=-1)

    # step function h: 1 at DC/Nyquist, 2 on positive freqs, 0 on negative freqs
    h = torch.zeros(T, dtype=x.dtype, device=x.device)
    if T % 2 == 0:
        h[0] = h[T // 2] = 1.0
        h[1 : T // 2] = 2.0
    else:
        h[0] = 1.0
        h[1 : (T + 1) // 2] = 2.0
    return torch.fft.ifft(Xf * h, dim=-1)


def phase_amplitude(z: torch.Tensor, eps: float = 1e-6):
    """Split analytic ``z`` into (unit phase vector, amplitude).

    Stays complex throughout; the only singular point (|z| -> 0) is clamped.
    """
    amp = z.abs()
    phase_unit = z / amp.clamp_min(eps)
    return phase_unit, amp
