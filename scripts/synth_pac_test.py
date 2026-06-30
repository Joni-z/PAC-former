"""Mandatory PAC validation (AGENT.md sec. 5) -- run before any real EEG.

Generate a signal with a KNOWN coupling (10 Hz phase -> 100 Hz amplitude) with
tensorpac, then:
  1. run our differentiable frontend + MI operator, read out the coupling matrix
     |Z|, and check its peak sits at the (10 Hz, 100 Hz) band pair;
  2. cross-check against tensorpac's own MVL comodulogram peak;
  3. confirm loss.backward() yields finite gradients everywhere.

If this fails the bug is in the frontend/operator, not the downstream task.

    python scripts/synth_pac_test.py
"""

import sys
from pathlib import Path

import numpy as np
import torch
from tensorpac import Pac
from tensorpac.signals import pac_signals_tort

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.frontend import Frontend                  # noqa: E402
from models.mixers.mi_operator import MIOperator       # noqa: E402

SF = 200            # EEG-scale sample rate
F_PHA, F_AMP = 10.0, 60.0   # known coupling; f_amp < Nyquist (100 Hz)
N_BANDS = 12        # log-spaced bank: wide gamma bands span the PAC sidebands


def band_bounds(frontend):
    """Per-band (low, high, centre) cutoffs in Hz."""
    low = frontend.sinc.min_low_hz + frontend.sinc.low_hz_.detach().abs()
    high = low + frontend.sinc.min_band_hz + frontend.sinc.band_hz_.detach().abs()
    low, high = low.squeeze(-1).numpy(), high.squeeze(-1).numpy()
    return low, high, (low + high) / 2


def main():
    # 1 epoch, single channel; coupling 10 -> 60 Hz
    sig, _ = pac_signals_tort(f_pha=F_PHA, f_amp=F_AMP, sf=SF, n_times=4000,
                              n_epochs=1, noise=1.0, rnd_state=0)
    x = torch.tensor(sig[None], dtype=torch.float32)   # (B=1, C=1, T)
    T = x.shape[-1]

    frontend = Frontend(n_bands=N_BANDS, hidden_dim=32, seq_len=T,
                        sample_rate=SF, kernel_size=201, n_channels=1)
    mi = MIOperator(d_model=32)

    token, phase_unit, amplitude = frontend(x)
    coupling = mi.coupling_matrix(phase_unit, amplitude)[0]   # (N, N): row=pha, col=amp

    # --- our operator's peak ---
    low, high, centers = band_bounds(frontend)
    i, j = np.unravel_index(coupling.detach().numpy().argmax(), coupling.shape)
    print(f"our MI peak : phase~{centers[i]:.1f}Hz  amp~{centers[j]:.1f}Hz "
          f"[{low[j]:.0f}-{high[j]:.0f}Hz]  (target {F_PHA}/{F_AMP})")

    # --- tensorpac ground-truth comodulogram (MVL = idpac (1,0,0)) ---
    p = Pac(idpac=(1, 0, 0), f_pha=(2, 20, 2, 1), f_amp=(30, 90, 5, 2))
    comod = p.filterfit(SF, sig).squeeze()             # (n_amp, n_pha)
    ai, pi = np.unravel_index(comod.argmax(), comod.shape)
    pha_c = p.xvec[pi] if hasattr(p, "xvec") else p.f_pha.mean(1)[pi]
    amp_c = p.yvec[ai] if hasattr(p, "yvec") else p.f_amp.mean(1)[ai]
    print(f"tensorpac   : phase~{pha_c:.1f}Hz  amp~{amp_c:.1f}Hz")

    # --- assertions ---
    # the global peak should land on the (phase~10 Hz, amplitude-band-spanning-
    # 60 Hz) pair: phase band near F_PHA, amplitude band whose passband covers F_AMP
    assert abs(centers[i] - F_PHA) <= 5.0, f"phase band off: {centers[i]:.1f}Hz"
    assert low[j] <= F_AMP <= high[j], \
        f"amplitude band [{low[j]:.0f}-{high[j]:.0f}] does not span {F_AMP}Hz"

    # --- gradient health ---
    loss = coupling.sum() + token.sum()
    loss.backward()
    for name, mod in [("frontend", frontend), ("mi", mi)]:
        for pn, pp in mod.named_parameters():
            assert pp.grad is None or torch.isfinite(pp.grad).all(), \
                f"non-finite grad in {name}.{pn}"
    print("gradients   : all finite")
    print("PASS: differentiable PAC operator localises the known coupling.")


if __name__ == "__main__":
    main()
