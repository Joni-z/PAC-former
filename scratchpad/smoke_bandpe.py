"""Smoke test for the band_pe ablation switch (AGENT.md sec. 13.28 Link 5).

Checks all three modes build/forward/backprop in both the supervised and MAE
paths, that `none` really contributes nothing, that `index` does not depend on
the filter bank's Hz values (the property "hz" claims to buy), and that a config
with no `band_pe` key is bit-for-bit the old default.
"""
import sys
import torch

sys.path.insert(0, "/scratch/zz5070/PACLock")

from models.build import build_model                 # noqa: E402
from models.pretrain import MAEPretrain              # noqa: E402
from models.triaxial import BandPE                   # noqa: E402

BASE = dict(
    dataset="tuab", arch="triaxial", freq_mixer="attention",
    n_channels=4, seq_len=400, sample_rate=200, num_classes=2, n_bands=4,
    d_model=32, depth=2, dropout=0.0, kernel_size=51, patch_len=100,
    n_heads=4, mask_ratio=0.5, mask_mode="crossfreq",
)


def cfg(**kw):
    c = dict(BASE); c.update(kw); return c


def main():
    torch.manual_seed(0)
    x = torch.randn(2, BASE["n_channels"], BASE["seq_len"])
    ok = True

    for mode in ("hz", "index", "none"):
        m = build_model(cfg(band_pe=mode))
        y = m(x); y.sum().backward()
        g = sum(p.grad.abs().sum().item() for p in m.parameters() if p.grad is not None)
        mae = MAEPretrain(cfg(band_pe=mode))
        l = mae(x); l = l[0] if isinstance(l, tuple) else l
        npar = sum(p.numel() for p in m.parameters())
        print(f"[1] band_pe={mode:5s} sup_finite={torch.isfinite(y).all().item()} "
              f"grad={g:8.1f} mae_loss={l.item():.5f} params={npar}")
        ok &= torch.isfinite(y).all().item() and g > 0 and torch.isfinite(l).item()

    # 2. `none` really emits zeros; `index` ignores the Hz values; `hz` does not
    band_hz_a = torch.tensor([[4.0, 2.0], [10.0, 4.0], [30.0, 8.0], [60.0, 16.0]])
    band_hz_b = band_hz_a * 1.7           # a different filter bank
    for mode in ("hz", "index", "none"):
        pe = BandPE(8, n_bands=4, mode=mode).eval()
        with torch.no_grad():
            a, b = pe(band_hz_a), pe(band_hz_b)
        depends = not torch.allclose(a, b)
        zero = bool(a.abs().sum() == 0)
        print(f"[2] band_pe={mode:5s} shape={tuple(a.shape)} "
              f"depends_on_Hz={depends} all_zero={zero}")
        ok &= a.shape == (4, 8)
        ok &= {"hz": depends, "index": not depends, "none": not depends}[mode]
        ok &= (mode == "none") == zero

    # 3. default == hz, exactly
    torch.manual_seed(1); d = build_model(cfg())
    torch.manual_seed(1); h = build_model(cfg(band_pe="hz"))
    same = all(torch.equal(a, b) for a, b in zip(d.state_dict().values(),
                                                 h.state_dict().values()))
    print(f"[3] config without band_pe identical to band_pe=hz: {same}")
    ok &= same

    print("\nALL GREEN" if ok else "\nFAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
