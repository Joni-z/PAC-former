"""Smoke test for freq_mixer='none' (the 2-axis / CBraMod-style control).

Checks, on CPU with a tiny config:
  1. supervised TriAxialPACLock builds + forwards + backprops with freq_mixer=none
  2. MAEPretrain builds + forwards + backprops with freq_mixer=none
  3. NO cross-band information flow: d(output at band j) / d(input at band i) == 0
     for i != j  -- this is what makes it the honest 2-axis control and the
     mechanism probe for cf_mixed.
  4. freq_mixer=none adds zero parameters vs. the freq mixer it replaces
  5. existing configs are untouched: freq_mixer=attention still mixes bands
"""
import sys
import torch

sys.path.insert(0, "/scratch/zz5070/PACLock")

from models.build import build_model
from models.pretrain import MAEPretrain

BASE = dict(
    dataset="tuab", arch="triaxial", n_channels=4, seq_len=400, sample_rate=200,
    num_classes=2, n_bands=4, d_model=32, depth=2, dropout=0.0, kernel_size=51,
    patch_len=100, n_heads=4, mask_ratio=0.5,
)


def cfg(**kw):
    c = dict(BASE)
    c.update(kw)
    return c


def main():
    torch.manual_seed(0)
    x = torch.randn(2, BASE["n_channels"], BASE["seq_len"])
    ok = True

    # 1. supervised
    m = build_model(cfg(freq_mixer="none"))
    y = m(x)
    y.sum().backward()
    g = sum(p.grad.abs().sum().item() for p in m.parameters() if p.grad is not None)
    print(f"[1] supervised none: logits {tuple(y.shape)} finite={torch.isfinite(y).all().item()} "
          f"grad_sum={g:.1f}")
    ok &= torch.isfinite(y).all().item() and g > 0

    # 2. MAE
    mae = MAEPretrain(cfg(freq_mixer="none", mask_mode="crossfreq"))
    loss = mae(x)
    loss = loss[0] if isinstance(loss, tuple) else loss
    loss.backward()
    print(f"[2] MAE none: loss={loss.item():.5f} finite={torch.isfinite(loss).item()}")
    ok &= torch.isfinite(loss).item()

    # 3. no cross-band flow (the defining property)
    from models.triaxial import TriAxialEncoder
    for name, expect_mix in (("none", False), ("attention", True)):
        torch.manual_seed(0)
        enc = TriAxialEncoder(depth=2, d_model=16, freq_mixer=name, n_heads=2, dropout=0.0)
        B, C, nb, P, D = 1, 2, 4, 3, 16
        g_in = torch.randn(B, C, nb, P, D, requires_grad=True)
        cpl = torch.rand(B, C, P, nb, nb)
        out = enc(g_in, cpl)
        # gradient of band 3's output w.r.t. all inputs
        out[:, :, 3].sum().backward()
        other = g_in.grad[:, :, :3].abs().sum().item()   # bands 0..2
        same = g_in.grad[:, :, 3].abs().sum().item()
        mixed = other > 1e-8
        print(f"[3] freq_mixer={name:9s} grad into OTHER bands={other:.3e} "
              f"same band={same:.3e} -> cross-band mixing={mixed} (expect {expect_mix})")
        ok &= (mixed == expect_mix) and same > 1e-8

    # 4. parameter count
    n_none = sum(p.numel() for p in build_model(cfg(freq_mixer="none")).parameters())
    n_attn = sum(p.numel() for p in build_model(cfg(freq_mixer="attention")).parameters())
    print(f"[4] params none={n_none} attention={n_attn} (none must be smaller)")
    ok &= n_none < n_attn

    print("\nALL GREEN" if ok else "\nFAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
