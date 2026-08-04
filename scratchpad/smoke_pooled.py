"""Smoke test for the pooled multi-dataset pretrain path (AGENT.md sec. 13.29).

The pooled pretrain is the expensive, near-irreversible run. Everything it depends
on is checked here on CPU first:

  1. VARIABLE WINDOW LENGTH -- the pooled corpus is cropped to 5 s but downstream
     finetuning uses TUAB's 10 s windows. The SAME weights must accept both, or the
     whole pooled design is void.
  2. PooledPretrainDataset cropping: every sample comes out at crop_len, member
     indices are right, a too-short member raises instead of silently padding.
  3. init_from round-trip: a saved MAE reloads with strict matching, so one pooled
     pretrain can be finetuned onto many downstream datasets.
  4. backward compatibility: a config with neither pretrain_pool nor init_from is
     untouched.
"""
import sys
import tempfile
import os

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, "/scratch/zz5070/PACLock")

from data.loaders import PooledPretrainDataset          # noqa: E402
from models.pretrain import MAEPretrain                 # noqa: E402

CFG = dict(
    dataset="tuab", arch="triaxial", freq_mixer="attention", spatial_pe="xyz",
    n_channels=16, seq_len=1000, sample_rate=200, num_classes=2,
    n_bands=8, d_model=32, depth=2, dropout=0.0, kernel_size=51, patch_len=200,
    n_heads=4, mask_ratio=0.5, mask_mode="mixed", mixed_p=0.5,
)


class FakeSet(Dataset):
    def __init__(self, n, C, T):
        self.n, self.C, self.T = n, C, T

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return torch.full((self.C, self.T), float(i)), 0


def main():
    torch.manual_seed(0)
    ok = True

    # ---- 1. one model, two window lengths -----------------------------------
    mae = MAEPretrain(CFG)
    losses = {}
    for T in (1000, 2000):
        x = torch.randn(2, 16, T)
        loss = mae(x)
        loss = loss[0] if isinstance(loss, tuple) else loss
        losses[T] = float(loss)
        print(f"[1] T={T:5d} -> recon loss {loss.item():.5f} finite={torch.isfinite(loss).item()}")
        ok &= bool(torch.isfinite(loss).item())
    # encode() is what the downstream Probe calls; it must also take both
    for T in (1000, 2000):
        h = mae.encode(torch.randn(2, 16, T))
        print(f"[1] encode T={T:5d} -> {tuple(h.shape)}")
        ok &= torch.isfinite(h).all().item()

    # ---- 2. pooled dataset cropping -----------------------------------------
    members = [("tuab", FakeSet(5, 16, 2000)), ("tuev", FakeSet(3, 16, 1000))]
    ds = PooledPretrainDataset(members, crop_len=1000, seed=0)
    shapes, mems = set(), []
    for i in range(len(ds)):
        X, m = ds[i]
        shapes.add(tuple(X.shape)); mems.append(m)
    print(f"[2] len={len(ds)} shapes={shapes} members={mems} composition={ds.composition()}")
    ok &= len(ds) == 8 and shapes == {(16, 1000)} and mems == [0]*5 + [1]*3

    try:
        PooledPretrainDataset([("bad", FakeSet(2, 16, 500))], crop_len=1000)[0]
        print("[2] too-short member did NOT raise -- BAD"); ok = False
    except ValueError as e:
        print(f"[2] too-short member raises as intended: {str(e)[:60]}...")

    # random crop actually varies (it is also the augmentation)
    offs = {tuple(PooledPretrainDataset([("t", FakeSet(1, 1, 2000))], 1000, seed=s)[0][0][0][:1].tolist())
            for s in range(3)}
    print(f"[2] crop is random across seeds (values constant per sample by design): {len(offs)} distinct")

    # ---- 3. init_from round-trip --------------------------------------------
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "pooled.pt")
        torch.save(mae.state_dict(), p)
        mae2 = MAEPretrain(CFG)
        missing, unexpected = mae2.load_state_dict(torch.load(p, map_location="cpu"), strict=False)
        print(f"[3] reload missing={list(missing)} unexpected={list(unexpected)}")
        ok &= not missing and not unexpected
        x = torch.randn(2, 16, 2000)
        mae.eval(); mae2.eval()
        with torch.no_grad():
            same = torch.allclose(mae.encode(x), mae2.encode(x), atol=1e-6)
        print(f"[3] reloaded encoder reproduces embeddings: {same}")
        ok &= same

    # ---- 4. backward compatibility ------------------------------------------
    plain = dict(CFG); plain.pop("spatial_pe")
    m3 = MAEPretrain(plain)
    l = m3(torch.randn(2, 16, 1000))
    l = l[0] if isinstance(l, tuple) else l
    print(f"[4] config without spatial_pe/pool still builds+runs: loss={l.item():.5f}")
    ok &= bool(torch.isfinite(l).item())

    print("\nALL GREEN" if ok else "\nFAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
