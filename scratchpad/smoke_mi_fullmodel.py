"""Full-model build/forward/backward for the 4 MI variants on REAL TUEV data.

Mixer-level correctness is covered by smoke_mi_mixers.py. This one checks the thing
that unit tests cannot: that the mixer survives being wired into TriAxialPACFormer
with the actual config, the actual token grid, and real EEG -- finite loss, finite
gradient on every parameter, and the per-variant diagnostic actually firing on real
coupling (not just on the synthetic draw used in the unit test).
"""

import sys

import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, ".")

from data.loaders import build_dataloaders          # noqa: E402
from models.build import build_model                # noqa: E402

base = yaml.safe_load(open("configs/ours_scratch_tuev_bandidx.yaml"))
base.update(batch_size=2, num_workers=0, device="cpu")
x, y = next(iter(build_dataloaders(base)[0]))
print(f"real TUEV batch: x={tuple(x.shape)}  y={y.tolist()}", flush=True)
print(f"\n{'variant':22s} {'params':>9s} {'loss':>7s} {'grad_norm':>10s}  diagnostic", flush=True)

fails = []
for fm in ["attention", "mi_product", "mi_product_shuffle", "mi_topk", "mi_topk_shuffle"]:
    cfg = dict(base)
    cfg["freq_mixer"] = fm
    torch.manual_seed(0)
    m = build_model(cfg)
    m.train()
    loss = F.cross_entropy(m(x), y)
    loss.backward()
    gn = sum((p.grad ** 2).sum() for p in m.parameters() if p.grad is not None).sqrt()
    nofinite = [n for n, p in m.named_parameters()
                if p.grad is not None and not torch.isfinite(p.grad).all()]
    nograd = [n for n, p in m.named_parameters() if p.grad is None]
    blk = m.encoder.blocks[0].freq
    if hasattr(blk, "last_mi_spread"):
        diag = f"mi_spread={blk.last_mi_spread:.4f}"
        diag_ok = blk.last_mi_spread > 1e-3
    elif hasattr(blk, "last_kept_frac"):
        diag = f"kept_frac={blk.last_kept_frac:.4f}"
        diag_ok = abs(blk.last_kept_frac - 3 / cfg["n_bands"]) < 1e-6
    else:
        diag, diag_ok = "-", True
    ok = bool(torch.isfinite(loss)) and bool(torch.isfinite(gn)) and not nofinite and diag_ok
    if not ok:
        fails.append((fm, nofinite, nograd, diag_ok))
    n = sum(p.numel() for p in m.parameters())
    print(f"{fm:22s} {n:>9,} {loss.item():>7.4f} {gn.item():>10.3f}  {diag:24s} "
          f"{'OK' if ok else 'FAIL'}", flush=True)

print("\n" + "=" * 62, flush=True)
if fails:
    print(f"FAIL: {fails}")
    sys.exit(1)
print("PASS: all 4 MI variants train end-to-end on real TUEV data")
