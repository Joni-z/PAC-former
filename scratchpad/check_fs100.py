"""Verify the fs=100 TUAB control configs on real data before burning GPU slots.

The test they enable: at 200 Hz the crossfreq mask hides 50-98 Hz; at 100 Hz it hides
24-48 Hz. Sleep-EDF is 100 Hz and has NO content above 48 Hz at all, so if cf_mixed's
margin over random MAE disappears when TUAB is bandwidth-limited to 100 Hz, that is the
Sleep-EDF loss explained (AGENT.md sec. 13.28 Link 1).
"""
import sys
import numpy as np
import torch
import yaml

sys.path.insert(0, "/scratch/zz5070/PAC-former")
from models.pretrain import MAEPretrain     # noqa: E402
from data import build_dataloaders          # noqa: E402

for fs in (200, 100):
    e = np.linspace(1.0, fs / 2 - 2.0, 9)
    print(f"fs={fs:3d} bands " + " ".join(f"{a:.0f}-{b:.0f}" for a, b in zip(e[:-1], e[1:]))
          + f"  | crossfreq hides {e[4]:.0f}-{e[-1]:.0f} Hz")

ok = True
for p in ("configs/pretrain_tuab_cf_mixed_ft_fs100.yaml",
          "configs/pretrain_tuab_random_ft_fs100.yaml"):
    cfg = yaml.safe_load(open(p))
    cfg["num_workers"], cfg["batch_size"] = 0, 4
    m = MAEPretrain(cfg)
    tr, _, _, _ = build_dataloaders(cfg)
    X, _ = next(iter(tr))
    loss = m(X)
    loss = loss[0] if isinstance(loss, tuple) else loss
    h = m.encode(X)
    print(f"{p}\n   real batch {tuple(X.shape)} -> loss {loss.item():.5f} "
          f"grid {tuple(h.shape)} finite={torch.isfinite(loss).item()}")
    # P must match the 200 Hz runs (10 patches) so ONLY frequency content differs
    ok &= torch.isfinite(loss).item() and h.shape[3] == 10 and X.shape[-1] == 1000

print("\nALL GREEN" if ok else "\nFAILED")
sys.exit(0 if ok else 1)
