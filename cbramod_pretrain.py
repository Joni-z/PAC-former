"""Does OUR objective help CBraMod's backbone? (goal 2026-07-25, item 2)

CBraMod-scratch beats ours on TUEV (§13.40-B). Our contribution is the OBJECTIVE, not the
backbone, so the strong move is to show the objective helps even CBraMod's SOTA backbone.

CBraMod tokenises by (channel, 1-second time-patch) — it has NO frequency-band axis, so our
band-grid crossfreq mask does not transfer literally. We map its SPIRIT ("predict hidden high
bands from visible low bands") into CBraMod's own token space as a spectral objective:

  * crossfreq (ours):  low-pass every patch (zero the high half of its 101-bin rFFT), feed the
                       low-passed signal, reconstruct the ORIGINAL full patch. The model must
                       recover 50-100 Hz content from 0-50 Hz — cross-frequency prediction.
  * random (CBraMod's native pretext): mask whole (channel, patch) tokens with CBraMod's own
                       `mask_encoding`, reconstruct the masked patches.

Both share the backbone and the raw-patch MSE target; only what is hidden differs. Phase 2
finetunes the same classifier head as baseline_cbramod.py, so the whole comparison is:
same backbone, same finetune, ONE objective swapped.

    python cbramod_pretrain.py --config configs/cbramod_pt_chbmit.yaml --obj crossfreq
"""

import argparse
import copy
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference/CBraMod"))

from eval import compute_metrics, select_key                      # noqa: E402
from models.cbramod import CBraMod                                # noqa: E402
from baseline_cbramod import CBraModNpy, run_epoch                # noqa: E402


def lowpass_patches(x, keep=0.5):
    """(B,C,S,200) -> low-passed: zero the top (1-keep) of each patch's rFFT."""
    B, C, S, P = x.shape
    spec = torch.fft.rfft(x, dim=-1)                              # (...,101)
    n = spec.shape[-1]
    cut = int(round(n * keep))
    spec[..., cut:] = 0
    return torch.fft.irfft(spec, n=P, dim=-1)


class CBraModPretrain(nn.Module):
    def __init__(self, obj, mask_ratio=0.5):
        super().__init__()
        self.obj, self.mask_ratio = obj, mask_ratio
        self.backbone = CBraMod(in_dim=200, out_dim=200, d_model=200,
                                dim_feedforward=800, seq_len=30, n_layer=12, nhead=8)

    def forward(self, x):                                         # x: (B,C,S,200)
        if self.obj == "crossfreq":
            inp = lowpass_patches(x, keep=0.5)
            rec = self.backbone(inp)                              # reconstruct full patch
            return nn.functional.mse_loss(rec, x)                # MSE on ALL positions
        # random: CBraMod-native whole-token masking
        B, C, S, _ = x.shape
        mask = (torch.rand(B, C, S, device=x.device) < self.mask_ratio)
        rec = self.backbone(x, mask=mask.long())
        m = mask.unsqueeze(-1).expand_as(x)
        return nn.functional.mse_loss(rec[m], x[m]) if m.any() else rec.sum() * 0.0


class Classifier(nn.Module):
    """CBraMod backbone (loaded) + two-layer flatten head, same as baseline_cbramod."""

    def __init__(self, backbone, n_classes, n_ch, n_patch, dropout=0.1):
        super().__init__()
        self.backbone = backbone
        self.backbone.proj_out = nn.Identity()
        flat = n_ch * n_patch * 200
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(flat, 200), nn.ELU(), nn.Dropout(dropout),
            nn.Linear(200, n_classes))

    def forward(self, x):
        return self.head(self.backbone(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--obj", choices=["crossfreq", "random"], required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    seed = cfg.get("seed", 0)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    root, native = cfg["data_root"], cfg["native_rate"]
    n_patch, shift = cfg["duration_s"], cfg.get("label_shift", 0)
    sets = [CBraModNpy(root, s, native, 200, 200, shift) for s in ("train", "val", "test")]
    bs, nw = cfg.get("batch_size", 64), cfg.get("num_workers", 8)
    ld = [DataLoader(d, batch_size=bs, shuffle=(i == 0), drop_last=(i == 0),
                     num_workers=nw, pin_memory=True) for i, d in enumerate(sets)]
    train_loader, val_loader, test_loader = ld
    ncls = cfg["num_classes"]

    # ---- Phase 1: pretrain backbone with the chosen objective ----
    pt = CBraModPretrain(args.obj, cfg.get("mask_ratio", 0.5)).to(device)
    opt = torch.optim.AdamW(pt.parameters(), lr=cfg.get("pretrain_lr", 1e-4), weight_decay=1e-4)
    print(f"[cbramod-pt] obj={args.obj} params={sum(p.numel() for p in pt.parameters())/1e6:.2f}M")
    for ep in range(cfg.get("pretrain_epochs", 15)):
        pt.train(); losses = []
        for X, _ in tqdm(train_loader, leave=False):
            X = X.to(device, non_blocking=True)
            loss = pt(X)
            opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
        print(f"[pretrain] epoch {ep:3d} | recon {np.mean(losses):.5f}", flush=True)

    # ---- Phase 2: finetune classifier on the pretrained backbone ----
    model = Classifier(pt.backbone, ncls, cfg["n_channels"], n_patch, cfg.get("dropout", 0.1)).to(device)
    cw = None
    if cfg.get("class_weights"):
        counts = np.bincount(sets[0].labels - shift, minlength=ncls).astype(np.float64)
        cw = torch.FloatTensor(counts.sum() / (ncls * np.clip(counts, 1, None))).to(device)
    criterion = nn.CrossEntropyLoss(weight=cw)
    fopt = torch.optim.Adam(model.parameters(), lr=cfg.get("finetune_lr", 1e-4),
                            weight_decay=cfg.get("weight_decay", 1e-5))
    key = select_key(ncls, cfg); ees = cfg.get("eval_every_steps", 0)
    best, best_state, best_ep = -np.inf, None, -1

    def mid(step):
        nonlocal best, best_state, best_ep
        _, vl, vy = run_epoch(model, val_loader, device, criterion)
        m = compute_metrics(vy, vl, ncls)
        print(f"[probe] ep {ep} step {step} | " + " ".join(f"val_{k}={v:.4f}" for k, v in m.items()), flush=True)
        if m[key] > best:
            best, best_ep = m[key], ep
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})

    for ep in range(cfg.get("probe_epochs", 10)):
        tr, _, _ = run_epoch(model, train_loader, device, criterion, fopt, eval_hook=mid, eval_every_steps=ees)
        _, vl, vy = run_epoch(model, val_loader, device, criterion)
        m = compute_metrics(vy, vl, ncls)
        print(f"[probe] epoch {ep:3d} | loss {tr:.4f} | " + " ".join(f"val_{k}={v:.4f}" for k, v in m.items()), flush=True)
        if m[key] > best:
            best, best_ep = m[key], ep
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})

    model.load_state_dict(best_state)
    _, tl, ty = run_epoch(model, test_loader, device, criterion)
    print(f"[cbramod-pt] obj={args.obj} test@best (ep {best_ep}) | "
          + " ".join(f"{k}={v:.4f}" for k, v in compute_metrics(ty, tl, ncls).items()))


if __name__ == "__main__":
    main()
