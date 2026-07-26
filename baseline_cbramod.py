"""CBraMod baseline on OUR splits + eval (goal 2026-07-24: add CBraMod alongside BIOT).

Same contract as baseline_biot.py: CBraMod's own model, but our train/val/test SPLITS and
our `eval.compute_metrics` and `eval.select_key`, so the model is the controlled variable.

ONE deliberate difference from baseline_biot.py, learned in §13.35: CBraMod is fed its OWN
input normalisation, not ours. CBraMod trained on `µV / 100` (reference/CBraMod
datasets/*_dataset.py: `return data/100`). Our stored signals are in Volts (~1e-5), so
µV/100 = stored * 1e6 / 100 = stored * 1e4 (95th-pct abs -> ~0.32, the O(0.3) scale CBraMod
expects). Forcing our 95th-percentile normalisation onto a model pretrained on µV/100 would
cripple it exactly the way the wrong learning rate crippled BIOT in §13.35. The controlled
variables are the splits, the task, and the eval — not the input scaling, which is intrinsic
to a pretrained model like its tokenizer. Kept identical for both init modes so CBraMod-scratch
and CBraMod-pretrained sit on the same footing.

Input geometry: CBraMod wants (B, C, S, 200) — S one-second patches of 200 samples at 200 Hz.
Our npy is (N, C, T); TUEV's is 250 Hz (T=1250) so it is resampled to 200 Hz (T=1000) first,
matching what our own TUEV loader does.

    python baseline_cbramod.py --config configs/cbramod_tuab.yaml --init pretrained
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
from scipy.signal import resample
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference/CBraMod"))

from eval import compute_metrics, select_key           # noqa: E402
from models.cbramod import CBraMod                      # noqa: E402

WEIGHTS = "reference/CBraMod/pretrained_weights/pretrained_weights.pth"


class CBraModNpy(Dataset):
    """Reads the SAME {split}_signals.npy / {split}_labels.npy our pipeline uses (so the
    splits are identical), but applies CBraMod's own normalisation and patch geometry."""

    def __init__(self, root, split, default_rate, target_rate=200, patch=200, label_shift=0):
        self.signals = np.load(os.path.join(root, f"{split}_signals.npy"), mmap_mode="r")
        self.labels = np.load(os.path.join(root, f"{split}_labels.npy"))
        self.default_rate, self.target_rate = default_rate, target_rate
        self.patch, self.label_shift = patch, label_shift
        self.dur = self.signals.shape[-1] // default_rate           # seconds

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        x = np.asarray(self.signals[i], dtype=np.float32)           # (C, T) Volts
        if self.default_rate != self.target_rate:
            x = resample(x, self.dur * self.target_rate, axis=-1)
        x = x * 1e4                                                 # Volts -> µV/100
        C, T = x.shape
        x = x.reshape(C, T // self.patch, self.patch)              # (C, S, 200)
        return torch.FloatTensor(x), int(self.labels[i]) - self.label_shift


class CBraModClassifier(nn.Module):
    """CBraMod backbone + their two-layer flatten head (all_patch_reps_twolayer)."""

    def __init__(self, n_classes, n_ch, n_patch, pretrained, dropout=0.1):
        super().__init__()
        self.backbone = CBraMod(in_dim=200, out_dim=200, d_model=200,
                                dim_feedforward=800, seq_len=30, n_layer=12, nhead=8)
        if pretrained:
            sd = torch.load(WEIGHTS, map_location="cpu")
            self.backbone.load_state_dict(sd)                       # strict: verified exact
        self.backbone.proj_out = nn.Identity()
        flat = n_ch * n_patch * 200
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 200), nn.ELU(), nn.Dropout(dropout),
            nn.Linear(200, n_classes),
        )

    def forward(self, x):
        return self.head(self.backbone(x))                          # (B,C,S,200)->(B,C,S,200)->logits


def run_epoch(model, loader, device, criterion, opt=None, eval_hook=None, eval_every_steps=0):
    train = opt is not None
    model.train(train)
    losses, logits_all, y_all = [], [], []
    for step, (X, y) in enumerate(tqdm(loader, leave=False)):
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).long()
        with torch.set_grad_enabled(train):
            logits = model(X)
            loss = criterion(logits, y)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
        logits_all.append(logits.detach().float().cpu().numpy()); y_all.append(y.cpu().numpy())
        if train and eval_every_steps and (step + 1) % eval_every_steps == 0:
            eval_hook(step + 1); model.train(True)
    return float(np.mean(losses)), np.concatenate(logits_all), np.concatenate(y_all)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--init", choices=["pretrained", "scratch"], required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    seed = cfg.get("seed", 0)
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    root = cfg["data_root"]
    default_rate = cfg["native_rate"]
    n_patch = (200 * cfg["duration_s"]) // 200                      # = duration_s
    shift = cfg.get("label_shift", 0)
    sets = [CBraModNpy(root, s, default_rate, 200, 200, shift) for s in ("train", "val", "test")]
    bs, nw = cfg.get("batch_size", 64), cfg.get("num_workers", 8)
    train_loader, val_loader, test_loader = (
        DataLoader(d, batch_size=bs, shuffle=(i == 0), drop_last=(i == 0),
                   num_workers=nw, pin_memory=True) for i, d in enumerate(sets))

    ncls = cfg["num_classes"]
    model = CBraModClassifier(ncls, cfg["n_channels"], n_patch,
                              pretrained=(args.init == "pretrained"),
                              dropout=cfg.get("dropout", 0.1)).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[cbramod] init={args.init} patches={n_patch} params={n_par/1e6:.2f}M classes={ncls}")

    # class weights for TUEV (severe imbalance), matching our own runs
    cw = None
    if cfg.get("class_weights"):
        counts = np.bincount(sets[0].labels - shift, minlength=ncls).astype(np.float64)
        cw = torch.FloatTensor(counts.sum() / (ncls * np.clip(counts, 1, None))).to(device)
    criterion = nn.CrossEntropyLoss(weight=cw)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 1e-4),
                           weight_decay=cfg.get("weight_decay", 1e-5))

    key = select_key(ncls, cfg)
    eval_every_steps = cfg.get("eval_every_steps", 0)
    best_val, best_state, best_ep = -np.inf, None, -1

    def mid(step):
        nonlocal best_val, best_state, best_ep
        _, vl, vy = run_epoch(model, val_loader, device, criterion)
        m = compute_metrics(vy, vl, ncls)
        print(f"[cbramod] epoch {ep:3d} step {step:6d} | "
              + " ".join(f"val_{k}={v:.4f}" for k, v in m.items()), flush=True)
        if m[key] > best_val:
            best_val, best_ep = m[key], ep
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})

    for ep in range(cfg.get("epochs", 20)):
        tr, _, _ = run_epoch(model, train_loader, device, criterion, opt,
                             eval_hook=mid, eval_every_steps=eval_every_steps)
        _, vl, vy = run_epoch(model, val_loader, device, criterion)
        m = compute_metrics(vy, vl, ncls)
        print(f"[cbramod] epoch {ep:3d} | loss {tr:.4f} | "
              + " ".join(f"val_{k}={v:.4f}" for k, v in m.items()), flush=True)
        if m[key] > best_val:
            best_val, best_ep = m[key], ep
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})

    _, tl, ty = run_epoch(model, test_loader, device, criterion)
    print("[cbramod] test@last  | " + " ".join(f"{k}={v:.4f}" for k, v in compute_metrics(ty, tl, ncls).items()))
    model.load_state_dict(best_state)
    _, tl, ty = run_epoch(model, test_loader, device, criterion)
    best = compute_metrics(ty, tl, ncls)
    print(f"[cbramod] test@best  | (epoch {best_ep}, val_{key}={best_val:.4f}) "
          + " ".join(f"{k}={v:.4f}" for k, v in best.items()))


if __name__ == "__main__":
    main()
