"""BIOT baseline run on OUR pipeline (AGENT.md sec. 13.24 checklist item 2).

Why this exists: BIOT's published numbers come from BIOT's own splits, loaders and
training loop. Quoting them next to ours is not a controlled comparison. This runner
puts BIOT's *own* model through **our** dataloaders, **our** splits and **our**
`eval.compute_metrics`, so the only thing that differs between a BIOT row and one of
our rows is the model (and, optionally, its pretraining).

Two modes, both needed and answering different questions:

  --init pretrained  loads `EEG-PREST-16-channels.ckpt` into the encoder. This is
                     BIOT *state B* -- a real foundation model, pretrained on
                     thousands of hours. It is the TARGET NUMBER our pooled
                     pretrain has to beat. It is NOT comparable to our current
                     single-dataset runs.
  --init scratch     random init. This is *state 1* -- pure architecture, no
                     pretraining. Put beside OUR from-scratch model it is a fair
                     BACKBONE-vs-BACKBONE comparison, and it is valid to make
                     before we have done any pooled pretraining.

Data alignment (sec. 13.26): our TUABLoader/TUEVLoader are verbatim ports of BIOT's,
including the 95th-percentile normalisation, so the tensors BIOT sees here are the
tensors it expects. TUAB/TUEV only -- CHB-MIT is resampled to 200 Hz in our pipeline
vs BIOT's native 256 Hz and is NOT comparable.

Deviation from BIOT's own recipe, stated for the record: BIOT's binary script uses
`n_classes=1` + BCE; we build `n_classes=cfg["num_classes"]` + CrossEntropy so the
loss/metric path is byte-identical to our own runs. Equivalent in expressiveness;
it keeps the pipeline the controlled variable.

Model selection: best epoch by validation, then test ONCE on that checkpoint. Reported
alongside the last-epoch number so the two are never confused.

    python baseline_biot.py --config configs/biot_tuab.yaml --init pretrained
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
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference/BIOT"))

from data import build_dataloaders          # noqa: E402
from eval import compute_metrics, select_key  # noqa: E402
from model.biot import BIOTClassifier       # noqa: E402

CKPT_DIR = "reference/BIOT/pretrained-models"
# Which BIOT checkpoint matches which montage. 16-channel PREST is the one whose
# channel count matches our TUAB/TUEV bipolar montage; the 18-channel checkpoints
# cannot be loaded into a 16-channel encoder.
PRETRAINED = {16: "EEG-PREST-16-channels.ckpt"}

def run_epoch(model, loader, device, criterion, opt=None,
              eval_hook=None, eval_every_steps=0):
    """``eval_hook(step)`` fires every ``eval_every_steps`` optimiser steps.

    Must stay in lockstep with train.py's version (AGENT.md 13.36): if only one
    side of a comparison can find its best checkpoint mid-epoch, the comparison
    measures validation resolution rather than the models.
    """
    train = opt is not None
    model.train(train)
    losses, all_logits, all_y = [], [], []
    for step, (X, y) in enumerate(tqdm(loader, leave=False)):
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).long()
        with torch.set_grad_enabled(train):
            logits = model(X)
            loss = criterion(logits, y)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
        losses.append(loss.item())
        all_logits.append(logits.detach().float().cpu().numpy())
        all_y.append(y.cpu().numpy())
        if train and eval_every_steps and (step + 1) % eval_every_steps == 0:
            eval_hook(step + 1)
            model.train(True)          # the hook validates, which flips to eval
    return float(np.mean(losses)), np.concatenate(all_logits), np.concatenate(all_y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--init", choices=["pretrained", "scratch"], required=True,
                    help="pretrained = BIOT state B (target number); "
                         "scratch = architecture-only control (state 1)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    seed = cfg.get("seed", 0)
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader, class_weights = build_dataloaders(cfg)

    n_ch = cfg["n_channels"]
    model = BIOTClassifier(
        n_classes=cfg["num_classes"],
        n_channels=n_ch,
        n_fft=cfg.get("token_size", 200),          # BIOT default t=200
        hop_length=cfg.get("hop_length", 100),     # BIOT default t-p=100
    )
    if args.init == "pretrained":
        if cfg.get("sampling_rate", cfg["sample_rate"]) != 200:
            raise ValueError("BIOT's released checkpoints are 200 Hz only; "
                             f"config asks for {cfg.get('sampling_rate')} Hz")
        if n_ch not in PRETRAINED:
            raise ValueError(f"no released BIOT checkpoint for {n_ch} channels "
                             f"(have {sorted(PRETRAINED)})")
        path = os.path.join(CKPT_DIR, PRETRAINED[n_ch])
        state = torch.load(path, map_location="cpu")
        model.biot.load_state_dict(state)
        print(f"[biot] loaded pretrained encoder: {path}")
    else:
        print("[biot] random init (architecture-only control, no pretraining)")
    model = model.to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[biot] init={args.init} n_channels={n_ch} params={n_par/1e6:.2f}M "
          f"num_classes={cfg['num_classes']}")

    w = class_weights.to(device) if class_weights is not None else None
    criterion = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 1e-4),
                           weight_decay=cfg.get("weight_decay", 1e-5))

    key = select_key(cfg["num_classes"], cfg)
    best_val, best_state, best_ep = -np.inf, None, -1
    # Mid-epoch validation (13.36). 0 = off => identical to the previous behaviour.
    eval_every_steps = cfg.get("eval_every_steps", 0)

    def mid_epoch_eval(step):
        nonlocal best_val, best_state, best_ep
        _, vl_, vy_ = run_epoch(model, val_loader, device, criterion)
        m_ = compute_metrics(vy_, vl_, cfg["num_classes"])
        print(f"[biot] epoch {ep:3d} step {step:6d} | "
              + " ".join(f"val_{k}={v:.4f}" for k, v in m_.items()), flush=True)
        if m_[key] > best_val:
            best_val, best_ep = m_[key], ep
            best_state = copy.deepcopy({k: v.detach().cpu()
                                        for k, v in model.state_dict().items()})

    for ep in range(cfg.get("epochs", 20)):
        tr_loss, _, _ = run_epoch(model, train_loader, device, criterion, opt,
                                  eval_hook=mid_epoch_eval,
                                  eval_every_steps=eval_every_steps)
        _, vl, vy = run_epoch(model, val_loader, device, criterion)
        m = compute_metrics(vy, vl, cfg["num_classes"])
        print(f"[biot] epoch {ep:3d} | loss {tr_loss:.4f} | "
              + " ".join(f"val_{k}={v:.4f}" for k, v in m.items()), flush=True)
        if m[key] > best_val:
            best_val, best_ep = m[key], ep
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})

    # last-epoch test (what our pretrain.py currently reports) ...
    _, tl, ty = run_epoch(model, test_loader, device, criterion)
    last = compute_metrics(ty, tl, cfg["num_classes"])
    print("[biot] test@last  | " + " ".join(f"{k}={v:.4f}" for k, v in last.items()))

    # ... and the model-selected test, which is what BIOT itself reports.
    model.load_state_dict(best_state)
    _, tl, ty = run_epoch(model, test_loader, device, criterion)
    best = compute_metrics(ty, tl, cfg["num_classes"])
    print(f"[biot] test@best  | (epoch {best_ep}, val_{key}={best_val:.4f}) "
          + " ".join(f"{k}={v:.4f}" for k, v in best.items()))


if __name__ == "__main__":
    main()
