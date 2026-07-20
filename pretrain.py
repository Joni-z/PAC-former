"""Masked-reconstruction pretraining + linear probe, in one job.

    python pretrain.py --config configs/pretrain_tuab_crossfreq.yaml

Phase 1: pretrain MAEPretrain (models/pretrain.py) on the train split, labels
ignored. Phase 2: freeze frontend+encoder, mean-pool tokens, train a single
linear layer on the labels, report test metrics. So every pretraining run comes
back with a downstream number to compare mask modes (random vs crossfreq) and
against from-scratch supervised (train.py) -- the whole point of the ablation.
"""

import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import wandb
import yaml
from tqdm import tqdm

from data import build_dataloaders
from eval import compute_metrics
from models.pretrain import MAEPretrain


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pretrain_epoch(model, loader, device, opt):
    model.train()
    losses = []
    for X, _ in tqdm(loader, leave=False):
        X = X.to(device, non_blocking=True)
        loss = model(X)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    return float(np.mean(losses))


class Probe(nn.Module):
    """Frozen encoder + a single trainable linear layer on mean-pooled tokens."""

    def __init__(self, mae, num_classes):
        super().__init__()
        self.mae = mae
        self.fc = nn.Linear(mae.recon[0].in_features, num_classes)

    def forward(self, x):
        with torch.no_grad():
            h = self.mae.encode(x)                    # (B, C, nb, P, D)
        h = h.mean(dim=(1, 2, 3))                     # (B, D)
        return self.fc(h)


def probe_epoch(model, loader, device, criterion, opt=None):
    train = opt is not None
    model.train(train); model.mae.eval()             # keep encoder in eval always
    losses, logits_all, y_all = [], [], []
    for X, y in tqdm(loader, leave=False):
        X, y = X.to(device, non_blocking=True), y.to(device).long()
        with torch.set_grad_enabled(train):
            logits = model(X)
            loss = criterion(logits, y)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
        logits_all.append(logits.detach().cpu().numpy()); y_all.append(y.cpu().numpy())
    return float(np.mean(losses)), np.concatenate(logits_all), np.concatenate(y_all)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = yaml.safe_load(open(ap.parse_args().config))
    set_seed(cfg.get("seed", 0))
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(project=cfg.get("wandb_project", "pac-former"),
               name=cfg.get("wandb_run_name", f"pretrain-{cfg['dataset']}-{cfg.get('mask_mode')}"),
               config=cfg)

    train_loader, val_loader, test_loader, class_weights = build_dataloaders(cfg)
    mae = MAEPretrain(cfg).to(device)
    n_params = sum(p.numel() for p in mae.parameters())
    task = cfg.get("pretrain_task", cfg.get("mask_mode", "mae"))
    print(f"[pretrain {task}] {n_params/1e6:.2f}M params on {device}")
    wandb.summary["n_params"] = n_params

    # ---- Phase 1: masked-reconstruction pretraining ----
    opt = torch.optim.AdamW(mae.parameters(), lr=cfg.get("lr", 3e-4),
                            weight_decay=cfg.get("weight_decay", 1e-4))
    for epoch in range(cfg.get("pretrain_epochs", 30)):
        loss = pretrain_epoch(mae, train_loader, device, opt)
        loss_name = "align_loss" if task == "phase_align" else "recon_loss"
        print(f"[pretrain] epoch {epoch:3d} | {loss_name} {loss:.5f}")
        wandb.log({"pretrain_epoch": epoch, loss_name: loss})

    ckpt = f"checkpoints/{cfg.get('wandb_run_name', 'pretrain')}.pt"
    import os; os.makedirs("checkpoints", exist_ok=True)
    torch.save(mae.state_dict(), ckpt)
    print(f"saved encoder -> {ckpt}")

    # ---- Phase 2: linear probe on the labels ----
    probe = Probe(mae, cfg["num_classes"]).to(device)
    opt = torch.optim.Adam(probe.fc.parameters(), lr=cfg.get("probe_lr", 1e-3))
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None)
    key = "auroc" if cfg["num_classes"] == 2 else "balanced_accuracy"
    best, best_state = -1.0, None
    for epoch in range(cfg.get("probe_epochs", 30)):
        tr, *_ = probe_epoch(probe, train_loader, device, criterion, opt)
        _, vl, vy = probe_epoch(probe, val_loader, device, criterion)
        m = compute_metrics(vy, vl, cfg["num_classes"])
        print(f"[probe] epoch {epoch:3d} | loss {tr:.4f} | " +
              " ".join(f"val_{k}={v:.4f}" for k, v in m.items()))
        wandb.log({"probe_epoch": epoch, "probe_train_loss": tr,
                   **{f"probe_val_{k}": v for k, v in m.items()}})
        if m[key] > best:
            best, best_state = m[key], {k: v.cpu() for k, v in probe.state_dict().items()}

    if best_state is not None:
        probe.load_state_dict(best_state)
    _, tl, ty = probe_epoch(probe, test_loader, device, criterion)
    tm = compute_metrics(ty, tl, cfg["num_classes"])
    print("[probe] test | " + " ".join(f"{k}={v:.4f}" for k, v in tm.items()))
    wandb.log({f"probe_test_{k}": v for k, v in tm.items()})
    wandb.finish()


if __name__ == "__main__":
    main()
