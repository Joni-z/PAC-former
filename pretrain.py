"""Masked-reconstruction pretraining + linear probe, in one job.

    python pretrain.py --config configs/pretrain_tuab_crossfreq.yaml

Phase 1: pretrain MAEPretrain (models/pretrain.py) on the train split, labels
ignored. Phase 2: freeze frontend+encoder, mean-pool tokens, train a single
linear layer on the labels, report test metrics. So every pretraining run comes
back with a downstream number to compare mask modes (random vs crossfreq) and
against from-scratch supervised (train.py) -- the whole point of the ablation.
"""

import argparse
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import wandb
import yaml
from tqdm import tqdm

from data import build_dataloaders
from eval import compute_metrics, select_key
from models.pretrain import MAEPretrain


def _expand_env(value):
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    return value


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pretrain_epoch(model, loader, device, opt):
    model.train()
    losses = []
    for X, dataset_idx in tqdm(loader, leave=False):
        X = X.to(device, non_blocking=True)
        loss = model(X, dataset_idx=dataset_idx)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    return float(np.mean(losses))


class Probe(nn.Module):
    """Head on mean-pooled encoder tokens. Two eval protocols (AGENT.md sec 13.24):
      * finetune=False (linear probe): encoder frozen, only `fc` trains -- the
        cheap representation-quality readout used during dev.
      * finetune=True (full finetune): encoder trains end-to-end with the head --
        the protocol BIOT/CBraMod/LaBraM report, needed for a fair Tier-B number.
    """

    def __init__(self, mae, num_classes, finetune=False):
        super().__init__()
        self.mae = mae
        self.finetune = finetune
        self.fc = nn.Linear(mae.recon[0].in_features, num_classes)

    def forward(self, x):
        with torch.set_grad_enabled(self.finetune and torch.is_grad_enabled()):
            h = self.mae.encode(x)                    # (B, C, nb, P, D)
        h = h.mean(dim=(1, 2, 3))                     # (B, D)
        return self.fc(h)


def probe_epoch(model, loader, device, criterion, opt=None,
                eval_hook=None, eval_every_steps=0, scheduler=None):
    """``eval_hook(step)`` fires every ``eval_every_steps`` optimiser steps.

    Same rationale and contract as train.py's version (AGENT.md 13.36): on the
    large datasets one epoch is ~9k updates, every model peaks on validation
    inside the first epoch, and per-epoch validation cannot resolve that. Kept
    here as well as in train.py/baseline_biot.py so the finetune protocol does not
    silently differ from the supervised one. Default 0 = off = unchanged.
    """
    train = opt is not None
    model.train(train)
    if not model.finetune:
        model.mae.eval()                             # linear probe: encoder frozen in eval
    losses, logits_all, y_all = [], [], []
    for step, (X, y) in enumerate(tqdm(loader, leave=False)):
        X, y = X.to(device, non_blocking=True), y.to(device).long()
        with torch.set_grad_enabled(train):
            logits = model(X)
            loss = criterion(logits, y)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
                if scheduler is not None:
                    scheduler.step()
        losses.append(loss.item())
        logits_all.append(logits.detach().cpu().numpy()); y_all.append(y.cpu().numpy())
        if train and eval_every_steps and (step + 1) % eval_every_steps == 0:
            eval_hook(step + 1)
            model.train(True)                        # the hook validates -> eval mode
            if not model.finetune:
                model.mae.eval()                     # keep the frozen encoder frozen
    return float(np.mean(losses)), np.concatenate(logits_all), np.concatenate(y_all)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = _expand_env(yaml.safe_load(open(ap.parse_args().config)))
    set_seed(cfg.get("seed", 0))
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(project=cfg.get("wandb_project", "pac-former"),
               name=cfg.get("wandb_run_name", f"pretrain-{cfg['dataset']}-{cfg.get('mask_mode')}"),
               config=cfg,
               mode="disabled")

    train_loader, val_loader, test_loader, class_weights = build_dataloaders(cfg)
    mae = MAEPretrain(cfg).to(device)
    n_params = sum(p.numel() for p in mae.parameters())
    task = cfg.get("pretrain_task", cfg.get("mask_mode", "mae"))
    print(f"[pretrain {task}] {n_params/1e6:.2f}M params on {device}")
    wandb.summary["n_params"] = n_params

    # ---- Phase 1: masked-reconstruction pretraining ----
    # Three mutually exclusive sources for the pretrained weights:
    #   init_from        -- skip phase 1, load an existing checkpoint. A pooled
    #                       pretrain is expensive; this is how ONE such run gets
    #                       finetuned onto many downstream datasets.
    #   pretrain_pool    -- pretrain on the pooled multi-dataset corpus (state 3,
    #                       the foundation model). Phase 2 still finetunes on
    #                       cfg['dataset'], so the two phases can differ.
    #   (neither)        -- pretrain on cfg['dataset'] itself (state 2). Default,
    #                       so every existing config behaves exactly as before.
    os.makedirs("checkpoints", exist_ok=True)
    ckpt = f"checkpoints/{cfg.get('wandb_run_name', 'pretrain')}.pt"
    init_from = cfg.get("init_from")

    if init_from:
        if cfg.get("pretrain_pool"):
            raise ValueError("set init_from OR pretrain_pool, not both")
        state = torch.load(init_from, map_location=device)
        missing, unexpected = mae.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise ValueError(f"init_from={init_from} does not match this config: "
                             f"missing={list(missing)[:5]} unexpected={list(unexpected)[:5]}")
        print(f"[phase1] SKIPPED -- loaded encoder from {init_from}")
    else:
        if cfg.get("pretrain_pool"):
            from data import build_pretrain_pool
            pt_loader, pool_ds = build_pretrain_pool(cfg)
            comp = ", ".join(f"{n}={c} ({f:.1%})" for n, (c, f) in pool_ds.composition().items())
            print(f"[phase1] POOLED corpus: {len(pool_ds)} windows | {comp}")
            wandb.summary["pool_size"] = len(pool_ds)
            wandb.summary["pool_composition"] = comp
        else:
            pt_loader = train_loader
        opt = torch.optim.AdamW(mae.parameters(), lr=cfg.get("lr", 3e-4),
                                weight_decay=cfg.get("weight_decay", 1e-4))
        for epoch in range(cfg.get("pretrain_epochs", 30)):
            loss = pretrain_epoch(mae, pt_loader, device, opt)
            loss_name = "align_loss" if task == "phase_align" else "recon_loss"
            print(f"[pretrain] epoch {epoch:3d} | {loss_name} {loss:.5f}", flush=True)
            wandb.log({"pretrain_epoch": epoch, loss_name: loss})
            # Pooled runs are long; checkpoint every epoch so a node fault costs
            # one epoch instead of the whole run.
            torch.save(mae.state_dict(), ckpt)
        torch.save(mae.state_dict(), ckpt)
        print(f"saved encoder -> {ckpt}")

    # ---- Phase 2: linear probe (default) or full finetune (probe_mode: finetune) ----
    finetune = cfg.get("probe_mode", "linear") == "finetune"
    probe = Probe(mae, cfg["num_classes"], finetune=finetune).to(device)
    if finetune:
        lr = cfg.get("finetune_lr", 1e-4)
        head_lr = cfg.get("finetune_head_lr")
        if head_lr is None:
            params = probe.parameters()
        else:
            params = [
                {"params": probe.mae.parameters(), "lr": lr},
                {"params": probe.fc.parameters(), "lr": head_lr},
            ]
    else:
        params = probe.fc.parameters()
        lr = cfg.get("probe_lr", 1e-3)
    print(f"[phase2] mode={'finetune' if finetune else 'linear-probe'} lr={lr}")
    if finetune and cfg.get("finetune_optimizer") == "adamw":
        opt = torch.optim.AdamW(
            params, lr=lr,
            weight_decay=cfg.get("finetune_weight_decay", 0.01),
        )
    else:
        opt = torch.optim.Adam(params, lr=lr)
    scheduler = None
    if finetune and cfg.get("finetune_scheduler") == "cosine":
        total_steps = len(train_loader) * cfg.get("probe_epochs", 30)
        warmup_steps = int(cfg.get("finetune_warmup_steps", 0.05 * total_steps))

        def lr_scale(step):
            if warmup_steps and step < warmup_steps:
                return max(step, 1) / warmup_steps
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_scale)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None)
    key = select_key(cfg["num_classes"], cfg)
    best, best_state = -1.0, None
    eval_every_steps = cfg.get("eval_every_steps", 0)      # 0 = off (13.36)

    def mid_epoch_eval(step):
        nonlocal best, best_state
        _, vl_, vy_ = probe_epoch(probe, val_loader, device, criterion)
        m_ = compute_metrics(vy_, vl_, cfg["num_classes"])
        print(f"[probe] epoch {epoch:3d} step {step:6d} | " +
              " ".join(f"val_{k}={v:.4f}" for k, v in m_.items()), flush=True)
        wandb.log({"probe_epoch_frac": epoch + step / max(len(train_loader), 1),
                   **{f"probe_val_{k}": v for k, v in m_.items()}})
        if m_[key] > best:
            best = m_[key]
            best_state = {k: v.detach().cpu().clone()
                          for k, v in probe.state_dict().items()}

    for epoch in range(cfg.get("probe_epochs", 30)):
        tr, *_ = probe_epoch(probe, train_loader, device, criterion, opt,
                             eval_hook=mid_epoch_eval,
                             eval_every_steps=eval_every_steps,
                             scheduler=scheduler)
        _, vl, vy = probe_epoch(probe, val_loader, device, criterion)
        m = compute_metrics(vy, vl, cfg["num_classes"])
        print(f"[probe] epoch {epoch:3d} | loss {tr:.4f} | " +
              " ".join(f"val_{k}={v:.4f}" for k, v in m.items()))
        wandb.log({"probe_epoch": epoch, "probe_train_loss": tr,
                   **{f"probe_val_{k}": v for k, v in m.items()}})
        if m[key] > best:
            best = m[key]
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in probe.state_dict().items()
            }

    if best_state is not None:
        probe.load_state_dict(best_state)
        if cfg.get("finetune_output"):
            output = cfg["finetune_output"]
            os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
            torch.save(best_state, output)
            print(f"saved finetuned model -> {output}")
    _, tl, ty = probe_epoch(probe, test_loader, device, criterion)
    tm = compute_metrics(ty, tl, cfg["num_classes"])
    print("[probe] test | " + " ".join(f"{k}={v:.4f}" for k, v in tm.items()))
    wandb.log({f"probe_test_{k}": v for k, v in tm.items()})
    wandb.finish()


if __name__ == "__main__":
    main()
