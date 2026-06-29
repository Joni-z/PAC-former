"""Training loop. Config-driven so the only thing that varies across an ablation
run is the YAML (specifically ``mixer``).

    python train.py --config configs/synthetic_mi.yaml
"""

import argparse

import numpy as np
import torch
import torch.nn as nn
import wandb
import yaml
from tqdm import tqdm

from data import build_dataloaders
from eval import compute_metrics
from models.build import build_model


def run_epoch(model, loader, device, criterion, optimizer=None):
    train = optimizer is not None
    model.train(train)
    losses, all_logits, all_y = [], [], []
    for X, y in tqdm(loader, leave=False):
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True).long()
        with torch.set_grad_enabled(train):
            logits = model(X)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        losses.append(loss.item())
        all_logits.append(logits.detach().cpu().numpy())
        all_y.append(y.cpu().numpy())
    return np.mean(losses), np.concatenate(all_logits), np.concatenate(all_y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))

    torch.manual_seed(cfg.get("seed", 0))
    np.random.seed(cfg.get("seed", 0))
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(
        project=cfg.get("wandb_project", "pac-former"),
        name=cfg.get("wandb_run_name", f"{cfg['dataset']}-{cfg['mixer']}"),
        config=cfg,
    )

    train_loader, val_loader, test_loader, class_weights = build_dataloaders(cfg)
    model = build_model(cfg).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.get("lr", 1e-3),
        weight_decay=cfg.get("weight_decay", 1e-5),
    )
    # class_weights is set only for TUEV (severe class imbalance); None elsewhere
    # falls back to nn.CrossEntropyLoss's default uniform weighting.
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{cfg['mixer']}] {n_params/1e6:.2f}M params on {device}")
    wandb.summary["n_params"] = n_params

    eval_every = cfg.get("eval_every", 1)  # run val every N epochs
    best, best_state = -1.0, None
    key = "auroc" if cfg["num_classes"] == 2 else "balanced_accuracy"
    for epoch in range(cfg.get("epochs", 20)):
        tr_loss, *_ = run_epoch(model, train_loader, device, criterion, optimizer)
        log = {"epoch": epoch, "train_loss": tr_loss}

        if (epoch + 1) % eval_every == 0:
            _, val_logits, val_y = run_epoch(model, val_loader, device, criterion)
            m = compute_metrics(val_y, val_logits, cfg["num_classes"])
            log.update({f"val_{k}": v for k, v in m.items()})
            print(f"epoch {epoch:3d} | train_loss {tr_loss:.4f} | val " +
                  " ".join(f"{k}={v:.4f}" for k, v in m.items()))
            if m[key] > best:
                best, best_state = m[key], {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            print(f"epoch {epoch:3d} | train_loss {tr_loss:.4f}")

        wandb.log(log)

    if best_state is not None:
        model.load_state_dict(best_state)
    _, test_logits, test_y = run_epoch(model, test_loader, device, criterion)
    test_m = compute_metrics(test_y, test_logits, cfg["num_classes"])
    print("test | " + " ".join(f"{k}={v:.4f}" for k, v in test_m.items()))
    wandb.log({f"test_{k}": v for k, v in test_m.items()})
    wandb.finish()


if __name__ == "__main__":
    main()
