"""Training loop. Config-driven so the only thing that varies across an ablation
run is the YAML (specifically ``mixer``).

    python train.py --config configs/synthetic_mi.yaml
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
from models.build import build_model


def run_epoch(model, loader, device, criterion, optimizer=None, forward_kwargs=None,
              aux_weight=0.0):
    train = optimizer is not None
    model.train(train)
    losses, aux_losses, all_logits, all_y = [], [], [], []
    for X, y in tqdm(loader, leave=False):
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True).long()
        with torch.set_grad_enabled(train):
            logits = model(X, **(forward_kwargs or {}))
            loss = criterion(logits, y)
            if train and aux_weight > 0:
                # crossfreq-reconstruction auxiliary (build.py sec. 13.15): a second
                # masked pass through the shared frontend/encoder, added to CE so
                # the objective forces low->high coupling info to be preserved.
                aux = model.crossfreq_aux_loss(X)
                loss = loss + aux_weight * aux
                aux_losses.append(aux.item())
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        losses.append(loss.item())
        all_logits.append(logits.detach().cpu().numpy())
        all_y.append(y.cpu().numpy())
    mean_aux = float(np.mean(aux_losses)) if aux_losses else 0.0
    return np.mean(losses), np.concatenate(all_logits), np.concatenate(all_y), mean_aux


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))

    seed = cfg.get("seed", 0)
    random.seed(seed)          # augment.py picks the per-batch augmentation via
                                # random.randint -- this module's seed is NOT set
                                # by torch/numpy seeding and was the main source
                                # of run-to-run variance before this fix.
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
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
    patience = cfg.get("patience", 0)      # 0 = no early stopping
    best, best_state, since_best = -1.0, None, 0
    key = "auroc" if cfg["num_classes"] == 2 else "balanced_accuracy"
    aux_weight = cfg.get("aux_recon_weight", 0.0)
    for epoch in range(cfg.get("epochs", 20)):
        tr_loss, _, _, tr_aux = run_epoch(
            model, train_loader, device, criterion, optimizer, aux_weight=aux_weight
        )
        log = {"epoch": epoch, "train_loss": tr_loss}
        if aux_weight > 0:
            log["train_aux_recon"] = tr_aux
        # MI diagnostic: the coupling mixer's learned pac_scale per layer. In v2
        # (tri-axial) the mixer lives at block.freq; in v1 (flat) at block.mixer.
        # Watching pac_scale is how we see whether the now time-resolved coupling
        # is actually being used (contrast v1, where it collapsed to ~0 on
        # CHB-MIT because the coupling was averaged into mush -- AGENT.md 9.17).
        for i, block in enumerate(model.encoder.blocks):
            mix = getattr(block, "mixer", None)
            if mix is None:
                mix = getattr(block, "freq", None)
            if mix is None:
                continue
            if hasattr(mix, "last_gate"):
                log[f"gate/layer{i}"] = mix.last_gate
            if hasattr(mix, "pac_scale"):
                log[f"pac_scale/layer{i}"] = mix.pac_scale.item()

        if (epoch + 1) % eval_every == 0:
            _, val_logits, val_y, _ = run_epoch(model, val_loader, device, criterion)
            m = compute_metrics(val_y, val_logits, cfg["num_classes"])
            log.update({f"val_{k}": v for k, v in m.items()})
            print(f"epoch {epoch:3d} | train_loss {tr_loss:.4f} | val " +
                  " ".join(f"{k}={v:.4f}" for k, v in m.items()))
            if m[key] > best:
                best, best_state, since_best = m[key], {k: v.cpu() for k, v in model.state_dict().items()}, 0
            else:
                since_best += 1
                if patience and since_best >= patience:
                    print(f"early stop at epoch {epoch} (no val {key} gain for {patience} evals)")
                    wandb.log(log)
                    break

        wandb.log(log)

    if best_state is not None:
        model.load_state_dict(best_state)
    _, test_logits, test_y, _ = run_epoch(model, test_loader, device, criterion)
    test_m = compute_metrics(test_y, test_logits, cfg["num_classes"])
    print("test | " + " ".join(f"{k}={v:.4f}" for k, v in test_m.items()))
    wandb.log({f"test_{k}": v for k, v in test_m.items()})
    if cfg.get("arch") == "triaxial" and cfg.get("freq_mixer") == "phase":
        for phase_mode in ("magnitude", "scramble"):
            _, ab_logits, ab_y, _ = run_epoch(
                model, test_loader, device, criterion,
                forward_kwargs={"phase_mode": phase_mode},
            )
            ab_m = compute_metrics(ab_y, ab_logits, cfg["num_classes"])
            print(f"test_{phase_mode} | " + " ".join(
                f"{k}={v:.4f}" for k, v in ab_m.items()
            ))
            wandb.log({f"test_{phase_mode}_{k}": v for k, v in ab_m.items()})
    wandb.finish()


if __name__ == "__main__":
    main()
