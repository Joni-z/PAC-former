"""Distributed, resumable foundation pretraining for PAC-Former.

This entry point intentionally performs phase 1 only.  One expensive pooled
checkpoint is produced once, then the existing ``pretrain.py`` finetune path
loads its plain ``mae_state.pt`` on each downstream task.

Single GPU:
    python foundation_pretrain.py --config configs/foundation/pacformer_base.yaml

Multi GPU:
    torchrun --standalone --nproc_per_node=8 foundation_pretrain.py \
        --config configs/foundation/pacformer_base.yaml
"""

import argparse
import json
import math
import os
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel

from data import build_pretrain_pool
from models.pretrain import MAEPretrain


def _expand_env(value):
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    return value


def _merge(base, override):
    merged = dict(base)
    for key, value in override.items():
        if (
            key in merged and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path):
    path = Path(path)
    with open(path) as handle:
        cfg = yaml.safe_load(handle)
    parent = cfg.pop("inherits", None)
    if parent:
        parent_path = Path(parent)
        if not parent_path.is_absolute():
            parent_path = path.parent / parent_path
        cfg = _merge(load_config(parent_path), cfg)
    cfg = _expand_env(cfg)
    if not cfg.get("pretrain_pool"):
        raise ValueError("foundation pretraining requires a non-empty pretrain_pool")
    return cfg


def distributed_context():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("distributed foundation pretraining requires CUDA")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    device = torch.device(
        f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    )
    return rank, local_rank, world_size, device


def set_seed(seed, rank):
    seed = int(seed) + int(rank)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cosine_schedule(optimizer, warmup_steps, total_steps):
    warmup_steps = int(warmup_steps)
    total_steps = int(total_steps)

    def scale(step):
        if warmup_steps and step < warmup_steps:
            return max(step, 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def atomic_torch_save(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def unwrap(model):
    return model.module if isinstance(model, DistributedDataParallel) else model


def save_checkpoint(output_dir, model, optimizer, scheduler, scaler,
                    cfg, epoch, global_step):
    output_dir = Path(output_dir)
    state = {
        "model": unwrap(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "config": cfg,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "torch_rng": torch.get_rng_state(),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
    }
    atomic_torch_save(state, output_dir / "latest.pt")
    # This plain state dict is deliberately compatible with pretrain.py's
    # ``init_from`` path.  Downstream jobs do not need optimizer state.
    atomic_torch_save(state["model"], output_dir / "mae_state.pt")


def restore_checkpoint(path, model, optimizer, scheduler, scaler, device):
    # This is our own full training-state checkpoint (it intentionally contains
    # Python/NumPy RNG state in addition to tensors), so weights_only cannot load
    # it on PyTorch >=2.6.
    state = torch.load(path, map_location=device, weights_only=False)
    unwrap(model).load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state.get("scaler", {}))
    if "torch_rng" in state:
        torch.set_rng_state(state["torch_rng"].cpu())
    if "numpy_rng" in state:
        np.random.set_state(state["numpy_rng"])
    if "python_rng" in state:
        random.setstate(state["python_rng"])
    return int(state["epoch"]) + 1, int(state["global_step"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--override", action="append", default=[],
        help="top-level YAML override, e.g. --override mixed_p=0.75",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    for item in args.override:
        if "=" not in item:
            raise ValueError(f"override must be key=value, got {item!r}")
        key, raw = item.split("=", 1)
        cfg[key] = yaml.safe_load(raw)
    rank, local_rank, world_size, device = distributed_context()
    set_seed(cfg.get("seed", 0), rank)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    loader, pool = build_pretrain_pool(cfg, rank=rank, world_size=world_size)
    model = MAEPretrain(cfg).to(device)
    if world_size > 1:
        # MAEPretrain retains the phase-alignment head so old checkpoints stay
        # compatible; that head is unused in the reconstruction recipe.
        model = DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=True
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.get("lr", 2e-4),
        betas=tuple(cfg.get("betas", (0.9, 0.95))),
        weight_decay=cfg.get("weight_decay", 0.05),
    )
    epochs = int(cfg.get("pretrain_epochs", 50))
    accumulation = int(cfg.get("gradient_accumulation", 1))
    if accumulation < 1:
        raise ValueError("gradient_accumulation must be positive")
    updates_per_epoch = math.ceil(len(loader) / accumulation)
    total_steps = updates_per_epoch * epochs
    scheduler = cosine_schedule(
        optimizer, cfg.get("warmup_steps", min(10_000, total_steps // 10)),
        total_steps,
    )

    precision = cfg.get("precision", "bf16")
    if precision not in ("fp32", "fp16", "bf16"):
        raise ValueError("precision must be fp32/fp16/bf16")
    use_amp = device.type == "cuda" and precision != "fp32"
    amp_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and precision == "fp16")

    output_dir = Path(cfg.get("output_dir", "checkpoints/foundation-pacformer-base"))
    start_epoch, global_step = 0, 0
    resume = args.resume or cfg.get("resume")
    if resume:
        start_epoch, global_step = restore_checkpoint(
            resume, model, optimizer, scheduler, scaler, device
        )

    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        composition = pool.composition()
        n_params = sum(p.numel() for p in unwrap(model).parameters())
        effective_batch = (
            cfg.get("batch_size", 32) * world_size * accumulation
        )
        print(
            f"[foundation] {n_params / 1e6:.2f}M params | {world_size} rank(s) | "
            f"effective batch {effective_batch} | {len(pool):,} windows"
        )
        print("[foundation] corpus " + ", ".join(
            f"{name}={count:,} raw ({fraction:.1%})"
            for name, (count, fraction) in composition.items()
        ))
        sampler = getattr(loader, "batch_sampler", None)
        if hasattr(sampler, "probabilities"):
            print("[foundation] sampled " + ", ".join(
                f"{name}={probability:.1%}"
                for name, probability in zip(pool.names, sampler.probabilities)
            ))
        with open(output_dir / "resolved_config.yaml", "w") as handle:
            yaml.safe_dump(cfg, handle, sort_keys=False)

    log_every = int(cfg.get("log_every_steps", 50))
    clip = float(cfg.get("gradient_clip", 1.0))
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(start_epoch, epochs):
        sampler = getattr(loader, "batch_sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        model.train()
        running_loss = 0.0
        running_microsteps = 0

        for microstep, (signals, dataset_idx) in enumerate(loader):
            signals = signals.to(device, non_blocking=True)
            should_update = (
                (microstep + 1) % accumulation == 0
                or microstep + 1 == len(loader)
            )
            sync = (
                model.no_sync()
                if isinstance(model, DistributedDataParallel) and not should_update
                else nullcontext()
            )
            with sync:
                with torch.autocast(
                    device_type=device.type,
                    dtype=amp_dtype,
                    enabled=use_amp,
                ):
                    loss = model(signals, dataset_idx=dataset_idx)
                    scaled_loss = loss / accumulation
                scaler.scale(scaled_loss).backward()

            running_loss += float(loss.detach())
            running_microsteps += 1
            if not should_update:
                continue

            if clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1

            if global_step % log_every == 0:
                values = torch.tensor(
                    [running_loss, running_microsteps],
                    dtype=torch.float64, device=device,
                )
                if world_size > 1:
                    dist.all_reduce(values, op=dist.ReduceOp.SUM)
                if rank == 0:
                    mean_loss = values[0].item() / values[1].item()
                    record = {
                        "epoch": epoch,
                        "step": global_step,
                        "loss": mean_loss,
                        "lr": scheduler.get_last_lr()[0],
                    }
                    print("[foundation] " + json.dumps(record), flush=True)
                running_loss = 0.0
                running_microsteps = 0

        if world_size > 1:
            dist.barrier()
        if rank == 0:
            save_checkpoint(
                output_dir, model, optimizer, scheduler, scaler,
                cfg, epoch, global_step,
            )
            print(f"[foundation] saved epoch {epoch} -> {output_dir}", flush=True)
        if world_size > 1:
            dist.barrier()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
