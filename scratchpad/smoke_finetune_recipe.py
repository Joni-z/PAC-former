"""Fast checks for the foundation downstream finetuning additions."""

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pretrain import Probe, _expand_env, probe_epoch


class FakeMAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_weight = nn.Linear(3, 4)
        self.recon = nn.Sequential(nn.Linear(4, 4))

    def encode(self, x):
        h = self.encoder_weight(x)
        return h[:, None, None, None, :]


def main():
    os.environ["PACFORMER_SMOKE_ROOT"] = "/tmp/data"
    expanded = _expand_env({"root": "${PACFORMER_SMOKE_ROOT}/tuab"})
    assert expanded["root"] == "/tmp/data/tuab"
    print("[1] environment path expansion OK")

    probe = Probe(FakeMAE(), num_classes=2, finetune=True)
    optimizer = torch.optim.AdamW([
        {"params": probe.mae.parameters(), "lr": 1e-4},
        {"params": probe.fc.parameters(), "lr": 1e-3},
    ])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: 1.0 - 0.1 * step
    )
    x = torch.randn(4, 3)
    y = torch.tensor([0, 1, 0, 1])
    before = probe.mae.encoder_weight.weight.detach().clone()
    probe_epoch(
        probe, [(x, y)], "cpu", nn.CrossEntropyLoss(),
        optimizer, scheduler=scheduler,
    )
    assert scheduler.last_epoch == 1
    assert not torch.equal(before, probe.mae.encoder_weight.weight)
    assert optimizer.param_groups[1]["lr"] > optimizer.param_groups[0]["lr"]
    print("[2] discriminative LR + per-step scheduler + encoder update OK")
    print("ALL GREEN")


if __name__ == "__main__":
    main()
