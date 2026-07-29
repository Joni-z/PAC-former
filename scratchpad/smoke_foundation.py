"""CPU smoke tests for the big-cluster foundation-pretrain handoff."""

import tempfile
from pathlib import Path
import sys
import gc

import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.loaders import PooledPretrainDataset, DatasetMixtureBatchSampler
from foundation_pretrain import (
    atomic_torch_save,
    cosine_schedule,
    restore_checkpoint,
    save_checkpoint,
)
from models.pretrain import MAEPretrain


class FakeSet(Dataset):
    def __init__(self, n, channels=16, samples=1000):
        self.n, self.channels, self.samples = n, channels, samples

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        return torch.randn(self.channels, self.samples), 0


def tiny_cfg():
    return {
        "dataset": "tuab",
        "pretrain_pool": [
            {"name": "tuab", "data_root": "unused"},
            {"name": "sleepedf", "data_root": "unused"},
        ],
        "arch": "triaxial",
        "freq_mixer": "attention",
        "spatial_pe": "xyz",
        "band_pe": "index",
        "n_channels": 16,
        "sample_rate": 200,
        "n_bands": 8,
        "d_model": 16,
        "depth": 1,
        "n_heads": 4,
        "dropout": 0.0,
        "kernel_size": 31,
        "patch_len": 200,
        "mask_mode": "mixed",
        "mixed_p": 0.5,
        "mask_ratio": 0.5,
        "recon_loss": "band_balanced_smooth_l1",
    }


def main():
    torch.manual_seed(0)

    # Temperature sampling probabilities and homogeneous batches.
    ds = PooledPretrainDataset(
        [("large", FakeSet(100)), ("small", FakeSet(25))],
        crop_len=1000,
    )
    sampler = DatasetMixtureBatchSampler(
        ds, batch_size=4, alpha=0.5, n_batches=20, seed=3,
    )
    assert torch.allclose(
        torch.tensor(sampler.probabilities),
        torch.tensor([2 / 3, 1 / 3], dtype=torch.float64),
    )
    for batch in sampler:
        member = [int(i >= ds.offsets[1]) for i in batch]
        assert len(set(member)) == 1
    print("[1] sqrt-size sampler probabilities + homogeneous batches OK")

    # DDP rank partition has equal length and a deterministic epoch schedule.
    s0 = DatasetMixtureBatchSampler(ds, 4, n_batches=20, rank=0, world_size=2)
    s1 = DatasetMixtureBatchSampler(ds, 4, n_batches=20, rank=1, world_size=2)
    assert len(list(s0)) == len(list(s1)) == 10
    auto = DatasetMixtureBatchSampler(ds, 4, rank=0, world_size=2)
    assert auto.n_batches == 30 and len(auto) == 15  # 31 -> DDP-safe floor
    old = list(s0)
    s0.set_epoch(1)
    assert old != list(s0)
    print("[2] distributed sampler partition + set_epoch OK")

    # Runtime montage coordinates: the same xyz MLP accepts 16-ch TUAB and
    # 2-ch Sleep-EDF batches. A mixed-dataset batch is rejected.
    model = MAEPretrain(tiny_cfg())
    with torch.no_grad():
        loss16 = model(torch.randn(1, 16, 1000), dataset_idx=torch.zeros(1))
        loss2 = model(torch.randn(1, 2, 1000), dataset_idx=torch.ones(1))
    assert torch.isfinite(loss16) and torch.isfinite(loss2)
    try:
        model._spatial_encoding(16, torch.device("cpu"), torch.tensor([0, 1]))
        raise AssertionError("mixed-dataset batch should fail")
    except ValueError:
        pass
    print("[3] runtime montage selection + mixed-batch guard OK")

    # Band-balanced loss gives equal band weight even with unequal mask counts.
    pred = torch.zeros(1, 1, 2, 3)
    target = torch.tensor([[[[1.0, 1.0, 1.0], [3.0, 0.0, 0.0]]]])
    mask = torch.tensor([[[[True, True, True], [True, False, False]]]])
    got = model._reconstruction_loss(pred, target, mask)
    expected = torch.tensor((0.5 + 2.5) / 2)  # Smooth-L1(1), Smooth-L1(3)
    assert torch.allclose(got, expected)
    print("[4] band-balanced Smooth-L1 reduction OK")

    # Full resumable checkpoint plus plain downstream-compatible state dict.
    del model
    gc.collect()
    checkpoint_model = torch.nn.Linear(2, 2)
    opt = torch.optim.AdamW(checkpoint_model.parameters(), lr=1e-3)
    sched = cosine_schedule(opt, warmup_steps=1, total_steps=4)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    with tempfile.TemporaryDirectory() as directory:
        save_checkpoint(
            directory, checkpoint_model, opt, sched, scaler, tiny_cfg(), 2, 7
        )
        clone = torch.nn.Linear(2, 2)
        opt2 = torch.optim.AdamW(clone.parameters(), lr=1e-3)
        sched2 = cosine_schedule(opt2, 1, 4)
        scaler2 = torch.amp.GradScaler("cuda", enabled=False)
        epoch, step = restore_checkpoint(
            Path(directory) / "latest.pt",
            clone, opt2, sched2, scaler2, torch.device("cpu"),
        )
        assert (epoch, step) == (3, 7)
        plain = torch.load(Path(directory) / "mae_state.pt", map_location="cpu")
        clone.load_state_dict(plain, strict=True)
    print("[5] resumable + downstream-compatible checkpoints OK")

    print("ALL GREEN")


if __name__ == "__main__":
    main()
