"""Datasets and dataloaders.

TUABLoader / TUEVLoader are ported from BIOT (``utils.py``) so preprocessing --
the 0.95-quantile amplitude normalisation, resampling, label convention --
matches the literature exactly. Do not "improve" these; comparability depends
on them being identical to BIOT.

SyntheticPACDataset is ours, for end-to-end pipeline smoke tests before TUAB/
TUEV access arrives: each sample is a known phase->amplitude coupled signal and
the label is whether coupling is present.
"""

import os
import pickle

import numpy as np
import torch
from scipy.signal import resample
from torch.utils.data import Dataset, DataLoader


# --------------------------------------------------------------------------- #
# Ported from BIOT (ycq091044/BIOT, utils.py) -- keep identical.
# --------------------------------------------------------------------------- #
class TUABLoader(Dataset):
    """TUAB binary abnormal/normal. 200 Hz default, 10 s windows."""

    def __init__(self, root, files, sampling_rate=200):
        self.root, self.files = root, files
        self.default_rate, self.sampling_rate = 200, sampling_rate

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["X"]
        if self.sampling_rate != self.default_rate:
            X = resample(X, 10 * self.sampling_rate, axis=-1)
        X = X / (np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True) + 1e-8)
        return torch.FloatTensor(X), sample["y"]


class TUEVLoader(Dataset):
    """TUEV 6-class event. 256 Hz default, 5 s windows; labels 1..6 -> 0..5."""

    def __init__(self, root, files, sampling_rate=200):
        self.root, self.files = root, files
        self.default_rate, self.sampling_rate = 256, sampling_rate

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["signal"]
        if self.sampling_rate != self.default_rate:
            X = resample(X, 5 * self.sampling_rate, axis=-1)
        X = X / (np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True) + 1e-8)
        Y = int(sample["label"][0] - 1)
        return torch.FloatTensor(X), Y


# --------------------------------------------------------------------------- #
# Ours: synthetic PAC classification for early pipeline validation.
# --------------------------------------------------------------------------- #
class SyntheticPACDataset(Dataset):
    """Binary task: signal with theta->gamma PAC (label 1) vs. uncoupled (0)."""

    def __init__(self, n=512, n_channels=4, seq_len=2000, sample_rate=200,
                 f_phase=8.0, f_amp=60.0, seed=0):
        self.n, self.n_channels = n, n_channels
        self.seq_len, self.fs = seq_len, sample_rate
        self.f_phase, self.f_amp = f_phase, f_amp
        self.rng = np.random.default_rng(seed)
        self.labels = self.rng.integers(0, 2, size=n)

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        t = np.arange(self.seq_len) / self.fs
        coupled = bool(self.labels[index])
        X = np.zeros((self.n_channels, self.seq_len), dtype=np.float32)
        for c in range(self.n_channels):
            phase = 2 * np.pi * self.f_phase * t + self.rng.uniform(0, 2 * np.pi)
            low = np.sin(phase)
            mod = (1 + np.sin(phase)) / 2 if coupled else 1.0  # amp gated by low phase
            high = mod * np.sin(2 * np.pi * self.f_amp * t)
            noise = 0.3 * self.rng.standard_normal(self.seq_len)
            X[c] = low + 0.5 * high + noise
        X = X / (np.quantile(np.abs(X), 0.95, axis=-1, keepdims=True) + 1e-8)
        return torch.FloatTensor(X), int(self.labels[index])


def _split_files(root):
    return (
        os.listdir(os.path.join(root, "train")),
        os.listdir(os.path.join(root, "val")),
        os.listdir(os.path.join(root, "test")),
    )


def build_dataloaders(cfg: dict):
    """Return (train, val, test) loaders for the dataset named in ``cfg``."""
    name = cfg["dataset"]
    bs, nw = cfg.get("batch_size", 64), cfg.get("num_workers", 4)
    rate = cfg.get("sampling_rate", cfg["sample_rate"])

    if name == "synthetic":
        common = dict(n_channels=cfg["n_channels"], seq_len=cfg["seq_len"],
                      sample_rate=cfg["sample_rate"])
        sets = [SyntheticPACDataset(n=n, seed=s, **common)
                for n, s in [(cfg.get("n_train", 512), 0),
                             (cfg.get("n_val", 128), 1),
                             (cfg.get("n_test", 128), 2)]]
    elif name in ("tuab", "tuev"):
        Loader = TUABLoader if name == "tuab" else TUEVLoader
        root = cfg["data_root"]
        tr, va, te = _split_files(root)
        sets = [Loader(os.path.join(root, sp), fs, rate)
                for sp, fs in [("train", tr), ("val", va), ("test", te)]]
    else:
        raise KeyError(f"unknown dataset '{name}'")

    return tuple(
        DataLoader(ds, batch_size=bs, shuffle=(i == 0),
                   drop_last=(i == 0), num_workers=nw)
        for i, ds in enumerate(sets)
    )
