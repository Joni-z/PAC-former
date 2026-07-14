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


class TUABNpyLoader(Dataset):
    """Same normalization/output as TUABLoader, but reads a consolidated
    (signals, labels) npy pair instead of one pickle.load per __getitem__.
    Used by TUAB/TUEP/TUSZ, which all share TUABLoader's {"X","y"} pkl format
    and all have 100k-400k+ per-window files -- the same IO-bound bottleneck
    scripts/consolidate_sleepedf.py fixed for Sleep-EDF (~15 it/s ceiling
    regardless of GPU speed). Run scripts/consolidate_pkl_dataset.py once per
    split to produce the npy files this reads.
    """

    def __init__(self, root, split, sampling_rate=200):
        self.signals = np.load(os.path.join(root, f'{split}_signals.npy'), mmap_mode='r')
        self.labels = np.load(os.path.join(root, f'{split}_labels.npy'))
        self.default_rate, self.sampling_rate = 200, sampling_rate

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        X = np.asarray(self.signals[index], dtype=np.float32)
        if self.sampling_rate != self.default_rate:
            X = resample(X, 10 * self.sampling_rate, axis=-1)
        X = X / (np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True) + 1e-8)
        return torch.FloatTensor(X), int(self.labels[index])


class TUEVNpyLoader(Dataset):
    """Same normalization/output as TUEVLoader, npy-backed (see TUABNpyLoader
    docstring). `labels` npy stores raw 1..6 (matches TUEVLoader's `label[0]`
    before the -1 shift, so consolidate_pkl_dataset.py doesn't need to know
    about the shift)."""

    def __init__(self, root, split, sampling_rate=200):
        self.signals = np.load(os.path.join(root, f'{split}_signals.npy'), mmap_mode='r')
        self.labels = np.load(os.path.join(root, f'{split}_labels.npy'))
        self.default_rate, self.sampling_rate = 256, sampling_rate

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        X = np.asarray(self.signals[index], dtype=np.float32)
        if self.sampling_rate != self.default_rate:
            X = resample(X, 5 * self.sampling_rate, axis=-1)
        X = X / (np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True) + 1e-8)
        return torch.FloatTensor(X), int(self.labels[index]) - 1


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


def _tuab_sets(root, rate):
    """TUAB/TUEP/TUSZ (all share this loader/pkl format): preprocessing already
    wrote disjoint train/val/test folders (subject-disjoint split happens at
    preprocessing time, like BIOT).

    Auto-detects a consolidated npy pair per split (see
    scripts/consolidate_pkl_dataset.py, which writes `{split}_signals.npy` /
    `{split}_labels.npy` directly into `root` -- same convention as
    SleepEDFLoader/consolidate_sleepedf.py) and uses the fast mmap-backed
    TUABNpyLoader if present; otherwise falls back to the original
    one-pickle-per-window TUABLoader (slow on these 100k-400k+ file
    datasets, but correct and always available).
    """
    sets = []
    for split in ("train", "val", "test"):
        if os.path.exists(os.path.join(root, f"{split}_signals.npy")):
            sets.append(TUABNpyLoader(root, split, rate))
        else:
            split_dir = os.path.join(root, split)
            sets.append(TUABLoader(split_dir, os.listdir(split_dir), rate))
    return sets


def _tuev_class_weights(root, files, n_classes=6):
    """Inverse-frequency class weights from the TUEV training split.

    TUEV is severely imbalanced (background/eye-movement events dominate;
    spike-wave etc. are rare), which made batch_size=128 training unstable --
    large batches from a skewed distribution give very noisy gradient signal
    for the rare classes. This is a one-time pass over the training pickles
    (label only) before training starts.
    """
    counts = np.zeros(n_classes)
    for f in files:
        with open(os.path.join(root, f), "rb") as fh:
            label = int(pickle.load(fh)["label"][0]) - 1
        counts[label] += 1
    weights = counts.sum() / (n_classes * np.clip(counts, 1, None))
    return torch.FloatTensor(weights)


class SleepEDFLoader(Dataset):
    """Sleep-EDF Cassette, 5-class sleep staging (W/N1/N2/N3/REM).

    Reads from a consolidated (signals, labels) npy pair per split rather
    than one pkl per 30s epoch -- ~128k small random-access file opens per
    epoch starved the GPU (~15 it/s ceiling regardless of GPU speed). mmap
    lets the OS page-cache the whole split (~1.8GB train) after epoch 1.
    Run ``scripts/consolidate_sleepedf.py`` once to produce these files.
    """

    def __init__(self, root, split):
        self.signals = np.load(os.path.join(root, f'{split}_signals.npy'), mmap_mode='r')
        self.labels = np.load(os.path.join(root, f'{split}_labels.npy'))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        X = np.asarray(self.signals[index], dtype=np.float32)
        X = X / (np.quantile(np.abs(X), q=0.95, axis=-1, keepdims=True) + 1e-8)
        return torch.FloatTensor(X), int(self.labels[index])


def _sleepedf_class_weights(labels, n_classes=5):
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    weights = counts.sum() / (n_classes * np.clip(counts, 1, None))
    return torch.FloatTensor(weights)


def _sleepedf_sets(root):
    sets = [SleepEDFLoader(root, subset) for subset in ('train', 'val', 'test')]
    class_weights = _sleepedf_class_weights(sets[0].labels)
    return sets, class_weights


def _tuev_sets(root, rate):
    """TUEV: preprocess_tuev.py only writes processed_train/processed_eval (no
    val split). Val is carved out of train here, by subject, with the same
    seed/fraction/logic as BIOT's run_multiclass_supervised.py
    (prepare_TUEV_dataloader) so the split is identical to the literature.

    Auto-detects consolidated npy files (see
    scripts/consolidate_pkl_dataset.py --format tuev-split, which reproduces
    this exact subject split once and writes train/val/test npy directly
    into `root`) and uses the fast mmap-backed TUEVNpyLoader if present;
    otherwise falls back to the original one-pickle-per-window TUEVLoader.
    """
    if os.path.exists(os.path.join(root, "train_signals.npy")):
        sets = [TUEVNpyLoader(root, split, rate) for split in ("train", "val", "test")]
        counts = np.bincount(sets[0].labels, minlength=7)[1:7].astype(np.float64)
        class_weights = torch.FloatTensor(counts.sum() / (6 * np.clip(counts, 1, None)))
        return sets, class_weights

    rng = np.random.default_rng(4523)
    train_files = os.listdir(os.path.join(root, "processed_train"))
    test_files = os.listdir(os.path.join(root, "processed_eval"))

    # sorted(), not list(set(...)): PYTHONHASHSEED randomizes string-set
    # iteration order per-process, so an unsorted list(set(...)) here made
    # rng.choice below draw a *different* val subject subset on every
    # process launch despite the fixed seed=4523 (found & fixed 2026-07-13,
    # see AGENT.md). sorted() gives a process-independent, truly
    # reproducible order for rng.choice to sample from.
    train_sub = sorted(set(f.split("_")[0] for f in train_files))
    val_sub = set(rng.choice(train_sub, size=int(len(train_sub) * 0.1), replace=False))
    train_sub = set(train_sub) - val_sub

    val_files = [f for f in train_files if f.split("_")[0] in val_sub]
    train_files = [f for f in train_files if f.split("_")[0] in train_sub]

    train_dir = os.path.join(root, "processed_train")
    class_weights = _tuev_class_weights(train_dir, train_files)
    sets = [
        TUEVLoader(train_dir, train_files, rate),
        TUEVLoader(train_dir, val_files, rate),
        TUEVLoader(os.path.join(root, "processed_eval"), test_files, rate),
    ]
    return sets, class_weights


def build_dataloaders(cfg: dict):
    """Return (train, val, test, class_weights) for the dataset named in ``cfg``.

    ``class_weights`` is ``None`` except for TUEV, where it's an inverse-
    frequency weight tensor (see ``_tuev_class_weights``) meant to be passed
    into ``nn.CrossEntropyLoss(weight=...)``.
    """
    name = cfg["dataset"]
    bs, nw = cfg.get("batch_size", 64), cfg.get("num_workers", 4)
    rate = cfg.get("sampling_rate", cfg["sample_rate"])
    class_weights = None

    if name == "synthetic":
        common = dict(n_channels=cfg["n_channels"], seq_len=cfg["seq_len"],
                      sample_rate=cfg["sample_rate"])
        sets = [SyntheticPACDataset(n=n, seed=s, **common)
                for n, s in [(cfg.get("n_train", 512), 0),
                             (cfg.get("n_val", 128), 1),
                             (cfg.get("n_test", 128), 2)]]
    elif name == "tuab":
        sets = _tuab_sets(cfg["data_root"], rate)
    elif name == "tuep":
        sets = _tuab_sets(cfg["data_root"], rate)  # same pkl format/loader as TUAB
    elif name == "tusz":
        sets = _tuab_sets(cfg["data_root"], rate)  # same pkl format/loader as TUAB
    elif name == "chbmit":
        sets = _tuab_sets(cfg["data_root"], rate)  # same pkl format/loader as TUAB
    elif name == "tuev":
        sets, class_weights = _tuev_sets(cfg["data_root"], rate)
    elif name == "sleepedf":
        sets, class_weights = _sleepedf_sets(cfg["data_root"])
    else:
        raise KeyError(f"unknown dataset '{name}'")

    loaders = tuple(
        DataLoader(
            ds, batch_size=bs, shuffle=(i == 0), drop_last=(i == 0),
            num_workers=nw, pin_memory=True,
            persistent_workers=nw > 0, prefetch_factor=4 if nw > 0 else None,
        )
        for i, ds in enumerate(sets)
    )
    return (*loaders, class_weights)
