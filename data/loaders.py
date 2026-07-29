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
from torch.utils.data import Dataset, DataLoader, Sampler, get_worker_info


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
        self.seed = int(seed)
        self._worker_rng = None
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


# --------------------------------------------------------------------------- #
# Pooled multi-dataset corpus for foundation-model pretraining (AGENT.md 13.29)
# --------------------------------------------------------------------------- #
class PooledPretrainDataset(Dataset):
    """Several datasets' TRAIN splits concatenated into one unlabelled corpus.

    This is the corpus the foundation-model pretrain (state 3) runs on. Labels are
    dropped -- the second return value is the member index, kept only so a run can
    report how its batches were composed.

    Two hard constraints decide membership, and they are why this class refuses
    rather than silently coerces:

      * **one montage.** Every member must expose the same 16-channel bipolar
        montage, because SpatialPE's xyz coordinates are looked up per dataset
        (models/montage.py) and a pooled batch has one coordinate table. Sleep-EDF
        (2 channels, a different montage) therefore cannot join this pool -- which
        is consistent with the measured result that crossfreq masking *hurts* on
        Sleep-EDF (sec. 13.28 Link 1), so nothing is lost by excluding it.
      * **one sample rate.** Members are pulled at 200 Hz via each loader's own
        `sampling_rate` argument, so the sinc filterbank's band edges mean the same
        thing in every batch.

    Windows differ in length (TUAB/TUSZ/CHB-MIT 10 s, TUEV 5 s), so every sample is
    **random-cropped** to a common `crop_len`. On the 10 s members the random offset
    doubles as augmentation; on a member already at `crop_len` it is a no-op. A
    member shorter than `crop_len` is an error, not a pad -- padding would inject
    silence the reconstruction target would then have to explain.
    """

    def __init__(self, members, crop_len, seed=0):
        # members: list of (name, torch Dataset yielding (X (C,T), y))
        if not members:
            raise ValueError("pooled corpus is empty")
        self.names = [n for n, _ in members]
        self.sets = [d for _, d in members]
        self.crop_len = int(crop_len)
        self.sizes = [len(d) for d in self.sets]
        self.offsets = np.cumsum([0] + self.sizes)
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return int(self.offsets[-1])

    def composition(self):
        total = len(self)
        return {n: (s, s / total) for n, s in zip(self.names, self.sizes)}

    def __getitem__(self, index):
        m = int(np.searchsorted(self.offsets, index, side="right") - 1)
        X, _ = self.sets[m][index - self.offsets[m]]
        T = X.shape[-1]
        if T < self.crop_len:
            raise ValueError(
                f"pooled member '{self.names[m]}' yields T={T} < crop_len="
                f"{self.crop_len}; lower pool_crop_len or drop that member"
            )
        if T > self.crop_len:
            # Dataset objects are copied into DataLoader workers.  Constructing a
            # Generator in __init__ would clone identical RNG state into every
            # worker, correlating crop offsets.  Seed lazily from PyTorch's unique
            # worker seed instead.
            if self._worker_rng is None:
                info = get_worker_info()
                worker_seed = info.seed if info is not None else torch.initial_seed()
                self._worker_rng = np.random.default_rng(
                    (self.seed + worker_seed) % (2**63 - 1)
                )
            off = int(self._worker_rng.integers(0, T - self.crop_len + 1))
            X = X[..., off:off + self.crop_len]
        return X, m


class DatasetMixtureBatchSampler(Sampler):
    """Homogeneous, temperature-balanced batches for a pooled EEG corpus.

    A normal shuffled ``ConcatDataset`` samples datasets in direct proportion to
    their number of windows.  At foundation-model scale that makes the largest
    corpus define most optimiser steps and lets dataset identity become a useful
    shortcut.  This sampler chooses a dataset with probability

        p(dataset=i) proportional to len(dataset_i) ** alpha

    and then samples one complete batch from that dataset.  ``alpha=1`` is
    proportional sampling, ``alpha=0`` is uniform over datasets, and the default
    foundation recipe uses ``alpha=0.5``.

    Keeping every batch dataset-homogeneous also makes variable channel layouts
    possible: the model can use one coordinate table for the whole batch rather
    than padding unrelated montages into a fake common channel axis.
    """

    def __init__(self, dataset, batch_size, alpha=0.5, n_batches=None,
                 seed=0, rank=0, world_size=1):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.alpha = float(alpha)
        requested_batches = n_batches
        self.n_batches = int(
            requested_batches if requested_batches is not None
            else max(1, len(dataset) // self.batch_size)
        )
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.epoch = 0
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.world_size < 1 or not 0 <= self.rank < self.world_size:
            raise ValueError(f"invalid distributed rank {self.rank}/{self.world_size}")
        if requested_batches is None:
            # Drop at most world_size-1 global batches so every DDP rank executes
            # exactly the same number of collectives. Corpus sizes need not be a
            # lucky multiple of batch_size*world_size.
            self.n_batches -= self.n_batches % self.world_size
            if self.n_batches == 0:
                raise ValueError(
                    "pooled corpus is too small for one batch on every DDP rank"
                )
        elif self.n_batches % self.world_size:
            raise ValueError(
                f"pool_batches_per_epoch={self.n_batches} must be divisible by "
                f"world_size={self.world_size}"
            )
        sizes = np.asarray(dataset.sizes, dtype=np.float64)
        if np.any(sizes <= 0):
            raise ValueError(f"pooled datasets must be non-empty, got {sizes.tolist()}")
        weights = np.power(sizes, self.alpha)
        self.probabilities = weights / weights.sum()

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.n_batches // self.world_size

    def __iter__(self):
        # One deterministic RNG stream defines the global batch schedule.  Each
        # DDP rank consumes a disjoint strided subset, so all ranks take exactly
        # the same number of optimiser steps.
        rng = np.random.default_rng(self.seed + 1_000_003 * self.epoch)
        member_ids = rng.choice(
            len(self.dataset.sets), size=self.n_batches,
            p=self.probabilities,
        )
        for batch_id, member in enumerate(member_ids):
            local = rng.integers(
                0, self.dataset.sizes[int(member)],
                size=self.batch_size,
            )
            if batch_id % self.world_size != self.rank:
                continue
            base = int(self.dataset.offsets[int(member)])
            yield (base + local).tolist()


def build_pretrain_pool(cfg: dict, rank=0, world_size=1):
    """DataLoader over the pooled corpus described by ``cfg['pretrain_pool']``.

        pretrain_pool:
          - {name: tuab,   data_root: .../v3.0.1/edf/processed}
          - {name: tuev,   data_root: .../v2.0.1/edf}
          - {name: tusz,   data_root: .../v2.0.6/edf/processed}
          - {name: chbmit, data_root: .../chb_mit/processed}
        pool_crop_len: 1000        # 5 s at 200 Hz -- the shortest member's window
    """
    rate = cfg.get("pool_sample_rate", 200)
    crop = cfg.get("pool_crop_len", 1000)
    members = []
    for spec in cfg["pretrain_pool"]:
        name, root = spec["name"], spec["data_root"]
        train_name = spec.get("train_name")
        if train_name and name == "tuev":
            members.append((name, TUEVNpyLoader(root, train_name, rate)))
            continue
        if train_name and name in ("tuab", "tuep", "tusz", "chbmit"):
            members.append((name, TUABNpyLoader(root, train_name, rate)))
            continue
        if name == "tuev":
            sets, _ = _tuev_sets(root, rate)
        elif name in ("tuab", "tuep", "tusz", "chbmit"):
            sets = _tuab_sets(root, rate)
        else:
            raise KeyError(f"dataset '{name}' cannot join the pooled corpus "
                           f"(needs the shared 16-ch bipolar montage at {rate} Hz)")
        members.append((name, sets[0]))          # TRAIN split only
    ds = PooledPretrainDataset(members, crop, seed=cfg.get("seed", 0))
    nw = cfg.get("num_workers", 4)
    if "pool_sampling_alpha" in cfg or world_size > 1:
        sampler = DatasetMixtureBatchSampler(
            ds,
            batch_size=cfg.get("batch_size", 32),
            alpha=cfg.get("pool_sampling_alpha", 0.5),
            n_batches=cfg.get("pool_batches_per_epoch"),
            seed=cfg.get("seed", 0),
            rank=rank,
            world_size=world_size,
        )
        loader = DataLoader(
            ds, batch_sampler=sampler, num_workers=nw, pin_memory=True,
            persistent_workers=nw > 0, prefetch_factor=4 if nw > 0 else None,
        )
    else:
        # Backward-compatible path: every existing pooled smoke/config keeps the
        # original proportional, sample-level shuffle unless it opts in.
        loader = DataLoader(
            ds, batch_size=cfg.get("batch_size", 32), shuffle=True, drop_last=True,
            num_workers=nw, pin_memory=True,
            persistent_workers=nw > 0, prefetch_factor=4 if nw > 0 else None,
        )
    return loader, ds


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
