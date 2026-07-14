"""Consolidate per-window pkl files (TUAB/TUEP/TUSZ's {"X","y"} format, or
TUEV's {"signal","label"} format) into one (signals, labels) npy pair per
split -- same fix as scripts/consolidate_sleepedf.py, generalized.

TUAB/TUEV/TUEP/TUSZ all still use `TUABLoader`/`TUEVLoader`'s per-file
`pickle.load` (data/loaders.py), the exact pattern that starved the GPU on
Sleep-EDF's ~128k files (~15 it/s ceiling regardless of GPU speed, IO-bound
not compute-bound). These four datasets have 113k-416k files each -- as bad
or worse than Sleep-EDF was. This script packs each split into one
contiguous npy so the OS can page-cache it after epoch 1.

Usage:
    conda activate pacformer
    python scripts/consolidate_pkl_dataset.py --format tuab \\
        --processed_dir tuh_eeg/v3.0.1/edf/processed --splits train val test
    python scripts/consolidate_pkl_dataset.py --format tuev \\
        --processed_dir tuh_eeg/v2.0.1/edf \\
        --splits processed_train processed_eval --out_names train eval
    python scripts/consolidate_pkl_dataset.py --format tuab \\
        --processed_dir tuh_eeg/tuh_eeg_seizure/v2.0.6/edf/processed --splits train val test
"""

import argparse
import os
import pickle

import numpy as np


def consolidate_tuab_format(split_dir, out_dir, out_name):
    """{"X": (16,2000) float64, "y": int} -- TUAB/TUEP/TUSZ all use this.

    Writes the signals array via a disk-backed memmap (np.lib.format.open_memmap)
    instead of building it in RAM first: TUAB's train split alone is
    ~38GB as float32 (409k windows x 16 x 2000 x 4 bytes), which OOM'd a
    32GB job when held fully in memory. memmap keeps peak RAM to ~one
    sample at a time regardless of split size.
    """
    files = sorted(os.listdir(split_dir))
    n = len(files)
    with open(os.path.join(split_dir, files[0]), 'rb') as fh:
        sample_shape = pickle.load(fh)['X'].shape

    signals = np.lib.format.open_memmap(
        os.path.join(out_dir, f'{out_name}_signals.npy'), mode='w+',
        dtype=np.float32, shape=(n, *sample_shape))
    labels = np.empty(n, dtype=np.int64)
    for i, f in enumerate(files):
        with open(os.path.join(split_dir, f), 'rb') as fh:
            sample = pickle.load(fh)
        signals[i] = sample['X'].astype(np.float32)
        labels[i] = int(sample['y'])
    signals.flush()
    np.save(os.path.join(out_dir, f'{out_name}_labels.npy'), labels)
    print(f'{out_name}: {n} windows -> {out_name}_signals.npy {signals.shape} '
          f'+ {out_name}_labels.npy')


def consolidate_tuev_format(split_dir, out_dir, out_name):
    """{"signal": (16,1250) float64, "label": array([1..6])} -- TUEV.
    memmap-backed, see consolidate_tuab_format docstring for why."""
    files = sorted(os.listdir(split_dir))
    n = len(files)
    with open(os.path.join(split_dir, files[0]), 'rb') as fh:
        sample_shape = pickle.load(fh)['signal'].shape

    signals = np.lib.format.open_memmap(
        os.path.join(out_dir, f'{out_name}_signals.npy'), mode='w+',
        dtype=np.float32, shape=(n, *sample_shape))
    labels = np.empty(n, dtype=np.int64)
    for i, f in enumerate(files):
        with open(os.path.join(split_dir, f), 'rb') as fh:
            sample = pickle.load(fh)
        signals[i] = sample['signal'].astype(np.float32)
        labels[i] = int(sample['label'][0])  # keep raw 1..6; Loader does -1
    signals.flush()
    np.save(os.path.join(out_dir, f'{out_name}_labels.npy'), labels)
    print(f'{out_name}: {n} windows -> {out_name}_signals.npy {signals.shape} '
          f'+ {out_name}_labels.npy')


def consolidate_tuev_with_split(root, out_dir):
    """TUEV only: reproduces data/loaders.py::_tuev_sets' subject-based
    train/val split EXACTLY (same seed=4523, same 10% val fraction, same
    "split filename on '_' for subject id" logic) so the consolidated npy
    files represent the identical split already used for every existing
    TUEV result -- consolidating must not silently change which subjects
    are in train vs. val.
    """
    train_files = sorted(os.listdir(os.path.join(root, "processed_train")))
    test_files = sorted(os.listdir(os.path.join(root, "processed_eval")))

    rng = np.random.default_rng(4523)
    # sorted(), not list(set(...)) -- see data/loaders.py::_tuev_sets for why
    # (PYTHONHASHSEED randomizes set iteration order per-process, breaking
    # reproducibility of rng.choice below despite the fixed seed).
    train_sub = sorted(set(f.split("_")[0] for f in train_files))
    val_sub = set(rng.choice(train_sub, size=int(len(train_sub) * 0.1), replace=False))
    train_sub = set(train_sub) - val_sub

    val_files = [f for f in train_files if f.split("_")[0] in val_sub]
    train_files = [f for f in train_files if f.split("_")[0] in train_sub]

    def dump(files, split_dir, out_name):
        n = len(files)
        with open(os.path.join(split_dir, files[0]), 'rb') as fh:
            sample_shape = pickle.load(fh)['signal'].shape
        signals = np.lib.format.open_memmap(
            os.path.join(out_dir, f'{out_name}_signals.npy'), mode='w+',
            dtype=np.float32, shape=(n, *sample_shape))
        labels = np.empty(n, dtype=np.int64)
        for i, f in enumerate(files):
            with open(os.path.join(split_dir, f), 'rb') as fh:
                sample = pickle.load(fh)
            signals[i] = sample['signal'].astype(np.float32)
            labels[i] = int(sample['label'][0])  # raw 1..6, Loader does -1
        signals.flush()
        np.save(os.path.join(out_dir, f'{out_name}_labels.npy'), labels)
        print(f'{out_name}: {n} windows -> {out_name}_signals.npy {signals.shape} '
              f'+ {out_name}_labels.npy')

    train_dir = os.path.join(root, "processed_train")
    dump(train_files, train_dir, 'train')
    dump(val_files, train_dir, 'val')
    dump(test_files, os.path.join(root, "processed_eval"), 'test')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--format', required=True, choices=['tuab', 'tuev', 'tuev-split'])
    ap.add_argument('--processed_dir', required=True)
    ap.add_argument('--splits', nargs='+',
                     help='subdirectory names under processed_dir to read (not used for tuev-split)')
    ap.add_argument('--out_names', nargs='+', default=None,
                     help='output basename per split (defaults to --splits)')
    args = ap.parse_args()

    if args.format == 'tuev-split':
        # writes directly into processed_dir (== root, e.g. tuh_eeg/v2.0.1/edf)
        consolidate_tuev_with_split(args.processed_dir, args.processed_dir)
        return

    assert args.splits, '--splits is required for --format tuab/tuev'
    out_names = args.out_names or args.splits
    assert len(out_names) == len(args.splits)

    fn = consolidate_tuab_format if args.format == 'tuab' else consolidate_tuev_format
    for split, out_name in zip(args.splits, out_names):
        split_dir = os.path.join(args.processed_dir, split)
        fn(split_dir, args.processed_dir, out_name)


if __name__ == '__main__':
    main()
