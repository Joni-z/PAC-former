"""Consolidate per-epoch Sleep-EDF pkl files into one (signals, labels) npy
pair per split.

The original preprocess_sleepedf.py writes one pkl file per 30s epoch
(~128k files total). Reading that many small files with random access on a
shared filesystem is IO-bound and starves the GPU (~120 files/s single
process, ~1000 files/s across 8 dataloader workers -> ~15 it/s ceiling at
batch_size=64, regardless of GPU speed). Packing each split into one
contiguous array file lets the OS page-cache the whole thing (~1.8GB train
split) after the first epoch, removing the per-sample file-open cost.

Usage:
    conda activate pacformer
    python scripts/consolidate_sleepedf.py \\
        --processed_dir sleep_edf/processed
"""

import argparse
import os
import pickle

import numpy as np


def consolidate(split_dir, out_dir, split):
    files = sorted(os.listdir(split_dir))
    n = len(files)
    signals = None
    labels = np.empty(n, dtype=np.int64)

    for i, f in enumerate(files):
        with open(os.path.join(split_dir, f), 'rb') as fh:
            sample = pickle.load(fh)
        sig = sample['signal'].astype(np.float32)
        if signals is None:
            signals = np.empty((n, *sig.shape), dtype=np.float32)
        signals[i] = sig
        labels[i] = int(sample['label'])

    np.save(os.path.join(out_dir, f'{split}_signals.npy'), signals)
    np.save(os.path.join(out_dir, f'{split}_labels.npy'), labels)
    print(f'{split}: {n} epochs -> {split}_signals.npy {signals.shape} + {split}_labels.npy')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--processed_dir', default='sleep_edf/processed')
    args = ap.parse_args()

    for split in ('train', 'val', 'test'):
        consolidate(os.path.join(args.processed_dir, split), args.processed_dir, split)


if __name__ == '__main__':
    main()
