"""Preprocess Sleep-EDF Cassette into per-epoch pkl files for PAC-Former.

Each output pkl:  {'signal': np.float32 (C, T), 'label': int}
  C = 2 channels (Fpz-Cz, Pz-Oz), T = 3000 (30 s @ 100 Hz)
  label: 0=W  1=N1  2=N2  3=N3(+N4)  4=REM

Uses E0-PSG files (102 recordings, 51 subjects × 2 nights).
Subject-disjoint split by sorted rank (70/15/15):
  train  subjects ranked 0-35  (~70%)
  val    subjects ranked 36-43 (~15%)
  test   subjects ranked 44+   (~15%)

Usage:
    conda activate pacformer
    python scripts/preprocess_sleepedf.py \\
        --raw_dir  sleep_edf/raw \\
        --out_dir  sleep_edf/processed
"""

import argparse
import os
import pickle
import re
from pathlib import Path

import numpy as np
import pyedflib

STAGE_MAP = {
    'Sleep stage W': 0,
    'Sleep stage 1': 1,
    'Sleep stage 2': 2,
    'Sleep stage 3': 3,
    'Sleep stage 4': 3,   # N4 -> N3 (AASM convention)
    'Sleep stage R': 4,
}
EEG_CHANNELS = ['EEG Fpz-Cz', 'EEG Pz-Oz']
EPOCH_SEC = 30
WAKE_MARGIN_MIN = 30      # keep at most this many minutes of wake around sleep


def read_psg(path):
    f = pyedflib.EdfReader(str(path))
    labels = [s.strip() for s in f.getSignalLabels()]
    fs = int(f.getSampleFrequency(0))
    signals = []
    for ch in EEG_CHANNELS:
        idx = labels.index(ch)
        signals.append(f.readSignal(idx).astype(np.float32))
    f.close()
    return np.stack(signals, axis=0), fs   # (C, T), fs


def read_hypnogram(path):
    f = pyedflib.EdfReader(str(path))
    ann = f.readAnnotations()   # (onset_arr, duration_arr, description_arr)
    f.close()
    return ann


def process_recording(psg_path, hyp_path):
    signal, fs = read_psg(psg_path)   # (C, T)
    onsets, durations, descs = read_hypnogram(hyp_path)

    ep_len = EPOCH_SEC * fs
    n_epochs = signal.shape[-1] // ep_len
    labels = np.full(n_epochs, -1, dtype=int)

    for onset, dur, desc in zip(onsets, durations, descs):
        stage = STAGE_MAP.get(desc.strip(), -1)
        if stage < 0:
            continue
        ep_start = int(float(onset) / EPOCH_SEC)
        ep_count = max(1, int(float(dur) / EPOCH_SEC))
        for i in range(ep_start, min(ep_start + ep_count, n_epochs)):
            labels[i] = stage

    # Trim excess wake far from the sleep period
    sleep_mask = labels > 0
    if sleep_mask.any():
        first_s = int(np.where(sleep_mask)[0][0])
        last_s = int(np.where(sleep_mask)[0][-1])
        margin = int(WAKE_MARGIN_MIN * 60 / EPOCH_SEC)
        labels[:max(0, first_s - margin)] = -1
        labels[min(n_epochs, last_s + margin + 1):] = -1

    epochs = []
    for i in range(n_epochs):
        if labels[i] < 0:
            continue
        ep = signal[:, i * ep_len: (i + 1) * ep_len]
        epochs.append({'signal': ep, 'label': labels[i]})
    return epochs


def parse_subject(fname):
    """SC4[SS][N]E0-PSG.edf -> int(SS)"""
    m = re.match(r'SC4(\d{2})\d', fname)
    return int(m.group(1)) if m else -1


def find_hypnogram(raw, psg_name):
    """SC4011E0-PSG -> look for SC4011E*-Hypnogram.edf"""
    prefix = psg_name[:7]   # e.g. 'SC4011E'
    candidates = sorted(raw.glob(f"{prefix}*-Hypnogram.edf"))
    return candidates[0] if candidates else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw_dir', default='sleep_edf/raw')
    ap.add_argument('--out_dir', default='sleep_edf/processed')
    args = ap.parse_args()

    raw = Path(args.raw_dir)
    out = Path(args.out_dir)

    psg_files = sorted(raw.glob('SC*E0-PSG.edf'))
    print(f"Found {len(psg_files)} E0-PSG files in {raw}")

    # Build subject-disjoint split from sorted unique subject IDs
    all_subjects = sorted({parse_subject(p.name) for p in psg_files if parse_subject(p.name) >= 0})
    n = len(all_subjects)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    train_set = set(all_subjects[:n_train])
    val_set = set(all_subjects[n_train:n_train + n_val])
    test_set = set(all_subjects[n_train + n_val:])
    print(f"Subjects: {n} total | train={len(train_set)} val={len(val_set)} test={len(test_set)}")

    for subset in ('train', 'val', 'test'):
        (out / subset).mkdir(parents=True, exist_ok=True)

    def split_name(subj):
        if subj in train_set: return 'train'
        if subj in val_set:   return 'val'
        return 'test'

    totals = {'train': 0, 'val': 0, 'test': 0}
    for psg in psg_files:
        subj = parse_subject(psg.name)
        if subj < 0:
            print(f"  skip {psg.name}: can't parse subject ID")
            continue

        hyp = find_hypnogram(raw, psg.stem.replace('-PSG', ''))
        if hyp is None:
            print(f"  skip {psg.name}: no matching hypnogram")
            continue

        subset = split_name(subj)
        try:
            epochs = process_recording(psg, hyp)
        except Exception as e:
            print(f"  ERROR {psg.name}: {e}")
            continue

        rec_id = psg.stem.replace('-PSG', '')
        for idx, ep in enumerate(epochs):
            pkl_path = out / subset / f"{rec_id}_{idx:04d}.pkl"
            with open(pkl_path, 'wb') as fh:
                pickle.dump(ep, fh)
        totals[subset] += len(epochs)
        print(f"  {psg.name} + {hyp.name}: {len(epochs)} epochs -> {subset}")

    print("\nSplit totals:")
    for k, v in totals.items():
        print(f"  {k}: {v}")
    print(f"Grand total: {sum(totals.values())} epochs")


if __name__ == '__main__':
    main()
