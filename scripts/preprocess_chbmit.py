"""Raw CHB-MIT .edf -> BIOT-format pickles, event-level seizure detection.

Same task shape as TUSZ (preprocess_tusz.py): event-level binary
seizure/background labels from per-file annotations, BIOT-style seizure-window
oversampling for the extreme class imbalance, patient-disjoint split. Unlike
TUAB/TUEV/TUEP/TUSZ, CHB-MIT ships its signals *already* in a 16-channel
bipolar montage (channel names like "FP1-F7" directly, no referential-to-bipolar
differencing needed) -- confirmed by inspecting the raw edf channel names, not
assumed. Resampled to 200Hz to match the rest of this project's convention
(BIOT's own CHB-MIT recipe, reference/BIOT/datasets/CHB-MIT/process*.py, stays
at native 256Hz and needs a separate manual per-patient "clean" pass because
some sessions have extra/reordered channels -- we sidestep that by reading
channel names directly per file and skipping files missing our 16 target
channels, logged, rather than hand-curating parameters per patient).

Patient split matches BIOT's own CHB-MIT split (reference/BIOT/datasets/CHB-MIT/process2.py)
for comparability: test={chb23,chb24}, val={chb21,chb22}, train=the rest.

Seizure times parsed from each patient's chbNN-summary.txt (handles both the
single-seizure "Seizure Start/End Time:" and multi-seizure numbered
"Seizure N Start/End Time:" formats).

    python scripts/preprocess_chbmit.py
"""

import glob
import os
import pickle
import re
from multiprocessing import Pool

import mne
import numpy as np

ROOT = "/scratch/zz5070/PAC-former/chb_mit/raw"
WIN_SAMPLES = 2000   # 10s @ 200Hz, matches TUAB/TUEP/TUSZ
OVERLAP_THRESH = 0.5
TARGET_RATE = 200

TARGET_CHANNELS = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
]

TEST_PATS = {"chb23", "chb24"}
VAL_PATS = {"chb21", "chb22"}


def parse_summary(summary_path):
    """Returns {filename: [(start_s, end_s), ...]} for every file with >=1 seizure."""
    text = open(summary_path, "r", errors="ignore").read()
    blocks = text.split("File Name: ")[1:]
    out = {}
    for block in blocks:
        fname = block.split()[0]
        starts = [int(x) for x in re.findall(r"Seizure\s*\d*\s*Start Time:\s*(\d+)", block)]
        ends = [int(x) for x in re.findall(r"Seizure\s*\d*\s*End Time:\s*(\d+)", block)]
        if starts:
            out[fname] = list(zip(starts, ends))
    return out


def window_label(win_start_s, win_end_s, seiz_intervals):
    win_dur = win_end_s - win_start_s
    overlap = 0.0
    for s, e in seiz_intervals:
        overlap += max(0.0, min(win_end_s, e) - max(win_start_s, s))
    return 1 if (overlap / win_dur) >= OVERLAP_THRESH else 0


def resolve_channel(ch_names, target):
    """CHB-MIT has a duplicated 'T8-P8' channel in most recordings; mne
    renames the second occurrence to 'T8-P8-1' (first stays 'T8-P8' or
    becomes 'T8-P8-0' depending on mne version). Try exact name, then the
    '-0' suffix, else None (caller skips the file)."""
    if target in ch_names:
        return target
    if f"{target}-0" in ch_names:
        return f"{target}-0"
    return None


def split_and_dump(params):
    edf_path, dump_folder, seiz_intervals, tag = params
    stem = os.path.splitext(os.path.basename(edf_path))[0]
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        ch_names = raw.ch_names
        resolved = [resolve_channel(ch_names, c) for c in TARGET_CHANNELS]
        if any(c is None for c in resolved):
            missing = [t for t, r in zip(TARGET_CHANNELS, resolved) if r is None]
            with open("chbmit-process-error-files.txt", "a") as f:
                f.write(f"{edf_path} missing channels: {missing}\n")
            return
        raw.pick(resolved)
        raw.reorder_channels(resolved)
        raw.resample(TARGET_RATE)
        data = raw.get_data()  # (16, T) at TARGET_RATE
    except Exception as e:
        with open("chbmit-process-error-files.txt", "a") as f:
            f.write(f"{edf_path} ({e})\n")
        return

    total_samples = data.shape[1]

    # (1) Non-overlapping background scan
    n_windows = total_samples // WIN_SAMPLES
    for i in range(n_windows):
        start = i * WIN_SAMPLES
        win_start_s = start / TARGET_RATE
        win_end_s = (start + WIN_SAMPLES) / TARGET_RATE
        label = window_label(win_start_s, win_end_s, seiz_intervals)
        dump_path = os.path.join(dump_folder, f"{tag}_{stem}_{i}.pkl")
        pickle.dump(
            {"X": data[:, start:start + WIN_SAMPLES], "y": label},
            open(dump_path, "wb"),
        )

    # (2) Seizure oversampling (BIOT-style, same recipe as preprocess_tusz.py):
    # dense overlapping windows (stride 5s) spanning [start-1s, end+1s].
    for idx, (s, e) in enumerate(seiz_intervals):
        lo = max(0, int((s - 1.0) * TARGET_RATE))
        hi = min(int((e + 1.0) * TARGET_RATE), total_samples)
        for start in range(lo, hi, 5 * TARGET_RATE):
            if start + WIN_SAMPLES > total_samples:
                break
            dump_path = os.path.join(dump_folder, f"{tag}_{stem}_s{idx}_add{start}.pkl")
            pickle.dump(
                {"X": data[:, start:start + WIN_SAMPLES], "y": 1},
                open(dump_path, "wb"),
            )


if __name__ == "__main__":
    processed = os.path.join(os.path.dirname(ROOT), "processed")
    dump_dirs = {split: os.path.join(processed, split) for split in ("train", "val", "test")}
    for d in dump_dirs.values():
        os.makedirs(d, exist_ok=True)

    patients = sorted(d for d in os.listdir(ROOT) if d.startswith("chb") and os.path.isdir(os.path.join(ROOT, d)))

    parameters = []
    for pat in patients:
        split = "test" if pat in TEST_PATS else "val" if pat in VAL_PATS else "train"
        summary = os.path.join(ROOT, pat, f"{pat}-summary.txt")
        if not os.path.exists(summary):
            with open("chbmit-process-error-files.txt", "a") as f:
                f.write(f"{pat}: no summary.txt\n")
            continue
        seiz_by_file = parse_summary(summary)
        for edf in sorted(glob.glob(os.path.join(ROOT, pat, "*.edf"))):
            fname = os.path.basename(edf)
            intervals = seiz_by_file.get(fname, [])
            parameters.append((edf, dump_dirs[split], intervals, split))

    print(f"{len(patients)} patients, {len(parameters)} edf files to process "
          f"(test={sorted(TEST_PATS)}, val={sorted(VAL_PATS)})")
    with Pool(processes=int(os.environ.get("SLURM_CPUS_PER_TASK", 8))) as pool:
        pool.map(split_and_dump, parameters)
