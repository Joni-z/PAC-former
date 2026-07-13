"""Raw TUSZ (TUH EEG Seizure Corpus) .edf -> BIOT-format pickles.

TUSZ is event-level binary classification (seizure vs background), same task
shape as TUEV/CHB-MIT rather than TUEP's whole-session diagnosis label -- each
10s window gets its own label from the corpus's own term-based ``.csv_bi``
annotations (``TERM,start,stop,seiz|bckg,confidence``), which apply to the
whole recording (not per-channel). Reuses TUAB's exact 16-channel bipolar
montage extraction and 200 Hz / 2000-sample (10s) windowing verbatim
(``preprocess_tuab.py``) -- same corpus family, same channel naming
(``EEG FP1-REF`` etc.), same ``01_tcp_ar`` montage standardization.

TUSZ ships its own train/dev/eval split (unlike TUEP) -- mapped here to our
train/val/test, no re-splitting needed and no subject leakage risk.

A window is labeled seiz (1) if it overlaps any seiz interval by >= 50% of
its duration, else bckg (0).

    python scripts/preprocess_tusz.py
"""

import csv
import glob
import os
import pickle
from multiprocessing import Pool

import mne

ROOT = "/scratch/zz5070/PAC-former/tuh_eeg/tuh_eeg_seizure/v2.0.6/edf"
CHANNEL_STD = "01_tcp_ar"
WIN_SAMPLES = 2000   # 10s @ 200Hz, matches TUAB/TUEP
OVERLAP_THRESH = 0.5


def parse_csv_bi(path):
    """Returns list of (start, stop) seizure intervals in seconds."""
    intervals = []
    with open(path) as f:
        rows = [l for l in f if not l.startswith("#")]
    reader = csv.DictReader(rows)
    for row in reader:
        if row["label"].strip() == "seiz":
            intervals.append((float(row["start_time"]), float(row["stop_time"])))
    return intervals


def window_label(win_start_s, win_end_s, seiz_intervals):
    win_dur = win_end_s - win_start_s
    overlap = 0.0
    for s, e in seiz_intervals:
        overlap += max(0.0, min(win_end_s, e) - max(win_start_s, s))
    return 1 if (overlap / win_dur) >= OVERLAP_THRESH else 0


def split_and_dump(params):
    edf_path, dump_folder, tag = params
    csv_path = edf_path[:-4] + ".csv_bi"
    if not os.path.exists(csv_path):
        with open("tusz-process-error-files.txt", "a") as f:
            f.write(edf_path + " (no csv_bi)\n")
        return
    try:
        seiz_intervals = parse_csv_bi(csv_path)
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        raw.resample(200)
        ch_name = raw.ch_names
        raw_data = raw.get_data()
        channeled_data = raw_data.copy()[:16]
        channeled_data[0] = raw_data[ch_name.index("EEG FP1-REF")] - raw_data[ch_name.index("EEG F7-REF")]
        channeled_data[1] = raw_data[ch_name.index("EEG F7-REF")] - raw_data[ch_name.index("EEG T3-REF")]
        channeled_data[2] = raw_data[ch_name.index("EEG T3-REF")] - raw_data[ch_name.index("EEG T5-REF")]
        channeled_data[3] = raw_data[ch_name.index("EEG T5-REF")] - raw_data[ch_name.index("EEG O1-REF")]
        channeled_data[4] = raw_data[ch_name.index("EEG FP2-REF")] - raw_data[ch_name.index("EEG F8-REF")]
        channeled_data[5] = raw_data[ch_name.index("EEG F8-REF")] - raw_data[ch_name.index("EEG T4-REF")]
        channeled_data[6] = raw_data[ch_name.index("EEG T4-REF")] - raw_data[ch_name.index("EEG T6-REF")]
        channeled_data[7] = raw_data[ch_name.index("EEG T6-REF")] - raw_data[ch_name.index("EEG O2-REF")]
        channeled_data[8] = raw_data[ch_name.index("EEG FP1-REF")] - raw_data[ch_name.index("EEG F3-REF")]
        channeled_data[9] = raw_data[ch_name.index("EEG F3-REF")] - raw_data[ch_name.index("EEG C3-REF")]
        channeled_data[10] = raw_data[ch_name.index("EEG C3-REF")] - raw_data[ch_name.index("EEG P3-REF")]
        channeled_data[11] = raw_data[ch_name.index("EEG P3-REF")] - raw_data[ch_name.index("EEG O1-REF")]
        channeled_data[12] = raw_data[ch_name.index("EEG FP2-REF")] - raw_data[ch_name.index("EEG F4-REF")]
        channeled_data[13] = raw_data[ch_name.index("EEG F4-REF")] - raw_data[ch_name.index("EEG C4-REF")]
        channeled_data[14] = raw_data[ch_name.index("EEG C4-REF")] - raw_data[ch_name.index("EEG P4-REF")]
        channeled_data[15] = raw_data[ch_name.index("EEG P4-REF")] - raw_data[ch_name.index("EEG O2-REF")]
    except Exception:
        with open("tusz-process-error-files.txt", "a") as f:
            f.write(edf_path + "\n")
        return

    stem = os.path.splitext(os.path.basename(edf_path))[0]
    total_samples = channeled_data.shape[1]
    RATE = 200

    # (1) Non-overlapping background scan: every 10s window gets its own label.
    n_windows = total_samples // WIN_SAMPLES
    for i in range(n_windows):
        start = i * WIN_SAMPLES
        win_start_s = start / RATE
        win_end_s = (start + WIN_SAMPLES) / RATE
        label = window_label(win_start_s, win_end_s, seiz_intervals)
        dump_path = os.path.join(dump_folder, f"{tag}_{stem}_{i}.pkl")
        pickle.dump(
            {"X": channeled_data[:, start : start + WIN_SAMPLES], "y": label},
            open(dump_path, "wb"),
        )

    # (2) Seizure oversampling (BIOT-style, datasets/CHB-MIT/process2.py): for
    # each seizure interval, densely extract overlapping 10s windows (stride 5s)
    # spanning [start-1s, end+1s], all labeled 1. This counters the extreme
    # class imbalance (~2% positives) at the data level, matching the reference
    # recipe so our numbers are comparable to BIOT's seizure-detection setup.
    for idx, (s, e) in enumerate(seiz_intervals):
        lo = max(0, int((s - 1.0) * RATE))
        hi = min(int((e + 1.0) * RATE), total_samples)
        for start in range(lo, hi, 5 * RATE):
            if start + WIN_SAMPLES > total_samples:
                break
            dump_path = os.path.join(dump_folder, f"{tag}_{stem}_s{idx}_add{start}.pkl")
            pickle.dump(
                {"X": channeled_data[:, start : start + WIN_SAMPLES], "y": 1},
                open(dump_path, "wb"),
            )


if __name__ == "__main__":
    split_map = {"train": "train", "dev": "val", "eval": "test"}
    processed = os.path.join(ROOT, "processed")
    dump_dirs = {out: os.path.join(processed, out) for out in split_map.values()}
    for d in dump_dirs.values():
        os.makedirs(d, exist_ok=True)

    parameters = []
    for corpus_split, out_split in split_map.items():
        edfs = glob.glob(os.path.join(ROOT, corpus_split, "*", "*", CHANNEL_STD, "*.edf"))
        print(f"{corpus_split} -> {out_split}: {len(edfs)} edf files")
        for edf in edfs:
            parameters.append((edf, dump_dirs[out_split], out_split))

    print(f"{len(parameters)} total edf files to process")
    with Pool(processes=int(os.environ.get("SLURM_CPUS_PER_TASK", 8))) as pool:
        pool.map(split_and_dump, parameters)
