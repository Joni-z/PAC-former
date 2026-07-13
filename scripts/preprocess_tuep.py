"""Raw TUEP (TUH EEG Epilepsy Corpus) .edf -> BIOT-format pickles.

TUEP is patient-level binary classification (epilepsy vs no-epilepsy), same
task shape as TUAB (normal/abnormal) -- so this reuses TUAB's exact 16-channel
bipolar montage extraction, 200 Hz resample, and 10 s (2000-sample) windowing
verbatim (``preprocess_tuab.py``), just adapted to TUEP's directory layout:
``00_epilepsy/<patient>/<session>/01_tcp_ar/*.edf`` (label=1) and
``01_no_epilepsy/<patient>/<session>/01_tcp_ar/*.edf`` (label=0), rather than
TUAB's flat train/eval folders. TUEP ships no train/eval split of its own, so
we make one here: patient-disjoint 70/15/15, shuffled with the same seed BIOT
uses for TUAB (12345) for consistency across this project's datasets.

Only the ``01_tcp_ar`` montage is used (matches TUAB's ``CHANNEL_STD``): it
covers 96/100 epilepsy patients and 82/100 no-epilepsy patients, the other
montages (02_tcp_le, 03_tcp_ar_a) cover far fewer and are skipped for
consistency, same reasoning as TUAB/TUEV standardizing on one montage.

    python scripts/preprocess_tuep.py
"""

import glob
import os
import pickle
from multiprocessing import Pool

import numpy as np
import mne

ROOT = "/scratch/zz5070/PAC-former/tuh_eeg/tuh_eeg_epilepsy/v3.1.0"
CHANNEL_STD = "01_tcp_ar"


def split_and_dump(params):
    edf_path, dump_folder, label, tag = params
    try:
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
        with open("tuep-process-error-files.txt", "a") as f:
            f.write(edf_path + "\n")
        return
    stem = os.path.splitext(os.path.basename(edf_path))[0]
    for i in range(channeled_data.shape[1] // 2000):
        dump_path = os.path.join(dump_folder, f"{tag}_{stem}_{i}.pkl")
        pickle.dump(
            {"X": channeled_data[:, i * 2000 : (i + 1) * 2000], "y": label},
            open(dump_path, "wb"),
        )


def patients_with_montage(class_dir):
    """List of (patient_id, [edf paths under 01_tcp_ar across all sessions])."""
    out = {}
    for patient in sorted(os.listdir(class_dir)):
        pdir = os.path.join(class_dir, patient)
        if not os.path.isdir(pdir):
            continue
        edfs = glob.glob(os.path.join(pdir, "*", CHANNEL_STD, "*.edf"))
        if edfs:
            out[patient] = edfs
    return out


def split_patients(patient_ids, a=0.7, b=0.85, seed=12345):
    ids = sorted(patient_ids)
    rng = np.random.RandomState(seed)
    rng.shuffle(ids)
    n = len(ids)
    return ids[: int(n * a)], ids[int(n * a) : int(n * b)], ids[int(n * b) :]


if __name__ == "__main__":
    epilepsy = patients_with_montage(os.path.join(ROOT, "00_epilepsy"))
    no_epilepsy = patients_with_montage(os.path.join(ROOT, "01_no_epilepsy"))
    print(f"epilepsy patients: {len(epilepsy)}, no_epilepsy patients: {len(no_epilepsy)}")

    e_train, e_val, e_test = split_patients(epilepsy.keys())
    n_train, n_val, n_test = split_patients(no_epilepsy.keys())

    processed = os.path.join(ROOT, "processed")
    dump_dirs = {split: os.path.join(processed, split) for split in ("train", "val", "test")}
    for d in dump_dirs.values():
        os.makedirs(d, exist_ok=True)

    parameters = []
    for split, pids in (("train", e_train), ("val", e_val), ("test", e_test)):
        for pid in pids:
            for edf in epilepsy[pid]:
                parameters.append((edf, dump_dirs[split], 1, "epi"))
    for split, pids in (("train", n_train), ("val", n_val), ("test", n_test)):
        for pid in pids:
            for edf in no_epilepsy[pid]:
                parameters.append((edf, dump_dirs[split], 0, "noepi"))

    print(f"{len(parameters)} edf files to process")
    with Pool(processes=int(os.environ.get("SLURM_CPUS_PER_TASK", 8))) as pool:
        pool.map(split_and_dump, parameters)
