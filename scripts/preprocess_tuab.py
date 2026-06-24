"""Raw TUAB .edf -> BIOT-format pickles, ported verbatim from BIOT
(``datasets/TUAB/process.py``) with only the root path adjusted to our local
copy of the corpus. Do not change the channel montage, resampling, or window
logic -- that is what keeps `data/loaders.py::TUABLoader` output comparable to
published BIOT numbers.

    python scripts/preprocess_tuab.py
"""

import os
import pickle
from multiprocessing import Pool

import numpy as np
import mne

ROOT = "/scratch/zz5070/PAC-former/tuh_eeg/v3.0.1/edf"
CHANNEL_STD = "01_tcp_ar"


def split_and_dump(params):
    fetch_folder, sub, dump_folder, label = params
    for file in os.listdir(fetch_folder):
        if sub in file:
            print("process", file)
            file_path = os.path.join(fetch_folder, file)
            raw = mne.io.read_raw_edf(file_path, preload=True)
            raw.resample(200)
            ch_name = raw.ch_names
            raw_data = raw.get_data()
            channeled_data = raw_data.copy()[:16]
            try:
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
                with open("tuab-process-error-files.txt", "a") as f:
                    f.write(file + "\n")
                continue
            for i in range(channeled_data.shape[1] // 2000):
                dump_path = os.path.join(dump_folder, file.split(".")[0] + "_" + str(i) + ".pkl")
                pickle.dump(
                    {"X": channeled_data[:, i * 2000 : (i + 1) * 2000], "y": label},
                    open(dump_path, "wb"),
                )


if __name__ == "__main__":
    np.random.seed(12345)  # match BIOT's run_binary_supervised.py seed

    train_val_abnormal = os.path.join(ROOT, "train", "abnormal", CHANNEL_STD)
    train_val_a_sub = list(set(item.split("_")[0] for item in os.listdir(train_val_abnormal)))
    np.random.shuffle(train_val_a_sub)
    train_a_sub, val_a_sub = (
        train_val_a_sub[: int(len(train_val_a_sub) * 0.8)],
        train_val_a_sub[int(len(train_val_a_sub) * 0.8) :],
    )

    train_val_normal = os.path.join(ROOT, "train", "normal", CHANNEL_STD)
    train_val_n_sub = list(set(item.split("_")[0] for item in os.listdir(train_val_normal)))
    np.random.shuffle(train_val_n_sub)
    train_n_sub, val_n_sub = (
        train_val_n_sub[: int(len(train_val_n_sub) * 0.8)],
        train_val_n_sub[int(len(train_val_n_sub) * 0.8) :],
    )

    test_abnormal = os.path.join(ROOT, "eval", "abnormal", CHANNEL_STD)
    test_a_sub = list(set(item.split("_")[0] for item in os.listdir(test_abnormal)))
    test_normal = os.path.join(ROOT, "eval", "normal", CHANNEL_STD)
    test_n_sub = list(set(item.split("_")[0] for item in os.listdir(test_normal)))

    train_dump_folder = os.path.join(ROOT, "processed", "train")
    val_dump_folder = os.path.join(ROOT, "processed", "val")
    test_dump_folder = os.path.join(ROOT, "processed", "test")
    for d in (train_dump_folder, val_dump_folder, test_dump_folder):
        os.makedirs(d, exist_ok=True)

    parameters = []
    for sub in train_a_sub:
        parameters.append([train_val_abnormal, sub, train_dump_folder, 1])
    for sub in train_n_sub:
        parameters.append([train_val_normal, sub, train_dump_folder, 0])
    for sub in val_a_sub:
        parameters.append([train_val_abnormal, sub, val_dump_folder, 1])
    for sub in val_n_sub:
        parameters.append([train_val_normal, sub, val_dump_folder, 0])
    for sub in test_a_sub:
        parameters.append([test_abnormal, sub, test_dump_folder, 1])
    for sub in test_n_sub:
        parameters.append([test_normal, sub, test_dump_folder, 0])

    with Pool(processes=int(os.environ.get("SLURM_CPUS_PER_TASK", 8))) as pool:
        pool.map(split_and_dump, parameters)
