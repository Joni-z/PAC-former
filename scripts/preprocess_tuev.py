"""Raw TUEV .edf/.rec -> BIOT-format pickles, ported verbatim from BIOT
(``datasets/TUEV/process.py``; itself adapted from
github.com/Abhishaike/EEG_Event_Classification), root path adjusted only.
Do not change the channel montage or event-windowing logic -- comparability to
published BIOT numbers depends on it matching exactly.

    python scripts/preprocess_tuev.py
"""

import os

import mne
import numpy as np
import pickle
from tqdm import tqdm

ROOT = "/scratch/zz5070/PAC-former/tuh_eeg/v2.0.1/edf"


def BuildEvents(signals, times, EventData):
    numEvents, _ = EventData.shape  # numEvents = rows of the .rec file
    fs = 250.0
    numChan, _ = signals.shape
    features = np.zeros([numEvents, numChan, int(fs) * 5])
    offending_channel = np.zeros([numEvents, 1])
    labels = np.zeros([numEvents, 1])
    offset = signals.shape[1]
    signals = np.concatenate([signals, signals, signals], axis=1)
    for i in range(numEvents):
        chan = int(EventData[i, 0])
        start = np.where(times >= EventData[i, 1])[0][0]
        end = np.where(times >= EventData[i, 2])[0][0]
        features[i, :] = signals[:, offset + start - 2 * int(fs) : offset + end + 2 * int(fs)]
        offending_channel[i, :] = int(chan)
        labels[i, :] = int(EventData[i, 3])
    return [features, offending_channel, labels]


def convert_signals(signals, Rawdata):
    signal_names = {k: v for v, k in enumerate(Rawdata.info["ch_names"])}
    return np.vstack((
        signals[signal_names["EEG FP1-REF"]] - signals[signal_names["EEG F7-REF"]],
        signals[signal_names["EEG F7-REF"]] - signals[signal_names["EEG T3-REF"]],
        signals[signal_names["EEG T3-REF"]] - signals[signal_names["EEG T5-REF"]],
        signals[signal_names["EEG T5-REF"]] - signals[signal_names["EEG O1-REF"]],
        signals[signal_names["EEG FP2-REF"]] - signals[signal_names["EEG F8-REF"]],
        signals[signal_names["EEG F8-REF"]] - signals[signal_names["EEG T4-REF"]],
        signals[signal_names["EEG T4-REF"]] - signals[signal_names["EEG T6-REF"]],
        signals[signal_names["EEG T6-REF"]] - signals[signal_names["EEG O2-REF"]],
        signals[signal_names["EEG FP1-REF"]] - signals[signal_names["EEG F3-REF"]],
        signals[signal_names["EEG F3-REF"]] - signals[signal_names["EEG C3-REF"]],
        signals[signal_names["EEG C3-REF"]] - signals[signal_names["EEG P3-REF"]],
        signals[signal_names["EEG P3-REF"]] - signals[signal_names["EEG O1-REF"]],
        signals[signal_names["EEG FP2-REF"]] - signals[signal_names["EEG F4-REF"]],
        signals[signal_names["EEG F4-REF"]] - signals[signal_names["EEG C4-REF"]],
        signals[signal_names["EEG C4-REF"]] - signals[signal_names["EEG P4-REF"]],
        signals[signal_names["EEG P4-REF"]] - signals[signal_names["EEG O2-REF"]],
    ))


def readEDF(fileName):
    Rawdata = mne.io.read_raw_edf(fileName)
    signals, times = Rawdata[:]
    eventData = np.genfromtxt(fileName[:-3] + "rec", delimiter=",")
    Rawdata.close()
    return [signals, times, eventData, Rawdata]


def load_up_objects(BaseDir, OutDir):
    for dirName, _, fileList in tqdm(os.walk(BaseDir)):
        print("Found directory:", dirName)
        for fname in fileList:
            if fname[-4:] != ".edf":
                continue
            print("\t", fname)
            try:
                signals, times, event, Rawdata = readEDF(os.path.join(dirName, fname))
                signals = convert_signals(signals, Rawdata)
            except (ValueError, KeyError):
                print("something funky happened in", dirName, fname)
                continue
            feats, offending, labels = BuildEvents(signals, times, event)
            for idx, (signal, off_chan, label) in enumerate(zip(feats, offending, labels)):
                sample = {"signal": signal, "offending_channel": off_chan, "label": label}
                with open(os.path.join(OutDir, fname.split(".")[0] + "-" + str(idx) + ".pkl"), "wb") as f:
                    pickle.dump(sample, f)


if __name__ == "__main__":
    train_out_dir = os.path.join(ROOT, "processed_train")
    eval_out_dir = os.path.join(ROOT, "processed_eval")
    os.makedirs(train_out_dir, exist_ok=True)
    os.makedirs(eval_out_dir, exist_ok=True)

    load_up_objects(os.path.join(ROOT, "train"), train_out_dir)
    load_up_objects(os.path.join(ROOT, "eval"), eval_out_dir)
