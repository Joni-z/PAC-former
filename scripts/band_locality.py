"""Where does each dataset's label live in frequency? (AGENT.md sec. 13.28, Link 1)

Why: cf_mixed beats standard MAE on TUAB/TUEV/TUSZ/CHB-MIT but LOSES on Sleep-EDF.
Without an explanation that is itself measured, "crossfreq masking helps on EEG" is a
claim with a known counterexample sitting next to it. This turns the explanation into
a number that can be checked BEFORE the pooled pretrain -- and it is what decides which
datasets belong in that pretrain.

Mechanism under test: the crossfreq objective hides the UPPER half of the bands and makes
the model predict them from the visible LOWER half. That is only a useful thing to spend
capacity on if the upper half actually carries label-relevant structure. If a dataset's
label lives entirely in the low bands, crossfreq masking makes the model model something
irrelevant, and standard random masking should win.

Method (deliberately model-free, so it cannot be an artefact of our architecture):
  * band edges = `torch.linspace(1, fs/2 - 2, n_bands+1)` -- the SAME linear split the
    sinc filterbank is initialised with (models/frontend/sinc.py:51), so "upper half"
    means here exactly what it means to the mask.
  * feature = log power per (channel, band) from a real FFT, i.e. the same quantity the
    SSL objective reconstructs (`log mean amplitude`).
  * a plain multinomial logistic regression on those features, fit on train, scored on
    test. Three feature sets: LOW half only, HIGH half only, ALL.

Read the output as: `high_only` is the ceiling on what crossfreq masking can be teaching
the model about. If `high_only` is at chance, the objective is asking for noise.

    python scripts/band_locality.py                 # all datasets it can find
    python scripts/band_locality.py sleepedf tuab
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.linear_model import LogisticRegression      # noqa: E402
from sklearn.metrics import balanced_accuracy_score, roc_auc_score  # noqa: E402
from sklearn.preprocessing import StandardScaler         # noqa: E402

from data import build_dataloaders                       # noqa: E402

N_BANDS = 8
MAX_WINDOWS = 4000          # per split; enough for a stable logistic fit, cheap on CPU

# (dataset, data_root, sample_rate, num_classes, n_channels, seq_len)
DATASETS = {
    "tuab":    ("/scratch/zz5070/PACLock/tuh_eeg/v3.0.1/edf/processed", 200, 2, 16, 2000),
    "tuev":    ("/scratch/zz5070/PACLock/tuh_eeg/v2.0.1/edf",           200, 6, 16, 1000),
    "tusz":    ("/scratch/zz5070/PACLock/tuh_eeg/tuh_eeg_seizure/v2.0.6/edf/processed", 200, 2, 16, 2000),
    "chbmit":  ("/scratch/zz5070/PACLock/chb_mit/processed",            200, 2, 16, 2000),
    "sleepedf": ("/scratch/zz5070/PACLock/sleep_edf/processed",                   100, 5, 2,  3000),
}


def band_edges(fs, n_bands=N_BANDS):
    """Exactly the sinc filterbank's init (models/frontend/sinc.py:40,51)."""
    return np.linspace(1.0, fs / 2 - 2.0, n_bands + 1)


def band_logpower(X, fs, n_bands=N_BANDS):
    """(N, C, T) -> (N, C*n_bands) log mean band power."""
    N, C, T = X.shape
    spec = np.abs(np.fft.rfft(X, axis=-1)) ** 2
    freqs = np.fft.rfftfreq(T, d=1.0 / fs)
    edges = band_edges(fs, n_bands)
    out = np.empty((N, C, n_bands), dtype=np.float32)
    for b in range(n_bands):
        sel = (freqs >= edges[b]) & (freqs < edges[b + 1])
        out[:, :, b] = spec[:, :, sel].mean(axis=-1) if sel.any() else 0.0
    return np.log(out + 1e-8).reshape(N, C * n_bands), out.shape


def drain(loader, cap):
    xs, ys = [], []
    n = 0
    for X, y in loader:
        xs.append(np.asarray(X)); ys.append(np.asarray(y))
        n += len(y)
        if n >= cap:
            break
    return np.concatenate(xs)[:cap], np.concatenate(ys)[:cap]


def score(Xtr, ytr, Xte, yte, num_classes):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(sc.transform(Xtr), ytr)
    Z = sc.transform(Xte)
    pred = clf.predict(Z)
    out = {"bal_acc": balanced_accuracy_score(yte, pred)}
    if num_classes == 2:
        out["auroc"] = roc_auc_score(yte, clf.predict_proba(Z)[:, 1])
    return out


def main():
    wanted = sys.argv[1:] or list(DATASETS)
    half = N_BANDS // 2
    print(f"{'dataset':<9} {'fs':>4} {'split':<20} " + " ".join(f"{k:>9}" for k in
          ("low_bacc", "high_bacc", "all_bacc", "low_auc", "high_auc", "all_auc")))
    for name in wanted:
        if name not in DATASETS:
            print(f"{name}: unknown", file=sys.stderr); continue
        root, fs, ncls, nch, seq = DATASETS[name]
        if not os.path.exists(root):
            print(f"{name:<9} {fs:>4} MISSING {root}", file=sys.stderr); continue
        cfg = dict(dataset=name, data_root=root, sample_rate=fs, sampling_rate=fs,
                   n_channels=nch, seq_len=seq, num_classes=ncls,
                   batch_size=256, num_workers=0)
        try:
            tr_l, _, te_l, _ = build_dataloaders(cfg)
            Xtr, ytr = drain(tr_l, MAX_WINDOWS)
            Xte, yte = drain(te_l, MAX_WINDOWS)
        except Exception as e:                                   # noqa: BLE001
            print(f"{name:<9} load failed: {type(e).__name__}: {e}", file=sys.stderr)
            continue

        Ftr, _ = band_logpower(Xtr, fs)
        Fte, _ = band_logpower(Xte, fs)
        Ftr = Ftr.reshape(len(Ftr), nch, N_BANDS)
        Fte = Fte.reshape(len(Fte), nch, N_BANDS)
        sets = {
            "low":  (Ftr[:, :, :half].reshape(len(Ftr), -1), Fte[:, :, :half].reshape(len(Fte), -1)),
            "high": (Ftr[:, :, half:].reshape(len(Ftr), -1), Fte[:, :, half:].reshape(len(Fte), -1)),
            "all":  (Ftr.reshape(len(Ftr), -1), Fte.reshape(len(Fte), -1)),
        }
        res = {k: score(a, ytr, b, yte, ncls) for k, (a, b) in sets.items()}
        e = band_edges(fs)
        span = f"lo{e[0]:.0f}-{e[half]:.0f} hi{e[half]:.0f}-{e[-1]:.0f}Hz"
        row = [res[k]["bal_acc"] for k in ("low", "high", "all")]
        row += [res[k].get("auroc", float("nan")) for k in ("low", "high", "all")]
        print(f"{name:<9} {fs:>4} {span:<20} " + " ".join(f"{v:>9.4f}" for v in row),
              flush=True)
        print(f"{'':<9} {'':>4} n_train={len(ytr)} n_test={len(yte)} chance_bacc={1/ncls:.3f}")


if __name__ == "__main__":
    main()
