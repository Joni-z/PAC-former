"""Metrics, defined to match BIOT exactly so numbers are comparable.

BIOT uses pyhealth's ``binary_metrics_fn`` / ``multiclass_metrics_fn``; those are
thin wrappers over scikit-learn with the conventions reproduced here:

  * binary    (TUAB) : balanced accuracy (0.5 threshold) + AUROC
  * multiclass (TUEV): balanced accuracy + weighted F1 + Cohen's kappa

Aggregation is sample/segment level -- the same level BIOT reports -- so do not
add recording-level pooling here without matching BIOT's own logic.
"""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    roc_auc_score,
)
from scipy.special import softmax


def binary_metrics(y_true: np.ndarray, logits: np.ndarray) -> dict:
    """``logits``: (N,) or (N, 2). Returns balanced accuracy + AUROC + PR-AUC.

    pr_auc (average precision) is the primary metric for the highly imbalanced
    seizure-detection setting (TUSZ, CHB-MIT): with ~2-10% positives AUROC is
    misleadingly high, so PR-AUC is what the seizure-detection literature and
    BIOT report as the headline number.
    """
    logits = np.asarray(logits)
    prob = softmax(logits, axis=1)[:, 1] if logits.ndim == 2 else 1 / (1 + np.exp(-logits))
    pred = (prob > 0.5).astype(int)
    return {
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "auroc": roc_auc_score(y_true, prob),
        "pr_auc": average_precision_score(y_true, prob),
    }


def multiclass_metrics(y_true: np.ndarray, logits: np.ndarray) -> dict:
    """``logits``: (N, C). Returns balanced accuracy + weighted F1 + Cohen kappa."""
    pred = np.asarray(logits).argmax(axis=1)
    return {
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "f1_weighted": f1_score(y_true, pred, average="weighted"),
        "cohen_kappa": cohen_kappa_score(y_true, pred),
    }


def compute_metrics(y_true, logits, num_classes: int) -> dict:
    return binary_metrics(y_true, logits) if num_classes == 2 \
        else multiclass_metrics(y_true, logits)


def select_key(num_classes: int, cfg: dict | None = None) -> str:
    """Validation metric used to pick the best epoch. Matches BIOT's own protocol
    so a BIOT row and one of our rows are selected the same way.

    BIOT (reference/BIOT):
      * run_binary_supervised.py:348      EarlyStopping(monitor="val_auroc")
      * run_multiclass_supervised.py:288  EarlyStopping(monitor="val_cohen")

    **Correction (2026-07-23):** multiclass previously selected on
    ``balanced_accuracy`` here and in ``pretrain.py``. That silently deviated from
    BIOT, and it is not metric-neutral: on TUEV our model's validation bal-acc and
    kappa are *anti-correlated* across epochs (§13.30), so bal-acc selection picked
    a low-kappa epoch for us and a good one for BIOT. Every TUEV number produced
    before this fix is selected on the wrong metric and must be re-run.

    ``cfg['select_metric']`` overrides, for the rare case where a run deliberately
    optimises something else -- but the default is the comparable one.
    """
    if cfg and cfg.get("select_metric"):
        return cfg["select_metric"]
    return "auroc" if num_classes == 2 else "cohen_kappa"
