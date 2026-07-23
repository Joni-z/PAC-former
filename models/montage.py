"""Electrode coordinates for montage-agnostic spatial positional encoding
(AGENT.md sec. 13.23 A / 13.4).

Every dataset in this project feeds a fixed **bipolar** montage (a channel is an
electrode *pair*, e.g. FP1-F7), so a channel is encoded by the concatenated xyz
of its two 10-20 endpoints -> a 6-D coordinate. Coordinates were extracted once
from `mne.channels.make_standard_montage("standard_1020")` (TUH's legacy T3/T4/
T5/T6 mapped to modern T7/T8/P7/P8) and normalised by a ~0.1 m head radius so
they sit in ~[-1.2, 1.2]; hardcoded here so runtime has no mne dependency.

`coords_for(dataset)` returns an (n_channels, 6) float array, or None for an
unknown montage (SpatialPE then falls back to its learned index embedding, so
existing behaviour is unchanged when coordinates are absent).
"""

import numpy as np

# 16-channel bipolar montage shared by TUAB/TUEV/TUSZ/CHB-MIT (BIOT recipe,
# scripts/preprocess_tuab.py). Order matches the preprocessing channel order.
_BIPOLAR_16 = [
    [-0.2944, 0.8392, -0.0699, -0.7026, 0.4247, -0.1142],   # FP1-F7
    [-0.7026, 0.4247, -0.1142, -0.8416, -0.1602, -0.0935],  # F7-T3
    [-0.8416, -0.1602, -0.0935, -0.7243, -0.7345, -0.0249], # T3-T5
    [-0.7243, -0.7345, -0.0249, -0.2941, -1.1245, 0.0884],  # T5-O1
    [0.2987, 0.849, -0.0708, 0.7304, 0.4442, -0.12],        # FP2-F8
    [0.7304, 0.4442, -0.12, 0.8508, -0.1502, -0.0949],      # F8-T4
    [0.8508, -0.1502, -0.0949, 0.7306, -0.7307, -0.0254],   # T4-T6
    [0.7306, -0.7307, -0.0254, 0.2984, -1.1216, 0.088],     # T6-O2
    [-0.2944, 0.8392, -0.0699, -0.5024, 0.5311, 0.4219],    # FP1-F3
    [-0.5024, 0.5311, 0.4219, -0.6536, -0.1163, 0.6436],    # F3-C3
    [-0.6536, -0.1163, 0.6436, -0.5301, -0.7879, 0.5594],   # C3-P3
    [-0.5301, -0.7879, 0.5594, -0.2941, -1.1245, 0.0884],   # P3-O1
    [0.2987, 0.849, -0.0708, 0.5184, 0.543, 0.4081],        # FP2-F4
    [0.5184, 0.543, 0.4081, 0.6712, -0.109, 0.6358],        # F4-C4
    [0.6712, -0.109, 0.6358, 0.5567, -0.7856, 0.5656],      # C4-P4
    [0.5567, -0.7856, 0.5656, 0.2984, -1.1216, 0.088],      # P4-O2
]

# Sleep-EDF Cassette 2-channel montage (Fpz-Cz, Pz-Oz).
_SLEEP_2 = [
    [0.0011, 0.8825, -0.0171, 0.004, -0.0917, 1.0024],      # Fpz-Cz
    [0.0032, -0.8111, 0.8261, 0.0011, -1.1489, 0.1466],     # Pz-Oz
]

_BY_DATASET = {
    "tuab": _BIPOLAR_16,
    "tuev": _BIPOLAR_16,
    "tusz": _BIPOLAR_16,
    "chbmit": _BIPOLAR_16,
    "tuep": _BIPOLAR_16,
    "sleepedf": _SLEEP_2,
    "sleep": _SLEEP_2,
}

COORD_DIM = 6


def coords_for(dataset: str | None):
    """(n_channels, 6) coordinate array for a known montage, else None."""
    if dataset is None:
        return None
    table = _BY_DATASET.get(str(dataset).lower())
    if table is None:
        return None
    return np.asarray(table, dtype=np.float32)
