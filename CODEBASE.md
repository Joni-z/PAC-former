# PAC-Former codebase

Implementation of the design in [AGENT.md](AGENT.md) (rationale in [README.md](README.md)).
The whole repo exists to support **one controlled ablation**: same backbone, same
training, swap only the token mixer — vanilla self-attention vs. CoTAR vs. our
differentiable Modulation-Index (MI) operator. Switching mixer is a single line
in a YAML config.

## Layout

```
configs/                 one YAML = one full run (the only thing that differs across the ablation is `mixer`)
data/loaders.py          TUAB/TUEV loaders ported from BIOT + SyntheticPACDataset (ours) + build_dataloaders
models/
  frontend/
    sinc.py              learnable SincNet bandpass bank, log/constant-Q spaced   [OURS]
    analytic.py          differentiable FFT-Hilbert -> unit phase vector + amplitude (never an angle)  [OURS]
    __init__.py          Frontend: raw EEG -> (band tokens, phase, amplitude)
  mixers/
    base.py              TokenMixer interface every mixer obeys (shape/dtype in == out)
    attention.py         baseline: multi-head self-attention   (ported from TeCh)
    cotar.py              baseline: CoTAR                        (ported from TeCh)
    mi_operator.py        ★ OURS: directional MVL operator, the contribution
    __init__.py            name -> class registry (build_mixer)
  block.py               norm -> mixer -> FFN; mixer injected by config
  encoder.py             stack of blocks; threads phase/amplitude through as kwargs
  head.py                mean-pool + linear classifier
  build.py               cfg dict -> assembled PACFormer
train.py                 config-driven training loop
eval.py                  metrics matching BIOT exactly (balanced acc / AUROC / weighted-F1 / Cohen kappa)
scripts/
  test_mixers.py          AGENT.md §3 acceptance check: all 3 mixers interchangeable, finite grads
  synth_pac_test.py       AGENT.md §5 mandatory PAC validation against tensorpac
  preprocess_tuab.py      raw TUAB .edf -> BIOT-format pickles (ported from BIOT/datasets/TUAB/process.py)
  preprocess_tuev.py      raw TUEV .edf/.rec -> BIOT-format pickles (ported from BIOT/datasets/TUEV/process.py)
preprocess_tuab.slurm     slurm job for the TUAB preprocessing script (CPU only, ~3200 files, 59GB)
preprocess_tuev.slurm     slurm job for the TUEV preprocessing script (CPU only, ~500 files, 19GB)
reference/                read-only clones of BIOT / TeCh / SincNet / tensorpac (gitignored)
```

`[OURS]` files are written from scratch; everything else is ported/adapted from
the reference repos (see [AGENT.md](AGENT.md) §2). Do not "improve" the ported
data preprocessing or metrics — comparability to the literature depends on them
staying identical to BIOT.

## The MI operator in one paragraph

Frequency bands are tokens. Instead of `softmax(QKᵀ)V`, the MI operator builds a
**directional** coupling matrix with the Mean-Vector-Length form,
`Z[i,j] = mean_t A_j(t)·e^{iφ_i(t)}` (row `i` = low-frequency phase / modulator,
column `j` = high-frequency amplitude / modulated). It then follows CoTAR's
`aggregate → redistribute` skeleton (so it is a legitimate attention replacement,
not a pooling): aggregate the modulator tokens weighted by `softmax_i|Z[i,j]|`,
concat onto each token, project. Three numerical choices from the design doc are
implemented and matter: stay in complex arithmetic (`z/|z|`, never `atan2`);
mean-centre the amplitude envelope before the MVL sum (debiasing, removes the
spurious low-frequency peak); and use a log/constant-Q filterbank so high-
frequency amplitude bands are wide enough to span the `f_amp ± f_phase`
sidebands that carry the coupling.

## Environment

```bash
conda create -n pacformer python=3.11 -y && conda activate pacformer
pip install -r requirements.txt
```

`requirements.txt` now pins the CUDA build (`+cu126`) since the `pacformer` env
runs real training on H200 GPU nodes. If you ever need a CPU-only env (e.g. for
quick login-node smoke tests), install
`torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu`
instead — without `--no-deps`, since the CUDA build needs the bundled
`nvidia-*-cu12` packages as real dependencies, not just a wheel swap.

## Quickstart (synthetic task — no real data needed)

```bash
# 1. interface check — all three mixers interchangeable (run first)
python scripts/test_mixers.py
# 2. mandatory PAC validation — operator localises a known 10->60 Hz coupling
python scripts/synth_pac_test.py
# 3. end-to-end ablation on the synthetic task (swap only `mixer`)
python train.py --config configs/synthetic_mi.yaml
python train.py --config configs/synthetic_attention.yaml
python train.py --config configs/synthetic_cotar.yaml
```

## Real data: TUAB / TUEV

The raw TUH EEG corpus is already downloaded at `tuh_eeg/`:

- `tuh_eeg/v3.0.1/edf/{train,eval}/{abnormal,normal}/01_tcp_ar/*.edf` — TUAB
- `tuh_eeg/v2.0.1/edf/{train,eval}/<subject>/*.edf` (+ `.rec` annotations) — TUEV

These are raw `.edf` files, not the pickle format BIOT's loaders expect, so
preprocessing has to run once before training:

```bash
# CPU-bound, IO-heavy — run as a slurm job, not on the login node
sbatch preprocess_tuab.slurm   # -> tuh_eeg/v3.0.1/edf/processed/{train,val,test}/*.pkl
sbatch preprocess_tuev.slurm   # -> tuh_eeg/v2.0.1/edf/processed_{train,eval}/*.pkl
```

Both scripts are verbatim ports of BIOT's `datasets/{TUAB,TUEV}/process.py` —
same channel montage (16 derived bipolar channels), same resampling (200Hz),
same windowing (TUAB: 10s non-overlapping; TUEV: 5s around each labelled
event) — only the root path was changed to point at `tuh_eeg/`. Verified file-
by-file against a few real recordings before being trusted (output shapes:
TUAB `(16, 2000)` per window, TUEV `(16, 1250)` per event).

TUAB's train/val/test split happens *during* preprocessing (subject-disjoint,
80/20 train/val, matching `run_binary_supervised.py`'s seed 12345), so
`data/loaders.py::_tuab_sets` just reads the three resulting folders. TUEV's
upstream script only produces `processed_train`/`processed_eval` — the 10%
subject-held-out validation split is carved out at load time in
`data/loaders.py::_tuev_sets`, reproducing BIOT's `prepare_TUEV_dataloader`
split exactly (same seed 4523, same `f.split("_")[0]` subject key).

Once preprocessing has run:

```bash
python train.py --config configs/tuab_mi.yaml
python train.py --config configs/tuev_mi.yaml
# swap mixer: configs/tuab_attention.yaml, configs/tuab_cotar.yaml, etc.
```

`configs/tuab_*.yaml` / `configs/tuev_*.yaml` already point `data_root` at the
preprocessing scripts' output paths. Verified end-to-end (forward + backward,
correct logits shape) against the real channel/window shapes before being
trusted, with `device: cpu` for that smoke test — set `device: cuda` for an
actual training run on a GPU node.

## Open ablation choices (not closed decisions — AGENT.md §9)

- MI redistribute step is concat+MLP (matching CoTAR). Gating and matrix-mixing
  are the other two variants to try.
- Filterbank spacing defaults to `log` (constant-Q); `linear` is available for
  ablation via `SincBandpass(spacing=...)`.
