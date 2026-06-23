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
    cotar.py             baseline: CoTAR                        (ported from TeCh)
    mi_operator.py       ★ OURS: directional MVL operator, the contribution
    __init__.py          name -> class registry (build_mixer)
  block.py               norm -> mixer -> FFN; mixer injected by config
  encoder.py             stack of blocks; threads phase/amplitude through as kwargs
  head.py                mean-pool + linear classifier
  build.py               cfg dict -> assembled PACFormer
train.py                 config-driven training loop
eval.py                  metrics matching BIOT exactly (balanced acc / AUROC / weighted-F1 / Cohen kappa)
scripts/
  test_mixers.py         AGENT.md §3 acceptance check: all 3 mixers interchangeable, finite grads
  synth_pac_test.py      AGENT.md §5 mandatory PAC validation against tensorpac
reference/               read-only clones of BIOT / TeCh / SincNet / tensorpac (gitignored)
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

`requirements.txt` pins CPU torch (this was built on a login node). On a GPU node
install the matching CUDA `torch`/`torchaudio` instead; nothing else changes.

## Quickstart

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

Real EEG (TUAB/TUEV) needs an NEDC data application (AGENT.md §7); once granted,
only `data/loaders.py` paths change. Point a config at `dataset: tuab` /
`dataset: tuev` with `data_root: <path to processed splits>`.

## Open ablation choices (not closed decisions — AGENT.md §9)

- MI redistribute step is concat+MLP (matching CoTAR). Gating and matrix-mixing
  are the other two variants to try.
- Filterbank spacing defaults to `log` (constant-Q); `linear` is available for
  ablation via `SincBandpass(spacing=...)`.
