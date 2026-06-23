# PAC-Former Codebase Build Guide (for coding agent)

## 0. What this project is

We are building **PAC-Former**: a differentiable phase-amplitude coupling (PAC)
operator that replaces self-attention in a frequency-domain EEG encoder.
Frequency bands are tokens. The mixer that moves information between tokens is
a learnable Modulation Index (MI) operator instead of QKᵀ softmax attention.

The codebase's entire reason for existing is to support one controlled
ablation: **same backbone, same training, swap only the token mixer** between
(a) vanilla self-attention, (b) CoTAR, (c) our MI operator. Every design
decision below exists to keep that swap clean. If a change makes the mixer
swap messier, it's the wrong change.

**Do not silently deviate from this guide.** If something here turns out to
be wrong or impossible once you're in the code, stop and flag it rather than
improvising a different design.

---

## 1. Repo layout

```
configs/                  # one yaml = one full run config (which mixer, hparams)
data/                     # dataset loading + preprocessing (ported, not invented)
models/
  frontend/
    sinc.py               # learnable SincNet-style bandpass  [OURS]
    analytic.py           # differentiable Hilbert -> phase/amplitude per band [OURS]
  mixers/
    base.py                # abstract interface ALL mixers must satisfy
    attention.py           # baseline: vanilla self-attention
    cotar.py                # baseline: ported from TeCh
    mi_operator.py          # ★ OURS: the actual contribution
  block.py                 # norm -> mixer -> FFN, mixer injected via config
  encoder.py                # stack of L blocks
  head.py                   # pooling + classification head
  build.py                  # config -> assembled model
train.py                    # training loop (ported skeleton)
eval.py                      # metrics (ported, must match BIOT exactly)
scripts/
  synth_pac_test.py          # validation harness, see Section 5 — build this EARLY
```

**Rule:** files marked `[OURS]` or inside `mixers/mi_operator.py` are written
from scratch. Everything else should be ported/adapted from the reference
repos in Section 2, not reinvented.

---

## 2. Reference repos — what to take from each

Clone these into a `vendor/` or `reference/` scratch directory for reading;
do not import them live as a dependency.

| Repo | Take this | Do NOT take |
|---|---|---|
| `github.com/ycq091044/BIOT` | `data/` preprocessing for TUAB/TUEV, dataset splits, `eval.py` metric definitions (balanced acc, AUROC, weighted F1, Cohen's kappa), train loop skeleton, baselines: SPaRCNet, ContraWR, CNN-Transformer, FFCL, ST-Transformer | its tokenization architecture — not relevant to our mixer-swap design |
| `github.com/Levi-Ackman/TeCh` | the CoTAR module itself, ported as our CoTAR baseline mixer (`mixers/cotar.py`) | its time-domain patch/backbone architecture — we are frequency-domain, do not align to it |
| `github.com/mravanelli/SincNet` | reference for the learnable sinc bandpass formula (only 2 learned params per filter: low/high cutoff) | as-is audio config; adapt to EEG sample rate / band count |
| `github.com/braindecode/braindecode` | EEGNet, ShallowConvNet, EEGConformer, SPaRCNet implementations for baseline tier 2; also useful for auto-downloadable small datasets (MOABB/MNE) during early dev | — |
| `github.com/eeyhsong/EEG-Conformer` | structural reference for a CNN+self-attention EEG model — useful as a sanity check for what "swap out the attention" should look like | — |
| `github.com/EtienneCmb/tensorpac` | ground-truth PAC computation (MVL/Tort/Ozkurt) and `pac_signals_tort` synthetic signal generator — **required** for Section 5 validation | — |
| `github.com/pactools/pactools` | secondary cross-check for tensorpac, optional | — |
| `github.com/935963004/LaBraM` | reference numbers only (table-filling), optionally a fine-tuned checkpoint for tier-3 comparison | do not reimplement, do not try to beat it as a primary claim |

If a needed detail isn't covered by these repos, stop and ask rather than
guessing — especially for TUAB/TUEV preprocessing, where small deviations
break comparability.

---

## 3. The mixer interface — build this before anything else

This is the load-bearing contract of the whole repo. All three mixers must
satisfy it exactly, including shape and dtype, so swapping one line in a
config is the only thing that changes between ablation runs.

`models/mixers/base.py`:

```python
class TokenMixer(nn.Module):
    """
    Input:  x of shape (batch, n_bands, hidden_dim)
            — n_bands tokens, each a learned representation of one frequency band
    Output: same shape (batch, n_bands, hidden_dim)
    No mixer may change n_bands or hidden_dim.
    No mixer may assume anything about how x was produced (don't reach back
    into the frontend for phase/amplitude — if the MI operator needs phase
    and amplitude directly, those must be passed alongside x explicitly,
    see Section 4).
    """
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError
```

Acceptance check before writing any mixer: write a unit test that
instantiates each of the three mixers with the same `n_bands`/`hidden_dim`,
feeds them the same random tensor, and asserts identical output shape and
dtype, and that `.backward()` runs without NaN/Inf on all three. This test
must exist and pass before integrating any mixer into `block.py`.

---

## 4. Frontend contract (`frontend/sinc.py`, `frontend/analytic.py`)

- `sinc.py`: learnable bandpass bank. Each band has exactly 2 learned
  parameters (low cutoff, high cutoff), following SincNet. Output: filtered
  signal per band, shape `(batch, n_bands, time)`.
- `analytic.py`: differentiable analytic signal via FFT-based Hilbert
  transform (multiply by sign function in frequency domain, inverse FFT).
  **Do not call `torch.atan2` or any explicit `arg()` anywhere in this file.**
  Stay in complex representation throughout:
  - phase vector for band b: `z_b / |z_b|` (unit complex number, never an angle)
  - amplitude for band b: `|z_b|`
  - clamp `|z_b|` away from 0 (e.g. `+1e-6`) before dividing, since that's the
    only singular point left once `arg()` is avoided.
- Output of the frontend going into `mi_operator.py` should expose both:
  (a) the pooled per-band token `x` (batch, n_bands, hidden_dim) for the
      generic mixer interface, and
  (b) per-band phase/amplitude tensors, since the MI operator's aggregate
      step needs these directly, not just the post-projection token.
  Pass (b) as an explicit auxiliary argument to `forward()`, not by having
  the mixer reach backward into frontend internals.

---

## 5. Mandatory validation step — do this before touching real EEG

Build `scripts/synth_pac_test.py` immediately after `mi_operator.py` has a
forward pass, **before** wiring it into the full classification pipeline.

Steps:
1. Use `tensorpac.signals.pac_signals_tort` to generate a synthetic signal
   with a known coupling (e.g. 10 Hz phase → 100 Hz amplitude).
2. Run our differentiable frontend + MI operator on it; extract the
   coupling matrix `|Z|`.
3. Independently compute the ground-truth comodulogram with `tensorpac`'s
   own MVL/Tort estimator on the same signal.
4. Assert our operator's peak coupling location matches the known
   (10 Hz, 100 Hz) pair, and assert `loss.backward()` produces no NaN/Inf
   gradients anywhere in the frontend or operator.

Do not proceed to wiring up real EEG training until this test passes. If it
fails, the bug is in the operator or frontend, not in the downstream task —
debug here, not in the full pipeline.

---

## 6. Build order

Follow this order. Each step should be independently runnable/testable
before moving to the next — do not write steps 2-5 all at once and debug
backward.

1. **Plumbing first.** Wire up `data/` (ported from BIOT) + a trivial
   baseline (e.g. EEGNet from braindecode) + `train.py` + `eval.py`. Get one
   real number out. This proves the pipeline and metrics are correct, with
   zero of our own novel code involved — any bug here is plumbing, not method.
2. **Frontend in isolation.** Implement `sinc.py` + `analytic.py`. Validate
   with synthetic input (not yet via the classifier) that phase/amplitude
   outputs look right and gradients are finite.
3. **Mixer interface + attention baseline.** Implement `base.py` and
   `attention.py`, wire frontend → block → encoder → head with the vanilla
   attention mixer. This is the frequency-domain transformer baseline and the
   first real point of comparison.
4. **MI operator.** Implement `mi_operator.py`. Run Section 5's synthetic PAC
   test. Only after it passes, swap it into the same encoder from step 3.
5. **CoTAR baseline.** Port from TeCh into `cotar.py`, slot into the same
   encoder.
6. **Tier-2 baselines.** EEGNet, SPaRCNet, EEG-Conformer, TeCh-original — run
   as separate models (not through our mixer interface), for the "is our
   encoder competitive at all" sanity context.
7. **Tier-3 numbers.** Pull LaBraM/BIOT/EEGPT/CBraMod published numbers into
   the results table. Do not reimplement or fine-tune these unless
   specifically asked later.

---

## 7. Dataset notes

- TUAB (binary, normal/abnormal) and TUEV (6-class event) require an access
  application to `help@nedcdata.org` before they can be downloaded — this has
  unpredictable lead time. Submit this early and do NOT block steps 1-5 on it.
- For early development, use braindecode/MOABB auto-downloadable datasets
  (e.g. BCI Competition IV-2a) to validate the pipeline end-to-end on real
  (if small) EEG before TUAB/TUEV access arrives.
- Once TUAB/TUEV access is granted, only the `data/` loader needs to change;
  everything upstream of it should be untouched if the interfaces above were
  respected.

---

## 8. Metrics — must match BIOT exactly

- TUAB (binary): balanced accuracy + AUROC
- TUEV (6-class): balanced accuracy + weighted F1 + Cohen's kappa

Use the same aggregation logic as BIOT's `eval.py` (segment-level vs.
recording-level aggregation matters — copy it, don't re-derive it), or our
numbers will not be comparable to anything in the literature.

---

## 9. Things to flag back to the user rather than deciding alone

- Any case where TUAB/TUEV preprocessing details aren't fully specified by
  the BIOT repo and require a judgment call.
- Any case where the MI operator's redistribute step (Section "reweight by
  |Z| and modulate the corresponding high-frequency token") admits multiple
  reasonable implementations (gating vs. matrix-mixing vs. concat+MLP like
  CoTAR). Default to concat+MLP (matching CoTAR) for the first working
  version, but flag that the other two are open ablation choices, not closed
  decisions.
- Any numerical instability that survives the unit-complex-vector trick in
  Section 4 — don't patch it ad hoc, surface it.