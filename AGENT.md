# PAC-Former — Status & Build Guide (for coding agent)

## 0. What this project is

> **Project positioning (updated 2026-07-12, post-meeting with PI — this
> supersedes any earlier framing in this doc that reads as "we are competing
> head-to-head with CoTAR"):**
> **PAC-Former's actual goal is an EEG foundation model.** CoTAR/attention are
> not the opponents we're trying to beat in a vacuum — they are a source of
> *time-series architecture ideas/tricks* we are evaluating for inclusion in
> the eventual foundation-model backbone. The mixer-swap ablation
> (attention/cotar/mi, §9 below) is how we validate individual design
> choices (does this idea help, on which tasks, why), not the end goal itself.
> **Large-scale self-supervised pretraining is the next planned phase** once
> the backbone design (frontend + mixer) is settled — see §11.

**PAC-Former**: a differentiable phase-amplitude coupling (PAC) operator that
replaces self-attention in a frequency-domain EEG encoder. Frequency bands are
tokens. The mixer that moves information between tokens is a learnable
Modulation Index (MI) operator instead of QKᵀ softmax attention.

The codebase's mixer-swap machinery exists to support one controlled
ablation: **same backbone, same training, swap only the token mixer** between
(a) vanilla self-attention, (b) CoTAR, (c) our MI operator. Every design
decision exists to keep that swap clean. If a change makes the mixer swap
messier, it's the wrong change. This ablation is a *validation tool* for the
backbone design, not the project's deliverable — see the positioning note
above.

**Do not silently deviate from this guide.** If something here turns out to
be wrong or impossible once you're in the code, stop and flag it rather than
improvising a different design.

---

## 1. Repo layout

```
configs/                  # one yaml = one full run config (which mixer, hparams, dataset)
data/                     # dataset loading + preprocessing (TUAB/TUEV ported from BIOT,
                           #  Sleep-EDF Cassette written by us)
models/
  frontend/
    sinc.py                # learnable SincNet-style bandpass          [OURS]
    analytic.py            # differentiable Hilbert -> phase/amplitude [OURS]
    conv.py                # diagnostic-only plain conv tokenizer (no band structure)
    __init__.py             # Frontend module: sinc -> per-band conv patch tokenizer
                             # -> (band x patch) tokens + phase/amplitude
  mixers/
    base.py                # abstract interface ALL mixers must satisfy
    attention.py            # baseline: vanilla self-attention
    cotar.py                 # baseline: ported from TeCh (aggregate -> redistribute)
    mi_operator.py           # ★ OURS: PAC-biased cross-band attention
  augment.py                 # jitter / frequency-mask / time-mask / channel-drop / drop
  block.py                    # norm -> mixer -> FFN, mixer injected via config
  encoder.py                   # stack of L blocks
  head.py                       # pooling + classification head
  build.py                       # config -> assembled model
train.py                         # training loop (checkpoints on best val balanced_accuracy)
eval.py                           # metrics (ported, matches BIOT exactly)
scripts/
  synth_pac_test.py                # synthetic-PAC validation harness (Section 5)
  preprocess_tuab.py / preprocess_tuev.py   # ported from BIOT
  download_sleepedf.sh / preprocess_sleepedf.py   # ours, PhysioNet Sleep-EDF Cassette
```

**Rule:** files marked `[OURS]` or inside `mixers/mi_operator.py` are written
from scratch. Everything else should be ported/adapted from the reference
repos in Section 2, not reinvented.

---

## 2. Reference repos — what to take from each

Clone into `reference/` (gitignored, read-only, do not import live).

| Repo | Take this | Do NOT take |
|---|---|---|
| `github.com/ycq091044/BIOT` | `data/` preprocessing for TUAB/TUEV, dataset splits, `eval.py` metric definitions (balanced acc, AUROC, weighted F1, Cohen's kappa), train loop skeleton, published baseline/BIOT numbers (Section 9) | its tokenization architecture |
| `github.com/Levi-Ackman/TeCh` | the CoTAR module itself, ported as our CoTAR baseline mixer (`mixers/cotar.py`) | its time-domain patch/backbone architecture — we are frequency-domain |
| `github.com/mravanelli/SincNet` | reference for the learnable sinc bandpass formula (2 learned params per filter: low/high cutoff) | as-is audio config; adapted to EEG sample rate / band count |
| `github.com/EtienneCmb/tensorpac` | ground-truth PAC computation (MVL/Tort/Ozkurt) and `pac_signals_tort` synthetic signal generator — used in `synth_pac_test.py` | — |

---

## 3. The mixer interface

`models/mixers/base.py`:

```python
class TokenMixer(nn.Module):
    """
    Input:  x of shape (batch, n_bands * n_patches, hidden_dim)
    Output: same shape
    No mixer may change token count or hidden_dim. The MI operator receives
    phase_unit/amplitude as explicit kwargs (never reaches back into the
    frontend).
    """
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError
```

---

## 4. Frontend (`models/frontend/__init__.py`)

Current (v2) design — band-preserving conv patch tokenizer:

- sinc bandpass bank, per channel: `(B, C, n_bands, T)`
- per band: `Conv1d(C -> hidden, kernel=patch_len, stride=patch_len)` — mixes
  channels (learned, not averaged) and patchifies time
- tokens flattened to `(B, n_bands * P, hidden)`, `P = seq_len // patch_len`;
  band identity is recoverable (`P = N // n_bands`) so the MI operator can
  build a band x band coupling matrix
- `phase_unit` / `amplitude` exposed per **channel**, per band, at full time
  resolution — `(B, C, n_bands, T)`, never averaged across channels before
  the Hilbert transform (v3 fix, Section 9.7) — passed to the MI operator as
  explicit kwargs, which computes coupling per channel and averages `|Z|`
  across channels

**v1 design (deprecated, see Section 9 for why):** mean-pooled over all
channels, and collapsed each band's whole time course into a single token via
`Linear(seq_len -> hidden)`. This threw away both spatial and temporal
structure and capped TUAB AUROC at 0.79 regardless of mixer — see the
diagnostic in Section 9.

`analytic.py` rules (unchanged): FFT-based Hilbert transform, stay in complex
representation throughout — `phase_unit = z / |z|` (unit complex, never an
explicit `atan2`/`arg()`), `amplitude = |z|`, clamp `|z|` away from 0 before
dividing.

---

## 5. MI operator (`models/mixers/mi_operator.py`) — current design

**PAC-biased cross-band attention.** For target band j attending to source
band i:

```
attn[j, i]   = Q_j . K_i / sqrt(d_k)  +  pac_scale * coupling[i, j]
weight[j, i] = softmax_i(attn[j, i])
core_j       = sum_i weight[j, i] * V_i
```

`coupling` is the Canolty (2006) MVL score, mean-centered amplitude debiasing
(removes the spurious low-frequency term from amplitude being strictly
positive), **normalized by a fixed constant (`NORM_CONST = 100.0`)** rather
than the per-channel amplitude std (see §9.10 — the std-based Ozkurt
normalization divides by ~0 on flat/dead channels in 16-channel clinical EEG
and produced NaN; a fixed divisor removes that failure mode entirely, and
`pac_scale` being learned absorbs whatever overall magnitude is left). Applies
only to the real (per-channel, 4D) path; the single-channel/synthetic 3D path
used by unit tests is unchanged and still uses std-based normalization.
`pac_scale` is a learned scalar (init 1.0). This strictly generalizes pure
attention (`pac_scale -> 0`) and pure PAC-weighted aggregation (frozen Q/K,
large `pac_scale`), so the model can learn, per layer, how much to lean on the
physiological prior vs. a data-driven cross-band relationship. Redistribute
step is concat + MLP (matches CoTAR).

**Complexity note:** despite having learned Q/K/V, this mixer is **not**
O(N²) in the token count N = n_bands·P. Q/K/V are computed on `band_repr`
(the per-band mean over the P patches, `(B, n_bands, D)`), so the attention
matrix is a fixed `n_bands × n_bands` (n_bands=32 is a constant hyperparameter,
independent of sequence length) — O(n_bands²) = O(1) w.r.t. sequence length.
The only terms that scale with sequence length (T or N) are linear: the
coupling einsum (O(n_bands²·C·T)) and the concat+MLP redistribute (O(N·D²)).
This is a relevant selling point for large-scale pretraining (sub-quadratic
vs. standard self-attention's O(N²)) — at the cost of the cross-band attention
only ever seeing band-level time-averaged summaries, not patch-level detail
(patch-level detail is only reintroduced at the final concat with `x_b`).

**v1 design (deprecated):** fixed PAC coupling matrix used directly as
softmax aggregation weights, no learned Q/K/V. Tied CoTAR on Sleep-EDF but
had no mechanism to exceed it (Section 9).

Validated with `scripts/synth_pac_test.py` (10 Hz phase -> 60 Hz amplitude
synthetic signal via `tensorpac`): recovers the correct coupling peak,
finite gradients throughout. Run this after any change to the operator,
**before** touching real EEG.

---

## 6. Metrics — must match BIOT exactly

- TUAB (binary): balanced accuracy + AUROC
- TUEV (6-class): balanced accuracy + weighted F1 + Cohen's kappa
- Sleep-EDF (5-class): balanced accuracy + weighted F1 + Cohen's kappa
  (kappa is the standard reported metric in the sleep-staging literature)

Checkpoint selection during training (`train.py`) is on **val balanced_accuracy**
(`auroc` for binary tasks), not kappa — kappa is noisier epoch-to-epoch on
smaller val splits. The final test metrics come from the best-val checkpoint,
not the last epoch trained.

---

## 7. Datasets

| Dataset | Task | Classes | Split | Notes |
|---|---|---|---|---|
| TUAB | normal/abnormal EEG | 2 | BIOT-style | 16-channel, requires access application |
| TUEV | EEG event type | 6 | BIOT-style | 16-channel, severe class imbalance -> inverse-frequency `CrossEntropyLoss` weights |
| Sleep-EDF Cassette | sleep stage (AASM) | 5 (W/N1/N2/N3/REM) | subject-disjoint, 70/15/15 by sorted subject rank | 2-channel (Fpz-Cz, Pz-Oz), 30s epochs @ 100Hz; downloaded via `scripts/download_sleepedf.sh` (PhysioNet SHA256SUMS-driven, not HTML crawl), preprocessed via `scripts/preprocess_sleepedf.py` |
| TUEP | epilepsy diagnosis | 2 | patient-disjoint, 70/15/15 (own split, TUEP ships none) | 16-channel, same montage as TUAB; `scripts/preprocess_tuep.py`. **Judged not usable**: attention/cotar baselines ≈ chance (bacc 0.49/0.50) — session-level diagnosis label doesn't reflect every 10s window, label granularity mismatch not a mixer problem. Not pursuing further. |
| TUSZ | seizure detection (event-level) | 2 | corpus-native train/dev/eval -> train/val/test | 16-channel, same montage as TUAB; `scripts/preprocess_tusz.py`, labels from `.csv_bi` interval annotations, BIOT-style seizure-window oversampling (§9.10) for the extreme (~2%) class imbalance; `eval.py` now reports **pr_auc** (primary metric for this task, AUROC is misleading under this imbalance) alongside balanced_accuracy/auroc. |
| CHB-MIT | seizure detection | 2 | — | pediatric, 23 subjects, ~42GB; downloaded via `scripts/download_chbmit.sh` (parallel, `xargs -P 10`). Resume logic originally checked file *existence* only (`[[ -f ]]`), so 0-byte files left behind by repeated interrupted runs (login-node process kills, §"login node has no GPU/compute" in memory) were wrongly treated as complete and never retried — found 41 such files after an apparently-"done" run; fixed to check non-empty (`[[ -s ]]`). Preprocessing not yet ported (BIOT's `reference/BIOT/datasets/CHB-MIT/process*.py` is the template to follow, same recipe as TUSZ). |

---

## 8. Build order (for reference; steps 1-6 are done)

1. Plumbing: `data/` + trivial baseline + `train.py` + `eval.py` — done.
2. Frontend in isolation (`sinc.py` + `analytic.py`) — done, v1 then
   redesigned to v2 (Section 4/9).
3. Mixer interface + attention baseline — done.
4. MI operator + `synth_pac_test.py` validation — done, v1 then redesigned to
   PAC-biased attention (Section 5/9).
5. CoTAR baseline — done.
6. Three-way ablation on TUAB, TUEV, Sleep-EDF — done for v1 MI + v2 frontend;
   **in progress** for v2 MI (PAC-biased attention) on Sleep-EDF.

---

## 9. Experimental log (chronological, with results)

> **MI-operator version naming (canonical, use these going forward):**
> - **mi v1** = fixed PAC coupling used directly as softmax aggregation weights,
>   no learned Q/K/V (earliest design; tested on TUAB/TUEV/Sleep-EDF).
> - **mi v2** = **PAC-biased cross-band attention** (learned single-head Q/K/V +
>   learnable-scaled PAC coupling as an additive attention bias + concat/MLP
>   redistribute), but with **channel-averaged** phase/amplitude feeding the
>   coupling. Intermediate design; still tied CoTAR on Sleep-EDF (kappa 0.51216).
> - **mi v3** = mi v2 with the **per-channel** coupling fix (coupling computed
>   per channel then |Z|-averaged, §9.7) + the global seed-control fix (§9.8).
>   This is the current `mi_operator.py`, and the version that **beats CoTAR
>   3/3 seeds on Sleep-EDF** (mean kappa 0.5354 vs 0.4996). v2→v3 differs only by
>   channel-mean→per-channel coupling (plus the seed fix, which was global).
> - **mi v4** = mi v3 + multi-head Q/K/V + per-head pac_scale + gated injection
>   (tried, no net gain, **reverted** — see §9.9).
>
> NOTE: sections §9.6–§9.9 below were written before this renaming and label the
> PAC-biased-attention design "v2" (both channel-mean and per-channel variants)
> and the multi-head+gating design "v3". Read those older labels as: old
> "v2 channel-mean" = **v2**, old "v2 per-channel" = **v3**, old "v3" = **v4**.
> (Frontend versions v1/v2/v3 are a *separate* numbering and are unaffected.)

### 9.1 Old frontend (v1), TUAB / TUEV — first three-way ablation

Frontend v1: sinc filter -> mean-pool over 16 channels -> `Linear(seq_len ->
hidden)` per band (one token per band, no time structure).

| Dataset | Mixer | balanced_acc | AUROC | kappa | f1_weighted |
|---|---|---|---|---|---|
| TUAB | attention | 0.7140 | 0.7901 | — | — |
| TUAB | cotar | 0.7208 | 0.7968 | — | — |
| TUAB | mi (v1) | 0.7142 | 0.7896 | — | — |
| TUEV | attention | 0.3825 | — | 0.1954 | 0.4924 |
| TUEV | cotar | 0.3903 | — | 0.2237 | 0.5616 |
| TUEV | mi (v1) | 0.3661 | — | 0.2373 | 0.5541 |

All three mixers tied on both datasets — no differentiation, and both
datasets sat well below BIOT's own from-scratch numbers (TUAB BIOT-vanilla
balanced_acc 0.7925; TUEV BIOT-vanilla kappa 0.4482). Suspected a frontend
bottleneck, not a mixer problem.

### 9.2 Frontend diagnostic (TUAB, attention mixer only)

Isolated the frontend as the variable: compared the v1 band frontend against
a plain conv patch tokenizer with no frequency-band decomposition at all.

| Frontend | balanced_acc | AUROC |
|---|---|---|
| diag_sinc (v1 band frontend) | 0.7149 | 0.7918 |
| diag_conv (no bands, plain conv patch tokenizer) | 0.8023 | 0.8765 |

Confirmed the frontend was the bottleneck, via two specific losses: (1)
mean-pooling over 16 channels discards spatial info (TUAB abnormalities are
often localized to a few electrodes), (2) collapsing each band's full time
course into one token discards temporal structure entirely.

### 9.3 Redesigned frontend (v2) — band-preserving conv patch tokenizer, TUAB

New frontend keeps band structure (needed for MI's band x band coupling) but
patch-tokenizes each band in (channel, time) the same way the winning
diagnostic conv tokenizer did (Section 4). Backbone and training also scaled
up to match: `n_bands` 12->32, `d_model` 64->128, `depth` 4->6, `lr` 1e-3->1e-4,
`epochs` 20->50 with `patience`-based early stopping, added augmentation.

| Mixer | balanced_acc | AUROC |
|---|---|---|
| attention | 0.7959 | 0.8764 |
| cotar | 0.7953 | 0.8730 |
| mi (v1) | ~0.81 (val peak; run preempted before test) | — |

`attention`'s balanced_acc (0.7959) exactly matches BIOT's **best pre-trained**
number (6-dataset-pretrained BIOT, from-scratch training on our side, no
pretraining) — TUAB is essentially saturated across methods (SPaRCNet 0.7896,
ST-Transformer 0.7966, BIOT-vanilla 0.7925 all cluster in the same band), so
no mixer differentiation on this task is expected going forward.

### 9.4 Redesigned frontend (v2), TUEV

| Mixer | balanced_acc | kappa | f1_weighted |
|---|---|---|---|
| attention | 0.4205 | 0.2925 | 0.6223 |
| cotar | 0.4227 | 0.2272 | 0.5591 |
| mi (v1) | 0.3503 | 0.2517 | 0.5974 |

All three still below BIOT-vanilla (kappa 0.4482) — backbone/training gap
vs. the BIOT paper remains here, separate from the mixer question. MI (v1)
is the weakest of the three. **Conclusion: PAC is not a valid inductive bias
for TUEV.** PAC assumes low-frequency phase modulates high-frequency
amplitude; TUEV's 6-way event discrimination depends on local waveform shape,
not cross-band coupling — a task-mismatch, not an implementation bug.

### 9.5 Switch to Sleep-EDF Cassette — a task where PAC has real grounding

Delta-spindle (N2/N3) and theta-gamma (REM) PAC coupling are established
physiological markers for exactly the classes being predicted here, unlike
TUAB/TUEV. Built the full pipeline from scratch (download, preprocess,
loader — Section 7).

| Mixer | balanced_acc | kappa | f1_weighted |
|---|---|---|---|
| attention | 0.5757 | 0.4629 | 0.6594 |
| cotar | 0.6052 | 0.5146 | 0.6911 |
| mi (v1) | 0.6036 | 0.5112 | 0.6934 |

mi (v1) and cotar are essentially tied, and both clearly beat attention —
confirms PAC *is* a useful signal here (unlike TUEV), but v1's fixed,
unlearned coupling-as-softmax-weights aggregation isn't extracting more value
than CoTAR's fully-learned aggregation.

### 9.6 MI redesign (v2) — PAC-biased cross-band attention

To beat CoTAR rather than tie it, added learned per-layer Q/K/V attention
with the PAC coupling as a learnable-scaled additive bias (Section 5 design).
First training attempt (job 12168972) was preempted around epoch 46 with no
final test metrics, but early val_kappa (epoch 6: 0.557) already exceeded
CoTAR's final val_kappa (0.544) — promising but inconclusive.

**Resolved (job 12236503):**

| Mixer | val_balanced_accuracy | test_balanced_accuracy | test_kappa | test_f1_weighted |
|---|---|---|---|---|
| cotar | — | — | 0.5146 | — |
| mi (v1) | — | — | 0.5112 | — |
| mi (v2) | 0.65234 | 0.62933 | **0.51216** | 0.68898 |

MI (v2) essentially ties CoTAR (0.51216 vs. 0.5146) and does not clear MI (v1)
either — the early-epoch promise (val_kappa 0.557 at epoch 6) did not survive
to convergence. Per the branch pre-registered above: **tied → architecture
problem, not a data problem.** See Section 9.7 for the diagnosis and fix
attempted next, rather than adding a second dataset at this margin.

### 9.7 Frontend v3 — per-channel PAC coupling (fixes a channel-mean bottleneck)

Diagnosis: the v2 frontend (Section 4) fixed the token path's channel-mean
bottleneck (per-band Conv1d mixes channels instead of averaging, Section 9.3)
but left the phase/amplitude path — the thing MI's coupling matrix is
actually computed from — on `hilbert(filtered).mean(dim=1)`, i.e. averaging
the analytic signal across the 2 Sleep-EDF channels (Fpz-Cz, Pz-Oz) *before*
computing phase/amplitude. If the two channels differ in phase alignment or
amplitude scale (plausible — central vs. occipital derivations often carry
different spindle/delta PAC strength), this can cancel or dilute exactly the
coupling signal the MI operator is supposed to exploit — the same class of
bug as the pre-9.3 token bottleneck, just left unfixed on the PAC-computation
side because MI hadn't been tested yet.

Fix: `Frontend.forward` no longer averages across channels before the
Hilbert transform — `phase_unit`/`amplitude` are now `(B, C, n_bands, T)`
(per channel) instead of `(B, n_bands, T)`. `MIOperator.coupling_matrix`
computes the MVL coupling **per channel** and averages the resulting
non-negative `|Z|` scores across channels (which cannot cancel the way
averaging the raw analytic signal can) rather than averaging the phase
signal itself. `coupling_matrix` still accepts the old 3D
(no-channel-dim) shape for the single-channel synthetic/unit tests
(`scripts/test_mixers.py`, `scripts/synth_pac_test.py`) — both re-verified
passing after the change, including a 2-channel gradient-finiteness check.

**Resolved (job 13230733, 48m35s):**

| Mixer | val_balanced_accuracy | test_balanced_accuracy | test_kappa | test_f1_weighted |
|---|---|---|---|---|
| cotar | — | — | 0.5146 | — |
| mi (v1) | — | — | 0.5112 | — |
| mi (v2, channel-mean) | 0.65234 | 0.62933 | 0.51216 | 0.68898 |
| mi (v2, per-channel fix) | 0.62177 | 0.62104 | **0.5138** | 0.68995 |

The per-channel fix moved test_kappa from 0.51216 to 0.5138 — inside normal
seed-to-seed noise, and still short of CoTAR's 0.5146. **The channel-mean
bottleneck is not the (or not the main) reason MI fails to clear CoTAR** —
this hypothesis is now reasonably ruled out. The frontend change is still
correct on principle (per-channel coupling avoids the theoretical
cross-channel phase-cancellation failure mode) and stays in, but the search
for why MI doesn't beat CoTAR moves elsewhere: candidates are (a) `pac_scale`
collapsing toward 0 during training (the model learning to ignore the PAC
bias entirely, i.e. degenerating to plain QK attention regardless of input
quality), (b) the concat+MLP redistribute step (Section 10's flagged open
ablation — gating instead), (c) the operator design itself rather than any
one input/output detail. Next: inspect the learned `pac_scale` from the
13230733 checkpoint before deciding between (b) and (c).

### 9.8 Seed bug found and fixed; controlled mi vs. cotar comparison

No checkpoints are saved to disk (`train.py`'s `best_state` only lives in
memory for the run's duration), so inspecting `pac_scale` required adding
per-layer wandb logging (`train.py`, in the epoch loop: `for i, block in
enumerate(model.encoder.blocks): if hasattr(block.mixer, "pac_scale"): ...`)
and rerunning. That rerun (job 13233599) landed test_kappa **0.5792** —
a large jump with *no functional code change* from the prior 0.5138 run
(same config, same `seed: 0`). That gap (0.5138 -> 0.5792 from supposedly
identical settings) was bigger than the effect we were trying to measure,
so it was investigated before trusting the number.

**Root cause: `seed` was never fully controlling the training run.**
`train.py` only called `torch.manual_seed` and `np.random.seed`. But
`models/augment.py:109` picks which augmentation to apply each forward pass
via Python's built-in `random.randint(...)` — the `random` module's seed was
never set anywhere, so its state was whatever the OS/interpreter initialised
it to, different every process launch. With one `random.randint` call per
training batch across 60 epochs, this alone was enough to make "the same
config, same seed" produce meaningfully different training trajectories.
GPU-side non-determinism (cuDNN algorithm auto-selection) is a smaller,
secondary contributor.

Fix (`train.py`, top of `main()`): seed `random`, `numpy`, `torch` (CPU
*and* CUDA via `torch.cuda.manual_seed_all`) from the same `cfg["seed"]`,
and set `torch.backends.cudnn.deterministic = True` /
`torch.backends.cudnn.benchmark = False`. Verified in isolation (running
`RandomAugment` twice from the same reset seed pair now yields an identical
augmentation-choice sequence, where before the fix it did not).

**Controlled rerun, same seed=0, both mixers (jobs 13247588 mi / 13247279
cotar):**

| Mixer | test_balanced_accuracy | test_kappa | test_f1_weighted |
|---|---|---|---|
| cotar | 0.5963 | 0.5107 | 0.6913 |
| mi (v2, per-channel + seed fix) | 0.6130 | **0.5199** | 0.6886 |

With randomness actually controlled, MI beats CoTAR by +0.0092 kappa — real
but modest, nowhere near the 0.5792 outlier (now understood to be an
especially favourable, unseeded augmentation draw, not a genuine
improvement). Single-seed still isn't enough to fully rule out this seed
itself being favourable; **recommended next step is 1-2 more seeds each
(e.g. seed 1, 2) before treating "MI > CoTAR on Sleep-EDF" as a settled
result** for writeup purposes.

**Resolved — seeds 1 and 2 (jobs 13357185/13357186 mi, 13357187/13357188
cotar):**

| seed | cotar test_kappa | mi test_kappa | mi lead |
|---|---|---|---|
| 0 | 0.5107 | 0.5199 | +0.0092 |
| 1 | 0.5003 | 0.5491 | +0.0488 |
| 2 | 0.4878 | 0.5373 | +0.0495 |
| **mean** | **0.4996** | **0.5354** | **+0.0358** |

MI beats CoTAR on **3/3 seeds**, no sign reversal, and seed 0 (the first
controlled run) turns out to be the *weakest* margin of the three, not a
favourable outlier — so the earlier caution about single-seed noise is now
resolved in MI's favor. **This is currently the most solid evidence in the
project that PAC-biased cross-band attention (v2 MI) genuinely outperforms
CoTAR on Sleep-EDF**, once (a) the channel-mean frontend bug (9.7) and (b)
the `random`-seed bug (9.8) are both fixed. Treat "MI > CoTAR on Sleep-EDF"
as settled for now; revisit only if a future architecture/data change
reopens the question.

### 9.9 MI v3 attempt — multi-head + gated injection (tried, reverted, no net gain)

Two changes were tried on top of the settled v2 result (jobs 13364126/27/554,
seeds 0/1/2): (a) made the cross-band Q/K/V multi-head (4 heads, matching the
`SelfAttention` baseline exactly, with a per-head learnable `pac_scale`
instead of one shared scalar) so that `pac_scale -> 0` degenerates to the
*same* 4-head attention being compared against, instead of a strictly weaker
single-head one; (b) replaced the concat+MLP redistribute step with a sigmoid
gate (`gate = sigmoid(W_g[token; core])`, `out = gate * core_proj(core)`)
added via the Block's residual, so a token can suppress cross-band injection
entirely (`gate -> 0`) instead of always absorbing it.

| seed | cotar | mi v2 | mi v3 (multi-head + gate) | v2 lead | v3 lead |
|---|---|---|---|---|---|
| 0 | 0.5107 | 0.5199 | 0.5347 | +0.0092 | +0.0240 |
| 1 | 0.5003 | 0.5491 | 0.5490 | +0.0488 | +0.0487 |
| 2 | 0.4878 | 0.5373 | 0.4892 | +0.0495 | +0.0014 |
| **mean** | 0.4996 | **0.5354** | 0.5243 | +0.0358 | +0.0247 |

**No net gain — reverted.** v3's mean kappa (0.5243) is *lower* than v2's
(0.5354) and its seed-to-seed spread nearly doubled (0.0292 -> 0.0598 range).
seed 0 improved, seed 1 was flat, but seed 2 collapsed to a near-tie with
CoTAR (+0.0014, down from v2's +0.0495) — still a 3/3 sweep, but a much less
clean one. Per-head `pac_scale` values did show real differentiation across
heads (e.g. seed1 layer5: heads 0.037/0.029/**0.185**/0.076) confirming the
multi-head split lets heads specialize, but overall magnitudes were much
smaller than v2's single-head values (v2 reached 0.4-0.6 in deep layers; v3
heads mostly stayed under 0.1) — the gate may be absorbing part of the
"should this token use cross-band info" decision that `pac_scale` alone used
to carry, without a net accuracy benefit and with added optimization
instability (more free parameters: per-head scale + gate projection, on a
dataset not large enough to reliably fit them).

`mi_operator.py` and `train.py`'s `pac_scale` logging were reverted to the
v2 (single-head, concat+MLP) design, which remains the reported result.
Multi-head + gating is not ruled out as a good idea in principle, but this
attempt did not validate it — revisit only with a clearer hypothesis for why
seed 2 collapsed, not as a blind retry.

### 9.10 NaN bug on 16-channel clinical EEG (TUEV/TUEP/TUSZ) — found, fixed

Running the (Sleep-EDF-winning) MI operator on TUEV and TUEP surfaced a real
bug, not a task-mismatch result: `train_loss` was `nan` from epoch 0, all
per-layer `pac_scale` values went `nan`, and the model collapsed to
predicting a single class (TUEV: balanced_accuracy exactly 1/6 = 0.1667,
f1≈0, kappa=0 — pure chance on 6 classes). On TUEP (binary) the same NaN
crashed the job outright (`roc_auc_score` raises on NaN input). Sleep-EDF (2
channels) and TUAB were never affected.

**Root cause:** `coupling_matrix`'s Ozkurt normalization divides by the
per-channel amplitude std (`denom = sqrt(mean(amp_c**2))`, clamped to
`1e-6`). On Sleep-EDF's 2 hand-picked channels this is safe, but 16-channel
clinical TUH montages routinely have flat/dead channels in parts of a
recording (electrode pop-off, saturation, artifact) — when a channel's
amplitude variance is near-zero, the division blows up, and one bad channel
in a 16-channel batch is enough to poison the whole batch via the einsum ->
softmax chain. This is a direct side effect of the v2->v3 per-channel-coupling
fix (§9.7): channel-averaging (old v2) diluted a bad channel's contribution
across all 16; per-channel computation (v3) isolates and amplifies it instead.

**Fix:** replaced the per-channel-std Ozkurt normalization with a **fixed
divisor** (`NORM_CONST = 100.0`, module-level constant in `mi_operator.py`)
on the real/4D (per-channel) path only — removes the division-by-near-zero
failure mode entirely; the learnable `pac_scale` absorbs whatever coupling
magnitude is left, so this is not expected to change Sleep-EDF behavior
materially (untested at time of writing — re-verify Sleep-EDF numbers are
still consistent with §9.8 after this change, not assumed). The 3D
(single-channel/synthetic) path used by `synth_pac_test.py`/`test_mixers.py`
is unchanged and still passes.

**Resolved (job 13460000):** `train_loss` was finite for all epochs (0
occurrences of `nan` in the log) — fix confirmed on real 16-channel data, not
just the synth/unit tests.

| Mixer | bacc | kappa | f1_weighted |
|---|---|---|---|
| attention | 0.4205 | 0.2925 | 0.6223 |
| cotar | 0.4227 | 0.2272 | 0.5591 |
| mi (v3, fixed) | 0.4193 | 0.2569 | 0.5843 |

mi is now a real, trustworthy number (previously 0.1667/0/0 — pure NaN-induced
chance) and lands between attention and cotar, still the weakest on kappa —
consistent with §9.4's "PAC doesn't fit TUEV" conclusion, just no longer
contaminated by a crashed run.

### 9.11 TUSZ three-way ablation — first real result (single seed)

Jobs 13460123 (attention) / 13460124 (cotar) / 13460125 (mi), seed 0, on the
BIOT-style-oversampled TUSZ data (§7). `eval.py`'s new `pr_auc` (§7) is the
primary metric given the ~2-6% positive rate; `train_loss` was finite
throughout all three runs (§9.10 fix holds on TUSZ too).

| Mixer | bacc | AUROC | PR-AUC |
|---|---|---|---|
| attention | 0.5748 | 0.7645 | 0.4262 |
| cotar | 0.6122 | 0.7987 | 0.4694 |
| mi | 0.5873 | 0.7985 | **0.4697** |

mi and cotar are effectively **tied** on PR-AUC/AUROC (both clearly beat
attention, mirroring the Sleep-EDF pattern of "structured mixer > plain
attention"), but unlike Sleep-EDF's clean 3/3-seed mi win, **mi does not
separate from cotar here**. Single seed only — per the seed-workflow
convention (dev/tuning = seed 0 only), do not treat this as settled; multi-seed
would be needed before drawing a conclusion either way.

### 9.12 IO bottleneck on TUAB/TUEV/TUEP/TUSZ — same class of bug as Sleep-EDF, fixed

Discovered while investigating why TUSZ attention (job 13460123) took 3h39m
(anomalously slow vs. other runs): TUAB/TUEV/TUEP/TUSZ all still use
`TUABLoader`/`TUEVLoader`'s one-`pickle.load`-per-`__getitem__` pattern
(`data/loaders.py`) — the exact IO-bound bottleneck `consolidate_sleepedf.py`
was written to fix for Sleep-EDF's ~128k files (small-file random-access
reads starve the GPU regardless of GPU speed). These four datasets have
**113k-416k files each** — as bad as or worse than Sleep-EDF was (TUAB
409,455; TUSZ 416,362; TUEP 184,739; TUEV 113,353).

**Fix:** generalized the consolidation approach. `scripts/consolidate_pkl_dataset.py`
packs per-window pkls into one `{split}_signals.npy` + `{split}_labels.npy`
pair per split (same convention as `consolidate_sleepedf.py`: written directly
into the dataset's root processed dir). New `TUABNpyLoader`/`TUEVNpyLoader`
(`data/loaders.py`) read these via `mmap_mode='r'`. `_tuab_sets`
(TUAB/TUEP/TUSZ, which all share TUABLoader's `{"X","y"}` format) and
`_tuev_sets` **auto-detect** the consolidated files and use the fast path if
present, otherwise transparently fall back to the original per-pkl loader —
no config or call-site changes needed, and nothing breaks for
not-yet-consolidated datasets.

Two bugs surfaced while building this, both fixed:

1. **OOM on TUAB.** The first version built the whole split array in RAM
   before writing (`np.empty((n, *shape))`); TUAB's train split alone is
   ~38GB as float32, which OOM'd a 32GB job (TUSZ's ~29.5GB train barely
   fit, was not a coincidence-free margin). Fixed by writing through a
   disk-backed memmap (`np.lib.format.open_memmap`) instead of an in-RAM
   array — peak RAM is now ~one sample regardless of split size.
2. **TUEV's val split was not actually reproducible across process launches**
   despite a fixed `rng.choice(..., seed=4523)` — see §9.13, found while
   consolidating TUEV and getting a different train/val split size than a
   fallback run in the same session.

**Status: done.** TUAB (job 13482061, memmap fix, 41min, no OOM), TUEV, and
TUSZ all consolidated successfully — verified `_tuab_sets`/`_tuev_sets` pick
up the npy files automatically and return correct sizes/shapes/dtypes/labels.
TUEP not consolidated (already judged not usable, §7). Any future
training run on these three datasets automatically gets the speedup with no code
changes.

### 9.13 TUEV val split was non-deterministic across process launches (PYTHONHASHSEED) — found & fixed

While consolidating TUEV (§9.12), the consolidation script's train/val
subject split (76772/7160) didn't match a `_tuev_sets` fallback run from the
same session (74813/9119) — same total (83932), different split. Root cause:

```python
train_sub = list(set(f.split("_")[0] for f in train_files))   # BUG
val_sub = rng.choice(train_sub, size=..., seed=4523)
```

`PYTHONHASHSEED` is unset in this environment (confirmed: `echo $PYTHONHASHSEED`
is empty), so Python randomizes string hashing **per process** by default.
`set()` iteration order for strings depends on hash values, so
`list(set(...))`'s order is different every time a new Python process starts
— even though `rng.choice`'s numeric seed (4523) is fixed, it draws from a
differently-ordered input array each run, so **the actual val subjects
selected differ from run to run**. Verified directly: three separate `python3
-c "print(list({'aaa','bbb',...}))"` invocations gave three different
orderings.

This is the same *class* of bug as the Sleep-EDF `random`-module seed issue
(§9.8): code that looks seeded but isn't actually deterministic across
process launches. **Practical impact:** every historical TUEV run (v1
frontend, v2 frontend, all three mixers, the §9.10 NaN-fix rerun) was
evaluated against a *different* random 10% subset of train subjects held out
as val, despite `seed=4523` appearing in the code. This does not overturn the
"PAC doesn't fit TUEV" conclusion (no run was systematically favored — it's
symmetric noise across mixers, not bias toward one), but it means TUEV
mixer-to-mixer comparisons have had an uncontrolled noise source this whole
time, on top of whatever comes from training-run seeding itself.

**Fix:** `train_sub = sorted(set(...))` instead of `list(set(...))`, in both
`data/loaders.py::_tuev_sets` and `scripts/consolidate_pkl_dataset.py`.
`sorted()` gives a process-independent order for `rng.choice` to sample from.
Verified: two separate process launches with the fix now select the identical
29 val subjects. TUEV consolidated npy files were regenerated with the fix
(job 13481333) after this was found — the file counts in §9.12 postdate the
fix. **Not yet re-verified:** whether this changes any of the historical TUEV
result numbers (§9.4/§9.10) beyond noise — not assumed either way; flag if
re-running TUEV comparisons and numbers move more than expected run-to-run
variance.

### 9.14 CHB-MIT three-way ablation — first result, mi wins clearly (single seed)

Jobs 13560489 (attention) / 13560490 (cotar) / 13560491 (mi), seed 0, on the
consolidated CHB-MIT npy data (§9.12-style `_tuab_sets` fast path — CHB-MIT
shares TUAB's loader/pkl format). All three early-stopped cleanly (patience
12), no NaN/OOM issues.

| Mixer | best-val epoch | val AUROC (peak) | test AUROC | test bacc | test PR-AUC |
|---|---|---|---|---|---|
| attention | 0 | 0.764 | 0.642 | 0.509 | 0.044 |
| cotar | 17 | 0.773 | 0.635 | 0.514 | 0.045 |
| **mi** | 5 | **0.862** | **0.735** | 0.529 | ~0.30+ |

mi is the clear winner here, unlike TUSZ (§9.11, mi/cotar tied) — test AUROC
0.735 vs. ~0.64 for the other two, and per-epoch val PR-AUC sits in the
0.27-0.36 range for mi vs. 0.02-0.24 for attention/cotar, on CHB-MIT's heavily
imbalanced event-level positive rate. attention is notably the worst: its best
val AUROC is at epoch 0 and degrades monotonically after, with val
balanced_accuracy stuck at ~0.50 (chance level) throughout — it does not
appear to learn a useful discriminative signal at all on this dataset.
balanced_accuracy for all three mixers stays low (~0.50-0.53) at the default
0.5 threshold given the class imbalance; AUROC/PR-AUC are the more informative
metrics here. Single seed only — per the seed-workflow convention (dev/tuning
= seed 0 only), multi-seed averaging would be needed before this is a settled
result for reporting.

---

## 10. Things to flag back to the user rather than deciding alone

- Any case where TUAB/TUEV preprocessing details aren't fully specified by
  the BIOT repo and require a judgment call.
- Any change to the MI operator's redistribute step (currently concat+MLP,
  matching CoTAR) — gating and matrix-mixing are open ablation choices, not
  closed decisions.
- Any numerical instability that survives the unit-complex-vector trick in
  Section 4 — don't patch it ad hoc, surface it.

---

## 11. Roadmap (post-2026-07-12 repositioning)

Per the §0 positioning update, the project's actual deliverable is an EEG
foundation model, not "mi beats cotar." Current/near-term plan:

1. **Finish validating the backbone design** via the mixer-swap ablation —
   this is still the right tool for testing individual ideas (does
   per-channel PAC coupling help, does multi-head/gating help, etc.), just
   not the paper's headline result. Sleep-EDF's mi-v3-beats-cotar result
   (§9.8) and the TUEV/TUEP/TUSZ negative-or-pending results are inputs to
   backbone design decisions, not competing claims against CoTAR.
2. **Complexity is a real asset for this positioning**: the MI mixer is
   sub-quadratic (§5's complexity note) — cross-band attention operates on a
   fixed n_bands=32 summaries regardless of sequence length, unlike standard
   self-attention's O(N²). Worth keeping front-of-mind when scaling to long
   pretraining sequences.
3. **Large-scale self-supervised pretraining** is the next phase once the
   backbone is settled (no concrete plan/objective/data recipe decided yet as
   of this writing — do not assume a specific pretraining design without
   re-confirming with the user).
4. **TUSZ/CHB-MIT** (§7): being pursued as additional PAC-relevant datasets in
   service of goal 1, not as a pivot to a different benchmark suite (that
   "switch to Medformer benchmark" idea was discussed and explicitly not
   adopted — the user chose to stay on the BIOT/EEG-corpus lineage and
   reframe the goal instead, per §0).
