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

> **Architecture status (2026-07-15):** the shipped code is the v1 single-mixer
> design; the **plan of record is the §13 v2 spec** (tri-axial space/frequency/time
> tokenization, coupling as the frequency-axis mixer, coupling-based SSL as the
> keystone). §§3-5 describe the *current code*, not the target. Read §13 and
> §9.17 before making architecture decisions.

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

**3.2 Coupling caching.** The MI operator's band×band coupling matrix depends
only on the frontend's phase/amplitude, so `Encoder.forward` computes it once
and threads it to every block as `coupling=` (guarded by `hasattr(mixer,
"coupling_matrix")`, so attention/CoTAR are unaffected and the swap stays
clean). MI still recomputes it itself if called standalone without `coupling`
(the unit tests do this). Saves `depth−1` coupling einsums per forward.

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

> **Superseded 2026-07-15 — read §13 first.** This section documents the v1..v6
> single-mixer line, which is closed (§9.17: v5 lost the ceiling; the PAC prior
> was not carrying the supervised wins; the coupling was being averaged into
> mush). The architecture of record is the **§13 tri-axial v2 spec**, where this
> operator becomes the *frequency-axis* mixer with time-resolved, per-channel
> coupling. The description below still matches the code in
> `models/mixers/mi_operator.py` (currently v6, written but never trained).

**v6 (in code, not the plan of record): convex mix of an attention branch and
the v3 band-level operator.** Two branches plus an input-conditioned gate:

```
core_attn = MultiHeadSelfAttention(x)              # branch A: full token-level attention
core_pac  = redistribute( coupling-weighted cross-band aggregate of V )  # branch B: PAC
g[b, j]   = sigmoid( gate_mlp( coupling_stats(b, j) ) )   # per-(sample, target-band) in (0,1)
out       = core_attn  +  g * core_pac             # residual added by the block
```

Branch B, in detail: `V = v_proj(band_repr)` (band_repr = per-band mean over
patches); `weight[j,i] = softmax_i(coupling_scale · coupling[i,j])`;
`core_pac_j = Σ_i weight[j,i]·V_i`; broadcast over patches, then a CoTAR-style
`concat([x_b, core_pac]) → MLP` redistribute. `coupling` is the Canolty (2006)
MVL score with mean-centered amplitude debiasing (removes the spurious
low-frequency term from amplitude being strictly positive), **normalized by a
fixed constant (`NORM_CONST = 100.0`)** rather than the per-channel amplitude
std (see §9.10 — the std-based Ozkurt normalization divides by ~0 on flat/dead
channels in 16-channel clinical EEG and produced NaN). Applies only to the real
(per-channel, 4D) path; the single-channel/synthetic 3D path used by unit tests
still uses std-based normalization. **The coupling matrix depends only on the
frontend's phase/amplitude, so `Encoder` computes it once and threads it to
every block as `coupling=` — no per-layer recompute (§3.2).**

Why this shape (both problems solved by one change):
- **Floor ≥ baseline.** `core_attn` is *bit-for-bit* the vanilla-attention
  baseline mixer (`models/mixers/attention.py`), so when `g → 0` the PAC branch
  switches off and MI collapses to exactly the attention baseline — not to the
  crippled single-head band-level attention the old additive-bias design
  degenerated into on non-PAC data (that was the mechanism behind "MI平庸 on
  non-PAC datasets"). *Verified numerically*: with the gate forced to 0, MI's
  output equals a weight-tied `SelfAttention` to 0.0.
- **Ceiling preserved.** When `g` is high the directional PAC aggregation is
  injected → the physiological prior on PAC-strong tasks.
- **Adaptive & input-conditioned.** `g` is a function of 3 *scale-invariant*
  statistics of the coupling column for each target band (mean drive, peak
  drive, peakedness = peak−mean, after dividing the matrix by its own mean), so
  the model *detects* whether real cross-frequency structure is present per
  input/band and dials the prior accordingly. No longer a frozen per-layer
  scalar. Diagnostic: train.py logs `gate/layer{i}` = mean g per layer.
- **Not an attention variant.** PAC is a separate gated structural operator, no
  longer a bias folded into the QK logits — directly addresses the "looks like
  a self-attention variant" critique.

**Complexity note (deliberately O(N²) as of v5).** Branch A is full
token-level multi-head self-attention over N = n_bands·P tokens → O(N²·D).
This is an *intentional* trade made 2026-07-14 (PI steer): the older design
kept the cross-band attention on band-level summaries only (O(n_bands²),
sub-quadratic in sequence length) but never saw patch-level detail, and its
floor sat below baseline. We spend the O(N²) to (a) get patch-level attention
and (b) guarantee floor = attention baseline. Branch B stays O(n_bands²), and
the coupling einsum (O(n_bands²·C·T), now computed once for all layers) is the
only sequence-length term left in the PAC path. If the sub-quadratic property
is wanted back for long-sequence pretraining, branch A can later be swapped for
a linear-attention kernel without touching branch B or the gate.

**Deprecated designs:** v1 = fixed coupling matrix as softmax weights, no
learned Q/K/V (tied CoTAR, couldn't exceed). v3/v4 = PAC as an additive bias
`pac_scale · coupling[i,j]` on QK logits with a learned per-layer scalar
`pac_scale` (good ceiling on PAC-strong data, but floor below baseline on
non-PAC data + reads as an attention variant — the two problems v5 fixes).

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

### 9.15 MI redesign v5 — adaptive gated PAC branch over an attention floor

**Motivation.** Across the six-dataset sweep (§9.1–9.14) the v3/v4 MI operator
showed a consistent shape: it wins where PAC is physiologically load-bearing
(Sleep-EDF §9.8, CHB-MIT §9.14) but is *平庸 / no better, sometimes worse* on
datasets where PAC is not the driving signal (TUAB §9.3, TUEV §9.4). Root-cause
analysis (PI review, 2026-07-14) pinned this on the operator's structure, not
its hyperparameters:

- v3/v4 added PAC as an **additive bias on the QK logits**, scaled by a single
  learned per-layer scalar `pac_scale` (init 1.0). On non-PAC data training just
  drives `pac_scale → 0`, at which point MI is a **single-head, band-level**
  attention over `band_repr` (per-band mean over patches). But the *attention
  baseline* it's compared against is **multi-head, token-level**. So the
  degenerate MI is strictly *weaker* than baseline — the floor sat **below**
  baseline, which is exactly the "non-PAC 平庸" symptom.
- Secondary problem: "PAC = attention + a bias term" reads as a trivial
  self-attention variant (a repositioning/novelty risk, per the 7.13 meeting).

**Change (both problems, one redesign).** MI is now two parallel branches + an
input-conditioned gate (full spec in §5):

```
out = MultiHeadSelfAttention(x)  +  g · redistribute(PAC-coupling aggregate)
g   = sigmoid(gate_mlp(scale-invariant coupling stats)), per (sample, target band)
```

- **Floor = baseline, by construction.** Branch A is bit-for-bit the vanilla
  attention mixer; `g → 0` ⇒ MI ≡ attention baseline (verified: forced-zero gate
  gives `max|MI − weight-tied SelfAttention| = 0.0`). So non-PAC data can no
  longer push MI below baseline.
- **Ceiling kept.** `g` high ⇒ directional PAC aggregation injected.
- **Adaptive.** `g` reads three scale-invariant statistics of the coupling
  column per target band (mean drive, peak drive, peakedness) — it *detects*
  whether real cross-frequency structure exists in this input, per band. Directly
  implements the 7.13 "自适应调节对 PAC 先验的依赖" ask. train.py logs
  `gate/layer{i}`.
- **Not an attention variant.** PAC is a separate gated structural operator now,
  not entangled in the QK logits.

**Complexity.** Deliberately O(N²) now (branch A is token-level attention) —
a trade the PI signed off on: spend the quadratic cost to get patch-level
attention + a guaranteed floor, giving up the old sub-quadratic-in-seq-len
property. Branch A is swappable for a linear-attention kernel later if
long-sequence pretraining needs it back (§11). Coupling einsum is now computed
once per forward and reused across layers (§3.2), not `depth` times.

**Validation done (CPU, pre-EEG, all green):** `scripts/test_mixers.py`
(interface + finite grads), `scripts/synth_pac_test.py` (still localizes the
10→60 Hz synthetic coupling — `coupling_matrix` unchanged), full-model
forward/backward on synthetic data (every MI param gets finite non-None grad),
and the floor + caching equalities above. **Not yet run on real EEG** — the
six-dataset re-sweep with v5 is the next step; expectation to check is
"non-PAC datasets (TUAB/TUEV) now ≥ attention baseline (floor fix) while
PAC-strong (Sleep-EDF/CHB-MIT) keeps mi's lead (ceiling)."

### 9.16 CoTAR baseline sanity check — is CoTAR crippled, or just disadvantaged?

**Risk (PI review).** In the band-frontend ablations CoTAR sometimes trails
even vanilla attention (TUEV §9.4: CoTAR kappa 0.2272 < attention 0.2925). A
reviewer's first read of that is "you didn't implement/tune CoTAR properly",
which would taint the "MI > CoTAR" claims elsewhere. We need to separate
"baseline is broken" from "baseline is task-disadvantaged".

**Finding (code audit).** `models/mixers/cotar.py::CoTAR.forward` is numerically
equivalent to TeCh's `layers/Transformer_EncDec.py::CoTAR.forward`
(aggregate-MLP → softmax over tokens → single core → concat+MLP redistribute,
`d_core = d_model//4`) — **no port bug**. The block is also genuinely post-norm
like TeCh (only the old docstring wording said "Pre-mixer-norm"; fixed in
`block.py`). CoTAR's weakness is architectural: (a) our frontend adds **no
positional encoding** (TeCh does), which hurts CoTAR most since it's a
permutation-invariant global pool; (b) our band-preserving frontend deliberately
keeps 32-band structure for MI, which CoTAR immediately collapses to a rank-1
core — CoTAR is structurally disadvantaged *exactly where MI should win*.

**Check.** To show CoTAR is a healthy baseline in its *native* token space, give
both attention and CoTAR the plain `conv` frontend (bag-of-patch tokens, no band
split) and compare head-to-head. Configs: `tuev_diag_conv_{attention,cotar}.yaml`
(the embarrassing case) and `tuab_diag_conv_cotar.yaml` (pairs with the existing
`tuab_diag_conv.yaml` attention run, §9.2). Pass criterion: conv CoTAR ≳ conv
attention.

**Result — PASSED, and the sign flips (TUEV, seed 0, jobs 13933318/13933319):**

| frontend | attention kappa | cotar kappa | cotar − attention |
|---|---|---|---|
| band (main ablation, §9.4) | 0.2925 | 0.2272 | **−0.0653** (CoTAR loses) |
| conv (CoTAR's native token space) | 0.2303 | **0.2501** | **+0.0198** (CoTAR wins) |

Full conv-frontend test metrics — attention: bacc 0.3979 / kappa 0.2303 /
f1_w 0.5696; cotar: bacc 0.4088 / kappa 0.2501 / f1_w 0.5852. **CoTAR is ≥
attention on all three test metrics**, and val agrees (kappa 0.3596 vs 0.3263),
so it is not a single-metric artifact.

**Reading.** The *ordering* between CoTAR and attention reverses purely by
swapping the frontend, with the mixer code untouched. That is direct evidence
for the audit conclusion: **CoTAR is implemented correctly and behaves like a
healthy baseline when given bag-of-patch tokens; it under-performs in our main
ablation because the band-structured frontend disadvantages it** (it is a
permutation-invariant global pool, gets no positional encoding from our
frontend, and collapses the 32-band structure to a rank-1 core). So "MI >
CoTAR on band-structured PAC tasks" is a fair win over a working baseline, not
an artifact of a broken one — which was the review risk this check existed to
close.

**Caveats.** Single seed; the conv-space margin is small (+0.02 kappa). Note
also that TUEV conv attention (0.2303) is *worse* than band attention (0.2925)
— on TUEV the band frontend helps attention, so conv is not a strictly stronger
setup, it is just CoTAR's fairer one. This check does **not** touch the main
ablation (band frontend, same-backbone swap) — that setup is correct as-is.

**TUAB second data point (job 13933320):** conv CoTAR 0.7940 bacc / 0.8698
AUROC vs conv attention's 0.8023 / 0.8765 (§9.2) — essentially a tie, CoTAR a
hair below. So the strong form ("conv CoTAR always ≥ conv attention") does
*not* hold; the honest claim is **CoTAR is within noise of attention in its
native token space and clearly ahead on TUEV, i.e. it is a working baseline,
not a crippled one** — which is all this check needed to establish.

### 9.17 v5 failed on the PAC-strong datasets — and the two findings that explain it

**v5 result (seed 0, jobs 13933313-17), against the historical baselines:**

| dataset | v5 mi | reference | verdict |
|---|---|---|---|
| Sleep-EDF kappa | 0.5101 | v3 mi 0.5199 / cotar 0.5107 | ✗ v3's lead gone |
| CHB-MIT test AUROC | **0.5721** | v3 mi 0.735 / attn 0.642 / cotar 0.635 | ✗✗ **below both baselines** |
| TUAB bacc / AUROC | 0.7954 / 0.8748 | attention 0.7959 / 0.8764 | ≈ tie (floor held, no gain) |
| TUEV bacc / kappa | 0.4334 / 0.2825 | attention 0.4205 / 0.2925 | ≈ mixed |
| TUSZ AUROC / PR-AUC | 0.8001 / 0.5052 | v3 mi 0.7985 / 0.4697 | ✓ marginal gain |

v5's floor fix worked (TUAB/TUEV now match baseline) but **cost the ceiling**,
catastrophically on CHB-MIT. Gate diagnostics say why: on CHB-MIT `g` froze at
**~0.50** in 5 of 6 layers (0.9217, 0.5018, 0.4961, 0.4974, 0.4996, 0.5017) —
the additive form `attn + g·pac_delta` cannot express "pure band-level
operator", which is what v3 actually won with, so the gate just straddled and
did neither well. Sleep-EDF's gate did differentiate (0.20-0.85) and still lost
the ceiling.

**Finding 1 — the PAC prior was never carrying the supervised wins.**
`pac_scale` trajectories from the v3 runs:

| dataset | v3 `pac_scale` converged | v3 mi score | attention |
|---|---|---|---|
| Sleep-EDF (job 13247588) | 0.17 / 0.25 / 0.28 (nonzero) | kappa 0.5199 | cotar 0.5107 |
| CHB-MIT (job 13560491) | **5e-05 / -9e-05 / 0.0** (collapsed by ep4) | **AUROC 0.735** | **0.642** |

CHB-MIT's best checkpoint (epoch 5) already had `pac_scale ≈ 0`, yet beat
attention by ~0.09 AUROC. **So the band-level bottleneck — mean-pool the P
patches per band, mix across the n_bands summaries, broadcast back — is the
asset, not the PAC bias.** This kills the review-facing claim "we wrote PAC
into the mixer and that is why we win", on our own evidence. It does *not*
show PAC is useless: it shows **single-dataset supervised training has no
incentive to learn a cross-frequency mechanism** — it fits labels via the
cheapest route available. A prior only pays off if the *objective* demands it,
which is the core argument for making the SSL objective the keystone (§13).

**Finding 2 — we were computing the coupling into mush.**
`coupling_matrix` runs the einsum over the **entire** window and then
`.mean(dim=1)` over channels → **one 32×32 matrix per sample**, averaged over
16 electrodes and 2000 timesteps. Seizures are **focal** (few electrodes) and
**transient** (seconds). Averaging over all channels and all time leaves a
statistic that is near-constant across samples, i.e. carrying almost no
discriminative information. `pac_scale → 0` on CHB-MIT is then the *correct*
thing for the optimizer to do. **This is not the prior being wrong, it is the
prior being averaged away** — and it is a plain bug-class defect that should be
fixed regardless of what happens to the rest of the redesign. Fix: compute
coupling per **(channel, time-patch)**, which falls out naturally of the
tri-axial grid in §13.

**Status.** v5 is abandoned. A v6 (convex mix `(1-g)·attn + g·v3_band_op`,
spanning both endpoints exactly — verified `g→0` ≡ attention baseline and
`g→1` ≡ v3 operator to 0.0) was written and validated on CPU but **never
submitted**: the PI called a halt to redesign the architecture wholesale for
foundation-model readiness rather than keep patching the mixer (§13). The v6
code is the current state of `mi_operator.py` and is a reasonable fallback if
§13 stalls, but it does not address Finding 2 and is not the plan of record.

---

## 10. Things to flag back to the user rather than deciding alone

- Any case where TUAB/TUEV preprocessing details aren't fully specified by
  the BIOT repo and require a judgment call.
- The MI operator's two-branch structure (attention floor + gated PAC branch,
  §5/§9.15) and the gate's conditioning signal (scale-invariant coupling
  column stats) are the current committed design as of v5. The redistribute
  step (concat+MLP) and matrix-mixing remain open ablation choices; changing
  the branch structure or the floor-equals-baseline guarantee is a bigger
  decision — flag it.
- Any numerical instability that survives the unit-complex-vector trick in
  Section 4 — don't patch it ad hoc, surface it.

---

## 11. Roadmap

> **Superseded 2026-07-15.** The plan of record is now the **§13 v2 architecture
> spec** (tri-axial tokenization + coupling on the frequency axis + coupling-based
> SSL). The mixer-patching line this roadmap was written around (v1..v6) is
> closed — see §9.17 for why. Kept below for history; where it conflicts with
> §13, §13 wins.

Per the §0 positioning update, the project's actual deliverable is an EEG
foundation model, not "mi beats cotar."

1. ~~**Finish validating the backbone design** via the mixer-swap ablation.~~
   Done and concluded (§9.1-9.17). Outcome: the ablation is kept but *narrowed*
   to the frequency axis (§13.8), and its headline finding is uncomfortable —
   the PAC prior was not carrying the supervised wins (§9.17 Finding 1).
2. ~~**Complexity:** MI is O(N²) by deliberate choice (v5's token-level floor).~~
   Moot — v5 is abandoned. §13 recovers cheap mixing structurally via tri-axial
   factorization (frequency axis back to O(n_bands²), constant in sequence
   length), and explicitly rejects Mamba for the time axis (§13.5).
3. **Large-scale self-supervised pretraining** — no longer "the phase after the
   backbone is settled". Per §13.2 it is the **keystone**: the phase-conditioned
   amplitude reconstruction objective (§13.7) is what makes the frequency prior
   load-bearing at all. **First result in (§13.10, 2026-07-18/19): crossfreq
   masking beats random-mask MAE on 3/5 datasets (TUAB/CHB-MIT/TUSZ), loses on
   2/5 (TUEV/Sleep-EDF)** — partial, single-seed evidence the keystone bet pays
   off; not yet settled. Full recipe/schedule (curriculum across the other
   §13.7 objectives, multi-dataset joint pretrain) still undecided.
4. **TUSZ/CHB-MIT** (§7): additional PAC-relevant datasets, not a pivot to a
   different benchmark suite (the "switch to Medformer benchmark" idea was
   discussed and explicitly not adopted — stay on the BIOT/EEG-corpus lineage
   and reframe the goal instead, per §0).
5. **Build order for §13** (staged, per §13.9): (a) tri-axial skeleton + physics
   PE, attention on all three axes — prove montage-agnostic and no regression;
   (b) swap coupling into the frequency axis, re-run the narrowed ablation;
   (c) pretraining. Note §13.9(4): historical baselines are **not** comparable
   to v2 and must be re-run on the v2 skeleton.

---

## 12. PI meeting notes (chronological, verbatim from user)

Kept in the original Chinese as given by the user each time, not translated
or reworded — these are the source-of-truth record of what was reported to
and discussed with the PI, separate from this doc's own experimental log
(§9) and roadmap (§11).

### 2026.6.27

一、上次会议后做的工作

搭建PAC Former codebase包括可学习带通滤波器+可微Hilbert变换提取频带相位/幅度 实现三个可互换的token mixer(包含核心设计的differentiable directional Modulation Index operator)

接入了eeg数据集TUAB和TUEV,完成预处理,跑通了三个mixer在两个数据集上的训练流程,接入wandb做实验追踪

二、讨论内容

初步结果显示MI算子比self-attention稳定地强 但暂时还没超过CoTAR

TUEV上训练不稳定(过拟合,验证集指标噪声大),可能和类别严重不平衡有关

三、后续计划

修复TUEV的训练不稳定问题后重跑,确认MI算子的排名是否依然成立。

针对MI算子做消融实验(频带数量、重分配方式:concat+MLP/gating/matrix-mixing),看能否拉近和CoTAR的差距。

跑Tier-2 baseline(EEGNet等)和整理LaBraM/BIOT等文献数字,确认我们的频域encoder整体处在合理区间。

### 2026.7.2

一、上次会议后做的工作

调整模型backbone结构（频带数、隐藏维度、训练策略对齐）在TUAB、TUEV上跑到接近BIOT论文同量级的指标阈值。

换数据集到Sleep-EDF，跑了两版mi，mi和cotar的效果都略高于self-attention，且mi收敛更快。

二、讨论内容

数据集选择上TUAB、TUEV上PAC先验与任务不匹配，是否需要切换PAC生理学意义更明确的数据集。

当前mi算子设计相比cotar势不够明显，应该在各数据集上做优化而不是盲目切换数据集。

三、后续计划

继续修改模型架构：优化frontend，并尝试引入gating机制。

引入更多数据集验证mi的泛化表现，为后续做pretrain做准备。

### 2026.7.13

一、上次会议后做的工作

搜集及处理新数据集，为实验准备。

模型迭代到v4版本，在六个数据集上进行实验（CHB-MIT、Sleep-EDF、TUAB、TUEP、TUEV、TUSZ）。

二、讨论内容

v3版本目前表现最佳，但优势和数据集强相关：在PAC先验强的数据集上效果突出，在不太依赖PAC先验的常规EEG数据集上表现平庸，没有明显优势。

模型结构讨论了两点：一是当前设计容易被误解为self-attention的小变种，需要调整结构的顺序或在表述上凸显核心设计；二是复杂度目前是线性的，还有优化空间。

三、后续计划

继续改模型架构，方向可以是使其能自适应调节对PAC先验的依赖程度，在非PAC数据集上也表现更好。

寻找baseline，并筛选适合pretrain的数据集，为后续预训练做准备。

---

## 13. PAC-Former v2 architecture spec (decided 2026-07-15; skeleton built + running)

**Status: design of record. Supersedes the v1..v6 mixer-patching line (§9.15-9.17).**
The decision to stop patching the mixer and redesign wholesale came from the PI
on 2026-07-15: the mixer line was "应用面太局限" — it only made sense as a narrow
PAC-former, not as an EEG foundation model, which is the actual deliverable (§0).

**Built 2026-07-15** (supervised path; SSL/pretraining §13.7 NOT built yet):
- `models/frontend/triaxial.py` — `TriAxialFrontend` (grid tokens + patch-resolved
  per-channel coupling, §13.3/13.6) and `patch_coupling()`.
- `models/triaxial.py` — `BandPE`/`SpatialPE`/`rope`, axis mixers
  (`FreqCoupling`/`FreqAttention`/`FreqCoTAR` + `_MHA`), `TriAxialBlock`,
  `TriAxialEncoder`.
- `models/build.py` — `TriAxialPACFormer`, selected by `cfg["arch"]=="triaxial"`;
  v1 `PACFormer` kept intact. Frequency-axis mixer chosen by `cfg["freq_mixer"]`.
- Configs `configs/{chbmit,sleep,tuab,tuev,tusz}_v2_{coupling,attention,cotar}.yaml`
  (n_bands=8, batch=32 — the channel axis is restored so tokens are ~C× more).
- CPU-verified: all 3 freq mixers forward/backward with finite grads; coupling is
  time-resolved AND channel-specific; C=19 runs (encoder channel-dynamic). GPU
  canary (CHB-MIT coupling, 1.68M params) trains without OOM at batch 32.

**Supervised validation sweep running (seed 0):** coupling jobs 14117818-22
(chbmit/sleep/tuab/tuev/tusz), baselines 14117990-96. Reminder (§13.9-4): these
are NOT comparable to the §9.1-9.14 numbers (those used n_bands=32 + the
channel-collapsing frontend) — the v2 attention/cotar freq-mixer runs ARE the
right baselines. Reading the sweep (§13.2): supervised evidence tests the
*skeleton health* and *Finding-2 fix* (does time-resolved coupling now beat
attention on the freq axis, esp. CHB-MIT/TUSZ), NOT the SSL keystone.

Known simplifications in this build, to revisit before pretraining:
- `SpatialPE` is a learned per-index embedding, not yet the coordinate MLP
  (§13.4). Fine for single-montage validation; swap in coords for cross-montage.
- Coupling is recomputed inside the frontend each forward (not cached like the
  v1 encoder did); cheap relative to the tri-axial attention, revisit if needed.

### 13.1 Thesis

Existing EEG foundation models (BIOT, LaBraM, CBraMod, REVE) tokenize by
**time segment** and let the model discover frequency structure implicitly.
PAC-Former v2 makes the three physical axes of EEG — **space (electrode),
frequency (band), time (patch)** — explicit token dimensions, mixes along each
axis with a physically-appropriate operator, and **pretrains with an objective
that forces cross-frequency mechanism learning**. The frequency axis is mixed
by a directional coupling operator; it is always on and is the only channel
through which bands exchange information — no attention fallback path.

### 13.2 Novelty stack (four layers; each is load-bearing)

1. **Frequency-prior tri-axial tokenization.** Tokens are
   (electrode × band × time-patch), with explicit analytic-signal phase and
   amplitude carried alongside. Frequency is a *prior*, not a posterior.
2. **Time-resolved, channel-specific directional coupling** as the frequency
   axis mixer. Unlike QK attention (symmetric learned *similarity*), coupling
   is asymmetric and *mechanistic*: "does band i's phase drive band j's
   amplitude". Direction is physically real (slow gates fast).
3. **Phase-conditioned amplitude reconstruction (SSL) — the keystone.** See
   13.6. This is what turns layers 1-2 from decorative into load-bearing.
4. **Interpretability, for free.** The operator emits a time-resolved,
   per-electrode comodulogram at every layer — a readout neuroscientists
   already use. Competing foundation models are black boxes.

**Honest position on novelty (do not let this get lost):** on *supervised*
evidence alone, layer 2's claim is weak — §9.17 Finding 1 shows `pac_scale→0`
on CHB-MIT while still beating baseline, i.e. the band-level bottleneck, not
the PAC prior, won. Layer 3 is the answer to that, not a nice-to-have: a prior
only pays off when the objective demands it. **If the SSL objective does not
deliver, the novelty falls back to a prior our own ablations call weak. That is
the bet, and it should be taken with eyes open.** Note however that §9.17
Finding 2 (coupling averaged into mush) is an independent, plain defect — fixing
it is justified regardless of how the bet resolves.

### 13.3 Token grid

Worked example: 16 electrodes, 10 s @ 200 Hz, n_bands=8, patch_len=200.

```
raw (C=16, T=2000) + electrode coords (C, 3)
  --sinc filterbank (learnable cutoffs)-->  (16, 8, 2000)
  --differentiable Hilbert-->  phase_unit (16, 8, 2000), amplitude (16, 8, 2000)
  --per-(channel,band) conv patch tokenizer, stride=patch_len-->
       token grid (C=16, B=8, P=10), each d_model=128   = 1280 tokens
```

Each token means "electrode c, band b, second p".

**Critical change vs v1:** the current frontend convs the 16 channels *away*
(mixes them into hidden). v2 must **not** — the channel axis stays explicit,
otherwise (a) variable montages are impossible and (b) focal signal is averaged
away (§9.17 Finding 2).

**n_bands 32 → 8 (decided).** Token count is C·B·P, so bands are a 4x memory
multiplier: 16·8·10 = 1280 tokens vs 16·32·10 = 5120. Physiologically the
canonical bands (delta/theta/alpha/beta/low-gamma/high-gamma) are ~6-8; 32 was
only ever chosen to "give the mixer leverage" (§9.3-era) and is poor value in a
foundation-model setting.

### 13.4 Physics-aware positional encoding (closes the montage gap)

```
spatial : MLP(electrode xyz)          -> d_model
band    : MLP(center_freq, bandwidth) -> d_model
time    : RoPE over patch index
```

Encode electrodes by **coordinate, not index** — "channel 3" means nothing
across datasets, but xyz is universal. This is what makes the model
montage-agnostic and removes `n_channels` as a hardcoded hyperparameter (the
gap called out in the PI's competitive analysis; REVE-style in spirit).

Encoding bands by **center frequency, not index** buys the same flexibility on
the frequency axis: the filterbank becomes swappable (pretrain with one bank,
finetune with another).

### 13.5 Tri-axial encoder block

Each block mixes along one axis at a time (norm + residual per sub-mixer, FFN
at the end). Inspired by Crossformer / CBraMod's criss-cross and iTransformer's
variate-as-token, extended to three axes — the frequency axis being ours.

| # | axis | operator | applied over | cost |
|---|---|---|---|---|
| 1 | time | **RoPE self-attention** | P patches, per (c, b) fiber | O(P²), P=10 |
| 2 | space | self-attention + spatial PE | C electrodes, per (b, p) | O(C²)=256 |
| 3 | frequency | **directional coupling operator** | B bands, per (c, p) | O(B²)=64 |

**Why factorized:** full attention over all 1280 tokens is ~1.6M pairs.
Factorized, the three axes cost 100 + 256 + 64. **The factorization — not any
exotic sequence operator — is what wins the complexity fight.** The frequency
axis stays O(B²), i.e. constant in sequence length; the compute-primitive
advantage over standard O(N²) foundation models is preserved.

**Time axis = attention, NOT Mamba (decided 2026-07-15).** Mamba was considered
and rejected: (a) P≈10-30 (≤300 even for a 5-min window) — attention is already
free there, Mamba's linear-vs-quadratic edge needs P≫1000; (b) Mamba is causal,
our task is offline, so it would need a bidirectional variant for no gain;
(c) our frontend is *already* non-causal (whole-signal FFT Hilbert, centered
201-tap sinc kernels), so Mamba's streaming advantage is moot by construction;
(d) `mamba-ssm` needs compiled CUDA kernels — a real dependency risk against
torch 2.8.0+cu126 — for zero payoff; (e) it adds no novelty (commodity in 2026)
and introduces a confound that directly weakens attribution of our actual claim
("how much of the gain is just Mamba?"). **Keep the infrastructure boring so the
frequency-axis ablation stays sharp.** Hedge: implement the time-axis mixer
behind the same swappable interface as the frequency-axis mixer, so switching
later (sample-level resolution, or hours-long context) is a config change.

### 13.6 Coupling, computed properly

```
v1 (broken): einsum over the whole window, then .mean over channels
             -> ONE 32x32 matrix per sample          <- averaged into mush
v2:          computed within each patch, per channel
             -> coupling[c, p] : (16, 10, 8, 8)      <- time-resolved, focal-preserving
```

The frequency mixer at (c, p) uses `coupling[c, p]`. The space axis then
aggregates across electrodes (so focal events survive rather than being
pre-averaged), and the time axis propagates across patches. The MVL math itself
(unit complex phase vector, mean-centred amplitude debiasing, no `atan2`) is
unchanged and still validated by `scripts/synth_pac_test.py`.

### 13.7 Pretraining objectives

- **Keystone — phase-conditioned amplitude reconstruction.** Mask the
  *amplitude envelope* of a (channel, high band, patch); require reconstruction
  from the **other bands' phase** plus spatio-temporal context. The only
  structure that can solve this is phase→amplitude coupling, so the objective
  trains the mechanism directly and the model has no cheap route around it
  (this is the direct answer to §9.17 Finding 1).
- **Masked token reconstruction** (standard) — general representation.
- **Cross-electrode reconstruction** — mask an electrode, rebuild from
  neighbours — trains the space axis.

Mixed curriculum. No recipe/schedule decided yet.

### 13.8 Ablation contract (preserved, sharpened)

The old contract was "same backbone, swap the mixer". The new one is **"same
backbone, swap only the frequency-axis mixer"** (coupling / attention / cotar),
with the space and time axes held fixed. This is a cleaner scientific question
than before: *given identical spatial and temporal modelling, does the
directional coupling operator beat plain attention on the frequency axis?*
`models/mixers/base.py`'s TokenMixer contract needs restating in these terms —
it is now an axis mixer, not a global one.

### 13.9 Risks / open items

1. **Memory.** Token count C·B·P with the channel axis restored. n_bands=8
   keeps it at 1280; watch it if C or P grows.
2. **The bet.** See 13.2 — if the SSL objective does not pay, novelty rests on a
   prior our own supervised ablations call weak.
3. **Scope.** This is a rewrite of frontend output shape, encoder, PE, and the
   mixer contract — not a patch. Stage it: (a) tri-axial skeleton + PE with
   attention on all three axes, prove montage-agnostic + no regression;
   (b) swap coupling into the frequency axis, re-run the ablation;
   (c) pretraining.
4. Historical baselines (§9.1-9.14) were measured with n_bands=32 and the
   channel-collapsing frontend. **They are not directly comparable to v2
   numbers** — baselines must be re-run on the v2 skeleton before any claim.

### 13.10 First SSL-keystone result — coherence-gate backbone fails again, crossfreq MAE wins 3/5 (2026-07-18/19)

Two independent experiments, both single seed (seed 0):

**(a) Supervised sweep — new freq-mixer primitive.** `FreqCoherenceGate`
(`models/triaxial.py:158`, `freq_mixer="coherence"`) multiplies the softmax
attention probabilities by a coupling-derived sigmoid gate and renormalizes,
instead of `FreqCoupling`'s additive `pac_scale · coupling` bias (§9.17). Init
`gate_w=0` → uniform gate → exactly plain attention at init, so it can only
switch gating on if it helps — same graceful-degradation intent as v5's floor
guarantee (§9.15), applied to the multiplicative form instead. Jobs
14213997-14214001 (`configs/{ds}_v2_coherence.yaml`), vs. the existing v2
attention/coupling/cotar baselines (§13, jobs 14117818-96):

| dataset | coherence | attention | coupling | cotar |
|---|---|---|---|---|
| TUAB (bacc/auroc/pr_auc) | 0.797/0.868/0.858 | 0.794/0.867/0.865 | 0.797/0.872/0.869 | — |
| CHB-MIT (bacc/auroc/pr_auc) | 0.500/0.532/0.018 | (run failed) | 0.500/0.526/0.018 | 0.500/0.645/0.047 |
| Sleep-EDF (bacc/f1/kappa) | 0.622/0.689/0.509 | 0.601/0.692/0.511 | 0.626/0.702/0.531 | 0.619/0.731/0.572 |
| TUEV (bacc/f1/kappa) | 0.487/0.650/0.344 | 0.525/0.660/0.376 | 0.515/0.649/0.370 | — |
| TUSZ (bacc/auroc/pr_auc) | 0.631/0.829/0.583 | 0.697/0.826/0.577 | 0.654/0.835/0.605 | — |

No win anywhere; worse than at least one existing baseline on Sleep-EDF, TUEV,
TUSZ. **Same pattern as v5 (§9.17): a new competing-layer design for PAC,
evaluated under plain supervised training, does not separate from attention.**
This is now three redesigns in a row (v4 gating, v5 attn+gated-PAC-branch,
coherence multiplicative gate) failing to clear this bar under supervised
training — treat "architecture-side PAC layer, supervised loss" as a closed
question, not worth another redesign attempt. Reinforces §13.2's framing: the
prior only pays off if the *objective* forces it.

**(b) MAE pretrain sweep — the §13.7 keystone objective, first real run.**
`models/pretrain.py` (`MAEPretrain`): frontend + tri-axial encoder
(`freq_mixer="attention"`, no coupling matrix given to the encoder — see
below) pretrained via masked reconstruction of per-(electrode, band,
patch) log amplitude, then linear-probed. Two `mask_mode`s (`_mask`,
`models/pretrain.py:51`):
- `random` — standard MAE, independent Bernoulli per token, `mask_ratio=0.5`.
  Safety net / proven-paradigm control.
- `crossfreq` (**OURS**) — deterministically hide the entire upper half of
  bands (`m[:, :, nb//2:, :] = True`) for every electrode/patch, leaving only
  low bands visible. The only signal that can reconstruct hidden high-band
  amplitude is low-phase→high-amplitude coupling, so the objective forces the
  mechanism directly — this is §13.7's keystone, and the direct answer to
  §9.17 Finding 1 ("supervised training has no incentive to learn cross-
  frequency mechanism"). The true coupling matrix is zeroed before the
  encoder in this mode (`cpl = torch.zeros_like(coupling)`) since it's
  computed from the very bands being hidden and would leak the target.

Jobs 14214002-14214011 (`configs/pretrain_{ds}_{crossfreq,random}.yaml`,
`pretrain.slurm`; 30 pretrain epochs + 20 probe epochs each, checkpoint saved
before probing so a probe timeout doesn't lose the encoder):

| dataset | crossfreq (bacc/auroc-or-f1/pr_auc-or-kappa) | random | verdict |
|---|---|---|---|
| TUAB | 0.779 / 0.857 / 0.862 | 0.743 / 0.809 / 0.811 | crossfreq wins ✓ |
| CHB-MIT | 0.535 / 0.878 / **0.393** | 0.500 / 0.743 / 0.136 | crossfreq wins big (pr_auc ~3x) ✓ |
| TUSZ | 0.669 / 0.836 / 0.551 | 0.558 / 0.809 / 0.453 | crossfreq wins ✓ |
| TUEV | 0.447 / f1=0.563 / κ=0.271 | 0.472 / f1=0.644 / κ=0.356 | random wins ✗ |
| Sleep-EDF | 0.557 / f1=0.649 / κ=0.451 | 0.586 / f1=0.657 / κ=0.483 | random wins ✗ |

**Reading.** crossfreq beats random 3/5, and by a wide margin on the two
extreme-imbalance binary seizure tasks (CHB-MIT, TUSZ) plus TUAB; random wins
on the two multi-class tasks (TUEV 6-way, Sleep-EDF 5-way). This lines up with
a **binary-vs-multiclass split, not yet confirmed as causal** — one plausible
account is that hiding an entire frequency half teaches a coarse
"anomalous/coupled or not" signal that transfers to binary detection tasks but
under-serves fine-grained multi-way discrimination; not tested against
alternative explanations (e.g. dataset size, channel count) yet.

**Status: this is the first positive evidence for §13.2 layer 3 (the SSL
keystone) actually paying off** — but it is partial (3/5, single seed) and
does not close the question raised in §13.2 ("if the SSL objective does not
deliver, novelty rests on a prior our own ablations call weak"). Next steps,
not yet done: (1) explain the TUEV/Sleep-EDF losses before claiming the
keystone generally works; (2) multi-seed on at least the 3 winning datasets
before treating this as settled (per seed-workflow convention, dev used seed
0 only so far); (3) decide whether (a) is worth another architecture attempt
given (b) — current read is **no**, resource should shift to the pretrain
objective, not the mixer.

**Critical gap the §13.10(b) runs do NOT close: all 10 used
`freq_mixer="attention"`.** The coupling matrix was zeroed, so the directional
coupling OPERATOR (`FreqCoupling`) was never actually exercised under the
crossfreq objective. The flagship claim we'd publish — "the directional
coupling operator beats attention *when the objective forces it*" — lives in
the untested interaction cell of the 2×2 {freq_mixer: attention/coupling} ×
{objective: random/crossfreq}. Identified while reading the 2026-07-19
landscape/positioning doc ("Designing a Novel, Publishable EEG Foundation
Model…", repo root), which makes running this 2×2 its #1 recommendation.

### 13.11 The operator×objective 2×2 — leakage control + jobs submitted (2026-07-19)

The two missing cells (coupling+random, coupling+crossfreq) require feeding a
real coupling matrix to `FreqCoupling` under masking, which reintroduces the
target-leakage problem the §13.10(b) runs dodged by zeroing it.
`coupling[.., i, j] = mean_t(phase_i · amp_j)` within a patch, so any entry
whose driving band i or driven band j is a hidden token carries that band's
own amplitude/phase — i.e. the reconstruction target. Feeding the full matrix
trivializes the task.

**Leakage control (implemented, `models/pretrain.py` forward):** keep coupling
ONLY between band-tokens that are BOTH visible at each (channel, patch); zero
every entry touching a masked band. `vis=(~mask).permute(0,1,3,2)` →
`keep = vis.unsqueeze(-1) & vis.unsqueeze(-2)` → `cpl = coupling*keep`. For
crossfreq this leaves the **low→low block only** (the operator must still LEARN
low→high routing through its Q/K/V + pac_scale; the coupling prior cannot hand
it the answer); for random it leaves the visible-visible pairs. Applied
uniformly to both objective columns so the 2×2 does not confound objective with
leakage policy. Replaces the old crude `zeros_like if crossfreq else coupling`
(which was moot anyway — attention ignores coupling). CPU smoke test
(`scratchpad/smoke_coupling.py`): both modes give finite loss + finite grads,
`pac_scale` receives gradient, and the leak-check confirms **zero coupling mass
on hidden high bands** while the low→low prior (mass ~2.46) is preserved.

**Jobs (seed 0, `configs/pretrain_{ds}_{crossfreq,random}_coupling.yaml`,
freq_mixer=coupling):** 14246031-41 — submitted priority order: coupling+
crossfreq on tusz/tuab/chbmit (the interaction cell on the 3 datasets crossfreq
already won) first, then their coupling+random controls, then tuev/sleep for
completeness. Reading, once done: compare each dataset's 4 cells
(attn+random §13.10, attn+crossfreq §13.10, coup+random, coup+crossfreq). The
flagship result is coup+crossfreq **> all three others** — that's "operator ×
objective interaction". If coupling only matches attention but crossfreq still
wins, the paper falls back to an objective-only claim (doc Recommendation 5b).
Still single-seed dev; multi-seed the winners before any writeup.

**Outcome (2026-07-19) — the flagship interaction cell NEVER RAN.** Jobs
14246031-36 (the coupling variants on the 3 binary datasets tusz/tuab/chbmit,
BOTH crossfreq and random) were all **CANCELLED** before completion; only the
two multi-class datasets finished (14246038-41). So the 2×2 exists ONLY for
TUEV and Sleep-EDF — the two datasets crossfreq already LOST on in §13.10:

| dataset (kappa) | attn+random | attn+crossfreq | coup+random | coup+crossfreq |
|---|---|---|---|---|
| TUEV | 0.356 | 0.271 | 0.271 | 0.286 |
| Sleep-EDF | 0.483 | 0.451 | 0.468 | **0.493** |

On Sleep-EDF, coup+crossfreq (0.493) is the best of the four cells — a *hint*
the operator×objective interaction may be real — but it is on a dataset in the
"loser" column and single-seed, so not load-bearing. **The decisive test —
coupling+crossfreq on TUAB/CHB-MIT/TUSZ, the 3 datasets crossfreq won — is
still UNRUN and must be resubmitted.** This is the single most important open
cell in the whole project.

### 13.12 Phase-steered mixer + phase-alignment objective (2026-07-19)

Motivated by the same landscape/positioning doc, a **stronger, parameter-free
realization of the coupling prior** than any mixer tried before. This is now the
frontier of the project and the strongest novelty candidate.

**(a) `FreqPhaseSteered` mixer (`freq_mixer="phase"`, `models/triaxial.py:206`).**
Parameter-free directional cross-band communication through the *complex* PAC
vector. The frontend now returns `pac_vector[i,j] = mean_t A_j(t)·exp(i·φ_i(t))`
— coupling magnitude AND preferred physical phase (`return_pac_vector=True` when
freq_mixer=phase, `models/build.py:65`). For every target band j, messages arrive
only from slower bands i<j (strict lower-triangular mask); each source token is
rotated in adjacent 2-D feature planes by `angle(pac_vector[i,j])`, then
magnitude-normalised aggregated (complex batched matmul, avoids the
(M,target,source,D/2) tensor that dominated cost on 16-electrode TUSZ/CHB-MIT).
**No QK, no learned pac_scale, no gate, no value/output projection** — the ONLY
pathway across the frequency axis is the measured phase-amplitude geometry
itself. This is the sharpest possible form of §13.2's "the prior IS the
mechanism": unlike `coupling`/`coherence` (which add a *learnable* cross-band
path that supervised training then zeroes, §9.17/§13.10a), phase-steered
*cannot* route cross-band except through real PAC.

Built-in mechanism ablation (`train.py:127`): at test, re-run with
`phase_mode ∈ {magnitude` (zero the preferred phase, keep |PAC|), `scramble`
(randomise per-edge phase, keep |PAC|)`}` — the decisive test of whether the
measured phase actually carries the signal.

Supervised results, jobs 14253205-209 (first batch 14253034-38 cancelled),
seed 0:

| dataset (key metric) | normal | magnitude | scramble | vs best OTHER mixer |
|---|---|---|---|---|
| TUAB (auroc) | **0.873** | 0.824 | 0.873 | best (coupling 0.872) |
| TUEV (kappa) | **0.413** | 0.178 | 0.332 | best (attention 0.376) |
| Sleep-EDF (kappa) | 0.516 | 0.040 | 0.507 | mid (cotar 0.572) |
| TUSZ (auroc / bacc) | 0.835 / 0.594 | 0.770 / 0.633 | 0.803 / 0.582 | auroc ties coupling; bacc worse than attn 0.697 |
| CHB-MIT | **TIMEOUT** (14253209, hit 12h wall — NO RESULT) | — | — | — |

Reading: phase is the **best supervised mixer on TUAB and TUEV**, mid on Sleep,
mixed on TUSZ (auroc competitive, balanced-acc worse than attention). This is
the first mixer to actually clear the other mixers on any dataset under plain
supervised training (§13.10a said the class was closed — phase is a partial
counterexample, on 2/4 completed datasets). **Mechanism ablation is a SPLIT
verdict, and must be reported honestly:** zeroing the preferred phase entirely
(`magnitude`) is consistently and sometimes catastrophically damaging (Sleep
kappa 0.516→0.040; TUEV 0.413→0.178) → the phase geometry is genuinely
load-bearing, not decorative. BUT `scramble` (randomise per-edge phase) barely
hurts on TUAB/Sleep (and only moderately on TUEV) — if the *specific measured*
phase mattered, scramble should hurt as much as magnitude does. Current read:
the model relies on a phase rotation being *present/structured* more than on its
exact measured value — a real caveat to any "learns the true PAC phase" claim,
and the #1 thing to understand before building the paper on this mixer.
CHB-MIT is untested (timed out — needs >12h wall or fewer epochs).

**(b) `phase_align` objective (`models/pretrain.py:119`, `pretrain_task="phase_align"`).**
Contrastive BCE discriminating real PAC geometry (positive) from a
magnitude-matched phase-scramble (negative: permute the complex *unit* phase
across (electrode,patch) within each sample, keep every |Z| and all tokens).
Trains the phase mechanism directly, where amplitude-MAE only encouraged it
indirectly. Requires freq_mixer=phase; pooled pos/neg encodings → `align_head`.

Pretrain results (only tusz/sleep ran; the phase_align attempts 14253097/099/225
were cancelled, completed ones are 14255746 tusz / 14253226 sleep):

| dataset (key) | phase_align | phase_random (recon-MAE, phase mixer) | crossfreq-MAE §13.10 (attn) |
|---|---|---|---|
| TUSZ (auroc) | 0.773 | 0.814 | **0.836** |
| Sleep-EDF (kappa) | 0.343 | 0.436 | **0.451** |

**phase_align LOSES to both its own phase_random control and to the §13.10
crossfreq amplitude-MAE.** Diagnosed cause (from the loss curves):
`align_loss` collapses to ~0.0008 within 4 epochs — the contrastive task is
**trivially separable** (the phase-scrambled negative is too easy to tell
apart), so the encoder learns a shortcut discriminator that does not transfer,
while phase_random's recon_loss converges normally (→0.064 over 15 epochs) and
transfers better. **The phase_align objective is currently broken (too-easy
negatives); it needs harder negatives (e.g. small phase perturbations, or
mixing real geometries across samples) or a non-contrastive formulation before
it can fairly test the phase-steering hypothesis under SSL.**

**Open gaps in the phase line (none of these done):** (1) CHB-MIT phase
supervised timed out — rerun with a longer wall or fewer epochs; (2) phase_align
on TUAB/CHB-MIT/TUEV never submitted; (3) fix the too-easy-negative problem in
phase_align; (4) understand the scramble-vs-magnitude asymmetry in the mechanism
ablation; (5) the entire phase line is single-seed. Combined with §13.11's unrun
coupling+crossfreq flagship cell, the two highest-value TODOs are:
**resubmit coupling+crossfreq on TUAB/CHB-MIT/TUSZ, and fix+rerun phase_align.**

### 13.13 Full job ledger for the 2026-07-18/19 push (so nothing is lost again)

Every job from this multi-session push, so progress is never under-counted (a
prior status write missed the phase + coupling families entirely — they were
submitted by a parallel session/user). Verify with
`sacct -u zz5070 --starttime 2026-07-18T00:00 -X`.

- **14213997-14214001** — supervised `coherence` (5 ds). COMPLETED. §13.10a.
- **14214002-14214011** — MAE `{crossfreq,random}` × 5 ds, freq_mixer=attention. COMPLETED. §13.10b.
- **14246031-14246036** — MAE `{crossfreq,random}_coupling` on tusz/tuab/chbmit. **ALL CANCELLED** (flagship cell lost). §13.11.
- **14246038-14246041** — MAE `{crossfreq,random}_coupling` on tuev/sleep. COMPLETED. §13.11 table.
- **14253034-14253038** — supervised `phase` (5 ds), first attempt. **ALL CANCELLED** (superseded).
- **14253205-14253208** — supervised `phase` tusz/sleep/tuab/tuev. COMPLETED. §13.12a.
- **14253209** — supervised `phase` chbmit. **TIMEOUT** (no result). §13.12a.
- **14253097/099, 14253225** — `phase_align` tusz/sleep early attempts. CANCELLED.
- **14253098, 14253100** — `phase_random` tusz/sleep. COMPLETED. §13.12b.
- **14253226 (sleep), 14255746 (tusz)** — `phase_align`. COMPLETED. §13.12b.

Checkpoints for all completed pretrain runs: `checkpoints/<wandb_run_name>.pt`.
New source since §13.10: `FreqPhaseSteered` + `"phase"` in `FREQ_MIXERS`
(`models/triaxial.py`), `return_pac_vector` (`models/frontend/triaxial.py`),
`phase_mode` ablation (`models/build.py`, `train.py`), `_phase_alignment_loss`
+ visible-visible coupling leakage control (`models/pretrain.py`), configs
`configs/{ds}_v2_phase.yaml`, `configs/pretrain_{ds}_{phase_align,phase_random,*_coupling}.yaml`,
`scripts/test_phase_steered.py`, and the landscape doc at repo root.
