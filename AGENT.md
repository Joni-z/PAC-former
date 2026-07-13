# PAC-Former — Status & Build Guide (for coding agent)

## 0. What this project is

**PAC-Former**: a differentiable phase-amplitude coupling (PAC) operator that
replaces self-attention in a frequency-domain EEG encoder. Frequency bands are
tokens. The mixer that moves information between tokens is a learnable
Modulation Index (MI) operator instead of QKᵀ softmax attention.

The codebase's entire reason for existing is to support one controlled
ablation: **same backbone, same training, swap only the token mixer** between
(a) vanilla self-attention, (b) CoTAR, (c) our MI operator. Every design
decision exists to keep that swap clean. If a change makes the mixer swap
messier, it's the wrong change.

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

`coupling` is the Canolty (2006) MVL score with Ozkurt normalization and
van-Driel-style mean-centered amplitude debiasing (removes the spurious
low-frequency term from amplitude being strictly positive). `pac_scale` is a
learned scalar (init 1.0). This strictly generalizes pure attention
(`pac_scale -> 0`) and pure PAC-weighted aggregation (frozen Q/K, large
`pac_scale`), so the model can learn, per layer, how much to lean on the
physiological prior vs. a data-driven cross-band relationship. Redistribute
step is concat + MLP (matches CoTAR).

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

---

## 10. Things to flag back to the user rather than deciding alone

- Any case where TUAB/TUEV preprocessing details aren't fully specified by
  the BIOT repo and require a judgment call.
- Any change to the MI operator's redistribute step (currently concat+MLP,
  matching CoTAR) — gating and matrix-mixing are open ablation choices, not
  closed decisions.
- Any numerical instability that survives the unit-complex-vector trick in
  Section 4 — don't patch it ad hoc, surface it.
