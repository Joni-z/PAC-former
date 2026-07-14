# CFC-in-the-Frequency-Domain: EEG Novelty Notes

## Novelty / literature review: cross-frequency coupling as a differentiable structural prior for a frequency-domain EEG encoder

> **Positioning note (updated 2026-07-12):** this document was originally
> written as a standalone novelty argument for "PAC-Former vs. CoTAR." The
> project's actual goal is an **EEG foundation model**; CoTAR/attention are
> not opponents to beat head-to-head but a source of time-series
> architecture ideas being evaluated for inclusion in the eventual
> foundation-model backbone (see `AGENT.md` §0/§11). The literature survey
> below is still the correct grounding for **why cross-frequency coupling
> (CFC) is a validated, genuinely novel architectural idea worth folding
> into that backbone** — read every "vs. CoTAR" framing in this doc through
> that lens: CoTAR is the aggregate-redistribute template we borrow and
> extend into the frequency domain, and the ablation against
> attention/CoTAR/MI (`AGENT.md` §9) is how we validate the idea before
> spending pretraining compute on it, not the paper's headline claim.

### One-paragraph summary

* The concrete idea — designing a differentiable, end-to-end module that
  welds the directional cross-frequency coupling (CFC) hierarchy ("low-frequency
  phase modulates high-frequency amplitude") directly into the attention /
  token-interaction mechanism itself, as a frequency-domain analogue of CoTAR
  — appears genuinely novel; no existing paper combines all three defining
  elements at once. The closest prior art is ACCNet (*Neural Networks*,
  2025), which does use CFC architecturally, but as a learnable edge in a
  graph-attention network, not as a directional attention replacement inside
  a frequency-domain transformer.
* The surrounding space is crowded, but the crowding lands just to the side
  of this idea. Frequency-domain EEG encoders, spectral transformers,
  Fourier-as-attention replacements (FNet / GFNet / AFNO), and frequency-domain
  forecasting transformers (FEDformer / FreTS / Fredformer / FreEformer) all
  exist — but they treat frequency as a symmetric/global object; none encode
  a directional, hierarchical CFC prior. In deep learning, PAC/CFC is
  overwhelmingly used as a precomputed *feature*, not as an architectural
  inductive bias.
* The neuroscience grounding is solid and clinically motivated: phase-amplitude
  coupling is a well-established, directional ("low modulates high")
  phenomenon that is reliably altered in Alzheimer's disease and epilepsy —
  so this inductive bias earns its place. The cleanest contribution framing:
  a differentiable PAC / modulation-index operator that directionally
  organizes attention, benchmarked against ACCNet and standard self-attention.

### Key findings

1. **No fully overlapping prior art.** An extensive search across arXiv,
   OpenReview, Semantic Scholar, PubMed/PMC, and relevant venues found no
   paper simultaneously satisfying: (a) a differentiable PAC / modulation-index
   operator implemented inside attention computation; (b) an asymmetric,
   directional "low-frequency-phase → high-frequency-amplitude" hierarchy;
   (c) used to replace fully-connected self-attention in a frequency-domain
   EEG encoder.
2. **ACCNet is the biggest threat and must be addressed head-on.** It uses
   CFC architecturally (learning graph edges between adaptive frequency-band
   nodes, explicitly emphasizing low-high interactions), but lives in a GNN /
   graph-attention framework, is undirected, and has no differentiable
   modulation-index operator.
3. **"CFC as a feature" is the mainstream paradigm** (complex-valued CNNs on
   PAC comodulograms, autoencoders on CFC matrices). These don't threaten
   novelty, but confirm the field already recognizes CFC as discriminative.
4. **Frequency-domain EEG transformers and foundation models are numerous**,
   but their use of frequency concentrates on tokenization / reconstruction
   objectives (LaBraM, NeuroRVQ, TFM-Tokenizer) or symmetric cross-band
   attention — none of it directional CFC.
5. **Fourier-as-attention architectures** (FNet, GFNet, AFNO, SpectFormer)
   replace attention with global/symmetric frequency-domain filtering; none
   encode cross-frequency directionality.
6. **General time-series frequency-domain transformers** (FEDformer, FreTS,
   Fredformer, FITS, FourierGNN, FreEformer) operate in the frequency domain
   but treat frequency as a flat set; Dualformer (2026) introduces a
   depth-wise frequency hierarchy, but it is general-purpose and non-PAC.
7. **No differentiable FOOOF / specparam encoder exists** — periodic /
   aperiodic (1/f) decomposition is used offline; making it end-to-end is a
   related, similarly open opportunity.
8. **"Why has nobody just dropped classical MI (MVL) into attention" deserves
   a direct answer, not a dodge.** The search found this isn't simply "nobody
   thought of it" — there are three concrete technical barriers plus a
   motivation gap, which together form the "the door wasn't guarded, nobody
   tried to open it" part of the novelty argument, and should be addressed
   proactively in the paper rather than left for a reviewer to raise. See
   "Why this combination hasn't been done before" below.

---

## Detailed content

### Baseline paper background (TeCh / CoTAR)

This work builds on "Decentralized Attention Fails Centralized Signals:
Rethinking Transformers for Medical Time Series" (Guoqi Yu, Juncheng Wang,
Chen Yang, Jing Qin, Angelica I. Aviles-Rivero, Shujun Wang), ICLR 2026
(Oral), arXiv:2602.18473, OpenReview oZJFY2BQt2; code at
github.com/Levi-Ackman/TeCh (built on Medformer). Core argument: MedTS
signals are "centralized" while self-attention is "decentralized"
(fully-connected) — a structural mismatch. CoTAR (Core Token
Aggregation-Redistribution) replaces attention with a global core token:
aggregate first, then redistribute information back to each channel,
yielding linear complexity. Our analogous argument — self-attention treats
frequency as a fully-connected graph, mismatched with the directional CFC
hierarchy — is a clean structural parallel and a well-positioned frame.

**Positioning clarification: TeCh plays two distinct roles in this project,
which should be kept separate to avoid the misreading "we're just modifying
TeCh":**

1. **Argumentative ancestor** — the structural argument that "decentralized
   attention doesn't fit centralized signals," and the aggregate→redistribute
   paradigm, are the intellectual resources we inherit and transplant into
   the frequency domain. This part should be cited and compared against.
2. **Architectural template** — TeCh/CoTAR itself is a **time-domain** model
   (a Medformer-based patching backbone); its specific architectural details
   should not be copied wholesale into our frequency-domain encoder. Our
   backbone is derived from the frequency-domain CFC structure itself
   (learnable bandpass → analytic signal → band-as-token → MI operator).

This distinction sharpens the pitch: "TeCh showed decentralized attention
doesn't fit centralized **time-domain** medical signals; we carry that
insight into the **frequency domain** — where 'centralized' means the
cross-frequency-coupling hierarchy (low frequency governs high frequency)."
TeCh thus goes from "template being copied" to "springboard being extended,"
and the novelty boundary becomes clearer: TeCh should be one of the methods
compared against in experiments, not the parent of our method.

---

### Area 1 — CFC/PAC as a structural/architectural prior (most important)

* **ACCNet** (Dongyuan Tian, Yucheng Wang, Peiliang Gong, Zhewen Xu, Zhenghua
  Chen, Xiaohui Wei, Min Wu), *Neural Networks* vol. 191, 107853, 2025; DOI
  10.1016/j.neunet.2025.107853. Two modules: Adaptive Bands Decomposition
  (subject-specific frequency-band nodes) and a Cross-Frequency Coupling
  mechanism that learns personalized frequency relationships from a
  node-edge perspective, "specifically emphasizing interactions between
  low- and high-frequency components." This is an architectural-level CFC
  mechanism (not feature fusion) — but it does graph attention over
  frequency-band nodes with undirected edge learning, and is not a
  differentiable PAC operator replacing transformer self-attention. This is
  the closest prior art; distinguish clearly on three axes: (GNN vs.
  transformer), (undirected edges vs. directional phase→amplitude
  modulation), (no differentiable modulation-index operator).
* CFC-as-feature precedents (not a threat, but cited as the mainstream
  paradigm): complex-valued CNNs on SEEG epilepsy PAC comodulograms
  ("Classifying epileptic phase-amplitude coupling in SEEG using
  complex-valued convolutional neural network," *Frontiers in Physiology*,
  2022, DOI 10.3389/fphys.2022.1085530); autoencoders on CFC matrices for
  absence seizures (*Frontiers in Neuroinformatics*, 2025); DB-GNN
  (arXiv:2504.20744, 2025) for identifying CFC brain networks.
* Convergent work welding "physics" into attention, non-EEG / non-directional:
  Holographic Transformer (arXiv:2509.19331, 2025) integrates phase
  interference into self-attention (within-frequency, not cross-frequency).
  TransformEEG (*Applied Sciences* 15(24):13275, 2025) implicitly learns PAC
  via phase-swap data augmentation — no PAC-structured operator.

**Additional search finding worth calling out on its own:** Holographic
Transformer is currently the only work that genuinely puts "phase" explicitly
inside self-attention computation, and deserves treatment as a nearest
neighbor rather than being lumped into "convergent work." It implements
phase interference as a discrete interference operator inside attention, and
specifically designs a dual-head decoder to prevent "phase collapse" (phase
information getting squeezed out when the loss favors amplitude
optimization). It differs from our approach on three axes: (a)
within-frequency phase interference, not cross-frequency phase→amplitude
modulation; (b) symmetric, no directionality; (c) non-EEG, non-PAC. Its
"phase collapse" problem also flags an engineering risk worth watching in
our own operator design (see the normalization discussion in the MI-operator
section below).

---

### Area 1.5 — Modulation index design space: why the MVL form, and why this combination hasn't been done before

This section answers two questions that sit outside the literature survey
proper but are equally part of the novelty argument: (1) classical MI isn't
one formula but a family — which one do we pick, and why; (2) if the MVL
form is naturally attention-shaped, why has nobody done this in twenty
years?

#### (a) The MI design space

The modulation index for PAC has several historically competing definitions,
each essentially choosing a different "distance + aggregation" scheme:

* **MVL (Mean Vector Length, Canolty 2006)** — Z = (1/T)Σ A_high(t)·e^{iφ_low(t)},
  taking |Z| as coupling strength. This is the form we use, because it is
  **naturally differentiable and structurally isomorphic to attention**:
  weighting by a complex exponential and summing amplitude is the same
  "weight by phase/similarity, then aggregate amplitude/value" paradigm as
  softmax(QKᵀ)V. Its drawback is dependence on the absolute magnitude of
  high-frequency oscillation amplitude, requiring Ozkurt normalization
  (dividing by Σ|A|) or LayerNorm to prevent the model from learning "which
  band has more power" instead of "which band pair is coupled."
* **Ozkurt-normalized MVL** — adds a normalization term to MVL, equally
  differentiable, recommended to integrate directly into the operator design.
* **Tort's KL-MI (2010)** — bins phase, computes mean amplitude per bin,
  normalizes into a distribution, and computes KL divergence from uniform.
  This is the field's most commonly used and most robust version, with a
  clean information-theoretic interpretation, but **hard binning is itself
  non-differentiable** — keeping that interpretation while making it
  differentiable requires soft-binning (softmax-weighted in place of hard
  bins), a possibly worthwhile but non-essential extension.
* **Other corrected variants** (dPAC, dMI, eMI, wMI, GLM-CFC) — proposed to
  address boundedness, noise robustness, harmonic robustness, and short-data
  issues; mostly too computationally heavy for an attention inner loop, but
  usable as comparison points for "why MVL and not something else."

This design space itself doesn't threaten novelty (all MI variants are
offline analysis methods, none designed for a differentiable/end-to-end
setting), but proactively explaining "why MVL, not Tort-KL" in the methods
section signals a deliberate, weighed choice rather than grabbing whatever
formula was handy.

#### (b) Why this combination hasn't been done before — four technical reasons

This is worth discussing proactively in the paper because MVL's
"differentiability" and "structural similarity to attention" don't look like
new discoveries — if it's this obvious, why hasn't anyone welded it into
attention in twenty years? The search surfaced a stack of concrete technical
barriers, not a dead end:

1. **Complex-valued softmax fundamentally doesn't hold.** Standard attention
   relies on softmax(QKᵀ)V, and softmax requires real-valued inputs; porting
   attention to the complex domain, if similarity is computed via softmax,
   either degenerates to a constant function or becomes non-analytic
   (non-differentiable) in the complex domain. This is a recurring hard
   obstacle in the complex-domain attention literature. **Our workaround:**
   MVL doesn't need softmax at all — it is inherently a "complex-exponential-weighted
   sum, then take the modulus," so |Z| can be used directly as coupling
   strength without forcing the softmax paradigm — but this means the paper
   needs to explicitly state "this is not a complex-valued version of
   softmax attention, but a different aggregation mechanism," or risk being
   misread as dodging the complex-softmax problem rather than sidestepping it.
2. **MVL flattens the time axis — it's fundamentally a *readout*, not a
   *mixer*, and this is the deepest objection, requiring a direct response.**
   Standard attention outputs a "reweighted token sequence" (token×token
   mixing), while MVL's Z = (1/T)Σ A(t)·e^{iφ(t)} integrates an entire time
   segment down to a single (complex) scalar. The N×N comodulogram matrix
   does resemble an attention matrix in shape, but it lives in
   "frequency-pair space," not token space, and has no natural "value" to
   propagate downstream. This means "MVL replacing attention" isn't a
   literal drop-in — "what gets aggregated, how it gets redistributed" must
   be redefined. **This is exactly the bridge CoTAR provides**: CoTAR's
   paradigm is precisely "aggregate into a core token, then redistribute,"
   and MVL's time-integration step maps cleanly onto CoTAR's "aggregate"
   step, rather than being treated as a suspicious pooling operation. See
   the exact mapping in "MI operator's isomorphism with CoTAR" below — this
   is the central argument answering the reviewer objection "this is just
   pooling, why call it an attention replacement," and must be spelled out
   in the methods section.
3. **Gradients through phase extraction are unstable.** Instantaneous phase
   is the argument of the analytic signal, i.e. arctan2(imaginary, real);
   its gradient diverges as amplitude approaches 0 (denominator |z|²), and
   phase itself has a wrap discontinuity at 2π. **Workaround:** stay in
   complex representation throughout, never explicitly call arg() / atan2();
   use z_low/|z_low| as a unit phase vector directly in computation, only
   clamping where |z_low| approaches 0. This is the same class of problem as
   the SincNet t=0 NaN bug encountered earlier in the project, and the fix
   generalizes.
4. **Lack of motivation, not lack of capability.** Before CoTAR's argument
   that "self-attention is structurally mismatched with centralized
   signals" existed, the mainstream deep-learning + PAC approach (compute
   comodulogram offline → feed to CNN/autoencoder) was good enough; nobody
   had pressure to weld it end-to-end into attention. This is a "nobody
   bothered" gap, not a "tried and failed" dead end — good news for
   novelty, but it means barriers ①②③ above are real implementation risks
   that must be addressed one by one in the methods section, not assumed
   trivial.

#### (c) MI operator's isomorphism with CoTAR — answering "isn't this just pooling?"

This section is the detailed expansion of point (b)(2) above, and should be
part of the paper's core methods argument.

CoTAR's two steps are: ① **aggregate** — pool information from all tokens
into a single global core vector; ② **redistribute** — concatenate the core
vector back onto each token and project, achieving indirect token-to-token
interaction at linear complexity.

The MI operator maps onto these two steps exactly, rather than inventing a
separate scheme:

* **① Aggregate, corresponding to the time integration in Z = Σ A·e^{iφ}.**
  The difference: CoTAR aggregates into a symmetric core vector (all channels
  treated equally), while we aggregate into a **directional N×N coupling
  matrix** (rows = low-frequency modulators, columns = high-frequency
  modulated targets). This directed matrix is exactly the structure we add
  beyond CoTAR, and is the genuinely new piece in the "frequency domain +
  CFC" vs. "time domain + centralized signals" analogy.
* **② Redistribute, corresponding to using |Z| as a weight, letting
  low-frequency phase modulate the corresponding high-frequency token.**
  CoTAR's second step concatenates the core vector back onto each token and
  passes it through an MLP; our corresponding approach treats coupling
  strength |Z| as an attention-like weight, with frequency tokens as the
  weighted values — concretely implementable as concat+MLP (closest to
  CoTAR, recommended as the first version), elementwise gating, or using the
  coupling matrix directly as a mixing matrix on tokens — the latter two are
  ablation variants.

This mapping supports the argument: we are not "using a pooling operation
and calling it attention," but "following the exact same aggregate→redistribute
paradigm as CoTAR, only swapping the symmetric core token for a directional
CFC coupling matrix." When asked "how does this count as an attention
replacement," the answer is: "it is isomorphic to CoTAR, which has already
been accepted at ICLR and recognized as a valid attention replacement."

Separately, directionality **requires no extra design** in this
construction — it falls directly out of "who supplies phase, who supplies
amplitude" (low frequency supplies phase, high frequency supplies
amplitude) — something a symmetric attention structure structurally cannot
provide, and the core distinction from ACCNet's undirected edge learning.

---

### Area 2 — Frequency-domain EEG encoders / spectral transformers (2023–2026)

Crowded: AMDET (arXiv:2212.12134) has a spectral attention block doing
symmetric cross-band attention; Spectral Transformer (PSD→transformer);
TFormer (time-frequency cross-attention); AFTA (MDPI *Brain Sciences*
15(4):382, 2025) — an Adaptive Frequency Filtering Module combined with
time-domain attention for self-supervised seizure tasks (TUSZ/TUAB/TUEV,
AUROC 0.891); learnable filter banks via SincNet → Sinc-EEGNet
(arXiv:2101.10846); FreqDGT (arXiv:2506.22807), a frequency-adaptive dynamic
graph transformer. Recent surveys (Transformer-based EEG Decoding,
arXiv:2507.02320; MDPI *Sensors* 25(5):1293, 2025) note most EEG
transformers convert to time-frequency or fuse spectral features rather than
going deep on directional frequency structure — leaving room for the
"directional CFC" angle.

### Area 3 — EEG foundation models and frequency

LaBraM (ICLR 2024 spotlight) uses vector-quantized neural spectrum
prediction — its tokenizer reconstructs Fourier amplitude and phase. BIOT
(NeurIPS 2023) tokenizes biosignals into a sentence-style format. NeuroRVQ
(OpenReview m38Hle9Utx) explicitly reconstructs Fourier spectral amplitude A
and phase φ (via sin/cos). TFM-Tokenizer (arXiv:2502.16060) uses a
frequency-then-time paradigm with a Localized Spectral Window Encoder that
slices windows into frequency patches to model "cross-frequency
dependencies." CBraMod uses criss-cross (spatial/temporal) attention;
CodeBrain uses dual-domain tokenization; EEGPT (OpenReview lvS2b8CjG5) uses
spatiotemporal masked SSL; REVE uses 4D Fourier positional encoding.
Confirmed: their tokenizers heavily depend on frequency-domain
reconstruction/PSD, but none impose a directional CFC interaction prior —
TFM-Tokenizer's "cross-frequency dependency" is symmetric patch interaction,
not directional phase→amplitude modulation.

### Area 4 — Fourier/spectral as an attention replacement

FNet (fixed DFT for token mixing), GFNet (NeurIPS 2021 / T-PAMI; learns
global filters in the frequency domain, O(L log L), replacing
self-attention), AFNO (block-wise channel mixing + soft thresholding),
SpectFormer. All relax the "all tokens fully connected" assumption via
global, symmetric frequency-domain operations at log-linear complexity —
conceptually aligned with our complexity/efficiency motivation — but none
encode cross-frequency directionality or a "low modulates high" hierarchical
bias. Rarely applied to EEG; this itself is a gap.

### Area 5 — Periodic/aperiodic (1/f) decomposition in neural networks

FOOOF/specparam (Donoghue et al. 2020) parameterizes spectra as aperiodic
(offset, knee, exponent χ) + periodic (Gaussian peaks). Overwhelmingly used
as an offline analysis step. Searches for "differentiable FOOOF / aperiodic
exponent neural network / specparam end-to-end" turn up only offline
analysis papers, plus one preprint using FOOOF's stochastic fluctuations in
preprocessing (arXiv:2505.19009) — but no fully differentiable end-to-end
FOOOF encoder exists. This is a related, similarly open contribution that
could be combined with the CFC prior.

### Area 6 — General time-series frequency-domain transformers

FEDformer (frequency-enhanced decomposition, random Fourier mode selection);
FreTS (NeurIPS 2023, frequency-domain MLP, "global view" + "energy
compaction"); Fredformer (KDD 2024, debiasing frequency bands so the model
doesn't over-attend to high-energy/low-frequency components); FITS
(low-pass + complex linear); FourierGNN (Fourier Graph Operator); FreEformer
(arXiv:2501.13989, adds a learnable matrix on top of attention to fix a
low-rank issue); JTFT (joint time-frequency). None introduce cross-frequency
coupling or a directional frequency hierarchy; Fredformer's "frequency bias"
is the closest conceptual relative (concerned with how attention weights
frequencies), but it's about equalizing, not imposing a directional
hierarchy. Dualformer (arXiv:2601.15669, 2026) assigns high-frequency
components to shallow layers and low-frequency to deep layers — a frequency-hierarchy
architectural prior, but general-purpose, non-EEG, non-PAC.

### Area 7 — Neuroscience grounding (brief)

* **Real, directional, hierarchical:** PAC — high-frequency amplitude
  modulated by low-frequency phase — is classical and has direct evidence in
  humans. Canolty et al. (*Science* 2006, DOI 10.1126/science.1128115) used
  human ECoG recordings showing low-frequency theta (4–8 Hz) phase modulates
  high-gamma (80–150 Hz) band power, with stronger modulation at higher
  theta amplitude (peak coupling at ~146.2 Hz amplitude, ~5.6 Hz phase).
  Hippocampal theta-phase→gamma-amplitude coupling (Tort, Buzsáki) is a
  textbook example. Directionality/hierarchy is supported by Voytek et al.
  (*Frontiers in Human Neuroscience* 2010, PMC2972699): low-frequency
  oscillations may coordinate long-range communication across brain regions
  while high-gamma activity is more spatially localized — i.e., slow rhythms
  set a temporal frame within which fast, local activity is nested.
  Computational models (neural mass / cortical column, *PLoS Comput Biol*
  2016; oscillator network models) confirm directional generative
  mechanisms.
* **Altered in disease (clinical motivation):** In Alzheimer's disease,
  Prabhu et al. (*Brain Communications* 2024; 6(2):fcae121) found AD
  patients (n=50; age 60±8) vs. cognitively normal controls (n=35; age
  63±5.8) "show reduced theta-gamma PAC, with weakened coupling of gamma
  amplitude within the 6–8 Hz oscillation range," localized to left
  parahippocampal cortex (gamma amplitude 30–40 Hz coupled to theta 4–8 Hz /
  alpha 8–12 Hz phase, MEG). In epilepsy, PAC (low-frequency phase to HFO
  amplitude) is a recognized seizure-onset-zone (SOZ) biomarker: Cui et al.
  (*Cognitive Neurodynamics* 2023, DOI 10.1007/s11571-022-09915-x) used a
  mean-vector-length modulation index between low-frequency rhythms
  (0.5–24 Hz) and HFOs (80–560 Hz) on 20-second interictal ECoG for SOZ
  localization; Motoi et al. (medRxiv 2020.11.07.20226258) reported
  infraslow-HFA PAC distinguishing SOZ with "AUC of 0.926," rising starting
  ~87 seconds before seizure onset. This confirms the bias we want to
  encode does track the clinical states we're trying to classify.

---

## Overall novelty assessment

The proposed contribution — a differentiable PAC / modulation-index operator
that directionally organizes (or replaces) self-attention in a
frequency-domain EEG encoder, encoding the asymmetric "low-frequency-phase
modulates high-frequency-amplitude" hierarchy — is, per an extensive search,
novel. Its three defining elements (differentiable PAC operator; directional/asymmetric
hierarchy; attention replacement in a transformer) each appear individually,
or in adjacent form, but never combined:

* Differentiable PAC operator inside attention: not found.
* "Low→high" frequency hierarchy as an architectural bias: only found
  general-purpose and non-PAC (Dualformer, depth-wise), or undirected
  (ACCNet's graph edges, AMDET's cross-band attention).
* Attention replaced via spectral structure: only found symmetric/global
  (GFNet/AFNO/FNet), never with CFC directionality.

The biggest threat is ACCNet; secondary convergent ideas are Holographic
Transformer (phase inside attention, non-EEG, within-frequency) and
Dualformer (depth-wise frequency hierarchy, non-PAC).  None preempt the full
idea.

**Closing note:** "why has nobody done this before" has itself been directly
answered (see Area 1.5): the answer is a stack of three technical barriers —
complex-valued softmax not holding, a gap between MVL's time-integration and
mixer semantics that needs bridging, and unstable phase gradients — plus a
lack of clear motivation before CoTAR-style arguments existed, rather than
this direction having been tried and abandoned. Each of the three technical
barriers has a concrete workaround (skip softmax, bridge via CoTAR's
aggregate-redistribute paradigm, stay in complex representation throughout
without explicitly extracting phase angle). This should be written out
proactively in the methods section, turning it from "a potential reviewer
objection" into "a design tradeoff we've already thought through" — which is
itself part of the paper's argumentative strength.
