# PACLock 文献精读与项目对照

更新日期：2026-07-28

## 结论先行

1. PAC 放进深度网络并不是空白：
   Phase-Swap 已覆盖 phase/magnitude 自监督；SleepPACNet 已覆盖 Hilbert 低频复信号与
   高频分支融合；complex PAC CNN 已覆盖 coupling magnitude + preferred phase；
   PACNet 已覆盖 PAC 驱动的 subject/task-specific 频带选择。
2. 这些工作没有直接覆盖本项目的完整组合：可学习 sinc filterbank、显式
   electrode × band × time token grid、time-resolved per-channel complex coupling，以及
   PAC-aware pretraining。但论文贡献不能笼统写成“首次把 PAC 放进深度学习”。
3. 当前 `mi_product` / `mi_topk` 的 TUEV 负结果只足以否定当前测量与 mixer 组合，
   不足以否定 PAC。实现对所有 band pair 计算 MVL，MI mixer 又不限制
   slow-phase → fast-amplitude；filterbank 也没有对每个频率对保证 carrier sidebands。
4. 现有最稳定的实证仍然是 `cf_mixed` 有效，而“它通过学习 PAC routing 有效”目前
   没有证据。后续最值得做的不是继续调 mixer，而是先建立一个测量上正确、带负对照的
   PAC probe，再决定 PAC 应进入 objective、tokenizer，还是只作为分析指标。

## 一、四篇直接相近工作

### Phase-Swap（GCPR 2020）

- 方法交换两个片段的完整 Fourier phase 与 magnitude，训练二分类器判断是否交换。
  采样限定在同一 subject/session，目标是学习对跨 subject/session 稳健的表示。
- 它学到的是宽泛的 waveform–spectrum consistency，不是低频相位对高频幅值的定向
  调制，也没有 PAC comodulogram。
- 在 CHB-MIT 上原任务准确率达到 99.8%，说明任务过于容易并出现 shortcut；加入随机
  channel masking 后才重新获得下游收益。作者还发现 spatial coherence 会成为捷径。
- 对本项目的含义：若使用 phase/amplitude pretext，必须有 task difficulty 和 shortcut
  诊断。只展示 pretext loss 降低不能证明学到了 PAC。

### SleepPACNet（Scientific Reports 2026）

- 单通道 Fp1–Fp2、DOD 25 subjects、LOSO。低频 0.5–8 Hz 经 Hilbert 得到 real/imag，
  高频 8–30 Hz 保留滤波波形，二者进入 PAC CNN 分支；另有 raw-EEG 分支。
- accuracy 由普通 CNN 的 73.3% 提升到 75.7%，kappa 由 .623 到 .654；多重校正后，
  stage-wise F1 只有 REM 改善显著。手工 PAC 特征反而最差。
- 它是架构侧最接近的先例。差异空间在于：它不是显式可微 MVL，不做 band-pair
  routing，不保留 electrode × band × time 三轴，也没有预训练。
- 论文只有一个小数据集且大量结构/频带选择来自 trial-and-error。可以作为结果基线，
  不能把其 headline 当作 PAC 普适有效的证据。

### PACNet adaptive filters（Frontiers in Neuroscience 2023）

- 对 event-aligned ECoG trials 计算 ERPAC，为每个低频带选择两个高频峰（各 ±10 Hz），
  再把这些 subject/task-specific 子带送进 EEGNet 式并行分支。
- 核心贡献是 PAC-based frequency selection，不是 PAC matrix 直接参与特征混合。
- 相对固定 filterbank 的 PAC ablation 约 1.5–2 个点，强度比总表 headline 小。
- 需要注意频带选择与 CV 的边界：若在整个受试者全部 trials 上先选带再交叉验证，
  即便不使用类别标签，也引入了 test-distribution information。复现时必须明确
  selection 是 fold 内、train-only，还是 transductive。

### Complex-valued PAC CNN（Frontiers in Physiology 2023）

- 9 位 SEEG 患者、23 次 seizure、LOPO。以 Tort MI 为模、preferred phase 为角，
  构造 10×10 complex PAC image，再输入 complex CNN。
- AUC 约 .92；magnitude-only PAC + real CNN 约 .88，real/imag 普通 CNN 约 .89。
  说明 preferred phase 在该设置下有增量。
- 与本项目 `FreqPhaseSteered` / complex `pac_vector` 的概念最接近，但它是离线
  SEEG image classification，不是 end-to-end scalp-EEG transformer。

## 二、PAC 测量真正约束了什么

### 1. 高幅值频带必须包含调制侧带

若高频 carrier 为 \(f_h\)，低频调制为 \(f_l\)，幅度调制会产生
\(f_h-f_l,\ f_h,\ f_h+f_l\) 三个分量。高频滤波器只罩住 carrier 时，Hilbert envelope
可能几乎看不到调制。

- Zandvoort & Nolte 从 bispectrum 推导：窄带 PAC 中 amplitude-band 的有效带宽应与
  phase frequency 联动；1:1 配比表现最好，过窄漏侧带，过宽则混入其他能量并涂抹。
- Triplet-filter 工作主张分别罩住两侧带和 carrier，以较窄滤波器兼顾噪声抑制与
  detectability。
- State-space PAC 直接估计振荡成分与软频带，展示了普通窄带滤波删除 side lobes 后
  真实 PAC 会消失。

本项目的 sinc bank 采用 log spacing，代码注释认为高频带“自然够宽”，但宽度只由
高频 band 自身决定，并不针对每个 `(low phase, high amplitude)` pair 满足
sideband 条件。learnable cutoff 也没有 PAC-identifiability 约束。因而一个 pair 的
MVL 小，可能是没有耦合，也可能只是滤掉了其两侧带。

### 2. coupling 必须具有方向和有效 pair support

PAC 语义是 slow-band phase → faster-band amplitude。当前 frontend 的矩阵确实按
`Z[..., i, j] = mean phase_i * centered_amp_j` 保留了方向，但 `FreqMIProduct` 和
`FreqMITopology` 使用整个 `nb × nb` 矩阵；只有 `FreqPhaseSteered` 明确使用
`i < j` 的 lower-triangular support。

因此 MI mixer 的 top-k 可能选择 self-pair、fast-phase → slow-amplitude 或物理意义
很弱的相邻 pair。shuffle control 能判断 band-pair identity 是否有用，却不能修复
treatment 本身的 support 定义。

### 3. MVL 不是默认唯一正确答案

Hülsemann 等的模拟比较表明：

- MVL 在长数据、高 SNR、高采样、单峰耦合时通常最敏感，但对调制强度、宽度和
  amplitude scale 更敏感。
- Tort MI 在短、噪、低采样以及可能双峰的 coupling 下更稳健，代价是 phase-bin
  数成为额外自由度。
- PLV 没有稳定统治区；GLM-CFC 对 modulation strength 敏感，但慢很多且怕噪声。

所以正式机制实验至少应并列报告 debiased MVL 与 Tort MI；preferred phase 另行保留。
这不是为了挑最好指标，而是判断负结果来自 PAC 不存在还是 estimator mismatch。

### 4. “显著 PAC”不等于独立振荡间的生理耦合

- 尖峰、sharp edge、锯齿/非正弦波的谐波会同时产生低频相位和高频能量，形成强 PAC。
- 非平稳的共同输入可以同时改变慢相位和高频幅值，产生相关而无跨频因果作用。
- 条件间 PAC 差异可能只由低频/高频 power 或 SNR 差异造成。Aru 等建议进行
  power-stratification 或至少报告 power-matched control。
- bispectrum 的 inside/outside 与 harmonic pattern 可帮助区分 nested oscillation 和
  broadband transient；Kramer 等还建议 raw trace、high-frequency-triggered average、
  harmonic spectrum 与 bicoherence 检查。

对分类任务而言，伪 PAC 仍可能是好用的 marker；但论文应称其为
“phase–amplitude statistical dependency”，除非做了上述控制，不应直接宣称神经机制。

### 5. 时间平均会错过短暂耦合

- ERPAC 在每个 event time point 跨 trials 回归 phase 与 amplitude，适合 evoked、
  time-locked coupling；它与单 trial 内长窗口平均回答不同问题。
- State-space PAC 可追踪短窗口、非平稳的 coupling，并给 posterior interval，不需
  构造大量 surrogate。
- 当前实现按 patch 计算 time-resolved MVL，这一方向是合理差异点；但 patch 内样本数、
  低频周期数和 estimator 方差需要明确。1 秒 patch 对 1–2 Hz 仅含 1–2 个周期，
  单 patch MVL 的稳定性很可疑。

## 三、与当前代码和实验的对应

| 项目 | 当前实现 | 文献给出的风险 | 判断 |
|---|---|---|---|
| PAC estimator | centered amplitude complex MVL，最后取模 | 短窗/噪声/双峰下可能不如 Tort MI | 需要 metric-control |
| pair support | frontend 全 band pair；MI mixers 也全 pair | PAC 应主要是 slow phase → faster amplitude | treatment 定义有污染 |
| filterbank | 8 个 log-spaced learnable sinc bands | 每个 pair 应覆盖 carrier sidebands | 没有 pair-wise 保证 |
| 时间尺度 | 每 channel、每 patch | 低频周期不足会使估计高方差 | 应按低频自适应窗口或汇聚 |
| 归一化 | 固定除数 100 | estimator scale 与 amplitude/SNR 相关 | 适合数值稳定，不等于统计校准 |
| artifact control | shuffle band-pair matrix | sharp wave、power、SNR 可同时存活 | shuffle 不是充分负对照 |

因此 §13.41 最准确的表述仍是：“在当前 estimator、频带和 TUEV 协议下，没有证据
MI-guided mixer 优于 attention。”不应扩成“架构侧 PAC 已普遍被证伪”。取消继续调
`k` / temperature 是合理的，但若以后重新打开该线，应以修正 measurement 为新实验，
而不是在原 mixer 上继续 fishing。

## 四、EEG foundation model 文献告诉我们的评测现实

最新统一 benchmark 基本印证了项目背景判断：

- 不同工作使用不同预处理、split、adaptation（linear probe / full fine-tune）与指标，
  单篇 SOTA 排名很难横向比较。
- NeuroAtlas 在 42 个数据集上发现，同一 domain 内模型排名能显著变化；EEG FM
  并不稳定胜过通用 time-series FM。
- 2026 EEG-FM benchmark 发现 specialist scratch model 在许多任务仍有竞争力，
  linear probe 经常不足，大模型规模也不自动带来更好泛化。
- 多维评测发现 FM 的优势更集中在 sleep、mental-health 等 long-context tasks；
  short-window BCI 和 channel-constrained 设置下小型监督模型仍然很强。
- stress test 显示 frozen representation 的 dataset identity 几乎可被完美解码，
  random-initialized encoder 有时胜过 pretrained encoder。当前最干净的正结果之一是
  CHB-MIT held-out subjects 上的 ictal vs same-session interictal，而这仍不等于
  sample-precise onset detection 或跨医院临床迁移。

这给“做出一个 SOTA 表格”的实用启示是：任务、split、窗口、adaptation 和 metric
本身就是研究设计的一部分。最容易形成清晰故事的不是声称 universal EEG FM，而是
在 PAC 合理的长上下文或 event-related 场景上做强结果，再用一两个严格 control
证明收益不是纯 preprocessing 或 dataset identity。

## 五、频谱自监督方向的撞车边界

- BIOT 已使用 channel-wise STFT token。
- LaBraM 已用 VQ neural spectrum prediction 作为 masked target。
- TFM-Tokenizer 已把 single-channel time-frequency motifs 离散化，并用时频 masking；
  还能作为 BIOT/LaBraM 的 plug-in。
- BandVQ 已把 delta/theta/alpha/beta/gamma 分开 VQ，并做 masked code prediction。
- SpecMoE 已做 STFT 上的 time、frequency、time-frequency Gaussian masking 和
  spectral gating。
- MFM 是视觉论文，但“在频域 mask-and-predict”这一宽泛算法表述已有明确先例。

所以 `cf_mixed` 若只写成“cross-frequency masking”，新颖性偏弱。真正可区分的方向应是：
mask distribution 或 prediction target 由**有方向的 phase→amplitude dependency**
定义，并且有 matched phase destruction、sideband destruction 或 time-shift control
证明模型依赖的是该 dependency，而非“多看低频”。

## 六、建议的最小判别实验

在不重新开启大规模架构搜索的前提下，先做一个小而能判机制的 2×2：

1. `measurement = current MVL` vs `pair-valid PAC`
   （slow→fast mask + pair-aware sideband/triplet filter + MVL/Tort-MI 双指标）。
2. `pairing = real` vs `phase-destroyed control`
   （保持每带 power、幅度分布和自相关，破坏低相位与高幅值的相对时序）。

先在人工信号上验证：

- 真 PAC 随 coupling strength 单调；
- 无 sideband 时应检测不到；
- sharp-wave-only 不应被误报为 nested oscillation，或至少能被 bispectral/harmonic
  flag 标记；
- 不同 patch length 下 estimator bias/variance 可见。

再在一个 PAC 最有生理与 benchmark 支持的数据集上做 probe，而不是立即把它塞回
Transformer。若 real > control 且与标签相关，再比较三种进入方式：

1. 只作 auxiliary target；
2. 只作 tokenizer/filter selection；
3. 只作 hard topology。

每次只改变一个位置。这样才能真正回答“architecture 与 pretrain 能否叠加”，而不是
比较两个同时改变了测量、mask distribution 和优化难度的系统。

## 七、目前最有希望的论文叙事

保守但可证的版本：

> Frequency-aware masked pretraining improves clinical EEG transfer, but its benefit does
> not arise from explicit low-to-high reconstruction. PAC-informed analysis reveals when
> phase–amplitude dependencies are measurable and when architectural injection fails.

若上述 pair-valid control 得到正结果，可升级为：

> A sideband-valid, time-resolved PAC prior supplies complementary information only when
> its estimator and intervention preserve the defining phase–amplitude relationship.

第二个版本既避开“首次 PAC + deep learning”的撞车，也把当前的负结果转化为一个有
价值的发现：不是 PAC 名词本身带来收益，而是 measurement validity 决定先验是否可用。
