# PAC-Former 论文库

这个目录用于保存 PAC-Former 的方法学依据、相近工作和 EEG 评测背景。当前共
28 篇 PDF；每篇均已通过 PDF 头、页数和全文抽取检查。原始 PDF 在 `papers/`，
便于检索的纯文本在 `text/`，文件校验值在 `SHA256SUMS.txt`。

> `2021_Zandvoort_Nolte_Filter_Parameters_PAC_thesis_chapter.pdf` 是作者公开的完整
> 博士论文；目标论文 *Defining the Filter Parameters for Phase-Amplitude Coupling
> from a Bispectral Point of View* 位于第 5 章。

## 1. PAC 与深度学习：最直接的相近工作

| 文件 | 核心内容 | 与本项目的关系 |
|---|---|---|
| `2020_Lemkhenter_Favaro_Phase_Swap.pdf` | 交换同一受试者/记录片段的 Fourier phase 与 magnitude，判断二者是否匹配 | 与 phase/amplitude 自监督叙事直接相邻，但不是低频相位调制高频幅值的 PAC 算子 |
| `2023_Li_et_al_PACNet_Adaptive_Filters.pdf` | 用 ERPAC 为每位受试者和任务选择高频子带，再送入 EEGNet 式分支 | PAC 作为自适应频带选择；不是 token mixer |
| `2023_Li_et_al_Complex_PAC_CNN_SEEG.pdf` | 将 Tort MI 和 preferred phase 合成复数 PAC 图，输入复数 CNN | 与保留复数 PAC 向量/优选相位最接近 |
| `2026_Lee_et_al_SleepPACNet.pdf` | Hilbert 低频复信号与高频信号在 CNN 分支融合 | 与架构侧“显式低相位 + 高频分支”最接近 |

## 2. PAC 方法与可靠性

| 文件 | 用途 |
|---|---|
| `2006_Canolty_et_al_High_Gamma_Theta.pdf` | Canolty complex mean vector / preferred phase 的经典来源 |
| `2008_Kramer_et_al_Spurious_PAC_Sharp_Edges.pdf` | 尖峰、锐边和非正弦波形如何制造显著但非独立振荡耦合 |
| `2010_Tort_et_al_Modulation_Index.pdf` | 基于相位分箱和 KL divergence 的 Tort MI |
| `2013_Voytek_et_al_ERPAC.pdf` | 在每个事件时刻跨 trial 计算 time-resolved PAC |
| `2015_Aru_et_al_Untangling_CFC.pdf` | CFC 的滤波、非平稳性、功率混淆、surrogate 与因果解释清单 |
| `2017_Cole_Voytek_Waveform_Shape.pdf` | 非正弦波形形状、谐波与“伪 PAC” |
| `2018_Kovach_et_al_Bispectrum_and_PAC.pdf` | PAC 与 bispectrum 的关系；区分 nested oscillation 和 transient/sharp-wave |
| `2019_Hulsemann_et_al_PAC_Metrics_Comparison.pdf` | PLV、MVL、Tort MI、GLM-CFC 的模拟比较 |
| `2020_Chehelcheraghi_et_al_Triplet_Filter_PAC.pdf` | 以 carrier 及两侧边带组成 triplet filter，避免窄高频带漏掉调制 |
| `2021_Zandvoort_Nolte_Filter_Parameters_PAC_thesis_chapter.pdf` | 从 bispectrum 推导 amplitude-band 带宽与 phase frequency 的配比 |
| `2022_Soulat_et_al_State_Space_PAC.pdf` | state-space oscillator + 动态 PAC，自动估频并表达不确定性 |

## 3. EEG foundation model 与评测

| 文件 | 选择理由 |
|---|---|
| `2024_Yang_et_al_BIOT.pdf` | channel-wise STFT token 与跨数据集 biosignal 预训练基线 |
| `2024_Jiang_et_al_LaBraM.pdf` | VQ neural spectrum prediction 与 masked code pretraining |
| `2025_Wang_et_al_CBraMod.pdf` | criss-cross 时空建模；当前 TUEV 强基线 |
| `2025_Liu_et_al_REVE.pdf` | 25,000 subjects、跨 montage 的 4D position encoding |
| `2025_EEG_Bench_Clinical.pdf` | 临床 EEG foundation model 标准化比较 |
| `2026_Kommineni_et_al_Generalization_Framework.pdf` | full fine-tune、linear probe、LoRA、样本/通道约束的多维评测 |
| `2026_Liu_et_al_EEG_FM_Benchmark.pdf` | 12 个开源 FM、13 个数据集、统一 LOSO/few-shot 对比 |
| `2026_NeuroAtlas_Benchmark.pdf` | 42 个数据集和临床级指标；展示同任务跨数据集排名不稳定 |
| `2026_Zare_Stress_Testing_EEG_FMs.pdf` | dataset identity、随机初始化、标签置乱等 negative controls |

## 4. 频谱自监督与 tokenizer 相近工作

| 文件 | 选择理由 |
|---|---|
| `2023_Xie_et_al_Masked_Frequency_Modeling.pdf` | 视觉领域频域 mask-and-predict 的通用邻近工作，不是 EEG 论文 |
| `2026_Jathurshan_et_al_TFM_Tokenizer.pdf` | 单通道 time-frequency motif、离散 tokenizer、时频遮挡 |
| `2026_Sukhbaatar_et_al_BandVQ.pdf` | 五个经典频带分别 VQ，再做 masked code prediction |
| `2026_Darankoum_et_al_SpecMoE.pdf` | STFT 平滑时频遮挡和 spectral mixture-of-experts |

## 阅读产物

项目级结论、撞车矩阵、当前实现审计与建议实验见
[`notes/PAC_FORMER_LITERATURE_REVIEW.md`](notes/PAC_FORMER_LITERATURE_REVIEW.md)。

