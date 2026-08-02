# PAC-Former 大集群预训练冻结方案

> ## ⚠️ 2026-07-30：本方案的**目标函数部分已失去证据支撑，暂缓启动**（AGENT.md §13.44a）
>
> 本文冻结于 2026-07-28，当时 backbone 还是 raw-token 版本。此后 backbone 换成了
> **强制 gauge-invariant PAC interaction tokenizer**（§13.43），并在两个数据集上把
> objective × BandPE 的 2×2 完整重跑。结果推翻了本文两条核心依据：
>
> 1. **`cf_mixed` 优于标准随机 MAE —— 未被证实。** 六个同 BandPE 对比(CHB-MIT/TUEV ×
>    hz/index × raw/PAC)方向全不一致：两赢两输两平。§13.24 的 Tier A 判据在新架构下
>    **依然没有满足**。
> 2. **“Hz PE 与 objective 的频率先验发生干扰”（下表第 38 行的选型理由）—— 不成立。**
>    同样六个对比全打架。`band_pe: index` **保持不变**，但理由改为“没有证据支持改动”，
>    不再是“与 cf_mixed 配合最好”。
>
> **真正稳住的只有架构。** CHB-MIT 上 `random MAE + index`（无目标函数先验、无频率 PE、
> 只有 tokenizer）从 0.1575 升到 **0.5767**，单这一格就超过旧架构 2×2 的最好成绩
> 0.5472。四格散布从 0.3897 压到 0.1751 → 0.1295：先验一旦进了 token 几何，再从
> objective 或 PE 加同类先验就只剩噪声级抖动。
>
> **另有一个未解决的上游问题：TUEV 上预训练是倒退。** 同一个 tokenizer，纯监督
> 0.5688，预训练+微调最好格 0.5540，`cfm_idx` 只有 0.4726。在回答“这个架构下预训练
> 到底还有没有正收益”之前，比较 objective 变体是没有意义的。
>
> **2026-07-31 更新：目标函数已消融完毕（AGENT.md §13.45）。**
>
> 九个目标函数 × 三个数据集跑完，按预登记的 minimax 判据：
>
> **目标函数改为标准 `random` MAE。** `cf_mixed` 是最接近的落选者（worst-case −0.0143，
> 平局但从未赢过）；`crossfreq`（−0.049）、`bandrand`、以及新设计的判别式相位 pretext
> `pac_consistency`（TUEV +0.086 但另外两个数据集都输）全部淘汰。
>
> **但更要紧的是使用条件：预训练不是普遍增益。**
>
> ```
> TUEV       random MAE 0.5024  <  纯监督 0.5688     −0.067
> Sleep-EDF  random MAE 0.5572  <  纯监督 0.5732     −0.016
> CHB-MIT    random MAE 0.5767  >  纯监督 0.0180     +0.559
> ```
>
> **在两个监督能正常工作的数据集上，最优预训练方案依然不如不预训练。** 唯一的胜利来自
> CHB-MIT，而那里监督直接塌到基线正样本率（1% 正样本）。
>
> 所以记录方案是：**`random` MAE，且仅用于监督会失败的低标注/极不平衡任务**。这是救场
> 手段，不是通用先验。在当前规模（单数据集、15–30 epoch）下没有证据支持更强的主张；
> 大规模多语料下是否改变，未知且未测。
>
> **启动大集群前仍需确认的一件事**：上述结论全部来自单数据集小规模预训练。若大集群的
> 价值主张是"多语料大规模能翻盘"，那必须先用一个小规模的多语料对照验证这个假设，而不是
> 直接投产。本文其余部分——数据、preprocessing、DDP、采样、checkpoint、评测顺序——
> **仍然有效**，未受影响。

状态：可交付。本文定义要跑什么、为什么这样跑，以及论文能说什么。它取代旧的
“显式 PAC matrix 指导 attention”主线；旧实验仍保留为支持设计选择的负结果。

## 1. 方法一句话

**PAC-Former 是一个 electrode × frequency × time 三轴 EEG Transformer，使用
PAC-inspired asymmetric cross-frequency masking：一半 batch 做普通随机掩码，一半
隐藏高频 band token 并从低频与时空上下文重建其 log-amplitude。**

这里 PAC 是 corruption/masking distribution 的生理启发，不是把噪声很大的 MVL
矩阵硬塞进 attention，也不声称该目标唯一识别出了生物 PAC。

建议论文标题方向：

> **PAC-Former: Cross-Frequency Masked Pretraining for Generalizable EEG
> Representation Learning**

副叙事：

> A neural prior is more useful as a learning problem than as a fixed attention
> operator.

这句话有项目内部证据支撑：显式 coupling operator、MI product 和 MI topology 都没有
稳定胜过普通 frequency attention；相反，`cf_mixed` 在 CHB-MIT full-finetune 中相对
matched random-MAE 有 3.5× PR-AUC 增益。论文中应把 operator 结果作为设计诊断，
不要写成 PAC 在所有架构中都无效。

## 2. 冻结的模型

| 部件 | 最终选择 | 依据 |
|---|---|---|
| frontend | 8-band learnable sinc + differentiable Hilbert | 生理参数化、band 显式；现有正结果均基于此 |
| tokens | electrode × band × time patch | frequency axis 是 cross-frequency mask 可定义的前提 |
| encoder | factorized time / space / frequency attention | 从 scratch 去掉 frequency axis 在 TUAB/TUEV/TUSZ 都下降 |
| space PE | bipolar endpoint xyz MLP | TUEV/TUSZ 有增益，且支持跨 montage 坐标语义 |
| band PE | learned band index | 与 `cf_mixed` 配合最好；Hz PE 与 objective 的频率先验发生干扰 |
| capacity | d=256, depth=8, heads=8，约 8.6M 参数 | 比已验证的 1.65M 骨干有预训练容量，又不盲目上百 M |
| PAC operator | 不使用；frequency mixer 为普通 attention | MI product/top-k 与 shuffle control 都没有正证据 |

大模型没有从监督小数据的 wide/deep ablation 中直接外推。8.6M 是一个保守的
foundation-size 扩容：能吃更多无标签数据，但仍适合 5 秒、16×8×5 token grid 和
常规多卡训练。

## 3. 冻结的预训练目标

主配置：
[`configs/foundation/pacformer_base.yaml`](configs/foundation/pacformer_base.yaml)

每个 batch 以 0.5 概率选择：

1. **Random MAE**：独立随机隐藏 50% token，保证所有频段与任务的一般覆盖。
2. **Cross-frequency mask**：隐藏最高的 4/8 个 band，重建每个
   electrode × band × patch 的 log mean analytic amplitude。

两种 mask 都隐藏 50%，所以比较不受 corruption budget 混淆。重建使用
band-balanced Smooth-L1：每个被重建频带先独立求平均，再跨 band 平均，避免 EEG
低频能量和数据集尺度主导梯度，同时保留绝对 log-amplitude target。

不要在主 run 前调 `mixed_p`。0.5 是已有证据最稳的跨数据集折中；若主结果成立，
再用 `--override mixed_p=0.25/0.75` 做 dose-response，而不是把它包装成调参。

## 4. 数据与预处理

第一阶段使用 TUAB、TUEV、TUSZ、CHB-MIT 的 **train split only**：

- 统一到 200 Hz；
- 16-channel bipolar TCP montage；
- 每通道按绝对值 95% quantile 归一化，与 BIOT pipeline 对齐；
- 随机裁成 5 秒，10 秒数据的 crop 同时充当时移增强；
- 不额外固定 bandpass：learnable sinc frontend 负责频带分解；
- dataset sampling probability 与窗口数的平方根成正比，避免最大数据集垄断更新；
- batch 内只来自一个数据集，为运行时 montage 坐标和未来可变通道数保留接口。

TUSZ/CHB-MIT 的监督预处理包含由 seizure label 产生的密集 `_add` 重叠窗口。它们适合
下游训练，但不应进入“无标签”预训练，否则 SSL 已经间接使用标签，而且大量近重复
窗口会虚增 corpus size。运行
[`scripts/prepare_ssl_pool.sh`](scripts/prepare_ssl_pool.sh) 生成不含 `_add` 的
`ssl_train` mmap arrays；主配置已指向这些文件。

当前 runtime xyz 接口已经支持 batch 级 montage 切换。增加真正不同 montage 的新数据
时，还需要：

1. 在 `models/montage.py` 注册其电极/双极坐标；
2. 在 `build_pretrain_pool` 增加对应 loader adapter；
3. 保持同一 batch 内 montage 一致。

不要为了凑数据量把不同导联硬 padding 为 16 channel；那会把缺失模式变成 dataset ID。

## 5. 训练工程

入口：
[`foundation_pretrain.py`](foundation_pretrain.py)

已经支持：

- single GPU 或 `torchrun` DDP；
- BF16/FP16、gradient accumulation、gradient clipping；
- AdamW + warmup cosine schedule；
- dataset-balanced homogeneous batch sampler；
- 每 epoch 完整 checkpoint；
- optimizer/scheduler/scaler/RNG 的精确恢复；
- 单独输出 `mae_state.pt`，可直接被现有下游 `init_from` 加载；
- `${PACFORMER_DATA_ROOT}` 路径展开，迁移集群不需要改代码。

启动示例：

```bash
export PACFORMER_DATA_ROOT=/path/to/processed-data
export NPROC_PER_NODE=8
./scripts/prepare_ssl_pool.sh
./scripts/launch_foundation_pretrain.sh \
  configs/foundation/pacformer_base.yaml
```

断点恢复：

```bash
torchrun --standalone --nproc_per_node=8 foundation_pretrain.py \
  --config configs/foundation/pacformer_base.yaml \
  --resume checkpoints/foundation-pacformer-base/latest.pt
```

默认 8 GPU × 16/GPU × accumulation 2，effective batch 256，50 epochs。若显存不足，
优先降低 per-GPU batch 并增加 accumulation，保持 effective batch；不要先缩模型。

## 6. 必须跑的实验顺序

### Stage A：主 run 与归因 control

同预算完整预训练：

1. `pacformer_base.yaml`：主方法；
2. `pacformer_base_random.yaml`：matched standard-MAE control。

只有主方法在至少两个目标任务上稳定胜过 random-MAE，才能把收益归因到 asymmetric
cross-frequency masking。单独胜过外部模型但不胜 random control，只能说明 backbone/
scale 有效，不能说明 PAC-inspired objective 有效。

### Stage B：下游 full-finetune

```bash
python pretrain.py --config configs/foundation/finetune_tuab.yaml
python pretrain.py --config configs/foundation/finetune_tuev.yaml
python pretrain.py --config configs/foundation/finetune_tusz.yaml
python pretrain.py --config configs/foundation/finetune_chbmit.yaml
```

配置使用 encoder 5e-5、head 5e-4、AdamW、cosine schedule，并保留项目已有的
best-validation checkpoint 与大型数据集 step-level validation。

主指标：

| 数据集 | headline metric | 角色 |
|---|---|---|
| TUEV | Cohen's κ | 最适合展示三轴结构与多类事件识别 |
| TUSZ | PR-AUC | 大规模、极不均衡 seizure detection |
| CHB-MIT | PR-AUC | `cf_mixed` 已有最强内部正结果，但不与 BIOT paper row直接比较采样率 |
| TUAB | AUROC + PR-AUC | BIOT-aligned 外部锚点，但任务较饱和 |

每个下游 checkpoint 至少跑 seeds 0/1/2；预训练本身可以先固定 seed 0。报告
mean ± std，并同时保留 segment-level 标准指标和 subject/recording-level 补充指标。

### Stage C：只在主方法成立后做机制

- `pacformer_base_bandrand.yaml`：同样整带隐藏，但 band 随机选；
- `pacformer_base_lowfreq.yaml`：方向反转，隐藏低频；
- `mixed_p ∈ {0, .25, .5, .75, 1}` dose-response；
- 主 checkpoint 对 phase-shift / amplitude-preserving negative control 的 probe。

前两个 control 可以先跑 20–25 epochs；它们用于解释，不应抢主方法和 random control
的完整预算。`bandrand` 若追平主方法，故事应改成“structured band masking”，不能继续
声称 high-frequency direction 是关键。

## 7. SOTA 表格的组织

主表必须同时有两条比较线：

1. **外部竞争性**：BIOT、CBraMod、LaBraM、REVE，以及强 specialist baseline；
2. **内部归因**：scratch、random-MAE、PAC-inspired cross-frequency MAE。

优先使用本项目已经对齐的 TUAB/TUEV split 和指标跑公开 checkpoint；论文原表数字只作
参考，不把不同 preprocessing/split 的数字混成同一结论。TUSZ/CHB-MIT 主要依靠 matched
pipeline 内部比较，再补充 published rows。

最有希望的 headline 不是“一个模型在所有 EEG 数据集都统治”，而是：

> PAC-inspired cross-frequency masking gives a consistent advantage on
> clinically imbalanced event detection while remaining competitive on
> heterogeneous multiclass EEG.

如果 TUEV 不赢但 TUSZ/CHB-MIT 明显赢，可以把 scope 收到 clinical event detection；
这仍是完整故事。反过来，如果只有 TUAB 小幅赢，不足以支撑主论文。

## 8. 明确不做的事情

- 不恢复 MI product/top-k、coupling gate 或继续调其 temperature/k；
- 不声称 cross-frequency reconstruction 的唯一解是生物 PAC；
- 不使用 test data 做 filter selection、normalization statistics 或 pretraining；
- 不把 label-driven seizure oversampling 伪装成无标签语料规模；
- 不在主表混用 linear probe 与 full-finetune；
- 不通过为每个数据集单独挑 `mixed_p` 来制造“统一方法”。

这套方案的可卖点不是一个复杂新 layer，而是三件事形成闭环：

1. frequency-explicit tri-axial representation；
2. PAC-inspired directional corruption objective；
3. heterogeneous EEG 上防 dataset shortcut 的平衡、坐标化预训练。

