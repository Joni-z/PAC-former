# PAC-Former 大集群预训练 —— 交接方案

**更新 2026-07-31。** 依据：AGENT.md §13.43(架构)、§13.44a / §13.45(目标函数)。

---

## 0. 先读这一条

**预训练在当前证据下不是普遍增益。**

```
TUEV       最优预训练 0.5024  <  纯监督 0.5688     −0.067
Sleep-EDF  最优预训练 0.5572  <  纯监督 0.5732     −0.016
CHB-MIT    最优预训练 0.5767  >  纯监督 0.0180     +0.559
```

两个监督能正常工作的数据集上，九个目标函数里最好的那个仍然输给不预训练。唯一的
胜利来自监督直接塌到基线正样本率的 CHB-MIT。

**所以这次大 run 的目的不是"上预训练涨点"，而是回答一个明确的问题：**

> 把语料从单一来源扩到多来源，能不能翻转上面那三行。

如果翻不了，结论就是"这个架构不需要预训练"，那也是一个干净的结果，且省下后续所有
预训练投入。**判据必须是对比纯监督，不是对比其他预训练变体。**

---

## 1. 预训练数据（本次的核心决定）

### 用什么

| 语料 | 小时 | 群体 | 下游？ |
|---|---|---|---|
| **TUEP** | 513 | 癫痫/非癫痫门诊 | 否 |
| TUAB train | 829 | 异常/正常临床 | 是 |
| TUSZ train | 640 | 癫痫发作临床 | 是 |
| CHB-MIT train | 855 | 儿科头皮 | 是 |
| Sleep-EDF train | 649 | 健康睡眠、2 通道 | 是 |
| **合计** | **≈3,486** | **5 个语料 / 4 类人群** | |

**全部本地已预处理，无需下载。** 只用官方 train split，eval/test 从不接触。

### 为什么这样选

**关键依据：LaBraM 的数据量消融说小时数不是瓶颈。** 它的 Base 模型(5.8M)用
500 小时时，在 TUAB 上**超过**了自己 2500 小时的版本，在 TUEV 上达到 90%。

而我们失败的预训练用的是 TUEV(156h)和 CHB-MIT(855h)。**855h 已经超过
LaBraM Base 需要的 500h，却没有普遍收益。**

⇒ **瓶颈不是小时数，是语料多样性。** LaBraM 的 2534h 来自约 20 个语料，我们是 1 个。
所以本方案按**多样性**而非小时数组织：5 个语料、4 类人群（门诊癫痫 / 临床异常 /
儿科 / 健康睡眠），而不是把某一个语料堆大。

**TUEP 是唯一非下游语料，必须加。** 它已预处理成我们的格式（16 通道双极、200Hz、
10s 窗），且是 LaBraM 自己预训练语料的一部分（591h）。它提供唯一一份与评测完全
独立的数据。

**下游语料的 train split 可以用。** LaBraM 做过对照：把下游数据集放进预训练，
下游表现**不显著变化**。所以这不是作弊，但论文里要写明。若审稿关心，补一个
"仅 TUEP" 的对照即可。

**Sleep-EDF 只有 2 通道，照收。** xyz SpatialPE 支持可变 montage，已在 2 通道
Sleep-EDF 上验证过（监督 0.5732）。它是唯一的健康人群语料，多样性价值最高。

### 数据准备

```bash
export PACFORMER_DATA_ROOT=/scratch/zz5070/PAC-former
bash scripts/prepare_ssl_pool.sh     # 生成不含 _add 的 ssl_train mmap
```

TUSZ / CHB-MIT 的 `_add` 窗口是用标签生成的、且高度重叠，**仅从 SSL 池中剔除**
（微调仍然使用）。TUEP 需要新增一条 consolidate（当前脚本未含）。

---

## 2. 冻结的模型

| 部件 | 选择 |
|---|---|
| frontend | 8-band 可学习 sinc + 可微 Hilbert |
| **tokenizer** | **`pac_interaction` / `measured`** — 强制、规范不变的 PAC token |
| tokens | electrode × band × time patch |
| encoder | 三轴因子化 attention（time RoPE / space / freq） |
| freq mixer | **普通 attention**（PAC 只在 token 构造处进入） |
| space PE | 双极端点 xyz MLP |
| band PE | learned band index |
| capacity | d=256 / depth=8 / heads=8，约 8.6M |

主配置：[`configs/foundation/pacformer_base.yaml`](configs/foundation/pacformer_base.yaml)
（**需要更新**：加 `tokenizer_mode: pac_interaction`、`pac_token_mode: measured`，
并把 `pretrain_pool` 改成上面第 1 节的五个语料）。

架构依据：TUEV +0.081、Sleep-EDF +0.052、TUSZ 打平；三个参数对齐的匹配对照
（`uniform` / `concat` / `magnitude`）全部与 raw 基线打平（§13.43-J）。

---

## 3. 冻结的预训练目标

**标准 `random` MAE**，`mask_ratio: 0.5`，重建每个 (electrode, band, patch) 的
log mean analytic amplitude，band-balanced Smooth-L1。

九个目标函数 × 三个数据集消融的结论，完整数据见 AGENT.md §13.45。`cf_mixed` 是最接近的
落选者（minimax worst-case −0.0143，平局但从未赢）。

**两个必须带走的参数注意事项：**

- **`weight_decay` 必须与下游监督协议对齐。** TUEV 上 1e-4 → 1e-5 让结果涨 0.052，
  这个量级足以翻转结论。
- **`band_pe: index` 不动。** 原理由（与 cf_mixed 配合最好）已作废；新理由是
  六个 BandPE 对比方向全不一致，没有证据支持改动。

---

## 4. 数据预处理（保持不变）

BIOT 对齐：16 通道双极、200 Hz、per-channel q95 归一化、5 秒裁剪。
数据集采样概率正比于窗口数的平方根；batch 内只来自一个语料（为 runtime montage
坐标保留接口）。

---

## 5. 证据顺序与证伪规则

1. **full-budget 主 run vs 相同预算的纯监督**，在全部五个下游任务上。
   这是判据，不是与其他预训练变体比。
2. 通过后再跑 seed 0/1/2：TUEV κ、TUSZ/CHB-MIT PR-AUC、TUAB AUROC/PR-AUC。
3. 只有 (1) 为正才做 mask 方向的机制对照（band-random / low-frequency）。

**证伪规则：**

- 主 run 在 TUEV 和 Sleep-EDF 上仍输给纯监督 ⇒ **结论是"该架构不需要预训练"**，
  停止预训练线，不要再换目标函数（九个已经试过）。
- 主 run 打平 band-random 对照 ⇒ 贡献改称 structured band masking。
- 主 run 赢不了 random MAE ⇒ 不得把任何外部 SOTA 数字归因于 PAC-inspired objective。

---

## 6. 生产代码（已就绪）

- `foundation_pretrain.py`：torchrun/DDP、BF16/FP16、梯度累积、AdamW + warmup-cosine、
  梯度裁剪、逐 epoch checkpoint、含 RNG 的精确 resume、供下游加载的 `mae_state.pt`
- `DatasetMixtureBatchSampler`：sqrt 采样 + 同语料 batch + DDP 各 rank 不相交等步调度
- `MAEPretrain` / `SpatialPE` 的 runtime montage 选择
- `scripts/launch_foundation_pretrain.sh`、`scripts/prepare_ssl_pool.sh`
- 四个 full-finetune 配置（分层 LR + cosine）

本地烟测全绿（`scratchpad/smoke_foundation.py`、`configs/smoke/foundation.yaml`）。

---

## 7. 未完成

- **AMD / ROCm 迁移未开始。** 需评估 Hilbert 的 FFT、复数运算、DDP/BF16 三处兼容性。
- `configs/foundation/pacformer_base.yaml` 需按第 1、2 节更新（tokenizer + 语料池）。
- `scripts/prepare_ssl_pool.sh` 需增加 TUEP。
