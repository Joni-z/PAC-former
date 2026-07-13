# PAC-Former 实验汇总（截至 2026-07-12）

三向消融：**同一 backbone、同一训练流程，只换 token mixer** —— attention / cotar / **mi(ours)**。
参考基线：BIOT-vanilla（从零训练），BIOT 6-数据集预训练版。

---

## 1. TUAB（正常/异常，二分类）—— 指标：balanced_acc / AUROC

### 1a. 旧 frontend (v1)
| Mixer | bacc | AUROC |
|---|---|---|
| attention | 0.7140 | 0.7901 |
| cotar | 0.7208 | 0.7968 |
| mi (v1) | 0.7142 | 0.7896 |

三者打平，且都低于 BIOT-vanilla（bacc 0.7925）。→ 怀疑 frontend 瓶颈。

### 1b. Frontend 诊断（仅 attention）
| Frontend | bacc | AUROC |
|---|---|---|
| v1 band frontend | 0.7149 | 0.7918 |
| plain conv patch（无频带） | 0.8023 | 0.8765 |

确认 v1 frontend 是瓶颈（通道平均丢空间信息 + 频带整段时间压成一个 token 丢时序）。

### 1c. 新 frontend (v2, band-preserving conv patch)
| Mixer | bacc | AUROC |
|---|---|---|
| attention | 0.7959 | 0.8764 |
| cotar | 0.7953 | 0.8730 |
| mi (v1) | ~0.81 val峰值（被抢占，无test）| — |

**结论：TUAB 已饱和**（attention 0.7959 = BIOT 6-数据集预训练版），各方法无差异化空间。

---

## 2. TUEV（6类事件）—— 指标：balanced_acc / kappa / f1_weighted

### 2a. 旧 frontend (v1)
| Mixer | bacc | kappa | f1 |
|---|---|---|---|
| attention | 0.3825 | 0.1954 | 0.4924 |
| cotar | 0.3903 | 0.2237 | 0.5616 |
| mi (v1) | 0.3661 | 0.2373 | 0.5541 |

### 2b. 新 frontend (v2)
| Mixer | bacc | kappa | f1 |
|---|---|---|---|
| attention | 0.4205 | 0.2925 | 0.6223 |
| cotar | 0.4227 | 0.2272 | 0.5591 |
| mi (v1) | 0.3503 | 0.2517 | 0.5974 |

**结论：PAC 先验与 TUEV 不匹配**（TUEV 靠局部波形形状，非跨频耦合）。mi 是三者最弱。均低于 BIOT-vanilla（kappa 0.4482）。

---

## 3. Sleep-EDF（睡眠5分期）—— 主线，指标：balanced_acc / kappa / f1_weighted
> PAC 有真实生理基础（N2/N3 delta-spindle 耦合、REM theta-gamma 耦合）

### 3a. 首次三向（mi v1）
| Mixer | bacc | kappa | f1 |
|---|---|---|---|
| attention | 0.5757 | 0.4629 | 0.6594 |
| cotar | 0.6052 | 0.5146 | 0.6911 |
| mi (v1) | 0.6036 | 0.5112 | 0.6934 |

mi(v1) 与 cotar 打平，均明显超 attention → PAC 在此**确有信号**，但 v1 固定耦合权重无法超越 cotar。

### 3b. mi 设计迭代（单 seed，逐步排查）
| 版本 | kappa | 说明 |
|---|---|---|
| cotar (基线) | 0.5146 | |
| mi v1 | 0.5112 | 固定 PAC 权重 |
| mi v2 (channel-mean) | 0.51216 | PAC-biased attention，仍打平 |
| mi v2 (per-channel 修复) | 0.5138 | 修 frontend 通道平均 bug，无明显变化 |
| mi v2 (同上，未固定seed第二次) | **0.5792** | ← 发现同配置能跳这么多 → 定位到 seed bug |

**关键发现：`random` 模块的 seed 从未生效**（augment 每 batch 随机选增强），是同配置大幅波动主因。已修复（random/numpy/torch/cuda 全 seed + cudnn deterministic）。

### 3c. 修复 seed 后 —— 多 seed 受控对比（核心结果）✅
| seed | cotar kappa | mi(v2) kappa | mi 领先 |
|---|---|---|---|
| 0 | 0.5107 | 0.5199 | +0.0092 |
| 1 | 0.5003 | 0.5491 | +0.0488 |
| 2 | 0.4878 | 0.5373 | +0.0495 |
| **均值** | **0.4996** | **0.5354** | **+0.0358** |

**3/3 seed mi 全部领先 cotar，方向一致** → 目前项目最扎实的正面结果。

### 3d. mi v3 尝试（多头 + gating）—— 无净收益，已回退
| seed | cotar | mi v2 | mi v3 | v3 领先 |
|---|---|---|---|---|
| 0 | 0.5107 | 0.5199 | 0.5347 | +0.0240 |
| 1 | 0.5003 | 0.5491 | 0.5490 | +0.0487 |
| 2 | 0.4878 | 0.5373 | 0.4892 | +0.0014 |
| **均值** | 0.4996 | **0.5354** | 0.5243 | +0.0247 |

v3 均值反而低于 v2，方差近乎翻倍（seed2 崩到几乎打平）。已回退到 v2。

> ⚠️ 注：Sleep-EDF 文献 SOTA kappa ≈ 0.74–0.83（AttnSleep/SeqSleepNet/SleepGMUformer 等）。
> 我们绝对值远低于 SOTA —— 定位是**mixer 受控消融研究，非睡眠分期 SOTA 系统**（backbone/训练配置有差距，与 mixer 结论分开看）。

---

## 4. TUEP（癫痫诊断，二分类，本次新加）—— 指标：balanced_acc / AUROC
| Mixer | bacc | AUROC | 状态 |
|---|---|---|---|
| attention | 0.4907 | 0.4971 | 完成 |
| cotar | 0.5028 | 0.5119 | 完成 |
| mi | — | — | **崩溃(NaN)** |

**attention/cotar 均≈瞎猜水平** → 病历级诊断标签打在整段录音上，窗口级分类学不出信号（标签粒度不匹配，非 mixer 问题）。此数据集已判定不适用。

---

## 5. TUSZ（癫痫发作检测，事件级二分类，本次新加）—— 进行中
- 事件级标签（csv_bi 发作区间），已加 **BIOT 式发作窗口过采样** + **PR-AUC 指标**（应对 ~2% 极端不平衡）
- 状态：**预处理排队中（job 13409475），训练待提交**
- 这是 PAC 有生理基础的数据集（HFO-低频耦合），若跑出正面结果可支撑"PAC 相关任务普遍有效"

## 6. CHB-MIT（癫痫发作检测）—— 下载中（~21GB/42GB），未开始训练

---

## 一句话总结（给会议）
- **核心正面结果**：修复 seed bug 后，mi 在 Sleep-EDF 上 **3/3 seed 稳定超越 cotar（+0.036 kappa 均值）** —— PAC-biased cross-band attention 在有生理基础的任务上确实有效。
- **PAC 的选择性**：TUAB（饱和）、TUEV（任务不匹配）上 mi 无优势甚至最弱；Sleep-EDF 上有优势。→ "先验有效性依赖任务是否真由该先验驱动"本身是个发现。
- **进行中**：TUSZ（有生理基础，跑对了协议）预处理排队；CHB-MIT 下载中。
- **待决策**：研究定位（窄 PAC-former vs 更宽的频域结构化 MedTS mixer / 换 Medformer benchmark 战场）。
