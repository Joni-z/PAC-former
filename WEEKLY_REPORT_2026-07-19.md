# PAC-Former 周报 — MI v4 之后的进展(2026-07-19)

## 一句话主线

上周我们把 PAC(跨频耦合)当成**网络里的一个结构层**反复迭代(v3→v4→v5),想让它在监督训练下 work。这周**证明了这条路走不通、并诊断出确切原因**,于是做了一个战略转向:**把 PAC 从"架构层"改成"自监督预训练目标"**,拿到了第一批正面证据;同时探索了一个更强的 parameter-free 相位原语。

逻辑链条:**继续修 mixer(v5)→ 失败 → 找到两个根因 → 转向 SSL 目标 → 首个正面结果 → 探索 phase 原语**。

所有结果均为 seed 0(单 seed dev)。

---

## 1. MI v5 重设计(承接 v4 的问题)

**为什么要 v5**:v3/v4 在六数据集 sweep 上有个稳定的形状——**PAC 生理上重要的数据集赢**(Sleep-EDF、CHB-MIT),**PAC 非主导的数据集平庸甚至输**(TUAB、TUEV)。

根因(7/14 PI review 定的):v3/v4 把 PAC 当成加在 QK logits 上的**加性 bias**,由单个标量 `pac_scale` 缩放。非 PAC 数据上训练直接把 `pac_scale → 0`,退化成一个**单头、band 级**的 attention——而对照的 attention baseline 是**多头、token 级**的。所以退化后的 MI **严格弱于 baseline**,地板踩在 baseline 之下,这就是"非 PAC 平庸"的症状。

**v5 的改法**:两条并行分支 + 输入条件门控

```
out = 多头自注意力(x)  +  g · redistribute(PAC 耦合聚合)
g   = sigmoid(gate_mlp(尺度不变的耦合统计量))   # 每 (样本, 目标频带) 一个
```

- **地板 = baseline(构造保证)**:分支 A 就是原版 attention,`g→0` 时 MI ≡ baseline(实测 forced-zero gate 时 `max|MI − attention| = 0.0`)。非 PAC 数据再也压不到 baseline 以下。
- **天花板保留**:`g` 高时注入 PAC 聚合。
- **自适应**:`g` 读耦合列的三个尺度不变统计量(均值驱动 / 峰值驱动 / 峰度),按频带**检测**是否真有跨频结构。
- CPU 上全绿,但当时还没上真 EEG。

---

## 2. CoTAR baseline 健全性核查(堵一个 reviewer 的口)

**风险**:band frontend 的 ablation 里 CoTAR 有时连原版 attention 都不如(TUEV:CoTAR kappa 0.2272 < attention 0.2925)。reviewer 会说"你 CoTAR 没实现/调好",连带质疑所有"MI > CoTAR"的结论。

**做法**:给 attention 和 CoTAR 都换上朴素 conv frontend(CoTAR 的原生 token 空间)对打。

**结果——通过,而且符号翻转了(TUEV,seed 0):**

| frontend | attention kappa | cotar kappa | 差 |
|---|---|---|---|
| band(主 ablation) | 0.2925 | 0.2272 | **−0.065(CoTAR 输)** |
| conv(CoTAR 原生空间) | 0.2303 | **0.2501** | **+0.020(CoTAR 赢)** |

只换 frontend、mixer 代码不动,CoTAR 和 attention 的**排序就反转**。→ 证明 **CoTAR 实现正确、是个健康 baseline**,它在主 ablation 里输是因为 band frontend 对它不利(置换不变的全局池化、拿不到位置编码、把 32 band 塌成 rank-1)。所以"MI > CoTAR"是**对一个能打的 baseline 的公平胜利**。

(TUAB 第二个数据点:conv 下两者基本打平,所以诚实结论是"CoTAR 在原生空间 ≈ attention、TUEV 上更强",不是残废 baseline。)

---

## 3. ⭐ v5 在真 EEG 上失败 —— 以及解释它的两个关键发现(本周转折点)

**v5 结果(seed 0),对历史 baseline:**

| 数据集 | v5 mi | 参照 | 判定 |
|---|---|---|---|
| Sleep-EDF kappa | 0.5101 | v3 mi 0.5199 / cotar 0.5107 | ✗ v3 的领先没了 |
| CHB-MIT AUROC | **0.5721** | v3 mi 0.735 / attn 0.642 | ✗✗ **比两个 baseline 都低** |
| TUAB bacc/AUROC | 0.7954 / 0.8748 | attn 0.7959 / 0.8764 | ≈ 平(地板守住,无增益) |
| TUEV bacc/kappa | 0.4334 / 0.2825 | attn 0.4205 / 0.2925 | ≈ 混合 |
| TUSZ AUROC/PR | 0.8001 / 0.5052 | v3 mi 0.7985 / 0.4697 | ✓ 微弱增益 |

v5 的**地板修好了**(TUAB/TUEV 追平 baseline),但**代价是天花板没了**,CHB-MIT 上灾难性崩塌。门控诊断:CHB-MIT 上 `g` 在 6 层里 5 层冻在 **~0.50**,加性形式 `attn + g·pac_delta` **无法表达"纯 band 级算子"**,而 v3 恰恰是靠这个赢的,门控就骑墙、两头不讨好。

### 发现 1:PAC 先验从来就不是监督胜利的功臣

v3 的 `pac_scale` 收敛轨迹:

| 数据集 | v3 `pac_scale` 收敛值 | v3 mi 分 | attention |
|---|---|---|---|
| Sleep-EDF | 0.17 / 0.25 / 0.28(非零) | kappa 0.5199 | cotar 0.5107 |
| CHB-MIT | **5e-05 / -9e-05 / 0.0(第 4 轮就塌)** | **AUROC 0.735** | 0.642 |

CHB-MIT 最好的 checkpoint(epoch 5)`pac_scale` 已经 ≈ 0,却还赢 attention ~0.09 AUROC。**→ 真正的资产是那个 band 级瓶颈(把每 band 的 patch 均值池化、跨 band 混合、再广播回去),不是 PAC bias。** 这直接否掉了"我们把 PAC 写进 mixer 所以赢"这个 reviewer-facing 说法(用我们自己的证据)。它**不是说 PAC 没用**,而是说:**单数据集监督训练没有任何动力去学跨频机制**——它会走最省的路去拟合标签。**先验只有在目标函数逼它时才会兑现。**

### 发现 2:我们把耦合算成了一摊糊

`coupling_matrix` 在**整个窗口**上做 einsum 再对通道 `.mean` → **每个样本一个 32×32 矩阵**,在 16 电极 × 2000 时间步上平均掉了。而癫痫是**局灶的**(少数电极)、**瞬态的**(几秒)。全通道全时间一平均,得到的统计量在样本间近乎恒定、几乎没有判别信息。CHB-MIT 上 `pac_scale→0` 于是是优化器**正确**的选择。**→ 不是先验错了,是先验被平均没了**,而且是个 bug 类缺陷。修法:按 **(通道, 时间-patch)** 算耦合——这自然落在下面第 4 节的三轴网格里。

**结论**:v5 放弃。v6(凸组合 `(1-g)·attn + g·v3算子`,两端精确可达)写了、CPU 验了,但**从未提交**——PI 叫停,决定**整体重设计架构**面向 foundation model,而不是继续打补丁修 mixer。

---

## 4. 战略转向:v2 架构 + 把 SSL 目标当基石

这是承接两个发现的必然决定:

- 既然**监督训练没动力学 PAC**(发现 1)→ 那就**换成自监督目标去逼它学**。
- 既然**耦合被全局平均毁了**(发现 2)→ 新架构用**三轴网格**(电极 × 频带 × 时间-patch),按 (通道, patch) 算耦合。

新设计:频率原生 tokenization(sinc 频带 + 显式相位/幅度)+ 三轴 transformer,只把**频率轴 mixer** 拿来做 ablation。**贡献不再赌"更强的架构模块",而是赌"一个逼模型学 PAC 的预训练目标"。**

---

## 5. ⭐ 首批 SSL 基石结果(7/18–19,seed 0)

### (a) 监督 coherence-gate mixer:再次全面无胜

新原语 `FreqCoherenceGate`(把 attention 概率乘一个耦合导出的门再归一化,init 时 ≡ 纯 attention):

| 数据集 | coherence | attention | coupling | cotar |
|---|---|---|---|---|
| TUAB (bacc/auroc/pr) | 0.797/0.868/0.858 | 0.794/0.867/0.865 | 0.797/0.872/0.869 | — |
| CHB-MIT | 0.500/0.532/0.018 | (跑挂) | 0.500/0.526/0.018 | 0.500/0.645/0.047 |
| Sleep-EDF (bacc/f1/kappa) | 0.622/0.689/0.509 | 0.601/0.692/0.511 | 0.626/0.702/0.531 | 0.619/0.731/0.572 |
| TUEV | 0.487/0.650/0.344 | 0.525/0.660/0.376 | 0.515/0.649/0.370 | — |
| TUSZ | 0.631/0.829/0.583 | 0.697/0.826/0.577 | 0.654/0.835/0.605 | — |

**哪都没赢。** 这是连续第三个 PAC-as-a-layer 在监督下失败(v4 门控、v5、coherence)。→ **"架构侧 PAC 层 + 监督损失"当成已关闭的问题,不值得再试。** 正好印证发现 1。

### (b) ⭐ crossfreq MAE vs random MAE:核心赌注,首个正面证据

`crossfreq`(**我们的**)= mask 掉整个高频半区,只留低频可见,逼模型用 **低频相位→高频幅度耦合** 去重建高频幅度。`random` = 标准 MAE 对照。都跑 30 轮预训练 + 20 轮 linear probe。

| 数据集 | crossfreq | random | 判定 |
|---|---|---|---|
| TUAB (auroc) | 0.857 | 0.809 | crossfreq 赢 ✓ |
| CHB-MIT (auroc / pr_auc) | 0.878 / **0.393** | 0.743 / 0.136 | crossfreq 大赢(pr_auc ~3×)✓ |
| TUSZ (auroc) | 0.836 | 0.809 | crossfreq 赢 ✓ |
| TUEV (kappa) | 0.271 | 0.356 | random 赢 ✗ |
| Sleep-EDF (kappa) | 0.451 | 0.483 | random 赢 ✗ |

**读法**:crossfreq **3/5 赢**,且在两个极度不均衡的癫痫二分类(CHB-MIT、TUSZ)+ TUAB 上大幅领先;在两个多分类任务(TUEV 6 类、Sleep 5 类)上输。呈现一个**二分类 vs 多分类的分界**(尚未证因果)。**这是 SSL 基石真正开始兑现的首个正面证据**,但仍是部分的(3/5,单 seed)。

**外加一条强证据**:CHB-MIT 上所有从头监督全塌(0.53),但 crossfreq 预训练 → **0.878 AUROC**。**预训练本身有价值。**

---

## 6. operator × objective 的 2×2(有一个关键空洞)

**动机**:第 5(b) 节的 10 个 pretrain 全用 `freq_mixer=attention`、耦合矩阵被清零——所以**耦合算子本身从没在 crossfreq 目标下被真正测过**。旗舰主张"耦合算子在目标逼它时 beat attention"落在这个 2×2 的未测格子里。

已实现**泄漏控制**(只保留双方都可见的 band 之间的耦合,凡触及被 mask 频带的项清零;crossfreq 下只剩 low→low,算子仍须自己学 low→high 路由),CPU smoke 验了泄漏为零。

**⚠️ 但结局:旗舰格子从没跑成。** coupling+crossfreq/random 在三个二分类(tusz/tuab/chbmit)上的 6 个 job **全部 CANCELLED**;只有两个多分类完成:

| 数据集 (kappa) | attn+random | attn+crossfreq | coup+random | coup+crossfreq |
|---|---|---|---|---|
| TUEV | 0.356 | 0.271 | 0.271 | 0.286 |
| Sleep-EDF | 0.483 | 0.451 | 0.468 | **0.493** |

Sleep 上 coup+crossfreq(0.493)是四格最优——**耦合×目标交互可能真实**的一个 hint,但落在 crossfreq 已经输的数据集、单 seed,不 load-bearing。**决定性测试(coupling+crossfreq 在三个二分类赢家上)仍未跑,是全项目最重要的未跑格子。**

---

## 7. ⭐ phase-steered 原语 + phase_align 目标(当前前沿)

这是**比 coherence 更硬的 novelty 候选**。

### (a) `FreqPhaseSteered` mixer(parameter-free)

用**复数** PAC 向量 `pac_vector[i,j] = mean_t A_j(t)·exp(i·φ_i(t))`(同时保幅度和偏好相位)。每个目标频带 j 只接收慢频 i<j 的消息,源 token 按实测 PAC 相位**旋转**后幅度归一聚合。**没有 QK、没有可学 scale、没有门、没有 value/output 投影**——跨频唯一通道就是实测相位-幅度几何本身。这是"先验即机制"的最锋利形式:coupling/coherence 还有可学路径(会被监督训练清零),phase-steered **除了真实 PAC 无法跨频**。

**监督结果 + 内建机制消融**(`magnitude`=抹掉偏好相位;`scramble`=打乱每条边相位):

| 数据集 (关键指标) | normal | magnitude | scramble | vs 其他最佳 mixer |
|---|---|---|---|---|
| TUAB (auroc) | **0.873** | 0.824 | 0.873 | 最高(coupling 0.872) |
| TUEV (kappa) | **0.413** | 0.178 | 0.332 | 最高(attention 0.376) |
| Sleep (kappa) | 0.516 | 0.040 | 0.507 | 中(cotar 0.572) |
| TUSZ (auroc/bacc) | 0.835/0.594 | 0.770/0.633 | 0.803/0.582 | auroc 追平 coupling;bacc 差 |
| CHB-MIT | **TIMEOUT(撞 12h 墙,无结果)** | — | — | — |

**phase 是第一个在监督下真正超过其他 mixer 的**(TUAB、TUEV 最高)——第 5(a) 节说这条路关闭了,phase 是个部分反例。**机制消融是分裂结论,必须诚实报**:抹掉相位(magnitude)持续甚至灾难性掉点(Sleep kappa 0.516→**0.040**;TUEV 0.413→0.178)→ 相位几何**确实 load-bearing**;但**打乱相位(scramble)几乎不掉**(TUAB/Sleep)——若精确相位重要,scramble 本该同样伤。当前读法:模型靠"有结构化相位旋转"多过靠"精确相位值",这是任何"学到真 PAC 相位"主张前必须搞清的 caveat。

### (b) `phase_align` 对比目标 —— 目前是坏的(已诊断)

对比 BCE:区分真实 PAC 几何(正)vs 幅度匹配的相位打乱(负)。

| 数据集 (关键) | phase_align | phase_random(对照) | crossfreq-MAE §5(b) |
|---|---|---|---|
| TUSZ (auroc) | 0.773 | 0.814 | **0.836** |
| Sleep (kappa) | 0.343 | 0.436 | **0.451** |

**phase_align 输给自己的对照、也输给第 5(b) 节的 crossfreq。诊断出根因**:`align_loss` 4 个 epoch 就塌到 **0.0008**——对比任务**平凡可分**(打乱相位的负样本太好认),encoder 学了个捷径判别器、表示不迁移;而 phase_random 的 recon_loss 正常收敛(15 轮→0.064)迁移更好。**→ phase_align 当前坏在"负样本太简单",需要更难的负样本或非对比形式,才能真正测相位假设。**

---

## 本周净收获

1. **关闭了一条路**:三次证明 PAC-as-a-layer 在监督下不 work,并用 `pac_scale→0` + 仍赢 0.09 的证据**定位到根因**(监督无动力学跨频)。
2. **打开了一条路**:把 PAC 变成预训练目标,crossfreq MAE **3/5 赢**过标准 MAE,并在 CHB-MIT 把崩塌的监督从 0.53 救到 0.88——**SSL 基石首个正面证据**。
3. **修了一个 bug 类缺陷**:耦合从"全局平均成糊"改成按 (通道, patch) 计算。
4. **提出更强的 novelty**:parameter-free phase-steered mixer,监督下 TUAB/TUEV 最优,机制消融证明相位 load-bearing。
5. **诚实的未决项**:phase_align 目标坏了(已诊断);2×2 旗舰格子(coupling+crossfreq 于二分类)未跑;CHB-MIT phase 超时;全部单 seed。

## 下周计划

- 重跑 coupling+crossfreq 于 TUAB/CHB-MIT/TUSZ(补旗舰格子)
- 修 phase_align 的负样本、重跑
- 补 CHB-MIT phase(更长墙 / 更少轮)
- 3 个 crossfreq 赢家做多 seed 确认显著性
