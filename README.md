# PAC-Former

强制、规范不变的相位-幅值耦合(PAC)tokenization 算子 + 三轴 EEG Transformer。

**核心主张**:PAC 不作为 attention 的修饰,而是**定义 token 本身**。高频 token 旁边不保留
任何原始波形通路,先验无法被优化绕过。

```
h_j = a_j ⊙ Σ_{i<j} α_ij · exp(−i∠Z_ij) · p_i        (j > 0)
h_0 = a_0 ⊙ p_0                                       (根节点)

a_j   目标频带幅值特征        α_ij  归一化耦合强度 |Z|/Σ|Z|
p_i   慢频带相位特征          ∠Z_ij 测量得到的优选相位
```

## 主要结果(TUEV kappa,监督从零训练,seed 0)

| | 1.64M | 8.6M |
|---|---|---|
| raw token(逐频带波形卷积) | 0.4679 | 0.4498 |
| **PAC interaction token** | **0.5493** | **0.5379** |
| 差 | **+0.0814** | **+0.0881** |

同 pipeline、同划分、全部从零训练的对照:BIOT-scratch 0.4449、CBraMod-scratch 0.5017。
参数量仅 CBraMod 的 41%。

跨数据集:Sleep-EDF +0.052、TUSZ 打平、CHB-MIT(预训练下)见 AGENT.md §13.43-J。

## 四个匹配对照(全部参数量相同,TUEV)

| arm | 拆掉什么 | kappa |
|---|---|---|
| raw | 整个交互 | 0.4679 |
| `uniform` | 耦合强度 + 相位对齐 | 0.4734(打平) |
| `magnitude` | 只拆相位对齐 | 0.4826(打平) |
| `concat` | 只拆"强制乘积"(改为拼接) | 0.4838(打平) |
| `measured` | 无 | **0.5493** |

三个对照全部打平基线 ⇒ 排除"赢在乘法结构""赢在暴露相位特征""赢在强度加权"三种解释。
收益归因:优选相位对齐 +0.0667,强度加权 +0.0092(噪声)。

## 仓库结构

```
models/           模型(frontend/triaxial.py 是 PAC tokenizer)
configs/          当前实验配置
configs/_closed/  已封闭实验线,仅供复现 AGENT.md 中的负结果
scripts/          数据预处理与工具
scratchpad/       CPU 验证脚本(提交 GPU 作业前的正确性检查)
literature/       28 篇论文库 + 精读笔记
iclr2026/         论文
```

## 文档

| 文件 | 内容 |
|---|---|
| **`AGENT.md`** | 实验记录(时间序,含所有负结果与撤回)。**唯一的事实来源** |
| `BIG_CLUSTER_HANDOFF.md` | 大集群预训练交接:冻结的模型/目标函数/证据顺序 |
| `PRETRAIN_DATA_PLAN.md` | 预训练语料清单(18 语料/4,953 小时)与入池规则 |

## 环境

```bash
conda activate pacformer
sbatch train.slurm <config_name>      # 监督训练
sbatch pretrain.slurm <config_name>   # 预训练 + 微调
```
