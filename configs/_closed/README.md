# 已封闭实验线的配置(仅供复现 AGENT.md 中的负结果)

不要用于新实验。这些配置对应已被证据关闭的方向：

| 前缀 | 实验线 | 关闭依据 |
|---|---|---|
| `*_v2_*` | v2 频率轴 mixer(coupling / coherence / cotar / phase) | §13.20、§13.10a(coherence 门从未打开，结论作废) |
| `cand_*` | 监督候选结构搜索(wide / deep / nb16 / coupling) | §13.40-A：无一超过 base |
| `*_mi_product*`、`*_mi_topk*` | MI 矩阵调制 attention | §13.41：0 胜，随机对照打平 |
| `synthetic_*` | v1 合成信号验证 | v1 架构已废弃 |

当前架构见 `configs/pacint_*.yaml`(PAC interaction tokenizer)。

| `pretrain_*`、`sub_*`(无 `tokenizer_mode: pac_interaction`) | raw-token backbone 上的目标函数消融 | 架构已换为 PAC tokenizer(§13.43)；目标函数结论见 §13.45 |
