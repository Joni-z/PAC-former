# CFC频域EEG novelty
## Novelty / 文献调研:把跨频耦合做成频域 EEG encoder 的可微结构先验

### 一句话总结

* 你的具体想法——设计一个可微、端到端的模块,把"低频相位调制高频幅度"这一有方向的跨频耦合(CFC)层级直接焊进 attention / token 交互机制本身,作为 CoTAR 在频域的类比——看起来是真正 novel 的;没有任何现有论文同时具备它的三个定义性要素。 最接近的先例是 ACCNet(Neural Networks 2025),它确实在架构层面用了 CFC,但做成的是 graph-attention 网络里的可学习边,不是频域 transformer 里有方向的 attention 替代。
* 周边领域很拥挤,但拥挤的方向恰好和你错开。 频域 EEG encoder、spectral transformer、Fourier-as-attention 替代(FNet / GFNet / AFNO)、频域预测 transformer(FEDformer / FreTS / Fredformer / FreEformer)都已存在——但它们都把频率当成对称/全局的对象,没有一个编码了有方向、有层级的 CFC 先验。深度学习里的 PAC / CFC 绝大多数被当作预计算特征用,而不是架构层面的 inductive bias。
* 神经科学依据扎实、且有临床动机: 相位-幅度耦合是公认的、有方向的("低调制高")现象,而且在 AD 和癫痫里会被可靠地改变——所以这个 inductive bias 站得住。最干净的贡献定位是:把模块设计成一个有方向地组织 attention 的可微 PAC / modulation-index 算子,并与 ACCNet 和标准 self-attention 做正面对比。

### 核心发现

1. 没有完全撞车的先例。 在 arXiv、OpenReview、Semantic Scholar、PubMed/PMC 以及各期刊会议做了详尽检索后,没有论文同时满足:(a) 在 attention 计算内部实现可微的 PAC / modulation-index 算子;(b) 施加有方向、不对称的"低频相位→高频幅度"层级;(c) 用它替代频域 EEG encoder 里的全连接 self-attention。
2. ACCNet 是最大的威胁,必须正面处理。 它在架构上用了 CFC(在自适应频带节点之间学习 graph 边,明确强调低-高交互),但放在 GNN / graph-attention 框架里,是无向的,而且没有可微的 modulation-index 算子。
3. "CFC 当特征"是主流范式(在 PAC comodulogram 上跑复数 CNN、在 CFC 矩阵上跑 autoencoder)。这些不威胁 novelty,但说明领域已经认可 CFC 是有判别力的。
4. 频域 EEG transformer 和 foundation model 非常多,但它们对频率的使用集中在 tokenization / 重建目标(LaBraM、NeuroRVQ、TFM-Tokenizer)或对称的跨频带 attention——都不是有方向的 CFC。
5. Fourier-as-attention 架构(FNet、GFNet、AFNO、SpectFormer)用全局/对称的频域滤波替代 attention;没有一个编码跨频方向性。
6. 通用时序频域 transformer(FEDformer、FreTS、Fredformer、FITS、FourierGNN、FreEformer)在频域操作,但把频率当成一个扁平集合;Dualformer(2026)引入了按深度的频率层级,但是通用的、非 PAC 的。
7. 不存在可微的 FOOOF / specparam encoder——periodic / aperiodic(1/f)分解都是离线用的;把它做成端到端是一个相关的、同样开放的机会。
8. **【新增】"为什么没人把经典 MI(MVL)直接塞进 attention"本身值得正面回答,而不是绕过去。** 检索发现这不是单纯的"没人想到",而是有三个具体的技术门槛叠加,外加一个动机缺口——这构成了 novelty 论证里"敌人没来不是因为路被堵死,而是没人想破门"的那部分,需要在论文里主动讲清楚,否则审稿人会自己提出这个问题。详见下方"为什么这个组合此前没被做出来"一节。

---

## 详细内容

### 基线论文背景(TeCh / CoTAR)

你的工作建立在 "Decentralized Attention Fails Centralized Signals: Rethinking Transformers for Medical Time Series"(Guoqi Yu, Juncheng Wang, Chen Yang, Jing Qin, Angelica I. Aviles-Rivero, Shujun Wang)之上,ICLR 2026(Oral),arXiv:2602.18473,OpenReview oZJFY2BQt2;代码在 github.com/Levi-Ackman/TeCh(基于 Medformer)。核心论证:MedTS 信号是"中心化"的,而 self-attention 是"去中心化"的(全连接),二者结构失配。CoTAR(Core Token Aggregation-Redistribution)用一个全局 core token 替代 attention,先聚合再把信息重分配到各通道,带来线性复杂度。你的类比论证——self-attention 把频率当成一张全连接图,与有方向的 CFC 层级失配——是一个干净的结构平行,定位很好。

**【新增·定位澄清】TeCh 在本项目里扮演两个不同角色,要分开看,避免被误读为"在 TeCh 上改":**

1. **论证前身**——"去中心化 attention 配不上中心化信号"这个结构性论证,以及 aggregate→redistribute 这个范式,是我们要继承并迁移到频域的精神资源。这部分要引用、要对比。
2. **架构模板**——TeCh/CoTAR 本身是**时域**模型(基于 Medformer 的 patch 化骨架),其具体架构细节不应该被照搬进我们的频域 encoder。我们的骨架由频域 CFC 结构自身导出(可学习带通 → 解析信号 → band 当 token → MI 算子)。

这个区分让 pitch 更干净:"TeCh 证明了去中心化 attention 配不上中心化的**时域**医疗信号;我们把这个洞见搬到**频域**——这里的'中心化'就是跨频耦合的层级(低频统辖高频)。"TeCh 由此从"被抄的模板"变成"被延伸的跳板",novelty 的边界也更清楚:对比实验里 TeCh 应该是被对比的对象之一,而不是我们方法的母体。

---

### 领域 1 — CFC/PAC 作为结构/架构先验(最重要)

* **ACCNet**(Dongyuan Tian, Yucheng Wang, Peiliang Gong, Zhewen Xu, Zhenghua Chen, Xiaohui Wei, Min Wu),*Neural Networks* vol. 191, 107853, 2025;DOI 10.1016/j.neunet.2025.107853。两个模块:Adaptive Bands Decomposition(受试者特异的频带节点)和一个 Cross-Frequency Coupling 机制,从"节点-边视角学习个性化的频率关系,特别强调低频与高频成分之间的交互"。这是一个架构级的 CFC 机制(不是特征融合)——但它是在频带节点上做 graph attention、无向的边学习、而且不是替代 transformer self-attention 的可微 PAC 算子。这是最接近的 prior art;要在以下三点上明确区分:(GNN vs transformer)、(无向边 vs 有方向的相位→幅度调制)、(没有可微 modulation-index 算子)。
* CFC 当特征的先例(不威胁,但作为主流范式引用):在 SEEG 癫痫的 PAC comodulogram 上跑复数 CNN("Classifying epileptic phase-amplitude coupling in SEEG using complex-valued convolutional neural network," *Frontiers in Physiology*, 2022, DOI 10.3389/fphys.2022.1085530);在 CFC 矩阵上跑 autoencoder 做失神发作(*Frontiers in Neuroinformatics*, 2025);DB-GNN(arXiv:2504.20744, 2025)识别 CFC 脑网络。
* 把"物理"焊进 attention 的趋同工作,非 EEG / 非方向性: Holographic Transformer(arXiv:2509.19331, 2025)把相位干涉整合进 self-attention(同频内,不是跨频)。TransformEEG(*Applied Sciences* 15(24):13275, 2025)用 phase-swap 数据增强隐式学 PAC——没有 PAC 结构化的算子。

**【新增·补充检索】这一节额外确认的一点:Holographic Transformer 是目前唯一一个真的把"相位"显式塞进 self-attention 计算内部的工作,值得单独拎出来作为最近邻对比,而不只是归入"趋同工作"一笔带过。** 它把相位干涉做成一个离散干涉算子放进 attention,并专门设计了双头解码器来防止"相位塌缩"(loss 偏向幅度优化时相位信息被压没)。但它和我们的方案在三个维度都不同:(a) 同频内的相位干涉,不是跨频的相位→幅度调制;(b) 对称,没有方向性;(c) 非 EEG、非 PAC。它的"相位塌缩"问题也提示了我们自己模块设计时需要注意的一个工程风险(见下方 MI 算子设计部分的归一化讨论)。

---

### 领域 1.5 — 【新增小节】Modulation Index 设计空间:为什么选 MVL 形式、以及为什么这个组合此前没被做出来

这一节回答两个文献调研之外、但同样属于 novelty 论证的问题:(1) 经典 MI 不是一个公式而是一族,我们选哪个、为什么;(2) 如果 MVL 形式天然适合塞进 attention,为什么二十年里没人这么做。

#### (a) MI 的设计空间

PAC 的 modulation index 历史上有多个互相竞争的定义,每个本质上是在选不同的"距离 + 聚合"方式:

* **MVL(Mean Vector Length,Canolty 2006)**——Z = (1/T)Σ A_high(t)·e^{iφ_low(t)},取 |Z| 作为耦合强度。这是我们选用的形式,原因是它**天然可微,且结构上与 attention 同构**:复指数当权重、对幅度做加权求和,跟 softmax(QKᵀ)V 是同一个"用相位/相似度加权再聚合幅度/值"的范式。缺点是依赖高频振荡的绝对幅度大小,需要做 Ozkurt 归一化(除以 Σ|A|)或配合 LayerNorm 来避免模型学成"哪个频段功率大"而不是"哪对频段耦合强"。
* **Ozkurt 归一化 MVL**——在 MVL 上加一个归一化项,同样可微,建议直接整合进算子设计里。
* **Tort 的 KL-MI(2010)**——相位分 bin、算每 bin 平均幅度、归一化成分布后算 KL 散度对均匀分布的偏离。这是领域里最常用、最稳健的版本,有清晰的信息论解释,但**硬分桶这一步本身不可微**,要保留这个解释需要做 soft-binning(用 softmax 加权代替硬分桶)——这是一个可能值得写进方法、但非必需的扩展方向。
* **其它修正版**(dPAC、dMI、eMI、wMI、GLM-CFC)——为了解决有界性、抗噪、抗谐波、短数据等问题而提出,大多计算量偏重,不适合放进 attention 内循环,但可以作为"为什么我们选 MVL 形式而不是别的"的对比论据。

这个设计空间本身不构成新的 novelty 威胁(MI 的各种变体都是离线分析方法,没有一个是为可微/端到端场景设计的),但在方法部分主动交代"为什么是 MVL 不是 Tort-KL"会让审稿人觉得这是经过权衡的选择,而不是随手拿来一个公式。

#### (b) 为什么这个组合此前没被做出来——四个技术性理由

这一点之所以值得在论文里主动讨论,是因为 MVL 形式的"可微性"和"与 attention 的结构相似性"看起来都不是新发现——如果真的这么显然,为什么二十年里没人把它焊进 attention?检索后发现答案是几个具体的技术门槛叠加,而不是"此路不通":

1. **复数 softmax 本质上不成立。** 标准 attention 依赖 softmax(QKᵀ)V,而 softmax 要求实数输入;把 attention 搬到复数域时,若要用 softmax 计算相似度,要么退化为常函数,要么在复数域非解析（不可导）。这是复数域 attention 文献里反复出现的硬性障碍。**我们的绕法**:MVL 本身不需要 softmax,它天生是"复指数加权求和后取模",可以直接用 |Z| 作为耦合强度,不必勉强套用 softmax 范式——但这意味着论文里需要明确说明"这不是 softmax attention 的复数版本,而是另一种聚合机制",否则容易被误读为对复数 softmax 问题的回避而非绕开。
2. **MVL 把时间轴拍扁了——它本质是一个 readout,不是 mixer,这是最深的一层质疑,必须正面回应。** 标准 attention 输出的是"重新加权后的 token 序列"(token×token 混合),而 MVL 的 Z = (1/T)Σ A(t)·e^{iφ(t)} 是对整段时间积分成一个标量(复数)。comodulogram 那个 N×N 矩阵确实在形状上像 attention 矩阵,但它活在"频率对空间",不在 token 空间,且没有天然的"value"向下传递。这意味着"MVL 替代 attention"不是字面意义上的 drop-in,必须重新定义"聚合什么、怎么重分配"。**这正是 CoTAR 能提供的桥接结构**:CoTAR 的范式恰好就是"先聚合成 core token,再 redistribute",MVL 的时间积分步骤可以严格对应到 CoTAR 的"聚合"步骤,而不是被当作可疑的 pooling。具体映射关系见下方"MI 算子与 CoTAR 的同构对应"一节——这是回应"这只是个 pooling,凭什么叫 attention 替代"这类审稿质疑的核心论证,必须在方法部分讲清楚。
3. **穿过相位提取的梯度不稳定。** 瞬时相位是解析信号的辐角,即 arctan2(虚部, 实部),其梯度在振幅趋近 0 时发散(分母为 |z|²),相位本身在 2π 处还存在 wrap 不连续。**绕法**:全程停留在复数表示里,不显式调用 arg() / atan2,直接用 z_low/|z_low| 作为单位相位向量参与运算,只在 |z_low| 趋近 0 处做 clamp。这与项目早期遇到的 SincNet t=0 NaN bug 是同一类问题,解法也通用。
4. **缺乏动机,而非缺乏能力。** 在 CoTAR 提出"自注意力与中心化信号结构失配"这一论证之前,深度学习 + PAC 的主流做法(离线算 comodulogram → 喂 CNN/autoencoder)已经够用,没有人有压力把它做成端到端塞进 attention。这是一个"没人 bother 去做"的空白,而不是"试过但做不出来"的死路——这对 novelty 是好消息,但意味着①②③这三个技术门槛是真实存在的实现风险,需要在方法部分逐一交代解决方案,而不能假设它们是trivial的。

#### (c) MI 算子与 CoTAR 的同构对应——回答"这是不是只是个 pooling"

这一节是(b)中第 2 点的具体展开,是论文方法部分应当包含的核心论证链。

CoTAR 的两步是:① **聚合**——把各 token 的信息聚合进一个全局 core 向量;② **重分配**——把 core 向量拼回每个 token 再投影,实现 token 间的间接交互,带来线性复杂度。

MI 算子可以严格对应到这两步,而不是另起一套:

* **① 聚合,对应 Z = Σ A·e^{iφ} 的时间积分。** 区别在于 CoTAR 聚合出的是一个对称的 core 向量(所有通道一视同仁),我们聚合出的是一个**有方向的 N×N 耦合矩阵**(行 = 低频调制者,列 = 高频被调制者)。这个有向矩阵正是我们比 CoTAR 多出来的结构,也是"频域 + CFC"相对"时域 + 中心化信号"这一类比里真正新增的部分。
* **② 重分配,对应用 |Z| 作为权重、让低频相位去调制对应的高频 token。** CoTAR 第二步是把 core 向量拼回每个 token 再过 MLP;我们对应的做法是把耦合强度 |Z| 当作类似 attention 权重的角色,频率 token 作为被加权的 value,具体实现可以是拼接+MLP(与 CoTAR 最贴近、建议作为首个版本)、逐元素 gating,或把耦合矩阵当作 mixing 矩阵直接作用于 token——后两种可作为消融变体。

这个映射关系给出的论证是:我们不是"用一个 pooling 冒充 attention",而是"走了与 CoTAR 完全相同的 aggregate→redistribute 范式,只是把对称的 core token 换成了有方向的 CFC 耦合矩阵"。当被问"凭什么这能算 attention 替代"时,答案是"它与已被 ICLR 接收、被认可为 attention 替代的 CoTAR 同构"。

另外,方向性在这个构造里**不需要额外设计**,它直接来自"谁出相位、谁出幅度"的角色分配(低频供相位、高频供幅度)——这是对称 attention 结构性给不出的东西,也是相对 ACCNet(无向边学习)的核心区分点。

---

### 领域 2 — 频域 EEG encoder / spectral transformer(2023–2026)

这块很拥挤:AMDET(arXiv:2212.12134)有一个 spectral attention block 做对称跨频带 attention;Spectral Transformer(PSD→transformer);TFormer(时频 cross-attention);AFTA(MDPI *Brain Sciences* 15(4):382, 2025)——一个 Adaptive Frequency Filtering Module 与时域 attention 结合做自监督癫痫任务(TUSZ/TUAB/TUEV,AUROC 0.891);通过 SincNet 做可学习滤波器组 → Sinc-EEGNet(arXiv:2101.10846);FreqDGT(arXiv:2506.22807)频率自适应动态图 transformer。近期综述(Transformer-based EEG Decoding, arXiv:2507.02320;MDPI *Sensors* 25(5):1293, 2025)指出,大多数 EEG transformer 是转成时频或融合频谱特征,而不是深入有方向的频域结构——这给你的"有方向 CFC"角度留了空间。

### 领域 3 — EEG foundation model 与频率

LaBraM(ICLR 2024 spotlight)用 vector-quantized neural spectrum prediction——它的 tokenizer 重建 Fourier 的幅度和相位。BIOT(NeurIPS 2023)把生物信号 token 化成句子式格式。NeuroRVQ(OpenReview m38Hle9Utx)明确重建 Fourier 频谱幅度 A 和相位 φ(用 sin/cos 表示)。TFM-Tokenizer(arXiv:2502.16060)用 frequency-then-time 范式,有一个 Localized Spectral Window Encoder 把窗口切成频率 patch 来建模"跨频依赖"。CBraMod 用 criss-cross(空间/时间)attention;CodeBrain 用双域 tokenization;EEGPT(OpenReview lvS2b8CjG5)用时空 masked SSL;REVE 用 4D Fourier 位置编码。确认:它们的 tokenizer 高度依赖频域重建/PSD,但没有一个施加有方向的 CFC 交互先验——TFM-Tokenizer 的"跨频依赖"是对称的 patch 交互,不是有方向的相位→幅度调制。

### 领域 4 — Fourier/spectral 作为 attention 替代

FNet(用固定 DFT 做 token mixing)、GFNet(NeurIPS 2021 / T-PAMI;在频域学全局滤波器,O(L log L),替代 self-attention)、AFNO(分块通道混合 + 软阈值)、SpectFormer。它们都通过全局对称的频域操作放松了"所有 token 全连接"的假设,带 log-linear 复杂度——和你的复杂度/效率动机概念一致,但没有一个编码跨频方向性或"低调制高"的层级偏置。很少被用到 EEG 上;这本身就是个 gap。

### 领域 5 — 神经网络里的 periodic/aperiodic(1/f)分解

FOOOF/specparam(Donoghue et al. 2020)把频谱参数化为 aperiodic(offset、knee、exponent χ)+ periodic(高斯峰)。它绝大多数被当作离线分析步骤。搜 "differentiable FOOOF / aperiodic exponent neural network / specparam end-to-end" 找到的都是离线分析论文,以及一篇在预处理里用 FOOOF 的随机涨落建模预印本(arXiv:2505.19009)——但没有完全可微的端到端 FOOOF encoder。这是你可以和 CFC 先验组合的、相关且同样开放的贡献。

### 领域 6 — 通用时序频域 transformer

FEDformer(频率增强分解,随机 Fourier 模式选择);FreTS(NeurIPS 2023,频域 MLP,"global view" + "energy compaction");Fredformer(KDD 2024,给频带去偏,让模型不要过度关注高能量/低频成分);FITS(低通 + 复数线性);FourierGNN(Fourier Graph Operator);FreEformer(arXiv:2501.13989,在 attention 上加可学习矩阵修复 low-rank 问题);JTFT(联合时频)。没有一个引入跨频耦合或有方向的频率层级结构;Fredformer 的"frequency bias"是概念上最近的表亲(它关心 attention 怎么给频率加权),但它是均衡而非施加有方向的层级。Dualformer(arXiv:2601.15669, 2026)把高频成分分给浅层、低频分给深层——一个频率层级的架构先验,但通用、非 EEG、非 PAC。

### 领域 7 — 神经科学依据(简要)

* 真实、有方向、有层级: PAC——高频幅度被低频相位调制——是经典且在人类身上有直接证据的。Canolty et al.(*Science* 2006, DOI 10.1126/science.1128115)用人类 ECoG 记录,显示低频 theta(4–8 Hz)节律的相位调制 high gamma(80–150 Hz)band 的功率,且 theta 幅度越大调制越强(最大耦合在 ~146.2 Hz 幅度、~5.6 Hz 相位)。海马 theta 相位→gamma 幅度的关系(Tort, Buzsáki)是教科书级实例。方向性/层级性由 Voytek et al.(*Frontiers in Human Neuroscience* 2010, PMC2972699)支持:低频振荡可能协调脑区间的远程通信,而高频 high gamma 活动空间上更局限——即慢节律设定时间框架,快的局部活动嵌套其中。计算模型(neural mass / cortical column, *PLoS Comput Biol* 2016;振荡器网络模型)确认了有方向的生成机制。
* 疾病中被改变(临床动机): 在 AD 中,Prabhu et al.(*Brain Communications* 2024;6(2):fcae121)发现 AD 患者(n=50;年龄 60±8)相比认知正常对照(n=35;年龄 63±5.8)"表现出 theta-gamma PAC 降低,gamma 幅度在 6–8 Hz 振荡范围内的耦合减弱",位于左侧海马旁皮层(gamma 幅度 30–40 Hz 耦合到 theta 4–8 Hz / alpha 8–12 Hz 相位,MEG)。在癫痫中,PAC(低频相位到 HFO 幅度)是公认的发作起始区(SOZ)生物标志:Cui et al.(*Cognitive Neurodynamics* 2023, DOI 10.1007/s11571-022-09915-x)在 20 秒发作间期 ECoG 上,用 mean-vector-length modulation index 在低频节律(0.5–24 Hz)与 HFO(80–560 Hz)之间做 SOZ 定位;Motoi et al.(medRxiv 2020.11.07.20226258)报告 infraslow-HFA PAC 区分 SOZ 的 "AUC 为 0.926",且在发作起始前约 87 秒开始上升。这说明你想编码的这个 bias,确实跟踪着你要分类的临床状态。

---

## 总体 novelty 评估

你提出的贡献——一个有方向地组织(或替代)频域 EEG encoder 中 self-attention 的可微 PAC / modulation-index 算子,编码不对称的"低频相位调制高频幅度"层级——据详尽检索,是 novel 的。它的三个定义性要素(可微 PAC 算子;有方向/不对称层级;transformer 中的 attention 替代)各自单独出现过、或以邻近形式出现过,但从未被组合:

* attention 内的可微 PAC 算子: 没找到。
* 把"低→高"频率层级作为架构偏置: 只找到通用且非 PAC 的(Dualformer,按深度),或无向的(ACCNet 的 graph 边、AMDET 的跨频带 attention)。
* 通过频谱结构替代 attention: 找到的都是对称/全局的(GFNet/AFNO/FNet),从无 CFC 方向性。

最大威胁是 ACCNet;次级趋同想法是 Holographic Transformer(相位入 attention,非 EEG,同频内)和 Dualformer(按深度的频率层级,非 PAC)。没有一个抢占了完整想法。

**【新增·收尾】"为什么此前没人做"这一问题本身已经得到正面回答(见领域 1.5):答案是复数 softmax 不成立、MVL 的时间积分与 mixer 语义之间存在需要桥接的鸿沟、相位梯度不稳定这三个技术门槛叠加,外加 CoTAR 式论证出现之前缺乏明确动机——而不是这个方向已被尝试过并被放弃。三个技术门槛各有具体绕法(不用 softmax、借 CoTAR 的 aggregate-redistribute 范式搭桥、全程留在复数表示里不显式取相位角),这部分应该在方法论部分主动写出来,把它从"潜在的审稿质疑"转成"我们已经想清楚的设计权衡",这本身也是论文论证力的一部分。