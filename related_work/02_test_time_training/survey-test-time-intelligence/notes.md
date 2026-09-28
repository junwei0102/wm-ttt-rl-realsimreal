# A Survey on Self-Improving Test-Time Intelligence: Feedback-Driven Adapting, Learning, and Scaling at Inference

> arXiv [2609.01679](https://arxiv.org/abs/2609.01679) · Machine Intelligence Research, 2026（已录用：arXiv 注释为 "Accepted by Machine Intelligence Research"，期刊 DOI 10.1007/s11633-026-1694-1 暂未生效，卷期和页码未知） · 主题：02_test_time_training · 层级：survey
> 代码：none（综述，无代码）。配套论文列表 https://github.com/mr-eggplant/awesome_test_time_intelligence 已确认存在，是纯 paper list，没有代码和 benchmark 实现。据该 repo README，列表按 TTL / TTS / Intersection / Applications 共收录 489 篇，这个数字来自 repo，不来自论文。· 笔记依据：`source/arxiv.tex`（数字/公式，单文件）+ `paper.pdf`（图）

覆盖范围：PDF 共 76 页，正文到 p.53，`arxiv_references.bib` 有 512 条文献（另有 36 个 `@string` 宏，`grep -c '^@'` 得 548）。`arxiv.tex` 全文读完（Sec.1–8）。精读 Sec.2（概念框架）、Sec.3（TTL）、Sec.5（learning × scaling 交叉）、Sec.6.4（Embodied）、Sec.7（挑战）；Sec.4（TTS）读了分类和理论小节；Sec.6.1–6.3、6.5–6.6 只略读。公式编号按 PDF。作者团队以华南理工谭明奎组为主（共同一作 Shuaicheng Niu、Guohao Chen；通讯作者 Mingkui Tan、Cheng Deng）。EATA / SAR / FOA / ZOA / ZeroSiam / TLM 的 bib 作者栏里都有本综述作者（Niu、G. Chen 或 Tan 等）。综述的 TTA 部分写得最细，笔者推测和这一点有关。

## 一句话

提出 **feedback-driven Test-Time Intelligence (TTI)** 作为总框架，用两根轴给所有"部署时变好"的方法定位：一根是 **Update**，看是否用 test-time feedback 改写可写状态 $\phi$；另一根是 **Compute**，看推理时花了多少额外算力。在这张图上，TTA 是 TTL 的子集，TTS 是另一条路线，两者的交叉区是 learning-to-scale 和 test-time RL 这类混合系统。

## 问题设定

- **动机**（Sec.1）：train-then-infer 假设"冻结模型能应付所有测试场景"，面对分布偏移、用户差异、难度差异，这个假设越来越站不住（原文 "less tenable"）。TTA 社区（改参数）和 LLM 的 TTS 社区（加算力）术语各自为政，彼此的联系看不出来。
- **训练端**（Sec.3.1）： $\theta^{\ast}=\arg\min_\theta \mathcal L_{tr}(\theta;\mathcal D_{tr})$，其中 $\mathcal D_{tr}\sim p_{\mathcal S}(x,y)$。
- **Definition 1（TTL，Eq.1）**：测试输入 $\mathcal D_{te}=\lbrace x_j\rbrace$，外部反馈 $\mathcal F_{te}$ 可以为空。
  $$\phi^{+}=\operatorname{Update}(\phi;\mathcal D_{te},\mathcal F_{te})$$
  $\phi$ 可以是输入或 latent、参数或统计量、辅助模块、外部 memory；Update 可以是梯度优化、闭式或 running-statistics 更新、memory 写入，也可以是输入精修。场景同时覆盖 $p_{\mathcal S}\neq p_{\mathcal T}$（分布偏移）和 $p_{\mathcal S}=p_{\mathcal T}$（记忆和自我改进）。
- **TTS（Eq.7–8）**：标准推理是 $\hat y_{std}=\mathcal I_0(f_\theta,x)$，TTS 把它推广成 $\hat y_{tts}=\mathcal F(f_\theta,x;\mathbf c)$。 $\mathbf c$ 是预算配置（采样数、推理步数、搜索深度、验证轮数、工具调用次数），取 $\mathbf c=\mathbf c_0$ 时退化为标准推理， $\theta$ 始终不变。
- **两条边界判据**（Sec.2.1，最值得记）：
  1. *TTI vs training-time amortization*：部署时的信号如果只在之后的离线 fine-tune / 蒸馏里才起作用，就不算 TTI。
  2. *TTL vs 普通推理态演化*：KV cache、序列模型的 hidden state、预定 denoising 轨迹上的 diffusion latent 都**不算** TTL。只有被 test-time feedback **显式优化或写入**的 designated writable state 才算。

## 核心方法

这篇综述没有提出方法，这一节整理它的分类框架。

### 0. 顶层：Update × Compute 平面（Fig.3，Sec.2.2）

纵轴是 Model (State) Update（No change / State change），横轴是 Inference-Time Compute（Low / High）。经典 TTL 落在左上（改状态，额外算力少），TTA 是其中的一个椭圆；TTS 落在右下（算力多，不改状态）；阴影交叉区是 Sec.5 的 learning to scale 和 joint 系统。Fig.2 的时间线分三期：2020–21 Emergence、2022–23 Expansion、2024–26 Diversification。时间线分四色泳道：TTA（restore source performance）、Beyond TTA（build persistent capability）、TTS、Joint TTL×TTS（compute-derived feedback updates active state）。

### 1. TTL 的四个设计维度（Sec.3.1）

同一个方法可以在四个维度上各有一个坐标，这是一种 multi-view 分类。

**(a) Feedback signal**（Table 1，作者认为这是"最关键的设计决策"）

| 家族 | 代表 | 前提 | 典型失败 |
|---|---|---|---|
| Entropy & confidence | Tent, EATA, SAR, DeYO, COME | 源模型在目标域上仍然大致 calibrated | confirmation bias、class collapse、对 label shift 敏感 |
| Consistency & augmentation | MEMO, CoTTA, RMT, SPA, ActMAD | 增广保持语义，或 teacher–student 一致性有意义 | 增广无效、teacher 漂移、trivial agreement |
| Reconstruction & self-supervision | TTT(rotation), TTT-MAE, TTT++, TTTFlow | 训练期已经准备好辅助头或 decoder | pretext 与主任务不匹配 |
| Pseudo-label & self-training | T3A, AdaContrast, conjugate PL | 初始预测较可靠 | 噪声标签自我强化、prototype 污染 |
| External (env / tool / user) | PAD, FeedTTA, RLCF, Reward-Adaptation, SimATTA, COPR（正文 Sec.3.2.5 另举 TT-VLA、T³RL） | 有接口、延迟预算或人工成本可承受 | reward hacking、稀疏或延迟反馈、unsafe exploration |

**(b) Update target $\phi$**（Table 2）分五类：Input（diffusion restoration DDA、频域 TF-Cal）、Partial params（BN 统计量 AdaBN/Tent、层或通道选择、In-Place TTT 把 MLP block 的最终投影矩阵当 fast weights 原地更新）、Auxiliary params（prompt TPT/FOA、adapter/LoRA）、Backbone/Full（TTT、SHOT；MEMO、CoTTA）、External memory（prototype T3A、instance cache TDA/AdaNPC、semantic memory MemoryBank/PhysMem）。作者给的原则是："prefer the smallest update scope that captures the expected shift"；反馈质量差或没有回滚机制时，不要做持久的大容量更新。

**(c) Temporal horizon**（Sec.3.4，Fig.4）分两根**独立**的轴：
- *Supervision horizon*：一次更新用多少证据，单样本（MEMO、TPT、TTT-Layer 的逐 token 更新）或多样本（Tent 的 batch、RoTTA 的 history、TT-VLA 的交互轨迹）。
- *Persistence horizon*：更新后的状态保留多久，episodic（每个样本或每个域结束后 reset）或 continual（跨整条部署流保留）。例如 Tent 按 batch 更新，但是否 reset 取决于部署协议。

**(d) Objective scope**：一类是适应分布偏移（Sec.3.5）；另一类是 Beyond adaptation（Sec.3.6），包括 evolving memory、recursive self-improvement、personalization、环境或任务 specialization、用户交互反馈。

### 2. 适应分布偏移的机制（Sec.3.5）

- **偏移类型**：covariate shift 指 $p_{\mathcal S}(x)\ne p_{\mathcal T}(x)$ 且 $p(y|x)$ 不变（Def.2）；label shift 指 $p(y)$ 变而 $p(x|y)$ 不变（Def.3）；现实中常是两者复合。对齐目标（Eq.2）：
  $$\phi^{\ast}=\arg\min_\phi \mathcal D\big(q^{(\phi)}_ {\mathcal T}(z)\thinspace\Vert\thinspace q_{\mathcal S}(z)\big)+\lambda\mathcal R(\phi)$$
- **TTT vs FTTA**（Fig.5，Sec.3.5.2）。TTT 在训练期让共享 encoder $g$ 同时服务两个头：监督头 $\pi_C$ 和自监督头 $\pi_S$。测试时先用 $\mathcal L_{ttl}(\pi_S(g(x)))$ 更新 $g$，再用 $\pi_C$ 预测。下面的式子是笔者根据 Sec.3.5.2 的文字和 Fig.5 写的，不是论文编号公式（原文只说 "updates the shared encoder"，没写成梯度步）： $g^{+}=g-\eta\nabla_g\mathcal L_{ttl}$， $\hat y=\pi_C(g^{+}(x))$。FTTA 不改训练过程，典型是 Tent 的熵最小化（Eq.3） $\mathcal H(\hat y)=-\sum_c\hat y_c\log\hat y_c$，默认假设"置信 ≈ 正确"。取舍：TTT 需要源端准备，proxy 与任务对齐得更好；FTTA 即插即用，但更容易 collapse。
- **协议三维**（Sec.3.5.3）：source-dependent / source-light / source-free；offline / online；single-domain / continual。continual TTA 的两大难题是 catastrophic forgetting 和 error accumulation，对策分三类：可靠目标（样本筛选、校准）、稳定更新（Fisher 正则、teacher-anchoring restoration 即让预测或参数向稳定的 anchor 模型对齐、梯度过滤）、多模型解耦。
- **Forward-only**（Sec.3.5.4）。FOA 用 CMA-ES 搜 prompt，把解空间"从百万维降到几千维"： $\mathbf p_k^{(t)}\sim\mathbf m^{(t)}+\tau^{(t)}\mathcal N(0,\Sigma^{(t)})$（Eq.4）。ZOA 用 SPSA， $\hat g(\theta)=\frac{\mathcal L(x;\theta+c\epsilon)-\mathcal L(x;\theta)}{c}\epsilon^{-1}$（Eq.5），每个样本只需两次前向。另一支是 optimization-free：BN 统计量或特征均值校准、FFT 或 diffusion 输入净化、LAME 输出调整。
- **TTL 作为记忆**（Sec.3.6.1）：TTT-Layer 把 hidden state 当作模型，逐 token 做一步自监督梯度， $W_t=W_{t-1}-\eta\nabla_W\ell_{ssl}(x_t;W_{t-1})$（Eq.6），序列复杂度是 $\mathcal O(n)$。后续工作沿四个方向展开：更新规则（TTT-E2E、In-Place TTT）、容量管理（Titans、ATLAS）、plug-and-play memory、模态扩展（RoboTTT 用于机器人长 history）。

### 3. TTS 分类（Sec.4，Fig.6，Table 3/4）

- **内部算力**：iterative reasoning（CoT、Self-Refine、recurrent depth）、Best-of-N、self-consistency、search & planning（ToT、RAP 用 LM 当 world model 跑 MCTS、LATS）、adaptive computation（compute-optimal、s1 budget forcing）。
- **外部资源**：retrieval、tool use、verifier（PRM、GenRM）、multi-agent。
- **纯 scaling 的三个局限**（Table 5）：成本高；默认不会带来持久改进；收益递减甚至变得不稳定。

### 4. Learning × Scaling 交叉（Sec.5，Fig.8）

| 关系 | 子类 | 代表 | 在不在 TTI 定义内 |
|---|---|---|---|
| Scaling for learning | consensus 目标、verifier 反馈、search 蒸馏 | STaR, ReST^EM, Math-Shepherd, ReST-MCTS*, BOND；前身 Expert Iteration / AlphaGo Zero | **不在**（属于 training-time amortization，只做简述） |
| Learning to scale | 预算分配、策略路由、学习 scaling 行为 | SelfBudgeter, AdaptThink, Adaptive-RAG, RouteLLM, SCoRe | 控制器固定时算 TTS，在线更新控制器时算 TTL |
| Joint learning-and-scaling | test-time RL；self-synthesized supervision | TTRL, TTRV, T³RL, TTT-Discover, Alpha-RTL；SEAL, TT-SI, TTSR | 在 |

Test-time RL 按 reward 的来源分三类。① consensus：TTRL 用多数投票当伪标签，按笔者的写法可以记成 $r_i=\mathbb 1[\hat y_i=\mathrm{maj}(\hat y_{1:N})]$。② verified / stabilized：T³RL 引入工具验证，DARE、SCOPE、DDRL 做稳定化。③ task-grounded：TTT-Discover、Alpha-RTL 用可执行评测器。作者的结论是，诱导出的 reward 与正确性保持相关时，test-time RL 最可靠（"most reliable"）。

### 5. Embodied（Sec.6.4，Fig.11）

TTI 可以介入 perception→planning→action→environment 闭环的每一环。
1. *Policy adaptation*：可以只改感知或表征（AdaptFly、T4P），可以只改 planner 或 controller（BRIC、Centaur），也可以直接改 policy（PAD 继续优化 inverse-dynamics 目标；TT-VLA 把逐步任务进度当 dense reward 做 test-time RL）。
2. *Online skill improvement*：Voyager、LRLL 扩充技能库；Hong et al. 从失败试验中同时更新 reflection 模型和 policy。
3. *Planning-time scaling*：RoboMonkey 做采样加验证，RoVer 用 robot PRM，CoVer-VLA 做分层验证。

另外，Sec.3.2.5 把环境交互信号分成两类。一类是自监督信号：PAD、AugWM、MoVie（latent dynamics modeling），以及 AdaJEPA（执行一个 action chunk，用观测到的下一状态转移来适应 latent world model）。另一类是 outcome 信号：FeedTTA、TT-VLA。Sec.3.6.4 还列了 world model specialization（Worldagen、AdaPower）和 TTT-Parkour。

## 关键实验与证据

综述**没有自己的实验，也没有数值 meta-analysis**。Table 1–5 都是定性表（Table 3/4 的标题就是 "decision-oriented comparison"）：Table 1–4 列资源需求、适用场景和失败模式，Table 5 列纯 scaling 局限的成因、部署后果和代表证据。文中转述的实证或理论结论如下：
- **显存**：在 ViT-Base 上，Tent 需要 5,165 MB，纯推理只要 832 MB（Sec.3.5.4，引自 FOA）。这是 forward-only 方法的动机。
- **TTA 理论**（Sec.3.5.5）：原始 TTT 给出局部充分条件，辅助 loss 与主 loss 在共享参数上梯度对齐时，一小步自监督更新会降低主任务 loss。Entropy Enigma 指出，长时间做熵最小化会先涨后跌。PeTTA 把长期 collapse 和伪标签误差、类别结构、更新率联系起来。Protected TTA 和 risk monitoring 用来控制是否、何时继续适应：前者对其 entropy-shift detector 中的 online betting 更新给出 regret 保证，后者用 confidence sequences 控制随时间累积的误报。
- **TTS 理论和实证**（Sec.4.2, 4.2.6）：Large Language Monkeys 发现至少一个样本做对的比例随采样预算近似 log-linear 增长。Wu et al. 发现推理长度与准确率呈倒 U 关系。Chen et al. 发现多数投票随调用次数先升后降。RPC 分析 self-consistency：结合内部概率后，估计误差的收敛从 linear 变为 exponential。Huang et al. 指出 reward model 不完美时，增大 N 会加剧 reward hacking。Snell et al. 发现容易题适合顺序修正、难题适合更宽的搜索，而且在匹配算力下小模型有时能胜过大模型。结论是 **TTS 的收益是有条件的，不是单调的**。
- **作者提到的 benchmark / 数据集**：ImageNet-C 式合成 corruption（作者认为过窄）、WILDS、ACDC、VisDA（Sec.1 的偏移场景）；Evo-Memory（LLM agent 流式 TTL，发现多数 agent 缺乏动态 memory 管理）；Alfarra et al. 的计算时间受限评测协议（Sec.7.1 把它当作 matched test-time budget 评测的例子，并指出这类标准化评测仍然缺乏）。

## 局限与论文没说的

**作者承认的**（Sec.7.1）：开放世界中的长时程稳定性（collapse）；通用的学习和评估目标（例如 Tent 迁到分割效果不好）；效率离"接近单次前向"还很远；benchmark 过窄，评测没有同时覆盖准确率、延迟、显存、校准和稳定性；超参依赖离线调参，削弱了"自适应"的说法；理论只覆盖孤立机制。Table 7 的七个未来方向里，五个的可行性标为 High，Scientific Discovery 和 Deep Research 标为 Medium。

**我读出来的**：
- **Compute 轴只是示意**，没有统一的算力度量。Fig.3 里 TTL / TTA / TTS 各区域的位置是定性摆放，不能用来比较效率。
- **定义排除了机器人里最主流的适应方式**。按"普通推理态演化不算 TTL"这条判据，RMA 式的 history encoder / 隐式 system identification、RSSM 的 filtering、in-context adaptation 都被排除，全文也没讨论 sim-to-real / domain randomization（在 tex 里 grep 不到）。对本项目来说，这意味着"部署时适应"的另一半文献要去 real2sim2real 线补。上面几个例子是笔者按其定义做的归类，论文没有讨论。
- **Model-based RL 的 MPC / planning 没有被放进 TTS**。TD-MPC2 的 MPPI、Dreamer 的 imagination 本质上就是 compute 轴上的方法，但 Sec.4.4.3 只用一句话提到"world model 预测未来状态、支持 replanning"。其他涉及 world model 的地方是 LLM / 多模态语境：RAP 用 LM world model 评估 MCTS 分支（Sec.4.2.4），MindJourney 做 world-model-based test-time scaling（Sec.6.3.1）。全文没有出现 MPC。
- **Embodied 部分只有约 1 页**（p.46–48，含 Fig.11）。Sec.6.4 的 15 篇引用里有 10 篇是 2025–26 年的工作，其中 5 篇是 arXiv 预印本。Sec.6.4 开头提到 embodied 反馈 "often safety-critical"，但真机上持久更新带来的安全问题（不可回滚的物理后果）没有专门讨论。回滚只在两处泛泛提到：Sec.3.3 讲 update target（backbone / full-parameter 更新需要 rollback controls），Sec.3.6.2 讲 recursive self-improvement（需要 version isolation 和 rollback）。两处都没有涉及物理机器人。
- **作者组的工作占比偏高**（EATA、SAR、SAR²、FOA、ZOA、EVA-0、SPA、ZeroSiam、TLM、CoLA），TTA 细分类的粒度明显细于 TTS 和 embodied。
- **项目里几篇 TTT 架构线论文没有被引用**：ARC TTT、LaCT、One-Minute Video、KV-binding 争议文都不在 bib 里（按标题 grep 过）。关于"TTT layer 是否等价于 linear attention"的争论也完全缺席。

## 在脉络中的位置

- **前驱**：TTA 综述（Liang et al. IJCV 2025 等）、TTS 综述；奠基工作是 [TTT](../ttt-self-supervised/notes.md) 和 [Tent](../tent/notes.md)。
- **本项目里被它引用并定位的**：
  - [TTT layers](../ttt-layers/notes.md)、[Titans](../titans/notes.md)、[TTT-E2E](../ttt-e2e/notes.md)：归在 TTL for context memorization。
  - [TTRL](../ttrl/notes.md)、[TTT-Discover](../ttt-discover/notes.md)：归在 joint / test-time RL。
  - [PAD](../../05_intersections/pad/notes.md)：归在 environment-interaction feedback 和 direct policy adaptation。
- **本项目有、但它没覆盖的 TTT 线**：[ARC TTT](../ttt-arc-fewshot/notes.md)、[LaCT](../lact/notes.md)、[One-Minute Video](../ttt-one-minute-video/notes.md)、[KV-binding](../ttt-kv-linear-attention/notes.md)。读这几篇时可以用本文的 Update target / persistence horizon 两个维度去归类。
- **与本项目其他主题的联系**：
  - *World model*：AdaJEPA 和 Worldagen / AdaPower 说明 world model 本身可以是 TTL 的 $\phi$，可对照 [V-JEPA 2](../../01_world_model/vjepa2/notes.md) 的 latent planning。MPC / imagination（[TD-MPC2](../../01_world_model/tdmpc2/notes.md)、[DreamerV3](../../01_world_model/dreamerv3/notes.md)）对应 Compute 轴，这是笔者的映射。
  - *Real2sim2real*：sim-to-real gap 可以看成 covariate shift 加动力学偏移。[SimDist](../../05_intersections/simdist/notes.md) 属于"仿真预训练 + 部署交互更新"的 TTL；[ASAP](../../03_real2sim2real/asap/notes.md) 用真机数据离线训练 delta action，按本文定义属于 training-time amortization；[RMA](../../03_real2sim2real/rma/notes.md) 属于被排除的"普通推理态"。三者的归类都是笔者按本文判据做的。
  - *RL*（笔者的映射，综述没有讨论具体的 RL 算法，全文没有出现 PPO / GRPO）：[TTRL](../ttrl/notes.md) 直接沿用 GRPO（[DeepSeek-R1](../../04_rl/deepseek-r1/notes.md)），也验证了 [PPO](../../04_rl/ppo/notes.md)，新东西主要是 reward 由测试流上的 consensus 构造。但 test-time RL 不都只是换 reward：[TTT-Discover](../ttt-discover/notes.md) 把训练目标换成偏向最大值的 entropic objective。TT-VLA 可以和 [SimpleVLA-RL](../../04_rl/simplevla-rl/notes.md) 对照：前者是部署时 RL，后者是训练时 RL。[World-Gymnast](../../05_intersections/world-gymnast/notes.md) 属于 WM × RL × TTT 的交汇。

## 前置知识

1. Distribution shift 的类型（covariate / label shift）和 BatchNorm 统计量。
2. Entropy minimization 与伪标签（Tent、conjugate PL），以及 collapse 为什么会发生。
3. 自监督辅助任务（rotation、MAE）和"梯度对齐"直觉（TTT）。
4. Fast weights / 线性 RNN 视角（TTT-Layer）。
5. LLM 推理侧：CoT、Best-of-N、self-consistency、PRM / verifier、MCTS。
6. Policy gradient / GRPO（理解 test-time RL）；零阶优化（CMA-ES、SPSA）。

## 复现可行性

- 综述本身没有代码。上面的 paper list 可以当索引用。
- **作者推荐的评测方向**是多维、预算匹配的评测：同时报告 accuracy、latency、memory、compute overhead、calibration、stability，并在匹配的 test-time 预算下比较（Sec.7.1 举 Alfarra et al. 的计算时间受限协议为例）。Sec.1 引用的偏移场景有 ImageNet-C、WILDS、ACDC，其中合成 corruption 被 Sec.7.1 批评为过窄；agent memory 有 Evo-Memory。算力需求论文没有给出。
- **面向本项目的最小实验**（笔者建议，不来自论文）：在 MuJoCo 或 DMControl 的一个任务上训练一个 latent world model 加 MPC 的 agent，测试时改变质量或摩擦，制造动力学偏移。然后在 Update × Compute 平面上比较四种设置：
  - (a) 冻结基线；
  - (b) 只加 compute：增大 MPC 采样数或 horizon；
  - (c) 只加 update：用在线转移预测误差更新 dynamics（AdaJEPA / PAD 式）；
  - (d) (b)+(c) 同时用。

  另外对 episodic 和 continual 两种 persistence 分别画"return–额外算力"曲线，并画长流上的稳定性曲线。单卡可做。难点在于：怎么定义可比的 compute 预算；continual 设置下怎么检测 collapse、何时回滚。

## 综述强调、但本项目未收录的重要工作

1. **EATA**（ICML 2022）：过滤不可靠和冗余的样本，加 Fisher 正则抗遗忘。综述在 continual TTA 的 sample selection 和 Fisher-informed regularization 两类对策里都引用它。
2. **SAR**（ICLR 2023）：在 wild 场景（小 batch、混合偏移、label shift）下，把基于熵的样本筛选和 sharpness-aware 更新结合起来做熵最小化，分析了 collapse。
3. **CoTTA**（CVPR 2022）：continual TTA 的代表方法，用增广视图上平均的 teacher 预测当伪目标，结合权重平均和 stochastic restoration。
4. **Snell et al., Scaling LLM Test-Time Compute Optimally**（ICLR 2025）：compute-optimal TTS，说明策略选择取决于题目难度。
5. **SEAL, Self-Adapting Language Models**（NeurIPS 2025）：模型自己生成微调数据和更新指令，做持久的权重更新，是 joint 系统的代表。
6. **TT-VLA, On-the-Fly VLA Adaptation via Test-Time RL**（ACL 2026）：用逐步任务进度当 dense reward，在部署时对 VLA 做 RL。
7. **RoboMonkey**（CoRL 2025）：VLA 动作的采样加验证，是 planning-time scaling 的代表。
8. **AdaJEPA: An Adaptive Latent World Model**（arXiv 2606.32026）：用执行后观测到的转移在线适应 latent world model。
9. **AdaPower: Specializing World Foundation Models for Predictive Manipulation**（arXiv 2512.03538）：world foundation model 的测试时 specialization，并跨 episode 保留 memory。
10. **TTT-Parkour: Rapid Test-Time Training for Perceptive Robot Parkour**（arXiv 2602.02331）：在重建出的地形几何上做测试时训练，用于人形机器人跑酷。

## 自测问题

1. 按本文 Sec.2.1 的判据，TTT-Layer 的 $W_t$ 更新算 TTL，RSSM 的 recurrent state 更新和 KV cache 却不算，区别在哪？RMA 的 adaptation module 该怎么归类？
2. TTT 与 FTTA 的差别是"目标何时准备好"。各写出一个典型目标（rotation / MAE vs 熵），并说明为什么 FTTA 更容易 collapse，EATA / SAR 又是从哪两个角度去防 collapse 的。
3. Supervision horizon 和 persistence horizon 为什么是独立的？分别举出"单样本监督 + continual 保留"和"batch 监督 + episodic reset"的例子。
4. 为什么 STaR / ReST^EM 不算 TTI，而 TTRL 算？TTRL 的 consensus reward 在什么情况下会自我强化错误？T³RL 怎么缓解？
5. Best-of-N 里 coverage 随 N 增长，最终准确率却可能不升反降，请从 selector 的角度解释。再用 SPSA（Eq.5）说明 forward-only TTA 每一步的代价从哪里来。
