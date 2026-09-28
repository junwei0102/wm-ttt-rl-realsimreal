# End-to-End Test-Time Training for Long Context

> arXiv [2512.23675](https://arxiv.org/abs/2512.23675) · arXiv preprint（v2，2025-12-31；正文未注明会议）, 2025 · 主题：02_test_time_training · 层级：frontier
> 代码：https://github.com/test-time-training/e2e （JAX；论文摘要和 §3 都给出了这个链接，仓库页面已确认存在） · 笔记依据：`source/arxiv.tex`（数字/公式）+ `paper.pdf`（图）

## 一句话

本文把长上下文语言建模看成 continual learning 问题。架构只用 sliding-window attention（SWA）Transformer，测试时对已读的上下文继续做 next-token prediction 的梯度下降，把上下文压缩进最后 1/4 个 block 的 MLP 权重；训练时用 meta-learning（gradients of gradients）直接优化"TTT 之后的 loss"，从而学到适合 TTT 的初始化。测试时和训练时两处都是端到端（E2E）的。

## 问题设定

- **任务**：标准自回归语言建模。prefill 阶段读入 $x_0(\langle\texttt{BOS}\rangle),x_1,\dots,x_T$，decode 阶段输出 $\hat p_{T+1}$，测试 loss 为 $\texttt{CE}(\hat p_{T+1},x_{T+1})$（§2 开头）。
- **瓶颈**：full attention 的 prefill 是 $O(T^2)$，decode 是 $O(T)$，KV cache 随上下文长度线性增长。Mamba 2、Gated DeltaNet 这类 RNN 每个 token 的代价恒定，但上下文越长，它们越用不好上下文（Figure 1 左）。SWA 和 SWA/full 混合同样比不上 full attention。
- **目标**：在每个 token 代价恒定（prefill $O(T)$、decode $O(1)$）的前提下，让 loss 随上下文长度的下降方式与 full attention 一致。作者不追求逐字召回，追求的是"压缩"（§1）。
- **之前卡在哪**：
  1. dynamic evaluation（Mikolov 2013，Krause 2018）在测试时也做 next-token prediction 的 TTT，但训练时优化的是"开箱即用的 loss"，训练与测试不一致，本文称之为 TTT-naive。
  2. 以 [TTT layers](../ttt-layers/notes.md) 和 [LaCT](../lact/notes.md) 为代表的长上下文 TTT，inner loop 用逐层的 Key-Value Binding（KVB）重建损失，而不是网络末端的真实任务损失。它们还被当作"替换 self-attention 的架构设计"（§4.2.3）。另外，这类方法必须把隐状态放进 GPU 内部的单个芯片上，隐状态大小因此受限；TTT-E2E 的隐状态是普通 MLP 层，可以用标准工具跨 GPU 分片（§3.7）。
- **假设**：纯文本；预训练数据为 DCLM-Baseline，只保留长度 ≥8K 的文档；长上下文扩展与评测用 Books（评测用 held-out 划分）。不涉及分布偏移、奖励或控制。

## 核心方法

**1. Toy：TTT via next-token prediction**（§2.1）。取一个去掉全部 attention、只剩 MLP 的 Transformer $f(\cdot;W)$，它本身等价于 bigram。测试时把上下文当训练集，按顺序做 online GD：
$$\ell_t(W)=\texttt{CE}\big(f(x_{t-1};W),x_t\big),\qquad W_t=W_{t-1}-\eta\thinspace\nabla\ell_t(W_{t-1})$$
（`source/arxiv.tex` Eq.1–2），最后输出 $\hat p_{T+1}=f(x_T;W_T)$。 $x_1$ 的信息由此"写进"了更新后的 MLP（Figure 2 左）。

**2. Outer loop：learning to (learn at test time)**（§2.2）。关键在于 $\ell_t(W_{t-1})$ 同时扮演两个角色：它是 TTT 的训练损失，也是"以 $x_{<t}$ 为条件预测 $x_t$"的测试损失，因为 $W_{t-1}$ 从未见过 $x_t$。所以训练目标直接取
$$\mathcal L(W_0;X)=\frac1T\sum_{t=1}^T\ell_t(W_{t-1})\quad\text{(Eq.3, TTT-E2E)}\qquad\text{vs.}\qquad \mathcal L_{\texttt{naive}}(W_0;X)=\frac1T\sum_{t=1}^T\ell_t(W_0)\quad\text{(Eq.4, TTT-naive)}$$
对 $W_0$ 求梯度要穿过 inner loop 的更新。白板推导（笔者补充，非论文原式）： $\partial W_t/\partial W_{t-1}=I-\eta\nabla^2\ell_t(W_{t-1})$，因此 $\nabla_{W_0}\mathcal L$ 中含有 Hessian-vector product，也就是 gradients of gradients。这与 MAML 的结构相同，区别在于这里的 inner loop 学的是一条序列，而不是一个数据集（§4.4）。Figure 2 右的 toy 实验中，TTT-naive 只比 bigram 基线好一点，TTT-E2E（ $b=1$）几乎追平 full attention。

**3. Mini-batch TTT + sliding window**（§2.3）。online GD 有两个问题：串行（效率）和单 token 梯度容易爆炸（稳定性）。改为 mini-batch：
$$W_i=W_{i-1}-\eta\thinspace\frac1b\sum_{t=(i-1)b+1}^{ib}\nabla\ell_t(W_{i-1}),\qquad \mathcal L(W_0;X)=\frac1T\sum_{i=1}^{T/b}\sum_{t=(i-1)b+1}^{ib}\ell_t(W_{i-1})$$
（Eq.5–6）。副作用是 batch 内所有预测都用 $W_{i-1}$，模型在 batch 内又退化成 bigram，loss 随 batch 内位置上升，呈锯齿状（Figure 2 右， $b=16$ 的红线）。补救办法是把 attention 限制为窗口 $k$ 的 SWA，并要求 $k\ge b$，让 batch 内的上下文由 attention 负责。主实验取 $T=128$ K， $k=8$ K， $b=1$ K。作者在结论中把两者分工解释为：SWA 是 short-term memory，被 TTT 更新的权重是 long-term memory。

**4. 三个实现细节**（§2.3.1，Figure 3 左）：
- 只 TTT MLP 层。embedding、norm、attention 在 inner loop 中冻结，因为更新它们会导致 outer loop 不稳定。
- 只更新最后 1/4 个 block（760M 模型共 24 个 block，即最后 6 个）。梯度只回传到这 $L/4$ 个 block。
- 被更新的 block 中额外加一个**静态**的第二 MLP，作为存放预训练知识的"safe storage"，以缓解遗忘。为了保持总参数量不变，全网的 MLP hidden dim 相应缩小。

Outer loop 用标准优化器更新全部参数，包括 fast weights 的初始化 $W_0$、attention、embedding 等。Inner loop 只更新那几个 MLP，即 fast weights。

**5. 推理**：prefill 时每满 $b$ 个 token 做一次 TTT 梯度步。decode 时先用当前权重生成，攒满 $b$ 个自生成的 token 后再对它们做一次 TTT（§2.3.2）。因此 decode 延迟等于"SWA decode + 每 $b$ 个 token 一次 prefill 级的 TTT 步"（§3.7）。

**6. 另一条推导路径：从 TTT-KVB 出发**（§2.4，Table 1，760M、DCLM、8K）。TTT-KVB 在每层上做
$$\ell^{(l)}_ t=\big\Vert g(\theta_K^{(l)}x_t^{(l)};W^{(l)}_ {t-1})-\theta_V^{(l)}x_t^{(l)}\big\Vert^2,\quad z^{(l)}_ t=g(\theta_Q^{(l)}x_t^{(l)};W^{(l)}_ t)$$
（Eq.7–8）。作者分三步把它变成 TTT-E2E：
- ① 输出规则简化为 $z_t=g(\theta_Kx_t;W_{t-1})$（Eq.9），loss 从 2.818 变为 2.819，几乎无影响；
- ② **关键步**：把逐层 KVB 损失换成网络末端的 next-token 损失，同时去掉 $\theta_K,\theta_V$，得到 "TTT-E2E all layers MH"，loss 2.806；
- ③ 只更新最后 1/4 个 block，并换成常规 MLP（不再用 multi-head + LoRA），得到 TTT-E2E，loss 2.805。在 760M 规模下，隐状态从 18M 增到 88M（5×），H100 上的 prefill 延迟从 0.017 降到 0.0086 s/1K tokens。

原因在于 E2E 损失下，要更新某个 block 里的小 multi-head MLP，梯度得先穿过它上方的大 MLP；要继续往更下层传，还得再穿过下方的 attention 层。所以"少更新几个 block，但每个 block 的状态更大"更划算。SWA 基线的 loss 是 2.827。两条推导是对偶的：主推导是在测试时 E2E 的基础上把训练时也做成 E2E；这条推导是在训练时 E2E 的基础上把测试时也做成 E2E。从 RNN 视角看，TTT-E2E 是只有一个"RNN 层"的 RNN，隐状态就是这些 MLP 权重。

## 关键实验与证据

基本设置：先在 8K 上预训练（DCLM），再在目标长度上做 extension fine-tuning（Books，token 数为预训练的 5%），评测也在 Books 上。所有带窗口的基线都取 $k=8$ K。基线包括 full attention、SWA、SWA:full=5:1 混合、Mamba 2、Gated DeltaNet、TTT-KVB（LaCT）；后三者用官方代码，并做了两处增强，见附录 C。

- **随上下文长度的扩展**（Figure 1 左，3B 模型，164B tokens）：TTT-E2E 相对 full attention 的 loss Δ 在 8K–128K 基本持平，读图约为 −0.01。其他方法的 Δ 都随长度变差，SWA 在 128K 时读图约为 +0.05。Figure 9（附录）画的是绝对 loss。据图注，到 32K 为止所有方法的 loss 都随上下文变长而下降；32K 之后，SWA、Mamba 2、GDN、TTT-KVB 的 loss 反而上升，full attention 和 hybrid SWA/full 则在所有长度上都继续下降（读图 TTT-E2E 也一直下降）。作者的解释是：上下文越长，extension fine-tuning 时每个 outer-loop batch 中的序列越少，梯度方差越大，而这些方法又用不好更长的上下文，于是方差带来的损害超过了收益。
- **延迟**（Figure 1 右）：prefill 延迟与上下文长度无关，128K 时在 H100 上比 full attention 快 2.7×。
- **逐 token loss**（Figure 6，32K/128K）：TTT-E2E 是唯一在整个上下文范围内都低于 full attention 的方法，优势主要来自前段 token。在 $t<1$ K 时还没有发生任何 TTT 步，计算图与 full attention 完全相同，只有权重不同。作者的解释是：TTT-E2E 的权重只需在"当前 mini-batch"上表现好，而 full attention 的权重要兼顾所有未来位置。
- **消融**（Figure 4，760M）：
  - 窗口 $k$：TTT-E2E 对 $k$ 的敏感度与 SWA、GDN 相近。这组消融只做 8K 预训练、在 DCLM 上评测（窗口和 mini-batch 两组消融都是如此），所以 $k=8$ K 时 SWA 与 full attention 等价，此时 TTT-E2E 仍能把 loss 降低 0.018，说明它带来的是正交的增益，而不只是弥补 SWA 的缺口。
  - mini-batch $b$： $b$ 越大越差。 $b=8$ K 等于不做 TTT，此时 TTT-E2E 为 2.825，TTT-KVB 为 2.826，full attention 为 2.827，说明架构改动本身几乎不起作用。
  - **更新层数**（最重要的消融）：只更新 1 层或 3 层时，loss 不能像 full attention 那样随上下文扩展；更新 6 层或 12 层可以，且 12 层与 6 层效果相当，因此默认更新 1/4。
- **随训练算力的扩展**（Figure 5）：算力较小时 TTT-E2E 的优势随算力增加而缩小；模型超过约 760M、训练 token 超过约 48B 后，曲线趋于平行于 full attention。Gated DeltaNet 呈现同样的趋势。模型规模这一轴上仍有小幅上翘。另有两条 anecdotal 观察：Llama 3 tokenizer 比 Llama 2 tokenizer 在 3B 上让 TTT-E2E 相对 full attention 的优势多出约 0.01；用 SlimPajama 训练时，token 轴上同样出现上翘。
- **NIAH**（Table 2，RULER S-NIAH，3B@128K）：full attention 全面领先。128K 时 S-NIAH-1/2/3 的得分，full attention 为 0.99/0.86/0.64，TTT-E2E 为 0.06/0.05/0.03，与 SWA 和各 RNN 同属一档。作者明确承认"压缩"会丢掉细节，不擅长召回。
- **长序列 decode**（Figure 7）：prefill 8K 的 Books 文本后续写 8K，用 Qwen-3-8B-Base 给拼接后的序列打 loss，共 512 条序列。TTT-E2E 得到的 Qwen loss 低于 full attention。作者人工检查了约 20 个样本；采样设置为 temperature 1、top-p 0.95、repetition penalty 1.1（附录 D）。
- **训练代价**（Figure 8，H200）：训练延迟在 128K 时比 full attention 快 1.2×，在 8K 时慢 3.4×。每 token 的 FLOPs 基本恒定，但 8K→32K 区间延迟仍在上升，原因是 gradient checkpointing through time 的量随 $\log T$ 增加。

## 局限与论文没说的

**作者承认的**：
- 训练慢。gradients of gradients 远不如普通反向传播成熟，cuDNN FlashAttention 也不支持二阶梯度，而多数训练算力花在短上下文预训练上，8K 时慢 3.4× 是实打实的代价（§3.7）。
- 召回能力弱，NIAH 结果远差于 full attention。
- 三个实现细节可能只是当前实验设置下的产物（§2.3.1）。
- decode 评测只是 base model 上的有限尝试。
- tokenizer 和数据质量的影响只有 anecdotal 观察。
- 125M 和 350M 模型在 32K 以上做 fine-tuning 不稳定。
- 预训练时丢弃了短于 8K 的文档，原因是跨文档 reset MLP 会拖慢训练。

**笔者读出来的**：
- **评测口径窄**：主证据几乎全是 Books 上的 loss/perplexity，而 fine-tuning 也在 Books 上，领域是匹配的。没有测长文档 QA、多跳推理、代码等下游任务，"与 full attention 同样扩展"只在 loss 这一口径上成立。
- **规模外推**："在生产级规模上会保持同样趋势"的依据只有 ≤3B 模型、≤128K 上下文，且模型规模这一轴本身就有上翘。
- **短上下文延迟不占优**：Figure 1 右读图，8K–32K 区间 TTT-E2E 的 prefill 延迟并不低于 full attention，优势只在更长的上下文中出现。
- **服务端内存**：论文没有讨论并发服务的代价。每条序列都要持有自己的一份 fast weights（760M 模型就有 88M 参数），多用户 batch decode 的显存和吞吐都没有给出。
- **超参数与鲁棒性**：inner-loop 学习率 $\eta$ 的取值、以及它是否可学习，正文和附录都没有写。上下文中含噪声或对抗内容时权重会被写坏的问题，也没有测（作者只提到 self-generated tokens 可能起到类似 gating 的作用）。
- **数字小出入**：按 Table 3，2.7B 模型的基本配方是预训练 54B tokens、微调 2.7B tokens，取 3 倍后是 162B + 8.1B = 170.1B（只算预训练也是 162B），与摘要所说的"164B tokens"对不上，论文没有解释。

## 在脉络中的位置

- **前驱**：
  - dynamic evaluation（Mikolov 2013 / Krause 2018，即 TTT-naive 的原型）；
  - [TTT layers](../ttt-layers/notes.md)（TTT-KVB、mini-batch TTT、gradients of gradients）；
  - [LaCT / Test-Time Training Done Right](../lact/notes.md)（TTT-KVB 加 SWA 加大 chunk，是本文的基线和推导起点）；
  - [Titans](../titans/notes.md)（同属 KVB 变体）；
  - MAML（学 inner loop 初始化）；
  - Clark et al. 2022：在 full attention 模型末端加一个用 next-token loss 更新的 MLP fast weight，方法上最接近本文，但没有线性复杂度，也没有和 attention 交错；
  - Schmidhuber 1992 的 fast weight programmers；
  - [TTT with self-supervision](../ttt-self-supervised/notes.md)：本文在 §4.2.2 引用它作为"用 self-supervision 做 TTT、提升分布偏移下泛化"的代表工作（本文把 TTT 的思想追溯到 1970 年代的 locally weighted regression 和 dynamic evaluation 等更早的工作）。
- **后继 / 影响**：
  - [TTT-KVB 实为 linear attention](../ttt-kv-linear-attention/notes.md) 采用了本文的 TTT-KVB / TTT-E2E 二分法，并进一步论证 KVB 式 TTT 本质上是 learned linear attention 而非记忆，与本文"长上下文 TTT 不必记忆 KV"的观点呼应；
  - [TTT-Discover](../ttt-discover/notes.md) 出自同一团队（test-time-training 组织），沿用本文的 continual learning 叙述，把 TTT 做成"测试时对单个问题做 RL"；
  - [Test-time intelligence 综述](../survey-test-time-intelligence/notes.md) 把它归入神经 test-time memory 架构中"memory objectives and update rules"这一方向，视其为 TTT-Layer 之后改进训练目标的后续工作。
- **与本项目其他主题的联系**：
  - **World model**：长时程视频 world model（[Genie](../../01_world_model/genie/notes.md)、[Dreamer 4](../../01_world_model/dreamer4/notes.md)、[Cosmos](../../01_world_model/cosmos-wfm/notes.md)）都受上下文窗口的限制。本文"SWA 管短期记忆、fast weights 管长期记忆"的结构是一种恒定代价的记忆方案，[One-Minute Video](../ttt-one-minute-video/notes.md) 已在视频生成中用过 TTT(-KVB) 层（论文把它列入 TTT-KVB 一系）。本文 RoPE θ 的取法也参考了 million-length video world model 的工作（附录 B）。
  - **RL**：作者点出长 CoT 的 RL 生成正是 decode 时自训练的应用场景，但没有测试。[TTRL](../ttrl/notes.md) 和 TTT-Discover 是"测试时 RL"的另一条线。
  - **Real2sim2real**：[PAD](../../05_intersections/pad/notes.md) 被本文引用为"序列上的 TTT"，其结论是在机器人视频流上不 reset 更好。[RMA](../../03_real2sim2real/rma/notes.md) 把历史压缩成隐变量来做隐式的 system identification。本文则把历史压缩进权重，并用 meta-learning 学初始化，可以视为部署时适应的另一种实现（这是笔者的类比，论文未涉及控制）。

## 前置知识

1. Transformer 的 self-attention 与 KV cache 的复杂度；sliding-window attention；next-token prediction、cross entropy、perplexity。
2. mini-batch SGD 与 online GD 的区别（并行度、方差）。
3. meta-learning / bi-level optimization，尤其是 MAML 的 inner loop / outer loop，以及 gradients of gradients（Hessian-vector product）。
4. 固定大小隐状态的序列模型：linear attention、DeltaNet、Mamba 2、Gated DeltaNet，以及把它们统一起来的 TTT layer（KVB）视角。建议先读 [TTT layers](../ttt-layers/notes.md)。
5. fast weights / fast weight programmers；dynamic evaluation。
6. 工程层面：gradient checkpointing、RoPE θ 与上下文扩展、Chinchilla scaling。

## 复现可行性

- **代码**：https://github.com/test-time-training/e2e ，纯 JAX，用 uv 管理依赖、Hydra 管理配置、W&B 记录日志，支持 Slurm 多节点。据仓库 README（非论文）：要求 CUDA 12.8.1、cuDNN 9.8.0、NCCL 2.26.2；已用 Llama 3 tokenizer 处理好的 DCLM-8K 和 Books3 数据放在 GCS；只发布了 TTT-E2E 的少量 checkpoint，托管在 requester-pays 的 GCS bucket 上：125M/1B/3B 各有 DCLM 8K 预训练版和 Books 8K 版，128K 只有 3B（Books）一个。论文 §3 声明所有实验都能用该仓库复现。
- **算力**：论文**没有报告**总 GPU 时数。已知信息只有：多数训练在 GB200 上完成，prefill 延迟在单张 H100 上测，训练延迟在 H200 上测。数据量见 Table 3：125M 模型预训练 2.5B tokens、微调 125M；760M 模型预训练 15B、微调 750M；2.7B 模型预训练 54B、微调 2.7B。主实验再乘 3 倍。所有 3B 主结果都需要多卡集群。
- **最小可复现实验**：
  1. **Toy**（附录 A）：2 个 block，dim 384，6 个 head，上下文 128，DCLM 数据，token 数按 Chinchilla 配比，batch 为 125K tokens，学习率在 [2e-3, 6e-3] 内按 0.5e-3 步长搜索。复现 Figure 2 右中五条曲线：有 attention、无 attention、TTT-naive、TTT-E2E $b=1$、TTT-E2E $b=16$。重点验证两点：naive 与 E2E 之间的差距，以及 $b=16$ 时的锯齿。单卡应该可行（笔者估计）。
  2. **进阶**（笔者建议的缩小版）：125M 模型，在 DCLM 8K 上预训练并在 DCLM 上评测，对比 SWA（此设置下等价于 full attention）、TTT-E2E（ $b=1$ K）和 TTT-E2E（ $b=8$ K，即不做 TTT），看能否重现 Figure 4 中间一栏的趋势。原文这组消融用的是 760M 模型，对照组是 TTT-KVB。
- **预计难点**：
  - 二阶梯度的显存开销和 gradient checkpointing through time；
  - FlashAttention 不支持 double backward，速度会受影响；
  - outer loop 的稳定性：inner loop 若更新 embedding、norm 或 attention，outer loop 会不稳定，所以只 TTT MLP（§2.3.1）；另外作者发现 QK norm 能让 TTT-E2E 训练更稳定（附录 C）。这是两件独立的事，QK norm 并不是用来让 attention 可以被 TTT 的；
  - $\eta$ 等 inner-loop 超参数要从代码中查，论文没有给。

## 自测问题

1. 为什么 $\ell_t(W_{t-1})$ 既是 TTT 的训练损失又是公平的测试损失？Eq.3 与 Eq.4（TTT-naive）的梯度在计算上差在哪里？请写出两步 inner loop 时 $\partial\mathcal L/\partial W_0$ 中出现 Hessian 的位置。
2. 不加 attention、只做 mini-batch TTT 时，loss 为什么在每个 batch 内上升？为什么必须满足 $k\ge b$？ $b=8$ K 在本文设置下为什么等于"不做 TTT"？
3. 从 TTT-KVB 走到 TTT-E2E 的"关键步"具体删掉了哪两处差异？在 E2E 损失下，为什么"少更新几个 block、用大 MLP"比"每层都更新小的 multi-head MLP"更省算力，状态也更大？
4. $t<1$ K 时 TTT-E2E 还没有做任何梯度步，为什么 loss 已经低于 full attention？这对"meta-learning 学到的初始化"意味着什么？
5. TTT-E2E 在 loss 上随上下文的扩展追平了 full attention，NIAH 却几乎失败。用"压缩 vs 近乎无损的召回"解释这一现象，并设计一个能区分两者的评测。
