# Learning to (Learn at Test Time): RNNs with Expressive Hidden States

> arXiv [2407.04620](https://arxiv.org/abs/2407.04620) · ICML 2025（Spotlight poster，已在 PMLR v267 和 OpenReview 核实；arXiv v4 正文未注明会议），首版 2024 · 主题：02_test_time_training · 层级：foundational
> 代码：https://github.com/test-time-training/ttt-lm-jax （JAX，训练与复现论文）；https://github.com/test-time-training/ttt-lm-pytorch （PyTorch 朴素实现，README 称用于教程、不建议用来训练） · 笔记依据：`source/*.tex`（数字/公式）+ `paper.pdf`（图）

## 一句话

把 RNN 的隐状态换成一个模型 $f$ 的权重 $W$，把更新规则换成对自监督重建损失的一步梯度下降，这样每读一个 token 就在测试序列上"训练"一次（TTT layer）。自监督任务本身（K/V/Q 投影）在 outer loop 中与 next-token prediction 端到端学出。

## 问题设定

- **问题**：self-attention 在长上下文里效果好，但每个 token 的代价是 $O(t)$，KV cache 随长度线性增长；RNN 层（LSTM、Mamba）每个 token 的代价是 $O(1)$，但要把上下文压进固定大小的隐状态，长上下文时受限于隐状态的表达力。Figure 2（右）显示，在 Books 32k 上，Mamba 按 token index 统计的 perplexity 过了 16k 就不再下降，Transformer 则一直下降。
- **输入/输出**：sequence modeling layer，即自回归地把 $x_1..x_T$ 映射为 $z_1..z_T$（§2 脚注 1），可以直接替换 self-attention 或 Mamba 层。
- **假设**：标准语言模型预训练（next-token prediction），数据用 Pile（2k/8k）和 Books3（1k–32k）。不涉及分布偏移，也不涉及 RL 或控制。
- **之前卡在哪**：linear attention、DeltaNet、Mamba 的隐状态都是矩阵（线性），表达力有上限。作者的出发点是：自监督学习本来就擅长把海量数据压进模型权重，那就用它做"压缩启发式"（§1、§2.1）。

## 核心方法

**1. 三要素视角**（Figure 3）。任何序列层都可以写成"初始状态 + update rule + output rule"。Naive TTT 取 $W_0=f.\texttt{params()}$，每步代价 $O(1)$：
$$W_t = W_{t-1} - \eta\thinspace\nabla\ell(W_{t-1};x_t),\qquad z_t=f(x_t;W_t)$$
（`source/arxiv.tex` §2.1 Eq.1–2）。直观理解： $W$ 会"记住"梯度大的输入。

**2. 可学习的自监督任务**（§2.3 Eq.4–5）。多视角重建：
$$\ell(W;x_t)=\big\Vert f(\theta_K x_t;W)-\theta_V x_t\big\Vert^2,\qquad z_t=f(\theta_Q x_t;W_t)$$
$\theta_K x_t$ 是 training view， $\theta_V x_t$ 是 label view， $\theta_Q x_t$ 是 test view。**inner loop** 只优化 $W$，这时各个 $\theta$ 相当于损失函数的超参数。**outer loop** 用 next-token prediction 优化 $\theta_K,\theta_V,\theta_Q,\theta_\text{rest},\theta_\text{init}(=W_0),\theta_\text{lr}$，这时 $W$ 只是隐状态（Table 2）。反向传播要对 $\nabla\ell$ 再求导，也就是 gradients of gradients，和 meta-learning 相同（§2.2）。

**3. mini-batch TTT**（§2.4 Eq.6）。GD 的通式是 $W_t=W_{t-1}-\eta G_t=W_0-\eta\sum_{s\le t}G_s$，求出所有 $G$ 之后用一次 `cumsum` 就得到全部 $W_t$。三种取法：
- online GD： $G_t=\nabla\ell(W_{t-1};x_t)$，完全串行；
- batch GD： $G_t=\nabla\ell(W_0;x_t)$，完全并行，但 $W_t$ 离 $W_0$ 实际只有"一步"；
- **mini-batch GD**： $G_t=\nabla\ell(W_{t'};x_t)$， $t'$ 是上一个 mini-batch 的最后时刻，块内 $b$ 个梯度可以并行。全文取 $b=16$。

白板推导（线性 $f$，记 $k_s=\theta_Kx_s, v_s=\theta_Vx_s, q_t=\theta_Qx_t$，由于 $\nabla\ell=2(Wk_s-v_s)k_s^\top$）：
$$z_t = W_{t'}q_t - 2\eta\sum_{s=t'+1}^{t}(W_{t'}k_s - v_s)\thinspace(k_s^\top q_t)$$
与 linear attention 的 $z_t=\sum_s v_s k_s^\top q_t$ 相比，多出两项：一是误差修正项 $-W_{t'}k_s$（delta rule），二是块间通过梯度通道传过来的 $W_{t'}$。取 $b=1$、去掉 LN 和 residual，这个式子就退化为 DeltaNet（§4.2 原文："equivalent to TTT-Linear with inner-loop mini-batch size 1"）。

**4. Dual form**（§2.5 Eq.7–8，Appendix A）。primal 形式要逐个 materialize $d\times d$ 的 $G_t$，只能做外积，TensorCore 用不上。dual 形式不显式计算中间的 $G$ 和 $W$。在 $\theta=I$ 的特例下：
$$W_b=W_0-2\eta(W_0X-X)X^\top,\qquad Z=W_0X-2\eta\thinspace(W_0X-X)\thinspace\texttt{mask}(X^\top X)$$
其中 mask 保留 $s\le t$ 的上三角（含对角）、把下三角置零（原文："upper triangular mask with zeros"，类似 attention mask，只是用 0 而不是无穷大）。代价方面，算 $W_b$ 仍是 $O(bd^2)$，另外需要 $O(b^2d)$ 算 $z$。dual 形式用更高的理论复杂度换来 matmul 利用率，在 JAX/TPU 上训练 **>5×** 快于 primal（§2.5）。Appendix A 把它推广到任意深度的 MLP（可以理解为先做一次标准 backward，再带 mask 做一次 forward）。训练和 prefill 用 dual，decode 用 primal（§3.3）。

**5. 理论等价**（§2.6）。Theorem 1：线性 $f$ + batch GD + $\eta=1/2$ + $W_0=0$，此时 $W_t=\sum_s(\theta_Vx_s)(\theta_Kx_s)^\top$，恰好就是 linear attention。Theorem 2：把 learner 换成 Nadaraya-Watson 估计器，取核 $\kappa\propto e^{(\theta_Kx)^\top\theta_Qx'}$（Eq.9–10），就得到 self-attention。由此把 TTT layer 推广为任意 learner 的 train/predict（Figure 9）。

**6. 实例化与技巧**（§2.7）。 $f(x)=x+\texttt{LN}(f_\text{res}(x))$。TTT-Linear 的 $f_\text{res}=Wx$（ $W$ 为方阵）；TTT-MLP 为两层 MLP，隐层宽度是输入的 4×，接 GELU。learnable $W_0$ 可以稳定训练。learnable $\eta(x)=\eta_\text{base}\thinspace\sigma(\theta_\text{lr}\cdot x)$， $\eta_\text{base}$ 对 Linear 取 1、对 MLP 取 0.1（从 {0.01, 0.1, 1, 10} 里选不发散的最大值，Appendix C）。backbone 默认用 Mamba 式结构： $\theta_K$ 与 $\theta_Q$ 合并为一个投影，投影之后、进入 TTT layer 之前各接一个时间卷积（Conv K / Conv Q），输出再乘一个 GELU gate（Figure 13）。显存方面：默认会保存 $W_1,\dots,W_T$；用了 mini-batch 和 dual form 后仍需保存 $T/b$ 个 mini-batch 末尾的 $W$，所以再沿时间维做 gradient checkpointing（"through time"，Appendix C）。

## 关键实验与证据

协议照搬 Mamba 论文：模型规模为 125M/350M/760M/1.3B（Mamba 对应 130M–1.4B），使用 Chinchilla recipe，每个 batch 0.5M tokens，AdamW (0.9, 0.95)，cosine 衰减到 1e-5，10% warmup，weight decay 0.1。各规模的 token 数为 2.5B/7B/15B/26B（Table 3）。统一用 Llama tokenizer。评测指标只有 perplexity。

- **Table 1 消融**（125M，Pile， $T=2048$），逐行累加：
  - linear attention：15.91；改进版 linear attention：15.23；TTT 等价形式：15.23（验证 Theorem 1）；
  - + learnable $W_0$：15.27（+0.04，略差，但后面各行没有它就无法稳定训练）；
  - + LN 和 residual：14.05（−1.22）；
  - **+ mini-batch TTT（ $b$ 从 2048 改为 16）：12.35（−1.70，贡献最大）**；
  - + learnable $\eta$：11.99（−0.36）；
  - + Mamba backbone：11.09（−0.90）。
- **Figure 7**： $b$ 越小 perplexity 越好（ $b=16$ 时为 11.09）；前向时间先随 $b$ 增大而下降，再因 $O(Tbd)$ 项上升。 $b$ 在速度和质量之间折中。
- **Figure 4/14**：一步 GD 后有 $\ell(W_t;x_t)<\ell(W_{t-1};x_t)$；而且越往序列后面， $\ell(W_t;x_t)$ 相对 $\ell(W_0;x_t)$ 的差距越大，说明 inner loop 确实在学。
- **Pile（Figure 10）**：2k 下 TTT-Linear (M)、Mamba、Transformer 的曲线基本重合；8k 下 TTT-Linear (M) 和 TTT-MLP (M) 都明显优于 Mamba。
- **Books（Figure 11/15）**：2k 下 Mamba 略优于 TTT-Linear；32k 下 TTT (M) 两个变体都优于 Mamba，TTT-MLP (T) 也略优于 Mamba。作者总结的规律是：上下文越长，TTT 相对 Mamba 的优势越大（§3.1）。
- **Figure 2（右）**：Books 32k，所有方法的训练 FLOPs 与 Mamba 1.4B 对齐。Transformer 为 TF finetune，19 个 block，推理 FLOPs 为 2.8×（Appendix C）。TTT-Linear 和 TTT-MLP 在 16k 之后 perplexity 仍在下降。
- **backbone 效应**：在 Mamba backbone 下，TTT-MLP 最多与 TTT-Linear 持平；换到 Transformer backbone，TTT-MLP 明显更好。作者的假设是时间卷积对表达力弱的隐状态帮助更大（§3.1）。
- **速度**：TPU v5e-256 上 2k 上下文，每个训练 iteration Transformer 需 0.30s，TTT-Linear 需 0.27s（§3.3）。A100 推理 kernel（Figure 12，1.3B）：各 RNN 每 token 的延迟近似恒定，Transformer 则线性增长。prefill 时 TTT-Linear 最快，TTT-MLP 慢于 Mamba；decode 时 TTT-MLP 明显慢于 TTT-Linear 和 Mamba。

## 局限与论文没说的

**作者承认的**：
- TTT-MLP 在 memory I/O 上仍有困难，wall-clock 代价相对 FLOPs 增加得多（摘要、§4.2）；
- GPU 上只写了推理 kernel，没有训练 kernel（§3.3）；
- 没有得到干净的 scaling law 线性拟合（§3.1）；
- 规模只到 1.3B、上下文只到 32k；
- 视角只用了线性投影；
- 没测 hybrid 架构。

**我读出来的**：
1. **评测只有 perplexity**。没有下游任务，也没有 recall 类或 needle 类的长上下文测试，所以"更好地利用长上下文"只按 token-index perplexity 衡量。
2. **基线不全**。作者在 §4.2 称 DeltaNet 是最相关的工作，并讨论了 GLA、Mamba 2、Gated DeltaNet 等矩阵隐状态 RNN，但实验里只和 Transformer、Mamba 比。
3. **backbone 混杂**。主结果都用 Mamba backbone，Table 1 里它贡献了 −0.90。TTT (T) 在短上下文下常常不如 Mamba。
4. **Figure 16 的另一面**：
   - 所有从零训练的方法，训练上下文过长后 perplexity 都会变差；
   - 每个模型规模下 TF finetune 都是最好的（TF finetune 只在 4k 及以上有）；
   - 1.3B 时 TF pretrain 在 16k/32k 上也好于两个 TTT 变体（按参数量比较，不按 FLOPs）。
   因此 TTT 的卖点是"FLOPs 或延迟对齐之下优于 Mamba"，不是"优于 attention"。
5. **消融的覆盖面**。Table 1 是单一规模（125M）、单一长度（2k），没有给方差； $b=16$ 在长上下文、大模型上是否仍然合适，论文没有验证。后继的 [LaCT](../lact/notes.md) 恰好认为小 chunk 会导致 GPU 利用率低。
6. **稳定性敏感**。需要 learnable $W_0$， $\eta_\text{base}$ 取"不发散的最大值"，TTT-MLP 还要对 $\eta$ 做 warmup。Figure 14 显示中间层的 $\Vert x_t\Vert$ 持续上升。
7. **概念上要分清**：这里的 "test-time training" 不是部署期的 test-time adaptation。测试时 $\theta$ 全部冻结，每条序列都从 $W_0$ 重新开始，跨序列不积累任何东西，本质上是一种 learned in-context learning / RNN。
8. **没有涉及控制、视觉或多模态序列**。作者只在 Discussion 里提到：上下文更长时 $f$ 也需要更大，对 video 和 embodied agents 这类上下文可达百万到十亿量级的任务， $f$ 可以是卷积网络。

## 在脉络中的位置

- **前驱**：
  - [TTT with self-supervision](../ttt-self-supervised/notes.md)：每个测试样本定义一个学习问题，但那里的自监督任务是手工设计的；本文改为在 outer loop 里学出来；
  - TTT-MAE 与 TTT on video streams（自回归 TTT，§4.1.1 认为与本文最相关）；
  - fast weights 与 fast weight programmers（Schmidhuber 1992；Irie et al. 2021 已用 MLP 做 fast weights）；
  - linear attention（Katharopoulos 2020）、DeltaNet（Schlag 2021）；
  - dynamic evaluation（Krause 2018）；
  - MAML 式 bi-level optimization；
  - 作者的预稿 arXiv 2310.13807。
- **后继 / 影响**：
  - [Titans](../titans/notes.md)：深层神经记忆，用带动量和遗忘的 surprise 梯度更新；
  - [LaCT](../lact/notes.md)：大 chunk、Muon 优化器，扩展到视频和 3D；
  - [One-Minute Video](../ttt-one-minute-video/notes.md)：把 TTT-MLP 插进 DiT，用于长视频生成；
  - [TTT-E2E](../ttt-e2e/notes.md)：用 next-token prediction 做 TTT，并 meta-learn 初始化；
  - [TTT with KV binding = linear attention](../ttt-kv-linear-attention/notes.md)：把本文 Theorem 1 推广到更大一类 TTT（KV binding 类都可写成 learned linear attention），并借此挑战把 TTT 解释为"测试时记忆（memorization）"的叙事；
  - 综述见 [survey-test-time-intelligence](../survey-test-time-intelligence/notes.md)。
- **与本项目其他主题的联系**：
  - **World model**：[DreamerV3](../../01_world_model/dreamerv3/notes.md) 的 RSSM 用 GRU，是固定大小的隐状态；[Genie](../../01_world_model/genie/notes.md) 和 [Dreamer 4](../../01_world_model/dreamer4/notes.md) 的 dynamics 用 Transformer，每步代价随上下文增长。TTT layer 是介于两者之间的候选 sequence model，[One-Minute Video](../ttt-one-minute-video/notes.md) 已在视频生成里验证它可以充当长程记忆。
  - **Real2sim2real**：[RMA](../../03_real2sim2real/rma/notes.md) 的 adaptation module 从历史中隐式估计 extrinsics，相当于一种隐式的在线 system identification。TTT layer 则把"在部署历史上拟合模型"写成显式的 inner loop。控制任务里下一时刻的状态本身就是免费的监督信号，这一点与 [SimDist](../../05_intersections/simdist/notes.md)"部署时只更新 dynamics"的思路相通。
  - **RL**：[PAD](../../05_intersections/pad/notes.md) 被本文列为 TTT 在 robot manipulation 上的后续应用，但它用的是手工设计的 inverse dynamics 辅助任务，而且 encoder 是持续更新的；本文的任务可学，但状态每条序列都会重置。对 POMDP 下的 recurrent policy 来说，TTT layer 同样可以作为 memory 候选，不过论文没有测试。

## 前置知识

1. self-attention、KV cache，以及 linear attention 的递归形式（把 $\sum v_sk_s^\top$ 当作隐状态）；
2. RNN/SSM 的基本框架与 Mamba（它是本文的 baseline 和 backbone 来源）；
3. online / batch / mini-batch GD 的区别，以及 cumsum 形式的 GD；
4. meta-learning 与 bi-level optimization（inner/outer loop、gradients of gradients，如 MAML）；
5. 原始 TTT（Sun 2020）与 denoising autoencoder 式的重建任务；
6. fast weights、DeltaNet 的 delta rule；
7. perplexity 评测与 Chinchilla scaling 协议；gradient checkpointing；TensorCore 与 matmul 的硬件效率。

## 复现可行性

- **代码**：
  - JAX 仓库（基于 EasyLM）提供训练脚本，可复现论文实验；README 称提供 Llama-2 tokenize 后的 Pile 和 Books3（GCS bucket），支持 GPU 和 TPU VM。
  - PyTorch 仓库是朴素实现，README 明确不建议用它训练；另有单独的推理 kernel 仓库。
  - 两个仓库均已用 WebFetch 确认存在。
- **算力**：
  - 论文**没有报告**总 GPU/TPU 小时数，只说实验用 JAX 在 TPU 上跑，提到了 v5e-256 pod，致谢中说部分算力来自 Google TPU Research Cloud 和 Hyperbolic Labs。延迟测试用 A100 80G。
  - 最大配置为 1.3B、26B tokens、50000 steps（Table 3）。由 Figure 10/11 的横轴可见，最大规模的训练 FLOPs 在 $10^{20}$ 量级。
  - 全套 scaling 实验（4 个规模 × 多种长度 × 多个方法）超出一般学术预算。
- **数据**：Pile 可以获取；Books3 有版权争议（论文引用了 authorsguild2023），获取可能受限。
- **最小复现建议**：
  1. **CPU/numpy 验证**（1 小时内）：随机取 $X\in\mathbb{R}^{d\times b}$（ $d=64,b=16$），实现 primal（逐步 $W_t$）和 dual（Eq.7–8 的 mask 形式），检查两者输出一致；再令 $W_0=0,\eta=1/2$ 并采用 batch GD，验证与 linear attention 的输出一致（Theorem 1）。
  2. **125M、Pile 2k**：用 JAX 仓库只复现 Table 1 的两行，"TTT equivalence ($b=T$)" 对 "+mini-batch ($b=16$)"，目标值分别为 15.23 和 12.35。注意 Table 1 是逐行累加的，12.35 那一行已包含 learnable $W_0$ 与 LN + residual，不是只改 $b$。配置为 2.5B tokens、4800 steps（Table 3）。按常用的 $6ND$ 近似粗算约 $1.9\times10^{18}$ FLOPs（本笔记推算，非论文数字），几张 A100 可以承受。
  3. 可选：在同一设置下扫描 $b\in\lbrace1,4,16,64,2048\rbrace$，复现 Figure 7 左图的趋势。
- **预计难点**：
  - learnable $W_0$ 和 $\eta_\text{base}$ 带来的训练稳定性；
  - 反向传播穿过 inner-loop 梯度时的显存（需要 through-time checkpointing）；
  - GPU 上没有官方训练 kernel，所以 GPU 上的 wall-clock 数字不能直接对比；
  - Mamba backbone 的细节（Figure 13）需要对齐。

## 自测问题

1. 从 Eq.4 出发推导 Theorem 1：为什么线性 $f$、batch GD、 $W_0=0$、 $\eta=1/2$ 会给出 linear attention？改成 mini-batch GD 后，哪一项让它不再等价，这一项为什么提升了表达力？
2. 写出 $b=16$ 时第二个 mini-batch 内 $z_t$ 的表达式。从 $W_s$ 到 $W_t$ 有"cumsum"和"梯度"两条信息通道，它们各在什么条件下起作用？
3. 推导 dual form 中 $\Delta=(W_0X-X)\thinspace\texttt{mask}(X^\top X)$，说明 mask 为什么是上三角（含对角），以及它为什么在理论复杂度更高的情况下仍然更快。
4. 哪些量是 outer-loop 参数，哪些是 inner-loop 状态？为什么需要单独的 test view $\theta_Q$，它与 training view、label view 的分工是什么？
5. 证明 $b=1$、去掉 LN 和 residual 的 TTT-Linear 等价于 DeltaNet 的 delta rule。如果把 TTT layer 放进 world model 做 dynamics，inner-loop 的自监督 label 可以取什么？与 RMA 式的隐式 system identification 相比，这样做有什么不同？
