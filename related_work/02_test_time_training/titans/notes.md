# Titans: Learning to Memorize at Test Time

> arXiv [2501.00663](https://arxiv.org/abs/2501.00663) · NeurIPS 2025（Poster；已在 NeurIPS 2025 proceedings（Advances in NeurIPS vol. 38）和 OpenReview 核实。arXiv v1 于 2024-12-31 提交，本笔记依据 v1） · 主题：02_test_time_training · 层级：milestone
> 代码：none（官方）。`source/MainText/Conclusion.tex` 原文是 "implemented in Pytorch and JAX … intend to make the code … available soon"，截至本笔记没有找到官方仓库。社区复现有 https://github.com/lucidrains/titans-pytorch，README 自称 unofficial · 笔记依据：`source/*.tex`（数字/公式）+ `paper.pdf`（图）

## 一句话

把 RNN 的 hidden state 换成一个 MLP 的权重，作为 neural long-term memory（LMM）。测试时，模型对每个 token 用 associative memory loss $\|\mathcal M(\mathbf k_t)-\mathbf v_t\|^2$ 做梯度下降来写入记忆，这个梯度下降带 momentum（past surprise）和 weight decay（forget gate）。LMM 再和充当 short-term memory 的 attention、以及 persistent memory 组合，得到 Titans（MAC / MAG / MAL 三种接法）。

## 问题设定

- **要解决的问题**：长序列建模。Transformer 的依赖建模精确，但代价是 $O(N^2)$，context 长度也是固定的。linear Transformer 和 linear RNN/SSM 把历史压进固定大小的向量或矩阵 state，上下文一长就压缩不下（`MainText/Intro.tex`）；加法写入会导致 "memory overflow"，性能明显下降（`MainText/Background.tex`）。
- **memory 视角**：作者把所有序列模型统一成两步，write $\mathcal M_t=f(\mathcal M_{t-1},x_t)$ 和 read $y_t=g(\mathcal M_t,x_t)$（`MainText/Background.tex` Eq.6–7）。linear attention 就是加法写入 $\mathcal M_t=\mathcal M_{t-1}+K_t^\top V_t$（Eq.4）。由此提出五个问题：memory 用什么结构（Q1）、怎么更新（Q2）、怎么读取（Q3）、多个 memory 模块怎么组合（Q4）、是否需要 deep memory（Q5）。
- **输入/输出**：token 序列 $x\in\mathbb R^{N\times d_{in}}$，主任务是下一 token 预测；另外测了时间序列预测和 DNA 分类。不涉及 RL 或环境交互。
- **假设**：slow weights 在预训练（outer loop）中学好，测试时冻结。slow weights 包括 q/k/v 投影、attention、门控、persistent memory。测试时只有 memory MLP 的权重在**单条序列内部**被更新（inner loop），用的是无标签的自监督 key→value 关联信号（associative memory loss，Eq.12）。
- **之前卡在哪**：
  1. linear attention 用 Hebbian 式加法写入，写进去的东西擦不掉。
  2. DeltaNet 能按 key 替换旧值，但没有整体遗忘；Mamba2 有 gating，却不能删除某条记忆（`MainText/Experiments.tex` §5.3）。
  3. [TTT layers](../ttt-layers/notes.md) 已经把 hidden state 当成模型权重、用梯度更新，但没有 forget 机制，只用 momentary surprise，deep memory 的效果也没有做实验（`Appendix/MAL.tex` "LMM is Generalized TTT Layer"）。
  4. 现有 hybrid 模型几乎都是把 recurrent 层和 attention 层逐层堆叠（即 MAL 式）。

## 核心方法

**1. Neural long-term memory（§3.1）。** 先做投影 $\mathbf k_t=x_tW_K,\ \mathbf v_t=x_tW_V,\ \mathbf q_t=x_tW_Q$（Eq.11, 15）。memory $\mathcal M$ 是 $L_\mathcal M\ge1$ 层的 MLP，它的参数就是 fast weights。写入目标是 associative memory loss，后续文献称之为 KV binding（`source/MainText/Methods.tex` Eq.12）：
$$\ell(\mathcal M_{t-1};x_t)=\big\|\mathcal M_{t-1}(\mathbf k_t)-\mathbf v_t\big\|_2^2$$
最朴素的写法是一步 GD，作者把梯度称为 "surprise"（Eq.8）：$\mathcal M_t=\mathcal M_{t-1}-\theta_t\nabla\ell(\mathcal M_{t-1};x_t)$。它的问题是：一次大 surprise 之后梯度迅速变小，紧随其后的相关 token 就记不住了。于是作者把 surprise 拆成 past surprise 和 momentary surprise 两部分，再加上遗忘，得到最终形式（Eq.13–14）：
$$\mathcal M_t=(1-\alpha_t)\,\mathcal M_{t-1}+S_t,\qquad S_t=\eta_t\,S_{t-1}-\theta_t\,\nabla\ell(\mathcal M_{t-1};x_t)$$
- $\theta_t$：本步写入多少 momentary surprise，相当于 data-dependent 的学习率。
- $\eta_t$：momentum 的衰减系数。$\eta_t\to0$ 时丢掉上一步的 surprise，对应上下文切换；$\eta_t\to1$ 时完全沿用。
- $\alpha_t\in[0,1]$：即 weight decay，起 forget gate 的作用；论文称它与 modern RNN 中的 gating 机制密切相关。$\alpha_t\to0$ 时不影响已有记忆，$\alpha_t\to1$ 时清空整个 memory。

三个系数都是 $x_t$ 的函数，实验里按 token 取值，但具体的参数化方式论文没有给出。整体上，这就是"带 momentum 和 weight decay 的 SGD"沿序列维度展开。读取时只做前向、不更新权重：$y_t=\mathcal M^*(\mathbf q_t)$（Eq.15）。另外要注意，正文说 surprise 是 "gradient with respect to the input"，但公式里求的是对 memory 参数的梯度，这里以公式为准。

**白板推导：线性 memory 会退化成什么。** 取 $\mathcal M=W$、loss 为 $\tfrac12\|W\mathbf k_t-\mathbf v_t\|^2$，则 $\nabla_W\ell=(W\mathbf k_t-\mathbf v_t)\mathbf k_t^\top$。
- 当 $\eta_t=\alpha_t=0$ 时，$W_t=W_{t-1}(I-\theta_t\mathbf k_t\mathbf k_t^\top)+\theta_t\mathbf v_t\mathbf k_t^\top$，正好是 delta rule，即 DeltaNet（`Appendix/MAL.tex` Eq.34）。
- 再加上 $\alpha_t$，就对应 Gated DeltaNet（论文原话是 $\eta_t=0$ 时两者等价）。作者据此说 LMM 推广了 Gated DeltaNet、Longhorn 和 TTT layer：
  - 相对 Gated DeltaNet 多出三样：momentum、deep memory，以及 chunk 之间的非线性递推（chunk 内仍是线性递推）；
  - 相对 Longhorn，除上面三样外还多一个 forget gate；
  - 相对 TTT layer 的三点区别是：forget 机制（weight decay）、momentum，以及对 deep memory 的实验评估（`Appendix/MAL.tex`）。
- 如果把 loss 换成 $-\langle W\mathbf k_t,\mathbf v_t\rangle$，梯度变成 $-\mathbf v_t\mathbf k_t^\top$，更新是 $W_t=W_{t-1}+\theta_t\mathbf v_t\mathbf k_t^\top$，也就是 linear attention 的 Hebbian 写入（Eq.4）。这一步是我补的，论文没写。它说明选择什么 inner loss，就等于选择什么写入规则。

**为什么要 deep memory**：矩阵 memory 等价于 online linear regression，隐含假设历史里的依赖是线性的；两层及以上的 MLP 表达力严格更强（§3.1 "Memory Architecture"）。

**2. 并行训练（§3.2，Figure 1）。** 逐 token 递推没法用 matmul 并行。作者沿用 TTT layers 的做法，把序列切成大小为 $b$ 的 chunk。chunk 内所有 token 的梯度都在 chunk 起点的权重 $\mathcal M_{t'}$ 处计算，其中 $t'=t-\mathrm{mod}(t,b)$（Eq.16）：
$$\mathcal M_t=\beta_t\mathcal M_0-\sum_{i=1}^{t}\theta_i\frac{\beta_t}{\beta_i}\nabla\ell(\mathcal M_{t'};x_i),\qquad \beta_i=\prod_{j=1}^{i}(1-\alpha_j)$$
- 线性 memory 在第一个 chunk 内，累计梯度可以写成 $\Theta_b\mathbf B_b(W_0X-X)X^\top$（Eq.17），全部是 matmul。
- momentum 项 $S_t=\eta_tS_{t-1}-\theta_tu_t$（Eq.18）里的 $u_t$ 可以一次性全部算出，所以它是一个输入依赖的线性递推，可以用 parallel associative scan 求解。
- 结果就是 Figure 1 所说的"chunk 内线性（cumsum），chunk 间非线性（经过梯度）"。

所以实际执行的是 **chunk 级 mini-batch 更新**，不是 Eq.13–14 那样的逐 token 更新；$b$ 越大越快，但离真正的 online 更新越远。论文没有报告 $b$ 的取值。作者还提出可以把 $\alpha,\theta,\eta$ 设成 chunk 级常数，这时 Eq.18 变成 LTI 系统，可以用 global convolution 计算；不过实验中用的仍是 token 级系数。

**3. Persistent memory（§3.3）。** 在序列最前面拼上 $N_p$ 个可学习、与输入无关的 token（Eq.19）。作者给了三个动机：
- 存储任务知识，测试时冻结；
- FFN 可以看作 data-independent 的 attention（Eq.20），persistent memory 起类似作用；
- 缓解 causal attention 对序列开头 token 的偏置。

**4. 三种接法（§4）。**
- **MAC（Memory as a Context，Figure 2、3a，Eq.21–25）**：先把序列切成 segment $\mathsf S^{(t)}$，每个 segment 依次做下面几步。
  1. 用当前 segment 去查历史：$h_t=\mathcal M^*_{t-1}(\mathbf q_t)$。
  2. 拼接并在 segment 内做 full causal attention：$\tilde{\mathsf S}^{(t)}=[p_1..p_{N_p}]\,\|\,h_t\,\|\,\mathsf S^{(t)}$，$y_t=\mathrm{Attn}(\tilde{\mathsf S}^{(t)})$。
  3. 用 attention 的输出而不是原始输入写入 memory：$\mathcal M_t=\mathcal M_{t-1}(y_t)$；输出 $o_t=y_t\otimes\mathcal M^*_t(y_t)$。

  这样 attention 既决定要不要用长时记忆，也筛选哪些信息值得写入。测试时 persistent memory 冻结，attention 负责 in-context learning，memory 继续学习。
- **MAG（Memory as a Gate，Figure 4、3b，Eq.26–28）**：不切 segment。一支是带 persistent prefix 的 SWA，另一支是直接用输入更新的 LMM，两支经 gating 组合：$o=y\otimes\mathcal M(\tilde x)$，论文说 $\otimes$ 可以是任意非线性 gating。实验中两支输出各自用可学习向量归一化，再过非线性 $\sigma$。作者把 SWA 看作精确的短时记忆，把 LMM 看作 fading memory。
- **MAL（Memory as a Layer，Figure 5，Eq.29–31）**：$y=\mathcal M(\tilde x)$，$o=\mathrm{SW\text{-}Attn}(y)$。骨架采用 H3 式结构，也就是常见的逐层 hybrid。
- **LMM**：只保留 neural memory，不用 attention。

**实现细节（§4.4）**：
- 所有块都有残差连接；
- q/k/v 用 SiLU 激活，q 和 k 做 $\ell_2$ norm；
- q/k/v 投影之后各接一个 1D depthwise-separable conv；
- 输出投影之前做 normalization 和线性门控。

§4.4 还给了一个定理：Titans 能解 TC$^0$ 以外的问题，而 Transformer、对角线性递推和 DeltaNet 都局限在 TC$^0$。v1 没有给出证明。

**训练与推理流程。**
- **Outer loop**（`MainText/Experiments.tex` "Training"）：AdamW，lr 4e-4，cosine schedule，batch 0.5M tokens，weight decay 0.1，LLaMA 2 tokenizer（32K 词表），训练长度 4K。170M/340M/400M 模型各训 15B FineWeb-Edu token，760M 训 30B。梯度穿过 inner loop 反传到 $W_K,W_V,W_Q$ 和门控参数，论文称 $W_K,W_V$ 是 inner loss 的 "hyperparameters"。
- **推理**：slow weights 冻结。每进来一个 chunk/segment，先读 memory 再写 memory，计算量随序列长度线性增长。
- $\mathcal M_0$ 怎么得到，论文没有写明。我推测是按 TTT layers 的惯例由 outer loop 学出的初始化。

## 关键实验与证据

**语言建模与常识推理**（Table 1）。每格为 Wiki ppl ↓ / 平均 acc ↑：

| 规模 / tokens | Transformer++ | TTT | Gated DeltaNet | GDN-H2$^*$ | LMM | MAC | MAG | MAL |
|---|---|---|---|---|---|---|---|---|
| 340M / 15B | 31.52 / 42.92 | 27.44 / 44.51 | 27.01 / 45.42 | — | 26.18 / 46.17 | 25.43 / 47.36 | 25.07 / 47.54 | 24.69 / 46.55 |
| 400M / 15B | 30.63 / 45.64 | 26.11 / 46.86 | 25.47 / 47.26 | 24.19 / 47.69 | 25.03 / 47.83 | 25.61 / 48.65 | 23.59 / 48.60 | 23.93 / 47.87 |
| 760M / 30B | 25.21 / 48.69 | 24.17 / 47.32 | 21.18 / 49.69 | 19.88 / 51.49 | 20.04 / 51.56 | 19.93 / 52.51 | 18.61 / 52.50 | 19.07 / 50.97 |

- 在非 hybrid 模型里，LMM 最好。
- 平均 acc 上 MAC 和 MAG 都优于 MAL，而三者用的是同一套模块。作者由此论证：组合方式本身就影响效果，而现有 hybrid 模型几乎都是 MAL 式。但 Wiki ppl 上并非如此：340M 和 400M 时 MAL（24.69、23.93）都比 MAC（25.43、25.61）低。
- LMM 相对 Gated DeltaNet 的优势不大：400M 时平均 acc 只高 0.57（47.83 vs 47.26）。

**S-NIAH**（RULER，Table 2，测试长度 2K–16K；主实验的训练长度是 4K）。这组数最直接地说明 forget、momentum 和 deep memory 的价值。16K 长度下：

| 任务 | TTT | Mamba2 | DeltaNet | LMM | MAC |
|---|---|---|---|---|---|
| S-NIAH-W | 0.0 | 0.0 | 0.0 | 80.6 | 95.2 |
| S-NIAH-N | 4.4 | 0.0 | 5.4 | 80.2 | 97.4 |
| S-NIAH-PK | 88.4 | 5.4 | 71.4 | 96.2 | 98.4 |

论文没有注明这组实验用的模型规模。

**BABILong**（Figure 6，只有图、没有表）。分 few-shot 和 fine-tune 两种设定，baseline 数字取自 BABILong 原论文。以下是读图估计：
- fine-tune 设定下，Titans (MAC)-FT 在 $10^6$ token 处约 94%，在 $10^7$ 处约 69%；RMT-FT 在 $10^6$ 处约 43%。
- 作者称 Titans 的参数量比 Llama3.1-8B+RAG 少约 ×70，效果却更好（§5.4）。
- 摘要里 ">2M context" 的结论就来自这张图。

**Deep memory**（Figure 7、8，在 Pile 子集上训练，只测 LMM）：
- 170M、360M、760M 三个规模。作者称 $L_\mathcal M$ 越大，所有长度上的 ppl 都更低；参数量较小时，更深的 memory 对序列长度也更稳健。读图看这一说法并不严格成立：360M 在 2K 长度处 $L_\mathcal M=4$ 的 ppl 反而最高（约 13.26，高于 $L_\mathcal M=1$ 的约 13.19）；170M 上 $L_\mathcal M=3$ 与 $L_\mathcal M=4$ 交错，4K–16K 处 $L_\mathcal M=3$ 略低。
- 代价是吞吐：读图约为 $L_\mathcal M=1$ 时 44K tokens/s，$L_\mathcal M=4$ 时约 23–24K tokens/s，硬件未说明。吞吐不随序列长度变化。

**消融**（Table 5，以 LMM 为基座；三列依次为 LM ppl / Reasoning acc / Long Context acc）：

| 变体 | LM ppl ↓ | Reasoning acc ↑ | Long Context acc ↑ |
|---|---|---|---|
| LMM | 27.01 | 47.83 | 92.68 |
| Linear Memory | 28.49 | 46.97 | 85.34 |
| w/o Convolution | 28.73 | 45.82 | 90.28 |
| w/o Momentum | 28.98 | 45.49 | 87.12 |
| w/o Weight Decay | 29.04 | 45.11 | 85.60 |
| w/o Persistent Memory | 27.63 | 46.35 | 92.49 |
| + Attn (MAC) | 26.67 | 48.65 | 97.95 |
| + Attn (MAG) | 25.70 | 48.60 | 96.70 |
| + Attn (MAL) | 25.91 | 47.87 | 96.91 |

- 作者的排序是 weight decay > momentum > conv > persistent。
- 但在长上下文这一列掉得最多的是 **Linear Memory**，说明 deep memory 主要帮的是长程召回；persistent memory 对长上下文几乎没有影响。
- 表里 LMM 行的 27.01 / 47.83，正好等于 Table 1 中 400M LMM 的 (Wiki 25.03 + LMB 28.99)/2 和平均 acc。三行 +Attn 也对得上：例如 MAC 的 26.67 = (25.61 + 27.73)/2，48.65 等于 400M MAC 的平均 acc。由此推测消融是在 400M 规模上做的，ppl 列是 Wiki 和 LMB 两项的均值；论文没有说明这一点。

**其他任务**：
- **时间序列**（Table 3）：在 Simba 框架里把 Mamba 换成 LMM。ETTm1 的 MSE 为 0.358，Simba 为 0.383。Traffic 的 MSE 0.415 是最好的，但 MAE 0.289 不如 iTransformer 的 0.282。
- **DNA**（Table 4）：Enhancer Cohn 75.2、Enhancer Ens 89.6 两项最好；Human Reg. 89.3，低于 HyenaDNA 的 93.8。
- **训练吞吐**（Figure 9）：LMM 略慢于 Mamba2 和 Gated DeltaNet。MAL 比 LMM 和 Gated DeltaNet 都快，论文归因于用 FlashAttention 实现 SWA。但读图 MAL（约 40–41K tokens/s）在所有长度上仍略慢于 Mamba2（约 42K），2K/4K 时也慢于 Transformer++。

## 局限与论文没说的

**作者承认的**：
- v1 是"第一版"，更大模型的结果留到后续版本（`MainText/Experiments.tex` 脚注）。
- memory 越深训练越慢（Figure 8）；MAC 效果好但比 MAL 慢（§5.9）。
- 只用了简单的 MLP 作为 memory；更好的 memory 架构和 chunk 级参数化留作未来工作。

**我读出来的**：
- **规模与评测口径**：
  - 最大只到 760M 参数、30B token，训练长度 4K。
  - 400M 的 baseline 数字直接取自 Gated DeltaNet 论文（`Appendix/Exp-details.tex`）；BABILong 和 DNA 的 baseline 也都引用他人结果，训练流程并不同源。
  - 没有多 seed，也没有方差。
- **超参数与算力缺失**：chunk 大小 $b$、segment 长度、$N_p$、$N_l$、SWA 窗口、主实验用的 $L_\mathcal M$、$\alpha/\theta/\eta$ 的参数化、NIAH 和 BABILong few-shot 用的模型规模都没有给。**完全没有报告算力**，吞吐测试的硬件也没有说明。
- **文本与表格不一致**：
  - Intro 说评测了 "recall-intensive" 任务，v1 里找不到这组实验；TC$^0$ 定理没有证明。
  - Table 1 中 760M 的 Titans 行缺少 hybrid 标记 $^*$，760M MAL 的 SIQA 为 30.98，明显异常。
  - Figure 7 标注 360M，与其他地方的 340M 不一致。
  - §5.6 说 "outperforming all baselines"，但 Traffic 的 MAE 不是最优。
  - §5.8 说 MAL "faster than baselines"，但 Figure 9 中 Mamba2 在所有长度上都比 MAL 快。
  - §5.9 说 MAC 和 MAG 都优于 MAL，但 Table 5 的 Long Context 列里 MAL（96.91）高于 MAG（96.70）。
- **"测试时学习"的边界**：测试时更新的只有单条序列内部的 fast weights，属于 in-context 的 TTT。它不同于 [Tent](../tent/notes.md) 或 [TTT-self-supervised](../ttt-self-supervised/notes.md) 那样在部署期更新模型参数，论文也没有讨论跨序列或跨 episode 保留记忆。
- **机制本身受到质疑**：[ttt-kv-linear-attention](../ttt-kv-linear-attention/notes.md) 认为，KV binding 类 TTT 在 inner loop 最后一层为线性且无 bias 的前提下等价于 linear attention。它说 Titans 满足这些前提、结论预计可以推广过来，但因为没有官方实现，没有在 Titans 上做实验验证。它还指出：加深 inner MLP 的收益在各文献里并不一致；在 Frobenius inner loss 下，per-token 学习率可以被 $v_t$ 吸收，因此是冗余的。注意 Titans 的 inner loss 是 MSE（Eq.12），不是 Frobenius 内积，所以"学习率冗余"这一条不能直接套到 Titans 上。
- **长上下文证据面窄**：
  - ">2M" 的结论只来自 BABILong 一个 benchmark 上的小模型 fine-tune，而且在 $10^7$ 长度下也掉到约 69%（读图）。
  - forget gate 会不会在超长距离下把 needle 衰减掉，论文没有分析。
  - 2M 长度下的推理时延和显存没有报告；Figure 9 只给了训练吞吐，最长只到 16K。
- **可能的失败模式**：
  - inner loop 的学习率由数据决定，遇到分布外 token 时可能产生过大的更新；
  - 干扰信息多的长序列可能覆盖旧记忆；
  - chunk 太大时，"所有梯度都在 chunk 起点求"这个近似会变差。
- **没测的 setting**：视觉/视频、多模态、RL、机器人控制、world model rollout 都没有涉及。

## 在脉络中的位置

- **前驱**：
  - fast weight programs（Schmidhuber 1992）以及 Hebbian / delta rule；
  - linear Transformer；
  - DeltaNet、Gated DeltaNet、Longhorn；
  - [TTT layers](../ttt-layers/notes.md)：Titans 沿用了它的"hidden state = 模型权重"设定、inner/outer loop 结构，以及 mini-batch 并行化方法；
  - MNM（Munkhdalai 2019）；
  - RMT：作者称 MAC 是 RMT 的推广；
  - persistent memory 来自 Sukhbaatar 2019；
  - 更早的 TTT 思想见 [TTT with self-supervision](../ttt-self-supervised/notes.md)。
- **后继 / 影响**：
  - [LaCT](../lact/notes.md)：它的 Momentum 变体写明 "Following Titans"，由每个 token 预测一个标量 momentum，再在 chunk 内取平均（`lact/source/supp.tex`）；主方法用的是 Muon。LaCT 不用显式 weight decay，改为对 fast weights 做 L2 weight normalization（`lact/source/sections/03_method.tex`）。
  - [TTT-E2E](../ttt-e2e/notes.md)：把 TTT-KVB（KV Binding）列为最相关的 prior work 路线，Titans 是其中一个变体。实验里对照的 TTT-KVB baseline 是 LaCT 式实现，论文只说 Titans "follow a similar construction"，没有直接和 Titans 比。
  - [ttt-kv-linear-attention](../ttt-kv-linear-attention/notes.md)：把 KV binding 类 TTT 重新解释为 linear attention，并提出上面提到的质疑。
  - [One-Minute Video Generation](../ttt-one-minute-video/notes.md)：在 related work 中把 Titans 列为"更大、更非线性的 hidden state"方向。
  - 同作者的 MIRAS、Nested Learning 在它的基础上继续发展（本项目未收录，见 `related_work/index.json`）。
  - 综述：[survey-test-time-intelligence](../survey-test-time-intelligence/notes.md)。
- **与本项目其他主题的联系**：
  - **World model**：[Dreamer](../../01_world_model/dreamer/notes.md) 和 [DreamerV3](../../01_world_model/dreamerv3/notes.md) 中 RSSM 的确定性状态 $h_t$，正是 Titans 所说的 vector-valued memory。[Genie](../../01_world_model/genie/notes.md)、[DIAMOND](../../01_world_model/diamond/notes.md)、[Dreamer 4](../../01_world_model/dreamer4/notes.md) 这类依赖 attention 或有限帧历史的 world model，长 rollout 时的一致性受上下文长度限制。MAC 式的"SWA 短时记忆 + 测试时可写入的长时记忆"是给 world model 补长程记忆的一个候选结构。论文完全没测这个方向，它是本项目可以探索的交叉点。
  - **Real2sim2real / 部署期适配**：[RMA](../../03_real2sim2real/rma/notes.md) 用 adaptation module，从近期的 state-action 历史估计环境 extrinsics，相当于隐式的 system identification。[PAD](../../05_intersections/pad/notes.md) 在部署期用自监督 loss 对策略编码器做梯度更新。Titans 介于两者之间：它像 PAD 一样在测试时用梯度写入 fast weights，但写入规则本身是 outer loop 端到端学出来的。$\alpha_t$ 和 $\eta_t$ 恰好对应动力学发生突变时"何时丢掉旧的估计"。
  - **RL**：部分可观测环境里，policy 和 value 需要长程记忆，LMM 可以代替 LSTM 作为 recurrent policy 的记忆模块。但这没有经过验证，而且 inner loop 的梯度叠加上 [PPO](../../04_rl/ppo/notes.md) 这类方法高方差的 outer 梯度，稳定性未知。

## 前置知识

1. linear attention 及其 recurrent 形式；理解 "hidden state = associative memory" 这个视角。
2. delta rule、DeltaNet 和 Gated DeltaNet；gating / forget gate，例如 Mamba2、GLA。
3. [TTT layers](../ttt-layers/notes.md)：fast weights、inner loop / outer loop、mini-batch TTT 的 matmul 并行化。
4. 带 momentum 和 weight decay 的 SGD，并学会把它看成沿序列维度的递推。
5. parallel associative scan、sliding window attention、FlashAttention。
6. 评测：perplexity、RULER S-NIAH、BABILong。
7. （可选）TC$^0$ 与 state tracking。

## 复现可行性

- **代码**：没有官方代码。社区版 lucidrains/titans-pytorch 实现了 `NeuralMemory` 和 `MemoryAsContextTransformer`，但作者表示会尝试原文之外的 memory 结构，可能与论文有出入。
- **数据**：FineWeb-Edu（取 15B/30B token）、Pile 子集、RULER S-NIAH、BABILong、ETT/ECL/Traffic/Weather、GenomicsBenchmarks，全部公开，不需要仿真器。
- **算力**：论文**没有报告**硬件和训练时长。我按 6ND 粗估（不是论文里的数字，也没算 inner loop 的开销）：340M × 15B 约 $3\times10^{19}$ FLOPs，760M × 30B 约 $1.4\times10^{20}$ FLOPs。
- **预计难点**：
  1. 上面列的那些超参数全都缺，只能自己扫；
  2. deep memory 要做 chunk 并行，还要对 inner loop 反向传播，写出高效 kernel 不容易；
  3. 没有官方实现可以用来核对正确性。
- **最小可复现实验**：
  1. 在合成的 S-NIAH 或 associative recall 任务上，用 2–4 层、d≈256 的小模型，训练长度 2K、测试 2K–16K，依次对比四个变体：线性 memory、+ $\alpha_t$、+ $\eta_t$、$L_\mathcal M=2$。目标是复现 Table 2 的定性趋势：没有 forget 的模型随长度增加而崩溃，LMM 基本保持。
  2. 用 titans-pytorch 在 FineWeb-Edu 的小子集上，比较 170M 以下的 LMM、Gated DeltaNet 和 Transformer++，只看 Wiki ppl 的相对排序，对照 Table 1 的 340M 那一行。

## 自测问题

1. 从 $\tfrac12\|W\mathbf k_t-\mathbf v_t\|^2$ 出发，推出线性 memory 的更新式。为什么 $\eta_t=\alpha_t=0$ 时它就是 delta rule？为什么把 loss 换成 $-\langle W\mathbf k_t,\mathbf v_t\rangle$ 就得到 linear attention？
2. 只用 momentary surprise（Eq.8）时，一次大 surprise 之后会出什么问题，$S_t$ 怎么缓解它？$\eta_t\to0$ 和 $\alpha_t\to1$ 在行为上有什么区别？
3. chunk 并行时，chunk 内的梯度在哪个权重处求值？为什么这会得到"chunk 内线性、chunk 间非线性"？chunk 大小 $b$ 变大有什么代价？
4. MAC 为什么用 attention 输出 $y_t$ 写入 memory，而不用原始输入？测试时 persistent memory、attention、LMM 三类参数分别处于什么状态？
5. Table 5 中，对长上下文影响最大和几乎没有影响的组件分别是哪个？结合 ttt-kv-linear-attention 的论点，你会怎样设计实验来检验 deep memory 是否必要？
