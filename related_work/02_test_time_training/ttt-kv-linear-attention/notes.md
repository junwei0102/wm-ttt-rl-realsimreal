# Test-Time Training with KV Binding Is Secretly Linear Attention

> arXiv [2602.21204](https://arxiv.org/abs/2602.21204) · ICML 2026（`source/main.tex` 使用 `\usepackage[accepted]{icml2026}`）, 2026 · 主题：02_test_time_training · 层级：frontier
> 代码：https://github.com/nv-tlabs/tttla （入口页；实验代码在 https://github.com/JunchenLiu77/LaCT/tree/tttla 〔LLM、NVS〕 与 https://github.com/JunchenLiu77/ViTTT/tree/tttla 〔ImageNet〕，均已打开核实；论文本身只给出项目页 https://research.nvidia.com/labs/sil/projects/tttla/ ）· 笔记依据：`source/*.tex`（数字/公式）+ `paper.pdf`（图）

## 一句话

对于最后一层是线性、无 bias 的 TTT-KVB 层（inner loop 以 key 为输入、以 value 为目标做绑定损失的 TTT），最后一层权重每走一步梯度下降，都只是加上一个外积 $\phi(k)^\top g$。因此把 inner loop 展开后，输出总能写成 $o_t=\hat q_t\thinspace(S_0+\sum_i\hat k_i^\top\hat v_i)$ 这种 learned linear attention 的形式。作者据此认为，TTT 并不是"测试时记忆"，而是一种带可学习 kernel 的 linear attention，并在这个视角下化简模型、把计算改成并行形式。

## 问题设定

- **研究对象**：TTT 作为序列建模层，用来替代 softmax attention，而且只看 **TTT-KVB** 这一支（术语取自 [TTT-E2E](../ttt-e2e/notes.md)）。每层维护 fast weights $f_\theta$，对每个 token 或 chunk 以 $k$ 为输入、 $v$ 为目标，做一步在线 GD，损失可以是 MSE，也可以是 Frobenius 内积（dot-product loss）；更新之后，把 $q$ 送进 $f_\theta$ 得到输出（`sections/3.preliminary.tex`）。inner loop 反传任务损失的 TTT-E2E 不在讨论范围内，[用于分布偏移的 TTT](../ttt-self-supervised/notes.md) 也不在。
- **主流解释**：storage-and-retrieval。inner loop 把 KV 映射"记住"，query 负责"检索"。沿着这个叙事，后续设计越来越重：更深的 inner MLP、Muon 式优化器、momentum、weight normalization、per-token learning rate（LaCT、ViTTT、Titans、Atlas）。
- **此前的空白**：[TTT layers](../ttt-layers/notes.md) 只证明了一个特例：单层线性 fast weight、零初始化时，TTT 等价于 linear attention。多层 MLP 或带 momentum 的 TTT 还在不在这个框架里，一直没有讲清楚，所以也说不清"加深 MLP、换优化器"到底改进了什么。另外，现有实现都是逐块递归执行的，训练吞吐受限。
- **实验载体**：官方 LaCT（760M LLM、114M NVS）和 ViTTT-B（90M，ImageNet-1K）（Appendix A，`sections/appendix.tex`）。

## 核心方法

**(1) 与"记忆"解释矛盾的四个现象**（`sections/4.observations.tex`）
- 推理时增加 inner-loop 步数：inner loss 降了，任务指标反而变差（Figure 1）。
- inner loop 的 GD 改成梯度上升，并从头重训：指标持平，有时略好（Table 1）。
- t-SNE 下 Q 与 K 明显分成两团；V 与 O 在 Block 1/4 也明显分开，Block 7 则相邻、部分重叠（Figure 2，LaCT-NVS 的 Block 1/4/7）。query 落在 key 的分布之外，这不像是在做"检索"。
- 计算 TTT 输出时用 K 代替 Q：几乎不掉点（Table 1）。

**(2) Theorem 5.1：单步线性化**（`sections/5.ttt_as_linear_attn.tex` §5.1；证明见 `appendix.tex` §B）
设 $f(x)=\phi(x;\Theta)\thinspace W$，其中 $W\in\mathbb R^{D_h\times D_{out}}$ 线性、无 bias。对 key $k$ 走一步 GD，同时更新 $(W,\Theta)$。按链式法则， $\nabla_{W_t}\mathcal L=\phi_t(k)^\top\thinspace\partial\mathcal L/\partial f_t(k)$，因此

$$W_{t+1}=W_t+\phi_t(k)^\top g_t(k),\qquad g_t(k)\triangleq-\eta\thinspace\frac{\partial\mathcal L}{\partial f_t(k)}$$

$$o=\phi_{t+1}(q)\big(W_t+\phi_t(k)^\top g_t(k)\big)=\hat q\thinspace(S_0+\hat k^\top\hat v),\quad \hat q=\phi_{t+1}(q),\ \hat k=\phi_t(k),\ \hat v=g_t(k),\ S_0=W_t .$$

整个定理只依赖一件事：线性层对自身权重的梯度一定是"输入 × 上游梯度"的秩 1 外积，而这恰好就是 linear attention 往状态里写入的方式。

**(3) Theorem 5.2：沿序列展开**。每个 token 走一步 GD，用归纳法得到
$$o_t=\phi_{t+1}(q_t)\Big(W_0+\sum_{i=0}^{t}\phi_i(k_i)^\top g_i(k_i)\Big).$$
注意第 $i$ 项用的是当时的 $\Theta_i$。如果 $\Theta$ 也在更新，kernel 本身就随历史变化（论文称之为 dynamic kernel）。

**(4) Theorem 5.3：加 momentum**。设 $\Delta_t=\nabla\mathcal L_t+\alpha_t\Delta_{t-1}$，定义累积系数 $\beta_i^j=\prod_{s=i+1}^{j}\alpha_s$（ $i=j$ 时取 1）。交换求和次序后，整体形式不变，只有 value 变成按 momentum 加权： $\hat v_i=m_i(k_i)=g_i(k_i)\sum_{j=i}^{t}\beta_i^j$（`appendix.tex` §D）。

**(5) 两个实例**
- **LaCT**（§5.3 Eq.1，Appendix E）：inner 网络是 SwiGLU， $f(x)=(\mathrm{silu}(xW_0)\odot xW_2)W_1$；损失为 $\mathcal L=-\langle f(k),v\rangle$，所以 $\partial\mathcal L/\partial f=-v$；另有 per-token lr $\eta_t$、momentum $\alpha_t$，以及 Muon 式正交化 $\mathcal M$。展开后得到
$$o_t=\phi_{t+1}(q_t)\Big(W_{1,0}+\sum_{i=0}^{t}\mathcal M\big(\phi_i(k_i)^\top m_i\big)\Big),\qquad m_i=v_i\sum_{j=i}^{t}\eta_j\beta_i^j .$$
- **ViTTT**（§5.4，Appendix F/G）：GLU 形式 $f(x)=\mathrm{silu}(xW_0)\odot(xW_1)$，取 $\phi(x)=\mathrm{silu}(xW_0)$，单步更新后得到 $o_t=\phi_{t+1}(q_t)\odot\big(q_t(W_{1,t}+\eta\thinspace k_t^\top(v_t\odot\phi_t(k_t)))\big)$（Appendix F 的写法；正文 §5.4 省略了 $\eta$ 和下标 $t$），即 $\phi$ 同时门控 value 和输出。ViTTT 里的 3×3 depthwise conv fast weight 则等价于一个 sliding-window linear attention。

**(6) 用这个视角解释 (1) 的四个现象**（§5.2）
- 多走 inner steps，相当于在推理时换了一个训练时没见过的 attention 算子，属于 train–test mismatch。
- 梯度上升只是让 $\hat v$ 变号，这个符号会被可学习的 value 投影吸收。
- $q$ 只进入 $\phi_{t+1}(\cdot)$， $k$ 则进入 $\phi_t(\cdot)$ 和 $g_t(\cdot)$，两者本来就不是相似度计算里对称的两端，所以分布不同很正常。
- 用 K 代替 Q 之后，有效的 query/key 分别是 $\phi_{t+1}(k)$ 和 $\phi_t(k)$，仍然不同，attention 不会塌缩。

**(7) 化简路径与并行形式**（`sections/6.practical_implications.tex`）
以下六步依次叠加：
1. 只更新最后一层。 $\Theta$ 固定后， $\phi$ 变成静态的可学习 kernel。
2. 去掉 weight norm。此时状态更新满足结合律。
3. 多层 MLP 换成单层线性，于是 $\hat q=q$、 $\hat k=k$。
4. 去掉 per-token lr。在 Frobenius 损失下，它可以被 $v_t$ 吸收。
5. 去掉 momentum，于是 $\hat v=v$。
6. 去掉梯度正交化，得到标准 linear attention： $o=q(W+\sum_i k_i^\top v_i)$。

从第 2 步起， $N$ 个长度为 $L$ 的 chunk 可以一次算完（Appendix H）：
$$\mathbb O=\Phi(\mathbb Q)W_{1,0}+\big((\Phi(\mathbb Q)\Phi(\mathbb K)^\top)\odot\mathcal C^{\uparrow L}\big)\mathbb V,\qquad \mathcal C_{ti}=\sum_{j=i}^{t}\eta_j\prod_{s=i+1}^{j}\alpha_s\ (t\ge i).$$
lr 和 momentum 都被折叠进一个下三角的衰减 mask。Appendix I 说明了两种会破坏结合律、因而无法这样并行的情况：更新 $\Theta$（会出现嵌套的 silu/silu'），以及 weight norm（ $\mathrm{Norm}(A+B)\ne\mathrm{Norm}(A)+\mathrm{Norm}(B)$）。

**训练与推理**：outer loop 端到端地学习 QKV 投影、kernel $\phi$ 的参数以及初始 fast weight $W_{1,0}$（后者在 LaCT 中是可学习的初始 fast weight，见 LaCT 源码 `supp.tex`）。对 V2–V6 这类变体，推理时既可用递归形式（状态大小恒定），也可用并行形式，两者数学上等价（Appendix H）；Baseline 与 V1 只能递归执行。

## 关键实验与证据

**实验设置**（Appendix A）
- LLM：760M 的 LaCT-LLM，FineWeb-Edu 100B tokens，batch 4/GPU × 8×A100，20K iter，约 56 h。在 Book-3 的 2.5B tokens 上评测 perplexity，基于 Flame 代码库。
- NVS：12 层、768 维，共 114M 参数。RealEstate10K，128×128，batch 128/GPU × 4×A100，20K iter，约 38 h。训练用 2 输入视图 + 6 目标视图，评测用 2 + 3，只用 MSE 训练，以 PSNR 评测。
- 分类：ViTTT-B，90M，ImageNet-1K，batch 256/GPU × 2×H100，60 epoch，约 16 h，报告 top-1。

**反例实验**（Table 1，`sections/4.observations.tex`）：列依次为 LaCT-LLM ppl↓ / LaCT-NVS PSNR↑ / ViTTT top-1↑

| 模型 | LLM ppl | NVS PSNR | ViTTT top-1 |
|---|---|---|---|
| Baseline | 16.43 | 25.94 | 79.34 |
| Gradient Ascent | 16.19 | 25.85 | 79.61 |
| Replace Q with K | 16.18 | 25.95 | 79.18 |

**Figure 1（目测读图）**：inner-loop 步数从 1 增加到 64。NVS 的 PSNR 从约 26 掉到约 17，inner loss 从约 0 降到约 −60；LLM 的 ppl 从约 15.2 升到约 15.75，inner loss 在约 8 步处降到最低（约 −1.4），之后略有回升。注意 1 步处的 LLM ppl（约 15.2）与 Table 1/2 的 Baseline（16.43）对不上，原文未说明 Figure 1 的评测设置。

**化简路径与吞吐**（Table 2）：列依次为 ppl / PSNR / top-1 / 递归吞吐 / 并行吞吐（TTT 层，LLM，单 batch）

| 变体 | ppl | PSNR | top-1 | 递归 tok/s | 并行 tok/s |
|---|---|---|---|---|---|
| Baseline | 16.43 | 25.94 | 79.34% | 4.30M | N/A |
| V1 只更新最后一层 | **15.93** | **25.97** | **79.63%** | 10.60M | N/A |
| V2 去 weight norm | 16.31 | 25.93 | **79.63%**† | 11.02M | 30.18M |
| V3 MLP→单层线性 | 16.23 | 25.71 | 79.39% | 12.95M | 49.69M |
| V4 去 per-token lr | 16.12 | 25.70 | 79.39%† | 13.31M | 53.99M |
| V5 去 momentum | 15.97 | 25.70† | 79.39%† | 14.40M | 57.28M |
| V6 去正交化（=linear attention） | 16.80 | 25.73 | 79.54%* | 89.67M | 124.6M |

（† 表示该步对这个模型不适用、数值沿用上一行；* 表示 ViTTT 没有正交化，这一步改为去掉 gradient normalization。）

**Table 2 的几点解读**
- 化简到只更新最后一层（V1）时，三个任务都是最好成绩。
- 深层 MLP 对 NVS 有帮助（V2→V3：25.93→25.71）；梯度正交化对 LLM 有帮助（V5→V6：15.97→16.80）。
- 并行形式对递归形式的加速比由笔者从表中算出：V4 为 53.99/13.31≈4.06，对应正文"up to 4.0×"。
- V6 的递归吞吐从 14.40M 跳到 89.67M，主要来自去掉了 Muon 正交化，与并行化无关。
- Figure 3 给出了逐位置的 ppl（Book-3 2.5B tokens）：V6 在整个序列长度上都是最高的一条曲线，其余变体挤在一起。Table 2 报告的是 32k 长度处的 ppl。
- Figure 4：Variant 2 的并行形式相对原版 LaCT 有 **1.19×** 的端到端训练加速，收敛情况相当（图中 Variant 2 递归形式为 1.06×）。

**两处口径不一致**
- 引言说完整 TTT 只比化简后的变体好 "+0.87 ppl / +0.24 dB"，这组数是 V1 对 V6（16.80−15.93、25.97−25.73）。
- §6.1 说"只有轻微退化（+0.4 ppl / −0.2 dB）"，那组数是 Baseline 对 V6。
- 两处的参照物不同，读的时候要分清。

## 局限与论文没说的

**作者承认的局限**（`sections/7.conclusion.tex`）
- 理论只覆盖最后一层线性、无 bias 的情形；非线性最后一层留作未来工作。
- 实验只做了 LaCT 和 ViTTT。Titans、Atlas 被认为"满足定理假设"，但因为没有官方实现，没有做实证。

**笔者的批判性阅读**
1. **梯度上升实验几乎是必然结果（笔者推导）**。LaCT 和 ViTTT 的损失 $-\langle f(k),v\rangle$ 对 $v$ 是线性的，所以以 $v$ 为目标做梯度上升，等价于以 $-v$ 为目标做梯度下降（论文 §4.2 也指出翻转梯度等价于把损失取负）。Muon 的 Newton–Schulz 迭代和梯度 L2 normalization 都是奇函数，不影响这个等价。因此只要 $v$ 是 $x$ 的线性投影、且只被 inner loop 当作目标使用，"上升模型"在参数取 $W_v$ 时和"下降模型"在参数取 $-W_v$ 时就是同一个函数族。（LaCT-LLM 的 TTT 分支与 SWA 共享 QKV，见第 3 条，此时 $-W_v$ 也会翻转 SWA 的 value，严格等价还需要额外条件。）重训结果持平是数学上预料之中的，并不能区分"记忆"和"linear attention"两种解释：记住 $-v$、再用取反的读出，照样可以叫记忆。真正有区分力的检验应该用 MSE 损失（TTT-MLP、Titans），而论文没有做。
2. **Figure 1 是推理时的干预**。模型训练时按 LaCT 默认每个 chunk 只做一次 update（本文未明说训练步数，Figure 1 横轴从 1 起），推理时多走几步，掉点可以直接用 train–test mismatch 解释，作者自己也是这么解释的。干净的对照应该是训练和推理都走 $n$ 步。另外，Frobenius 损失没有下界，"inner loss 更低"本来就不等于"拟合更准"。
3. **Q→K 实验与 t-SNE 缺乏对照**。论文没说 Q→K 实验是否重训。t-SNE 也没有给 softmax attention 或普通 linear attention 的同类图作对照。另据 [LaCT](../lact/notes.md) 源码 `sections/03_method.tex`：LaCT-LLM 的 TTT 分支与等宽的 sliding window attention 共享 QKV、输出相加；NVS 则只用输入视图做一轮 update。在这种设置下 TTT 分支分担的工作有多少，本身就不清楚。
4. **带 Muon 的 Eq.1 并不严格成立**。Appendix E 把 $\mathcal M$ 移进了求和号，写成"对每个外积各自正交化"。但 LaCT 实际是对累积的动量做正交化，而 $\mathcal M$ 是非线性的，这一步交换求和次序不严格。所以对原版 LaCT 来说，"linear attention"更多是一个类比；只有 V6 这一端是严格等价。
5. **一般情形下的"等价"接近重言式**。一旦 $\Theta$ 也在更新， $\phi_{t+1}(q)$ 本身就把全部历史非线性地编码了进去，写成" $\hat q\cdot$ 状态"的形式没有约束什么。真正有实质内容的结论集中在 V1 及之后的变体上。
6. **MSE 损失没有覆盖到**。在 MSE 损失下， $g_i$ 依赖于当前状态 $W_i$（即 delta rule），Step 4/5 的吸收论证以及 Appendix H 的并行式都不能直接套用。Titans 用的是 MSE 损失，外加 momentum 和 weight decay 形式的遗忘门（见 [Titans](../titans/notes.md) 源码 `Appendix/MAL.tex`），定理没有显式写出衰减项。
7. **评测口径偏窄**。只看 ppl、PSNR、top-1，没有测 NIAH、MQAR 这类真正考记忆的召回任务，而 storage 解释恰恰在这类任务上才可能有优势。原文没有报告 seed 数，也没有误差棒，0.2–0.5 ppl 的差异（包括 V1 优于 Baseline）可能落在噪声范围内。
8. **并行式的细节没交代清楚**。Appendix H 的式子是 $NL\times NL$ 的带 mask 二次形式，并不是正文说的 prefix scan。而且对角块 $\mathcal C_{tt}=\eta_t$ 经 $\otimes\mathbf 1_{L\times L}$ 展开后是取值恒为 $\eta_t$ 的满块（块内没有因果 mask），相当于 update-then-apply；LaCT-LLM 实际用的是 shifted block-wise causal mask（apply-then-update，见 LaCT 源码 `sections/03_method.tex`）。论文没有说明实现上如何对齐。吞吐也只报告了单 batch 的数字。
9. **其他小问题**。按 20K iter × 32 条序列算（假设没有梯度累积），共 64 万条序列；要真的训满 100B tokens，每条序列需要约 15.6 万 tokens（笔者推算）。论文没写训练序列长度，"100B tokens"很可能指的是采样池。另外 Step 5 写的 " $g_t(k)=-v$" 符号不严谨：按定义应为 $g_t=\eta v$。

## 在脉络中的位置

- **前驱**：
  - linear attention（Katharopoulos 2020）。
  - fast weight programmers / DeltaNet（Schlag 2021）。论文转述 Yang 2024a 的结论：DeltaNet = 单层线性 + MSE 的 TTT（注意 `main.bib` 中这条引用 `yang2024gated` 实际是 GLA 论文 "Gated linear attention transformers with hardware-efficient training"）。
  - [TTT layers](../ttt-layers/notes.md) 的 Theorem 1（线性 + batch GD + 零初始化 = linear attention），本文把它推广到多层 MLP 和 momentum。
  - 被分析的具体对象：[LaCT](../lact/notes.md)、ViTTT（Han 2025，不在本项目）、[Titans](../titans/notes.md)。
  - [TTT-E2E](../ttt-e2e/notes.md) 提供了 "TTT-KVB" 这一术语，也是本文刻意排除的另一支。
- **后继 / 影响**：本项目里暂时没有以它为基础的后续工作。它直接挑战的是 LaCT、Titans 以及 [One-Minute Video](../ttt-one-minute-video/notes.md)（TTT-MLP 做长视频生成）背后"深层 fast weight = 更强记忆"的动机。它的实际用处是两条：一是只更新最后一层 + 并行形式可以作为更便宜的默认配置；二是做 TTT 层的消融时，应当以 linear attention（V6）作为基线。
- **与本项目其他主题的联系**：
  - **World model**：长时程视频或交互式 world model（[Genie](../../01_world_model/genie/notes.md)、[Cosmos](../../01_world_model/cosmos-wfm/notes.md)、[Dreamer 4](../../01_world_model/dreamer4/notes.md)）需要状态恒定的长上下文层。本文提醒，TTT-KVB 层的"记忆"能力上限更接近 linear attention，而不是"随用随学"。
  - **Real2sim2real**：项目里的"TTT"有两层含义，要分开。一是部署时为应对分布偏移而更新参数（[TTT-SS](../ttt-self-supervised/notes.md)、[Tent](../tent/notes.md)、[PAD](../../05_intersections/pad/notes.md)）；二是作为序列层的 TTT。本文只否定后者的"学习/记忆"叙事。按本文的视角，把 TTT-KVB 层放进 policy 来"适应 sim-to-real gap"，本质上是 in-context 的历史混合，和 [RMA](../../03_real2sim2real/rma/notes.md) 用 adaptation module 从历史推断隐变量属于同一类，而不是真正的在线学习（笔者观点）。
  - **RL**：递归形式每步只需 $O(1)$ 状态，适合需要长历史的实时控制策略。与 [TTRL](../ttrl/notes.md) 这类"测试时做 RL"不是一回事，不要混淆。

## 前置知识

1. **linear attention 的递归形式与并行形式**：状态 $S_t=\sum k_i^\top v_i$、特征映射 $\phi$、带衰减 mask 的并行形式（RetNet、GLA 一类）。
2. **[TTT layers](../ttt-layers/notes.md)**：inner loop 与 outer loop、fast weights、mini-batch GD 的 dual form。
3. **反向传播基础**：线性层对权重的梯度是外积；MSE 损失与 dot-product 损失下的上游梯度分别是什么（delta rule vs Hebbian）。
4. **带 momentum 的 SGD 展开**：累积系数 $\beta_i^j$、交换求和次序。
5. **LaCT 的结构**：SwiGLU fast weight、大 chunk、apply/update 的执行顺序、与 SWA 的混合、weight norm；以及 Muon（Newton–Schulz 正交化）。
6. **t-SNE 的解读注意事项**；NVS 与 PSNR 指标。

## 复现可行性

- **代码**：已开源。入口是 `nv-tlabs/tttla`，LLM/NVS 在 `JunchenLiu77/LaCT` 的 `tttla` 分支，ImageNet 在 `JunchenLiu77/ViTTT` 的 `tttla` 分支。入口页 README 没写依赖、许可证，也没说明是否提供 checkpoint。LLM 部分基于 Flame 代码库（Appendix A）。
- **算力**（Appendix A）：
  - 每个 LLM 变体 8×A100 × 约 56 h；每个 NVS 变体 4×A100 × 约 38 h；每个 ViTTT 变体 2×H100 × 约 16 h。
  - 完整复现 Table 2（笔者推算；带 † 的格子不需要单独训练）：LLM 7 次训练，约 3.1k A100·h；NVS 6 次（V5 为 †），约 0.9k A100·h；ViTTT 4 次（V2/V4/V5 为 †），约 130 H100·h。Table 1 的梯度上升行按原文是从头重训的，三个任务各需重训一次；Q→K 行是否需要重训，原文未说明。
- **数据**：FineWeb-Edu、Book-3、RealEstate10K、ImageNet-1K。
- **最小可复现实验建议**：
  1. **CPU/numpy 验证理论**，约 1 小时。实现只更新 $W_1$ 的 SwiGLU inner loop（Frobenius 损失、per-chunk 的 $\eta,\alpha$），检查逐 chunk 递归的输出与 Appendix H 的 mask 并行式一致。然后分别加上 $W_0/W_2$ 更新、weight norm，确认等价被打破。再验证"对 $v$ 做梯度上升 ≡ 对 $-v$ 做梯度下降"。最后给 $\mathcal M$ 接上 Newton–Schulz，量化 Eq.1 交换求和带来的误差。
  2. **单卡小规模对照**：选 ViTTT 或 LaCT-NVS（128×128，缩短 iter），只训 Baseline、V1、V6 三行，每行多跑几个 seed，看 Table 2 中不到 0.3 的差距是否超出噪声。
  3. **补做论文缺的关键检验**：把 LaCT 的 inner 损失换成 MSE，重做梯度上升实验，并加测 NIAH。
- **预计难点**：LLM 规模的算力开销；apply/update 顺序和 chunk 因果 mask 的实现细节；单 seed 结论的可信度。

## 自测问题

1. 从"线性层梯度 = 输入 × 上游梯度"出发推导 Theorem 5.1。为什么 $\hat q$ 用的是 $\phi_{t+1}$，而 $\hat k$ 用的是 $\phi_t$？如果 $\Theta$ 也在更新，这个"linear attention"还剩多少约束力？
2. 展开带 momentum 的更新，推出 $m_i=g_i\sum_{j\ge i}\beta_i^j$ 与 Appendix H 的 $\mathcal C_{ti}$。在 LaCT 中加入 Muon 正交化之后，为什么这一展开不再严格成立？
3. 在 Frobenius 损失下，为什么梯度上升与梯度下降是同一个函数族？换成 MSE 损失时会发生什么？这对 Table 1 的证据力度意味着什么？
4. 化简路径中，从哪一步开始可以并行？weight norm 和 dynamic kernel 各自破坏了什么性质？
5. 在 Table 2 里，哪两个组件真正带来了收益、分别对哪个任务有效？V6 的递归吞吐为什么会从 14.40M 跳到 89.67M？
