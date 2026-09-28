# One-Minute Video Generation with Test-Time Training

> arXiv [2504.05298](https://arxiv.org/abs/2504.05298) · CVPR 2025 · 主题：02_test_time_training · 层级：milestone
> 代码：https://github.com/test-time-training/ttt-video-dit（MIT 许可，已确认仓库存在；内含训练/采样代码和基于 ThunderKittens 的 TTT-MLP kernel。仓库只提供 CogVideoX-5B 预训练权重的获取方式，**论文最终权重没有放出**。摘要里给的项目页 https://test-time-training.github.io/video-dit 在 2026-09 访问时返回 404） · 笔记依据：`source/*.tex`（数字/公式）+ `paper.pdf`（图）

## 一句话

在预训练的 CogVideo-X 5B（Diffusion Transformer）里，每个 attention 层后面插一个带 tanh 门控、双向扫描的 **TTT-MLP 层**。self-attention 只在每个 3 秒片段内部做（local attention），跨片段的全局记忆交给 TTT 层：它的隐状态是一个两层 MLP 的权重，更新方式是对自监督重建损失做梯度下降。模型分 5 个阶段做 context extension 微调，之后能从故事板一次生成 63 秒、多场景的 *Tom and Jerry* 视频，人评 Elo 平均比第二名高 34 分。

## 问题设定

- **任务**：长视频的 text-to-video。输入是 Format 3 故事板：每个 3 秒片段对应一段 3–5 句的描述，场景边界用 `<scene start>` / `<scene end>` 标出。用户也可以只写 5–8 句的剧情梗概（Format 1），由 Claude 3.7 Sonnet 按 1→2→3 的顺序扩写（`source/sec/3_method.tex` §3.2）。输出是 63 秒视频，训练数据的分辨率是 720×480。
- **难点**：用标准 tokenizer，一分钟视频要超过 300k token。63 秒视频如果用 full attention，推理耗时是 local attention 的 11×，训练是 12×（§1，Figure 6）。线性复杂度的 RNN 层（Mamba、DeltaNet 等 linear attention 变体）的隐状态只是一个矩阵。作者认为，把几十万个向量压进秩只有几千的矩阵本身就很难，并以此解释为什么已有的 RNN 视频（如 LinGen）只有单场景、慢动作。
- **假设 / 条件**：
  - 从预训练的 5B DiT 出发。CogVideo-X 5B 原本只能生成 16 fps 下 3 秒（或 8 fps 下 6 秒）的视频。
  - 刻意把领域限定在 *Tom and Jerry*：81 集，1940–1948 年，每集约 5 分钟，合计约 7 小时。先用 Real-ESRGAN 超分到 720×480。
  - 标注由人完成：人工切分场景，每个 3 秒片段写一段平均 98 词（132 token）的描述（§3.3 脚注）。
  - 数据侧重多场景、长程故事和剧烈运动，不看重真实感。
- **"test-time"的含义**：这里没有任何 DiT 权重在测试时被更新。所谓 test-time training，指的是 TTT 层在前向时对当前这条输入序列跑 inner loop。这和 [TTT (Sun 2019)](../ttt-self-supervised/notes.md)、[Tent](../tent/notes.md) 那种"为应对分布偏移在测试时更新模型参数"是两回事。本文沿用的是 [TTT layers](../ttt-layers/notes.md) 的"序列建模层"含义。

## 核心方法

**1. TTT layer 回顾**（`source/sec/2_prelim.tex`）。把隐状态当作模型 $f$ 的权重 $W$，每读一个 token 就做一步梯度下降（Eq.1）：

$$W_t = W_{t-1} - \eta\thinspace\nabla\ell(W_{t-1};x_t)$$

自监督损失是多视角重建，用三组投影（Eq.4、Eq.5）：

$$\ell(W;x_t)=\Vert f(\theta_K x_t;W)-\theta_V x_t\Vert^2,\qquad z_t=f(\theta_Q x_t;W_t)$$

- inner loop 只优化 $W$。 $\theta_K,\theta_V,\theta_Q$ 在 outer loop 里学，作用类似 attention 的 K/V/Q。
- 对 $\nabla\ell$ 反向传播就是 gradients of gradients。
- TTT-MLP 的 $f(x)=x+\texttt{LN}(f_\texttt{MLP}(x))$，其中 $f_\texttt{MLP}$ 的隐层宽度是输入的 4×，激活用 GELU（§2.3）。
- inner-loop 学习率 $\eta$：TTT-MLP 取 0.1，TTT-Linear 取 1.0（附录 A）。

**2. 插进预训练 DiT**（`source/sec/3_method.tex` §3.1）

- **Gating**（Eq.6）：
  $$\texttt{gate}(\texttt{TTT},X;\alpha)=\tanh(\alpha)\otimes\texttt{TTT}(X)+X,\quad \alpha\in\mathbb{R}^d$$
  $\alpha$ 初始化为 0.1，所以 $\tanh(\alpha)\approx0.1$。这样随机初始化的 TTT 层一开始不会明显覆盖 $X$（预训练表征），同时 TTT 仍能对输出有所贡献（论文称这是沿用 Flamingo 的 standard practice）。
- **Bi-direction**（Eq.7）：DiT 是 non-causal 的，所以再反向扫一遍：
  $$\texttt{TTT}'(X)=\texttt{rev}(\texttt{TTT}(\texttt{rev}(X)))$$
- **修改后的 block**（Eq.8–12）：原 block 是 $X'=\verb|self_attn|(\texttt{LN}(X))$、 $Y=X'+X$。改成：
  $$Z=\texttt{gate}(\texttt{TTT},X';\alpha),\quad Z'=\texttt{gate}(\texttt{TTT}',Z;\beta),\quad Y=Z'+X$$
  正向和反向共享 $\theta_K,\theta_V,\theta_Q$，门控参数 $\alpha$、 $\beta$ 分开。MLP block 不动。

**3. 序列组织：local attention + global TTT**（§3.2，Figure 3 右）

- 每个 3 秒片段做成一个 sequence segment：先放该段文字的 text token，后接该段的 video token。 $n$ 个 segment 首尾拼接，得到文字和视频交错的长序列。
- self-attention 只在每个 segment 内部做。受预处理影响，相邻 segment 重叠 1 个 latent frame，即 1350 token。
- TTT 层扫描整条序列。所以**跨片段的信息只能经过 TTT 层（外加那 1 帧重叠）传递**。

**4. Non-causal mini-batch TTT**（§3.4，Eq.13–14）。一次用 $b=64$ 个 token 的平均梯度更新 $W$：

$$W_{ib}=W_{(i-1)b}-\frac{\eta}{b}\sum_{t=(i-1)b+1}^{ib}\nabla\ell(W_{(i-1)b};x_t),\qquad z_t=f(W_{ib};x_t)$$

- 和因果版 [TTT layers](../ttt-layers/notes.md) 的关键区别：mini-batch 内**所有** token 都用该批结束时的 $W_{ib}$ 输出，也就是说 batch 内每个 token 都"看到"了同批后面的 token。non-causal 的扩散模型允许这样做。
- 作者还观察到，梯度取平均能降低方差、让更新更稳。
- Eq.14 原文写的是 $x_t$；按 Eq.5 的写法，输入应当经过 $\theta_Q$ 投影。

**5. On-chip Tensor Parallel**（§3.5，附录 B，Figure 4）

- 问题：TTT-MLP 的隐状态 $W^{(1)},W^{(2)}$ 加上激活，放不进单个 SM 的 SMEM。
- 做法：把每个 SM 当成一张"GPU"来做 Megatron 式切分。 $W^{(1)}$ 按列切， $W^{(2)}$ 按行切。GELU 是逐元素的，所以前向算 inner loss 只需要一次 reduction，这次 AllReduce 用 Hopper 的 DSMEM 在 SM 之间完成。
- 效果：隐状态只在开头加载、结尾写回时经过 HBM。
- 其他工程：用 ThunderKittens 实现；借用 FlashAttention-3 的 producer-consumer 异步预取；沿序列维度做 gradient checkpointing；用 TMA 做异步写回。

**6. 训练与推理**

- **训练**：5 阶段 context extension（附录 Table 2）。

  | 视频长度 | ctx 长度 | 训练哪些参数 | LR | steps |
  |---|---|---|---|---|
  | 3 s | 18048 | 全模型 | TTT 与 gate 取 1e-4（cosine），预训练参数取 1e-5（constant） | 5000 |
  | 9 s | 51456 | 只训 TTT、gate、local attention 的 QKVO | 1e-5 | 5000 |
  | 18 s | 99894 | 同上 | 1e-5 | 1000 |
  | 30 s | 168320 | 同上 | 1e-5 | 500 |
  | 63 s | 341550 | 同上 | 1e-5 | 250 |

  第 2 阶段起冻结其余参数，目的是少遗忘预训练的世界知识。长视频训练数据由相邻的 3 秒片段拼接而成。
- 其他训练配置（附录 A）：
  - 损失用 v-prediction，1000 步噪声日程，最后一步强制 Zero-SNR；
  - AdamW， $(\beta_1,\beta_2)=(0.9,0.95)$，batch size 64，grad clip 0.1，weight decay 1e-4；
  - text dropout 0.1，FSDP2 混合精度。
- **推理**：DDIM 50 步；动态 CFG，强度从 1 增到 4；使用 negative prompt。
- 我的推断（论文没有明说）：每个去噪步都要把整条约 34 万 token 的序列送进网络，每个 TTT 层在正、反两个方向上各跑一遍完整的 inner loop，按 $T/b$ 算约 5.3k 个 mini-batch。 $W$ 会不会跨序列保留、 $W_0$ 是否可学，本文都没写；[TTT layers](../ttt-layers/notes.md) 里 $W_0$ 是 outer loop 学的。

## 关键实验与证据

**设置**（§4.1–4.2）

- 6 种方法，全部是线性复杂度：
  - local attention：不改原结构；
  - TTT-Linear、TTT-MLP；
  - Mamba 2：隐状态约是 TTT-Linear 的 4×、TTT-MLP 的 1/2；
  - Gated DeltaNet；
  - sliding-window attention：窗口 8192 token，约 1.5 秒视频。
- 除 local attention 外，其余方法都用同样的 gate + bi-direction 方式（§3.1）插入同一个 CogVideo-X 5B，改后参数量都是 7.2B。所有方法都用同一套微调配方（§3.3、附录 A）。
- 评测：
  - 维度取自 MovieGen 的 4 个轴：text following、motion naturalness、aesthetics、temporal consistency。temporal consistency 同时看场景内和跨场景的一致性。
  - 方式是盲测两两比较：每次随机给一个轴和一对同剧情视频。剧情是 Claude 生成的 100 个，每种方法每个剧情各生成 1 个视频。
  - 评审从 prolific 招募，偏好用 Chatbot Arena 的 Elo 汇总。

**主结果：63 秒**（Table 1）

| | Text | Motion | Aesthetics | Temporal | Avg |
|---|---|---|---|---|---|
| Mamba 2 | 985 | 976 | 963 | 988 | 978 |
| Gated DeltaNet | 983 | 984 | 993 | 1004 | 991 |
| Sliding window | **1016** | 1000 | 1006 | 975 | 999 |
| TTT-MLP | 1014 | **1039** | **1037** | **1042** | **1033** |

- TTT-MLP 平均比第二名 SWA 高 34 分。temporal consistency 比第二名 +38，motion naturalness +39。Table 1 的 caption 把这两个轴写成 "scene consistency" 和 "motion smoothness"，和表头名称不一致。
- text following 上 TTT-MLP 略低于 SWA。
- Figure 5 的定性对比：SWA 会改变厨房环境、房子颜色，还把 Jerry 偷派的情节重复了一次；Gated DeltaNet 在不同角度下 Tom 的外观不一致。

**18 秒淘汰赛**（附录 Table 3，约 100k token）

| | Local attention | TTT-Linear | Mamba 2 | Gated DeltaNet | SWA | TTT-MLP |
|---|---|---|---|---|---|---|
| Avg Elo | 962 | 1001 | 1005 | **1032** | 993 | 1004 |

- Gated DeltaNet 最好，比 Mamba 2 高 27，比 TTT-MLP 高 28。
- 结论反转：短上下文下矩阵隐状态的 GDN 更好，到 63 秒时 TTT-MLP 反超 GDN 42 分（1033 对 991，我算的差值）。这组"交叉"是支持"表达力强的隐状态在更长上下文里更有用"的最直接证据；不过这是我的解读，论文 §4.4 只说短上下文（约 100k token）下 GDN、Mamba 2 这类矩阵隐状态仍最有效。

**效率**（Figure 6，63 秒，都相对 local attention）

| | 推理 | 训练 |
|---|---|---|
| full attention | 11× | 12× |
| TTT-MLP | 2.5× | 3.8× |
| Gated DeltaNet | 1.8× | 1.8× |

TTT-MLP 的推理和训练分别比 GDN 慢 1.4× 和 2.1×（§4.4）。

**消融**：论文**没有**针对组件的消融，gate 的初值、bi-direction、 $b$、片段长度、阶段数都没做。唯一接近消融的是 TTT-Linear 对 TTT-MLP，但只在 18 秒上比了，两者只差 3 分（1001 对 1004）。

## 局限与论文没说的

**作者承认的局限**

- 短上下文下不如 GDN（Table 3）。
- wall-clock 比 GDN、Mamba 2 慢。kernel 卡在 register spill 和异步指令排序上（§6）。
- 视频有瑕疵（Figure 7）。作者说这类瑕疵在所有方法里都常见，并推测可能源于 5B 底模能力有限（§4.4）：
  - 3 秒片段交界处物体会变形，作者猜测是扩散模型在不同片段采到了不同 mode；
  - 物体会不自然地漂浮（重力效应没建模好）；
  - 光照变化与动作不一致（除非 prompt 里明确写出），视差（parallax）这类复杂运镜有时不准。
- 受资源限制只做到一分钟。
- gate + bi-direction 只是众多集成方式之一，自回归 backbone 可能需要另外的集成策略（§6）。

**我读出来的问题**

1. **评测的统计力不明**：
   - 评审人数、比较次数、Elo 置信区间都没报。每个剧情每种方法只采 1 个样本，看不到 seed 方差。
   - 30 分量级的 Elo 差距是否显著，无法判断。
   - 没有任何自动指标，例如 FVD、VBench 或角色一致性指标。
2. **淘汰规则不自洽**：
   - TTT-Linear（1001）被淘汰，分数更低的 SWA（993）却留了下来。结果是 63 秒上从来没有直接比较 TTT-Linear 和 TTT-MLP。而这组对比恰恰最能单独检验"MLP 隐状态更 expressive"这一核心假设。
   - 在 18 秒上，Mamba 2（1005）也略高于 TTT-MLP。
3. **baseline 的公平性**：所有 RNN baseline 都套用同一套 gate + bi-direction 集成和同一套超参，而这套方案看起来是为 TTT 设计的。论文没有说明是否为 Mamba 2 或 GDN 单独调过参。
4. **只测了一个领域**：只有卡通。"能迁移到通用视频"是作者的信念，没有实验。*Tom and Jerry* 数据集没有放出（仓库的 dataset 文档只讲自建数据的格式），最终权重也没放出，所以 Table 1 无法原样复现。
5. **结构性失效模式**：attention 被严格限制在 3 秒片段内，片段交界处的连续性全靠 TTT 层和 1 帧重叠。Figure 7 里的边界变形，很可能正是这种设计造成的，而不只是底模的问题。
6. **生成方式是整段同时去噪**：一分钟视频在 50 个去噪步里整体生成，non-causal，既不能流式生成，也不接受 action 条件。所以它**不是**可交互的 world model，不能直接用于 RL rollout。
7. **算力门槛高**：训练折合 256 张 H100 跑 50 小时（§1）。kernel 依赖 Hopper 的 DSMEM，仓库明确说 TTT-MLP 训练只支持 H100。Figure 6 的时延没有写测试硬件。
8. **$b=64$ 没做消融**。后来的 [LaCT](../lact/notes.md) 指出，16 或 64 这类小 mini-batch 的 FLOPs 利用率常常不到 5%。

## 在脉络中的位置

- **前驱**：
  - [TTT layers](../ttt-layers/notes.md)（Sun et al. 2024）：直接来源，TTT-MLP、多视角重建损失、mini-batch TTT 都出自这里。本文把它从语言模型搬到 non-causal 的视频 DiT 上。
  - [TTT with self-supervision](../ttt-self-supervised/notes.md)：Yu Sun 最早的 TTT 思想。
  - 骨干与 baseline：CogVideoX / DiT，Mamba 2、Gated DeltaNet、Longformer 的 SWA。
  - 集成技巧：Flamingo 的 tanh gating；bi-direction（论文称之为 standard trick，引用 Mo & Tian 2024 的 bidirectional SSM 扩散 Mamba）；LLM 的分阶段 context extension（Xiong et al. 2023）。
  - 评测与工程：MovieGen 的评测轴，Chatbot Arena 的 Elo，FlashAttention-3 / ThunderKittens。
  - 更远的源头：Fast Weight Programmers。[Titans](../titans/notes.md) 被论文列为 recent work：隐状态更大、更非线性，并用更复杂的优化方法更新。
- **后继 / 影响**：
  - [LaCT](../lact/notes.md)：不用小 mini-batch，改用 2K–1M token 的大 chunk 做更新，扩展到 14B 的自回归视频扩散模型（56K token）。
  - [TTT-KV = linear attention](../ttt-kv-linear-attention/notes.md)：点名本文（Dalal et al., 2025）属于为提高"memorization"保真度而把设计做复杂的一类工作。它论证 KV binding 式 TTT 本质上是一种学到的 linear attention，直接挑战了本文"隐状态更 expressive"的解释。
  - [TTT-E2E](../ttt-e2e/notes.md)：inner loop 不再用 KV 重建，而是在测试时直接做 next-token prediction。
- **与本项目其他主题的联系**：
  - **World model**：[Genie](../../01_world_model/genie/notes.md)、[DIAMOND](../../01_world_model/diamond/notes.md)、[Cosmos](../../01_world_model/cosmos-wfm/notes.md)、[Dreamer 4](../../01_world_model/dreamer4/notes.md)、[Ctrl-World](../../05_intersections/ctrl-world/notes.md) 这类视频式 world model，都有长时程一致性和记忆的问题。本文给了一种配方：local attention 管片段内，一个 expressive 的 recurrent memory 管全局。它可以和 RSSM 的固定大小 GRU 状态（[DreamerV3](../../01_world_model/dreamerv3/notes.md)）、以及 Transformer 的 KV cache 对照。要真正当 world model 用，还得解决 action 条件和因果的自回归 rollout，作者自己也说自回归 backbone 可能需要别的集成方式。
  - **RL**：[WMPO](../../05_intersections/wmpo/notes.md)、[World-Gymnast](../../05_intersections/world-gymnast/notes.md) 在视频 world model 里用 RL 训练 VLA。rollout 越长，越需要跨片段一致的记忆。但 TTT-MLP 的推理是 local attention 的 2.5×，这个代价在 RL 的海量 rollout 里会被放大。
  - **TTT / real2sim2real**：本文的 $W$ 每条序列都重新训练，不会持续适应分布偏移。这和 [PAD](../../05_intersections/pad/notes.md) 那种部署时用 self-supervised auxiliary task 更新策略参数不同。一个可能的交叉点（论文未涉及、属于我的推测）：把 TTT 隐状态当作 in-context 的 system identification 载体，类似 [RMA](../../03_real2sim2real/rma/notes.md) 用历史推断环境参数。

## 前置知识

1. **Latent video diffusion 与 DiT**：v-prediction、DDIM、CFG、adaLN；VAE 如何把视频变成 token。
2. **self-attention 的二次复杂度**；local attention 和 sliding-window attention 的区别。
3. **Linear attention / modern RNN**：Mamba 2、DeltaNet 和 Gated DeltaNet 的矩阵隐状态及其 update rule。
4. **[TTT layers](../ttt-layers/notes.md)**：inner loop 与 outer loop， $\theta_K/\theta_V/\theta_Q$，mini-batch TTT，fast weights，gradients of gradients（meta-learning）。
5. **微调预训练大模型的技巧**：零附近初始化的门控（Flamingo），分阶段 context extension。
6. **GPU 存储层次**：HBM 与 SMEM、kernel fusion（FlashAttention）、Megatron 式 Tensor Parallel。读 §3.5 需要。
7. **Elo / Bradley–Terry 式人类偏好评测**。

## 复现可行性

- **代码**：官方仓库 MIT 许可，提供 `train.py`、`sample.py` 和 TTT-MLP kernel（子模块 `ttt-tk`）。构建 kernel 需要 CUDA 12.3+ 和 gcc 11+。README 写明 TTT-MLP 训练只支持 H100。代码里 TTT-MLP 层有 `use_kernel` 开关，关掉时走 `ttt/models/ssm/ops/ttt_mlp.py` 的 PyTorch 实现，但 README 没有把这条路径列为受支持的训练方式。
- **权重与数据**：只能拿到 CogVideoX-5B 预训练权重，最终权重没有放出。*Tom and Jerry* 数据没有放出。仓库文档只规定了自建数据的格式：一个 JSONL 文件，含 `scene_start` / `scene_end` 和文本；视频按 1 秒一块预先算好 VAE latent，文本预先算好 T5 embedding。论文说"annotations available"，但对应的项目页目前是 404。
- **算力**：整个训练折合 256 张 H100 跑 50 小时（§1）。论文没有报告各阶段分别用了多少算力。步数和 batch size 见 Table 2 和附录 A。另外还要算上人工标注故事板和人评的成本。
- **最小可复现实验**（建议）：
  1. **机制级 toy 实验**（1–8 卡，纯 PyTorch）：
     - 按 Eq.13–14（ $b=64$）实现 TTT-MLP，加上 bi-direction 和 tanh gate，插进一个小型 non-causal DiT 或 Transformer。
     - 数据用合成的多片段长序列：第 1 段设定一个物体的身份或颜色，后面第 $k$ 段必须复现。
     - attention 只在片段内做。比较 local only、+TTT-Linear、+TTT-MLP、+Mamba 2 / Gated DeltaNet（可用 `fla` 库）。
     - 随片段数增加，用自动一致性指标检验 Table 3 → Table 1 的"交叉"是否出现。**顺带补上论文缺的长上下文 TTT-Linear vs TTT-MLP 对比**。
  2. **官方仓库部分复现**（H100）：
     - 只跑 stage 1–2（3 s / 9 s，ctx 18048 / 51456），数据用自建的小规模动画片段。
     - 检查 $\alpha=0.1$ 的门控能否在前期保住预训练质量。
     - 测 TTT-MLP 相对 local attention 的时延，对照 Figure 6 的 2.5× / 3.8×。
- **预计难点**：
  - kernel 绑定 Hopper 的 DSMEM；
  - gradients of gradients 带来的显存压力；
  - 34 万 token 的序列需要 FSDP 加 checkpointing；
  - 故事板的人工标注很贵；
  - Elo 人评噪声大、成本高，没有现成的自动指标可以替代。

## 自测问题

1. 写出 TTT layer 的 update rule 和 output rule（Eq.1、4、5）。在 inner loop 和 outer loop 里分别更新什么？为什么说 $\theta_K,\theta_V,\theta_Q$ 是 inner-loop 损失的"超参数"？
2. non-causal mini-batch TTT（Eq.13–14）中，第 $i$ 批里的 token 用的是 $W_{ib}$。这和因果版 TTT layers 有什么不同？如果把它搬进逐帧自回归的 world model，会出什么问题？
3. gate 的 $\alpha$ 为什么初始化为 0.1，而不是 0？提示：当 $\tanh(\alpha)=0$ 时，TTT 参数能收到多少梯度？
4. 在这个架构里，第 1 个 3 秒片段的信息要经过哪些路径才能影响第 20 个片段？据此解释为什么"片段交界处物体变形"是可以预见的失效模式。
5. 18 秒时 GDN 比 TTT-MLP 高 28 分，63 秒时 TTT-MLP 比 GDN 高 42 分。这支持什么假设？要让这个结论站得住，还缺哪些实验和统计信息？
