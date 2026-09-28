# Genie: Generative Interactive Environments

> arXiv [2402.15391](https://arxiv.org/abs/2402.15391) · ICML 2024（Oral） · 主题：01_world_model · 层级：foundational
> 代码：none（论文未提供官方代码；`main.tex` Broader Impact 声明不发布模型权重、训练数据及数据样例；社区的 JAX 复现 Jafar：https://github.com/flairox/jafar，非官方）· 笔记依据：`source/*.tex`（数字/公式）+ `paper.pdf`（图）

## 一句话

Genie 只用**没有动作标注**的互联网 2D 平台游戏视频（30k 小时），就训出一个 11B 参数、能逐帧控制的 world model。关键在于 **latent action model（LAM）**：它用一个只有 8 个码字的 VQ 码本当信息瓶颈，把"相邻两帧之间最主要的变化"压缩成离散的 latent action。推理时用户按下这 8 个"按键"之一，就能驱动模型生成下一帧。

## 问题设定

- **目标**：提出 *generative interactive environment*。给一张图（文生图、手绘草图、真实照片）作为 prompt，生成一个可以逐帧"玩"的环境。
- **输入/输出**：训练时只用视频帧 $\bm x_{1:T}$，不用动作、文本和奖励。推理时输入 prompt 帧 $x_1$ 加上用户每步选的离散动作 $a_t\in[0,|A|)$，输出下一帧 $\hat x_{t+1}$。
- **假设**：视频里存在一个"可控主体"，它的变化可以用极少的离散动作概括；相邻帧的变化主要由这个主体引起。环境是 16 帧、10 FPS 的短窗口，没有奖励，也没有终止信号。
- **前人卡在哪**（`main.tex` Table 1 与 §4）：
  - 经典 world model（[World Models](../world-models/notes.md)、[Dreamer](../dreamer/notes.md)、IRIS 等）需要 video + action 数据，只能用环境自己采的数据。
  - 视频生成模型只能用文本做 video-level 控制，做不到逐帧控制。
  - GAIA-1 和 [UniSim](../../05_intersections/unisim/notes.md) 需要文本和动作标注。
  - Playable Video Generation（PVG）虽然也用 latent action，但只在单个特定领域的数据上训练，不能靠 prompt 生成新环境。

## 核心方法

整体有三个组件，都基于 ST-transformer（Figure 3）。训练分两阶段：先训 video tokenizer 并冻结，再**并行联合训练** LAM（输入像素）和 dynamics model（输入 token）（`main.tex` §2.1）。

**1. ST-transformer**（Figure 4，`main.tex` §2）

每个 block 依次做三件事：
- spatial attention：在同一帧的 $1\times H\times W$ 个 token 之间做注意力；
- temporal attention：同一空间位置跨 $T$ 帧做注意力，带 causal mask；
- 一层 FFW。论文特意去掉了 spatial 层后面的 FFW，把参数省下来给其他部分，作者说这样效果"显著更好"，但没有给数字。

复杂度可以这样推（下面是我按论文"主导项随帧数线性增长"这句话做的推导）：
- 一个 block 的注意力开销约为 $O\big(T(HW)^2 + HW\,T^2\big)$，对比全时空注意力的 $O\big((THW)^2\big)$。
- 当 $HW\gg T$ 时，主导项 $T(HW)^2$ 随帧数线性增长。

**2. Video tokenizer（ST-ViViT）**（Figure 6）

这是一个 VQ-VAE，把视频编码成每帧一组离散 token：$\bm x_{1:T}\in\mathbb R^{T\times H\times W\times C}\mapsto \bm z_{1:T}\in\mathbb I^{T\times D}$。encoder 和 decoder 都是 causal ST-transformer，所以 $z_t$ 里含有 $\bm x_{1:t}$ 的信息。

论文只说用了 "standard VQ-VAE objective"，没有写出公式。标准形式（van den Oord 2017）是：

$$\mathcal L_{\text{VQ}}=\|x-\hat x\|_2^2+\|\mathrm{sg}[z_e]-e\|_2^2+\beta\|z_e-\mathrm{sg}[e]\|_2^2$$

配置：200M 参数，patch 4，码本 1024 个码字、每个 32 维（`main.tex` §3 Training Details、Table 7）。

有个细节可以推算出来（论文没有明说）：942B tokens ÷ (125k steps × batch 512) ≈ 14,720 ≈ 16 帧 × 920，即每序列约 1.47 万 token、每帧约 920 个（论文里的 token 数都是取整值，用 Table 10 的 750B ÷ (200k × 256) 算出来是 ≈14,650，量级一致）。920 = 40×23，一种可能的解释是 160×90 补到 92 行后按 patch 4 切分——这纯属推测，论文没给 padding 方式。这个量级和论文说的"视频可达 $O(10^4)$ tokens"对得上。

**3. Latent Action Model**（Figure 5，`main.tex` §2.1）

LAM 由 encoder、VQ 码本和 decoder 组成（下式按 §2.1 的文字整理，论文没有给显式公式；encoder 先输出连续的 latent action，再经 VQ 量化）：

$$\tilde a_t=\mathrm{VQ}\big(\mathrm{Enc}(\bm x_{1:t},x_{t+1})\big)\in\{e_1,\dots,e_{|A|}\},\qquad \hat x_{t+1}=\mathrm{Dec}(\bm x_{1:t},\tilde{\bm a}_{1:t})$$

它用 VQ-VAE 目标训练，重建的是像素 $x_{t+1}$。输入归一化到 $[0,1]$，decoder 输出过 sigmoid（Appendix C.1）。

为什么这样能学到"动作"：
- decoder 只能看到历史帧和 $\tilde a_t$，看不到未来帧，所以 $\tilde a_t$ 必须编码"过去到未来最有意义的变化"。
- $|A|=8$ 意味着每步只有 3 bit 的信息量，这个瓶颈逼着它只编码可控主体的运动，而不是整帧内容。
- 作者提到，增加码字数（动作数）有收益（原文只说 "benefit"，没说是哪项指标），代价是人和 AI agent 的可玩性下降（Appendix C.1，没有给数字）。

配置：300M 参数，patch 16，码本 8 个码字、每个 32 维，encoder 和 decoder 各 20 层、d_model 1024（Table 5）。

推理时 **LAM 只保留 VQ 码本**，其余部分全部丢弃。

**4. Dynamics model**（Figure 7，`main.tex` §2.1）

这是一个 decoder-only 的 MaskGIT ST-transformer。输入是 $\bm z_{1:T-1}$ 和 $\mathrm{sg}[\tilde{\bm a}_{1:T-1}]$，输出 $\hat{\bm z}_{2:T}$。

按 §2.1 的文字整理成公式如下（论文没有给显式公式）：

$$\mathcal L_{\text{dyn}}=\sum_{t=2}^{T}\mathrm{CE}\big(\hat z_t,\;z_t\big),\quad \hat z_t=f_\theta\big(\bm m\odot\bm z_{1:t-1},\ \mathrm{sg}[\tilde{\bm a}_{1:t-1}]\big)$$

其中：
- 只对 $z_{2:T-1}$ 做 Bernoulli mask，mask 率 $r\sim U(0.5,1)$；prompt 帧 $z_1$ 从不被 mask。
- 论文没有写 CE 是否只在被 mask 的位置上计算。

三个值得注意的设计：
- **stop-gradient**：dynamics 的 loss 不会回传到 LAM，LAM 只靠自己的重建目标来塑形。
- **加性嵌入**：latent action 以 additive embedding 的方式注入（LAM 和 dynamics model 都这样做），而不是按训练 world model 的常见做法（IRIS、TWM）把 $t$ 时刻的动作拼接到对应帧上。作者说这样可控性更好（没有给数字）。
- **大模型训练稳定性**：用 bfloat16 和 QK norm。

**5. 推理**（Figure 8，`main.tex` §2.2）

1. $x_1$ 经 tokenizer 编码得到 $z_1$。
2. 用户选一个整数 $a_1$，用它去 VQ 码本里查出 $\tilde a_1$。
3. dynamics 用 25 步 MaskGIT 采样生成 $z_2$（temperature 2，random sampling）。
4. tokenizer decoder 把 $z_2$ 解码成 $\hat x_2$。
5. 按自回归方式重复上面几步。

如果把真实视频经 LAM 推断出的动作序列喂回去，可以"重演"原视频；换一组动作，就得到新的轨迹。

**6. 可控性指标**（`main.tex` §3 Metrics）

$$\Delta_t\text{PSNR}=\text{PSNR}(x_t,\hat x_t)-\text{PSNR}(x_t,\hat x_t')$$

其中 $\hat x_t$ 用从真实帧推断出的 latent action 生成，$\hat x_t'$ 用随机采样的 latent action 生成。所有实验都在 $t=4$ 报告这个指标，值越大说明动作的影响越大。

## 关键实验与证据

**数据**（§3，Appendix B.1）

- 按关键词筛出平台游戏视频，切成 16s 片段、10 FPS、160×90 分辨率：共 55M 片段，约 244k 小时。
- 然后用 ResNet18 质量分类器过滤：分类器有 11M 参数，训练数据是 10k 条人工标注，花了约 10 人时。过滤后剩 6.8M 片段、30k 小时。
- 过滤的效果（Table 4，580M 模型）：FVD 从 61.4 降到 54.8，数据量只有原来的略多于 10%（原文 "just over 10%"；6.8M/55M≈12%），结果反而更好。
- Robotics 数据：RT-1 的约 130k 条演示，加上一份 simulation 数据，再加 QT-Opt 的 209k 条真机 episode，全部只当视频用，不用动作。

**Scaling**（Figure 9，Table 10/11）

- 模型规模：dynamics 从 41M 扩到 2.7B，每个规模都训 200k steps、batch 256、750B tokens。最终 training loss 随规模单调下降。其中 2.7B 用了 256 TPUv3 训 16 天，计算量 $6.91\times10^{21}$ FLOPs。
- batch size：固定 2.3B 模型，batch 取 128/256/448（分别对应 1.9M/3.8M/6.6M tokens），loss 同样随 batch 增大而下降。
- 注意：这里比较的都是 training loss，不是 FVD。

**最终模型**（§3.1，Table 12）

- dynamics 10.1B 参数：48 层、d_model 5120，batch 512，训 125k steps，用 256 TPUv5p，计算量 $6.6\times10^{22}$ FLOPs。
- 加上 tokenizer 和 LAM，总计 10.7B 参数（论文统称 11B），训练用了 942B tokens。
- 论文没有报告训练用了多长时间。

**定性结果**

- 用 OOD prompt 能生成有游戏感的运动：Imagen2 生成图、草图、真实照片（Figure 10）。
- 能学出视差效果（Figure 12）。
- latent action 在不同起始帧上语义一致：Platformers 上对应 left/right/jump/no-op（Figure 17），Robotics 上对应 down/up/left（Figure 13）。
- 能模拟可形变物体，比如薯片袋（Figure 11，来自 Robotics 模型）。

**Robotics**

2.5B 模型，超参数沿用 Platformers 的设置，测试集 FVD 82.7（§3.2）。

**用来训练 agent**（Figure 15，Appendix E）

流程：
1. 用冻结的 Platformers LAM 给 CoinRun 专家视频打 latent action 标签，专家视频来自 R2D2 策略。
2. 训练 BC 策略 $\pi(a_t\mid x_t)$ 预测这些 latent action。
3. 用少量带真实动作的专家样本建一个字典 $D:a\mapsto\{u\}$，把 latent action 映射到真实动作，执行时 $u_t\sim D[a_t]$。

结果：只要约 200 个 expert samples，easy 和 hard 两档都能达到 oracle BC（直接用真实动作训练）的水平，且明显高于 random。评估口径是 100 个 samples 中通关关卡的平均百分比，5 个 seed，95% CI（Figure 15 caption）。

**消融**

Table 2（LAM 的输入用像素还是 token）：

| | 数据集 | 参数 | FVD↓ | ΔtPSNR↑ |
|---|---|---|---|---|
| Token-input | Platformers | 2.3B | 38.8 | 1.33 |
| Pixel-input | Platformers | 2.5B | 40.1 | 1.91 |
| Token-input | Robotics | 1B | 257.8 | 1.65 |
| Pixel-input | Robotics | 1B | 136.4 | 2.07 |

用像素输入时可控性都更好。作者的解释是 tokenization 丢掉了运动信息。

Table 3（tokenizer 架构，patch 10）：

| | 参数 | 显存 | FVD↓ | ΔtPSNR↑ |
|---|---|---|---|---|
| ViT | 230M | 0.3GB | 114.5 | 1.39 |
| C-ViViT | 225M | 1.6GB | 272.7 | 1.37 |
| ST-ViViT | 205M | 0.9GB | 81.4 | 1.66 |

C-ViViT 做全时空注意力，显存占用最大，结果却最差，作者认为是过拟合。

Tokenizer 的 batch size（Table 6）：batch 从 64 增到 384，重建 PSNR 从 35.7 升到 36.5。

## 局限与论文没说的

**作者承认的**（§5 与 Broader Impact）

- 自回归 transformer 会幻觉出不真实的未来。
- 只有 16 帧记忆，长时程一致性差。
- 推理速度约 1 FPS，达不到实时交互。
- 没有放出权重和数据，复现门槛高。

**我读出来的**

1. **主模型没有定量评测**。论文里标了参数量的 FVD 和 ΔtPSNR 都来自 ≤2.5B 的变体（Table 2、Table 4、Robotics 2.5B；Table 3 没注明 dynamics 规模），11B 主模型一个 FVD 都没报。FVD 也只在自家变体之间比较，没有和外部视频模型或 world model 对比。OOD prompt 泛化（照片、草图）只有定性展示。
2. **ΔtPSNR 衡量的是"对动作有多敏感"，不衡量"动作语义是否一致、是否正确"**。只在 $t=4$ 报告，是很短的时程。一个对随机动作输出噪声很大的模型，ΔtPSNR 也会很高。跨 prompt 的动作一致性只有 Figure 13/17 的图示。
3. **它本身并不是一个 RL 环境**。Genie 没有奖励和终止信号，论文也没有在 Genie 里做任何 imagination / latent rollout 来训练 policy。所谓"训练 agent"，实际上是在 CoinRun 上用 LAM 标签做 BC。CoinRun 正好也是 2D 平台游戏，和训练数据同一类型，所以对"跨域"的说服力有限。latent 到真实动作的映射仍然需要专家动作标签，而且字典查表不看观测，只适用于离散动作。
4. **8 个离散动作的瓶颈**默认场景里"只有一个可控主体"。放到机器人的连续高维动作上，表达力明显不够。镜头运动、其他角色、环境本身的随机性，也可能被 latent action 吸收，和主体动作混在一起。Robotics 部分只有 FVD / ΔtPSNR（§3.2、Table 2）和定性图，没有做规划或 policy 评测；[V-JEPA 2](../vjepa2/notes.md) 正是拿这一点批评这类模型。
5. **口径上有些小问题**：
   - 11B 和 10.7B 两种说法并存；
   - 硬件一处写 TPUv5p，另一处写 TPUv5；
   - Table 2 中 Platformers 那组参数没有对齐（2.3B 对 2.5B）；
   - tokenizer 消融用的是 patch 10，和主模型的 patch 4 不一致；
   - 分辨率只有 160×90。
6. **算力门槛**：$6.6\times10^{22}$ FLOPs，外加一个无法复现的私有数据集。

## 在脉络中的位置

**前驱**

- 按动作条件预测下一帧的 world model：[World Models](../world-models/notes.md)、[Dreamer](../dreamer/notes.md)、IRIS / TWM。
- 视频 token 化与 MaskGIT 生成：VQ-VAE、MaskGIT、Phenaki（C-ViViT）、MaskViT、TECO；ST-transformer 来自 Xu et al. 2020。
- latent action 这条线：PVG、ILPO（Edwards 2019）、LAPO（Schmidt & Jiang 2024）；VPT 则是用有标注数据训 IDM，再给大规模视频打标签。

**后继 / 影响**

- Genie 2/3：只发了博客，不在本项目里。
- [DIAMOND](../diamond/notes.md)：同期工作，用 diffusion，并在 world model 里真正训练了 RL agent。
- [Cosmos](../cosmos-wfm/notes.md)：把 Genie 归入自回归视觉 world model。
- [Dreamer 4](../dreamer4/notes.md)：大部分知识来自无标签视频，只用少量动作配对数据学动作条件，并且真的在 world model 里训练 agent。这补上了 Genie 没做的那一半。
- [V-JEPA 2](../vjepa2/notes.md)：批评这条线只生成"看起来合理"的视频，没有实际用来控制机器人。
- [SimDist](../../05_intersections/simdist/notes.md)：以 Genie 为例子，反对像素级重建目标。
- [survey](../survey-wm-robot-learning/notes.md) 里的 DreamGen 等工作，用 latent action 或 IDM 从生成视频中恢复动作。注意：survey 里的 "Genie Envisioner" 是另一篇工作，不是 Genie。

**与本项目其他主题的联系**

- **RL**：Genie 提出"用生成模型造环境，缓解 RL 环境不够丰富的问题"，但没有真正闭环。[WMPO](../../05_intersections/wmpo/notes.md)、[World-Gymnast](../../05_intersections/world-gymnast/notes.md)、[Ctrl-World](../../05_intersections/ctrl-world/notes.md)、[RWM](../../05_intersections/robotic-world-model/notes.md) 用真实动作空间和奖励信号，在 world model 里做 policy optimization，可以作为对照。LAM + BC 属于 imitation from observation。
- **TTT**：Genie 适应新 prompt 完全靠 in-context 条件化，不更新参数。LAM 的重建误差本身是一个天然的 self-supervised 目标，可以拿来当部署时的 TTT 辅助任务，参考 [TTT](../../02_test_time_training/ttt-self-supervised/notes.md)、[PAD](../../05_intersections/pad/notes.md)。16 帧记忆的瓶颈，则可以对照 [TTT layers](../../02_test_time_training/ttt-layers/notes.md)、[One-Minute Video](../../02_test_time_training/ttt-one-minute-video/notes.md) 里用 fast weights 扩展长上下文的做法。
- **real2sim2real**："一张照片变成可玩环境"只能算最松散意义上的 real-to-sim：没有显式几何，没有物理参数，也不保证物理一致。显式重建的 digital twin 走的是另一条路，例如 [RialTo](../../03_real2sim2real/rialto/notes.md)、[SplatSim](../../03_real2sim2real/splatsim/notes.md)、[PhysTwin](../../03_real2sim2real/phystwin/notes.md)（可形变物体用 physics-informed 方法建模，可以和 Figure 11 的薯片袋对比）。对一个学出来的模拟器来说，sim-to-real gap 具体表现为幻觉和漂移，参见 [Reality Gap survey](../../03_real2sim2real/survey-reality-gap/notes.md) 的开放问题。

## 前置知识

1. **VQ-VAE**：码本、straight-through 梯度估计、commitment loss，以及用码本大小当信息瓶颈的思路。
2. **ViT / patch 化**，以及 factorized 时空注意力（ST-transformer）的复杂度分析。
3. **MaskGIT**：双向 masked token 预测，加上迭代并行解码和温度采样。
4. **自回归视频生成**的 tokenize-then-predict 范式（Phenaki / MaskViT）。
5. **动作条件 world model** 的基本形式 $p(o_{t+1}\mid o_{\le t},a_t)$，参见 [World Models](../world-models/notes.md)。
6. **inverse dynamics / imitation from observation**：BCO、ILPO、VPT，以及 behavioral cloning。
7. **评测指标** FVD、PSNR；Procgen CoinRun 环境。

## 复现可行性

- **代码**：官方不开源，也不放权重和数据。社区有 JAX 复现 [Jafar](https://github.com/flairox/jafar)，README 称复现了 Appendix F 的 CoinRun case study，单张 L40S 不到一周可以跑完。
- **全规模复现不可行**：数据集是私有的 30k 小时游戏视频，dynamics 需要 256 TPUv5p、$6.6\times10^{22}$ FLOPs。

**论文给出的小规模方案**（Appendix F）

- 规模：单张中档 TPU/GPU，一周以内。
- 数据：Procgen CoinRun hard 模式，用随机策略采集，不做 action repeat；level seed 取 0 到 10,000，每个关卡采 1000 步，共 10M transitions。
- tokenizer（Table 15）：8 层、d_model 512，batch 48×16 帧，16G 显存，训 3 天完成 300k steps。
- LAM（Table 16）：6 个码字。
- dynamics（Table 17）：12 层、d_model 512，采样温度 1.0，25 步 MaskGIT。
- LAM 和 dynamics 用 batch 36×16 联合训练 200k steps，优化器配置见 Table 9。

**最小实验建议**

照 Appendix F 跑通之后，补一个论文没做的定量检验。采数据时记下真实动作（不参与训练），然后在 held-out 关卡上：
1. 算 latent code 和真实动作之间的混淆矩阵 / NMI，把"动作一致性"从图示变成数字；
2. 报告 $t=4$ 时的 ΔtPSNR；
3. 做三组消融：
   - $|A|\in\{2,6,16,64\}$，检验"码字越多有收益、但可玩性下降"（Appendix C.1）；
   - LAM 用像素输入还是 token 输入（复现 Table 2 的趋势）；
   - 动作用加性嵌入还是拼接。

难度 medium。

**预计难点**

- LAM 的 VQ 码本可能坍缩或利用率低。这是 VQ 的常见问题，论文没有讨论。
- 论文没说 160×90 的 padding 方式、MaskGIT 的 mask schedule、dynamics CE 是否只在被 mask 位置计算，需要自己选。
- CoinRun 的 case study 是用 random policy 采的数据，和互联网视频的分布差很远，得到的结论不一定能迁移到大规模设置。

## 自测问题

1. 为什么 LAM 的 decoder 只能看 $\bm x_{1:t}$ 和 $\tilde{\bm a}_{1:t}$？如果把 $|A|$ 从 8 调到 1024，$\tilde a_t$ 会退化成什么？这对"可玩性"意味着什么？
2. dynamics model 对 latent action 用了 stop-gradient。如果去掉它，dynamics 的交叉熵 loss 会怎样反过来塑造 LAM，可能出现什么捷径？
3. 写出 ST-transformer 一个 block 的注意力复杂度，并与全时空注意力比较。结合 Table 3 解释：为什么 C-ViViT 显存更大，FVD 反而更差？
4. ΔtPSNR 能衡量什么、不能衡量什么？构造一个 ΔtPSNR 很高、但 latent action 在不同 prompt 下语义不一致的情形。
5. 在 CoinRun BC 实验里，作者用"latent 到真实动作的映射不含当前观测信息"来论证 latent action 是一致的。这个论证依赖什么前提？换成连续动作的机器人，这套流程需要改哪一步？
