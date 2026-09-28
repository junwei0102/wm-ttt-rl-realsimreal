# Mastering Diverse Domains through World Models

> arXiv [2301.04104](https://arxiv.org/abs/2301.04104) · Nature 640, 647–653 (2025)，Nature 版标题为 "Mastering diverse control tasks through world models"（本笔记依据 arXiv v2，2024-04） · 主题：01_world_model · 层级：milestone
> 代码：https://github.com/danijar/dreamerv3（JAX，MIT；论文正文只写 "available on the project website"，仓库存在已确认） · 笔记依据：`source/*.tex`（数字/公式）+ `paper.pdf`（图）

## 一句话

DreamerV3 仍是"RSSM world model + 在 latent imagination 中训练 actor-critic"的框架，关键贡献是一组让各项损失对信号量级不敏感的稳健化技巧：symlog、symexp twohot、KL balancing + free bits、1% unimix、percentile return normalization。有了它们，一套固定超参数就能跑通 8 个 domain、150 多个任务，并第一次在不用人类数据、不用 curriculum 的情况下，于 Minecraft 中从零拿到钻石。

## 问题设定

- **要解决的问题**：RL 算法换一个 domain 往往就要重新调参，比如 SAC 的熵系数（`sections/related.tex`），或者此前的 world model（论文引 DreamerV1）要按环境的视觉复杂度设定不同的 representation loss 权重（`sections/algorithm.tex`）。作者要的是"out of the box"：跨 domain 不改超参数。
- **输入/输出**：观测 $x_t$ 可以是图像（CNN）、向量（MLP），也可以两者都有；动作可离散也可连续；奖励是标量，稠密或稀疏都行，量级可以差好几个数量级。输出是随机策略 $\pi_\theta(a_t\mid s_t)$。
- **假设**：在线交互，数据收集与训练并发；只用环境 reward，没有 demonstration；允许部分可观测，由循环状态 $h_t$ 汇总历史；每个任务单独训练一个 agent（不做 multi-task）。
- **之前卡在哪**：(1) 用 MSE 回归大目标会发散，用 Huber/abs 回归又学得慢，用 running statistics 归一化会给优化带来非平稳性（`sections/algorithm.tex` "Robust predictions"）。(2) representation 正则的强度依赖场景：3D 场景要强正则，静态背景、像素级细节重要的游戏要弱正则。(3) 熵正则的合适尺度同时取决于 reward 的尺度和频率。advantage normalization 在稀疏奖励下会放大噪声，按 std 归一化在 std≈0 时会爆炸，而 SAC/MPO 那种约束式目标熵在稀疏奖励下探索慢（`sections/algorithm.tex` "Actor learning"）。

## 核心方法

三个网络：world model $\phi$、critic $\psi$、actor $\theta$，都用 replay 数据并发训练（Figure 3）。

**1. World model（RSSM）**，`sections/algorithm.tex` Eq.1：
$$h_t=f_\phi(h_{t-1},z_{t-1},a_{t-1}),\quad z_t\sim q_\phi(z_t\mid h_t,x_t),\quad \hat z_t\sim p_\phi(\hat z_t\mid h_t),$$
$$\hat r_t\sim p_\phi(\hat r_t\mid h_t,z_t),\quad \hat c_t\sim p_\phi(\hat c_t\mid h_t,z_t),\quad \hat x_t\sim p_\phi(\hat x_t\mid h_t,z_t)$$
- $z_t$ 是一组 categorical（softmax）分布的采样，梯度用 straight-through 回传。$c_t\in\{0,1\}$ 是 episode continuation flag。模型状态为 $s_t=\{h_t,z_t\}$。
- 损失（Eq.2–3），其中 $\beta_{\rm pred}=1,\ \beta_{\rm dyn}=1,\ \beta_{\rm rep}=0.1$：
$$\mathcal L(\phi)=\mathbb E_{q_\phi}\Big[\textstyle\sum_{t=1}^T\beta_{\rm pred}\mathcal L_{\rm pred}+\beta_{\rm dyn}\mathcal L_{\rm dyn}+\beta_{\rm rep}\mathcal L_{\rm rep}\Big]$$
$$\mathcal L_{\rm pred}=-\ln p_\phi(x_t\mid z_t,h_t)-\ln p_\phi(r_t\mid z_t,h_t)-\ln p_\phi(c_t\mid z_t,h_t)$$
$$\mathcal L_{\rm dyn}=\max\big(1,\ \mathrm{KL}[\mathrm{sg}(q_\phi(z_t\mid h_t,x_t))\,\|\,p_\phi(z_t\mid h_t)]\big),\quad \mathcal L_{\rm rep}=\max\big(1,\ \mathrm{KL}[q_\phi(z_t\mid h_t,x_t)\,\|\,\mathrm{sg}(p_\phi(z_t\mid h_t))]\big)$$
  白板上讲这三点：
  - **KL balancing**：同一个 KL 拆成两项，靠 stop-gradient 决定"谁向谁靠"。$\mathcal L_{\rm dyn}$ 让 prior（dynamics predictor）去追 posterior，权重 1；$\mathcal L_{\rm rep}$ 让 posterior 向 prior 靠、变得更可预测，权重只有 0.1，这样 representation 不会因为追求可预测而丢掉信息。
  - **free bits**：$\max(1,\cdot)$ 表示 KL 低于 1 nat 时该项取常数，梯度为 0。这样可以避免"dynamics 容易预测、representation 却没信息"的退化解，并把优化让给 $\mathcal L_{\rm pred}$。作者认为正是"free bits + 小 $\beta_{\rm rep}$"化解了上面说的"3D 场景要强正则、像素细节游戏要弱正则"的两难。
  - **1% unimix**：encoder、dynamics predictor 和 actor 的 categorical 都取 99% 网络输出加 1% 均匀分布，防止出现零概率和无穷大的 log 概率（`sections/methods.tex` "Distributions"）；对 encoder 和 dynamics predictor 而言，这让分布不可能变成确定性的，从而避免 KL 尖峰（`sections/algorithm.tex`）。
- 向量观测在输入 encoder 之前、以及作为 decoder 的目标时，都先做 symlog 变换（见下文 4）。

**2. Critic**（Eq.4–5）。先从 replay batch 中每个 posterior 状态出发，用 actor 加 dynamics predictor $\hat z$ 想象一条轨迹，得到 $s_{1:T},a_{1:T},r_{1:T},c_{1:T}$，然后算 bootstrapped λ-return：
$$R^\lambda_t=r_t+\gamma c_t\big((1-\lambda)v_t+\lambda R^\lambda_{t+1}\big),\quad R^\lambda_T=v_T,\qquad \mathcal L(\psi)=-\textstyle\sum_t\ln p_\psi(R^\lambda_t\mid s_t)$$
- 超参数：$\gamma=0.997$（discount horizon 333），$\lambda=0.95$，imagination horizon $H=15$（Table 4；正文写的是 $T=16$，两处不一致）。critic 输出分布，读出值为 $v_t=\mathbb E[v_\psi(\cdot\mid s_t)]$。
- 另外三个稳定手段：(a) critic 向自身参数的 EMA（decay 0.98）的输出做正则，作用类似 target network，但计算 return 时用的是当前 critic；(b) 在 replay 轨迹上也施加 critic loss（$\beta_{\rm repval}=0.3$），以想象得到的 $R^\lambda$ 作为起点处的 value annotation；(c) reward head 和 critic 的输出层权重初始化为 0，避免训练初期凭空"幻想"出大 reward 或大 value。

**3. Actor**（Eq.6–7）。离散和连续动作都用 REINFORCE，不像 DreamerV1 那样沿 dynamics 反传：
$$\mathcal L(\theta)=-\textstyle\sum_t\mathrm{sg}\!\Big(\frac{R^\lambda_t-v_\psi(s_t)}{\max(1,S)}\Big)\log\pi_\theta(a_t\mid s_t)+\eta\,\mathrm H[\pi_\theta(a_t\mid s_t)],\qquad S=\mathrm{EMA}\big(\mathrm{Per}(R^\lambda,95)-\mathrm{Per}(R^\lambda,5),\ 0.99\big)$$
- 被归一化的是 **return 的尺度**，不是 advantage。分母取 $\max(1,S)$，意思是只把大 return 缩小，小于 $L=1$ 的 return 保持原样，避免在稀疏奖励时放大近零的噪声。这样固定的 $\eta=3\times10^{-4}$ 就能自动做到"奖励稀疏时多探索、奖励够得着时多利用"。用 5–95 百分位而不是 min/max，是为了对随机环境中的离群 episode 稳健。
- 注意：Eq.6 字面上写的是 $+\eta\mathrm H$，按最小化理解会降低熵，和正文"用熵正则鼓励探索"的说法矛盾，应理解为 $-\eta\mathrm H$。这是我的判断，论文没有更正。

**4. Robust predictions**（Eq.8–12）
$$\mathrm{symlog}(x)=\mathrm{sign}(x)\ln(|x|+1),\quad \mathrm{symexp}(x)=\mathrm{sign}(x)(\exp|x|-1),\quad \mathcal L=\tfrac12\big(f(x,\theta)-\mathrm{symlog}(y)\big)^2,\ \hat y=\mathrm{symexp}(f)$$
- symlog 在原点附近近似恒等，对大值做对数压缩，而且保留符号，所以可以处理负 reward。
- **symexp twohot**（reward head 和 critic 使用）：网络输出一组 bin 上的 softmax，bin 为 $B=\mathrm{symexp}([-20,\dots,+20])$，读出 $\hat y=\mathrm{softmax}(f)^\top B$。训练目标 $\mathrm{twohot}(y)$ 只在相邻两个 bin $k,k+1$ 上按线性插值分配权重（Eq.12），损失为 $-\mathrm{twohot}(y)^\top\log\mathrm{softmax}(f)$。关键在于：对 logits 的梯度是 $\mathrm{softmax}(f)-\mathrm{twohot}(y)$，每个分量都有界，与 $|y|$ 的大小无关。这个梯度是我自己推的；论文的原话是 "decoupling the size of the gradients from the size of the targets"。bin 的个数论文没有给出。

**5. 工程配方**（`sections/methods.tex`，Table 4）
- 网络：block-diagonal GRU（8 blocks）、RMSNorm + SiLU。
- 优化：AGC（clip 阈值为权重范数的 30%），加 LaProp（$\epsilon=10^{-20}$，先做 RMSProp 归一化再加 momentum）；lr $4\times10^{-5}$。
- 数据：batch $16\times64$；uniform replay + online queue，容量 $5\times10^6$；latent state 写回 replay。
- 不使用任何 annealing、prioritized replay、weight decay 或 dropout。作者提到 prioritized replay 能进一步提分，但为了实现简单没有用。
- 训练量用 replay ratio 控制。例如 Atari 上 replay ratio 32、action repeat 4、batch $16\times64$，折合每 128 个 env step 做 1 次梯度更新，200M 步总共约 1.5M 次更新。
- 模型规模 12M–400M 由 hidden size $d$ 派生：GRU 单元数 $8d$，CNN 基础通道数与每个 latent 的 code 数都是 $d/16$（Table 3；注意表中 12M 一列的 recurrent units 写的是 1024，与 $8d=2048$ 不符，原表如此）。所有超参数不随规模变化。

**训练/推理流程**：交互时 encoder 和 RSSM 在线滤波得到 $s_t$，actor 直接采样动作，**没有 lookahead planning**。训练时交替执行：world model 在真实序列上的更新 → 从 posterior 状态出发的 imagination → critic 与 actor 的更新。

## 关键实验与证据

所有数字来自 `tables/*.tex`。每个 agent 都在 1 张 A100 上训练；Dreamer 和 PPO 每个 benchmark 跑 5 个 seed，其中 ProcGen 只有 1 个，BSuite 和 Minecraft 各 10 个。

| Benchmark（预算） | Dreamer | 主要对照 | 出处 |
|---|---|---|---|
| Atari 57（200M frames）gamer median / mean | 830% / 3381% | MuZero 693% / 3054%；PPO 180% / 892% | Table 6 |
| ProcGen 16（50M，hard）normalized mean | 66.01 | PPG 64.89；本文 PPO 42.80；原版 PPO 41.16 | Table 7 |
| DMLab 30（100M）human mean capped | 71.4 | IMPALA@1B 66.3；IMPALA@100M 31.0；PPO 35.9；R2D2+@10B 85.4 | Table 8 |
| Atari100k（400K）gamer mean / median | 125% / 49% | IRIS 105/29；TWM 96/51；SPR 62/40；EffMuZero 190/109（改了环境设置） | Table 9–10 |
| DMC Proprio 18（500K）mean / median | 871 / 754 | DMPO 801；D4PG 792 / 733；PPO 94 | Table 11 |
| DMC Visual 20（1M）mean / median | 861 / 786 | DrQ-v2 770 / 705；CURL 479；PPO 94 | Table 12 |
| BSuite 23 task mean / category mean | 66% / 63% | Boot DQN 60 / 57；PPO 49 / 47 | Table 13 |
| Minecraft Diamond（100M）return | 9.1 | IMPALA 7.1；Rainbow 6.3；PPO 5.1 | Table 5 |

- **Minecraft**：10 个 seed 全部在 100M 步内至少拿到过一次钻石，IMPALA、Rainbow、PPO 为 0%（Figure 5、Figure 9）。但到预算结束时，只有 **0.4% 的 episode** 拿到钻石（Figure 9 caption）。训练用 1 张 GPU、9 天，对比 VPT 的 720 张 GPU、9 天（`sections/related.tex`）；Table 2 记为 8.9 GPU days，使用 64 个远程 CPU 环境。
- **Robustness 消融**（Figure 6a，14 个任务，完整曲线见 Figure 17）：分别去掉 obs symlog、把 retnorm 换成 advnorm、把 symexp twohot 换成 Huber、去掉 KL balance & free bits，以及全部去掉。影响最大的是 KL 目标，其次是 return normalization 和 symexp twohot。每个技巧往往只对一部分任务关键。
- **Learning signal 消融**（Figure 6b，Figure 18）：停掉 reward 和 value 梯度对 representation 的塑形后，性能只小幅下降；停掉 reconstruction 梯度则大幅下降（图中约为完整版的 1/4，这是目测值）。结论是 Dreamer 主要依赖 world model 的无监督目标，作者由此提出可以利用无监督数据做预训练。
- **Scaling**（Figure 6c/d，Crafter 与 DMLab Goals）：12M→400M 参数时性能单调上升，而且所需环境交互更少；replay ratio 从 1 增到 64 时性能也可预测地提升。
- **PPO baseline 的可信度**：在 ProcGen 上，本文 PPO 的 42.80 略高于原版 PPO 的 41.16（Table 7），说明这个 baseline 不是弱化版本。

## 局限与论文没说的

**作者承认的局限**
- Minecraft 中钻石的单 episode 成功率只有 0.4%，"leaving a challenge"。
- DMLab 只跑了 100M 步，10B 跑不起；ProcGen 因为算力只跑 1 个 seed。
- 为了实现简单没有用 prioritized replay。
- 未来方向：从互联网视频学世界知识、跨 domain 共享一个 world model。

**我读出来的**
- **"单一配置"有前提**。模型规模和 replay ratio 是按 benchmark 分别选的：DMC 用 12M、其余用 200M；replay ratio 从 32 到 1024 不等（Table 2）。论文的说法是：replay ratio "按各 benchmark 的步数预算选择"，控制算力与数据效率的取舍；两个 control suite 用 12M 是因为它与 200M 性能相同且快得多（`sections/methods.tex` "Computational choices"）。但这两个量实际上仍是按 benchmark 设定的。
- **定向探索缺失**。BSuite 的 exploration 类别只有 0.01，Deep Sea 为 0.00，而 Boot DQN 分别是 0.68 和 1.00（Table 13）。只靠熵正则的探索无法处理需要深度探索的问题。Minecraft 的成功依赖 12 个 milestone reward，外加若干环境设置：沿用 prior work 的 block breaking 设置（`sections/experiments.tex`），作者自己的改动包括 flat categorical action space、去掉挖到钻石矿即终止、jump 键在后台按住 200ms（`sections/methods.tex` "Minecraft"），另外 MineRL v0.4.4 本身提供 abstract crafting actions。
- **对比口径不统一**。DMLab 正文说"超过 1B 步的 IMPALA 和 R2D2+"，但 Table 8 里只有 R2D2+ 在 10B 步的数（85.4 > 71.4）。Atari100k 中 Dreamer 用训练 episode 评测，没有单独的 eval episode（Table 10）。大部分 baseline 数字直接取自文献。ProcGen 上 Bigfish、Fruitbot、Miner 明显落后（Table 7）。
- **model 误差没有量化**。只给了开环视频预测的定性图（Figure 4、Figure 7），没有 compounding error 的指标。imagination 只有 15 步，长程部分靠 critic bootstrap。
- **没测的 setting**：真实机器人（DMC 都是仿真）、offline RL、multi-task/迁移、动力学随时间变化的非平稳环境、部署期分布偏移。每个任务都从零训练，没有测跨任务复用 world model。
- **算力门槛**。单次运行在 Atari 上是 7.7 GPU days，ProcGen 16.1，Minecraft 8.9（Table 2，口径应为单任务单 seed，论文没有明说）。全文 150 多个任务 × 多 seed 的总算力论文没有报告。
- **文本不一致**：$T=16$ 与 $H=15$；Eq.6 中熵项的符号；$\mathcal L_{\rm pred}$ 段落说 reward 用 symlog squared loss，后文又说 reward 用 symexp twohot。复现时应以代码为准。

## 在脉络中的位置

- **前驱**：Dyna（在模型中学习行为）；[World Models](../world-models/notes.md)（在梦中训练 controller）；PlaNet 的 RSSM；[Dreamer (V1)](../dreamer/notes.md)（latent imagination 中的 actor-critic，仅限连续控制）；DreamerV2（离散 latent 与 KL balancing，在 Atari 上超过人类）。critic 的分布式 twohot 回归借鉴自 MuZero 和 C51；EMA critic 类似 DQN 的 target network（论文引的是 Nature 2015 版 DQN；本项目收录的 2013 arXiv 版 [DQN](../../04_rl/dqn/notes.md) 还没有单独的 target network，只是在每轮优化时固定上一轮参数 $\theta_{i-1}$）。
- **后继 / 影响**：[Dreamer 4](../dreamer4/notes.md) 是同一作者的后继，改用可扩展的 Transformer world model，纯离线在 Minecraft 中拿到钻石。[TD-MPC2](../tdmpc2/notes.md) 走不重建观测、做 planning 的对照路线，并把 DreamerV3 作为主要 baseline。[DIAMOND](../diamond/notes.md) 在 Atari100k 上与 DreamerV3 对比像素级扩散 world model。同一作者线的 [DayDreamer](../../05_intersections/daydreamer/notes.md) 把 Dreamer 骨架搬到真机上在线学习（时间上早于 V3）。[Robotic World Model](../../05_intersections/robotic-world-model/notes.md) 延续了"learned simulator 中训练策略"的思路；[SimDist](../../05_intersections/simdist/notes.md) 则是在仿真中预训练 world model，部署时用 MPPI 做 planning，只用真实数据更新 latent dynamics，不在 world model 里训练 policy。
- **与本项目其他主题的联系**：
  - **RL**：actor 部分就是 REINFORCE 加熵正则。论文专门论证了为什么不用 [PPO](../../04_rl/ppo/notes.md) 的 advantage normalization、也不用 [SAC](../../04_rl/sac/notes.md) 的约束式目标熵，这对理解跨任务 RL 的稳定性很有价值。
  - **TTT**：learning signal 消融显示，representation 主要由自监督的 reconstruction 塑形。这与 [TTT](../../02_test_time_training/ttt-self-supervised/notes.md) 和 [PAD](../../05_intersections/pad/notes.md) 在部署时用自监督损失更新 encoder 的思路相契合（这是我的联系，论文没有测试）。另外，Dreamer 的 world model 在交互中一直在线更新，本身就是一种持续适应，但论文没有研究分布偏移下的适应。
  - **real2sim2real**：从真实交互学出的 world model 本身就是一个 learned simulator，不存在手工仿真器的 sim-to-real gap，换来的是 model error。RSSM 的 $h_t$ 从历史中汇总动力学信息，和 [RMA](../../03_real2sim2real/rma/notes.md) 用历史做隐式 system identification 的思路相通（这是我的类比，论文没有测试）。

## 前置知识

1. MDP、return、discount，以及 actor-critic、REINFORCE / policy gradient、熵正则（见 [PPO](../../04_rl/ppo/notes.md)、[SAC](../../04_rl/sac/notes.md)）。
2. TD(λ) 与 bootstrapped λ-return、target network。
3. VAE / ELBO、KL 散度、free bits、straight-through estimator（离散 latent）。
4. RSSM：确定性路径 $h_t$ 加随机路径 $z_t$，prior 与 posterior 的区别（[Dreamer](../dreamer/notes.md)）。
5. 分布式 value（C51/MuZero 的分类式回归）、twohot 编码。
6. 工程层面：GRU、RMSNorm、自适应梯度裁剪（AGC）、Adam 与 LaProp 的区别、replay ratio。

## 复现可行性

- **代码**：官方 JAX 实现 https://github.com/danijar/dreamerv3（MIT）。README 给出的入口是 `python dreamerv3/main.py --configs <env> <size>`，可以用 `--run.train_ratio` 设定 replay ratio。仓库持续更新，要对齐论文数字，建议固定到与 arXiv v2（2024-04）接近的版本。
- **算力**（Table 2，单张 A100 的 GPU days）：Visual Control 0.1（12M 模型）、Proprio Control 0.3（12M）、Atari100k 0.1、BSuite 0.5、DMLab 2.9、Atari 7.7、Minecraft 8.9、ProcGen 16.1。完整复现全部 benchmark 和 seed 需要的 GPU-天数很大，论文没有给出总数。
- **环境**：DMC、Atari（ALE，sticky actions）、ProcGen 容易装。DMLab 需要编译。Minecraft 依赖 MineRL v0.4.4（Minecraft 1.11.2，需要 Java），还要加上作者自己改过的 Diamond 环境。
- **最小可复现实验**：
  - 在 DMC Visual Control 上用 12M 模型、1M 步、replay ratio 512，跑 Walker Walk（目标约 961）和 Cartpole Swingup Sparse（目标约 759），3 个 seed，对照 Table 12。
  - 在 Cartpole Swingup Sparse 上加两个消融：(a) 把 return normalization 换成 advantage normalization；(b) 去掉 free bits。检验正文 "Actor learning" 部分的论点：稀疏奖励下 advantage normalization 会放大噪声、妨碍探索。注意 Figure 6a/17 的 14 个消融任务不含 Cartpole Swingup Sparse（其中的稀疏奖励 DMC 任务是 Visual Control Acrobot Sparse），而且论文的消融是把 KL balance 和 free bits 一起去掉，没有单独去掉 free bits，所以这是新实验，不是复现论文数字。
  - 算力约 0.1 GPU days/run × 12 runs，单卡几天可以完成。
- **预计难点**：JAX/CUDA 版本对齐；评测口径（训练 episode 还是独立 eval）；seed 方差（论文报 5 seeds 的均值 ± 1 std）；复现 Minecraft 需要 10 seeds × 约 9 GPU days，还要额外的 CPU 环境 worker，个人条件下基本不可行。

## 自测问题

1. $\mathcal L_{\rm dyn}$ 和 $\mathcal L_{\rm rep}$ 是同一个 KL，为什么要用 stop-gradient 拆开，并给出 1 : 0.1 的权重？KL 低于 1 nat 时 free bits 让梯度变成什么？它防止的是哪种退化解？
2. 把 advantage 除以它的 std，和把 return 除以 $\max(1,S)$，在"稀疏奖励、return 几乎全为 0"时分别会发生什么？为什么用 5–95 百分位而不是 min/max？
3. 推导 twohot 交叉熵对 logits 的梯度，说明为什么它与目标值 $y$ 的量级无关；对比直接对 $y$ 做 MSE 时的梯度。
4. 写出 imagination 中 λ-return 的递推式，说明 $c_t$、$\gamma$、$\lambda$ 和 $R^\lambda_T=v_T$ 各自的作用。为什么 horizon 只有 15 步也能处理 333 步的有效 discount horizon？
5. learning signal 消融（Figure 6b）说明 representation 主要由哪种梯度塑形？这对"部署时用自监督损失做 test-time adaptation"意味着什么？又有什么风险？
