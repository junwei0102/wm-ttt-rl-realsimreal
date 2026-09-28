# Proximal Policy Optimization Algorithms

> arXiv [1707.06347](https://arxiv.org/abs/1707.06347) · arXiv preprint, 2017（OpenAI；未在会议或期刊正式发表） · 主题：04_rl · 层级：foundational
> 代码：论文未给 PPO 的代码链接（只在脚注 3 给了 baseline "vanilla PG, adaptive stepsize" 的实现链接 berkeleydeeprlcourse/homework hw4）。OpenAI 官方实现在 https://github.com/openai/baselines（已访问确认：README 把 `ppo1` 标为 "obsolete version"，`ppo2` 为现行实现；整个 repo 状态为 "Maintenance"） · 笔记依据：本篇 arXiv 源码 `source/ppo-arxiv.tex` 只是用 `\includepdf` 包了一份 `ppo-min.pdf`，没有 .tex 正文，所以**公式、表格和数字都取自 `paper.pdf`**，编号沿用 PDF 中的 Eq./Table/Figure。

## 一句话

PPO 把 TRPO 的 KL 硬约束换成对概率比 $r_t(\theta)=\pi_\theta/\pi_{\theta_{old}}$ 做 clip 再取 min 的一阶 surrogate objective。这个目标对把比值推出 $[1-\epsilon,1+\epsilon]$、且让目标"变好"的变化不再给奖励，从而可以在同一批 on-policy 数据上安全地跑多个 epoch 的 minibatch SGD/Adam。

## 问题设定

- **设定**：标准 MDP 下的 on-policy model-free policy gradient。策略 $\pi_\theta(a|s)$ 是随机策略（连续控制用高斯分布，Atari 用离散分布），advantage 由学到的 $V(s)$ 通过 GAE 估计。奖励来自仿真器（MuJoCo、Roboschool、ALE）；观测处理（如 Atari 帧栈）原文未报告，Atari 只说网络结构沿用 [Mni+16]。
- **想同时满足的三点**（§1）：scalable（大模型、并行实现）、data efficient、robust（跨任务不调超参）。
- **之前的方法卡在哪**（§1、§2）：
  - DQN 在连续控制 benchmark 上没被证明有效（§1 脚注 1）。
  - vanilla PG（A2C/A3C）的数据效率低：每批数据只做一步梯度。如果在 $L^{PG}$ 上对同一条轨迹做多步优化，"empirically it often leads to destructively large policy updates"（§2.1）。
  - TRPO 需要共轭梯度加二阶近似，实现复杂，而且与带噪声的结构（dropout）、参数共享（policy 与 value 共享，或加辅助任务）不兼容（§1）。
  - TRPO 理论其实支持用 KL penalty（Eq. 5）代替约束，但固定的 $\beta$ 很难跨任务选，甚至在同一任务的不同训练阶段也不合适（§2.2）。

## 核心方法

**① 从 PG 到 surrogate。** 常用的梯度估计量与对应 loss（Eq. 1、2）：

$$\hat g=\hat{\mathbb E}_t\big[\nabla_\theta\log\pi_\theta(a_t|s_t)\hat A_t\big],\qquad L^{PG}(\theta)=\hat{\mathbb E}_t\big[\log\pi_\theta(a_t|s_t)\hat A_t\big]$$

TRPO 的做法（Eq. 3–4）是 $\max_\theta \hat{\mathbb E}_t[r_t(\theta)\hat A_t]$，约束为 $\hat{\mathbb E}_t[\mathrm{KL}[\pi_{\theta_{old}}(\cdot|s_t),\pi_\theta(\cdot|s_t)]]\le\delta$。把目标记为 CPI surrogate（Eq. 6，出自 conservative policy iteration [KL02]）：

$$L^{CPI}(\theta)=\hat{\mathbb E}_t\Big[\tfrac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}\hat A_t\Big]=\hat{\mathbb E}_t[r_t(\theta)\hat A_t]$$

白板上可以补一步（我的推导，不在论文里）：$\nabla_\theta r_t=r_t\nabla_\theta\log\pi_\theta$，而在 $\theta_{old}$ 处 $r_t=1$，所以 $\nabla L^{CPI}|_{\theta_{old}}=\nabla L^{PG}|_{\theta_{old}}$。也就是说第一步梯度与 vanilla PG 相同。ratio 的作用是：$\theta$ 离开 $\theta_{old}$ 后，仍能用旧策略的样本做 importance-weighted 估计，这就是能复用数据的原因。它的问题是没有约束，$L^{CPI}$ 会推动过大的更新。

**② Clipped surrogate objective（Eq. 7，本文核心）。**

$$L^{CLIP}(\theta)=\hat{\mathbb E}_t\Big[\min\big(r_t(\theta)\hat A_t,\ \mathrm{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat A_t\big)\Big],\quad \epsilon=0.2\ \text{(文中示例值)}$$

按 $\hat A$ 的符号逐项看（对应 Figure 1 的两张图）：

| 情况 | 单项目标等价于 | 梯度何时为 0 |
|---|---|---|
| $\hat A>0$ | $\min(r,1+\epsilon)\hat A$ | $r>1+\epsilon$：已经足够提高该动作概率，不再奖励 |
| $\hat A<0$ | $\max(r,1-\epsilon)\hat A$ | $r<1-\epsilon$：已经足够压低该动作概率，不再奖励 |

关键在于 min 的**不对称性**。只有当比值变化让目标"更好"时才被截断；如果比值往坏的方向走（例如 $\hat A>0$ 却 $r<1-\epsilon$），用的是未截断项，梯度仍然会把它拉回来。所以 $L^{CLIP}$ 是 $L^{CPI}$ 的 pessimistic lower bound，并且在 $r=1$ 处与 $L^{CPI}$ 一阶相等（§3）。Figure 2 给出直观证据：在 Hopper-v1 的第一次更新中沿更新方向插值，$L^{CPI}$ 单调上升，$L^{CLIP}$ 在更新后的参数处达到最大，此时 KL 约为 0.02。clip 相当于一个隐式的 trust region。

**③ 备选：Adaptive KL penalty（§4，Eq. 8）。** 目标为 $L^{KLPEN}=\hat{\mathbb E}_t[r_t\hat A_t-\beta\,\mathrm{KL}[\pi_{\theta_{old}},\pi_\theta]]$。每次更新后计算 $d=\hat{\mathbb E}_t[\mathrm{KL}]$：若 $d<d_{targ}/1.5$，令 $\beta\leftarrow\beta/2$；若 $d>1.5\,d_{targ}$，令 $\beta\leftarrow2\beta$。1.5 和 2 是启发式取值，作者说对它们不敏感。作者明确表示这一变体比 clip 差，保留它只是作为重要 baseline。

**④ 完整 actor-critic 目标（Eq. 9）。** policy 与 value 共享参数时：

$$L_t^{CLIP+VF+S}(\theta)=\hat{\mathbb E}_t\big[L_t^{CLIP}(\theta)-c_1L_t^{VF}(\theta)+c_2S[\pi_\theta] (s_t)\big],\quad L_t^{VF}=(V_\theta(s_t)-V_t^{targ})^2$$

$S$ 是 entropy bonus。论文没有写 $V_t^{targ}$ 的具体定义；常见实现（如 baselines）取 $\hat A_t+V_{old}(s_t)$，这一点不来自本文。

**⑤ Truncated GAE（Eq. 11–12）。** 每个 actor 跑 $T$ 步（$T\ll$ episode 长度），advantage 只看这 $T$ 步：

$$\delta_t=r_t+\gamma V(s_{t+1})-V(s_t),\qquad \hat A_t=\sum_{l=0}^{T-t-1}(\gamma\lambda)^l\delta_{t+l}$$

$\lambda=1$ 时退化成 A3C 的 $n$-step 估计量（Eq. 10）。**注意 PDF 的笔误**：Eq. 10 和 Eq. 11 最后一项的指数印成了 $T-t+1$，按求和结构应为 $T-t-1$。实现时一般写成反向递推 $\hat A_t=\delta_t+\gamma\lambda\hat A_{t+1}$（等价改写）。

**⑥ 训练循环（Algorithm 1）。** 每个 iteration：$N$ 个并行 actor 各用 $\pi_{\theta_{old}}$ 采 $T$ 步 → 计算 $\hat A_1..\hat A_T$ → 在这 $NT$ 个样本上以 minibatch 大小 $M\le NT$ 做 $K$ 个 epoch 的 Adam → $\theta_{old}\leftarrow\theta$。相对 vanilla PG，只需把 loss 从 $L^{PG}$ 换成 $L^{CLIP}$ 或 $L^{KLPEN}$（§5）。推理时只用 $\pi_\theta$；value 网络和 clip 都只在训练中起作用。

## 关键实验与证据

**1. Surrogate 消融（§6.1，Table 1）。** 7 个 MuJoCo `-v1` 任务（HalfCheetah、Hopper、InvertedDoublePendulum、InvertedPendulum、Reacher、Swimmer、Walker2d），每个训练 1M timesteps，3 个 seed。评分方式：每个 run 取最后 100 个 episode 的平均回报，按环境归一化（随机策略记 0，最好结果记 1），再对 21 个 run 取平均。策略网络为 MLP 2×64、tanh，输出高斯均值，std 为 "variable standard deviations"（沿用 [Sch+15b; Dua+16]）；policy 与 value 不共享参数，也不加 entropy bonus。其余超参见 Table 3：$T=2048$，Adam $3\times10^{-4}$，10 epochs，minibatch 64，$\gamma=0.99$，$\lambda=0.95$。

| 变体 | avg. normalized score |
|---|---|
| No clipping or penalty | **−0.39** |
| Clipping $\epsilon$=0.1 / **0.2** / 0.3 | 0.76 / **0.82** / 0.70 |
| Adaptive KL $d_{targ}$=0.003 / 0.01 / 0.03 | 0.68 / 0.74 / 0.71 |
| Fixed KL $\beta$=0.3 / 1 / 3 / 10 | 0.62 / 0.71 / 0.72 / 0.69 |

结论：多 epoch 复用数据时，**必须有某种约束**。不加约束得到负分，主要因为 HalfCheetah 上的结果比随机策略还差。在有约束的变体里，clip 以 $\epsilon=0.2$ 最好。作者还试过在 log 空间做 clip，结果没有更好（§6.1）。

**2. 与其他算法比较（§6.2，Figure 3）。** 对比对象为 TRPO、CEM、adaptive-stepsize vanilla PG、A2C、A2C + trust region（ACER 式），都是调过的实现，PPO 用 $\epsilon=0.2$。论文原话是 PPO "outperforms the previous methods on almost all"。放大图后可见：HalfCheetah、Walker2d、Reacher 上 PPO 明显领先；Swimmer 上 TRPO 的最终曲线高于 PPO；Hopper 上 PPO、TRPO、vanilla PG 末端接近。论文没有给出这张图的数值表。

**3. 高维展示（§6.3，Figure 4/5，Table 4）。** Roboschool 3D humanoid 任务：前进（训练 50M timesteps）、Flagrun（目标每 200 步或到达后随机换位，训练 100M）、FlagrunHarder（被方块砸、需要爬起来，训练 100M）。超参：$T=512$，15 epochs，minibatch 4096，actors 32（locomotion）/ 128（flagrun），log std 从 −0.7 线性退火到 −1.6。**Adam stepsize 按 KL 目标值自适应调节**（Table 4 caption，具体目标值未给）。这一部分只有 PPO 的学习曲线，没有 baseline。

**4. Atari（§6.4，Table 2/5/6，Figure 6）。** 49 个游戏，40M frames（10M timesteps），3 个 trial，对手是调过的 A2C 和 ACER，三者网络结构相同，都沿用 [Mni+16]。PPO 超参见 Table 5：8 actors，$T=128$，3 epochs，minibatch 32×8，Adam $2.5\times10^{-4}\times\alpha$，clip $0.1\times\alpha$，$\alpha$ 从 1 线性退火到 0，$c_1=1$，$c_2=0.01$。Table 2 统计各算法"赢"的游戏数：

| 口径 | A2C | ACER | PPO | Tie |
|---|---|---|---|---|
| 全程平均 episode 回报（偏向学得快） | 1 | 18 | **30** | 0 |
| 最后 100 个 episode（偏向最终性能） | 1 | **28** | 19 | 1 |

所以 PPO 的优势在 sample complexity 上，**最终性能不如 ACER**，这与 §1 的表述"similarly to ACER though it is much simpler"一致。我按 Table 6 的最终分数做了逐项两两统计（非论文数字）：PPO 对 A2C 为 40 胜、8 负、1 平（Venture 三者都是 0）。三方比较得到 ACER 27、PPO 19、平 2（Pong 上 ACER 与 PPO 显示值都是 20.7；Venture 三者均为 0.0）、A2C 1，与 Table 2 相差 1 局，应该是显示精度的舍入造成的。几个典型差距：Enduro（A2C/ACER 0.0，PPO 758.3）、Kangaroo（45.3 / 50.0 / 9928.7）；反过来 Gopher 上 ACER 37802.3，PPO 只有 2932.9。

## 局限与论文没说的

- **作者承认的**：KL penalty 不如 clip；adaptive KL 偶尔会出现 KL 明显偏离 $d_{targ}$ 的更新（§4）；Atari 最终性能输给 ACER（Table 2）。
- **"lower bound" 只是相对 $L^{CPI}$ 而言**：§1 说它是"pessimistic estimate (i.e., lower bound) of the performance of the policy"。但文中只论证了 $L^{CLIP}\le L^{CPI}$，没有 TRPO 那样的单调改进定理。clip 也**不硬性限制** ratio 或 KL：它只让出界样本的梯度为 0，这些样本仍会被共享参数和其他样本的梯度带着走，epoch 数 $K$ 越大越容易越界。Figure 2 的 KL≈0.02 只是 Hopper 第一次更新的单个例子。
- **实现细节决定成败，但论文没写**：观测和回报归一化、advantage 归一化、value clipping、梯度裁剪、正交初始化等细节只存在于 baselines 代码中。后来 Engstrom et al. 2020（"Implementation Matters in Deep Policy Gradients"）和 Andrychowicz et al. 2021（"What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical Study"）专门研究了这些 code-level 选择的影响，这两篇不在本项目中。
- **评测口径**：MuJoCo 只有 3 个 seed，归一化方式以"最好结果 = 1"为基准；Figure 3 只有曲线；baseline 声称"tuned"，但调参协议没有公开。Atari 用胜场数而不是 human-normalized score。回报都是训练过程中（stochastic policy 采样得到的）episode 回报，文中没有描述单独的 evaluation。
- **没测的 setting**：§5 说固定长度 $T$ 步采样的实现风格"well-suited for use with recurrent neural networks"，但文中没有明确的 recurrent policy 实验（MuJoCo 用 MLP，Atari 只说沿用 [Mni+16] 的网络结构）；MuJoCo 实验明确不共享参数（Roboschool 未说明），"兼容参数共享"这一优势只在 Atari 上间接体现（Table 5 设了 VF coeff $c_1=1$）；没有真机实验，也没有 sim-to-real；困难探索问题没有解决（Table 6 中 MontezumaRevenge 为 42.0，Venture 为 0.0）。
- **on-policy 的样本效率**：数据用完 $K$ 个 epoch 就丢掉。它只是比 A2C 省样本，与 off-policy 方法（如 [SAC](../sac/notes.md)）相比并不占优。本文唯一的 off-policy 对手是 Atari 上的 ACER（带 experience replay），按最终性能 PPO 还输给了它（Table 2）；连续控制部分没有与 off-policy 方法比较。
- **算力与 wall-time**：摘要说 PPO 在 sample complexity、simplicity 与 wall-time 之间取得了好的折中，但**全文没有报告任何硬件、训练时长或 wall-clock 数据**。

## 在脉络中的位置

- **前驱**：REINFORCE [Wil92]；conservative policy iteration [KL02]（$L^{CPI}$ 的来源）；TRPO [Sch+15b]（surrogate 加 KL 约束）；GAE [Sch+15a]；A3C/A2C [Mni+16]（多 actor、$T$ 步截断）；ACER [Wan+16]。同期工作 Heess et al. [Hee+17] 用 adaptive-KL 版 PPO 学习 3D locomotion。系统背景见 [RL overview](../survey-rl-overview/notes.md)。
- **后继 / 影响**：
  - **sim-to-real locomotion 的默认优化器**：[legged_gym](../../03_real2sim2real/legged-gym/notes.md) 是大规模并行 PPO，并且用了"按 KL 自适应学习率"（desired KL 0.01），与本文 Table 4 按 KL 目标调 Adam stepsize 的做法一致（legged_gym 自己注明参照的是 Heess et al. [Hee+17]）；[RMA](../../03_real2sim2real/rma/notes.md) Phase 1、[ASAP](../../03_real2sim2real/asap/notes.md) 的 tracking policy 和 delta action model 用的都是 PPO；[RialTo](../../03_real2sim2real/rialto/notes.md) 在 digital twin 里做 PPO + BC fine-tuning。
  - **生成式策略 / VLA 的 RL fine-tuning**：[DPPO](../dppo/notes.md) 把去噪链看成内层 MDP，再套用 PPO；[π_RL](../pi-rl/notes.md) 为 flow VLA 构造可计算的 log-likelihood，使 PPO/GRPO 能够使用。
  - **GRPO 一支**：[DeepSeek-R1](../deepseek-r1/notes.md) 的 GRPO 保留了 PPO 的 clipped ratio，去掉了 value network，改用组内归一化的回报作 advantage，并加上对 reference policy 的 KL。[SimpleVLA-RL](../simplevla-rl/notes.md)、[TTRL](../../02_test_time_training/ttrl/notes.md)、[WMPO](../../05_intersections/wmpo/notes.md) 都沿用 GRPO。RLHF（InstructGPT）也以 PPO 为优化器，不过这篇不在本项目中。
- **与本项目其他主题的联系**：
  - *World model*：PPO 只需要样本和 $\log\pi$，把环境当黑箱，所以能直接在 learned simulator 的 imagination 里训练。[Robotic World Model](../../05_intersections/robotic-world-model/notes.md) 就是这样做的，此时 on-policy 的样本代价由廉价的 imagined rollout 承担。它与 [Dreamer](../../01_world_model/dreamer/notes.md) 穿过可微 dynamics 反传 value 梯度的做法形成 likelihood-ratio 与 pathwise gradient 的对照。
  - *TTT*：test-time RL 中，[TTRL](../../02_test_time_training/ttrl/notes.md) 默认用 GRPO（也验证了 PPO）；[TTT-Discover](../../02_test_time_training/ttt-discover/notes.md) 则指出直接套用 PPO/GRPO 优化期望回报不适合 discovery 问题，改用 entropic objective 的 policy gradient（advantage 中加 KL penalty，β 按 KL 约束自适应）。clip 提供的隐式 trust region，是部署时做小步在线更新可借用的一种稳定性来源。
  - *Real2sim2real*：PPO、[domain randomization](../../03_real2sim2real/domain-randomization/notes.md) 加大规模并行仿真，构成了目前 locomotion sim-to-real 的标准配方（见 [reality gap survey](../../03_real2sim2real/survey-reality-gap/notes.md)）。本文 Table 4 用了 32–128 个 actors 和 4096 的大 minibatch，可以看作这一路线的早期形态。

## 前置知识

1. MDP、return、discount $\gamma$；value function 与 advantage $A(s,a)=Q-V$。
2. Policy gradient theorem 与 REINFORCE，以及 baseline 为什么能降低方差。
3. Importance sampling 与 on-policy / off-policy 的区别，也就是为什么 ratio 能让旧数据估计新策略的目标。
4. TRPO：KL trust region、CPI surrogate、共轭梯度加 line search 的动机（读本文前建议先读 TRPO §3–6）。
5. GAE：$\lambda$ 在偏差与方差之间的权衡，TD residual $\delta_t$。
6. Actor-critic 与 A2C/A3C 的并行采样；Adam；entropy regularization。

## 复现可行性

- **代码**：openai/baselines `ppo2` 已开源（README 把 `ppo1` 列为 "obsolete version, left here temporarily"，repo 状态为 "Maintenance"）。论文本身没有给 PPO 的代码链接。
- **环境与依赖**：论文用 Gym MuJoCo `-v1`、Roboschool、ALE。`-v1` 环境和 Roboschool 如今都已废弃，现代 Gymnasium MuJoCo 版本的动力学和 reward 有差异，数值无法与 Table 1 或 Figure 3 直接对比。Atari 通过 ALE 仍可用。
- **算力**：**论文没有报告**任何硬件、GPU 或 wall-clock。MuJoCo 实验用 2×64 MLP、1M 步，按我的判断单机 CPU 就能跑；Roboschool 用 50M–100M 步，Atari 49 个游戏 × 3 seed × 10M timesteps，成本主要在仿真采样上。
- **最小可复现实验**（对应 Table 1 与 Figure 2）：
  1. 在 Hopper 和 HalfCheetah（现代 MuJoCo 版本）上跑三组：{no clip/penalty、clip $\epsilon=0.2$、adaptive KL $d_{targ}=0.01$}，各 3 个 seed，1M 步。严格使用 Table 3 超参，MLP 2×64 tanh，policy/value 分离，不加 entropy。
  2. 指标：最后 100 个 episode 的平均回报；每次更新记录 approx-KL 和 $|r-1|>\epsilon$ 的样本比例。
  3. 在第一次更新时复现 Figure 2：沿 $\theta_{old}\to\theta_{new}$ 插值，画出 $L^{CPI}$、$L^{CLIP}$、KL 三条曲线。
  4. 预期在定性上复现：不加约束时 HalfCheetah 会崩；$L^{CLIP}$ 在更新点附近达到峰值。
- **预计难点**：未写入论文的 code-level trick 会显著影响结果，需要逐项开关做对照；3 个 seed 方差很大；环境版本不同导致无法对齐绝对数值；GAE 公式的指数笔误可能误导手写实现。

## 自测问题

1. 设 $\epsilon=0.2$、$\hat A_t=-1$。当 $r_t=1.5$ 和 $r_t=0.7$ 时，该样本对 $\theta$ 的梯度分别是否为 0？为什么说 min 只在"变好"的方向上截断？
2. 证明 $L^{CLIP}$ 与 $L^{CPI}$ 在 $\theta_{old}$ 处一阶相等，并说明 PPO 第一个 minibatch 的梯度与 vanilla PG 有什么关系。那么 clip 究竟在第几步开始起作用？
3. 为什么在 $L^{PG}$ 上对同一批数据做多个 epoch 是"not well-justified"的，而 $L^{CPI}$ 加上 clip 就可以？clip 能否保证 $\mathrm{KL}\le\delta$？
4. 从 Eq. 11–12 推出 $\lambda=1$ 时的 Eq. 10，并指出 PDF 中指数的笔误。$\lambda$ 从 0 变到 1 时，偏差和方差分别怎样变化？
5. TRPO 为什么难以处理 policy/value 参数共享或 dropout？PPO 的 Eq. 9 是如何处理的？在 learned world model 里训练策略时，PPO 与 Dreamer 的 actor 梯度在"对模型的要求"上有什么根本区别？
