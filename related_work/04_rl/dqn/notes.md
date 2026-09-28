# Playing Atari with Deep Reinforcement Learning

> arXiv [1312.5602](https://arxiv.org/abs/1312.5602) · NIPS 2013 Deep Learning Workshop（正文未写 venue，源码使用 `nips13submit_e.sty`）, 2013 · 主题：04_rl · 层级：foundational
> 代码：none（论文未给出代码。2015 年 Nature 扩展版有官方 Lua/Torch 实现 https://github.com/google-deepmind/dqn，已于 2022-07 归档；该实现带 target network，和本文算法不同）· 笔记依据：`source/*.tex`（数字/公式）+ `paper.pdf`（图）

## 一句话

用一个 CNN 直接从原始像素回归所有动作的 Q 值，用 Q-learning 的 TD target 加 SGD 训练。关键想法是 **experience replay**：把转移存进 replay memory 后均匀随机采样，打破样本间的相关性，并平滑非平稳的数据分布。这使深度网络与 off-policy TD learning 的组合在 7 个 Atari 游戏上稳定训练，而且所有游戏共用同一套架构和超参数。

## 问题设定

- **环境**：Arcade Learning Environment（ALE）里的 Atari 2600 模拟器 $\mathcal{E}$，可能是随机的。原始观测 $x_t$ 是 $210\times160$、60Hz 的 RGB 画面（`source/intro.tex`），动作集合 $\mathcal{A}=\lbrace1,\dots,K\rbrace$ 是离散的，所测游戏中 $K$ 在 4 到 18 之间（`source/method.tex`）。reward $r_t$ 是分数变化。
- **可观测性**：只看当前屏幕属于 partially observed，存在 perceptual aliasing。作者因此把完整历史 $s_t=x_1,a_1,\dots,a_{t-1},x_t$ 当作状态，假设 episode 有限长，这样得到一个很大但有限的 MDP（`source/background.tex`）。
- **不给什么**：不提供游戏专属信息、手工视觉特征或模拟器内部状态，只有画面、reward、终止信号和合法动作集合。7 个游戏用同一个网络结构和同一套超参数。
- **之前卡在哪**（`source/intro.tex`、`background.tex` Related Work）：
  1. 视觉 RL 靠手工特征加线性 value function，效果受特征质量限制。
  2. 深度学习默认 i.i.d. 样本和固定分布，而 RL 的样本高度相关，而且分布随策略变化；reward 稀疏、有噪声、有延迟。
  3. Q-learning 配非线性函数逼近或 off-policy 学习可能发散（Tsitsiklis & Van Roy；Baird），所以主流转向了有收敛保证的线性逼近。
  4. NFQ 用 RPROP 做 batch 更新，每轮计算量与数据集大小成正比；DFQ（Lange & Riedmiller）先训 autoencoder 再做 NFQ，不是端到端。

## 核心方法

**1. 目标：Bellman 最优方程**（`source/background.tex` Eq.1）。定义 $R_t=\sum_{t'=t}^{T}\gamma^{t'-t}r_{t'}$， $Q^{\ast}(s,a)=\max_\pi\mathbb{E}[R_t\mid s_t=s,a_t=a,\pi]$，它满足

$$Q^{\ast}(s,a)=\mathbb{E}_ {s'\sim\mathcal{E}}\Big[r+\gamma\max_{a'}Q^{\ast}(s',a')\thinspace\Big|\thinspace s,a\Big]$$

**2. Q-network 的回归损失**（`background.tex` Eq.2，`eq:q-learning-loss`）。用网络 $Q(s,a;\theta)$ 逼近 $Q^{\ast}$，第 $i$ 轮最小化

$$L_i(\theta_i)=\mathbb{E}_ {s,a\sim\rho(\cdot)}\big[(y_i-Q(s,a;\theta_i))^2\big],\qquad y_i=\mathbb{E}_ {s'\sim\mathcal{E}}\big[r+\gamma\max_{a'}Q(s',a';\theta_{i-1})\mid s,a\big]$$

其中 $\rho$ 是 behaviour distribution。优化 $L_i$ 时 $\theta_{i-1}$ 保持固定。作者特别指出，target 依赖网络权重，这一点和监督学习不同。

**3. 梯度**（`background.tex` Eq.3，`eq:q-learning-gradient`）：

$$\nabla_{\theta_i}L_i=\mathbb{E}_ {s,a\sim\rho;\thinspace s'\sim\mathcal{E}}\Big[\big(r+\gamma\max_{a'}Q(s',a';\theta_{i-1})-Q(s,a;\theta_i)\big)\nabla_{\theta_i}Q(s,a;\theta_i)\Big]$$

由于优化 $L_i$ 时 $\theta_{i-1}$ 固定，Eq.3 对 $L_i(\theta_i)$ 本身就是精确梯度（原式省略了常数因子 $-2$）。但 Algorithm 1 的 target 用的是当前 $\theta$，却仍按 Eq.3 更新，相当于忽略 target 对 $\theta$ 的依赖，也就是 Sutton & Barto 所说的 semi-gradient（原文没有用这个词）；它不是对 TD 误差平方求全梯度（后者是 Baird 的 residual gradient）。每步只用单样本估计期望、每步都更新时，就退化成经典 Q-learning。这个算法是 **model-free**（不估计 $\mathcal{E}$），也是 **off-policy**（学习 greedy 策略，按 $\epsilon$-greedy 行动）。

**4. Deep Q-learning with Experience Replay**（`source/method.tex` Algorithm 1）。每一步的流程：
1. 以 $\epsilon$-greedy 选动作，执行后得到 $r_t,x_{t+1}$，令 $\phi_{t+1}=\phi(s_{t+1})$；
2. 把 $(\phi_t,a_t,r_t,\phi_{t+1})$ 存进容量为 $N$ 的 replay memory $\mathcal{D}$，只保留最近 $N$ 条；
3. 从 $\mathcal{D}$ 均匀采一个 minibatch，计算

$$y_j=\begin{cases}r_j & \phi_{j+1}\ \text{terminal}\cr r_j+\gamma\max_{a'}Q(\phi_{j+1},a';\theta) & \text{otherwise}\end{cases}$$

4. 对 $(y_j-Q(\phi_j,a_j;\theta))^2$ 做一步梯度下降。

**白板要点**：Algorithm 1 算 target 用的是**当前的 $\theta$**，不是 Eq.2 里冻结的 $\theta_{i-1}$。本文**没有独立的 target network**，定期复制参数的 target network 是 2015 年 Nature 版才加的。（项目里 [HIL-SERL](../hil-serl/notes.md) 的 `sections/methods.tex` 把 target network 引到了 mnih2013playing 上，这个出处不准确。）另外，算法中 "otherwise select $a_t=\max_a Q^{\ast}(\phi(s_t),a;\theta)$" 这一行的 $\max_a Q^{\ast}$ 应该是 $\arg\max_a Q$，是原文笔误。

**为什么需要 replay**（`method.tex`）：
- (a) 每条经验可以参与多次更新，data efficiency 更高；
- (b) 随机采样打破相邻样本的相关性，降低更新方差；
- (c) 在 on-policy 学习中，当前参数决定下一批数据，容易形成反馈循环，陷入局部最优甚至发散。replay 把 behaviour distribution 在许多历史策略上做了平均。

由于 replay 里的数据来自旧参数，学习**必须是 off-policy**，这也是选 Q-learning 的原因。作者承认，均匀采样加 FIFO 覆盖不区分哪些转移重要，并提出可以借鉴 prioritized sweeping。

**5. 预处理 $\phi$ 与网络**（`method.tex` §4.1）：
- RGB 转灰度，下采样到 $110\times84$，再裁出 $84\times84$ 的游戏区域。裁成正方形只是因为用了 Krizhevsky 的 GPU 卷积实现，它要求输入为正方形。
- 把最近 4 帧堆叠成 $84\times84\times4$ 的输入。
- 网络结构：Conv 16 个 $8\times8$ 卷积核、stride 4、ReLU → Conv 32 个 $4\times4$ 卷积核、stride 2、ReLU → FC 256 ReLU → 线性输出层，每个动作一个输出。按无 padding 推算（论文没写 padding），特征图尺寸依次为 $20\times20\times16$ → $9\times9\times32$。
- **每个动作一个输出头**：一次前向就能得到所有动作的 Q 值。如果像 NFQ/DFQ 那样把动作作为输入，代价会随 $K$ 线性增长。

**6. 训练设定**（`source/experiments.tex` §5）：
- 优化与探索：RMSProp，minibatch 32； $\epsilon$ 在前 1M 帧内从 1 线性退火到 0.1，之后固定为 0.1。
- 数据量：共训练 10M 帧，replay memory 保存最近 1M 帧。
- **Reward clipping**：训练时正 reward 记为 +1、负 reward 记为 −1、0 保持不变，评测时用原始分数。
- **Frame skip**： $k=4$，Space Invaders 例外取 $k=3$，因为 $k=4$ 时激光的闪烁周期会让它看不见。这是各游戏之间唯一的超参数差异。
- 评测：用 $\epsilon=0.05$ 的 $\epsilon$-greedy（以概率 $1-\epsilon$ 取 $\arg\max_a Q$）。论文没有报告纯 greedy 的评测结果。

## 关键实验与证据

**主结果**（`experiments.tex` Table 1）。前五行是平均分：Sarsa、Contingency、DQN 是 $\epsilon=0.05$ 的 $\epsilon$-greedy 跑固定步数的平均总分；Random 是均匀随机选动作的策略；Human 是一位专家每个游戏玩约两小时后的 median reward。后三行是单个最佳 episode 的分数：

| | B.Rider | Breakout | Enduro | Pong | Q*bert | Seaquest | S.Invaders |
|---|---|---|---|---|---|---|---|
| Random | 354 | 1.2 | 0 | −20.4 | 157 | 110 | 179 |
| Sarsa | 996 | 5.2 | 129 | −19 | 614 | 665 | 271 |
| Contingency | 1743 | 6 | 159 | −17 | 960 | 723 | 268 |
| **DQN** | **4092** | **168** | **470** | **20** | **1952** | **1705** | **581** |
| Human | 7456 | 31 | 368 | −3 | 18900 | 28010 | 3690 |
| HNeat Best | 3616 | 52 | 106 | 19 | 1800 | 920 | **1720** |
| HNeat Pixel | 1332 | 4 | 91 | −16 | 1325 | 800 | 1145 |
| DQN Best | **5184** | **225** | **661** | **21** | **4500** | **1740** | 1075 |

- 与使用手工特征的 Sarsa 和 Contingency 相比，DQN 在 7 个游戏上全部领先。这两个基线用了背景减除，并把 128 种颜色各作为一个通道，相当于先验的物体分割（§5.3）。
- HNeat Best 依赖手工物体检测器，而且只报告单个最佳 episode 的分数。DQN 的**平均分**在 Space Invaders 以外的 6 个游戏上都超过了它。摘要里"6/7 超过所有已有方法"指的就是这个比较。
- 超过人类专家的有 Breakout、Enduro、Pong 三个。Beam Rider 被作者称为 "close to human"，实际为 4092/7456≈55%。Q*bert、Seaquest、Space Invaders 差距很大，作者归因于这些游戏需要长时间尺度上的策略。
- **训练稳定性**（Figure 2）：每个 epoch 是 50000 次 minibatch 更新，约 30 分钟。平均 episode reward（ $\epsilon=0.05$，10000 步）曲线噪声很大；而在训练前用随机策略收集一组固定状态，追踪其上的平均 $\max_a Q$，曲线平滑上升。作者报告"所有实验都没有出现发散"。
- **Value function 可视化**（Figure 3，Seaquest 中 30 帧的一段）：敌人出现（A）时预测值跳升，鱼雷即将命中（B）时达到峰值，敌人消失（C）后回落。
- **消融：没有。** 论文没有做"去掉 replay"或"不做 reward clipping"的对照，所以 replay 的必要性在本文只有论证（§4），没有实验证据。

## 局限与论文没说的

**作者自己承认的**：
- reward clipping 让 agent 分不清不同大小的 reward；
- replay 是均匀采样加覆盖，不区分重要转移；
- 需要长时间尺度策略的游戏离人类水平很远；
- 没有理论收敛保证。

**我读出来的**：
1. **超参数不全，难以精确复现**：没有报告 $\gamma$、RMSProp 的学习率等参数、开始学习前 replay 的预热量（minibatch 只说了从 $\mathcal{D}$ 均匀随机采样）。"1M 帧"和"10M 帧"指的是模拟器帧还是 agent 步（frame skip 之后）也没说清。Figure 2 横轴到 100 epoch，即 5M 次更新；Algorithm 1 是每个 agent 步更新一次，按 10M 模拟器帧、 $k=4$ 算只有 2.5M 步，两者对不上。
2. **评测口径粗糙**：没有 seed 数、方差或置信区间；Table 1 的"fixed number of steps"没给具体值；人类分数来自一位专家玩约 2 小时后的中位数，作者自己说明它明显高于 Bellemare 等人报告的人类分数。"DQN Best"取最佳单局，本质上是挑最大值。基线数字直接取自各自的论文，交互预算与协议未必一致。
3. **"单一超参数"有例外**：Space Invaders 的 $k=3$ 是作者注意到 $k=4$ 时激光因闪烁周期而不可见后，针对这一个游戏单独改的。
4. **4 帧堆叠 ≠ 完整历史**：形式化时假设状态是完整序列 $s_t$，实际 $\phi$ 只保留 4 帧。超过 4 帧的部分可观测性（比如需要记住屏幕外的信息）没有处理，这可能也是长时间尺度游戏表现差的原因之一。
5. **稳定性只是经验结论**：没有 target network，而 bootstrapping、非线性逼近和 off-policy 恰好同时存在。作者只根据 7 个游戏上的平滑 $\max Q$ 曲线声称"没有发散"。 $\max Q$ 上升本身并不能说明估计准确，max 算子带来的高估问题在这里无从检验（后来的 Double DQN 专门处理这个问题）。
6. **只适用于离散小动作空间**：结构上对每个动作输出一个值，并且需要 $\arg\max$，不能直接用于连续控制。
7. **训练和测试是同一个未修改的游戏**：形式化里作者只说 $\mathcal{E}$ "may be stochastic"，论文没有提到 sticky action 或 no-op 起始之类的随机化（原文未报告），评测里能看到的随机性来源只有 $\epsilon=0.05$。论文没有测跨游戏泛化，也没有测扰动下的鲁棒性。作者批评 HNeat 利用确定性漏洞，但 DQN 在这一点上也没有受过检验。
8. **算力**：只给出每个 epoch 约 30 分钟，没有写 GPU 型号。按 Figure 2 的 100 epoch 推算约 50 小时，这是我根据图注换算的，论文没有写总训练时间。

## 在脉络中的位置

- **前驱**：Q-learning（Watkins & Dayan 1992）；TD-Gammon（Tesauro 1995，用 MLP 做 on-policy value learning）；experience replay（Lin 1993，当时用于低维状态）；NFQ（Riedmiller 2005）和 DFQ（Lange & Riedmiller 2010）；ALE 基准（Bellemare et al. 2013）；Krizhevsky et al. 2012（AlexNet）的 GPU 卷积实现；ReLU（原文引的是 Jarrett et al. 2009 和 Nair & Hinton 2010）。
- **后继 / 影响**：
  - DQN 系列：Nature 版 DQN（2015，加入 target network，扩展到更多游戏），以及 Double DQN、Prioritized Experience Replay（正是本文 §4 末尾提出的方向）、Dueling、Rainbow。
  - [SAC](../sac/notes.md)：继承 replay 与 target value network，把 off-policy actor-critic 推广到连续动作；其源码中 target network 引用的是 mnih2015human。
  - [HIL-SERL](../hil-serl/notes.md)：夹爪的离散动作单独用一个 DQN critic（外加 target network）训练（`sections/methods.tex`）。
  - [IQL](../iql/notes.md)：offline RL 直接面对"固定 replay 数据 + $\max_{a'}$ 查询分布外动作"会高估的问题，改用 expectile 回归来避开。
  - [DayDreamer](../../05_intersections/daydreamer/notes.md)：用 DQN 路线的 Rainbow 作为离散动作任务的基线。
  - [PAD](../../05_intersections/pad/notes.md)：在引言第一句把本文列为"CNN + 深度 RL 从像素出发取得成功"的代表工作之一（`introduction.tex`）。
  - 还有 [RL 综述](../survey-rl-overview/notes.md)。
- **与本项目其他主题的联系**：
  - **World model**：DQN 明确是 model-free 方法（不估计 $\mathcal{E}$），而它奠定的 Atari/ALE 基准后来成了 world model agent 的主要比较场地，例如 [DreamerV3](../../01_world_model/dreamerv3/notes.md) 和 [DIAMOND](../../01_world_model/diamond/notes.md) 都在 Atari 上评测。[World Models](../../01_world_model/world-models/notes.md) 的 Table 1 也列了一个 DQN 基线，但那是 CarRacing-v0 上的结果，不是 Atari。replay buffer 也是后来训练 world model 的数据容器；DreamerV3 的 EMA critic 和 DIAMOND 的 stop-gradient target 都能追溯到 DQN 路线的 target 处理。
  - **TTT**：本文在训练用的同一个游戏上评测，没有描述任何测试时适应机制，也没有测训练和测试环境不同的情形。作者观察到"权重的小变化会引起访问状态分布的大变化"（§5.1，用来解释 reward 曲线为什么噪声大），但这是策略自身引起的状态分布变化，和 TTT 针对的训练与测试之间的 distribution shift 不是一回事。后者参见 [TTT self-supervised](../../02_test_time_training/ttt-self-supervised/notes.md) 和 [PAD](../../05_intersections/pad/notes.md)。
  - **Real2sim2real**：10M 帧的交互量只有在可以 reset、能快速步进的模拟器里才现实。frame skip 本身就是利用模拟器步进便宜来换取数据量。这种样本需求正是机器人 RL 要在仿真里训练，再用 [domain randomization](../../03_real2sim2real/domain-randomization/notes.md) 或大规模并行（[legged-gym](../../03_real2sim2real/legged-gym/notes.md)）迁移到真机的原因，也是用学到的 [Robotic World Model](../../05_intersections/robotic-world-model/notes.md) 充当 simulator 的动机。
  - **RL 内部**：DQN 代表 value-based / off-policy 一支；[PPO](../ppo/notes.md) 及其在 VLA/LLM 上的后继（[DPPO](../dppo/notes.md)、[SimpleVLA-RL](../simplevla-rl/notes.md)、[DeepSeek-R1](../deepseek-r1/notes.md) 的 GRPO）代表 on-policy policy gradient 一支。

## 前置知识

1. MDP、return、action-value $Q$、Bellman 最优方程；value iteration 与 Q-learning（Sutton & Barto 第 3–6 章）。
2. TD learning 与 bootstrapping；on-policy 与 off-policy 的区别； $\epsilon$-greedy 探索。
3. 函数逼近下 TD 的发散问题（Tsitsiklis & Van Roy 1997、Baird 1995），以及 semi-gradient 与 residual gradient 的区别。
4. CNN 基础（卷积、stride、ReLU）与 SGD/RMSProp。
5. POMDP 与 perceptual aliasing，以及为什么帧堆叠能近似 Markov 状态。

## 复现可行性

- **代码**：本文没有官方代码。最接近的官方实现是 Nature 版的 https://github.com/google-deepmind/dqn，已归档，README 要求 LuaJIT、Torch7、Xitari（ALE 的 fork）和 Linux，并且带 target network。现代社区实现如 CleanRL 的 `cleanrl/dqn_atari.py`（已确认存在）默认使用 target network 和 Adam。想复现 2013 版需要改动：去掉 target network（或设为每步同步）、换成 RMSProp、 $\epsilon$ 终值设为 0.1。
- **环境**：Gymnasium + `ale-py`（从 ale-py 0.9 起 ROM 已随 PyPI 包附带，不必再单独安装）。注意 `ALE/*-v5` 环境默认 `repeat_action_probability=0.25`（sticky action）、`frameskip=4`；本文没有提到 sticky action，复现时要显式设为 0，Space Invaders 还要把 frame skip 改成 3。
- **算力**：论文只给出每个 epoch（5 万次更新）约 30 分钟（Figure 2 图注），没有写 GPU 型号和总时长。网络很小（按无 padding 推算约 68 万参数），单张消费级 GPU 就够用，瓶颈在模拟器步进和 replay 内存：1M 条 $84\times84\times4$ 的 uint8 转移，如果不做帧共享，内存占用会很大。
- **最小可复现实验**：Pong，按 §5 的设定（RMSProp、batch 32、replay 1M、 $\epsilon$ 从 1 到 0.1 退火 1M、 $k=4$、reward clipping、 $84\times84\times4$ 输入、2 层卷积加 FC256），对照 Table 1 的 DQN 20 / Human −3。设三组：(a) 2013 原版，无 target network；(b) 加 target network；(c) 去掉 replay（容量等于 batch size）。同时画 Figure 2 的两种指标： $\epsilon=0.05$ 下的平均 reward，和固定随机状态集上的平均 $\max Q$。这样也补上了论文缺的 replay 消融。
- **预计难点**： $\gamma$、学习率和预热步数都要自己选；"帧"的口径不明确，会影响退火和 replay 的实际长度；论文没报 seed 数和方差，复现时至少要跑 3 个 seed 才能判断和 Table 1 是否一致；Seaquest、Q*bert 在论文里本身就离人类水平很远，不适合作为第一个复现目标。

## 自测问题

1. 为什么用了 experience replay 就必须 off-policy？如果把 Q-learning 换成 SARSA 直接用 replay 数据训练，target 里会出现什么不一致？
2. Eq.2 的 target 用冻结的 $\theta_{i-1}$，而 Algorithm 1 用当前的 $\theta$。两者在优化动力学上有什么区别？Algorithm 1 按 Eq.3 更新时，忽略了哪一项梯度？Nature 版的 target network 对应 Eq.2 里的哪个量？
3. 网络为什么对每个动作单独输出一个 Q 值，而不是把动作作为输入？这个设计换到连续动作空间会在哪一步失效（提示： $\max_{a'}$）？
4. 把 reward clipping 到 $\lbrace-1,0,+1\rbrace$ 后，Q 的数值范围和最优策略可能怎样变化？举一个会因此学错策略的游戏情形。
5. 为什么作者用"固定随机状态集上的平均 $\max_a Q$"来监控训练？这个指标平滑上升能说明什么，不能说明什么（比如高估、策略质量）？
