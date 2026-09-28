# Dream to Control: Learning Behaviors by Latent Imagination

> arXiv [1912.01603](https://arxiv.org/abs/1912.01603) · ICLR 2020 · 主题：01_world_model · 层级：foundational
> 代码：https://github.com/danijar/dreamer（论文正文给的入口是 https://danijar.com/dreamer，`sections/experiments.tex`）· 笔记依据：`source/*.tex`（数字/公式）+ `paper.pdf`（图）

## 一句话

在 PlaNet 的 RSSM latent space 里从真实经验出发 imagine $H$ 步轨迹，用 $\lambda$-return 训练 value model，再把 value 估计的**解析梯度**沿可微的 latent dynamics 反传给 actor。这样 policy 只靠 latent imagination 就能学到超出 imagination horizon 的长时程行为，不需要 online planning。

## 问题设定

- **任务**：visual control，建模为 POMDP（`sections/overview.tex`）。输入是 $64\times64\times3$ 图像，动作是 1–12 维连续向量，reward 在 $[0,1]$，每个 episode 1000 步，初始状态随机；所有任务统一用 action repeat $R=2$（`sections/experiments.tex`）。附录 C 还测了离散动作 + early termination 的 Atari / DMLab。
- **假设**：可以在线与环境交互、能 reset；reward 可观测；不给低维状态，只给像素；连续任务共用一套超参数，离散任务共用另一套。
- **之前的方法卡在哪**（`sections/intro.tex`，第 4 条出自 `sections/related.tex`）：
  1. 无论是用参数化 policy（Dyna、World Models、SOLAR）还是用 online planning（PETS、PlaNet）去最大化 imagined reward，只考虑固定 imagination horizon 内的 reward 都会导致行为**短视**。此外 PlaNet 每一步都要在线规划，训练耗时约 11 h / $10^6$ 步，Dreamer 约 3 h（`sections/experiments.tex`）。
  2. 现有方法为了对 model error 更鲁棒，多用 derivative-free 优化，没有利用神经网络 dynamics 本身能提供的解析梯度。
  3. model-free 方法（D4PG 等）样本效率很低，对比表里 D4PG 用了 $10^8$ 步。
  4. World Models 采用两阶段训练，控制器是在 imagination 里 evolve 出来的线性控制器。

## 核心方法

**1. World model（RSSM，沿用 PlaNet）**。`sections/dynamics.tex` Eq.9 定义了四个组件：

$$p_\theta(s_t\mid s_{t-1},a_{t-1},o_t)\ \text{(representation)},\quad q_\theta(s_t\mid s_{t-1},a_{t-1})\ \text{(transition)},\quad q_\theta(o_t\mid s_t),\quad q_\theta(r_t\mid s_t)$$

记号约定（`sections/overview.tex`）： $p$ 表示在真实环境中生成样本的分布（如用真实观测 $o_t$ 做 filtering 的 representation model）， $q$ 表示它们的近似，用于 latent imagination。transition 是 RSSM；representation 是 RSSM 加 CNN encoder；observation model 是 transposed CNN，只用来提供学习信号。latent 是 30 维对角 Gaussian，其余函数都是 3 层 300 单元的 ELU MLP（附录 A）。训练目标是 ELBO / VIB（Eq.10）：

$$\mathcal{J}_ {\mathrm{REC}}=\mathrm{E}_ p\Big(\sum_t \ln q(o_t|s_t)+\ln q(r_t|s_t)-\beta\thinspace\mathrm{KL}\big(p(s_t|s_{t-1},a_{t-1},o_t)\thinspace\Vert\thinspace q(s_t|s_{t-1},a_{t-1})\big)\Big)+\text{const}$$

连续任务取 $\beta=1$，KL 低于 3 free nats 时截断（附录 A）。附录 B 从 information bottleneck 推出这个界，KL 项的作用是限制每一步从当前图像里读入的信息量，逼模型更多依赖过去的状态，从而学到长程依赖。

**2. Imagination MDP**（`sections/behavior.tex`）。由于 $s_t$ 是 Markov 的，latent dynamics 本身就构成一个完全可观测的 MDP。从 replay 序列的 posterior 状态 $s_t$ 出发（每批是 $B=50$ 条、长度 $L=50$ 的序列，其中每个状态都作为起点，即每批 $50\times50=2500$ 个起点，这个数由附录 A 与 Algorithm 1 推出），交替执行 $a_\tau\sim q_\phi(a_\tau|s_\tau)$ 和 $s_\tau\sim q(s_\tau|s_{\tau-1},a_{\tau-1})$，并预测 $r_\tau$，一共 imagine $H=15$ 步。这个过程完全不解码图像，所以能并行 imagine 大量轨迹。

**3. Actor 与 critic**（Eq.2–3）。actor 输出 tanh-transformed Gaussian，并通过 reparameterization 采样：

$$a_\tau=\tanh\big(\mu_\phi(s_\tau)+\sigma_\phi(s_\tau)\thinspace\epsilon\big),\quad \epsilon\sim\mathcal{N}(0,\mathbb{I})$$

均值先过 tanh 再乘 5，标准差用 softplus（附录 A）。value model $v_\psi(s_\tau)$ 估计的是 state value，不是 Q 值。

**4. Value 估计**（Eq.4–6）：

$$\mathrm{V}_ {\mathrm R}(s_\tau)=\mathrm{E}\Big(\sum_{n=\tau}^{t+H}r_n\Big),\qquad
\mathrm{V}^k_{\mathrm N}(s_\tau)=\mathrm{E}\Big(\sum_{n=\tau}^{h-1}\gamma^{n-\tau}r_n+\gamma^{h-\tau}v_\psi(s_h)\Big),\ h=\min(\tau+k,t+H)$$

$$\mathrm{V}_ \lambda(s_\tau)=(1-\lambda)\sum_{n=1}^{H-1}\lambda^{n-1}\mathrm{V}^n_{\mathrm N}(s_\tau)+\lambda^{H-1}\mathrm{V}^H_{\mathrm N}(s_\tau)$$

$\mathrm{V}_ {\mathrm R}$ 只累加 horizon 内的 reward，是消融实验里的 "No value" 版本。 $\mathrm{V}_ \lambda$ 是对不同 $k$ 的 n-step 估计做指数加权平均，用来平衡 bias 和 variance，取 $\gamma=0.99,\lambda=0.95$。实现时常写成标准 TD($\lambda$) 递归： $\mathrm{V}_ \lambda(s_\tau)=r_\tau+\gamma[(1-\lambda)v(s_{\tau+1})+\lambda\mathrm{V}_ \lambda(s_{\tau+1})]$。注意这个递归式不是论文原文。

**5. 目标函数**（Eq.7–8）：

$$\max_\phi\ \mathrm{E}_ {q_\theta,q_\phi}\Big(\sum_{\tau=t}^{t+H}\mathrm{V}_ \lambda(s_\tau)\Big),\qquad
\min_\psi\ \mathrm{E}_ {q_\theta,q_\phi}\Big(\sum_{\tau=t}^{t+H}\tfrac12\big\Vert v_\psi(s_\tau)-\mathrm{V}_ \lambda(s_\tau)\big\Vert^2\Big)$$

这是整篇论文的关键。critic 回归 $\mathrm{V}_ \lambda$ 时对 target 做 stop-gradient。actor 的梯度则是**解析的**，链路为 $\mathrm{V}_ \lambda\to(\hat r_\tau, v_\psi(s_\tau))\to s_\tau\to a_{\tau-1}\to\phi$，用 stochastic backprop 穿过 transition 与 reward/value 网络，属于 BPTT 式的 pathwise gradient。behavior learning 阶段 world model 固定，不更新。离散动作用 straight-through gradients。

**6. 与经典 actor-critic 的区别**。A3C、[PPO](../../04_rl/ppo/notes.md) 用 REINFORCE 加 value baseline，Dreamer 则直接穿过 value model 反传梯度。DDPG（deterministic actor-critic）、[SAC](../../04_rl/sac/notes.md)（reparameterized actor-critic）虽然也穿过 critic 反传梯度，但只最大化一步的 Q 值，不穿过 transition；MVE、STEVE 用 learned dynamics 做多步 Q-learning target。Dreamer 的梯度穿过多步 dynamics，所以只需要 state value $V(s)$。

**7. Early termination 与离散任务**。world model 额外预测 discount：用二分类器拟合 soft label $0$ 或 $\gamma$，再用预测 discount 的累积乘积给 Eq.7–8 的各项加权。离散任务的其他设置： $H=10$， $\beta=0.1$，reward 过 tanh，探索用 $\epsilon$-greedy， $\epsilon$ 在前 200,000 个 gradient steps 内从 0.4 线性降到 0.1（附录 A）。

**8. 可替换的表征目标**。contrastive 版本（Eq.11–12）把 observation model 换成 state model $q_\theta(s_t|o_t)$，并用 NCE 估计 state marginal： $\mathcal{J}_ S^t=\ln q(s_t|o_t)-\ln\sum_{o'}q(s_t|o')$。reward-only 版本只预测 reward。

**训练与推理流程**（Algorithm 1，`algos/agent.tex`，数值见附录 A）：
- 先用随机动作收集 $S=5$ 个 episode。
- 之后循环：做 $C=100$ 个训练步，然后与环境交互 1 个 episode。每个训练步先用 replay 中采样的真实序列（ $B=50$ 条、长 $L=50$）更新 world model，再从这些序列的 posterior 状态出发 imagine，用同一批 imagined 轨迹同时更新 actor 和 critic（附录 A："the same trajectories are used to update both action and value models"）。
- 交互时逐步用 $p_\theta$ 对历史做 filtering 得到 $s_t$，执行 actor 的 mode 并加 $\mathcal{N}(0,0.3)$ 探索噪声。
- 优化器是 Adam，学习率为 world model $6\times10^{-4}$、value $8\times10^{-5}$、actor $8\times10^{-5}$；梯度范数超过 100 时裁剪。
- 作者发现 latent overshooting、entropy bonus、target network 都不需要。
- 推理时每步只需用 representation model（CNN encoder + RSSM）做一步 filtering 得到 $s_t$，再做一次 actor 前向，不做在线规划。

## 关键实验与证据

- **主结果**：DMC 20 个视觉任务（附录 G 表、Figure 6）。Dreamer 与重跑的 PlaNet 是 5 seeds 平均（Figure 6 caption），A3C、D4PG 的分数取自 Tassa et al. (2018)。

  | 方法 | 输入 | 环境步数 | 平均分（Dreamer/PlaNet 为 20 任务；A3C/D4PG 缺 Quadruped Run/Walk，只有 18 任务） |
  |---|---|---|---|
  | Dreamer | pixels | $5\times10^6$ | **823.39** |
  | D4PG | pixels | $10^8$ | 786.32 |
  | PlaNet（以 $R=2$ 重跑） | pixels | $5\times10^6$ | 332.97 |
  | A3C | proprio | $10^8$ | 243.70 |

  - **长程或稀疏任务优势明显**：Cartpole Swingup Sparse 上 Dreamer 812.22，PlaNet 0.64，D4PG 482.00；Acrobot Swingup 上 Dreamer 365.26，D4PG 91.70。
  - **Dreamer 明显落后 D4PG 的任务**：Finger Spin 498.88 vs 985.70；Reacher Hard 817.05 vs 957.10；Finger Turn Easy 825.86 vs 971.40。
- **算力**（`sections/experiments.tex`）：每个训练 run 用 1 张 V100 加 10 个 CPU 核。每 $10^6$ 环境步约 3 小时；PlaNet 需要 11 小时，D4PG 达到相近性能需要 24 小时。
- **Value model 的作用**（Figure 4，消融 imagination horizon）：在 Cartpole Swingup、Cheetah Run、Quadruped Walk、Walker Walk 上，horizon 从很短到 40 变化时，Dreamer（ $\mathrm V_\lambda$）对 horizon 更鲁棒（作者原话是 "more robust to the horizon length"），但并非处处稳定：horizon 极短时 Cartpole Swingup、Quadruped Walk 上也有下降，Walker Walk 在 $H=40$ 时明显下降、低于 No value。No value（ $\mathrm V_{\mathrm R}$）和 PlaNet 在 horizon 短时明显更差。Figure 7 的 caption 说，Dreamer 在 acrobot、hopper 这类需要长时程 credit assignment 的任务上成功，只优化 horizon 内 imagined reward 的做法（No value、PlaNet）只在 walker 这类 reactive 任务上成功。从 Figure 7/10 的曲线看：Acrobot Swingup、Cartpole Swingup Sparse 上 No value 和 PlaNet 几乎为 0；Hopper Hop/Stand 上 No value 能学到一部分但明显低于 Dreamer，PlaNet 接近 0；Walker Walk 上三者接近。正文称 Dreamer 在 20 个任务里 16 个胜出、4 个持平。
- **表征目标**（Figure 8/11）：reconstruction 在多数任务上最好。contrastive 大约解决一半任务，在 Finger Spin 上反而更好。reward-only "不够用"。作者据此认为，表征学习的进步很可能（"likely"）转化为 Dreamer 的性能提升。
- **离散动作与早停**（附录 C，Figure 9）：Atari 用 sticky actions，另测了 DMLab 的 Collect Good Objects 和 Watermaze。结果只以曲线呈现，没有汇总数字。作者自己的措辞是在"这个子集"的 Atari 游戏和 DMLab 的 object collection level 上学到了成功的行为（Watermaze 不在这句话里），并在附录 C 引 SimPLe 指出纯靠 world model 学习的 agent 在这些领域还没有竞争力。
- **长期预测质量**（Figure 5）：在两条 hold-out 轨迹上给 5 帧 context，只给动作做 open-loop 预测 45 步。作者称 RSSM 的长期预测准确；图中第一条轨迹在最后几步已能看出与真实帧有偏差。
- **Action repeat 敏感性**（Figure 12，2 seeds，使用旧版超参数，只跑到 $10^6$ 步）：作者结论是 $R=2$ 在各任务上整体最好。但从曲线看并非每个任务都最好，例如 Acrobot Swingup、Cartpole Swingup Sparse 上 $R=4$ 更高，Finger Spin 上 $R=1$ 更高。

## 局限与论文没说的

**作者承认的局限**：
- 在 Atari、DMLab 上，纯 world model agent 还不具竞争力。
- 表征学习是瓶颈，reward-only 不够，扩展到更复杂的视觉环境要靠更好的 representation learning（附录 C、Conclusion）。

**我读出来的问题**：
1. **比较口径不齐**：
   - D4PG 和 A3C 的分数直接抄自 Tassa et al.，与 Dreamer 的环境步数差 20 倍，而且 A3C 用的是 proprio 输入。
   - 平均分的任务集合不同：A3C/D4PG 没有 Quadruped Run/Walk，786.32 和 243.70 是 18 个任务的平均；Dreamer 的 823.39 是 20 个任务的平均，且包含它得分很高的两个 Quadruped 任务。只算 18 个共同任务时 Dreamer 约为 813.8（笔者按附录 G 表推算），仍高于 D4PG。
   - PlaNet 是以 $R=2$ 重跑的（没有按任务调 action repeat），脚注承认结果与 PlaNet 原文不同，所以 332.97 这个数可能偏弱。
   - Figure 6 图例写 D4PG 用 1e8 步，Figure 7/8/10/11/12 的图例却写 1e9 步（A3C 同样），与表格矛盾。
   - 正文一处写 "19 tasks with horizon of 20"，与附录 A 的 $H=15$ 和"20 任务"也对不上。
   - "16 胜 4 平"是看曲线得出的判断，没有做统计检验。
2. **缺少关键消融**：没有扫 $\lambda$；没有在 imagination 内把 analytic gradient 和 REINFORCE 做对比；也没有分析 actor 是否在利用 model error（model exploitation）。world model 没有 ensemble 或不确定性估计，梯度穿过一个有误差的可微模型，actor 有可能专门去钻模型漏洞。这一点论文完全没讨论。
3. **探索很弱**：只加了高斯噪声。困难探索任务没测，Atari 的曲线也显示多个游戏离 Rainbow 很远（Figure 9）。
4. **Setting 窄**：
   - 全部在仿真中进行，图像只有 $64\times64$，训练和测试环境同分布，只随机化了初始状态。
   - 没有 dynamics shift 或视觉 shift，也没有真机实验（真机要等 [DayDreamer](../../05_intersections/daydreamer/notes.md)）。
   - "同一套超参数"只在连续或离散各自内部成立，两者之间 $H$、 $\beta$、reward 处理都不一样。
5. **Reconstruction 目标的代价**：模型要花容量去重建与任务无关的像素，背景复杂时可能失效。这正是后续 decoder-free 路线（如 [TD-MPC2](../tdmpc2/notes.md)）的出发点。
6. **算力门槛不高但总量不小**：按"3 小时 / $10^6$ 步"推算，一个 $5\times10^6$ 步的 run 约 15 小时。20 任务 × 5 seeds 共 100 个 run，这是推算值，论文没给总 GPU 时数。

## 在脉络中的位置

- **前驱**：
  - [World Models](../world-models/notes.md)：两阶段训练加 imagination 里的 evolved controller。
  - PlaNet（Hafner et al. 2018，不在本项目中）：提供了 RSSM 和 reconstruction 目标，本文沿用。
  - Dyna（Sutton 1991）：learning in imagination 的框架。
  - deterministic / reparameterized actor-critic（DPG、DDPG、[SAC](../../04_rl/sac/notes.md)）与 SVG：pathwise value gradient 的来源。
  - MVE / STEVE：用学到的模型做多步 value target。
  - VAE 的 reparameterization。
- **后继 / 影响**：
  - "RSSM + imagination actor-critic"成为 Dreamer 系列的共同骨架。[DreamerV3](../dreamerv3/notes.md) 在此基础上加入各种稳健化技巧，扩展到 150 多个任务；[Dreamer 4](../dreamer4/notes.md) 把 world model 换成可扩展的 Transformer，并完全在离线数据学出的模型里做 RL。
  - [DayDreamer](../../05_intersections/daydreamer/notes.md) 把 Dreamer 直接用于真实机器人。
  - [Robotic World Model](../../05_intersections/robotic-world-model/notes.md) 把"在 learned simulator 里训 policy"用到腿足机器人，但 imagination 里的优化器换成了 PPO。
  - 在像素级视频 world model 内做 VLA RL 的 [WMPO](../../05_intersections/wmpo/notes.md) 和 [World-Gymnast](../../05_intersections/world-gymnast/notes.md)，也延续了"policy 在 world model 里训练"这一范式。
  - 对照路线：[TD-MPC2](../tdmpc2/notes.md) 不重建观测，并保留在线规划；[DIAMOND](../diamond/notes.md) 在像素空间用 diffusion 做 world model。
- **与本项目其他主题的联系**：
  - **RL**：Dreamer 用 analytic value gradient，[PPO](../../04_rl/ppo/notes.md) 用 likelihood-ratio gradient，[SAC](../../04_rl/sac/notes.md) 用一步 Q 梯度，这是理解三者 variance/bias 差异的好样本。 $\mathrm V_\lambda$ 和 PPO 里 GAE 用的 $\lambda$ 加权是同一个 TD($\lambda$) 思想。
  - **real2sim2real**：world model 本质上是从环境交互数据学出来的 learned simulator（本文的"环境"本身是 DMC 仿真，不是真机），imagination 就相当于"在仿真里训练"。sim-to-real gap 在这里表现为 model error。Dreamer 在线持续收集数据、重训模型来缩小 gap，没有用 [domain randomization](../../03_real2sim2real/domain-randomization/notes.md) 或显式的 system identification。可对照 [survey-reality-gap](../../03_real2sim2real/survey-reality-gap/notes.md)，以及"仿真预训练 world model、真实环境只更新 dynamics"的 [SimDist](../../05_intersections/simdist/notes.md)。
  - **TTT**：Dreamer 的 world model 在整个学习过程中一直在线更新（Algorithm 1 交替进行交互和训练），但论文不区分训练和部署阶段，也没有针对 test-time 分布偏移的显式适应机制。它的 reconstruction 和 KL 损失是现成的 self-supervised 目标，可以对照 [PAD](../../05_intersections/pad/notes.md) 和 [TTT](../../02_test_time_training/ttt-self-supervised/notes.md) 的思路，在部署时更新 encoder 或 dynamics。这个方向论文没有做，属于开放问题。

## 前置知识

1. POMDP / MDP，return 与 discount。
2. VAE：ELBO、KL、reparameterization trick。
3. PlaNet 的 RSSM：确定性 RNN 路径加随机 latent，filtering（posterior）与 prior 的区分。
4. Actor-critic 与 TD($\lambda$) / $\lambda$-return 的 bias–variance 权衡。
5. REINFORCE（score function）与 pathwise（reparameterized）两种梯度估计器的区别，BPTT。
6. SAC 的 tanh-Gaussian actor（DDPG 是 deterministic actor）。
7. straight-through estimator（离散动作）；InfoNCE（contrastive 变体，选读）。
8. DMC benchmark 与 action repeat。

## 复现可行性

- **代码**：官方仓库 https://github.com/danijar/dreamer（已用 WebFetch 确认存在，MIT 许可）。依赖 TensorFlow 2.2（GPU）、TensorFlow Probability、dm_control，另有 pandas 和 matplotlib。README 中的运行命令是 `python3 dreamer.py --logdir ... --task dmc_walker_walk`，同时说明已有支持 Atari 与 DMC 的 DreamerV2 代码。
- **算力**：论文写明每个 run 用 1×V100 加 10 CPU 核，约 3 h / $10^6$ 环境步（`sections/experiments.tex`）。
- **数据与环境**：不需要离线数据，只需 MuJoCo + dm_control；在服务器上跑还需要配置无头渲染（EGL 或 OSMesa，这是 dm_control 的通用要求，仓库 README 未提）。
- **最小复现实验**（单卡，约 1–2 天）：
  1. 选两个任务：walker_walk（reactive）和 cartpole_swingup_sparse 或 acrobot_swingup（长时程）。
  2. 对比 Dreamer（ $\mathrm V_\lambda$）与 No value（ $\mathrm V_{\mathrm R}$，关掉 critic），每个任务跑 $10^6$ 步、3 seeds。
  3. 预期现象来自 Figure 7/10：walker_walk 上两者接近，稀疏或长时程任务上 No value 几乎学不会。
  4. 可加一组 $H\in\lbrace5,15,40\rbrace$，看 Dreamer 对 horizon 是否鲁棒（对应 Figure 4）。
- **预计难点**：
  - TF 2.2 与 CUDA / cuDNN 的老版本兼容，新机器上可能需要 docker 或降级。
  - 无头 MuJoCo 渲染。
  - 结果对 free nats、梯度裁剪、探索噪声敏感，seed 方差大。
  - 若改用 PyTorch 社区实现，需要逐项核对附录 A 的超参数，否则比较会失去意义。

## 自测问题

1. Dreamer 的 critic 为什么只学 state value $v(s)$，而 DDPG/SAC 必须学 $Q(s,a)$？写出 actor 梯度从 $\mathrm V_\lambda$ 流回 $\phi$ 的完整路径。
2. 为什么 critic 的 target $\mathrm V_\lambda$ 要 stop-gradient，而 actor 的目标不能 stop？behavior learning 阶段如果不固定 world model 参数，会发生什么？
3. 从 bias–variance 的角度比较 $\mathrm V_{\mathrm R}$、 $\mathrm V^k_{\mathrm N}$、 $\mathrm V_\lambda$，并用它解释 Figure 4：为什么 No value 和 PlaNet 在短 horizon 下崩掉，Dreamer 受的影响却小得多？
4. imagined 轨迹为什么从 replay 序列中每个 posterior 状态 $s_t$ 出发，而不是只从 episode 初始状态出发？ $p_\theta$ 和 $q_\theta$ 在训练与 imagination 中各自扮演什么角色？
5. 把 Dreamer 用到离散动作加 early termination 的任务上，需要改哪三处？（提示：动作梯度估计器、discount 预测与加权、KL/horizon 等超参数）
