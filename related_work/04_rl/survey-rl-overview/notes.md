# Reinforcement Learning: An Overview

> arXiv [2412.05265](https://arxiv.org/abs/2412.05265) · arXiv 预印本（Kevin P. Murphy 单作者的书式综述；本笔记依据 v5，2025-12-01，253 页）, 2024 · 主题：04_rl · 层级：survey
> 代码：综述没有配套代码库。§6.4 的 Tinker 示例代码在 https://github.com/probml/pyprobml/blob/master/scripts/tinker_train_gem.py（已核实存在，改自 GEM 仓库 https://github.com/axon-rl/gem）· 笔记依据：`source/*.tex`（数字/公式）+ `paper.pdf`（图）

**本笔记覆盖的章节**
- 精读：Ch1 全章（`intro.tex`）；Ch4 MBRL 除 §4.5 successor representation 以外的全部内容（`model-rl.tex`、`planning.tex`）；§2.5 DQN 系列（`value-rl.tex`）；§3.1–3.3、§3.6.4、§3.6.8（`policy-rl.tex`）；§6.2.1–6.2.5 的部分、§6.3.3–6.3.5、§6.4（`llm.tex`、`code.tex`）；§7.3.2、§7.6、§7.7（`other-rl.tex`、`offline.tex`）。
- 只看了目录：Ch5 MARL、§3.4–3.5、§7.1 regret、§7.2 exploration、§7.4 intrinsic motivation、§7.5 HRL、§7.8 AIXI。

## 一句话

这是一本约 250 页的教科书式综述。它先用一张 agent–environment 通用图模型（Fig 1.2）统一 POMDP、MDP、contextual MDP 等设定，再按"学什么（value / policy / model）"分章讲算法，world model 和 LLM×RL 也用同一套记号。对本项目最有用的是 Ch4：它按"训练目标 × 使用方式"给 world model 分类（Table 4.1/4.2），并用博弈论解释 model 与 policy 交替训练为什么行得通。

## 问题设定

- **核心问题**：环境模型未知时，求一个 policy 使期望累积回报最大，即 $V_\pi(s_0)=\mathbb{E}[\sum_t R(s_t,a_t)\mid s_0,\pi]$、$\pi^*=\arg\max_\pi\mathbb{E}_{p_0}[V_\pi(s_0)]$（`intro.tex` §1.1.1，eqn:valueFn）。
- **通用模型**（§1.1.3，Fig 1.2，扩展自 Powell 的 universal modeling framework）：
  - 环境侧：hidden state 按 $w_{t+1}=M(w_t,a_t,\epsilon^w_t)$ 演化，发出观测 $o_{t+1}=O(w_{t+1},\epsilon^o_{t+1})$。
  - agent 侧：内部状态 $z_t$ 分两步更新，先 predict 再 update：
    $$z_{t+1}=U\big(P(z_t,a_t),\,E(o_{t+1})\big)$$
    其中 $E$ 是 encoder。可选再加一个 decoder $\hat o_{t+1}=D(z_{t+1|t})$。
  - policy 为 $a_t\sim\pi(z_t;\theta_t)$，参数由学习算法按较慢的时间尺度更新：$\theta_t=\mathcal{A}(\theta_{t-1},a_t,z_t,r_t)$。
  - 所以系统里有三个相互作用的随机过程：环境状态、agent 的 belief、policy 参数。后面各种 world model 都是这张图的特例：$P$ 对应 latent dynamics，$E$ 对应 encoder，$D$ 对应 decoder。
- **典型设定**（§1.2）：
  - POMDP：belief state $b_t=p(w_t\mid h_t)$ 是最优 policy 的充分统计量，但精确求解"wildly intractable"。
  - MDP：$w_t=o_t=s_t$ 的特例。
  - goal-conditioned MDP。
  - contextual MDP：动力学和奖励依赖一个隐藏的静态 context，例如程序化生成的关卡。它本质上是一种 POMDP，Ghosh2021 称之为 epistemic POMDP。
  - contextual bandit、belief-state MDP。
- **作者列出的跨章难点**（§1.3.4–1.3.6、§4.1）：
  - partial observability：frame stacking 抓不住长程依赖；RNN policy 一般不会主动做信息收集动作，除非经过 meta-learning。
  - exploration–exploitation。
  - reward 设计：non-Markovian reward、reward hacking、sparse reward、shaping。
  - model-free 样本效率低：OpenAI 的机械手需要 100 年的模拟数据。
  - function approximation 下训练不稳定（deadly triad，§2.5.2.5）。

## 核心方法

这篇综述没有提出方法，这一节整理它的组织框架。

**1. 三轴分类**（§1.3，Table 1.1）

综述从三个维度给 RL 方法分类：
- 学什么：value、policy、model，或它们的组合。
- 怎么表示：tabular，或参数化（用神经网络即 deep RL）。
- 动作由谁产生：on-policy 或 off-policy。

Table 1.1 的归类：

| 方法 | 学的函数 | on/off-policy |
|---|---|---|
| SARSA | $Q$ | on |
| Q-learning | $Q$ | off |
| REINFORCE | $\pi$ | on |
| A2C | $\pi, V$ | on |
| TRPO/PPO | $\pi, A$ | on |
| DDPG | $a=\pi(s),\ Q$ | off |
| SAC | $\pi, Q$ | off |
| MBRL | $p(s'\mid s,a)$ | off |

Ch2–4 按第一轴分章（Ch2 value-based、Ch3 policy-based、Ch4 model-based），之后是 Ch5 MARL、Ch6 LLMs and RL、Ch7 其他（regret、exploration、distributional、intrinsic、HRL、imitation、offline、AIXI）。

**2. Value-based**（Ch2）

Bellman 最优方程（§1.3.1）：
$$Q^*(s,a)=R(s,a)+\gamma\,\mathbb{E}_{s'}\big[\max_{a'}Q^*(s',a')\big]$$
由它得到 TD 更新 $Q\leftarrow Q+\eta[r+\gamma\max_{a'}Q(s',a')-Q]$（完整写法见 Algorithm 1，Tabular Q-learning）。

Deadly triad（§2.5.2.5）：function approximation、bootstrapping、off-policy 三者同时出现时训练会不稳定。根本原因是 bootstrapping 并不在最小化一个固定的目标函数。target network、gradient TD、two time-scale、layer norm 等（§2.5.2.6–2.5.2.10）是书中列出的针对它的解法；double Q-learning / double DQN（§2.5.3）针对的是另一个问题 maximization bias，不是 deadly triad。

**3. Policy-based**（Ch3）

- **Policy gradient theorem**（`policy-rl.tex` eqn:PGthm，§3.1.4）：
  $$\nabla_\theta J=\mathbb{E}_{\rho^\gamma_\pi(s)\,\pi_\theta(a|s)}\big[Q^{\pi_\theta}(s,a)\,\nabla_\theta\log\pi_\theta(a|s)\big]$$
  减去 baseline $b(s)=V(s)$ 就得到 advantage 形式（eqn:PGthmBaseline）。
- **GAE**（eqn:GAE，§3.2.2）：
  $$\delta_t=r_t+\gamma v_{t+1}-v_t,\qquad A_t=\delta_t+\gamma\lambda A_{t+1}$$
  $\lambda$ 控制 bias 与 variance 的权衡。
- **PPO**（eqn:PPOupdate，§3.3.3）：
  $$\max_\pi\ \mathbb{E}_{p_{\pi_k}}\big[\min(\rho_k A^{\pi_k},\ \mathrm{clip}(\rho_k,1-\epsilon,1+\epsilon)A^{\pi_k})\big],\qquad \rho_k=\frac{\pi(a|s)}{\pi_k(a|s)}$$
  Algorithm 9 在 actor-critic + GAE 的基础上用 clipped ratio，并对每批 rollout 做 $M$ 次更新。
- **扩散 policy 的 PPO**（§3.3.3.2）：DPPO 把每个去噪步当作一个 inner MDP；NCDPO 固定噪声后直接穿过整条去噪链反传。
- **RL as inference**（§3.6）：把先验固定为均匀分布的 KL control 就是 maxent RL。SAC（§3.6.8）的目标是 $\mathbb{E}[R+\alpha\mathcal{H}(\pi(\cdot|s))]$，soft value 为 $V(s)=\mathbb{E}_{a\sim\pi}[Q-\alpha\log\pi]$。tabular 情形下可以解出 $\pi^*\propto\exp(Q^*/\alpha)$、$V^*=\alpha\log\sum_a e^{Q^*/\alpha}$（eqn:softValueFn）。

**4. Model-based**（Ch4，本项目重点）

**(a) 两种用法**（§4.1）

- **decision-time planning**：每一步都从当前状态向前搜索（MPC、MCTS）。优点是可以在 reward-free 数据上训练 world model，再针对任意 reward 做规划；缺点是慢。
- **background planning**：用 world model 生成 imagined trajectories，拿来训练一个反应式 policy，相当于把规划的开销摊销掉。
- Algorithm 12（MBRL agent）把所有组合写进同一个循环：model 更新有 online / replay 两种，policy 更新有 on/off-policy × 真实数据/想象数据四种。

MPC 的目标（eqn:MPC，§4.2.4）：
$$a^*_{t:t+d}=\arg\max\ \mathbb{E}_{\mathcal{T}}\Big[\sum_{h=0}^{d}R(s_{t+h},a_{t+h})+\hat V(s_{t+d+1})\Big]$$
只执行第一个动作，然后重新规划。
- 在随机环境中，open-loop 规划是次优的（§4.2.4.1）。
- 常用求解器：LQR/DDP、random shooting、CEM、MPPI。TD-MPC 先用 policy prior 生成 $J<K$ 条候选轨迹，其余从高斯中采样。
- MuZero 靠 reward、value、policy 三种预测来训练 latent model（eqn:MZloss）。EfficientZero 在此基础上加了 self-prediction loss；书中转述 EfficientZero 原文的结果：三处改动（self-prediction、value prefix、Reanalyse）都有帮助，其中 self-prediction 收益最大（§4.2.2.2）。

**(b) 博弈视角：为什么 model 和 policy 交替训练能收敛**（§4.3.1，Rajeswaran2020）

把 MBRL 写成两个玩家的博弈：
$$\underbrace{\max_\pi J(\pi,\hat M)}_{\text{policy player}},\qquad \underbrace{\min_{\hat M}\ \mathbb{E}_{(s,a)\sim\mu^{\pi}_{M}}\mathrm{KL}\big(M(\cdot|s,a)\,\|\,\hat M(\cdot|s,a)\big)}_{\text{model player}}$$
- 关键在于：model loss 定义在当前 policy 于**真实环境**中诱导的访问分布 $\mu^\pi_M$ 上。
- 在 Nash 均衡处，policy 在真实环境中接近最优。与最优 policy 的差距由各个 $\epsilon$ 以及 $\mu^{\pi^*}_{M}$ 与 $\mu^{\pi^*}_{\hat M}$ 之间的 TV 距离决定。
- 两个玩家同时做梯度更新（GDA）不稳定，因此改成 Stackelberg 博弈，有两种排序：
  - PAL（policy as leader）：先拟合一个局部模型，再对 policy 做小步的保守更新，例如 natural actor-critic（§3.2.4）或 TRPO。
  - MAL（model as leader）：先在当前模型里把 policy 训练到最优，再用 data aggregation 保守地更新模型。MAL 得到的模型覆盖范围更"全局"。

Dyna（§4.3.2）的 policy 同时用真实数据和想象数据训练，$\pi_{k+1}=\pi_k+\eta\nabla J(\pi_k,\hat D_k\cup D_k)$，所以它是 model-free 与 model-based 的混合。MBPO = Dyna + SAC + 模型 ensemble（ensemble 思路来自 PETS），并用短 rollout 抑制 compounding error（§4.4.4.1）。

**(c) 按训练目标给 world model 分类**（§4.4，Table 4.1 / 4.2）

- **预测观测（generative WM）**。
  - latent 形式是一个 controlled HMM（eqn:unrollLatent）：
    $$\prod_i \pi(a_i|z_i)\,\mathcal{M}(z_{i+1}|z_i,a_i)\,R(r_i|z_{i+1},a_i)\,D(o_i|z_i)$$
    用 amortized VI 训练，即学一个 encoder $E(z_t|o_t)$ 近似后验。
  - Dreamer（§4.4.1.3，Fig 4.5）的 world model 损失：
    $$\mathcal{L}^{WM}=\mathbb{E}_q\Big[\sum_t -\beta_o\ln D(o_t|z_t,h_t)+\beta_z\,\mathrm{KL}\big(E(z_t|h_t,o_t)\,\|\,P(z_t|h_t)\big)\Big]$$
    actor-critic 部分用 λ-return $G^\lambda_t=r_t+\gamma((1-\lambda)V(h_t)+\lambda G^\lambda_{t+1})$ 作为目标。
  - 缺点（§4.4.1）：要重建与任务无关的干扰因素；推理慢；训练与测试时的观测过程可能存在分布偏移。
- **预测其他目标（non-generative WM）**。
  - 动机是 objective mismatch（§4.4.2.1）：容量有限的模型为了降低 MSE，会优先预测背景像素，而不是对控制重要的小物体。
  - Table 4.2 有三列（Table 4.1 则是"观测预测 / value+self prediction"× "background planning / online planning / exploration"）：
    - 损失：OP（observation）、ZP（latent state prediction）、RP（reward）、VP（value）、PP（policy）、POP（partial observation）。
    - policy 的输入：observables 或 latents。
    - 用法：Dyna（background planning）、MCTS/MPC（decision-time planning）、Aux（只作辅助损失，单阶段端到端训练）。
  - 代表方法的归类：
    - Diamond、Delta-IRIS：OP，policy 输入为 observables，用于 Dyna。
    - Dreamer：OP，policy 输入为 latent，用于 Dyna。
    - MuZero：RP+VP+PP，用于 MCTS。
    - EfficientZero：MuZero 的三项再加 ZP。
    - TD-MPC：RP+VP+ZP，用于 MPC-CEM。
    - DreamingV2：VP+ZP，用于 Dyna。
    - Denoised MDP：POP。
  - Ni2024 证明：同时满足 ZP 和 RP 的表征就满足 value equivalence，即对 $Q^*$ 是充分的（§4.4.2.3）。
  - self-prediction 用 EMA 目标来防止表征 collapse（eqn:EZP，§4.4.2.7）：
    $$\mathcal{L}_{\rm EZP}=\|M_\theta(E_\phi(h,a))-\mathrm{sg}(E_{\bar\phi}(h'))\|_2^2,\qquad \bar\phi_t=\rho\phi_t+(1-\rho)\bar\phi_{t-1}$$
    BYOL、JEPA、V-JEPA、SPR、EfficientZero 都属于这一类。线性 encoder 下已证明不会 collapse（BYOL-π）；非线性情形下是否收敛仍是 open problem。另一条防 collapse 的路线是信息论正则（VICReg 类）。
  - TD-MPC2（§4.4.2.12）的训练损失：
    $$\mathcal{L}=\mathbb{E}\sum_{t=0}^{H}\lambda^t\Big(\|z'_t-\mathrm{sg}(E(o'_t))\|^2+\mathrm{CE}(\hat r_t,r_t)+\mathrm{CE}(\hat q_t,q_t)\Big)$$
    reward 和 Q 离散化后用 cross-entropy 训练，做法对应 §7.3.2 的 two-hot / HL-Gauss。policy prior 在 latent rollout 上用 SAC 式目标训练，并作为 MPPI 的 proposal。
- **与 planning 联合训练**（§4.4.3–4.4.4）：
  - differentiable planning：VIN、TreeQN、D-TSN，都是双层优化。
  - 用统一的变分下界同时优化 model 和 policy：MNM、Aligned Latent Models。
- **应对模型误差**（§4.4.4）：从真实访问过的状态开始 rollout、用短 rollout、trajectory-level 扩散模型、PILCO/GP-MPC 传播不确定性、在 uncertainty set 上优化最坏情况、在 MBRL 与 MFRL 之间动态切换。

**5. LLMs and RL**（Ch6）

- **LLM 生成即 MDP**（eqn:LLMMDP）：转移是确定性的，$s_t=\mathrm{concat}(s_{t-1},a_t)$。单轮任务等价于 contextual bandit；奖励只在终点给出，并对 $\pi_{\rm ref}$ 加 KL 惩罚。
- **GRPO**（§6.2.4）：每个 prompt 采样 $J$ 个回答（通常约 8 个），组内归一化得到 advantage $\hat A^j_n=(R^j_n-\mu_n)/\sigma_n$；目标是 token 级 clip，并按序列长度归一化。
- **DAPO**：非对称 clip，$\epsilon_{\rm high}=0.28>\epsilon_{\rm low}=0.2$，用来防止 entropy collapse。
- **GSPO**：把 importance ratio 改成序列级。
- **多轮任务的 ReBN**（§6.2.5、§6.4.3）：$A^n_t=(G^n_t-\mu)/\sigma$，其中 $\mu,\sigma$ 取自 batch 内所有步的 return。
- **反方向 "LLMs for RL"**（§6.3）：LLM 可用于输入预处理、生成 reward（§6.3.2：Motif 用 LLM 给偏好标注做 AI feedback、Eureka 让 LLM 写 reward 代码、code as reward、直接用 VLM 定义 reward；MineCLIP 不在这里，而在 §1.3.6.2）、充当 world model（UniSim、code world model、GIF-MCTS）、充当 policy，以及 in-context RL。

## 关键实验与证据

综述没有在统一 benchmark 上做对照实验，书中的数字都是转述原论文。作者自己跑的 RL 训练实验只有 §6.4.6 的 Tinker 示例（另有 Fig 7.1 这类由作者 notebook 生成的 Thompson sampling 示意图）。

- **样本效率动机**（§4.1）：
  - Rainbow 需要数百万帧，人类几分钟就能达到同等水平；OpenAI 的机械手用了 100 年模拟数据。
  - 所引理论：Zhu2024 指出学 dynamics 的样本复杂度低于学 policy（这是综述原文的说法；所引论文的标题讲的是 representation complexity）；脚注引 Richens2025，称能完成任意多步 goal-conditioned 任务的 agent 必然隐式地学到了 world model。
- **DQN 系列**（§2.5.4，Fig 2.9）：
  - Rainbow 合并了 6 项改进：double DQN、PER、C51、n-step（$n=3$）、dueling、noisy nets。它是 2018 年 Atari-200M 上的 SOTA，每项改进的边际收益递减。
  - BBF 是 2024 年 Atari-100k 上的 SOTA。最重要的两项是更大的 Impala ResNet 和更高的 UTD；其余还有 soft reset、$n$ 从 10 退火到 3、$\gamma$ 从 0.97 升到 0.997、SPR 损失。
  - Agent57 用了 800 亿帧，MEME 把它降到 2 亿帧。Atari-100k 只包含 26 个游戏。
- **Deadly triad**（Fig 2.6）：Baird counterexample 有 7 个状态，$\gamma=0.99$，线性参数 $w\in\mathbb{R}^8$。behavior policy 按 6/7 与 1/7 的比例选两种动作时，TD(0) 发散；on-policy 的线性 TD(0) 则能保证收敛。
- **World model**（§4.4.1.3）：
  - DreamerV2 是第一个在 Atari 上超过人类的 imagination-based agent。
  - DreamerV3 首次在没有人类数据的情况下在 Minecraft 挖到钻石，用了 17 天训练。
  - DreamerV4 在 2500 小时带动作标注、无 reward 的 VPT 视频上离线训练 world model，用 shortcut 技巧把扩散采样降到 1–2 步。policy 先用 behavior cloning 初始化，再完全在 imagination 中用 actor-critic 微调，不混入真实环境数据。
  - DinoWM（§4.4.2.11）：预训练表征有时有帮助，有时从头训练反而更好。
- **Open-loop 规划的例子**（§4.2.4.1，Fig 4.3）：open-loop 下，先向上的两种方案期望回报都是 15，先向下的两种都是 20；closed-loop 可以保证 30。
- **Tinker 实验**（§6.4.6，Fig 6.11）：
  - 任务：GEM 的 `GuessTheNumber-v0-easy`，数字范围 1–10，最多 4 回合。最优策略是二分查找，期望回合数为 $\log_2 10=3.32$。
  - 配置：Qwen3-4B-Instruct-2507，LoRA rank 8，学习率 $4\times10^{-5}$，$\gamma=0.9$，每步 32 个 episode，共 20 步 ReBN。在笔记本上约 20 分钟。
  - 结果：平均 reward 从 0.8 升到 1.0；每回合 action token 数从约 900 降到约 500；回合数始终在 3 左右，因为 base model 本来就会二分；token entropy 逐步下降。
- **实验规范**（§1.3.7）：RL 实验方差大，推荐报告 IQM（0.25–0.75 分位的截尾均值），并用 bootstrap 或高斯近似给出置信区间（Agarwal2021）。
- **书中提到的 benchmark**：
  - 在线：ALE / Atari-200M / Atari-100k、CALE（连续动作）、MineDojo。
  - offline：D4RL（原文误拼为 "DR4L"）、RL Unplugged、OGBench、D5RL。
  - 其他：程序化生成环境（用于 contextual MDP 的泛化评测）、GEM（LLM agent 环境）。

## 局限与论文没说的

**作者承认的局限**
- LLM×RL 发展太快，书里只 "briefly mention a few highlights"（§6.2）。
- self-predictive world model 在非线性 encoder 下的收敛性（§4.4.2.7）、以及把基于 rate reduction 的 CTRL 博弈推广到序列数据（§4.4.2.9），都是 open problem。
- generative world model 的三个问题：要建模干扰因素、慢、训练与测试之间存在观测分布偏移（§4.4.1）。
- GRPO 需要环境能 reset，无法对中间状态做 credit assignment，统计效率低（§6.2.4 脚注）。
- offline RL 本质上比 online RL 受限（§7.7.4）。

**我读出来的问题**
1. **几乎不涉及 real2sim2real**。检索全部 `.tex`，找不到 domain randomization、sim-to-real、system identification、digital twin。唯一相关的是 §4.1 提到机械手用了 100 年模拟数据，但没说它依赖 DR。这条线请看 [Reality Gap 综述](../../03_real2sim2real/survey-reality-gap/notes.md)。
2. **TTT 只是擦边提及**。§6.3.5.2 一句话把 ATLAS、MesaNet 归为"基于 test-time training 的现代 RNN"；§6.2.4 讲 test-time compute 和 inference-time scaling（从 tilted distribution 采样）。完全没有 TTA，例如 Tent 或 PAD 那种部署时的自监督适应。
3. **机器人 RL 的前沿缺席**。VLA 只在脚注里给了定义；没有 π_RL、SimpleVLA-RL、HIL-SERL；offline RL 部分讲了 CQL、MOPO、MOREL，但没有 IQL。
4. **没有统一复评**。数字都转述自原论文；"At the time of writing this document (2024)" 这类 SOTA 表述（§2.5.4.7 BBF）会过时。书是单作者且持续改版，章节号在不同版本间会变，引用时应注明版本。
5. **部分内容被 `\eat{}` 隐藏**，在 PDF 里看不到，例如 PAL/MAL 的实验对比、Unified RL 的实验结论。这些不应引用。
6. **Table 4.1/4.2 的分类维度只有"训练目标/损失 × 用法"（Table 4.2 另加 policy 输入一列）**，没有涉及规模（数据量、参数量），也没有讨论怎样评测 world model 作为 simulator 的保真度，而后者正是 Cosmos、Genie、Ctrl-World 关心的问题。
7. **几处小瑕疵**：§1.3.1 的 Q-learning 更新式漏写了 "$Q(s,a)+\eta$"；Dreamer 一节把 λ-return 叫作 GAE。

## 在脉络中的位置

- **前驱**：Sutton & Barto 的教科书；Kochenderfer 等人的 *Algorithms for Decision Making*（§4.2 的 planning 部分基于其第 9 章）；Powell 的 universal modeling framework（Fig 1.2 由它扩展而来）；Levine 等人的 offline RL 教程；Ni2024 的 self-predictive RL（§4.4.2 的部分内容以它为基础）。
- **后继 / 影响**：这是一本入门参考书，不是方法论文，因此谈不上"后继方法"。本项目中被它直接讨论的工作有：
  - world model：[World Models](../../01_world_model/world-models/notes.md)、[Dreamer](../../01_world_model/dreamer/notes.md)、[DreamerV3](../../01_world_model/dreamerv3/notes.md)（书中引的是 2025 年 Nature 版）、[Dreamer 4](../../01_world_model/dreamer4/notes.md)、[TD-MPC2](../../01_world_model/tdmpc2/notes.md)、[DIAMOND](../../01_world_model/diamond/notes.md)、[V-JEPA 2](../../01_world_model/vjepa2/notes.md)。
  - 交叉线：[DayDreamer](../../05_intersections/daydreamer/notes.md)、[UniSim](../../05_intersections/unisim/notes.md)。
  - RL：[PPO](../ppo/notes.md)、[SAC](../sac/notes.md)、[DPPO](../dppo/notes.md)、[DeepSeek-R1](../deepseek-r1/notes.md)、[DQN](../dqn/notes.md)（书中引的是 2015 年 Nature 版）。
- **与本项目其他主题的联系**：
  - **World model**：Ch4 是 01 线的理论底座。
    - §4.3.1 的博弈视角说明了为什么 [Robotic World Model](../../05_intersections/robotic-world-model/notes.md)、[WMPO](../../05_intersections/wmpo/notes.md)、[World-Gymnast](../../05_intersections/world-gymnast/notes.md) 这类"在 world model 里训 policy，再回真实环境收数据"的循环，要求模型在 policy 的访问分布上足够准确。
    - objective mismatch 解释了 TD-MPC2 和 V-JEPA 2 为什么选择不带 decoder。
    - 书中把 UniSim 的用法归为"图像空间中的 MPC + random shooting"。
  - **TTT**：
    - contextual MDP / epistemic POMDP（§1.2.4）可以作为 test-time adaptation 和 [RMA](../../03_real2sim2real/rma/notes.md) 式在线辨识的形式化框架（这是我的连接）。
    - decision-time planning 本身就是一种 test-time compute（我的连接；书中 §6.2.4.11 只指出 LLM 的 SMC steering 与 §4.2.3 的 SMC 规划密切相关）。
    - §6.3.5.2 把基于 TTT 的现代 RNN（书中点名的是 ATLAS、MesaNet）和 xLSTM、Mamba 一起列为 world model 或 policy 的候选 backbone。[TTT layers](../../02_test_time_training/ttt-layers/notes.md) 和 [Titans](../../02_test_time_training/titans/notes.md) 属于同一类，但书中没有引用这两篇。
  - **Real2sim2real**（以下都是我的连接，书里没有这样写）：MAL 的 data aggregation、DAgger（§7.7.4.2）、MOPO/MOREL 的不确定性惩罚、在 uncertainty set 上优化最坏情况的 policy，这些都可以作为 [SimOpt](../../03_real2sim2real/simopt/notes.md) 和 [ASAP](../../03_real2sim2real/asap/notes.md)"用真实数据校准仿真、对模型误差保持保守"的理论语言；[domain randomization](../../03_real2sim2real/domain-randomization/notes.md) 本质上就是在 contextual MDP 的 context 分布上求泛化。
  - **RL 线内部**：
    - PPO + GAE 被 [legged-gym](../../03_real2sim2real/legged-gym/notes.md) 采用。
    - GRPO 被 [SimpleVLA-RL](../simplevla-rl/notes.md) 和 [TTRL](../../02_test_time_training/ttrl/notes.md) 采用。
    - offline-to-online（§7.7.4）与 [π\*0.6](../pi-star-06/notes.md)、[HIL-SERL](../hil-serl/notes.md) 的设定相近（我的连接）。

## 前置知识

1. MDP/POMDP、Bellman 方程、动态规划（value iteration、policy iteration）。
2. MC 与 TD、bootstrapping、function approximation 下的不稳定性。
3. score-function 梯度估计与 baseline 方差缩减。
4. 概率图模型与变分推断（ELBO、amortized VI、KL），这是 Dreamer 和 RL as inference 的基础。
5. 自监督表征学习：stop-gradient、EMA target（BYOL/JEPA）。
6. 基础最优控制：LQR、MPC、CEM、MPPI。
7. 基础博弈论：Nash 均衡、Stackelberg 博弈（§4.3.1 和 Ch5 要用）。
8. Transformer 生成与 KV cache（Ch6 要用）。

## 复现可行性

- **推荐工具**（Table 1.3，综述本身没有代码库）：
  - JAX：Stoix（含 MBRL）、PureJaxRL、JaxRL、Stable Baselines Jax、Jax Baselines、Rejax、Dopamine、Rlax、Acme。
  - PyTorch：CleanRL、Stable Baselines 3、TianShou（含 offline RL）。
- **评测口径**：IQM + bootstrap 置信区间（§1.3.7）；样本效率用 Atari-100k，offline 用 D4RL / OGBench。
- **书里现成的最小实验**：§6.4 的 Tinker GuessTheNumber，超参见上文，依赖 Tinker 云服务和 GEM 环境。书中说采样和梯度计算都在云端执行，只报了"笔记本上约 20 分钟"，没有给 GPU 型号和数量。预期结果按 Fig 6.11：reward 从 0.8 升到 1.0，action token 从约 900 降到约 500。
- **不依赖 Tinker 的替代方案**（CPU 即可跑）：
  1. 用 CleanRL 在 CartPole 上跑 Algorithm 9（PPO + GAE），扫 $\lambda\in\{0,0.95,1\}$，验证 eqn:GAE 中的 bias–variance 权衡。
  2. 在网格世界上跑 Algorithm 17（Tabular Dyna-Q），比较每个真实步配 $N=0$ 与 $N>0$ 次 imagined 更新时的样本效率。
  3. 复现 Baird counterexample，观察 deadly triad 导致的发散（Fig 2.6）。
- **预计难点**：书中的 SOTA 数字（Rainbow、BBF、DreamerV3）要回到原论文的代码去复现；Tinker 需要服务访问权限；token 级 loss 与长度归一化的细节要读 §6.4.1（书中明确说 Tinker 示例按 transition 平均、不除以 token 数）和 §6.4.4 的代码。

## 自测问题

1. 在 Fig 1.2 的通用模型里，Dreamer 的 sequence model、prior、encoder、decoder 分别对应 $U,P,E,D$ 中的哪一个？为什么 contextual MDP 是 POMDP 的一种特例？
2. MBRL 博弈中，model loss 为什么必须定义在 $\mu^\pi_M$ 上，而不是任意的数据分布上？PAL 和 MAL 各自在哪一侧做保守更新？为什么 MAL 得到的模型更"全局"？
3. 容量有限、按像素 MSE 训练的 world model，为什么可能对控制没有用？ZP+RP 为什么足以保证 value equivalence？只用 ZP 为什么会 collapse？EMA target 起什么作用？
4. 推导 $A_t=\delta_t+\gamma\lambda A_{t+1}$，并说明 $\lambda=0$ 和 $\lambda=1$ 时分别退化成什么。GRPO 和 ReBN 用什么替代了 critic？代价是什么？
5. 用 §4.2.4.1 的例子算出 open-loop 与 closed-loop 的期望回报，并解释 MPC 为什么在随机环境中次优，以及每步重新规划为什么能缓解这个问题。

## 综述强调、但本项目未收录的重要工作

1. **MuZero**：Schrittwieser et al. 2020，arXiv 1911.08265。§4.2.2.2：用 RP/VP/PP 训练 latent model，配合 MCTS。`index.json` 已把它列为概念选读后剔除。
2. **EfficientZero**：Ye et al.，NeurIPS 2021，bib 中没有 arXiv ID。§4.2.2.2：加入 self-prediction loss、value prefix 和 Reanalyse。
3. **PETS**：Chua et al. 2018，arXiv 1805.12114。§4.3.2：概率 ensemble 动力学模型的起点。
4. **MBPO**：Janner et al. 2019，arXiv 1906.08253。§4.3.2 / §4.4.4.1：Dyna + SAC + ensemble，用短 rollout 抑制 compounding error。
5. **PlaNet**：Hafner et al. 2019，arXiv 1811.04551。§4.4.1.3：Dreamer 的前身，规划用 MPC。
6. **Bridging State and History Representations: Understanding Self-Predictive RL**：Ni et al. 2024，arXiv 2401.08898。§4.4.2 用它的 OP/ZP/RP/VP 条件组织 Table 4.2。
7. **A Game Theoretic Framework for Model Based RL**：Rajeswaran et al.，ICML 2020，bib 中没有 arXiv ID。§4.3.1 的 PAL/MAL 出自这里。
8. **Objective Mismatch in Model-based RL**：Lambert et al.，L4DC 2020，bib 中没有 arXiv ID。§4.4.2.1。
9. **GAE**：Schulman et al. 2016，arXiv 1506.02438（§3.2.2）；以及 **TRPO**，arXiv 1502.05477（§3.3.2）。
10. **DeepSeekMath（GRPO 的出处）**：Shao et al. 2024，arXiv 2402.03300（§6.2.4）；以及 DAPO（arXiv 2503.14476）和 GSPO（arXiv 2507.18071）。
