# Reinforcement Learning

> 目录：`related_work/04_rl/` · 收录 10 篇论文和 1 篇综述 · 最后更新：2026-09-28
> 本文只转述各篇 `notes.md` 的内容，数字都能在对应论文的 `source/` 或 `paper.pdf` 里找到出处。带"≈"的数字是笔记从图上读出的近似值。未收录的论文只给 arXiv ID，这些 ID 已于 2026-09-28 在 arxiv.org 上核对过标题。

## 这条线在解决什么问题

强化学习（RL）研究的是：环境动力学未知、也没有逐步的正确动作标注时，智能体怎样只靠交互得到的标量 reward，学到一个使期望累积回报最大的 policy。MDP 与 POMDP 的形式化见 [Murphy 综述](./survey-rl-overview/notes.md) Ch1。

RL 比监督学习难在三处：
- 数据由当前策略自己产生，数据分布随策略一起变化；
- reward 稀疏且有延迟，需要做 credit assignment；
- value 学习依赖 bootstrapping，它和 function approximation、off-policy 数据同时出现时训练会不稳定，即 deadly triad。

所以这条线一直在三对矛盾之间取舍：
- 样本效率：on-policy 数据用完即丢，off-policy 和 offline 方法则复用数据；
- 训练稳定性：靠 target network、trust region、保守的 value 估计来维持；
- 策略表达力与似然是否可算：从 Gaussian 到 diffusion / flow，再到 token 化的 LLM / VLA。

2024 年以后，RL 的主要用法从"从零训练"变成"对预训练策略做后训练"。核心问题随之变成三个：reward 或 verifier 从哪来；rollout 在哪里跑（仿真器、真机还是 world model）；生成式策略的 log-likelihood 怎么算。

在本项目里，RL 负责把 world model 的预测变成决策与控制，也是 sim-to-real 训练和 test-time RL 共用的优化器。

---

## 1 发展脉络

两条支线的大致谱系如下（粗体为本项目收录）：

```
            value-based / off-policy                    policy gradient / on-policy
2013        **DQN** (experience replay)
2015        DDPG (continuous actions)                   TRPO (KL trust region), GAE
2017                                                    **PPO** (clipped surrogate)
2018        **SAC**, TD3 (max-ent, twin Q)                 |
2020-21     CQL, **IQL** (offline, in-sample)              |
2023-24     RLPD -> SERL -> **HIL-SERL** (real robot)      +-- **DPPO** (diffusion policy, two-layer MDP)
2024-25                                                    +-- GRPO (DeepSeekMath) -> **DeepSeek-R1** (RLVR)
2025        **pi*0.6 / RECAP**                             +-- **SimpleVLA-RL** (token VLA + GRPO, sim)
            (real-world VLA, advantage conditioning)       +-- **pi_RL** (flow VLA + PPO, sim)
2025-26     ---- world model as simulator: WMPO / World-Gymnast (see 05_intersections) ----
```

逐步看，每一步都是在补上一步留下的缺口：

| 时间 | 工作 | 上一步卡在哪 | 这一步怎么解决 | 留给下一步的问题 |
|---|---|---|---|---|
| 2013 | **[DQN](./dqn/notes.md)** | 视觉 RL 靠手工特征加线性 value；Q-learning 配非线性函数逼近可能发散 | CNN 从像素直接回归每个动作的 Q。experience replay 打破样本相关性并复用数据，这要求学习必须是 off-policy 的，所以选 Q-learning | 只能处理离散动作（要取 $\max_a$）；max 会带来 overestimation；2013 版没有 target network，稳定性只有经验证据 |
| 2015 | Nature 版 DQN（加 target network；无 arXiv）、Double DQN（1509.06461） | TD target 随参数漂移；max 造成高估 | 定期复制参数的 target network；用 online 网络选动作、target 网络评估 | 仍然只能处理离散动作 |
| 2015 | TRPO（1502.05477）、GAE（1506.02438） | vanilla PG 在同一批数据上做多步优化，会产生破坏性的大更新 | 用 KL trust region 约束每次更新；GAE 用 λ 调节 advantage 估计的 bias 与 variance | 要用共轭梯度加二阶近似，实现复杂，也不兼容参数共享和 dropout |
| 2015 | DDPG（1509.02971） | DQN 不能用于连续动作 | 由 deterministic actor 和 Q 组成的 off-policy actor-critic | actor 与 Q 强耦合，对超参数极其敏感，在 Humanoid 上学不动 |
| 2017 | **[PPO](./ppo/notes.md)** | TRPO 实现复杂 | 对概率比先 clip 再取 min，得到一阶的 surrogate，同一批 on-policy 数据可以跑 K 个 epoch | on-policy 数据用完即丢，样本效率仍然低；效果依赖大量没写进论文的 code-level trick |
| 2018 | **[SAC](./sac/notes.md)**（同期有 TD3，1802.09477） | on-policy 样本贵；DDPG 脆弱；soft Q-learning 的收敛依赖近似采样器 | 在 maximum entropy 目标下推导 soft policy iteration，再近似成 off-policy actor-critic：重参数化的 tanh-Gaussian actor、twin Q、target V | 温度（等价于 reward scale）要逐任务调；策略是单峰的；Humanoid 仍要约 10M 步，真机上无法从零训练 |
| 2020–21 | CQL（2006.04779）→ **[IQL](./iql/notes.md)** | 只有固定数据集时， $\max_{a'}$ 会查询数据里没有的动作，函数近似在这些动作上外推出错并被策略利用 | CQL 压低 OOD 动作的 Q。IQL 用 expectile regression 只在数据内的动作上逼近 max，保留 multi-step DP，再用 AWR 提取策略，全程不查询 OOD 动作 | AWR 本质上是加权 BC，很难超出数据中最好的动作；τ/β 要按领域调；offline-to-online 仍需 1M 个在线步 |
| 2023–24 | RLPD（2302.02948）→ SERL（2401.16013）→ **[HIL-SERL](./hil-serl/notes.md)** | 真机上 sample complexity 高、reward 难以指定、优化不稳定 | RLPD（SAC 加 prior data，prior data 与在线数据对半采样）、success classifier 充当 reward、人类在线干预。干预片段同时写入 demo buffer 和 RL buffer。13 项任务在 1–2.5 h 内达到 100%（timing belt 例外，用了 6 h） | 每个任务都从头训练一个小网络；人力成本没有量化；能否扩展到 VLA 规模未知 |
| 2024 | **[DPPO](./dppo/notes.md)** | diffusion policy 没有显式似然，PG 无法直接用；Q-learning 类的 diffusion RL 在稀疏 reward 下，微调一开始就掉点甚至崩溃 | 把每一步去噪当成内层 MDP 的一步，嵌进环境 MDP。单步去噪是高斯分布，似然可算，于是可以直接用 PPO 微调；探索是贴着示范流形的结构化探索（on-manifold） | 需要大量 on-policy 样本（Robomimic 3e6–4e7 步，Furniture 7e8–1e9 步），只能在仿真里做；同一环境步的各去噪步共用一个 advantage，信用分配粗 |
| 2024–25 | GRPO（DeepSeekMath，2402.03300）→ **[DeepSeek-R1](./deepseek-r1/notes.md)** | PPO 需要一个和策略差不多大的 value model，只有 outcome reward 时很难训；neural RM 在大规模 RL 里会被 reward hacking | GRPO 对每题采 G 个回答，用组内 (r−mean)/std 作为 advantage，从而去掉 critic；推理任务只用规则可验证的奖励（RLVR）。R1-Zero 不经 SFT 就学出了长 CoT，AIME 2024 pass@1 从 15.6 升到 77.9 | 要能从同一个 prompt 重复采样（相当于能 reset），verifier 的成本也要近乎为零；组内全对或全错时没有梯度；ratio 是 token 级还是序列级、std 归一化带来的偏差，论文都没交代 |
| 2025 | **[SimpleVLA-RL](./simplevla-rl/notes.md)** | 把 LLM 的 RL 配方搬到 VLA 上时，rollout 要和环境多轮交互；多数 VLA 的动作解码给不出可采样的分布 | 选用 token 化动作的 OpenVLA-OFT，在 veRL 里做并行仿真闭环 rollout；GRPO 加 0/1 成功奖励，再加 dynamic sampling、clip-higher、高温度采样。每个任务只用 1 条示范起步，LIBERO-Long 从 17.3 升到 91.7 | 依赖可重置的仿真器和 `is_successful` 判定；初始成功率为 0 时 RL 学不动；只覆盖 token 化动作 |
| 2025 | **[π_RL](./pi-rl/notes.md)** | flow matching VLA（π0/π0.5）的动作由确定性 ODE 积分得到，既没有探索噪声，也算不出 log-likelihood | Flow-Noise（可学的噪声网络，one-layer MDP）和 Flow-SDE（换成边缘分布相同的 SDE，沿用 DPPO 的 two-layer MDP）都把每个去噪步写成高斯，然后用 PPO 在仿真中微调 action expert | 训练的是随机策略，部署的是确定性 ODE，两者之间没有改进保证；仍然依赖大规模仿真交互 |
| 2025 | **[π\*0.6（RECAP）](./pi-star-06/notes.md)** | 在真机上跑 on-policy PPO 太贵；flow 策略没有似然；AWR 会丢弃或大幅降权大部分数据 | 用 MC 回报训练 distributional value，把 n-step advantage 按比例阈值二值化成文本条件 "Advantage: positive/negative"，再用监督学习训练 flow VLA。推理时以 positive 为条件采样，就得到正则化的改进策略。示范、人类纠正和自主 rollout 统一放进 iterated offline RL | 奖励标签、场景复位和接管都靠人；理论只保证一步改进；关键设计缺少消融 |
| 2025–26 | → [WMPO](../05_intersections/wmpo/notes.md)、[World-Gymnast](../05_intersections/world-gymnast/notes.md)（05 交叉线） | SimpleVLA-RL 和 π_RL 依赖物理仿真器；π\*0.6 依赖真机和人 | 在学出来的视频 world model 里做闭环 rollout，对 VLA 做 GRPO | 成功信号需要另找来源（学到的 reward model 或 VLM 打分）；策略可能利用模型误差 |

**主线怎么读**：2024 年之后，两条支线汇合到同一个问题上：怎样对预训练的生成式策略做 RL 后训练。
- value / off-policy 支线（DQN → SAC → IQL → HIL-SERL → π\*0.6）的目标是提高样本效率，因此能在真机上用，代价是需要人提供示范、干预和标签。
- policy gradient / on-policy 支线（TRPO → PPO → DPPO / GRPO → SimpleVLA-RL / π_RL）的优势是稳定、实现简单，前提是 rollout 足够便宜，比如有可重置的仿真器，或者以后用 world model 代替。

---

## 2 技术分类（taxonomy）

一个 RL 方法可以沿四根轴来定位：
- **学什么**（Murphy Table 1.1）：value（DQN、IQL 的 critic）、policy（REINFORCE、GRPO），或两者都学（actor-critic：PPO、SAC、DPPO、π_RL）。学 model 的是 MBRL，属于 [01 world model](../01_world_model/) 和 [05 交叉线](../05_intersections/)。
- **数据从哪来**：on-policy online；off-policy online（用 replay buffer）；offline（固定数据集）；offline-to-online 或 iterated offline（prior data 加上新采集的数据）。
- **策略怎样参数化**：离散动作上取 argmax Q；Gaussian 或 tanh-Gaussian；diffusion；flow matching；自回归 token（LLM、token 化的 VLA）。参数化方式决定了策略更新能用 likelihood-ratio 梯度、reparameterization 梯度，还是只能做条件监督学习。
- **reward 从哪来**：仿真器的 dense reward、仿真器的成功判定、学出来的 success classifier、规则 verifier、neural RM / preference RM、人工标签。

按子方向整理：

| 子方向 | 核心问题 | 本项目收录 | 未收录的代表作（arXiv） |
|---|---|---|---|
| Value-based deep RL | 深度网络加 off-policy TD，怎样训得稳 | [DQN](./dqn/notes.md) | Double DQN 1509.06461；Prioritized Experience Replay 1511.05952；Dueling 1511.06581；C51（distributional RL）1707.06887；Rainbow 1710.02298；BBF 2305.19452 |
| Policy gradient / trust region（on-policy） | 步长控制、advantage 估计、数据复用 | [PPO](./ppo/notes.md) | TRPO 1502.05477；GAE 1506.02438；A3C 1602.01783；Implementation Matters 2005.12729；What Matters in On-Policy RL 2006.05990 |
| Off-policy actor-critic（连续控制） | 连续动作下的样本效率与稳定性 | [SAC](./sac/notes.md) | DDPG 1509.02971；TD3 1802.09477；SAC 自动温度版 1812.05905 |
| Offline RL | 固定数据集下 OOD 动作的外推误差 | [IQL](./iql/notes.md) | BCQ 1812.02900；AWR 1910.00177；CQL 2006.04779；Decision Transformer 2106.01345；Levine 等的教程 2005.01643 |
| Offline-to-online 与真实世界机器人 RL | prior data 加少量真机交互；reward 怎样指定；安全 | [HIL-SERL](./hil-serl/notes.md) | AWAC 2006.09359；RLPD 2302.02948；Cal-QL 2303.05479；IBRL 2311.02198；SERL 2401.16013；RL-100 2510.14830 |
| 生成式策略（diffusion / flow）的 RL | 策略没有可解析似然时，怎样做 policy improvement | [DPPO](./dppo/notes.md)、[π_RL](./pi-rl/notes.md) | DDPO 2305.13301；IDQL 2304.10573；ReinFlow 2505.22094；Flow-GRPO 2505.05470；CFGRL 2505.23458；DSRL 2506.15799；FPO 2507.21053 |
| RLHF 与 LLM 推理 RL（RLVR） | critic-free 的 policy gradient、可验证奖励、reward hacking | [DeepSeek-R1](./deepseek-r1/notes.md) | Christiano et al. 1706.03741；InstructGPT 2203.02155；DPO 2305.18290；DeepSeekMath（GRPO 出处）2402.03300；RLOO 2402.14740；DAPO 2503.14476；Dr. GRPO 2503.20783；GSPO 2507.18071；ScaleRL 2510.13786 |
| VLA 的 RL 后训练 | 大模型、稀疏成功奖励、昂贵的 rollout 三者同时出现 | 仿真 GRPO：[SimpleVLA-RL](./simplevla-rl/notes.md)；仿真 PPO：[π_RL](./pi-rl/notes.md)；真机 advantage conditioning：[π\*0.6](./pi-star-06/notes.md) | RL4VLA 2505.19789；GR-RL 2512.01801；RL Token 2604.23073；EXPO-FT 2605.25477 |
| Model-based RL / world model 内的 RL（跨线） | 用学到的模型代替环境生成 rollout | 在其他目录：[Dreamer](../01_world_model/dreamer/notes.md)、[DreamerV3](../01_world_model/dreamerv3/notes.md)、[TD-MPC2](../01_world_model/tdmpc2/notes.md)、[Dreamer 4](../01_world_model/dreamer4/notes.md)、[DayDreamer](../05_intersections/daydreamer/notes.md)、[RWM](../05_intersections/robotic-world-model/notes.md)、[WMPO](../05_intersections/wmpo/notes.md)、[World-Gymnast](../05_intersections/world-gymnast/notes.md) | PETS 1805.12114；MBPO 1906.08253；MuZero 1911.08265 |

以上子方向在 [Murphy 综述](./survey-rl-overview/notes.md) 里都有覆盖。不过该综述几乎不涉及 sim-to-real 和机器人 VLA RL，也没有收 IQL。

---

## 3 论文总表

| 阅读顺序 | 论文 | 年份 | 层级 | 一句话 | 代码 | 笔记 |
|---|---|---|---|---|---|---|
| 0（贯穿全程） | Reinforcement Learning: An Overview（2412.05265） | 2024 | survey | 约 250 页的书式综述：用 agent–environment 通用图模型统一 POMDP、MDP 和 contextual MDP，按"学 value / policy / model"分章，world model 和 LLM×RL 也放进同一套记号 | 无配套代码库（§6.4 的 Tinker 示例在 [pyprobml](https://github.com/probml/pyprobml/blob/master/scripts/tinker_train_gem.py)） | [notes](./survey-rl-overview/notes.md) |
| 1 | Playing Atari with Deep Reinforcement Learning（1312.5602） | 2013 | foundational | CNN 从像素回归每个动作的 Q 值，靠 experience replay 让深度网络加 off-policy TD 稳定训练；7 个 Atari 游戏中 6 个超过已有方法，3 个超过人类专家 | 本文无官方代码；Nature 版的 [google-deepmind/dqn](https://github.com/google-deepmind/dqn) 已归档且带 target network；可用 CleanRL `dqn_atari.py` 改出 2013 版 | [notes](./dqn/notes.md) |
| 2 | Proximal Policy Optimization Algorithms（1707.06347） | 2017 | foundational | 用对概率比 clip 后取 min 的一阶 surrogate 代替 TRPO 的 KL 硬约束，同一批 on-policy 数据可以安全地跑多个 epoch | [openai/baselines](https://github.com/openai/baselines)（ppo2） | [notes](./ppo/notes.md) |
| 3 | Soft Actor-Critic（1801.01290） | 2018 | foundational | 在最大熵框架下推出可证明收敛的 soft policy iteration，再近似成 off-policy actor-critic；在高维 MuJoCo 上，样本效率和跨 seed 稳定性都好于 DDPG 和 PPO | [haarnoja/sac](https://github.com/haarnoja/sac)（不再维护，README 指向 softlearning） | [notes](./sac/notes.md) |
| 4 | Offline RL with Implicit Q-Learning（2110.06169） | 2021 | milestone | 用 expectile regression 只在数据内的动作上逼近 max，保留 multi-step DP，最后用 AWR 提取策略，全程不查询 OOD 动作 | [ikostrikov/implicit_q_learning](https://github.com/ikostrikov/implicit_q_learning) | [notes](./iql/notes.md) |
| 5 | HIL-SERL（2410.21845） | 2024 | milestone | 在真机上做视觉 RL：RLPD、success classifier，再加人类在线干预（干预数据同时写入两个 buffer）。13 项任务在 1–2.5 h 内达到 100% 成功率（timing belt 例外，用了 6 h） | [rail-berkeley/hil-serl](https://github.com/rail-berkeley/hil-serl) | [notes](./hil-serl/notes.md) |
| 6 | Diffusion Policy Policy Optimization（2409.00588） | 2024 | milestone | 把去噪链建成内层 MDP，用 PPO 微调 BC 预训练的 diffusion policy；One-leg 任务零样本部署到真机，成功 16/20 | [irom-princeton/dppo](https://github.com/irom-princeton/dppo) | [notes](./dppo/notes.md) |
| 7 | DeepSeek-R1（2501.12948） | 2025 | milestone | GRPO 加规则可验证奖励：R1-Zero 不经 SFT 就学出长 CoT（AIME 2024 pass@1 从 15.6 升到 77.9）；R1 再用多阶段流程补上可读性和通用能力 | 只有权重，[deepseek-ai/DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1) 没有训练代码和数据 | [notes](./deepseek-r1/notes.md) |
| 8 | SimpleVLA-RL（2509.09674） | 2025 | frontier | 把 R1 式的 GRPO 加 0/1 成功奖励用到 token 化的 VLA（OpenVLA-OFT）上，在 veRL 中做并行仿真闭环 rollout；每个任务用 1 条示范起步，LIBERO-Long 从 17.3 升到 91.7 | [PRIME-RL/SimpleVLA-RL](https://github.com/PRIME-RL/SimpleVLA-RL) | [notes](./simplevla-rl/notes.md) |
| 9 | π_RL（2510.25889） | 2025 | frontier | Flow-Noise 和 Flow-SDE 把 flow VLA 的去噪步改写成高斯转移，使 PPO 能在仿真中在线微调 π0 和 π0.5 | [RLinf/RLinf](https://github.com/RLinf/RLinf) | [notes](./pi-rl/notes.md) |
| 10 | π\*0.6（2511.14759） | 2025 | frontier | RECAP：用 MC distributional value 算 advantage，二值化后作为文本条件，靠监督学习做真实世界的 iterated offline RL；在最难的任务上 throughput 提高到两倍以上 | 无（[openpi](https://github.com/Physical-Intelligence/openpi) 只提供 π0 和 π0.5） | [notes](./pi-star-06/notes.md) |

这里的阅读顺序和 `index.json` 的 `reading_order` 有一处不同：HIL-SERL 被提到了 DPPO 前面。原因是 HIL-SERL 直接建立在 SAC 和 offline-to-online 之上，和 IQL 放在同一阶段读更顺。

---

## 4 推荐学习路线

### 阶段 0：共同语言

**前置知识 checklist**
- [ ] MDP / POMDP、return、折扣 γ、 $V$ / $Q$ / advantage $A=Q-V$
- [ ] Bellman 期望方程与最优方程；value iteration、policy iteration
- [ ] MC 与 TD、bootstrapping；on-policy 与 off-policy；ε-greedy
- [ ] policy gradient theorem、score-function（likelihood-ratio）估计、baseline
- [ ] importance sampling

**要读**：[Murphy 综述笔记](./survey-rl-overview/notes.md)。原书读 Ch1（§1.1–1.3）、§2.5（Q-learning、DQN 系列、deadly triad）、§3.1–3.2（PG、actor-critic、GAE）。

**读完应能回答**
1. Table 1.1 里，SARSA 为什么是 on-policy，Q-learning 为什么是 off-policy？
2. deadly triad 由哪三个要素组成？Baird counterexample 里改成 on-policy 的线性 TD(0) 为什么就能保证收敛？
3. 在 PG 里减去 baseline，为什么不改变梯度的期望？GAE 取 λ=0 和 λ=1 时分别退化成什么？

### 阶段 1：深度 RL 的三块基石（DQN → PPO → SAC）

**前置知识 checklist**
- [ ] CNN；SGD / RMSProp / Adam
- [ ] 熵、KL 散度，以及 reverse KL（I-projection）为什么是 mode-seeking 的
- [ ] 重参数化技巧；tanh 变换下 log-density 的换元公式
- [ ] TRPO 的动机：surrogate 目标加 KL 约束（TRPO 原文 §3–6，或 Murphy §3.3）

**要读**：[DQN](./dqn/notes.md) → [PPO](./ppo/notes.md) → [SAC](./sac/notes.md)。原书配读 §3.3.3（PPO）和 §3.6.8（SAC）。

**读完应能回答**
1. 用了 experience replay，为什么就必须 off-policy？DQN 2013 版算 target 用的是哪组参数？和 Nature 版有什么不同？
2. 设 ε=0.2、 $\hat A<0$， $r_t=1.5$ 和 $r_t=0.7$ 的样本梯度分别是否为 0？clip 能不能保证 KL ≤ δ？
3. SAC 的 reward scale 为什么等价于 $1/\alpha$？scale 过大和过小时各自怎样失败？
4. PPO 用 likelihood-ratio 梯度，SAC 用 pathwise（重参数化）梯度。两者对"策略似然可算"和"critic 可微"各有什么要求？（后面 DPPO、π_RL、π\*0.6 走了三条不同的路，根源就在这里。）
5. 为什么样本便宜时（大规模并行仿真）常用 PPO，样本贵时（真机）常用 SAC 系方法？可对照 [legged-gym](../03_real2sim2real/legged-gym/notes.md) 和 [HIL-SERL](./hil-serl/notes.md)（HIL-SERL 在阶段 2 精读，这里只看它的训练设置）。

### 阶段 2：数据驱动与真实世界（IQL → HIL-SERL）

**前置知识 checklist**
- [ ] offline RL 中的 extrapolation error，以及 behavior policy 的 support
- [ ] quantile / expectile regression
- [ ] AWR / AWAC 的闭式解 $\pi\propto\pi_\beta e^{\beta A}$
- [ ] RLPD 的对称采样；DAgger / HG-DAgger
- [ ] （机器人部分）impedance control、齐次变换与 adjoint 映射

**要读**：[IQL](./iql/notes.md) → [HIL-SERL](./hil-serl/notes.md)。原书配读 §7.6（imitation learning）和 §7.7（offline RL）。

**读完应能回答**
1. 写出 IQL 的 Eq.5–7，说明为什么三者都不会查询 OOD 动作。τ=0.5 和 τ→1 分别对应什么算法？
2. 为什么要单独拆出一个 $V$、只对动作取 expectile？（即"lucky sample"论证）
3. HIL-SERL 为什么让干预片段同时写入两个 buffer，而策略自己的转移只写入 RL buffer？"反复长时间接管直到成功"为什么会导致 Q 被高估？
4. HIL-SERL Table 1b 中，去掉干预后 RAM 任务是 48，IBRL 是 75。这对"HIL-SERL 的 RL 部分优于其他 demo-RL 方法"这一说法意味着什么？

### 阶段 3：生成式策略与大模型的 RL 后训练（DPPO → DeepSeek-R1 → SimpleVLA-RL → π_RL → π\*0.6）

**前置知识 checklist**
- [ ] DDPM / DDIM（参数 η）；Diffusion Policy 与 action chunking
- [ ] flow matching：CFM loss、Euler 采样；score SDE 与 probability flow ODE
- [ ] LLM 后训练的基本词汇：SFT、CoT、pass@k、reward model（读 DeepSeek-R1 用）
- [ ] RLHF 中的 KL-to-reference 正则与 k3 估计量；GRPO；DAPO 的 clip-higher 和 dynamic sampling
- [ ] VLA 架构：OpenVLA-OFT 的 action token；π0/π0.5 的 VLM 加 action expert
- [ ] classifier-free guidance；distributional value function（离散 bin 加交叉熵）

**要读**：[DPPO](./dppo/notes.md) → [DeepSeek-R1](./deepseek-r1/notes.md) → [SimpleVLA-RL](./simplevla-rl/notes.md) → [π_RL](./pi-rl/notes.md) → [π\*0.6](./pi-star-06/notes.md)。原书配读 §3.3.3.2（diffusion policy 上的 PPO）和 §6.2.4（single-turn RL 算法：PPO、GRPO、DAPO、GSPO、RLOO）。

**读完应能回答**
1. DPPO 为什么把 $a^K\sim\mathcal N(0,I)$ 算作环境的一部分？ $\sigma^{\text{exp}}_ {\min}$ 和 $\sigma^{\text{prob}}_ {\min}$ 分别作用在哪一步？
2. 写出 GRPO 的 $A_i$。一组 16 个回答全对时梯度是多少？R1 为什么把 KL 放进 loss，而不是像 PPO 那样当作 per-token reward？
3. SimpleVLA-RL 为什么必须把 OFT 的"MLP + L1"动作头换成"token 头 + CE"？G=8 时，组内 1 条成功和 7 条成功两种情况下，成功轨迹的 advantage 各是多少？
4. Flow-Noise 和 Flow-SDE 的 PPO ratio 是在同一个概率对象上算的吗？为什么 K=1 时 SDE rollout 的成功率只有 9.4%？
5. π\*0.6 中，为什么 β=1 时"以 $I=\text{True}$ 为条件的行为策略"恰好就是正则化改进策略？它的 value function 估计的是哪个策略的 value？
6. 把 SimpleVLA-RL、π_RL、π\*0.6、HIL-SERL 放在一起比较：rollout 各在哪里跑？reward 从哪来？分别需要多少人工？

### 阶段 4：通向交叉线（world model × RL × TTT × sim-to-real）

**前置知识 checklist**
- [ ] Murphy Ch4：decision-time planning 与 background planning 的区别，§4.3.1 的 MBRL 博弈（PAL / MAL）
- [ ] Dreamer 在 imagination 里训练 actor-critic 的方式（[Dreamer](../01_world_model/dreamer/notes.md)）
- [ ] test-time training 的基本设定：部署时用自监督辅助任务更新参数（[TTT](../02_test_time_training/ttt-self-supervised/notes.md)，读 PAD 用）
- [ ] （读 SimDist 用）MPPI 规划与 TD-MPC 式 latent world model（[TD-MPC2](../01_world_model/tdmpc2/notes.md)）

**要读**：
- [Robotic World Model](../05_intersections/robotic-world-model/notes.md)：在 world model 里跑 PPO；
- [WMPO](../05_intersections/wmpo/notes.md)、[World-Gymnast](../05_intersections/world-gymnast/notes.md)：在 world model 里对 VLA 做 GRPO；
- [SimDist](../05_intersections/simdist/notes.md)：用仿真预训练 world model，再到真机快速适应，baseline 是 IQL / RLPD；
- [PAD](../05_intersections/pad/notes.md)：SAC 加部署时的自监督适应；
- [TTRL](../02_test_time_training/ttrl/notes.md)：测试时做 GRPO；
- sim-to-real 一侧对照 [legged-gym](../03_real2sim2real/legged-gym/notes.md)。

**读完应能回答**
1. GRPO 需要从同一个初始状态做 G 次 rollout。真实世界做不到，world model 怎样提供这种能力？代价是什么？
2. 在 world model 里训练策略时，PPO 的 likelihood-ratio 梯度和 Dreamer 穿过可微 dynamics 的 pathwise 梯度，对模型分别有什么要求？
3. 按 Murphy §4.3.1，model loss 应该定义在哪个分布上？这对"在 world model 里训出的策略会不会利用模型误差"意味着什么？
4. TTRL（测试时 RL）、PAD（部署时自监督适应）、π\*0.6（真机 iterated offline RL）三者比较：部署时有没有 reward？多久更新一次？更新哪些参数？

---

## 5 关键机制对比

### 5.1 算法机制

| 方法 | 策略参数化 | 数据 | 价值 / advantage 估计 | 策略更新 | 稳定性来源 |
|---|---|---|---|---|---|
| DQN | 对 Q 取 argmax（离散动作），ε-greedy 探索 | off-policy online，replay buffer | Q-learning 的 TD target（取 max） | 没有独立策略 | experience replay、reward clipping（2013 版没有 target network） |
| PPO | Gaussian / softmax | on-policy online | 学 $V$，用 GAE(λ) 算 advantage | likelihood-ratio PG，对 clipped ratio 做 K 个 epoch | clip ε=0.2，相当于隐式 trust region |
| SAC | tanh-Gaussian | off-policy online，replay buffer | soft Q / V；twin Q 取 min；V 有 EMA target | 重参数化 pathwise 梯度，最小化 $\mathbb E[\log\pi-Q]$ | 熵正则（等价于 reward scale）、twin Q、target V |
| IQL | Gaussian（std 与状态无关） | offline，可接 online fine-tune | $V$ 用 expectile 拟合， $Q$ 用 MSE 回归，都只用数据内的 (s,a) | AWR：以 $\exp(\beta(Q-V))$ 加权的 BC | 从不查询 OOD 动作；指数化的 advantage 截断到 100 |
| HIL-SERL | 连续动作用 Gaussian，夹爪另用 DQN 式 critic | 真机 off-policy online，加 demo buffer（RLPD 对半采样） | SAC 式的 Q 加 target network | SAC actor loss（自适应 α） | prior data 与在线数据对半采样；人类干预；impedance 控制器的 reference limiting 保证安全 |
| DPPO | diffusion policy，每个去噪步是高斯 | 仿真 on-policy online | $V(s_t)$ 加 GAE（只在环境步上算），再乘 $\gamma_{\text{denoise}}^k$ 分给各去噪步 | 对每个去噪步做 PPO | 按 clip fraction 调 ε；两个 σ 下限截断；只微调最后 K' 个去噪步 |
| DeepSeek-R1（GRPO） | 自回归 LLM | on-policy，每个 prompt 采 G=16 条 | 不用 critic，用组内 (r−mean)/std | clipped ratio，KL（k3 估计）放进 loss | β=0.001；每 400 步重置 reference model；R1 的推理 RL 阶段 ε=10 |
| SimpleVLA-RL | token 化动作（改版 OpenVLA-OFT） | 仿真 on-policy，同一初始状态采 G=8 条 | 不用 critic，对 0/1 奖励做组内归一化 | GRPO，去掉 KL 项 | clip 区间 [0.8, 1.28]、dynamic sampling、rollout 温度 T=1.6 |
| π_RL | flow matching，经 Flow-Noise / Flow-SDE 变成每步高斯 | 仿真 on-policy | critic 挂在 VLM 或 action expert 上，GAE 按 chunk 级别算 | PPO，只训练 action expert | PPO clip、噪声水平 a、冻结 VLM |
| π\*0.6 | flow-matching VLA 加 FAST token，以 "Advantage: positive/negative" 为条件 | 真机 iterated offline（自主 rollout、人类纠正、示范） | MC 训练的 distributional V（201 个 bin），N=50 的 n-step advantage 按比例阈值二值化 | 条件监督学习（CFGRL），推理时以 positive 为条件采样 | 通过 β 隐式地正则到 $\pi_{\text{ref}}$；每轮都从预训练 checkpoint 重训 |

### 5.2 数据、奖励与算力

| 方法 | reward 来源 | rollout 在哪 | 需要真实数据吗 | 交互 / 样本量 | 论文报告的算力 |
|---|---|---|---|---|---|
| DQN | 游戏分数，clip 到 {−1, 0, +1} | Atari 模拟器 | 不需要 | 10M 帧 | 每 epoch（5 万次更新）约 30 分钟，GPU 型号未报 |
| PPO | 仿真 dense reward | MuJoCo / Roboschool / Atari | 不需要 | MuJoCo 1M；Roboschool humanoid 50–100M；Atari 10M timesteps | 未报告 |
| SAC | 仿真 dense reward | MuJoCo | 不需要 | 1M–10M 步 | 未报告（2×256 MLP，CPU 可跑） |
| IQL | 数据集自带 | 无（offline）；fine-tune 时 1M 在线步 | 需要离线数据集（D4RL） | 1M 次梯度更新 | 单张 GTX1080 上 1M 次更新不到 20 分钟 |
| HIL-SERL | 学到的 success classifier（Jenga 由人标注） | 真机 | 需要：20–30 条 demo、约 1200 帧 classifier 数据，训练中还要人在线干预 | 每任务 1 万到 10.8 万条转移，1–2.5 h（timing belt 用了 6 h） | 单张 RTX 4090 |
| DPPO | 仿真 dense 或 sparse reward | 仿真（Furniture 用 1000 个 IsaacGym 环境） | 预训练要示范；真机只做零样本部署 | Robomimic 3e6–4e7 步；Furniture 7e8–1e9 步 | 每轮迭代 16.6–871 s；One-leg 在一张 L40 上约 41 h（笔记按图估算） |
| DeepSeek-R1 | 规则 verifier（对错加格式）；最后阶段加 RM | 文本生成，没有外部环境 | 不需要（要有题目和答案） | R1-Zero 共 10,400 步，每步 32 题，每题 16 个回答 | 147K H800 GPU·h |
| SimpleVLA-RL | 仿真 `is_successful` 给 0/1 | LIBERO / RoboTwin 并行仿真 | SFT 要示范（最少每任务 1 条）；sim2real 只做零样本部署 | LIBERO-Long 约 250–300 个 RL step | 8×A800，时长未报告 |
| π_RL | 仿真稀疏成功奖励 | LIBERO / ManiSkill / MetaWorld / CALVIN | few-shot SFT 需要示范 | LIBERO 每个 suite：500 epochs × 64 个环境 × 8 个 rollout epoch | 8×H100，总时长未报告 |
| π\*0.6 | 人工给 episode 成功标签，换算成 {−1, 0, −C_fail} | 真机（家庭、工厂） | 需要：数万小时预训练示范，外加每任务每轮数百条真机 episode 和专家接管 | 例如 T-shirt 任务每轮 300 条，box 任务每轮 600 条加 360 条纠正 | 未报告 |

### 读表要点

- **策略怎样参数化，决定了策略怎样更新。**
  - 似然可以解析计算：用 likelihood-ratio PG（PPO、GRPO）。
  - critic 可微、动作连续：用 reparameterization 梯度（SAC）。
  - 似然难算的 flow / diffusion 策略有两种做法：改写 MDP，使每一步的似然可算（DPPO、π_RL）；或者把 policy improvement 改成条件监督学习（π\*0.6）。
- **critic 的角色一直在变。**
  - DQN、SAC、IQL 用 TD 学 Q。
  - PPO、DPPO、π_RL 学 V，再用 GAE 算 advantage。
  - GRPO 系列干脆不要 critic，用组内统计量代替，前提是能从同一状态重复采样。
  - π\*0.6 用 MC 回归训练 distributional V。
- **需要真实数据的方法都依赖人。** HIL-SERL 和 π\*0.6 需要人提供示范、干预或纠正、reward 标注。
- **不需要真实数据的方法都依赖仿真器。** DPPO、SimpleVLA-RL、π_RL 依赖可重置的仿真器，并靠 domain randomization 做零样本 sim-to-real。本项目 [05 交叉线](../05_intersections/) 的出发点，就是用 world model 去替代这里的仿真器，或者替代真机上的人力。

---

## 6 开放问题与争议

1. **VLA 的 RL 该走 on-policy PG，还是 off-policy / offline 路线？**
   - [SimpleVLA-RL](./simplevla-rl/notes.md) 和 [π_RL](./pi-rl/notes.md) 在仿真里做 PPO / GRPO；[π\*0.6](./pi-star-06/notes.md) 在真机上做 advantage conditioning；[HIL-SERL](./hil-serl/notes.md) 在真机上用 SAC 系的 RLPD。
   - 证据互相矛盾，每条都有保留：
     - π\*0.6 的 Fig.11 中，同样的数据下 RECAP 约 ≈60/h，AWR 约 ≈30/h，PPO 约 ≈22/h，而且 PPO 要把约束压到 η=0.01 才稳定。但 AWR 和 PPO 的起点 checkpoint 与 RECAP 不同。
     - [DPPO](./dppo/notes.md) 的 Fig.6 中，RLPD、Cal-QL、IBRL 在 HalfCheetah 上样本效率更高，在 Robomimic Can/Square 上却学不起来。不过这些基线在 Robomimic 上被限制在 1e6 个样本。
     - [SimDist](../05_intersections/simdist/notes.md) 指出，RLPD / SERL 这类高频更新 critic 的做法容易造成 value overestimation。
   - 目前没有在同一平台、同样预算下的对照。

2. **critic-free 的 GRPO 能不能离开"可 reset"这个前提？**
   - GRPO 要从同一个初始状态重复 rollout G 次，而且 verifier 成本近乎为零（见 [R1 笔记](./deepseek-r1/notes.md)"局限"第 9 条）。Murphy §6.2.4 的脚注也指出，GRPO 需要环境能 reset，无法对中间状态做 credit assignment。
   - [π_RL](./pi-rl/notes.md) Table 9 中，PPO 优于 GRPO：π0.5 在 LIBERO-Long 上是 93.0 对 77.6。但论文里 GRPO 的超参数表是空的，对比是否公平无法核查。
   - [WMPO](../05_intersections/wmpo/notes.md) 和 [World-Gymnast](../05_intersections/world-gymnast/notes.md) 用 world model 来提供 reset，但模型误差会被策略利用到什么程度，还没有系统的测量。

3. **RL 后训练是在"激发"已有能力，还是真的扩展了能力？**
   - 支持"激发"的证据：
     - [R1](./deepseek-r1/notes.md) 附录 A.1 承认，V3-Base 的预训练语料里含 OpenAI 模型生成的内容。它的 G.1 节说 7B dense 模型做 RL 没有提升，而 Table 17 里 Qwen2-Math-7B 做 RL 却有效。
     - [SimpleVLA-RL](./simplevla-rl/notes.md) Table 7：用 0 条示范做 SFT 时，RL 前后成功率都是 0。
     - [π_RL](./pi-rl/notes.md)：在 MetaWorld ML45 的新任务上没有提升，作者的结论是 RL 只改善动作层面，不会泛化到新的任务目标。
     - [DPPO](./dppo/notes.md)：效果依赖预训练覆盖到的 mode。
   - 支持"扩展"的证据：
     - SimpleVLA-RL 在 LIBERO 的"9 个任务训练、1 个任务留出"设定下，unseen 任务成功率上升；RL 还学出了示范里没有的 pushcut 行为（只有定性展示）。
   - LLM 一侧的同类争议见 2504.13837。

4. **生成式策略的似然近似能不能接受？**
   - DPPO 让同一环境步的所有去噪步共用一个 advantage。
   - π_RL 的 Flow-Noise 优化的是整条去噪链的联合概率，不是动作的边缘概率；Flow-SDE 在 K 很小时离散误差很大（K=1 时 train 成功率只有 9.4%）。
   - DPPO（训练用 η=1，评估用 η=0）和 π_RL（训练用 SDE，部署用 ODE）都存在"训练的是随机策略，部署的是确定性策略"的问题，目前没有改进保证。
   - π\*0.6 用条件监督学习完全绕开了似然，代价是理论只保证一步改进。
   - [SAC](./sac/notes.md) 那种重参数化加 $\log\pi$ 的做法，不能直接用在 flow 策略上。

5. **trust region 到底有多重要？**
   - [PPO](./ppo/notes.md) 的 clip 并不硬性限制 KL。
   - [R1](./deepseek-r1/notes.md) 推理 RL 阶段用 ε=10，下界永远不起作用。
   - [SimpleVLA-RL](./simplevla-rl/notes.md) 去掉了 KL 并放宽了 clip。
   - [DPPO](./dppo/notes.md) 只用 clip，不加 KL 或 BC 正则。
   - π\*0.6 的 PPO baseline 却要把约束收得很紧才稳定。
   - 各篇的设定差别很大，目前没有对 trust region 的系统比较。

6. **reward 从哪来，以及 reward hacking。**
   - [R1](./deepseek-r1/notes.md) Figure 6：用 helpful RM 训练时，RM 分数上升，CodeForces pass@1 却下降，所以 RM 只在最后 400 步使用。
   - [HIL-SERL](./hil-serl/notes.md) 的 success classifier 准确率为 95–99%，剩下的误判可能被策略利用。
   - [SimpleVLA-RL](./simplevla-rl/notes.md) 的 pushcut 从另一面看，就是只看终态的 reward 可能引发 specification gaming。
   - π\*0.6 依赖人工标签。
   - world model 路线里，[World-Gymnast](../05_intersections/world-gymnast/notes.md) 用 GPT-4o 打 0/1 分，[WMPO](../05_intersections/wmpo/notes.md) 用学到的成功判别器。
   - 真实世界没有 `is_successful`。在机器人上，reward 本身就是瓶颈。

7. **评测口径与可复现性。**
   - [DQN](./dqn/notes.md) 没有报 seed 数和方差；[PPO](./ppo/notes.md) 只跑了 3 个 seed；[SAC](./sac/notes.md) 用的是 5 个 seed 的 min/max 阴影。
   - [HIL-SERL](./hil-serl/notes.md) 每个任务只训练一次；[SimpleVLA-RL](./simplevla-rl/notes.md) 和 [π_RL](./pi-rl/notes.md) 没有报方差；[π\*0.6](./pi-star-06/notes.md) 没有数值表，正文、图和附录之间有多处对不上。
   - PPO 的效果高度依赖 code-level trick（2005.12729、2006.05990）。
   - Murphy §1.3.7 建议报告 IQM 加 bootstrap 置信区间。本项目复现时照此执行。

8. **样本效率，以及能否用 world model 替代环境。**
   - 各方法的交互量：DPPO 为 3e6–1e9 步（Robomimic 到 Furniture）；π_RL 需要大规模仿真；HIL-SERL 每任务 1 万到 10.8 万条真机转移；π\*0.6 每轮数百条真机 episode 加人工。这正是 [05 交叉线](../05_intersections/) 想解决的问题。
   - 按 Murphy §4.3.1 的 MBRL 博弈，model 只需要在当前策略诱导的访问分布上准确，否则策略会利用模型误差（model exploitation）。
   - 几个具体问题都还没有实验：PPO 的 clip 能否缓解 model exploitation（[PPO 笔记](./ppo/notes.md)）；DPPO 的结构化探索在 world model 里是否还有优势（[DPPO 笔记](./dppo/notes.md)）；IQL 的 expectile critic 能否用在 imagined rollout 上（[IQL 笔记](./iql/notes.md)）。

9. **RL 用在部署时（test-time RL）。**
   - [TTRL](../02_test_time_training/ttrl/notes.md) 用 majority vote 作为伪奖励跑 GRPO；[PAD](../05_intersections/pad/notes.md) 在部署时不用 reward，只做自监督适应。
   - 部署数据量小、分布不平稳，PPO / GRPO 的 ε、epoch 数、KL 正则应该怎样重新设定，本项目收录的论文都没有回答（[PPO 笔记](./ppo/notes.md)、[SimpleVLA-RL 笔记](./simplevla-rl/notes.md) 的开放问题）。VLA 方向的尝试见扩展阅读里的 EVOLVE-VLA。

---

## 7 扩展阅读（未收录，但与本线相关）

以下从选题时的 `dropped_notable` 列表中挑出：

| 论文 | arXiv | 为什么值得读 |
|---|---|---|
| Trust Region Policy Optimization | 1502.05477 | PPO 的直接前身。读懂 KL trust region 和单调改进的动机，才能理解 clip 在近似什么 |
| High-Dimensional Continuous Control Using GAE | 1506.02438 | λ 的 bias–variance 权衡。R1 的 Figure 4 中 PPO 取 λ=1 优于 0.95，背景就在这里 |
| TD3（同时参考 DDPG 1509.02971） | 1802.09477 | clipped double Q 被 SAC、IQL、HIL-SERL 沿用；DDPG 是 SAC 要解决的脆弱性的来源 |
| Conservative Q-Learning（CQL） | 2006.04779 | "压低 OOD 动作的 Q"这一路线，与 IQL 的 in-sample 路线对照；DPPO 的基线 Cal-QL 建立在它之上 |
| RLPD | 2302.02948 | HIL-SERL 的底座。HIL-SERL 正文没写的 critic ensemble、UTD 等细节要回这篇查 |
| SERL | 2401.16013 | HIL-SERL 的前作和软件栈，适合了解真机 RL 系统的工程构成 |
| DeepSeekMath | 2402.03300 | GRPO 的原始出处；R1、SimpleVLA-RL、WMPO、TTRL 用的 GRPO 都源自这里 |
| DAPO | 2503.14476 | clip-higher、dynamic sampling、去掉 KL，SimpleVLA-RL 直接借用了这三项 |
| Group Sequence Policy Optimization（GSPO） | 2507.18071 | 把 importance ratio 改成序列级，对应 R1 笔记里"ratio 是 token 级还是序列级"这个开放问题 |
| Does RL Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model? | 2504.13837 | RLVR 是否扩展了 base model 能力边界的核心争议，对应上面开放问题 3 |
| DSRL（Steering Your Diffusion Policy with Latent Space RL） | 2506.15799 | 在 noise space 里做 RL，不改 diffusion policy 的权重，和 DPPO / π_RL 的做法形成对照；π\*0.6 把它归为"只训练模型一部分"的方法 |
| What Can RL Bring to VLA Generalization?（RL4VLA） | 2505.19789 | 实证研究。π_RL 冻结 VLM、只训 action expert 的依据之一就是它的"RL 主要改善动作泛化" |
| RL-100 / GR-RL / RL Token | 2510.14830 / 2512.01801 / 2604.23073 | 真实世界 VLA RL 的前沿工作（都没有代码），可和 π\*0.6、HIL-SERL 放在一起比较数据配方 |
| EXPO-FT | 2605.25477 | 2026 年样本高效的 VLA RL 微调，有真机代码，可作复现备选 |
| EVOLVE-VLA | 2512.14666 | 用环境反馈对 VLA 做 test-time training，是 RL 与 TTT 交叉的 VLA 案例，与 TTRL、PAD 对照阅读 |

同一领域的其他综述（每个主题只收了 1 篇）：Offline RL 教程 2005.01643；RL for Large Reasoning Models 综述 2509.08827；Deep RL for Robotics 真实世界成功案例综述 2408.03539。RLHF 的背景可以读 InstructGPT（2203.02155）和 DPO（2305.18290）。

---

## 8 复现推荐（先复现再提假设）

选择标准有三条：有官方或成熟代码；算力从低到高排成阶梯；复现出的数字能直接作为本项目交叉线（world model 里的 RL）的 baseline。三个推荐都属于 on-policy PG 支线，这是有意的：[RWM](../05_intersections/robotic-world-model/notes.md) 在 world model 里跑 PPO，[WMPO](../05_intersections/wmpo/notes.md) 和 [World-Gymnast](../05_intersections/world-gymnast/notes.md) 在 world model 里跑 GRPO，它们都沿用这条支线。复现结果放在 `复现/<slug>/`，并附复现报告。评测至少跑 3 个 seed，按 Murphy §1.3.7 报 IQM 加 bootstrap 置信区间；指标和失败样例按 [`experiment/evaluation.md`](../../experiment/evaluation.md) 固定下来。

### ① PPO：on-policy 支线的调试基准（难度 easy）

- **为什么先做**：PPO 是 DPPO、π_RL、legged-gym、RWM 的优化器，也是 GRPO 的母体。后面任何 on-policy 实验出了问题，都要回到这里判断是算法的问题还是实现的问题。论文本身没有写 code-level trick，自己把这些 trick 逐项开关一遍，是理解 PPO 行为最便宜的方式。
- **算力**：论文没有报告硬件。2×64 MLP、每个任务 1M 步，单机 CPU 就能跑（笔记的判断）。整组实验是 3 种设置 × 2 个环境 × 3 个 seed。
- **最小实验**：
  1. 在现代 MuJoCo 的 Hopper 和 HalfCheetah 上比较三种设置：不加约束、clip ε=0.2、adaptive KL（ $d_{targ}=0.01$）。
  2. 严格使用 Table 3 的超参数：T=2048，Adam 3e-4，10 个 epoch，minibatch 64，γ=0.99，λ=0.95，policy 与 value 分开，不加 entropy。
  3. 每次更新记录 approx-KL 和 clip fraction。
  4. 在第一次更新时，沿 $\theta_{old}\to\theta_{new}$ 插值，复现 Figure 2 中 $L^{CPI}$、 $L^{CLIP}$ 和 KL 三条曲线。
  5. 可选：参照 Murphy 的建议，在 CartPole 上扫 GAE 的 λ∈{0, 0.95, 1}。
- **怎样算复现成功**：结果定性一致即可。不加约束时 HalfCheetah 会崩溃； $L^{CLIP}$ 在更新点附近达到峰值；clip 不劣于 adaptive KL。Gymnasium 的环境版本和论文用的 `-v1` 不同，绝对数值没法直接比。

### ② DPPO：生成式策略 RL 的最小闭环（难度 medium）

- **为什么做**：它是"BC 预训练加 PG 微调"在机器人上的最小可复现版本。π_RL 的 Flow-SDE 直接沿用了它的 two-layer MDP；[X-Sim](../03_real2sim2real/x-sim/notes.md) 的 student 是 diffusion policy，[ACDC](../03_real2sim2real/acdc-digital-cousins/notes.md) 的 cousins 只做了 BC，两者都可以接 DPPO 做仿真内的 RL 微调。官方 repo 提供预训练 checkpoint，不用自己采示范。
- **算力**：Robomimic Can（state 输入）在 L40 上每轮迭代 42.0 s（Table A2）。笔记按图的横轴估算，一次运行约 35 分钟，单卡即可。依赖 mujoco-py 和 robomimic，安装比较麻烦。
- **最小实验**：
  1. 加载官方预训练的 DP（K=20），用 DPPO 微调最后 K'=10 个去噪步，超参数按 Table A7/A9。
  2. 对照组是 Gaussian-MLP 加 PPO，每组 3 个 seed。目标是复现 Fig.5 / A9 中接近 100% 的趋势。
  3. 消融一：critic 用 $\tilde V(s)$ 还是 $V(s,a^1)$（Fig.A2）。
  4. 消融二：在 Square 上比较 $\sigma^{\text{exp}}_ {\min}$ 取 0.01 和 0.1，预期 0.01 会崩（Fig.A4）。
- **怎样算复现成功**：Can 的成功率接近 100%；σ 下限的消融复现出崩溃和不崩溃的分界。

### ③ SimpleVLA-RL：VLA RL，也是 world model 内 RL 的直接 baseline（难度 medium，算力门槛高）

- **为什么做**：WMPO 的附录注明，除 world model 以外的设置都沿用本文；World-Gymnast 也明确沿用了本文的四个技巧。所以"把仿真器换成 world model"这类假设，必须先有它在物理仿真里的数字作参照。官方提供了 1 条示范 SFT 的 checkpoint，可以跳过冷启动。
- **算力**：论文只写了 8×A800 80GB 全参数训练（§4.1），没有报告时长。README 给的配置是单机 8×A800 或双机 16×A800。LIBERO 需要大量并行环境进程，CPU 和渲染也可能成为瓶颈。
- **最小实验**：
  1. 从 libero-10（即 LIBERO-Long）的 traj1 SFT checkpoint（17.3%）出发，按 §4.1 设置：G=8、T=1.6、clip [0.8, 1.28]、不加 KL、开 dynamic sampling。
  2. 跑约 250–300 个 RL step，用 greedy 解码在 10 个任务 × 50 个场景上评测，目标接近 Table 5 的 91.7。
  3. 消融：把温度改回 T=1.0，或关掉 dynamic sampling，检查能否复现 Figure 3 中约 15 个点的差距。
  4. 训练中每 20 步记录"结果有成功也有失败的组"所占比例和 token 熵。
- **怎样算复现成功**：LIBERO-Long 接近 91.7；消融的方向和量级与 Figure 3 一致。

**算力不够时的备选**：
- [IQL](./iql/notes.md)：先做纯 CPU 的表格 U-maze（τ=0.5 对 τ=0.95），再跑 D4RL antmaze-medium-play（τ∈{0.5, 0.7, 0.9}，对照 Table 1 的 71.2）。单张 GTX1080 上 1M 次更新不到 20 分钟。它的 critic 是 SimDist 的 baseline，也是理解 π\*0.6 为什么放弃 AWR 的前置。
- [SAC](./sac/notes.md) v1：HalfCheetah，扫 reward scale ∈ {1, 5, 30}，CPU 可跑。它是 HIL-SERL、PAD、Dreamer actor 的共同底座。

---

## 维护备注

- 2026-09-28 已修正 `index.json`：DQN 的 why_key 不再说 2013 版有 target network（target network 是 2015 年 Nature 版才加的，见 [DQN 笔记](./dqn/notes.md)）；π_RL 的 why_key 去掉了与 source v3 不符的"41.6% → 85.7%"（v3 Table 4 为 IND 38.4 → 78.8 / 77.8，见 [π_RL 笔记](./pi-rl/notes.md)）。
- DPPO 的 ICLR 2025 和 SimpleVLA-RL 的 ICLR 2026（均为 Poster）都已在 OpenReview 核实，与 `index.json` 一致；两篇论文正文都没有写录用信息。
