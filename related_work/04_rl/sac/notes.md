# Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor

> arXiv [1801.01290](https://arxiv.org/abs/1801.01290) · ICML 2018 · 主题：04_rl · 层级：foundational
> 代码：https://github.com/haarnoja/sac（论文脚注给出的官方实现，基于 TensorFlow 和 rllab；README 已注明不再维护，改指向 https://github.com/rail-berkeley/softlearning）· 笔记依据：`source/main_no_comments.tex`（数字/公式）+ `paper.pdf`（图）

## 一句话

先在 maximum entropy RL 框架下推导 soft policy iteration：交替做 soft policy evaluation，以及把策略按 KL 投影到 $\exp(Q)$ 的 policy improvement，并证明它能收敛到策略类 $\Pi$ 内的最优策略。再用 replay buffer、soft Q/V critic 和重参数化的随机 Gaussian actor 近似这个过程，得到最大熵框架下第一个 off-policy actor-critic 算法。它在 Humanoid 这类高维 MuJoCo 任务上，样本效率和跨 seed 稳定性都明显好于 DDPG、PPO 和 SQL。

## 问题设定

- **形式化**（Sec 3.1）：infinite-horizon MDP $(\mathcal S,\mathcal A,p,r)$。状态和动作都是连续的，转移密度 $p$ 未知，reward 有界 $r\in[r_{\min},r_{\max}]$。实验中的输入是 MuJoCo 低维 state，完全可观测，reward 为 dense。设定是 model-free 的 online 交互，经验存入 replay buffer $\mathcal D$。输出是随机策略 $\pi_\phi(\mathbf a|\mathbf s)$。
- **目标**（Eq.1，`source/main_no_comments.tex` `eq:maxent_objective`）：
  $$J(\pi)=\sum_{t=0}^{T}\mathbb E_{(\mathbf s_t,\mathbf a_t)\sim\rho_\pi}\big[r(\mathbf s_t,\mathbf a_t)+\alpha\,\mathcal H(\pi(\cdot|\mathbf s_t))\big]$$
  温度 $\alpha\to0$ 时退化为标准 RL。正文此后省略 $\alpha$，因为它总能通过把 reward 乘以 $\alpha^{-1}$ 吸收进 reward。所以实践中真正需要调的是 **reward scale（等价于 $1/\alpha$）**。带折扣的严格定义见 App A Eq.14。
- **此前方法卡在哪**（Sec 1–2）：① TRPO、PPO、A3C 等 on-policy 方法每个梯度步都需要新样本，样本效率差；② DDPG 是 off-policy 的，但 deterministic actor 与 Q 相互耦合，对超参极其敏感，很难扩展到 Humanoid（21 维动作）；③ 已有的最大熵 off-policy 方法 soft Q-learning（SQL）本质上是 Q-learning：它估计的是**最优** soft Q，actor 只是一个近似采样器（approximate sampler），收敛性取决于这个 sampler 对 $\exp(Q)$ 的近似程度，连续动作下还需要复杂的近似推断。

## 核心方法

**1. Soft policy evaluation（Sec 4.1, Eq.2–3, Lemma 1）。** 对固定的 $\pi$，反复应用 soft Bellman backup：
$$\mathcal T^\pi Q(\mathbf s_t,\mathbf a_t)\triangleq r(\mathbf s_t,\mathbf a_t)+\gamma\,\mathbb E_{\mathbf s_{t+1}\sim p}[V(\mathbf s_{t+1})],\qquad V(\mathbf s_t)=\mathbb E_{\mathbf a_t\sim\pi}[Q(\mathbf s_t,\mathbf a_t)-\log\pi(\mathbf a_t|\mathbf s_t)]$$
证明思路（App B.1 Eq.15）：把熵并入 reward，定义 $r_\pi=r+\gamma\mathbb E_{\mathbf s'}[\mathcal H(\pi(\cdot|\mathbf s'))]$。这样问题就变成标准的 policy evaluation，压缩映射的结论可以直接套用。原文定义 $r_\pi$ 时漏写了 $\gamma$，展开 Eq.2 即可看出应该有。$|\mathcal A|<\infty$ 的假设只是为了保证熵有界。

**2. Soft policy improvement（Eq.4, Lemma 2）。** 对每个状态，把新策略 information projection 到策略类 $\Pi$（例如 Gaussian 族）：
$$\pi_{\text{new}}=\arg\min_{\pi'\in\Pi}\mathrm D_{\mathrm{KL}}\Big(\pi'(\cdot|\mathbf s_t)\,\Big\|\,\frac{\exp(Q^{\pi_{\text{old}}}(\mathbf s_t,\cdot))}{Z^{\pi_{\text{old}}}(\mathbf s_t)}\Big)$$
白板推导（App B.2 Eq.16–19）：$\pi_{\text{old}}\in\Pi$ 本身是可行解，所以最优解的 KL 不会比它大。又因为 $\log Z$ 只依赖状态，可以消去，得到 $\mathbb E_{\mathbf a\sim\pi_{\text{new}}}[Q^{\text{old}}-\log\pi_{\text{new}}]\ge V^{\text{old}}(\mathbf s)$（Eq.18）。把这个不等式代入 soft Bellman 方程，逐步展开 $Q^{\text{old}}$，就得到 $Q^{\text{new}}\ge Q^{\text{old}}$（Eq.19）。
**Theorem 1**：$Q^{\pi_i}$ 单调不减且有上界，因此收敛。在不动点上，任何 $\pi\in\Pi$ 的 soft value 都不会更高，所以收敛到的是 **$\Pi$ 内**最优。这一结论与策略的参数化方式无关，是 SAC 相对 SQL 在理论上的卖点。

**3. 函数近似版 SAC（Sec 4.2）。** 两步都不跑到收敛，而是各做一步 SGD 交替进行。网络包括 $V_\psi$ 及其目标网络 $V_{\bar\psi}$、两个 $Q_{\theta_{1,2}}$，以及策略 $\pi_\phi$：

- **V 损失**（Eq.5–6）：$J_V(\psi)=\mathbb E_{\mathbf s_t\sim\mathcal D}\big[\tfrac12(V_\psi(\mathbf s_t)-\mathbb E_{\mathbf a_t\sim\pi_\phi}[Q_\theta(\mathbf s_t,\mathbf a_t)-\log\pi_\phi(\mathbf a_t|\mathbf s_t)])^2\big]$。**$\mathbf s$ 取自 buffer，$\mathbf a$ 从当前策略重新采样。**作者说，从理论上讲 V 可以直接由 Q 和 $\pi$ 用单个样本无偏估计，单独学一个 V 网络是为了让训练更稳定。
- **Q 损失**（Eq.7–9）：$J_Q(\theta)=\mathbb E_{(\mathbf s_t,\mathbf a_t)\sim\mathcal D}\big[\tfrac12(Q_\theta(\mathbf s_t,\mathbf a_t)-r-\gamma V_{\bar\psi}(\mathbf s_{t+1}))^2\big]$。$(\mathbf s,\mathbf a,r,\mathbf s')$ 全部来自 buffer，TD target 用 target V 计算。
- **策略损失**（Eq.10→12）：直接最小化 Eq.4 的期望 KL。$Z$ 与 $\phi$ 无关，可以丢掉。再用重参数化 $\mathbf a_t=f_\phi(\epsilon_t;\mathbf s_t),\ \epsilon_t\sim\mathcal N$（Eq.11）得到
  $$J_\pi(\phi)=\mathbb E_{\mathbf s_t\sim\mathcal D,\epsilon_t\sim\mathcal N}\big[\log\pi_\phi(f_\phi(\epsilon_t;\mathbf s_t)|\mathbf s_t)-Q_\theta(\mathbf s_t,f_\phi(\epsilon_t;\mathbf s_t))\big]$$
  梯度（Eq.13）为 $\hat\nabla_\phi J_\pi=\nabla_\phi\log\pi_\phi(\mathbf a_t|\mathbf s_t)+(\nabla_{\mathbf a_t}\log\pi_\phi(\mathbf a_t|\mathbf s_t)-\nabla_{\mathbf a_t}Q(\mathbf s_t,\mathbf a_t))\nabla_\phi f_\phi(\epsilon_t;\mathbf s_t)$。和 DDPG 的 actor 更新（最大化 $Q(\mathbf s,\mu(\mathbf s))$）对照着看：SAC 相当于"注入噪声的 DDPG 式 pathwise gradient，再加一个熵项"。因为 Q 可微，这里用重参数化，而不是方差更大的 likelihood-ratio（REINFORCE）估计。
- **Twin Q**：两个 Q 独立训练，在 **V 的 target（Eq.6）和策略梯度（Eq.13）**里取 $\min(Q_{\theta_1},Q_{\theta_2})$，做法沿用 TD3，用来抑制 overestimation。作者说单个 Q 也能学会 21 维 Humanoid，但两个 Q 会"显著加快"训练。
- **Target 网络**：只对 V 维护目标网络，用 EMA 更新 $\bar\psi\leftarrow\tau\psi+(1-\tau)\bar\psi$，$\tau=0.005$。另一种做法是每 1000 个梯度步做一次 hard copy（App E）。
- **动作边界**（App C Eq.20–21）：$\mathbf a=\tanh(\mathbf u)$（逐元素），$\mathbf u\in\mathbb R^D$ 服从无界 Gaussian，密度记为 $\mu(\mathbf u|\mathbf s)$（原文符号：这里 $\mu$ 是密度而不是均值；均值、协方差由网络输出，Sec 4.2），于是 $\log\pi(\mathbf a|\mathbf s)=\log\mu(\mathbf u|\mathbf s)-\sum_{i=1}^D\log(1-\tanh^2(u_i))$。不做这个修正，熵项就算错了。

**4. 训练与推理流程**（Algorithm 1 + Table 1）：每轮先用 $\mathbf a\sim\pi_\phi$ 与环境交互并把 transition 存入 $\mathcal D$，再从 $\mathcal D$ 采 minibatch，依次更新 $\psi\to\theta_{1,2}\to\phi\to\bar\psi$。默认 1 个 env step 对应 1 个 gradient step。评估时取策略均值作为确定性动作，作为 MAP action 的近似（Sec 5.2, Fig 3a）。
**超参**（Table 1）：所有任务共用 Adam、lr $3\cdot10^{-4}$、$\gamma=0.99$、buffer $10^6$。所有网络都是 2 层 × 256 单元、ReLU，minibatch 256，$\tau=0.005$。只有 reward scale 按环境单独设置（Table 2）：Hopper、Walker2d、HalfCheetah、Ant 为 5，Humanoid-v1（17 维动作）为 20，Humanoid (rllab)（21 维动作）为 10。

## 关键实验与证据

**设置**（Sec 5, 5.1）：OpenAI Gym 的 Hopper、Walker2d、HalfCheetah、Ant、Humanoid（均为 -v1），外加 rllab Humanoid。baseline 包括 DDPG、PPO、SQL（也配了两个 Q）、TD3（用作者提供的实现，属于同期工作），App E 中还有 Trust-PCL。每个算法跑 5 个 seed，每 1000 个环境步做一次 evaluation rollout。曲线画的是均值，阴影为 min/max。DDPG 和 PPO 评估时关掉噪声，SQL 带噪声评估，SAC 用均值动作。

**主结果（Figure 1）**：论文**没有数值表**。下面是从曲线末端目测的近似值，只用来把握量级：

| 任务（步数） | SAC | 其他方法（约） |
|---|---|---|
| Hopper（1M） | ~3.3k | PPO ~2.9k，TD3/SQL ~2.8k，DDPG ~1.7k |
| Walker2d（1M） | ~3.7k | **TD3 ~4.7k 最高**，PPO ~3.8k，SQL ~2.7k，DDPG ~1.1k |
| HalfCheetah（3M） | ~15k | TD3 ~9.8k，SQL ~8.7k，DDPG ~6.8k，PPO ~2.6k |
| Ant（3M） | ~6k | TD3 ~4.6k，SQL ~4.2k，PPO ~2.5–3k，DDPG <1k |
| Humanoid-v1（10M） | ~6k | PPO 后期追平（~6k+），SQL ~5.5k，DDPG ≈ 0.5k |
| Humanoid rllab（10M） | ~5.5k | PPO ~2.3k，SQL ~1.8k，TD3/DDPG 接近 0 |

作者的表述是："简单任务上与 baseline 相当，困难任务上在学习速度和最终性能上都大幅领先。" 另外几点：DDPG 在 Ant、Humanoid-v1、Humanoid (rllab) 上都没有进展；SQL 能学会所有任务，但更慢，渐近性能也更差；PPO 慢，是因为高维任务上它需要很大的 batch。

**消融（Sec 5.2 + App E）**：
- **随机 vs 确定性（Fig 2）**：在 Humanoid (rllab) 上把 SAC 和一个 deterministic 变体各跑 5 个 seed。这个变体不最大化熵、没有 V 网络、用固定的探索噪声，等价于"带 twin Q、hard target、去掉 target actor 的 DDPG"。结果是 deterministic 变体跨 seed 的方差大得多。这是"熵带来稳定性"最直接的证据。
- **但 Fig 4（App E）更值得细看**：deterministic 变体和 hard-target SAC "在除 Humanoid (rllab) 以外的所有任务上都与 SAC 相当"。目测在 Walker2d、Ant 和 Humanoid-v1 上，deterministic 变体的末端回报甚至高于 SAC（Walker2d 约 4.4k vs 3.7k，Ant 约 6.5k vs 6.0k，Humanoid-v1 约 7.2k vs 6.4k）。也就是说，SAC 相对 DDPG 的大部分优势，很可能来自 twin Q 和这套实现细节；**熵的独特贡献主要体现在最难任务的稳定性上**。
- **评估方式（Fig 3a, Ant）**：用均值动作做确定性评估，回报高于随机评估。
- **Reward scale（Fig 3b, Ant，取 1/3/10/30/100）**：scale 太小时，策略接近均匀分布，无法利用 reward（scale=1 的曲线几乎是平的）；scale 太大时，前期学得快，但策略过早变成近似确定性，陷入局部最优。作者称 reward scale 是**唯一需要调的超参**。
- **$\tau$（Fig 3c，取 0.0001/0.001/0.01/0.1）**：$\tau=0.1$ 时训练失稳，return 掉到负值；$\tau=0.0001$ 时学得慢。作者说可用区间很宽，所有任务统一取 0.005。hard-target 变体在每个 env step 做多个 gradient step（Table 1：非 humanoid 任务 4 步，humanoid 1 步）时效果更好，但计算量也更大。

## 局限与论文没说的

- **作者承认的**：对 reward scale（也就是温度）敏感，最优值随环境而变，必须逐任务调（Fig 3b, Table 2）。未来方向是引入二阶信息（trust region）和更有表达力的策略类（Sec 6）。
- **理论与实践之间的差距**：收敛证明要求 $|\mathcal A|<\infty$，而且每一步都要精确 evaluation 和精确 KL 投影。实际算法在连续动作上用函数近似，每步只做一次 SGD，同时是 off-policy 的（正好处在 deadly triad 的位置）。论文对这一版没有任何收敛保证。
- **消融缺口**：①"单独的 V 网络能稳定训练"这一说法没有给出对照实验；②"twin Q 显著加速"也只有文字描述，没有图；③ 如前所述，Fig 4 显示熵在 6 个任务中只有 1 个上有明显优势，标题式结论"熵带来性能提升"的证据面比较窄。
- **评测口径**：只有 5 个 seed，阴影是 min/max 而不是置信区间。任务全部是 MuJoCo proprioceptive state 加 dense reward，没有测像素输入、sparse reward、真机和 partial observability。画出来的是纯 reward return，不是 SAC 实际优化的熵增广目标。baseline 方面，SAC 的 reward scale 按环境调过；TD3 直接用作者代码，在两个 Humanoid 任务上几乎为 0，而且 Humanoid-v1 的 TD3 曲线只画到约 4M 步，看不出是否专门为 humanoid 调过参。
- **策略类是单峰的**：最大熵的动机之一是"捕捉多个近优模式"（Sec 3.2）。但实际用的是 tanh-Gaussian，I-projection（reverse KL）本身又是 mode-seeking，所以多峰的 $\exp(Q)$ 只会被投影到其中一个峰上。后来 diffusion/flow policy 的 RL 工作（见 DPPO、π*0.6）要解决的正是策略表达力问题，而 SAC 的重参数化加 $\log\pi$ 在这类策略上并不直接适用。
- **温度固定，且对所有状态相同**：温度靠人工调 reward scale 来设定。后续版本（Haarnoja et al. 2018b "Soft Actor-Critic Algorithms and Applications"，不在本项目中）改成了带目标熵约束的自动 $\alpha$ 调节，并去掉了 V 网络。本项目的 HIL-SERL 用的就是 adaptive $\alpha$。
- **样本量仍然不小**：Humanoid 的曲线画到 10M 步（Fig 1 横轴），其中 Humanoid (rllab) 上 SAC 要到约 6M 步以后才接近 5k 的平台（Fig 1f，目测）。要直接在真机上从零训练还差得很远。论文**没有报告任何算力和 wall-clock 时间**。

## 在脉络中的位置

- **前驱**：maximum entropy IRL / control as inference（Ziebart 2008/2010，Todorov 2008，Toussaint 2009，Rawlik 2012）；G-learning（Fox et al.，本文引为 2016）；soft Q-learning（Haarnoja 2017，SAC 的直接前身）；PGQ、PCL/Trust-PCL 和 soft Q 与 policy gradient 等价性的系列工作；DPG/DDPG 与 SVG(0)（SAC 的 pathwise 策略梯度可以看成它们的随机、最大熵版本）；replay buffer 来自 DQN（[DQN](../dqn/notes.md)，2013 版只有 experience replay），target network 则来自 Mnih et al. 2015 的 Nature 版 DQN（本文引用的是 2015 版，不在本项目中）；double Q-learning 与同期 TD3 的 clipped double Q。主要的 on-policy 对照是 [PPO](../ppo/notes.md)。
- **后继 / 影响**（本项目内）：
  - [PAD](../../05_intersections/pad/notes.md)：直接构建在 SAC 之上，部署时用 inverse dynamics 这个 self-supervised 任务做 test-time adaptation，是 TTT × RL 的直接交点。
  - [HIL-SERL](../hil-serl/notes.md)：基于 RLPD（SAC 系 off-policy 方法），actor loss 带 adaptive $\alpha$ 的熵正则，在真机上训练操作策略。
  - [DPPO](../dppo/notes.md)：把 SAC 系的 RLPD 作为 Gaussian 策略的对照 baseline。
  - [DayDreamer](../../05_intersections/daydreamer/notes.md)：在 A1 四足真机上用 SAC 作为 model-free baseline。
  - [TD-MPC2](../../01_world_model/tdmpc2/notes.md)：以 SAC 为主要的 model-free baseline，自身的 policy prior 也采用最大熵目标。
  - [Dreamer](../../01_world_model/dreamer/notes.md)：action model 沿用 SAC 的 tanh-transformed Gaussian（引用本文），并用重参数化采样来反传 value 梯度。
  - [DreamerV3](../../01_world_model/dreamerv3/notes.md)：把 SAC 列为 baseline（视觉 DMC 表），并批评"约束到固定目标熵"的做法在 sparse reward 下探索慢。注意 DreamerV3 这里引用的是本文，但目标熵约束（自动调 $\alpha$）其实出自 SAC 后续版本；本文 v1 用的是固定温度（即手调 reward scale）。
  - [π*0.6](../pi-star-06/notes.md)：指出 SAC 式的 reparameterized gradient 难以用于没有可解析 log-likelihood 的 flow matching 策略。
  - [Robotic World Model](../../05_intersections/robotic-world-model/notes.md)：以 PPO、SAC 这类 model-free RL 交互需求太高、难以直接用于真机作为动机。
  - [SimDist](../../05_intersections/simdist/notes.md)：以 RLPD 和基于 SAC 的端到端 RL 在新域 finetune 时容易崩溃（灾难性遗忘）作为动机。
- **与本项目其他主题的联系**：
  - **World model**：world model 当 simulator 用时，在 imagination 里优化策略的通常是 actor-critic。SAC 的 tanh-Gaussian、重参数化和熵正则几乎是这条线的默认组件（Dreamer、TD-MPC2）。
  - **TTT**：PAD 说明 SAC 策略可以在部署时借助 self-supervised auxiliary loss 继续更新：部署时没有 reward，只优化 inverse dynamics 损失，更新共享的 feature extractor $\pi_e$ 和自监督 head，action head $\pi_a$ 不直接更新。最大熵策略"对模型和估计误差鲁棒"（引言引用 Ziebart 2010）这一性质，也和部署时的分布偏移直接相关。
  - **Real2sim2real**：大规模并行仿真里的主流选择是 PPO（[legged-gym](../../03_real2sim2real/legged-gym/notes.md)），因为那时样本便宜。SAC 的位置在样本贵的一端：真机训练、sim 之后在真机上 finetune、带 demo 的 off-policy 学习。熵最大化和 domain randomization 都是通过注入随机性换取鲁棒性，但一个作用在策略上，一个作用在环境上。
  - **RL 主线**：理解 SAC 之后，可以对照 IQL（offline 情形下怎样避免 OOD 动作的 Q 查询），以及 RLHF/GRPO 里的 KL/熵正则。概览见 [RL survey](../survey-rl-overview/notes.md)。

## 前置知识

1. MDP、Bellman 方程、policy iteration（evaluation 和 improvement 交替进行）。
2. Actor-critic 与 policy gradient：likelihood-ratio 估计和 reparameterization（pathwise）估计的区别。
3. DQN 的 replay buffer 和 target network；DPG/DDPG 的 deterministic actor 更新。
4. 熵、KL 散度，以及 I-projection（reverse KL）为什么是 mode-seeking。
5. Q-learning 的 overestimation bias，以及 double Q / TD3 的 clipped double Q。
6. 随机变量换元公式（tanh squashing 下的 log-density）。
7. （加分项）control as inference 与 soft Q-learning：为什么最优最大熵策略满足 $\pi^*\propto\exp(Q^*)$。

## 复现可行性

- **代码**：官方仓库 https://github.com/haarnoja/sac（TensorFlow + rllab，README 标明已不再维护）。README 推荐改用 https://github.com/rail-berkeley/softlearning（TF），PyTorch 用户可以用 rlkit。另外，现在所有主流 RL 库都有 SAC 的实现，不过大多是去掉了 V 网络、带自动 $\alpha$ 的后续版本，要复现这篇论文的 v1 需要注意区分。
- **环境**：MuJoCo 加 Gym -v1 环境和 rllab Humanoid。现在的 Gymnasium v4/v5 与 v1 在观测和 reward 定义上有差别，回报数值不能和论文直接比。
- **算力**：论文**没有报告**硬件和训练时间。可以确定的只有：网络是 2×256 的 MLP，CPU 就能跑；训练步数为 Hopper/Walker2d 1M、HalfCheetah/Ant 3M、Humanoid 10M（Fig 1 横轴）；默认每个 env step 做 1 次 batch 256 的梯度更新（Table 1）。所以瓶颈在串行的步数上，而不在显存。
- **最小可复现实验**：用 PyTorch 自己写一个 SAC v1（V + target V + twin Q + tanh-Gaussian，约几百行），在 HalfCheetah 上跑 1M 步、5 个 seed，超参按 Table 1，reward scale 取 5，每 1000 步用均值动作评估一次。另外做两组对照：① 把 reward scale 扫到 {1, 5, 30}，验证 Fig 3b 里"太小时接近均匀、太大时过早确定性"的现象；② 去掉熵项、改用固定高斯噪声的 deterministic 变体，看它是否像 Fig 4 那样在 HalfCheetah 上与 SAC 相当。如果还有余力，再在 Humanoid 上跑 SAC 和 deterministic 变体各 5 个 seed，检验 Fig 2 的方差差异。
- **预计难点**：tanh 下 log-prob 的数值稳定性（$1-\tanh^2$ 接近 0 时要加 eps，或者改用 softplus 形式）；reward scale 与环境版本耦合，需要重新调；V 的 target 要用当前策略的新采样动作，而不是 buffer 里的动作，这里容易写错；训练曲线和评估曲线（均值动作）的口径要和论文一致。

## 自测问题

1. 为什么 Eq.4 里的配分函数 $Z(\mathbf s)$ 可以忽略？把 $J_\pi$ 写成 $\mathbb E[\log\pi-Q]$，和 DDPG 的 actor 目标相比，多了什么、少了什么？
2. 从 Eq.18 推出 $Q^{\pi_{\text{new}}}\ge Q^{\pi_{\text{old}}}$，每一步用到了什么？$|\mathcal A|<\infty$ 用在了哪里？如果 $\Pi$ 不包含 $\exp(Q)/Z$，定理保证的"最优"指什么？
3. $J_V$ 中的动作从哪里采样，$J_Q$ 中的 $(\mathbf s,\mathbf a)$ 又从哪里采样？为什么这样整个算法能完全 off-policy，而 PPO 不能复用旧数据？
4. 为什么 reward scale 等价于 $1/\alpha$？结合 soft value 的定义，解释 Fig 3b 中 scale 过小和过大时各自的失败机制。
5. twin Q 的 $\min$ 用在哪两个更新里？Fig 4 中 deterministic 变体在 5/6 个任务上与 SAC 相当，这对"熵带来性能提升"这一结论意味着什么？
