# Offline Reinforcement Learning with Implicit Q-Learning

> arXiv [2110.06169](https://arxiv.org/abs/2110.06169) · ICLR 2022（Poster） · 主题：04_rl（offline RL / offline-to-online RL） · 层级：milestone
> 代码：https://github.com/ikostrikov/implicit_q_learning（论文 App B：JAX + Flax；仓库 README 称基于 JAXRL；摘要脚注的显示文字是这个地址，超链接实际指向 `implicit_policy_improvement`，以显示地址为准，已确认可访问） · 笔记依据：`source/*.tex`（数字/公式）+ `paper.pdf`（图）

## 一句话

IQL 在 SARSA 式 TD 学习里加一个用 **expectile regression 拟合、只对动作取上 expectile** 的状态价值 $V$，Q 再用 MSE 回归 $r+\gamma V(s')$：训练全程只在数据集里出现过的 $(s,a)$ 上查询 Q，却仍然能做 multi-step dynamic programming，逼近"数据支撑内的最优 Q"；最后用 advantage-weighted regression（AWR）提取策略。

## 问题设定

- **设定**：offline RL，只有固定数据集 $\mathcal D=\{(s,a,r,s')\}$，由未知的 behavior policy $\pi_\beta$ 采集，训练时不能与环境交互。输出一个策略 $\pi_\phi(a|s)$；奖励可观测，状态为低维向量（D4RL），MDP 定义见 `source/iclr2022_conference.tex` §3。
- **卡点**：标准 TD 目标（Eq.1）
  $$L_{TD}(\theta)=\mathbb E_{(s,a,s')\sim\mathcal D}\big[(r(s,a)+\gamma\max_{a'}Q_{\hat\theta}(s',a')-Q_\theta(s,a))^2\big]$$
  里的 $\max_{a'}$ 会查询**没见过的动作**，函数近似在这些动作上可能给出错误的值，且因为策略被定义为"最大化估计的 Q"，这常常导致 overestimation。
- **两类旧方案及其代价**（§1–2）：
  1. **multi-step dynamic programming + 约束/正则**：约束策略别离 $\pi_\beta$ 太远（BCQ、BEAR、BRAC、TD3+BC、AWAC、CRR），或把 OOD 动作的 Q 压低（CQL、Fisher-BRC）。这些方法在训练中仍然要用 OOD 动作查询 Q，只是加了约束，于是存在"策略改进幅度 vs. 对分布偏移的脆弱性"这个 trade-off。作者认为 in-distribution 约束本身也未必足以阻止外推。
  2. **single-step 方法**：只做一步 policy iteration，用 SARSA 学 $Q^{\pi_\beta}$ 再贪心提取（Onestep RL、AWR），或者干脆不学价值（Decision Transformer）。这类方法完全不碰 OOD 动作，但在需要把多条次优轨迹"stitching"起来的任务（AntMaze）上表现很差。
- **目标**：只用 in-sample 的动作（*in-sample learning*），同时保留 multi-step DP。

## 核心方法

**① 想学的量**（Eq.4）：只在数据支撑内做 max 的 Q-learning
$$L(\theta)=\mathbb E_{(s,a,s')\sim\mathcal D}\Big[\big(r(s,a)+\gamma\max_{a'\in\mathcal A,\ \pi_\beta(a'|s')>0}Q_{\hat\theta}(s',a')-Q_\theta(s,a)\big)^2\Big].$$
直接实现需要知道支撑，所以要换个办法去"隐式地"求这个 max。

**② Expectile regression**（§4.1，无编号公式）：$\tau\in(0,1)$ 的 expectile 是非对称最小二乘的解
$$m_\tau=\arg\min_{m}\ \mathbb E_{x\sim X}\big[L_2^\tau(x-m)\big],\qquad L_2^\tau(u)=|\tau-\mathbb 1(u<0)|\,u^2 .$$
$\tau=0.5$ 就是 MSE（均值）；$\tau>0.5$ 时，比 $m$ 小的样本被降权，$m$ 被推向分布的上沿（Figure 1 左/中）。它可以条件化为 $m_\tau(x)$，用 SGD 训练，梯度无偏。Lemma 1（§4.4）：若 $X$ 支撑有界、上确界为 $x^*$，则 $\lim_{\tau\to1}m_\tau=x^*$。脚注 2 说也可以用 quantile（$\ell_1$），但 expectile 只需改 MSE，且"效果略好"（没给数字）。

**③ 为什么要拆出 V**（§4.2）：最直接的做法是对 TD 目标 $r+\gamma Q_{\hat\theta}(s',a')$ 做 expectile regression。但这个目标的随机性同时来自 $a'$ 和环境转移 $s'\sim p(\cdot|s,a)$。上 expectile 会奖励"运气好转到了好状态"的样本（"lucky" sample），并不代表真的存在一个好动作。所以作者单独学一个状态价值 $V_\psi$，**只对动作分布取 expectile**，再用普通 MSE 回传给 Q，这样对转移随机性仍然取均值：
$$L_V(\psi)=\mathbb E_{(s,a)\sim\mathcal D}\big[L_2^\tau\big(Q_{\hat\theta}(s,a)-V_\psi(s)\big)\big]\quad\text{(Eq.5)}$$
$$L_Q(\theta)=\mathbb E_{(s,a,s')\sim\mathcal D}\big[(r(s,a)+\gamma V_\psi(s')-Q_\theta(s,a))^2\big]\quad\text{(Eq.6)}$$
两个损失都只用到数据集里的 $(s,a)$。注意 Eq.6 不再需要下一步动作 $a'$，所以数据是 $(s,a,r,s')$ 元组就够了，不需要 SARSA 的五元组。

**④ 理论**（§4.4 + App A）：把 Eq.5/6 的最优解递归地定义为
$$V_\tau(s)=\mathbb E^{\tau}_{a\sim\mu(\cdot|s)}[Q_\tau(s,a)],\qquad Q_\tau(s,a)=r(s,a)+\gamma\,\mathbb E_{s'}[V_\tau(s')].$$
- Lemma 2：$\tau_1<\tau_2\Rightarrow V_{\tau_1}(s)\le V_{\tau_2}(s)$，证明套用 policy improvement 的展开。
- Corollary 2.1：$V_\tau(s)\le\max_{a:\pi_\beta(a|s)>0}Q^*(s,a)$，其中 $Q^*$ 是限制在数据支撑内的最优 Q。
- Theorem 3：$\lim_{\tau\to1}V_\tau(s)=\max_{a:\pi_\beta(a|s)>0}Q^*(s,a)$。

结论是：$\tau=0.5$ 对应 SARSA（评估 $\pi_\beta$），$\tau\to1$ 对应只在支撑内取 max 的 Q-learning，IQL 覆盖了两者之间的整个谱。$\tau$ 越大，max 近似得越好，但优化越难，因此 $\tau$ 作为超参数。

**⑤ 策略提取：AWR**（Eq.7）
$$L_\pi(\phi)=\mathbb E_{(s,a)\sim\mathcal D}\big[\exp\big(\beta(Q_{\hat\theta}(s,a)-V_\psi(s))\big)\log\pi_\phi(a|s)\big].$$
$\beta\to0$ 时退化为 BC，$\beta$ 越大越接近取 Q 的 argmax。原文称，按前人工作，这个目标学到的是"在分布约束下最大化 Q"的策略（引 Peters & Schaal、AWR、AWAC）。它同样只用数据集动作。策略**完全不影响**价值学习，所以提取可以在 TD 学完之后做，也可以并行做；原文说并行学习"提供了一种"把 IQL 用于 online fine-tuning 的方式（§4.3）。补充一点：Eq.7 的写法是一个要**最大化**的加权对数似然，而 Algorithm 1 对 $L_\pi$ 做梯度下降，实际实现时应加负号，原文这里符号不严谨。另外 Eq.3 中的 $a'\sim\pi_\beta(\cdot|s)$ 应为 $\pi_\beta(\cdot|s')$。

**⑥ 训练流程**（Algorithm 1 + §4.3）：每一步依次更新 $V$（Eq.5）、$Q$（Eq.6），再对目标网络做 Polyak 更新 $\hat\theta\leftarrow(1-\alpha)\hat\theta+\alpha\theta$；另外有一个 AWR 循环。$V$ 和 $\pi$ 的更新里使用 clipped double Q-learning，即取两个 Q 的最小值。推理时直接用 $\pi_\phi$ 输出动作，不做规划（采样还是取 Gaussian 均值：原文未报告）。

## 关键实验与证据

**超参数**（App B）：Adam，lr $3\cdot10^{-4}$；所有网络都是 2 层 256 单元 ReLU MLP；actor 学习率用 cosine schedule；策略是状态无关 std 的 Gaussian；目标网络 $\alpha=0.005$；指数化的 advantage clip 到 $(-\infty,100]$。AntMaze 用 $\tau=0.9,\beta=10$，locomotion 用 $\tau=0.7,\beta=3$，Kitchen/Adroit 用 $\tau=0.7,\beta=0.5$，并在策略网络上加 dropout 0.1。AntMaze 的奖励统一减 1；locomotion 的奖励除以数据集中最好与最差轨迹回报之差。

**Toy U-maze**（§5.1，Figure 2）：到达目标奖励 10，其余为 0；以 0.25 的概率被转移到随机状态；数据集是 1 条最优轨迹加 99 条均匀随机轨迹；$\gamma=0.9$。Onestep RL 学到的价值离目标稍远就迅速衰减到 0，策略很差。IQL（$\tau=0.95$）的价值图与真实 $V^*$ 很接近。这个实验直观展示了 multi-step DP 能把奖励信号沿轨迹传播回去。

**D4RL 主结果**（Table 1，归一化分数）：
- locomotion-v2 total：IQL 692.4，CQL 698.5，Onestep RL 684.6，TD3+BC 677.4，DT 672.6。IQL 与最强方法基本持平。在 hopper-medium-expert 上 IQL 只有 91.5，低于 10%BC 的 110.9。
- antmaze-v0 total：IQL **378.0**，CQL 303.6，TD3+BC 163.8，Onestep RL 125.3，DT 112.2。优势集中在 medium/large 上，例如 large-play 39.6 vs CQL 15.8，large-diverse 47.5 vs 14.9。one-step 方法和 DT 在 medium/large 上几乎全是 0。在 umaze-diverse 上 IQL 62.2 低于 CQL 84.0。
- 合计 1070.4 vs CQL 1002.1。Kitchen total 159.8（BC 154.5），Adroit total 118.1（BC 104.5，CQL 93.6）。作者自己也说 Kitchen/Adroit 区分度低，大多数 RL 方法和模仿学习差不多（分项结果见 App B Table 3）。
- **运行时间**：IQL 20m，CQL（作者的 JAX 重实现）80m，DT 960m（原实现）。原版 CQL 跑 1M 次更新要 4 小时以上（§5.2 Runtime）。结论里说单张 GTX1080 跑 1M 次更新不到 20 分钟（§6）。

**消融：$\tau$**（Figure 3）：在 4 个 medium/large AntMaze 上比较 $\tau\in\{0.5,0.7,0.9\}$，$\tau=0.9$ 的学习曲线远高于另外两个，$\tau=0.5$（即 SARSA）基本学不到东西。这直接说明"上 expectile ≈ 支撑内 max"就是 stitching 能力的来源。论文只给了曲线，没有对应的数值表。

**Offline→online fine-tuning**（Table 2，先 offline 训练，再做 1M 步 online RL，App C）：
- antmaze-v0 total：IQL 370.1→**473.7**，CQL 151.5→231.1，AWAC 107.7→108.3。IQL 在 large-play 上 25.5→46.0，在 medium-play 上 72.0→95.0。
- hand-v0 total（dexterous binary 任务）：IQL 38.1→**124.0**，AWAC 46.7→103.1，CQL 31.5→9.9。pen-binary 上 IQL 60.7 低于 AWAC 的 70.3（作者承认）。
- online 阶段每个环境步做 1 次梯度更新，其他设置与 offline 阶段相同。dexterous 任务用 $\tau=0.8,\beta=3$，offline 只训练 25000 步，并加 $\sigma=0.03$ 的 Gaussian 探索噪声。

## 局限与论文没说的

**作者承认的**：$\tau$ 越接近 1 越难优化，只能当超参数调（§4.4）；Kitchen/Adroit 区分度低；fine-tuning 在 pen-binary 上不如 AWAC；one-step 与 multi-step 方法很难直接对比（Table 1 脚注）；作者用 CQL 原实现复现的离线结果比原论文差（尤其 medium/large AntMaze），所以离线对比用 CQL 原论文报告的数字，而 Table 2 的 fine-tuning 只能用自己跑的 CQL（App C；App C 里两处都写成 Table 2，前一处应指 Table 1，原文笔误）。注意 Table 1 中 locomotion -v2 的 CQL 数字并非原文数字，而是用作者推荐的实现（young-geng/CQL）跑出来的（§5.2）。

**我读出来的**：
1. **有限 $\tau$ 时 $V_\tau$ 依赖动作频率，而不仅仅依赖支撑**。expectile 是对 $a\sim\mu(\cdot|s)$ 加权的统计量，如果好动作在数据里很稀少，就需要更接近 1 的 $\tau$ 才能"看见"它。Theorem 3 只是 $\tau\to1$ 时的极限结果，并且假设精确求解、表格化、没有近似误差，没有给出有限样本或函数近似下的分析。在连续动作加神经网络的情形下，"$\pi_\beta(a|s)>0$ 的支撑"本身就是一个模糊的概念。
2. **缺少关键消融**：没有实验比较"直接对 TD 目标做 expectile"（§4.2 那个被否定的版本）和拆出 V 的版本，所以"lucky sample"这个论证只停留在推理层面。也没有 $\beta$ 的消融、clipped double Q 的消融，expectile 与 quantile 的对比没有给数字。
3. **超参数按领域设置**（AntMaze、locomotion、Kitchen/Adroit、fine-tuning 各用一组 $\tau/\beta$），论文没有说明是怎么选的。在 offline RL 中，如果用 online 评估来调参，本身就与"纯离线"的设定相冲突。
4. **评测口径**：Table 1 没有报告标准差；AntMaze 只说用了 100 条评估轨迹，没写 seed 数；locomotion 用的是 -v2 数据集而 AntMaze 用 -v0；baseline 数字来源不一（有作者提供的、原文的、重实现的）。Table 1 和 Table 2 中 IQL 自己的离线数字也不完全一致（例如 umaze 分别是 87.5 和 86.7），说明两者来自不同的运行。
5. **策略提取的上限**：AWR 本质上是加权的 BC，策略很难超出每个状态下数据里最好的动作，"通过泛化超越 behavior"只能依靠函数近似。它用的是单峰 Gaussian，面对多峰数据时可能在模式之间取平均。exp 权重会大幅压低大部分样本的权重（这正是 [π*0.6](../pi-star-06/notes.md) 改用 advantage conditioning 的理由）。
6. **只在低维状态、仿真环境中测试**：没有图像输入、没有真实机器人，也没有 VLA 规模的策略。fine-tuning 需要 1M 个 online 步，对真机来说代价很高。

## 在脉络中的位置

- **前驱**：
  - SARSA / fitted Q evaluation 与 TD 学习（Eq.1 与 [DQN](../dqn/notes.md) 同源，原文没有引用 DQN）。
  - 约束或正则类 offline RL：BCQ（App D 专门对比：BCQ 中采样动作数 $N$ 的作用类似 IQL 的 $\tau$，但 BCQ 的生成模型仍然可能产生 OOD 动作）、BEAR、BRAC、CQL、Fisher-BRC（同一作者）、TD3+BC。
  - 加权回归式策略提取：RWR（Peters & Schaal）、AWR、AWAC、CRR；one-step 方法：Onestep RL。
  - 统计工具是 expectile regression，与 quantile regression（Koenker & Hallock）密切相关；distributional RL 中的 QR-DQN/IQN 是它的"近亲"，但那里建模的是转移随机性，IQL 建模的是动作随机性（§2）。
  - clipped double Q-learning 来自 TD3。
- **后继 / 影响**（本项目内能找到证据的）：
  - [π*0.6](../pi-star-06/notes.md)：把 IQL 归入 AWR 类方法（`source/main.tex` §policy extraction），认为这类方法会丢弃或大幅降权数据，因此改用 advantage conditioning；实验中 AWR 基线的吞吐量更低。
  - [DPPO](../dppo/notes.md)：以 IDQL 为基线，IDQL 的 critic 就是 IQL 式的 expectile V 加 MSE Q，actor 换成了 diffusion policy（`source/body/appendix.tex`；DPPO 原文没有直接提 IQL，这是按 IDQL 的损失形式对照得出的）。
  - [SimDist](../../05_intersections/simdist/notes.md)：在真实四足和操作任务上用 IQL 的 offline-to-online fine-tuning 作为基线，四足任务中 IQL 的 value function 在仿真里预训练。
  - [TD-MPC2](../../01_world_model/tdmpc2/notes.md)、[Dreamer 4](../../01_world_model/dreamer4/notes.md) 的笔记中都以 IQL 为参照，讨论如何对抗分布偏移。
  - 命名撞车：[RL 综述](../survey-rl-overview/notes.md) 的 MARL 章节里 "IQL" 指的是 Independent Q-Learning，与本文无关。
- **与本项目其他主题的联系**：
  - **World model**：IQL 是"model-free、in-sample"这一极端：从不查询没见过的动作。world model 当 simulator 训练 RL（[WMPO](../../05_intersections/wmpo/notes.md)、[World-Gymnast](../../05_intersections/world-gymnast/notes.md)、[Robotic World Model](../../05_intersections/robotic-world-model/notes.md)、[Dreamer 4](../../01_world_model/dreamer4/notes.md)）则是另一个极端：在模型里生成新动作的后果，要面对的是策略利用模型误差（model exploitation）。两者对付的是同一种分布偏移问题（这是我的解读，不是这些论文的原话）。IQL 的 expectile critic 也可以直接在 imagined rollout 数据上训练。
  - **TTT**：offline 预训练再 online fine-tuning 可以看作"部署后继续学习"，不过靠的是奖励，而 [PAD](../../05_intersections/pad/notes.md) 靠的是自监督。[TTT-Discover](../../02_test_time_training/ttt-discover/notes.md) 的 $e^{\beta R}$ 加权在形式上与 Eq.7 同源。
  - **real2sim2real**：offline RL 直接从真实日志学习，是一条绕过建仿真器的路；SimDist 则把 IQL 当作"仿真预训练 value → 真机 fine-tune"的基线，与 [HIL-SERL](../hil-serl/notes.md) 这类真机 online RL 形成对照。

## 前置知识

1. MDP、Bellman 方程、TD 学习与目标网络（[DQN](../dqn/notes.md)）；SARSA 与 Q-learning 的区别（on-policy 评估 vs. 取 max）。
2. Offline RL 的 extrapolation error、overestimation，以及 behavior policy、support 这些概念。
3. Advantage $A=Q-V$；AWR/AWAC 的推导（KL 约束下策略改进的闭式解是 $\pi\propto\pi_\beta e^{\beta A}$，再投影回参数化策略）。
4. Quantile / expectile regression（非对称损失估计条件统计量），以及 distributional RL 的基本思路。
5. Actor-critic 与 clipped double Q（TD3 / [SAC](../sac/notes.md)）。
6. D4RL 基准（locomotion / AntMaze / Kitchen / Adroit）与归一化分数，以及什么是 "stitching"。

## 复现可行性

- **代码**：官方 JAX/Flax 实现开源（上面的 GitHub 地址），README 称基于 JAXRL，提供 `train_offline.py` 和 `train_finetune.py`，以及 `configs/` 下的 mujoco/antmaze/antmaze_finetune/kitchen 配置。论文没有写依赖版本。
- **算力**：非常低。论文给出的是单张 GTX1080 上 1M 次更新不到 20 分钟（§6），Table 1 的 runtime 为 20m。完整复现 Table 1 需要"任务数 × seed 数"次运行（locomotion 用 10 个 seed），单卡几天内可以跑完。
- **数据/环境**：D4RL（MuJoCo locomotion -v2、AntMaze -v0、Kitchen/Adroit -v0）；fine-tuning 还需要能在线交互的 AntMaze 环境，以及 AWAC 用过的 dexterous binary 环境（pen/door/relocate-binary-v0，来自 Rajeswaran et al.，§5.3）。预计最大的难点是老版 D4RL 与 MuJoCo 的环境安装（这是经验判断，论文没有提到）。
- **最小可复现实验**：
  1. **表格版**（约 1 小时，纯 CPU）：自己搭 §5.1 的 U-maze（奖励 10，以 0.25 概率随机转移，1 条最优 + 99 条随机轨迹，$\gamma=0.9$），分别用 $\tau=0.5$ 和 $\tau=0.95$ 跑 Eq.5/6 的表格迭代，画价值图，对照 Figure 2(c)(d)。
  2. **D4RL 版**：在 antmaze-medium-play-v0 上用 $\tau\in\{0.5,0.7,0.9\}$、$\beta=10$ 跑 1M 步，复现 Figure 3 的趋势，并对照 Table 1 的 71.2。再在 hopper-medium-replay-v2 上用 $\tau=0.7,\beta=3$ 对照 94.7。
- **预计难点**：AntMaze 分数在 seed 间方差大，需要 100 条评估轨迹；奖励处理（AntMaze 减 1、locomotion 归一化）和 advantage clip 这些细节会影响结果；分数对 $\tau/\beta$ 敏感。

## 自测问题

1. 如果直接对 $r+\gamma Q(s',a')$ 做上 expectile regression，在随机转移的环境里会高估什么？为什么拆成"V 对动作取 expectile + Q 用 MSE"就能避免？这个修正在什么前提下才成立？
2. 写出 CQL、TD3+BC、BCQ 各自在哪一步会查询 OOD 动作，再说明 IQL 的 Eq.5、Eq.6、Eq.7 为什么都不会。
3. 说明为什么 $\tau=0.5$ 时 IQL 退化为评估 $\pi_\beta$ 的 SARSA，而 $\tau\to1$ 时趋近支撑内的 Q-learning。如果某个状态下好动作只占数据的 1%，$\tau=0.7$ 的 $V$ 会离支撑内的 max 多远，为什么？
4. 为什么 Onestep RL 在 locomotion medium-expert 上表现不错，在 AntMaze medium/large 上却几乎为 0？用 multi-step DP 和 stitching 解释 Figure 2 与 Figure 3。
5. AWR 中 $\beta\to0$ 和 $\beta\to\infty$ 分别对应什么？为什么要把 $\exp(\beta A)$ 截断到 100？为什么策略与价值解耦会让 online fine-tuning 更容易？
