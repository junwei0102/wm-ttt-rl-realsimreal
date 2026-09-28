# Closing the Sim-to-Real Loop: Adapting Simulation Randomization with Real World Experience

> arXiv [1810.05687](https://arxiv.org/abs/1810.05687) · ICRA 2019（arXiv v1 2018-10，v4 2019-03） · 主题：03_real2sim2real · 层级：foundational
> 代码：none（论文、arXiv 页和项目页 https://sites.google.com/view/simopt 都没有代码链接；论文称实验视频放在项目页，项目页另有摘要和 FAQ） · 笔记依据：`source/simopt.tex`（数字/公式）+ `paper.pdf`（图）

## 一句话

SimOpt 把 domain randomization 的参数分布 $p_\phi(\xi)$ 当成优化对象，交替执行三步：先在当前分布上用 PPO 训练 policy，再到真机上跑 3 条 rollout，最后用同一个 policy 在仿真里对大量采样的 $\xi$ 跑 closed-loop rollout，按"仿真观测轨迹与真实观测轨迹的 discrepancy"用 REPS 在 KL trust region 内更新一个 Gaussian 分布。这样只靠部分观测、不需要真实世界的 reward，就把 real→sim→real 的循环闭合了。

## 问题设定

- **输入**：一个参数化、不可微的 GPU 物理仿真器（NVIDIA Flex）；仿真参数 $\xi$，包括机器人关节 compliance/damping、action scaling，以及绳子、peg、柜子的物理量（Table I/II）；一个人为给定、偏保守的初始分布 $p_{\phi_0}$；可以在真机上执行 policy 并记录**部分观测** $o_t$（7 维关节角，加上 DART 从深度图跟踪到的物体 3D 位置，见 §IV-D）。
- **输出**：更新后的仿真参数分布 $p_\phi$，以及在它上面训练出来、直接部署的 policy $\pi_{\theta,p_\phi}$。
- **假设**：reward 只在仿真里计算（仿真有 full state），真实世界不需要 reward instrumentation，也不需要 full state，例如不用测绳子直径和 compliance（§IV-A）；真实观测在仿真里有同维度的对应量；场景初始位置可以用 DART 在仿真里初始化；仿真器的**模型结构**足以描述现实，只需要调参数。
- **之前卡在哪**（§I–II；Fig. 4/5 见 §IV-C，replay 真实动作一点见 Appendix A）：(1) DR 的分布靠专家手调，而且太宽反而有害：Fig. 4 里 peg 比孔大、绳子太短等实例根本无解；Fig. 5 里方差一大，policy 就变得保守，甚至训不出来。(2) 迭代式 sysID（Tan et al. 2016；Zhu et al. 2018）要把仿真器逐步设到真实状态、重放真实动作，这需要 full state；Zhu 2018 还要在真实世界估计 reward。(3) Grounded action transformation（Hanna & Stone 2017）要额外拟合真实 forward dynamics 和仿真 inverse dynamics；Farchy 2013 需要人在环挑参数。(4) 最接近的 adaptive EPOpt（Rajeswaran 2016）用 risk-sensitive objective 加粒子滤波做贝叶斯更新，并且只在仿真里验证过。

## 核心方法

**1. DR 目标**（`source/simopt.tex` Eq. 1）。仿真器本身是确定性的，作者借用 reparameterization trick 的说法，把随机性放到参数上：$\xi\sim p_\phi(\xi)$，动力学为 $P(s_{t+1}\mid s_t,a_t,\xi)$。

$$\max_\theta\ \mathbb{E}_{P_{\xi\sim p_\phi}}\big[\mathbb{E}_{\pi_\theta}[R(\tau)]\big]$$

**2. 理想的 SimOpt 目标**（Eq. 2）。找一个 $\phi$，使得在该分布上训练出的 policy，在仿真与真实世界里产生的观测轨迹尽量一致：

$$\min_\phi\ \mathbb{E}_{P_{\xi\sim p_\phi}}\Big[\mathbb{E}_{\pi_{\theta,p_\phi}}\big[D(\tau^{ob}_\xi,\tau^{ob}_{real})\big]\Big]$$

这是一个 bilevel 问题：policy 依赖 $\phi$，每换一个 $\phi$ 都要重新做一次 RL，再上真机评估，不可行。

**3. 迭代近似**（Eq. 3）。把 policy 固定为上一轮训练好的 $\pi_{\theta,p_{\phi_i}}$，用它同时采集真实轨迹、评估新分布：

$$\min_{\phi_{i+1}}\ \mathbb{E}_{P_{\xi_{i+1}\sim p_{\phi_{i+1}}}}\Big[\mathbb{E}_{\pi_{\theta,p_{\phi_i}}}\big[D(\tau^{ob}_{\xi_{i+1}},\tau^{ob}_{real})\big]\Big]\quad \text{s.t.}\quad D_{KL}(p_{\phi_{i+1}}\,\|\,p_{\phi_i})\le\epsilon$$

KL 约束的作用是让新分布不要走出旧 policy 的 trust region：policy 只在 $p_{\phi_i}$ 覆盖的场景里训练过，离开这个范围，它在仿真中的行为就不再能可靠地和真实轨迹对比。

**4. Discrepancy**（Eq. 4）。它是加权的 $\ell_1+\ell_2$ 距离，$W$ 是每个观测维度的重要性权重：

$$D=w_{\ell_1}\sum_{i=0}^T\big|W(o_{i,\xi}-o_{i,real})\big|+w_{\ell_2}\sum_{i=0}^T\big\|W(o_{i,\xi}-o_{i,real})\big\|_2^2$$

距离计算前还会先过一个 Gaussian filter 来缓解时间错位。Table III/IV 给出的取值：$w_{\ell_1}=0.5$，$w_{\ell_2}=1.0$，smoothing std 为 5 个 timestep，截断为 4。peg 任务里关节角权重 0.05，peg 位置（当前步和上一步）权重 1.0；drawer 任务里关节角权重 0.5，drawer position（3D，Table IV 的原词；正文说观测是 drawer handle 的 3D 位置）权重 1.0。注意 policy 的输入和计算 $D$ 用的观测不必相同。

**5. 分布更新：REPS**（§III-C）。$p_\phi=\mathcal N(\mu,\Sigma)$，真机实验用 full covariance。仿真器不可微，所以用基于 relative entropy policy search（Peters et al. 2010）的采样式、gradient-free 更新，把仿真器当 black box，只需要样本 $\xi_j\sim p_{\phi_i}$ 和代价 $c_j=D(\tau^{ob}_{\xi_j},\tau^{ob}_{real})$。**论文没有写出 REPS 的更新式**，下面是标准 episodic REPS 在 Gaussian 情形下的形式，供白板推导用：

$$q(\xi)\propto p_{\phi_i}(\xi)\,e^{-c(\xi)/\eta},\qquad \eta^*=\arg\min_{\eta>0}\ \eta\epsilon+\eta\log\frac1N\sum_{j=1}^N e^{-c_j/\eta}$$

$$w_j=e^{-c_j/\eta^*},\quad \mu'=\frac{\sum_j w_j\xi_j}{\sum_j w_j},\quad \Sigma'=\frac{\sum_j w_j(\xi_j-\mu')(\xi_j-\mu')^\top}{\sum_j w_j}$$

Table III/IV 中的 "KL-threshold 1.0" 就是 $\epsilon$，"Minimum temperature of sample weights 0.001" 应对应 $\eta$ 的下界，用来防止权重全部集中到单个样本上（论文没有解释这一项，这里是按 REPS 的常规做法推测）。

**6. 完整流程**（Algorithm 1，Fig. 3）。每轮 SimOpt 迭代：
① 在 $p_{\phi_i}$ 上用并行 PPO 训练 policy。sim2sim 实验里用上一轮权重做 warm start，第一轮之后 PPO 迭代数因此从 200 降到 10（§IV-C）；真机实验每轮分别做 100（peg）/200（drawer）iter RL，是否 warm start 原文没有说明（§IV-D）。
② 真机执行 3 条 rollout，记录 $\tau^{ob}_{real}$。
③ 采样 9600 个 $\xi$，用**同一个 closed-loop policy** 在仿真里 rollout，算出 $c(\xi)$。
④ 用 REPS 更新分布。每轮 SimOpt 迭代里重复③④若干次（peg 3 次，drawer 20 次），全部复用同一批真实 rollout，以节省真机次数（§IV-D）。

"Timesteps per simulation parameter sample = 453" 恰好等于 3×151（episode 长度 150）。我**推测**每个 $\xi$ 要对 3 条真实轨迹各跑一遍，论文没有明说。部署时就直接执行最终 policy，policy 不以 $\xi$ 或历史为条件，也没有在线适应。

**7. 为什么用 closed-loop 而不是 replay**（Appendix A）。绳子的弯曲状态只能初始化，不能持续跟踪，所以没法像 Tan 2016、Zhu 2018 那样逐步把仿真器设到真实状态。作者在仿真实验中观察到：把 cabinet X 位置从 cost 里去掉以后，closed-loop 执行仍能给出有意义的参数更新，open-loop 重放真实动作则不行。原因是 closed-loop policy 的行为依赖于具体场景，这种依赖会反映在关节轨迹上。这一点只有定性描述，没有数字。

## 关键实验与证据

- **DR 对方差敏感**（§IV-C，Fig. 5）：仿真 Franka 开抽屉，只随机化柜子横向位置 X；policy 是 2 层、每层 64 单元的 MLP，PPO 训练 200 iter。std 取 2 cm（$\sigma^2=7\text{e-}4$）时能开抽屉；$\sigma^2=0.001$ 和 $0.01$（10 cm）时只能够到把手，打不开抽屉。作者提醒，reward 曲线差距看起来不大，是因为 reaching 项占了主导。三轴都随机化时，只有 2 cm 能收敛，其他设置连把手都够不到；方差再大就会出现物理仿真崩溃、关节打到极限。Fig. 4 展示了 peg 任务的宽分布里大量实例无解。
- **sim2sim 迁移**（Fig. 6/7）：目标场景里柜子偏移 15 cm 时，需要 3 轮 RL+SimOpt 才能开抽屉；偏移 22 cm 时需要 5 轮。作者指出，用朴素 DR 覆盖这么大的偏移，std 至少要 10 cm，而 Fig. 5 已经显示这样训不出能开抽屉的 policy。Fig. 6 显示分布均值逐步从 source 移向 target，同时分布变窄。
- **真机 swing-peg-in-hole**（ABB Yumi，§IV-D-1）：每轮 SimOpt 做 100 iter RL（约 7 分钟）、3 条真机 rollout、3 次 REPS 更新（每次 9600 个样本）。第 1 轮后 peg 已接近孔但插不进；**2 轮后 20 次试验成功率 90%**。变化最大的是绳子参数和 action scaling（Table II），例如 action scaling 从 0.5 变成 [0.25…0.44]，rope bending damping 从 0.01 变成 0.49，rope torsion damping 从 0.1 变成 0.48。Fig. 9 中关节 compliance 与 damping 出现相关，作者的解释是二者可以互相补偿。
- **真机 drawer opening**（Franka Panda，§IV-D-2）：每轮 SimOpt 做 200 iter RL（约 22 分钟）、3 条真机 rollout、20 次 REPS 更新（每次 9600 个样本）。更新前，夹爪不垂直于把手，一根手指从下方顶把手，结果夹爪被顶开。**1 轮更新后，20 次试验全部打开抽屉**。Table I 中 drawer handle friction 从 0.001 变到 2.13，action scaling 从 0.26 变到 [0.19…0.35]。
- **实验缺口**：真机上既没有 baseline（固定 DR、宽 DR、纯 sysID），也没有更新前的量化成功率。$\epsilon$、$W$、真机 rollout 条数、open-loop 与 closed-loop 的对比都没有做定量消融。仿真里的定量结果只有 Fig. 5 的 DR 方差实验，以及论文称为 ablation 的 sim2sim 实验（只报告了所需迭代数 3/5 轮）。

## 局限与论文没说的

- **作者承认**：只学了 uni-modal Gaussian，未来想扩展到 multi-modal 分布和更复杂的生成式仿真模型；想加入视觉、触觉等高维模态，既用作 policy 观测，也作为随机化因素（§V）。在 §III-C 里，作者也提到可以换成可微仿真器、adversarial discriminator 形式的 $D$，但都没有实现。
- **方差会塌缩**：Eq. 3 是 risk-neutral 的期望 discrepancy，最优解是落在 $\arg\min_\xi c(\xi)$ 上的 Dirac 分布。KL 约束只能减慢收敛，目标里没有任何项会保持"合适的随机化宽度"。所以 SimOpt 本质上更接近"带退火随机化的迭代 sysID"，最终宽度取决于迭代次数和 $\epsilon$（Fig. 6 中分布确实在变窄）。论文只报告了最终均值，没有给出最终方差的数值。后来的 BayesSim、DROPO 等工作改用 posterior 或似然来刻画分布（[reality gap 综述](../survey-reality-gap/notes.md) 在 DR 部分提到 DROPO，在 open problems 的 neural posterior estimation 部分提到 BayesSim）。
- **可辨识性**：参数之间存在补偿，比如 compliance 和 damping，Fig. 9 就体现了这一点。每轮只有 3 条轨迹，又只在 $\pi_i$ 激发出的行为下做匹配，得到的解不唯一；学到的"action scaling"本质上是在吸收未建模的执行器误差。
- **结构性误差修不了**：参数调得再好，也无法补偿仿真器缺失的动力学项。后来的 [ASAP](../asap/notes.md) 用 delta action model 来学这部分残差。
- **评测口径**：每个机器人只有 1 个任务，每个任务只有 20 次试验，没有 seed 方差；sim2sim 只改了初始状态（柜子位置），没有测动力学参数的 sim2sim。
- **门槛**：需要 64 GPU 集群（型号未报告）；REPS 采样阶段的耗时没有报告；需要带 3D articulated model 的 DART 跟踪；$W$ 和 reward 权重都要手调；中间轮次的 policy 要直接上真机，存在安全风险；如果现实参数在每个 episode 之间都在变，单一分布加无记忆 policy 只能折中。

## 在脉络中的位置

- **前驱**：system identification（Ljung）；[domain randomization](../domain-randomization/notes.md)（Tobin 2017）、Peng 2018 dynamics randomization、OpenAI 灵巧手；EPOpt / adaptive EPOpt（Rajeswaran 2016）；Tan 2016 迭代 sysID；Zhu 2018；GAT（Hanna & Stone 2017）；REPS（Peters 2010）；[PPO](../../04_rl/ppo/notes.md)；同组 Liang et al. 2018 的 GPU 并行 RL 基础设施（Flex）。
- **后继 / 影响**：[reality gap 综述](../survey-reality-gap/notes.md) 把它归入"online and iterative system identification"，并与 ADR、DROPO 并列为自动化 DR 分布的方法。[ASAP](../asap/notes.md) 把它列为 offline SysID 中"explicitly identifying critical simulation parameters"的代表，自己则进一步学习 delta action 来补残差。[X-Sim](../x-sim/notes.md) 把它作为可兼容的 DR 选项引用，[digital cousins](../acdc-digital-cousins/notes.md) 把它归为 simulator augmentation，[SIMPLER](../simpler/notes.md) 把它当作复杂动力学操作任务上的 DR 案例。[PhysTwin](../phystwin/notes.md) 从视频反推可形变物体的物理参数，与 SimOpt 调绳子参数属于同一类 real-to-sim 物理参数估计。[RialTo](../rialto/notes.md) 走的是另一条路：重建几何与场景，而不是动力学。
- **与本项目其他主题的联系**：
  - *world model*：作者自己把 SimOpt 定位为 model-based RL，把物理仿真器当作 "strong prior of the world model"（§I）。$\xi$ 是一个低维、可解释的 world model 参数，SimOpt 用 black-box 方式拟合轨迹 discrepancy。[Dreamer](../../01_world_model/dreamer/notes.md)、[DayDreamer](../../05_intersections/daydreamer/notes.md) 则在真实数据上拟合整个网络。[SimDist](../../05_intersections/simdist/notes.md)（先在仿真里预训练 world model，再快速适应真实世界）和 [Robotic World Model](../../05_intersections/robotic-world-model/notes.md) 可以看作把"先验来自仿真、用少量真实数据校准"这一思路从参数化仿真器换成了神经网络。
  - *TTT*：SimOpt 的适应发生在**两次部署之间、仿真器这一侧**，policy 在部署时不变。[PAD](../../05_intersections/pad/notes.md)（部署时用自监督 loss 更新 policy encoder）和 [RMA](../rma/notes.md)（部署时从历史隐式估计环境参数）则在 policy 这一侧做在线适应。三者有一个共同点：真实世界都**不提供 reward**，只提供无标签的观测，靠自监督式的失配信号驱动更新。
  - *RL*：内层是 [PPO](../../04_rl/ppo/notes.md)；外层的 REPS 相当于在仿真参数空间上做 episodic policy search，其中的 KL trust region 与 TRPO/PPO 同源。[HIL-SERL](../../04_rl/hil-serl/notes.md) 走的是直接在真机上做 RL 的路线，正好和 SimOpt 形成对照。

## 前置知识

1. MDP、policy gradient、[PPO](../../04_rl/ppo/notes.md)（包括并行采样和 warm start）
2. [domain randomization](../domain-randomization/notes.md)，以及它和 system identification 的区别
3. REPS / KL-constrained episodic policy search：对偶函数、温度 $\eta$、加权最大似然；和 CEM、CMA-ES 类比会更容易理解
4. 多元 Gaussian、KL divergence、full covariance 的含义
5. 轨迹 discrepancy 与时间对齐（平滑、错位问题）；状态估计与物体跟踪（DART）作为背景了解即可

## 复现可行性

- **代码**：没有官方代码。仿真用的是 NVIDIA Flex 和 NVIDIA 内部的多 GPU RL 基础设施（Liang 2018），论文没有提供可用发布，复现时需要换成现代 GPU 仿真器（Isaac Lab、MuJoCo MJX、ManiSkill 等）。真机部分需要 Yumi 或 Panda、深度相机，以及带 3D 模型的 DART 跟踪。
- **算力**（§IV-D）：64 块 GPU，每块 150 个仿真 agent（64×150=9600，正好等于每次 REPS 更新的样本数），GPU 型号没有报告。RL 部分每轮 SimOpt 约 7 分钟（peg）或 22 分钟（drawer），REPS 部分耗时没有报告。PPO 超参数见 Table III/IV：clip 0.2，$\gamma=0.99$，$\lambda=0.95$，步长 5e-4，agent 数 100（peg）/400（drawer），episode 长度 150。
- **最小实验建议（sim2sim，单卡）**：在 Isaac Lab 或 MJX 里搭一个 Franka 开抽屉任务，把柜子 X 偏移 15 cm（或者改关节 damping、action scaling），用这个场景当"real"。$p_\phi$ 取 2–5 维 Gaussian。每轮做 PPO（warm start），取 3 条"real"轨迹，采样 512–1024 个 $\xi$（论文用 9600，这里先缩小），REPS 用 $\epsilon=1.0$，$\eta_{\min}=10^{-3}$，$D$ 按 Eq. 4 加 Gaussian 平滑。对比四组：(a) 固定初始 DR；(b) 宽 DR（std 10 cm）；(c) 用 open-loop 重放代替 closed-loop（对应 Appendix A）；(d) 不同 $\epsilon$，同时画出方差随迭代的变化。如果想先验证 REPS 更新逻辑，可以用 pendulum 以杆长为 $\xi$ 做玩具实验。
- **预计难点**：$W$ 和平滑参数要调；参数之间的补偿会导致均值漂移但行为不变；方差塌缩，REPS 权重可能退化到单样本；极端 $\xi$ 会让仿真崩溃，需要对采样做截断；如果要做真机，还需要把真实初始状态同步到仿真里。

## 自测问题

1. Eq. 2 为什么不能直接优化？Eq. 3 做了哪两处近似？KL 约束保护的具体是什么？如果去掉它，会出现什么失败模式？
2. 把目标设成"最小化期望 discrepancy"时，理论最优的 $p_\phi$ 是什么？这与"domain randomization"的初衷有什么矛盾？SimOpt 的哪些设计在事实上保留了随机化宽度？
3. 为什么在仿真里用 closed-loop policy rollout，而不是重放真实动作？当 cabinet X 位置不在 cost 里时，它的参数为什么仍然能被更新？
4. 写出 Gaussian 情形下 REPS 的一次更新：给定 $\{\xi_j,c_j\}$，怎样求出 $\eta$、$\mu'$ 和 $\Sigma'$？"minimum temperature" 起什么作用？
5. 比较 SimOpt、[RMA](../rma/notes.md) 和 [PAD](../../05_intersections/pad/notes.md)：适应发生在哪一侧（仿真器还是 policy）、在什么时间（部署之间还是部署时）、各自需要真实世界提供什么信号？
