# ASAP: Aligning Simulation and Real-World Physics for Learning Agile Humanoid Whole-Body Skills

> arXiv [2502.01143](https://arxiv.org/abs/2502.01143) · RSS 2025（已在 RSS XXI 论文集 p066 核实；arXiv v3 正文未标注） · 主题：03_real2sim2real · 层级：frontier
> 代码：https://github.com/LeCAR-Lab/ASAP · 笔记依据：`source/*.tex`（数字/公式）+ `paper.pdf`（图）

## 一句话

先在 IsaacGym 里用 PPO 训练人形机器人的 motion tracking policy，拿去真机跑出轨迹，再用 RL 学一个 **delta (residual) action model** $\pi^\Delta(s,a)$：把真实记录的动作加上修正量后在仿真里回放，使仿真状态逼近真实状态。之后把冻结的 $\pi^\Delta$ 接进仿真器，在"对齐后的仿真"里微调 policy，部署时去掉 $\pi^\Delta$。关键想法是**在动作空间学 sim-to-real 残差**，而不是辨识物理参数或学 residual state 动力学。

## 问题设定

- **任务**：Unitree G1（23 DoF，手腕固定）跟踪从人类视频重定向来的高动态全身动作，例如 C 罗 180° 转身跳、1.5 m 前跳、单腿平衡（Fig. 1）。
- **输入/输出**：actor 的观测是 5 步历史本体感知加 phase 变量 $\phi\in[0,1]$；输出 $a_t\in\mathbb{R}^{23}$ 为关节目标位置，交给 PD controller 执行（`sections/deepmimic.tex`）。
- **假设**：动力学不匹配是 sim-to-real 的主要瓶颈。real-to-sim 对齐阶段可以用 MoCap 拿到机体 base 的位置、速度和姿态；每个动作都能在真机上用 pretrained policy 安全地跑若干次。
- **之前的方法卡在哪**（`sections/intro.tex`）：
  - SysID 需要预先定义参数空间，真实动力学落在参数化之外就补不了，而且常依赖真实力矩测量，很多硬件拿不到。
  - DR 靠随机化换鲁棒，得到的 policy 过于保守，牺牲 agility。
  - 用真实数据学 dynamics model 的做法在无人机、地面车这类低维系统上成功过，在人形机器人上没有验证。

## 核心方法

整体分两阶段四步（Fig. 2）：(a) 预训练并采集真实轨迹 →(b) 训练 delta action model →(c) 冻结 $\pi^\Delta$ 后微调 policy →(d) 不带 $\pi^\Delta$ 部署。

**1. 数据：视频 → 机器人动作**（`sections/deepmimic.tex`，Fig. 3）
- 拍摄人类视频，用 TRAM 重建 SMPL 动作。
- 做 "sim-to-data" 清洗：用 MaskedMimic 在 IsaacGym 里跟踪这些动作，跟得上的才保留，以此去掉物理上不可行的片段。
- 两阶段重定向：先选 12 个人与机器人对应的 link，对 SMPL shape $\beta'$ 做梯度下降，使 rest pose 贴近 G1；再固定 $\beta'$，优化 pose，使 link 位置对齐。

**2. 预训练：phase-based motion tracking（DeepMimic 风格）**
- 这是 goal-conditioned RL，goal 只有 phase $\phi$，对单动作跟踪已经足够。本体感知定义为（`sections/deepmimic.tex` + `macros.tex`）：
$$s_t^{p}\triangleq[q_{t-4:t},\ \dot q_{t-4:t},\ \omega^{root}_{t-4:t},\ g_{t-4:t},\ a_{t-5:t-1}],\quad q,\dot q,a\in\mathbb{R}^{23},\ \omega,g\in\mathbb{R}^3$$
- 奖励 $r_t=\mathcal{R}(s^p_t,s^g_t)$，用 PPO 最大化 $\mathbb{E}[\sum_{t=1}^{T}\gamma^{t-1}r_t]$。
- 训练稳定的四个要点：
  - **Asymmetric actor-critic**：critic 看得到参考动作的全局位置、root 线速度等 privileged information；actor 只用本体感知和 $\phi$，所以真机上不需要 odometry。
  - **Termination curriculum**：偏离参考超过阈值就终止，阈值从 1.5 m 逐步收紧到 0.3 m。
  - **RSI**：随机采样 $\phi$，从参考动作对应时刻的状态初始化。
  - **奖励与 DR**：奖励分 penalty、regularization、task 三类（Table I）。DR 只用了很基础的一组：摩擦 $\mathcal U(0.2,1.1)$，P gain $\mathcal U(0.925,1.05)\times$，控制延迟 $\mathcal U(20,40)$ ms，每 10 s 推一次、$v_{xy}=0.5$ m/s（Table VI）。

**3. 训练 delta action model**（`sections/deltaA.tex`，Fig. 2b）
- 真实数据为 $\mathcal D^r=\{s^r_0,a^r_0,\dots,s^r_T,a^r_T\}$，其中 $s_t=[p^{base}_t,v^{base}_t,\alpha^{base}_t,\omega^{base}_t,q_t,\dot q_t]$（$\alpha^{base}_t\in\mathbb{R}^4$ 为四元数，$q_t,\dot q_t\in\mathbb{R}^{23}$），由 MoCap 和机载传感器共同记录（原文未逐项说明哪个量来自哪个传感器）。
- 把 $\pi^\Delta$ 当作一个 RL agent，它的"环境"是被修改过的仿真器：
$$\Delta a_t=\pi^\Delta_\theta(s_t,a_t),\qquad s_{t+1}=f^{sim}(s_t,\ a^r_t+\Delta a_t)$$
- 仿真机器人初始化到真实状态 $s^r_t$（原文写 "During each RL step"），然后回放真实动作加修正量。奖励衡量 $s_{t+1}$ 与 $s^r_{t+1}$ 的差距，项目和跟踪奖励同构（body position、VR 3-point、DoF position 等，Table II），另加动作幅值正则 $\exp(-\|\Delta a_t\|)-1$（原文写作 $a_t$）。优化器用 PPO。
- 训练 horizon 在 Fig. 10b 里做了消融（0.25–1.5 s），可见这是多步 rollout，而不是逐步重置。
- **为什么用 RL**：论文没有专门论证为什么用 RL 而不是监督学习来训练 $\pi^\Delta$。附录 D（`sections/appendix.tex`，正文引作 Section VIII-D）里有一条论证："要做多步匹配就得穿过 $f^{sim}$ 求导，这一般不可行；RL 等于做了一次 gradient-free 的多步匹配。"但它针对的是**使用** $\pi^\Delta$ 微调 policy 这一侧，比较的是 RL fine-tuning 与 fixed-point / 梯度法。把同一理由套到 $\pi^\Delta$ 的训练上，是笔者的推断。
- 直觉例子：仿真高估了电机力量，真机跳不起来。这时 $\pi^\Delta$ 会学着削弱下肢动作，让仿真复现真机的失败。

**4. 在对齐后的仿真中微调**（`sections/deltaA.tex`，无编号公式）
$$s_{t+1}=f^{\text{ASAP}}(s_t,a_t)=f^{sim}\big(s_t,\ a_t+\pi^\Delta(s_t,a_t)\big)$$
- 冻结 $\pi^\Delta$，从 pretrained policy 出发，用和 Table I 相同的奖励做 RL 微调（论文没有单独说明微调用的算法；预训练用的是 PPO）。部署时去掉 $\pi^\Delta$，policy 直接输出给真机。

**对照：两种免 RL 的用法**（附录 D，Eq. (1)–(4)；§V-B）
- 记 $\hat\pi(s)$ 为在仿真中表现良好的 nominal policy。假设单步一致 $f^{real}(s,\pi(s))=f^{sim}(s,\pi(s)+\pi^\Delta(s,\pi(s)))$，可推出 $\pi(s)=\hat\pi(s)-\pi^\Delta(s,\pi(s))$（Eq. 2）。
- 求解方式一是 fixed-point iteration：$y_{k+1}=\hat\pi(s)-\pi^\Delta(s,y_k)$（Eq. 3）。
- 方式二是对 $l(y)=\|y+\pi^\Delta(s,y)-\hat\pi(s)\|^2$ 做梯度下降（Eq. 4）。
- 两者都是短视的单步匹配。

**Baseline：DeltaDynamics**（`sections/appendix.tex` 附录 C，正文引作 Section VIII-C）
- 学习 residual state：$s^{real}_{t+1}-s^{sim}_{t+1}=f^\Delta_\theta(s^{real}_t,a^{real}_t)$。
- 用 K 步自回归 MSE 训练，K 按 schedule 逐渐增大。
- 训练好后，把预测出的 delta 直接加到仿真状态上。

## 关键实验与证据

设置：在 IsaacGym 训练。sim-to-sim 实验把 IsaacSim 和 Genesis 当作"真实世界"，共 43 个动作，分 easy/medium/hard 三档。另做了 IsaacGym → 真机 G1。指标包括：Succ（任一时刻平均 body 距离 > 0.5 m 记为失败）、$E_{g\text{-}mpjpe}$ / $E_{mpjpe}$（mm）、$E_{acc}$（mm/frame²）、$E_{vel}$（mm/frame）。

- **Open-loop 回放（Table III，Fig. 5）**：把测试环境的动作放回训练环境重放。
  - 1.0 s 时 IsaacSim 上的 $E_{g\text{-}mpjpe}$：OpenLoop 80.8，SysID 77.6，DeltaDynamics 68.1，ASAP 37.9。Genesis 上分别为 82.5 / 76.5 / 50.2 / 36.9。
  - 0.25 s 时各方法基本持平（19–24），说明 ASAP 的优势在长 horizon。
  - DeltaDynamics 的 $E_{mpjpe}$ 反而一直最低（1.0 s 时 IsaacSim 21.5，ASAP 22.9），但它的 $E_{g\text{-}mpjpe}$ 在 0.25 s 和 0.5 s（IsaacSim）都是最高的。Fig. 5 只画了单个踢球动作的 MPJPE 曲线，其中 DeltaDynamics 的误差随时间放大，论文把这归因于 overfitting。
- **Closed-loop 微调（Table IV）**：
  - Hard 档 Genesis：Vanilla 的 Succ/$E_{g\text{-}mpjpe}$ 为 82.9% / 175，SysID 100% / 186，DeltaDynamics 60.0% / 190，ASAP 100% / 129。
  - Hard 档 IsaacSim：$E_{g\text{-}mpjpe}$ 分别为 148 / 165 / 137 / 129。
  - ASAP 在所有设置上 Succ 都是 100%。DeltaDynamics 在 Easy-Genesis 和全部 medium/hard 设置上掉到 83.3%–60.0%（Hard 档为 IsaacSim 66.7%、Genesis 60.0%）。
  - ASAP 并非全面最优：Genesis-Easy 的 $E_{mpjpe}$ 为 Vanilla 70.1、ASAP 73.5；IsaacSim-Hard 为 Vanilla 51.6、ASAP 56.5。
  - IsaacSim 本身的 gap 很小：Easy 档 Oracle 97.5，Vanilla 107，ASAP 106。
- **真机（Table V）**：只和 Vanilla 比。
  - Kick（in-distribution）：$E_{g\text{-}mpjpe}$ 61.2→50.2，$E_{mpjpe}$ 43.5→40.1。
  - LeBron "Silencer"（OOD）：$E_{g\text{-}mpjpe}$ 159→112，$E_{mpjpe}$ 55.3→47.5。
  - 真机上只训练了 **4-DoF 踝关节** delta action model。理由有两个：全 23 DoF 需要 >400 条 clip，而采集时有两台 G1 损坏；G1 踝关节是连杆结构，gap 大。实际用了 100 条 clip，5 个任务各执行 30 次，另有 10 分钟 locomotion 数据（§IV-C）。
- **消融（Fig. 10，sim-to-sim；数据量消融写明用 IsaacSim 数据、在 IsaacGym 训练 $\pi^\Delta$）**：
  - 数据量 43/430/4300/43000：closed-loop MPJPE 为 534.02 / 104.95 / 97.51 / 98.15，430 之后基本饱和。
  - 训练 horizon：正文称 open-loop 在 1.5 s 最好。但热图里 0.5 s 评测点上，1.0 s 训练 horizon 的 25.30 低于 1.5 s 的 29.98；在 0.25 / 1.0 / 1.5 s 评测点上，1.5 s 训练 horizon 最低。closed-loop 在 1.0 s 最好（97.51；不微调为 145.00）。
  - action norm 权重 0 / 0.001 / 0.01 / 0.1 / 1.0：closed-loop 为 180.97 / 169.19 / 125.90 / 97.51 / 176.52。
- **用法对比（Fig. 11）**：RL 微调误差最低。fixed-point 和梯度法在前段就比不微调还差。
- **和随机动作噪声比（Fig. 12）**：用 $s_{t+1}=f^{sim}(s_t,a_t+\beta\delta_a)$，$\delta_a\sim\mathcal U[0,1]$ 微调，IsaacGym→Genesis。不微调 336.1，最佳噪声约 173.5，ASAP 126.9。
- **$\pi^\Delta$ 输出幅值（Fig. 13）**：下肢大于上肢，踝 pitch 最大（0.056 / 0.054），并且左右不对称。这说明 gap 有结构，均匀噪声覆盖不了。

## 局限与论文没说的

**作者承认的**（`sections/limitation.tex`）：
- 高动态动作导致电机过热，两台 G1 损坏，限制了真实数据的规模。
- 依赖 MoCap。
- 全 23 DoF 的 $\pi^\Delta$ 需要 >400 条 clip，数据需求大。

**笔者读出来的**：
- **真机证据很薄**：只有 2 个动作，只和 Vanilla（带 Table VI 的基础 DR）比，没有 SysID、DeltaDynamics 或更强 DR 的真机对照。Table V 没有标准差，也没写评测次数。结论里"sim-to-real 最多降 52.7%"在 Table V 里找不到对应（按表算，最大降幅是 LeBron $E_{g\text{-}mpjpe}$ 的约 30%）。
- **文本与图不一致**：
  - 随机噪声最佳 MPJPE，正文写 150，Fig. 12 图注和图写 173。
  - 数据量从 4300 到 43000，正文说误差"下降 0.65%"，图中是 97.51→98.15，实际上升了约 0.66%。
  - 训练 horizon：正文说 1.5 s 在 0.25 / 0.5 / 1.0 s 评测点上都最低，但 Fig. 10b 热图在 0.5 s 评测点上是 1.0 s 训练 horizon 最低（25.30 对 29.98）。
  - Table II 的 action norm 权重是 −0.2，消融最优却是 0.1。可能是 sim2sim 与真机设置不同，但论文没说明。
- **Baseline 偏弱**：SysID 只在每个参数 3 个离散值中做搜索（Table VII；原文只说 "search ... among certain discrete ranges"，没说是否是完整网格）。辨识的是 base CoM 偏移、base link 质量比和各关节 PD 增益比，没有辨识执行器模型或延迟。论文说 DR 会导致"过于保守"，但没有和强 DR 或 RMA 类在线适应做 closed-loop 对比。
- **$\pi^\Delta$ 是单模型**：论文没有讨论不确定性估计，也没有用 ensemble。policy 可能 exploit 它在数据覆盖外的错误修正。真机上的 OOD 泛化只在一个动作（LeBron）上验证过。
- **缺少迭代**：没有做多轮循环（用微调后的 policy 重新采数据、再训 $\pi^\Delta$），数据分布始终来自 pretrained policy。
- **动作空间修正能表达的 gap 有限**：动作空间修正无法表达与动作无关的外部扰动、接触/地面差异、感知噪声。修正加在 PD 目标上，本质上是 state-and-action-conditioned 的执行器补偿。
- **算力没报**：训练时长、GPU 型号、PPO 超参都没写。

## 在脉络中的位置

- **前驱**：
  - DeepMimic（phase tracking、RSI）与 [PPO](../../04_rl/ppo/notes.md)。
  - legged 系 sim-to-real 训练配方 [legged_gym](../legged-gym/notes.md)。
  - 参数辨识和分布调整 [SimOpt](../simopt/notes.md)，以及基础 DR [domain randomization](../domain-randomization/notes.md)。这两类是 ASAP 要超越的对象。
  - residual action 思路直接继承 RGAT（Karnan 2020，论文 §VI-C 引用），属于 Grounded Action Transformation 这条线。[Reality Gap 综述笔记](../survey-reality-gap/notes.md) 也这样归类，但这是笔记作者按综述的分类做的，综述正文并没有引用 ASAP。
  - 人形机器人数据管线沿用 H2O（he2024learning）的 shape-and-motion 两阶段重定向，以及 TRAM、MaskedMimic。OmniH2O 只在 odometry 问题上被引用。
- **后继 / 影响**（以下两条都是本项目笔记作者的归类，两篇综述正文都没有讨论 ASAP）：
  - [Reality Gap 综述笔记](../survey-reality-gap/notes.md) 按综述的 residual 分类，把它归入"改输入"的 residual 方法。Reality Gap 综述的 `references.bib` 有 ASAP 条目，但正文没有引用。
  - [Test-time intelligence 综述笔记](../../02_test_time_training/survey-test-time-intelligence/notes.md) 按该综述的判据，把它归为 training-time amortization（非 test-time）。
  - 与 [SimDist](../../05_intersections/simdist/notes.md)、[Robotic World Model](../../05_intersections/robotic-world-model/notes.md) 同属"用真实数据学模型修正仿真/替代仿真"的方向。
- **与本项目其他主题的联系**：
  - **World model**：DeltaDynamics 基线本质上是"物理仿真 + learned residual state model"的混合 world model。Fig. 5 展示了它 autoregressive 误差累积，这正是 [DreamerV3](../../01_world_model/dreamerv3/notes.md)、[RWM](../../05_intersections/robotic-world-model/notes.md) 这类 world model 要对付的 compounding error。ASAP 的选择是保留物理仿真器，只在动作空间残差化，用物理先验约束状态的合理性。
  - **TTT / TTA**：ASAP 是**离线**对齐，部署时 policy 固定，没有 inner loop。对照 [RMA](../rma/notes.md)（部署时从历史隐式辨识 extrinsics）和 [PAD](../../05_intersections/pad/notes.md)（部署期自监督更新）。作者在 future work 中提到要 "exploring adaptation techniques for delta action models to achieve sample-efficient few-shot alignment"，但没有说这种适应发生在部署时。把它做成部署时更新 $\pi^\Delta$ 的模块，是笔者看到的一个自然交叉点。
  - **RL**：RL 在这里用了两次。一次是 PPO 训 policy；另一次是把 real-to-sim 的动力学对齐表述成 RL 问题（奖励 = 仿真与真实状态的匹配），绕开对仿真器求导。论文把这定位为 SysID 的替代：不显式估计系统参数，而是学修正动作。这与 [HIL-SERL](../../04_rl/hil-serl/notes.md) 这类直接在真机上做 RL 的路线形成对照。

## 前置知识

1. PPO 与 actor-critic，asymmetric actor-critic / privileged information（[PPO](../../04_rl/ppo/notes.md)）。
2. DeepMimic 式 motion tracking：phase 变量、RSI、early termination。
3. 关节位置目标 + PD controller 的动作参数化。
4. sim-to-real gap、system identification 与 domain randomization 的区别（[DR](../domain-randomization/notes.md)、[SimOpt](../simopt/notes.md)）。
5. Residual learning：residual policy、grounded action transformation。
6. SMPL 人体模型与 motion retargeting；MPJPE 类指标。
7. POMDP 与 open-loop / closed-loop 评测的区别。

## 复现可行性

- **代码**：已开源，MIT，基于 HumanoidVerse。支持 IsaacGym Preview 4、IsaacSim 4.2.0、Genesis 0.2.1，Linux 平台，IsaacGym 需要 Python 3.8。README 里有三个阶段的命令（motion tracking、delta action 训练、微调），也附带了重定向好的 G1 动作。以上来自对 GitHub README 的 WebFetch，不是论文内容。README 示例：motion tracking 和微调用 `num_envs=4096`，delta action 训练用 `num_envs=5000`。
- **算力**：论文未报告 GPU 型号、数量和训练时长。
- **数据/仿真**：sim-to-sim 完全可复现，不需要真机。真机部分需要 G1、MoCap 和安全防护，门槛极高，而且作者报告有硬件损坏。
- **最小实验（建议）**：
  1. 选 Genesis 作为"真实"环境，挑 1–3 个 medium/hard 动作，在 IsaacGym 预训练 tracking policy。
  2. 在 Genesis 中 rollout 约 430 条轨迹。Fig. 10a（用的是 IsaacSim 数据，不是 Genesis）显示 430 条时 closed-loop MPJPE 已为 104.95，4300 条为 97.51。
  3. 用 horizon 1.0 s、action norm 权重 0.1 训练 $\pi^\Delta$。
  4. 冻结 $\pi^\Delta$ 后微调 policy，回到 Genesis 评测。
  5. 同时复现 Table III 的 0.25 / 0.5 / 1.0 s open-loop 回放，以及 Fig. 12 的随机噪声基线（$\beta=0.1$）。
  6. 验收标准：closed-loop MPJPE 明显低于 Vanilla 和噪声微调。
- **预计难点**：
  - $\pi^\Delta$ 的奖励权重和 horizon 敏感（Fig. 10b/c）。
  - 论文没有给出 task reward 的核函数形式与 PPO 超参，只能依赖代码。
  - 三个仿真器的机器人模型要一致。
  - 正文与图表有若干数字不一致，复现时应以代码 config 为准。

## 自测问题

1. delta action model 的"环境""动作""奖励"分别是什么？为什么不能直接对 $\|f^{sim}(s,a^r+\Delta a)-s^r_{t+1}\|$ 做监督 / 梯度下降，而要用 PPO？
2. 从单步一致假设推出 $\pi(s)=\hat\pi(s)-\pi^\Delta(s,\pi(s))$。fixed-point 和梯度法为什么在 Fig. 11 中表现比不微调还差？
3. DeltaDynamics（residual state）与 ASAP（residual action）在 open-loop 1.0 s 时，$E_{mpjpe}$ 和 $E_{g\text{-}mpjpe}$ 的排序为什么相反？这对"把 learned model 嵌进仿真器"有什么启示？
4. 为什么真机上只学 4-DoF 踝关节的 $\pi^\Delta$？论文给出的理由是数据不足和踝关节连杆结构。Fig. 13 的幅值分布来自 IsaacGym→IsaacSim，不是真机数据，它能在多大程度上支持这个选择？
5. 如果把 $\pi^\Delta$ 换成随机噪声 $\beta\delta_a$，policy 学到的是什么？为什么 Fig. 12 中噪声过大（$\beta=0.4$）反而变差？
