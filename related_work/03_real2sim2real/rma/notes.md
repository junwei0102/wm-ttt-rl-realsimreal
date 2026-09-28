# RMA: Rapid Motor Adaptation for Legged Robots

> arXiv [2107.04034](https://arxiv.org/abs/2107.04034) · RSS 2021 · 主题：03_real2sim2real · 层级：foundational
> 代码：论文正文未给代码链接；项目主页 https://ashish-kmr.github.io/rma-legged-robots/ 链到 https://github.com/antonilo/rl_locomotion（其 README 自述是 "Learning Visual Locomotion with Cross-Modal Supervision" 的代码，"builds on the code from RMA"，所以不是 RMA 原文代码的原样发布；GPL-3.0） · 笔记依据：`source/main.tex`、`source/supp-text.tex`（数字/公式）+ `paper.pdf`（Fig. 2/3/4/S3/S4）

## 一句话

在仿真里先用 PPO 训练一个以 privileged 环境参数的低维编码 $z_t$（extrinsics）为条件的 base policy，再用监督学习训练一个 adaptation module，让它只看最近 0.5 s 的 state-action 历史就能回归出 $z_t$。部署时不做任何梯度更新，也不在真机上 fine-tune，只靠这个在线隐式 system identification 在不到 1 s 内适应新地形和负载。

## 问题设定

- **任务**：四足机器人 Unitree A1（12 个驱动关节，约 12 kg）在未见过的真实地形和扰动下稳定向前走，这些条件包括沙地、泥地、油面、负载变化、台阶。训练完全在仿真中进行，部署时 zero-shot，不做仿真校准，也不在真实世界 fine-tune（`main.tex` Sec. I、III-C）。
- **输入/输出**：state $x_t\in\mathbb{R}^{30}$，由 12 个关节角、12 个关节速度、躯干 roll/pitch 和 4 个二值足端接触组成；action $a_t\in\mathbb{R}^{12}$ 是期望关节角，经 PD 转成力矩（Sec. IV-A）。只用 proprioception，没有视觉或地形感知。
- **可观测性**：环境参数 $e_t\in\mathbb{R}^{17}$ 包含负载质量及其位置（3）、12 个电机强度、摩擦（1）和局部地形高度（1）。仿真里可以读到它，真机上读不到，因此问题是部分可观测的。
- **奖励**：没有参考轨迹，也没有 trajectory generator，只用带 bioenergetics 动机的"自然"奖励（最小化做功和地面冲击），前进速度上限 0.35 m/s。
- **之前的方法卡在哪**：(1) domain randomization 的 policy 对 $e$ 不敏感，会学得过于保守（Sec. II 引 Luo et al.）。(2) 仿真校准路线，如 actuator net [Hwangbo 2019]、Tan et al. 2018 的电机建模，要先在每台真机上采数据。(3) 测试时优化 latent 的方法，如 Peng et al. 2020 用 AWR，需要在真实环境 rollout：脚注给出 50 个 5–10 s 的 episode，共 4–8 min，而机器人还没学会走时采这些数据容易摔坏。(4) 显式 SysID [Yu et al. 2017] 要把所有物理参数估准，这既难也不必要。

## 核心方法

**整体结构**（`main.tex` Fig. 2，Eq. 1–2）：两个阶段训练，部署时两个模块异步运行。

**Phase 1：base policy $\pi$ 与 environmental factor encoder $\mu$ 联合 RL。**
$$z_t=\mu(e_t),\qquad a_t=\pi(x_t,a_{t-1},z_t)\qquad(\texttt{main.tex}\ \text{Eq.1, Eq.2})$$
$$J(\pi)=\mathbb{E}_{\tau\sim p(\tau|\pi)}\Big[\textstyle\sum_{t=0}^{T-1}\gamma^t r_t\Big]\qquad(\text{Sec. III-A})$$
$\mu$ 是 3 层 MLP（隐层 256、128），把 17 维 $e_t$ 压成 8 维 $z_t$。$\pi$ 是 3 层 MLP，隐层 128。PPO 的梯度同时回传到 $\pi$ 和 $\mu$，所以 $z_t$ 不去重建物理参数，只编码"为了选对动作需要知道的环境信息"。作者认为这顺带绕开了可辨识性问题：几个参数如果对观测的影响相同，就不必区分（Sec. I 第二条 speculation）。

- **奖励**（Sec. III-A）共 10 项，系数依次为 20, 21, 0.002, 0.02, 0.001, 0.07, 0.002, 1.5, 2.0, 0.8：
  forward $\min(v_x^t,0.35)$；lateral/rotation $-\|v_y^t\|^2-\|\omega^t_{yaw}\|^2$；work $-|\boldsymbol\tau^\top(\mathbf q^t-\mathbf q^{t-1})|$；ground impact $-\|\mathbf f^t-\mathbf f^{t-1}\|^2$；smoothness $-\|\boldsymbol\tau^t-\boldsymbol\tau^{t-1}\|^2$；action magnitude $-\|\mathbf a^t\|^2$；joint speed $-\|\dot{\mathbf q}^t\|^2$；orientation $-\|\boldsymbol\theta^t_{roll,pitch}\|^2$；z acceleration $-\|v_z^t\|^2$；foot slip $-\|\mathrm{diag}(\mathbf g^t)\mathbf v_f^t\|^2$。
- **Curriculum**（`supp-text.tex` S1；扰动与地形部分见 main Sec. III-A）：第 3–10 项惩罚项再乘一个系数 $k_t$，$k_0=0.03$，按 $k_{t+1}=k_t^{0.997}$ 逐步增大到接近 1（$t$ 是迭代数）。不这样做，policy 会因为惩罚太重学不会走。质量、摩擦、电机强度等扰动的难度也线性递增；地形不设 curriculum。
- **PPO 细节**（supp S1）：15,000 iter，每 iter 80,000 transitions 分 4 个 mini-batch，每个跑 4 轮。loss 是 surrogate 加 0.5×value loss，ratio clip 到 [0.8, 1.2]，GAE $\lambda=0.95$，$\gamma=0.998$，不加 entropy 正则，Gaussian std 下限 0.2，Adam lr 5e-4。

**Phase 2：adaptation module $\phi$，监督回归 extrinsics。**
$$\hat z_t=\phi(x_{t-k:t-1},a_{t-k:t-1}),\quad k=50\ (0.5\,\text{s});\qquad \mathcal L=\|\hat z_t-z_t\|^2,\ z_t=\mu(e_t)\ \text{冻结}\qquad(\text{Sec. III-B})$$
- **为什么能从历史里推出 $z$**：$a$ 是"命令关节去哪"，$x$ 是"关节实际到了哪"。两者的偏差由摩擦、负载、电机强度决定，作者把这类比成 Kalman filter 从观测历史估计状态（Sec. I）。所以 $\phi$ 同时需要 $a$ 和 $x$ 的历史。
- **结构**（Sec. IV-B）：每步 $(x,a)$ 先经 2 层 MLP 嵌成 32 维，再过 3 层 1-D CNN 沿时间卷积，各层 [in, out, kernel, stride] 为 [32,32,8,4]、[32,32,5,1]、[32,32,5,1]，最后 flatten 并线性投影到 $\hat z_t\in\mathbb{R}^8$。（假设无 padding，我推算时间长度变化为 50→11→7→3，flatten 后 96 维；论文没写这一步。）
- **on-policy 数据**（DAgger 式，引 Ross et al. 2011）：如果用 $\pi$ 加真值 $z_t$ 滚出的轨迹训练 $\phi$，数据里只有"走得好"的状态，部署时一旦偏离，$\phi$ 就没见过那些状态。所以每次都用 **$\phi$ 自己预测的 $\hat z_t$** 驱动 $\pi$ 滚出轨迹，再用 $\mu(e_t)$ 作标签回归，反复迭代到收敛（supp Algorithm 1 Phase 2）。随机初始化的 $\phi$ 和不准的 $\hat z$ 恰好提供了探索。训练 1000 iter，batch 80,000，Adam lr 5e-4。

**部署：异步双频率**（Sec. III-C，supp Algorithm 2）。$\phi$ 以 10 Hz 更新 $\hat z_{\text{async}}$，$\pi$ 以 100 Hz 读最新的 $\hat z$ 输出期望关节角。两者之间没有中心时钟。PD 控制为 $\boldsymbol\tau=K_p(\hat{\mathbf q}-\mathbf q)+K_d(\hat{\dot{\mathbf q}}-\dot{\mathbf q})$，$\hat{\dot{\mathbf q}}=0$，真机上 $K_p=55$、$K_d=0.8$（Sec. IV-A）。这种解耦成立的前提是 extrinsics 变化慢、机器人状态变化快。作者也试过另一种做法，即单个 policy 直接吃完整历史，结果是步态不自然、仿真性能差、板载只能跑 10 Hz，也做不了异步（Sec. III-C，没有给数字）。

**要点**：部署时**不更新任何权重**。所谓 "adaptation" 就是 $\phi$ 的一次前向推理，相当于把 system identification 这个优化问题摊销成一个网络（Sec. I 第一条 speculation）。

## 关键实验与证据

**仿真设置**：RaiSim，fractal terrain 参数为 octaves 2、lacunarity 2.0、gain 0.25、z-scale 0.27。每个 episode 最长 1000 步，身高低于 0.28 m、roll 超过 0.4 rad 或 pitch 超过 0.2 rad 时提前终止（Sec. IV-A）。训练/测试参数范围见 **Table I**：

| 参数 | Friction | $K_p$ | $K_d$ | Payload (kg) | CoM (cm) | Motor strength | 每步 resample 概率 |
|---|---|---|---|---|---|---|---|
| 训练 | [0.05, 4.5] | [50, 60] | [0.4, 0.8] | [0, 6] | [-0.15, 0.15] | [0.90, 1.10] | 0.004 |
| 测试 | [0.04, 6.0] | [45, 65] | [0.3, 0.9] | [0, 7] | [-0.18, 0.18] | [0.88, 1.22] | 0.01 |

**仿真主结果 Table II**（`main.tex` `tab:sim-metrics`，3 个随机种子 × 每种子 1000 episodes）：

| 方法 | Success % | TTF | Reward | Distance m | Samples | Torque | Smoothness | Ground Impact |
|---|---|---|---|---|---|---|---|---|
| Robust (DR) | 62.4 | 0.80 | 4.62 | 1.13 | 0 | 527.59 | 122.50 | 4.20 |
| SysID（预测 $\hat e_t$） | 56.5 | 0.74 | 4.82 | 1.17 | 0 | 565.85 | 149.75 | 4.03 |
| AWR (Peng 2020) | 41.7 | 0.65 | 4.17 | 0.95 | 40k | 599.71 | 162.60 | 4.02 |
| RMA w/o Adapt | 52.1 | 0.75 | 4.72 | 1.15 | 0 | 524.18 | 106.25 | 4.55 |
| **RMA** | **73.5** | 0.85 | 5.22 | 1.34 | 0 | 500.00 | 92.85 | 4.27 |
| Expert（真值 $z_t$，上界） | 76.2 | 0.86 | 5.23 | 1.35 | 0 | 485.07 | 85.56 | 3.90 |

这张表的读法：
- RMA 与 Expert 只差 2.7 个点的成功率，说明从 0.5 s 历史估计 $z$ 几乎没有损失。
- **SysID 56.5 < Robust 62.4**。显式回归 17 维物理参数反而不如完全不做估计，这支持"预测 $z$ 而非 $e$"的设计。
- AWR 在参数不停 resample 的测试设置下适应太慢，只有 41.7。
- w/o Adapt 为 52.1，说明 base policy 离不开正确的 $z$。
- Ground Impact 一项 RMA 为 4.27，比 SysID、AWR、Robust 都差，所以并非每个指标都领先。

**真实世界室内**（Fig. 3，每种方法 5 次，明显失败的只跑 2 次）：

| 任务 | RMA | RMA w/o Adapt | A1 controller |
|---|---|---|---|
| Uneven Foam（success / TTF / dist） | 80 / 0.9 / 0.9 | 0 / 0.1 / 0.1 | 20 / 0.36 / 0.36 |
| Upward Incline | 100 / 1 / 1 | 20 / 1 / 0.6 | 100 / 1 / 1 |
| Mattress | 100 / 1 / 1 | 0 / 1 / 0.25 | 100 / 1 / 1 |
| Step Down-15（success） | 100 | 0 | 60 |
| Step Up-6 | 100 | 40 | 80 |
| Step Up-8 | 60 | 0 | 20 |

负载实验（Fig. 3 底部曲线、Sec. V-A）：caption 称 A1 controller 在 8 kg 时开始退化（图中 5→8 kg 之间成功率已从 100 下降，12 kg 时为 0；正文说它能超过官方标称的 5 kg，但会下沉并最终摔倒）；RMA w/o Adapt 在 8 kg 以上"基本不摔但也不前进"；RMA 最多能背 12 kg（等于 A1 自重），成功率较高。油面加塑料包脚的实验（Fig. 4）成功率 90%。

**户外**（Fig. 1、Sec. V-B）：沙地、泥地、土堆、高草、灌木 0 失败，徒步道下楼梯 70%，水泥堆和卵石堆 80%，下坡泥堆 100%。户外没有 baseline 对比，也没有报告试验次数。

**机制证据**：Fig. 4 画了油面实验中经 median filter 的 $\hat z$ 第 1、5 维，Fig. S3 画了往背上扔 5 kg 沙袋时的第 2、7 维。打滑或负载落下后 $\hat z$ 明显跳变（图中红色 "Adaptation" 区间，读图估计约 1–1.5 s；论文未给精确时长），步态周期随后恢复，力矩稳定在更高的水平，$\hat z$ 停在新值，说明它一直"记得"表面滑或者多了负载。

**泛化曲线**（Fig. S4，3 个种子 × 100 trials）：分别把 payload 扫到 15 kg、terrain height variation 扫到 0.25–0.40 m、friction 降到 0.0125。三组里 RMA 基本都是除 Expert 外最好的（个别点例外，例如 friction=0.8 时 DR 的 success 高于 RMA），但所有方法都随扰动变大而明显下降。

**消融覆盖**：真正做了的只有 "w/o Adapt"，以及 "SysID（预测 $e$ 而非 $z$）"这个对照。on-policy 与 expert 数据训练 $\phi$ 的比较、异步与单网络的比较，文中只有定性描述，没有数字。

## 局限与论文没说的

**作者承认**（Conclusion）：这是 blind 策略，只用 proprioception。下楼梯时突然踩空、岩石同时绊住多条腿这类大扰动仍会失败，作者认为下一步需要视觉等 exteroception。

**我读出来的**：
1. **任务很窄**：只练了"向前走、速度不超过 0.35 m/s"。没有速度或朝向指令跟踪，没有转向，也没有高速或跳跃等 agile 动作。
2. **真机证据薄**：每个任务 5 次，明显失败的只跑 2 次，户外不报 trial 数。真机 baseline 只有 A1 自带 MPC 和 RMA w/o Adapt，**没有真机上的 Robust (DR) baseline**，所以真机结果区分不出"适应"和"一个更好的 DR policy"各自的贡献。RMA w/o Adapt 时 $z$ 喂的是什么（零向量？均值？）文中没说。base policy 训练时一直有 $z$，突然拿掉 $z$ 等于喂了分布外输入，这个 ablation 可能偏弱。
3. **"OOD" 程度温和**：Table I 的测试范围只比训练范围略宽，比如 payload 从 6 kg 到 7 kg。适应能力本质上被训练时的随机化支撑集限制。$\phi$ 是固定网络，超出支撑集后没有机制继续学习（Fig. S4 中极低摩擦下成功率也在下降）。
4. **文本不一致**：Fig. 3 caption 说 step down 15 cm 成功率 80%、uneven foam 100%，而图内表格是 Step Down-15 为 100、Uneven Foam 为 80，像是两者对调了。Sec. IV-A 写"控制 100 Hz、仿真步长 0.025 s"，0.025 s 大于 10 ms 的控制周期，数字自相矛盾。curriculum 的失败模式在正文是"原地不动"，在 supp 是"learns to fall"。
5. **$z$ 的可解释性是挑选出来的**：Fig. 4 和 Fig. S3 各挑了 2 维展示，没有系统分析 $z$ 与各物理量的对应关系。
6. **$e_t$ 中的局部地形高度**：它被离散化并取四足最大值，但 $\phi$ 必须从 proprioception 推出它，这一维估得多准没有单独评测。$K_p/K_d$ 被随机化了却不在 $e_t$ 里，对这两个参数只是纯 DR。
7. **算力口径**：只写了"普通台式机 1 GPU"。RaiSim 是 CPU 仿真，CPU 核数没报告；训练曲线（Fig. S2）带阴影带，但没说明用了几个种子。

## 在脉络中的位置

- **前驱**：
  - [Domain Randomization](../domain-randomization/notes.md)：Robust baseline 就是这条路线，RMA 在它之上加了"可条件化的 $z$"。
  - Yu et al. 2017 UP-OSI：universal policy 加在线 SysID。RMA 把"估 $e$"改成"估 $z$"。
  - Peng et al. 2020：latent 加 AWR 在真机上优化，RMA 的 AWR baseline 即来自这里。
  - Lee et al. 2020：用 privileged teacher 做 legged locomotion，但依赖 trajectory generator 和 actuator net。
  - Hwangbo et al. 2019：惩罚项 curriculum。
  - DAgger（Ross et al. 2011）：Phase 2 的 on-policy 数据。
  - [PPO](../../04_rl/ppo/notes.md)：Phase 1 的训练算法。
- **后继 / 影响**：
  - "privileged encoder → 从历史回归 latent"这一范式被大量沿用。[ASAP](../asap/notes.md) 在 related work 中把 RMA 与 Kumar 2022 双足 RMA、Fu 2023 Deep Whole-Body Control、Qi 2023 In-hand Rotation via RMA 一起列为 regression-based 的在线适应方法。
  - [Reality Gap 综述](../survey-reality-gap/notes.md) 在 meta learning / adaptive control 和 privileged information 两处都以 RMA 为代表例子。
  - [DayDreamer](../../05_intersections/daydreamer/notes.md) 把 RMA 和多篇 locomotion 工作一起列入"仿真大规模随机化训练，部署时冻结"这一派，DayDreamer 自己则转向在真实世界直接学 world model。
  - 同期的 [legged-gym](../legged-gym/notes.md) 用单 GPU 上的大规模并行仿真把 ANYmal 的 locomotion 训练压到分钟级（平地不到 4 min，不平地形约 20 min）。它的机器人、任务和观测都与 RMA 不同，不能和 RMA 的约 24 h 直接对比。
- **与本项目其他主题的联系**：
  - **TTT / TTA**：RMA 是"部署时适应"，但**不做梯度更新**。适应完全靠在仿真里预先学好的推理映射（amortized SysID），这和 [TTT](../../02_test_time_training/ttt-self-supervised/notes.md)、[PAD](../../05_intersections/pad/notes.md) 在测试时用 self-supervised 目标更新权重、[Tent](../../02_test_time_training/tent/notes.md) 用 entropy minimization 这一无监督目标更新权重形成对照。一个自然的问题是：超出随机化支撑集后，能否在 $\phi$ 或 $\pi$ 上加一个测试时的自监督内循环。
  - **World model**：$\phi$ 从 $(x,a)$ 历史中提取动力学差异，相当于只学了"动力学的 latent 上下文"，没有学完整的 latent dynamics。[Robotic World Model](../../05_intersections/robotic-world-model/notes.md) 和 [SimDist](../../05_intersections/simdist/notes.md) 学的是显式 world model，可以与 RMA 对照：前者在 world model 的 imagination 中训练 policy，再部署到真机（作者把真机上的 online learning 留作未来工作）；后者在仿真中预训练 world model，然后在真实世界只用预测误差更新 latent dynamics，实现快速适应。
  - **Real2sim2real**：RMA 是**不做 real-to-sim** 的 sim2real，不校准仿真，只靠宽随机化加在线推断。[SimOpt](../simopt/notes.md) 用真实数据调整随机化分布，[ASAP](../asap/notes.md) 用真实数据学 delta action，[RialTo](../rialto/notes.md) 重建数字孪生。这三条路线都用真实数据修仿真，RMA 则把"适应"推到部署时的 policy 内部。

## 前置知识

1. PPO + GAE（Phase 1 全靠它；懂 clip ratio、value loss 和 $\gamma/\lambda$）
2. POMDP 与 system identification（为什么 $e_t$ 不可观测，以及在线 SysID 与离线 SysID 的区别，参考 UP-OSI）
3. Domain randomization 与"鲁棒但保守"的 trade-off
4. Privileged information / teacher-student 蒸馏，以及 DAgger 为什么能解决 covariate shift
5. 关节位置控制加 PD（action 是期望关节角而非力矩）
6. 1-D temporal CNN 对时间序列的感受野计算

## 复现可行性

- **代码**：原文没有官方代码链接。项目主页链到 `antonilo/rl_locomotion`，它基于 RMA 代码，但服务于后续的视觉 locomotion 工作；依赖 RaiSim 的特定 commit（`f0bb4407…`）、PyTorch、wandb。PPO 部分论文指向 RaisimGymTorch 作为参考实现（supp S1）。另外，社区通常用 legged-gym / Isaac Gym 重写 RMA，这不属于论文内容。
- **算力**（Sec. IV-B）：Phase 1 在"普通台式机、1 GPU"上约 24 h，仿真 1.2B 步；Phase 2 约 3 h，仿真 80M 步。
- **仿真 / 数据**：RaiSim、Unitree A1 URDF、内置 fractal terrain，不需要真实数据。真机部分需要 A1 和板载算力，并实现 100 Hz / 10 Hz 的异步。
- **最小可复现实验**（只用仿真，单 GPU 一天以内）：
  1. 在 legged-gym 或 RaiSim 中搭 A1 加 fractal 地形，按 Table I 随机化 friction/payload/CoM/motor strength，并按 0.004/步 resample；
  2. 按 Sec. III-A 的 10 项奖励加 $k_t$ curriculum，用 PPO 训练 $\pi+\mu$（$z\in\mathbb{R}^8$），得到 Expert；
  3. 冻结 $\pi,\mu$，用 on-policy rollout 训练 $\phi$（50 步历史，1-D CNN）；
  4. 在测试范围内按 0.01/步 resample，比较 Expert / RMA / RMA w/o Adapt / Robust（不给 $z$ 从头训）/ SysID（回归 $e$），对照 Table II 的相对排序；
  5. 补上论文缺的两个消融：用 expert rollout 训练 $\phi$ 与 on-policy 训练的对比，以及历史长度 $k\in\{10,25,50,100\}$；再做一次摩擦阶跃实验，画 $\hat z$ 的时间曲线，复现 Fig. 4 的定性行为。
- **预计难点**：10 项奖励的系数与 curriculum 很敏感。正文说不加 curriculum 会"原地不动"；supp Fig. S2 caption 说不加 $k_t$ 时机器人初期只有负经验，没法很快学会走。不过图里只画了加 $k_t$ 的训练曲线，没有不加 $k_t$ 的对照曲线。换仿真器后绝对数字不会与 Table II 对齐，只能比相对排序。真机 zero-shot 依赖 PD 增益、延迟和传感器细节。文中只给了真机 PD 增益（$K_p=55$、$K_d=0.8$），延迟和传感器细节都没给。

## 自测问题

1. 为什么 $\phi$ 回归的是 $z_t=\mu(e_t)$ 而不是 $e_t$？Table II 中 SysID（56.5）甚至低于 Robust（62.4），这说明了什么？在可辨识性上，$z$ 比 $e$ 好在哪里？
2. 如果用 $\pi(x,a,z_{\text{true}})$ 的 rollout 数据训练 $\phi$，部署时会出现什么分布偏移？用 $\hat z$ 驱动 rollout（DAgger 式）又为什么能缓解？
3. $\phi$ 的输入为什么必须同时含 $a_{t-k:t-1}$ 和 $x_{t-k:t-1}$？只给 $x$ 的历史会丢掉什么信息？
4. 部署时 RMA 更新了哪些参数？它的"适应"和 Tent/TTT 的测试时适应在机制上有何本质区别？遇到训练随机化范围之外的环境，两者各自可能怎样失败？
5. 为什么"$\pi$ 100 Hz + $\phi$ 10 Hz 异步"成立，而"单个网络直接吃完整历史"不行？这个设计依赖环境的什么假设？
