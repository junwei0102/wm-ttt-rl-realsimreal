# DayDreamer: World Models for Physical Robot Learning

> arXiv [2206.14176](https://arxiv.org/abs/2206.14176) · CoRL 2022 · 主题：05_intersections · 层级：foundational
> 代码：https://github.com/danijar/daydreamer（论文正文只给了项目页 https://danijar.com/daydreamer，见 `sections/abstract.tex`；仓库已确认存在，基于 TensorFlow 2）· 笔记依据：`source/*.tex`（数字/公式）+ `paper.pdf`（图）

## 一句话

作者没有提出新算法，在 DreamerV2（RSSM world model + latent imagination 中的 actor-critic）官方实现上主要加了**异步 actor/learner**，然后不用仿真器，直接在 4 台真实机器人上从零在线学习。全部实验共用一套超参数：A1 四足约 1 小时学会翻身、站立和行走；两台机械臂用 8–10 小时，从像素和稀疏 reward 学会 pick and place。核心想法是把 learned world model 当作"机器人自己学出来的快速仿真器"：真实交互只用来训练模型，大部分 trial-and-error 都在 latent imagination 里完成。

## 问题设定

- **设定**：在真实世界做 online RL，不用仿真器、示范或 offline 数据，每个任务单独从零训练（`sections/intro.tex`）。
- **输入/输出**：观测 $x_t$ 是多模态的（proprioception、RGB、depth），动作可以是连续的也可以是离散的。reward 来自手工函数或事件检测，world model 需要学会预测它。环境部分可观测：例如 Sphero 只有俯视 RGB，朝向只能从观测历史推断。
- **四个任务**（`sections/experiments.tex` §3.1–3.4）：

| 机器人 | 控制频率 | 观测 | 动作 | reward |
|---|---|---|---|---|
| Unitree A1 | 20 Hz | 电机角度、姿态、角速度 | 12 维连续（目标电机角，由 PD 控制器执行，经 Butterworth 滤波） | 5 项稠密 shaped reward |
| UR5 | 2 Hz | proprio + 第三人称 RGB | 离散（末端 XYZ 增量 + 夹爪开合） | 稀疏：抓到 +1（由夹爪部分闭合检测），在原 bin 松手 −1，放进对面 bin +10 |
| XArm | 约 0.5 Hz | 同 UR5，另加 depth | 离散 | 同 UR5 |
| Sphero Ollie | 2 Hz | 仅俯视 RGB | 连续电机力矩 | 稠密：到目标的负 L2 距离 |

- **之前的方法卡在哪**（`sections/related.tex`、abstract）：deep RL 需要大量 trial and error，难以在真机上部署。主流做法有三类：在仿真里加 domain/dynamics randomization 再迁移、用机器人 fleet 收集数据、依赖示范或任务先验。仿真复现不了真实世界的复杂性（比如软物体），存在建模误差，部署后也不会随世界变化而适应。已有的真机 model-based 工作也各有局限：Visual Foresight 要在像素空间生成图像来做规划，只能处理短时程任务；Yang et al. 依赖领域专用控制器；SOLAR 只在机械臂上演示了 reaching 和 pushing；PDDM 从状态观测学 dynamics 并做规划。作者的对比点是：本文用一套算法和超参数覆盖了 4 个不同任务。

## 核心方法

整体循环见 Figure 2：
1. 当前策略在真机上采集数据，存进 replay buffer（FIFO，容量 $10^6$）。
2. 从 buffer 采 $B=32$ 条长度 $T=32$ 的子序列，训练 world model。
3. 在 latent 空间 imagine 轨迹，训练 actor-critic。
4. 用新策略继续采集数据。

**1. World model（RSSM，`sections/method.tex` Eq.1）**

$$\mathrm{enc}_ \theta(s_t\mid s_{t-1},a_{t-1},x_t),\quad \mathrm{dyn}_ \theta(s_t\mid s_{t-1},a_{t-1}),\quad \mathrm{dec}_ \theta(s_t)\approx x_t,\quad \mathrm{rew}_ \theta(s_{t+1})\approx r_t$$

- $s_t$ 包含确定性的 recurrent state $h_t$ 和随机表示 $z_t$。encoder 是 posterior，能看到当前观测 $x_t$；dynamics 是 prior，只看历史和动作。imagination 时只用 prior。
- **传感器融合**：encoder 把所有模态融合成 $z_t$， $z_t$ 是离散 code，共 32 个 latent、每个 32 类（附录 D 超参数表）。论文没有给出融合网络的具体结构。
- decoder 只提供表示学习信号，并方便人查看模型预测，behavior learning 用不到它。Figure B.1 解码了 imagined 轨迹，可以看到 UR5 场景里静止的橙球变成绿球的幻觉。
- **损失**：本文只说各组件"通过 stochastic backpropagation 联合优化"，没写公式，附录 D 超参数表只给出 KL balancing = 0.8。按本文所基于的 DreamerV2，损失形式如下（**这是 DreamerV2 的公式，不是本文的**）：
$$\mathcal{L}(\theta)=\mathbb{E}\Big[\textstyle\sum_t -\ln p_\theta(x_t\mid s_t)-\ln p_\theta(r_t\mid s_t)+\alpha\thinspace\mathrm{KL}[\mathrm{sg}(q)\thinspace\Vert\thinspace p]+(1-\alpha)\thinspace\mathrm{KL}[q\thinspace\Vert\thinspace\mathrm{sg}(p)]\Big]$$
  其中 $q$ 是 encoder posterior， $p$ 是 dynamics prior， $\alpha=0.8$。这样 KL 项主要把 prior 拉向 posterior，而不是让 posterior 迁就 prior。注意这是简化写法：DreamerV2 原式还有 discount 预测项 $-\ln p_\theta(\gamma_t\mid s_t)$ 和 KL 缩放系数 $\beta$，本文是否使用以及取值都没有报告（原文未报告）。

**2. Actor-critic in imagination（Eq.2–4）**

- 从 replay 子序列中的每个 posterior 状态出发（DreamerV2 的做法，本文没有明写），用 dynamics prior 加 actor 往前 roll $H$ 步，reward 由 $\mathrm{rew}_ \theta$ 预测。 $H$ 在正文写的是 16，附录 D 超参数表写的是 15。起点数是 $32\times32=1024$，乘以约 16 步约为 16K，正好对应正文"单 GPU 上典型 batch 16K"的说法（这个对应是我推算的）。
- **λ-return**（Eq.3）：
$$V^\lambda_t \doteq r_t+\gamma\big((1-\lambda)\thinspace v(s_{t+1})+\lambda V^\lambda_{t+1}\big),\qquad V^\lambda_H\doteq v(s_H)$$
  它对 $N\in[1,H-1]$ 的所有 N-step return 做指数加权平均。末端的 bootstrap 项让 actor 能考虑 $H$ 步之外的 reward。 $\gamma=\lambda=0.95$。计算 λ-return 用的是一个缓慢更新的 target critic，更新间隔为 100（附录 D 超参数表）。
- **Critic**：回归 λ-return。论文只有文字描述，没有给公式；按 DreamerV2 的写法是平方误差 $\mathbb{E}[\sum_t\frac12(v(s_t)-\mathrm{sg}(V^\lambda_t))^2]$（这是 DreamerV2 的形式，不是本文给出的）。
- **Actor**（Eq.4，这里写的是 Reinforce 形式）：
$$\mathcal{L}(\pi)\doteq-\mathbb{E}_ {\pi,p_\phi}\Big[\textstyle\sum_{t=1}^{H}\ln\pi(a_t\mid s_t)\thinspace\mathrm{sg}\big(V^\lambda_t-v(s_t)\big)+\eta\thinspace\mathrm{H}[\pi(a_t\mid s_t)]\Big]$$
  离散动作（UR5、XArm）用 Reinforce，以 $v(s_t)$ 作 baseline。连续动作（A1、Sphero）用 reparameterization：直接最大化 $V^\lambda$，梯度沿可微的 dynamics 反传到动作。熵正则项 $\eta$ 防止策略坍缩成确定性策略，但附录 D 超参数表没有列出 $\eta$ 的数值。
- **梯度隔离**：actor 和 critic 的梯度不回传到 world model。否则策略会把模型"改"成对自己有利的样子，得到 incorrect and overly optimistic 的预测。

**3. 异步 actor/learner（论文描述的主要系统层改动）**

- learner 线程持续训练 world model 和 actor-critic；actor 线程并行计算动作，与环境交互（论文写的是 thread；开源仓库里 learner 和 actor 实际是两个分开启动的进程）。对 20 Hz 的 A1，这个设计是保证低延迟所必需的；对慢速机械臂，它让训练在等待环境时也不停下（`sections/experiments.tex` Implementation）。
- 因此 DreamerV2 里的 train frequency 超参数被删掉了，learner "without rate limiting"。每条真实样本被复用多少次，取决于 GPU 速度和控制频率。
- **训练/部署**：两者没有分开。部署时，actor 用 encoder 做 filtering 得到 $s_t$，再从 $\pi(a_t\mid s_t)$ 采样；与此同时 world model 和策略一直在更新。这就是论文所说 "adaptation" 的来源。

## 关键实验与证据

所有任务共用一套超参数（附录 D 超参数表，`tables/hparams.tex`）：MLP 4×512，LayerNorm+ELU，RSSM 512，lr $10^{-4}$，Adam $\epsilon=10^{-6}$，梯度裁剪 100，Start learning $10^4$。

- **A1 行走**（§3.1，Figure 4/8）：机器人从背朝下开始，全程没有 reset。约 5 分钟学会翻身落地，再过约 20 分钟学会站立，约 1 小时学会 pronking 步态，按目标速度前进。SAC 只学会翻身，既不会站也不会走，训练中还要人帮它摆脱腿部死锁。行走学会后，作者用长杆反复把机器人推倒；额外在线学习 10 分钟后，它能扛住轻推，被重推后也能迅速翻身站起。reward 见 Eq.5，共五项，每一项只有在它前面的各项都达到 0.7 以上时才激活，否则置 0：
$$r^{\text{upr}}=\tfrac{\hat z^T[0,0,1]-1}{2},\quad r^{\text{hip}}=1-\tfrac14\Vert q^{\text{hip}}+0.2\Vert_1,\quad r^{\text{shoulder}}=1-\tfrac14\Vert q^{\text{shoulder}}+0.2\Vert_1$$
$$r^{\text{knee}}=1-\tfrac14\Vert q^{\text{knee}}-1.0\Vert_1,\quad r^{\text{velocity}}=5\Big(\tfrac{\max(0,{}^{\mathcal B}v_x)}{\Vert{}^{\mathcal B}v\Vert_2}\thinspace\mathrm{clip}({}^{\mathcal B}v_x/0.3,-1,1)+1\Big)$$
  Figure 4 标注最大 reward 为 14，图中是**单次运行**，阴影表示每个时间 bin 内的 1 个标准差。
- **UR5 多物体 pick and place**（§3.2，Figure 5）：场景里有 3 个球。Z 轴只在持物时可以移动，夹爪到了正确 bin 上方会自动张开。Dreamer 在 8 小时内达到平均 2.5 objects/min，前约 2 小时几乎没有进展，之后才开始改善。Rainbow 和 PPO 只学会"抓起后在原 bin 丢下"的短视行为。人类基线来自录制 3 名操作者用摇杆操作 20 分钟（没说是每人 20 分钟还是合计）；Figure 5 里人类虚线大约在 3 附近（这是图上读数）。
- **XArm pick and place**（§3.3，Figure 6）：软物体用绳子系在夹爪上，观测多了 depth。Dreamer 10 小时达到 3.1 objects/min，与人类相当；Rainbow 停在"原 bin 抓放"的局部最优。作者还观察到 Dreamer 会拉绳子把物体从角落拽出来。
- **Sphero 导航**（§3.4，Figure 7）：只有 RGB 输入，每 100 步结束一个 episode，然后用随机的大功率动作把机器人打到新位置。Dreamer 在 2 小时内把平均到目标的距离降到 0.15（以场地尺寸为单位，按时间步平均）。DrQv2 的结果相近，这是四个任务里 Dreamer 唯一没有占优的一个。
- **适应**（Appendix A，Figure A.1）：XArm 在夜间训练。日出时强光照进场景，性能崩溃；算法不做任何改动继续训练，约 5 小时后恢复并超过原来的水平。作者说这比从零训练快，但没有给从零训练的对照曲线。
- **消融：没有。** 异步 learner、离散 latent、KL balancing、decoder、多模态融合都没有做消融，所以从本文无法判断哪个组件真正起作用。现有证据只有"Dreamer 整体 vs. 每个任务一个 model-free baseline"。

## 局限与论文没说的

**作者承认的**（`sections/discussion.tex`）：真机长时间训练会磨损硬件，可能需要人工干预或维修；Dreamer 和各 baseline 都需要训练更久才能看清上限；更难的任务也许要把真实世界的快速学习和仿真器结合起来。

**我读出来的：**

1. **统计口径弱**：A1 明确是单次运行，其余任务没说用了几个种子；A1 图的阴影是每个时间 bin 内的 1 个标准差，不是跨种子统计，其余图没说明阴影含义。"1 小时学会走"只是单条曲线上的一个时间点。
2. **baseline 可能吃亏**：SAC、Rainbow、PPO 的超参数都没有报告；图像和 proprio 的融合方式是把 proprio broadcast 成平面拼到 RGB 通道上；训练预算（墙钟时间）与 Dreamer 相同，而作者自己也推测 Rainbow 和 PPO 失败是因为需要更多经验。Sphero 上 DrQv2 与 Dreamer 打平，说明优势因任务而异。
3. **任务工程很重**：A1 的门控 reward 需要 base 速度和朝向，论文没说这些量怎么测。UR5 的 Z 轴只在持物时启用、到正确 bin 上方自动开爪（XArm 只说动作空间与 UR5 相同），大幅缩小了搜索空间。Sphero 的 reward 需要机器人位置，来源也没说明。另外还有人工干预：A1 走到场地边缘时被人搬回，SAC 被人解开腿部死锁。
4. **异步训练让样本效率和硬件绑定**：由于没有 train ratio，每条样本对应多少次梯度更新取决于 GPU 吞吐和控制频率。论文没报告梯度步数和 GPU 型号，换一套硬件，"1 小时 / 8 小时"的结论未必成立。
5. **"同一套超参数"对应的时间尺度并不相同**（以下是我的推算）： $\gamma=0.95$ 的有效视界约为 $1/(1-\gamma)=20$ 步，在 A1 上约 1 秒，在 UR5 上约 10 秒，在 XArm 上约 40 秒。同样，如果 Start learning $=10^4$ 按环境步计，XArm 要约 5.6 小时后才开始训练。Figure 6 前约 6 小时接近 0，看起来与此吻合；但 Figure A.1 里的 XArm 曲线在第 2–5 小时就已明显上升，这又与"前 5.6 小时不训练"矛盾。所以"按环境步计"这个解释并不自洽。论文没有说明这个数的单位，复现时需要核对。
6. **文中有几处不一致**：
   - imagination horizon 正文写 $H=16$，附录 D 超参数表写 15。
   - 按 Eq.5 字面， $r^{\text{upr}}\in[-1,0]$。这样五项上限合计 13，而且 $r^{\text{upr}}$ 永远达不到 0.7 的门控阈值，和 Figure 4 的"max reward 14"对不上。我怀疑原意是 $(\hat z^T[0,0,1]+1)/2$，应以代码为准。
   - Figure A.1 曲线峰值约 9 objects/min（日出前就已到约 4–5），而 Figure 6 最高约 3（图上读数），论文没说明两图是否来自同一次运行、是否用同一统计口径。
7. **没测的 setting**：
   - 都是单任务，目标位置和目标速度固定，没有泛化测试（新物体、新目标、新地形）。
   - world model 没有跨任务复用，intro 里说的"可复用"没有验证。
   - 没有使用 offline 数据或示范。
   - adaptation 只是继续训练：没研究遗忘（FIFO buffer 里新旧光照的数据混在一起），也没有从零训练的对照。
   - 安全性只靠滤波和保护壳，没有显式约束。
8. **imagination 质量没有量化**：Figure B.1 已经出现物体变色的幻觉，多物体场景下 model exploitation 的风险有多大，论文没有度量。

## 在脉络中的位置

- **前驱**：
  - [Dreamer](../../01_world_model/dreamer/notes.md)：latent imagination actor-critic 和 λ-return。
  - PlaNet 的 RSSM。
  - DreamerV2（本项目未收录）：离散 latent、KL balancing、按动作类型选择 Reinforce 或 reparameterization。本文直接基于它的官方实现。
  - [World Models](../../01_world_model/world-models/notes.md)：在 learned model 里训练策略的思想。
  - 真机 MBRL：Visual Foresight、SOLAR、PDDM。
  - baseline：[SAC](../../04_rl/sac/notes.md)、Rainbow（[DQN](../../04_rl/dqn/notes.md) 系）、[PPO](../../04_rl/ppo/notes.md)、DrQv2。
- **后继 / 影响**：
  - 同一作者线：[DreamerV3](../../01_world_model/dreamerv3/notes.md) 延续"固定超参数跨领域"的目标；[Dreamer 4](../../01_world_model/dreamer4/notes.md) 转向可扩展的 world model 和离线数据。
  - 用 learned simulator 训练真机策略：[Robotic World Model](../robotic-world-model/notes.md)；[SimDist](../simdist/notes.md) 先在仿真中预训练 world model，部署时再快速适应，正好对应本文 discussion 里"结合真实世界快速学习与仿真器"的方向。
  - 用生成式 world model 做 policy 评估或 RL：[UniSim](../unisim/notes.md)、[Ctrl-World](../ctrl-world/notes.md)、[WMPO](../wmpo/notes.md)、[World-Gymnast](../world-gymnast/notes.md)。它们从离线数据学 world model，本文则是在线从零学。
  - 真机 RL 的另一条路：[HIL-SERL](../../04_rl/hil-serl/notes.md) 走 model-free + 示范 + 人类干预路线；它的 related work 把本文列为真机 model-based RL 的已有工作，并称这类方法通常训练时间长。
  - [WM for robot learning survey](../../01_world_model/survey-wm-robot-learning/notes.md) 把本文列为"world model 当 simulator 训 policy"的早期代表。
- **与本项目其他主题的联系**：
  - **Real2sim2real**：本文走到了"完全不用 sim"的一端，对照面包括 [Domain Randomization](../../03_real2sim2real/domain-randomization/notes.md)、[legged_gym](../../03_real2sim2real/legged-gym/notes.md)（仿真中大规模并行）和 [RMA](../../03_real2sim2real/rma/notes.md)（仿真训练，部署时推断环境参数）。本文在 latent 空间做 16K 的大 batch 并行，相当于把 Isaac Gym 式的并行搬进了学到的模型。这样确实绕开了 reality gap（参见 [reality gap survey](../../03_real2sim2real/survey-reality-gap/notes.md)），代价是真机时间、硬件磨损和 reward 工程。[ASAP](../../03_real2sim2real/asap/notes.md) 用真实数据校正仿真动力学，介于两者之间。
  - **TTT / TTA**：A1 被推、XArm 遇上日出时的"adaptation"，本质是部署期继续做 online RL：world model 的损失（重建 + KL 的自监督部分，加上 reward 预测）和 reward 驱动的 actor-critic 一起更新，需要 reward，耗时从 10 分钟到 5 小时不等。可以对比三种做法：[TTT](../../02_test_time_training/ttt-self-supervised/notes.md) 用单样本自监督辅助任务；[Tent](../../02_test_time_training/tent/notes.md) 做熵最小化；[PAD](../pad/notes.md) 部署时只用自监督任务更新 encoder，不需要 reward。一个本文没回答的问题是：如果把 world model 的重建/KL 损失当作 TTT 辅助任务，部署时只更新 encoder，能否把视觉适应从小时级缩短到分钟级？
  - **RL**：Eq.3/4 就是 TD(λ) 加带 baseline 和熵正则的 policy gradient，只是 rollout 来自 learned model，而不是像 PPO/SAC 那样来自真实交互。

## 前置知识

1. RSSM 与 Dreamer 的 latent imagination（先读 [Dreamer 笔记](../../01_world_model/dreamer/notes.md)）
2. VAE/ELBO、reparameterization trick；DreamerV2 的离散 latent 与 KL balancing
3. Actor-critic、TD(λ)/λ-return、target network
4. Policy gradient 的两种估计器：Reinforce（score function + baseline）与通过可微 dynamics 反传
5. POMDP 与 filtering（为什么 Sphero 需要 recurrent state 来推断朝向）
6. Off-policy replay、actor/learner 异步架构、replay ratio
7. 机器人基础：PD 控制、控制频率、proprioception

## 复现可行性

- **代码**：官方仓库 https://github.com/danijar/daydreamer，依赖 TensorFlow 2、TensorFlow Probability 等。learner（GPU）和 actor 是两个独立进程，需在两个终端分别启动（论文正文说的是 thread）；仓库页面列出了 A1（含仿真配置）、XArm、UR5 的运行命令，没有 Sphero。
- **算力**：论文**没有报告** GPU 型号、数量和总梯度步数，只说约 16K 的 imagination batch 能在 "a single GPU" 上跑（`sections/method.tex`）。
- **硬件与数据**：没有数据集，全部在线采集。硬件包括：
  - Unitree A1，外加打印的保护壳（见致谢）；
  - UR5 或 XArm，加第三人称相机（XArm 用 RealSense RGB-D）；
  - Sphero Ollie，加俯视相机。

  墙钟时间：A1 约 1 小时，UR5 8 小时，XArm 10 小时，Sphero 2 小时。
- **最小复现实验**：
  1. **无硬件版**：用仓库的 A1 仿真配置，保留异步 actor/learner，让环境按 20 Hz 真实时间步进。机器人从背朝下初始化，reward 按 Eq.5 实现（注意 $r^{\text{upr}}$ 的符号问题），跑 1 小时墙钟，画出类似 Figure 4 的曲线。再用同步训练加固定 train ratio 做对照，并记录 learner 实际做了多少梯度步。这样正好补上论文缺的"异步 vs. 同步"消融和 replay ratio 数据。
  2. **低成本真机版**：复现 Sphero 导航。用廉价差速小车加俯视摄像头，reward 由视觉定位计算；预算 2 小时，与 DrQv2 对比，并检验"仅从图像历史推断朝向"是否成立。
- **难点**：
  - 真机的 reset 和安全：需要保护壳、指令滤波，还要人工把机器人搬回场地。
  - reward 所需的状态估计（速度、位置）论文没有交代。
  - 异步训练下结果依赖 GPU 吞吐。
  - 单次运行方差大，至少跑 3 个种子。
  - 旧版 TF2 依赖的环境不好搭。

## 自测问题

1. 为什么 actor 和 critic 的梯度不能回传到 world model？如果允许回传，imagined return 和真实表现会怎样背离？
2. 按 Eq.3 手推 $V^\lambda_t$ 的递推，并说明 $V^\lambda_H=v(s_H)$ 和 target critic 分别解决什么问题。在 $\gamma=0.95$ 时，A1（20 Hz）和 XArm（0.5 Hz）的有效物理视界各约多少？
3. 连续动作用 reparameterization，离散动作用 Reinforce。两者各要求 world model 具备什么性质（可微性、方差）？Eq.4 里 $\mathrm{sg}(V^\lambda_t-v(s_t))$ 起什么作用？
4. 去掉 train ratio 之后，每条真实样本被用于更新的次数由什么决定？这会怎样影响"1 小时学会走路"在另一台 GPU 上的可复现性？
5. XArm 遇到日出后的"adaptation"与 Tent、PAD 这类 test-time adaptation 有什么本质区别？分别更新了哪些参数、用了什么监督信号、需要多长时间？
