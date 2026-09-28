# Precise and Dexterous Robotic Manipulation via Human-in-the-Loop Reinforcement Learning

> arXiv [2410.21845](https://arxiv.org/abs/2410.21845) · Science Robotics 10(105), eads5033 (2025)（已由 Crossref 核实；arXiv 页面没有 journal-ref，本地 PDF 是 2025-03 的 v3）· 主题：04_rl · 层级：milestone
> 代码：https://github.com/rail-berkeley/hil-serl（论文只给出项目页 https://hil-serl.github.io/；repo 已用 WebFetch 确认存在）· 笔记依据：`source/*.tex`（数字/公式）+ `paper.pdf`（图）

## 一句话

HIL-SERL 是一套直接在真机上跑的视觉 RL 系统。它以 RLPD（SAC 加 prior data）为底座，训练中允许人随时用 SpaceMouse 接管机器人。人的纠错数据同时写进 demo buffer 和 RL buffer，再用 RL 而不是监督学习去优化。靠这一点，13 个精密装配、双臂协作和动态操作任务在 1 到 2.5 小时内（timing belt 例外，用了 6 小时）做到 100% 成功率，cycle time 平均快 1.8 倍（Table 1）。

## 问题设定

- **形式化**（`sections/methods.tex` §Preliminaries）：MDP $\{\mathcal S,\mathcal A,\rho,\mathcal P,r,\gamma\}$。状态是多路相机图像（裁剪后 resize 到 128×128）加本体感知（tcp pose/vel、F/T、夹爪状态；两个动态任务还用关节 q/dq），动作是末端 6D twist（双臂 12D，有夹爪时另加离散夹爪动作），两个动态任务则是 3 维前馈 wrench（Jenga 为 $F_x,F_z,\tau_z$，flipping 为 $F_x,F_z,\tau_y$），控制频率 10 Hz（Table 2–12）。
- **reward**：稀疏的二值 reward，由每个任务单独训练的 success classifier 给出。只有 Jenga 例外，由人在 episode 结束时标注（Table 11）。有夹爪的任务额外对夹爪动作加一个小的负惩罚，数值论文没给（`sections/results.tex`）。
- **环境**：完全在真实世界训练，不用仿真器。reset 靠脚本或人工完成，初始状态在小范围内随机化（2–8 cm，部分任务不随机）。
- **之前卡在哪**：真实世界 RL 有三个难点，sample complexity、reward 难以指定、优化不稳定（`sections/intro.tex`）。前作 SERL 只用离线 demo，聚焦 horizon 较短的简单任务，没有涉及双臂协作和动态操作（`sections/related_work.tex`）。模仿学习（BC、HG-DAgger、Diffusion Policy）受限于人类数据本身的次优，也没有从自身失败中学习的机制。

## 核心方法

**1. RL 底座：RLPD**（`sections/methods.tex` Eq.1–2）。每个训练 batch 一半取自 prior data（demo buffer），一半取自在线数据（RL buffer）。critic 和 actor 按 SAC 的方式更新：

$$\mathcal L_Q(\phi)=\mathbb E_{s,a,s'}\Big[\big(Q_\phi(s,a)-(r(s,a)+\gamma\,\mathbb E_{a'\sim\pi_\theta}[Q_{\bar\phi}(s',a')])\big)^2\Big]$$

$$\mathcal L_\pi(\theta)=-\mathbb E_{s}\Big[\mathbb E_{a\sim\pi_\theta}[Q_\phi(s,a)]+\alpha\,\mathcal H(\pi_\theta(\cdot\mid s))\Big]$$

$Q_{\bar\phi}$ 是 target network，$\alpha$ 自适应调节，策略是 Gaussian。两点需要注意：Eq.1 的 target 里没有 SAC 标准 soft backup 中的 $-\alpha\log\pi$ 项；RLPD 的 critic ensemble、UTD 比例等细节正文和补充材料都没写。

**2. 系统架构**（Fig.2）。actor 进程在机器人上执行当前策略，learner 进程从两个 buffer 等量采样做 RLPD 更新，两者异步通信并定期同步参数。所有相机共用一个在 ImageNet 上预训练的 ResNet-10，各路 embedding 拼接后再和本体感知编码（64 维）合并，接 256×256 的 MLP。数据增强用 random crop。backbone 是否冻结论文没说。

**3. Human-in-the-loop**（`sections/methods.tex` §3.4，全文的核心）。rollout 过程中，人可以在任意时刻 $t_i$ 接管，每次最多接管 $N$ 步，一条轨迹里可以多次接管。执行的动作为

$$a^*_t=\begin{cases}a_{itv}, & \text{人在干预}\\ a_{RL}, & \text{否则}\end{cases}$$

数据的去向是整套方法的关键：**干预片段同时写入 demo buffer 和 RL buffer，策略自己产生的转移只写入 RL buffer。** 这样干预数据在 50/50 采样下被放大权重，而且是在策略实际访问到的错误状态附近给出的，属于 on-distribution 的纠错。它和 HG-DAgger 的区别是：纠错数据交给 RL 按 reward 优化，而不是直接监督回归（同 RLIF 的思路）。操作上的经验是：训练早期多干预，然后逐步减少；只在具体出错时纠正，其余时间让机器人自己探索。作者特别警告，不要反复用"长时间接管直到任务成功"的方式干预，这会让 value function 在训练早期被高估，导致训练不稳定（§Training Process）。

**4. Reward classifier**（§Training Process，Supp. B）。靠遥操作采约 200 个正样本、1000 个负样本（约 10 条轨迹，大约 5 分钟）。模型是 ResNet-10 加两层 MLP，用交叉熵训练，Adam 3e-4，训 100 iterations。出现 false positive / false negative 时要人工补采数据。正文称在评估集上准确率一般 >95%；各任务表中报告的是 95–99%（SSD 和 IKEA top panel 为 95%；Jenga 不用 classifier）（Table 2–12）。

**5. 夹爪用单独的 DQN critic**（Eq.3）。连续动作 MDP $\mathcal M_1$ 和离散夹爪 MDP $\mathcal M_2$ 共享同一个状态。离散动作是 {open, close, stay}，双臂时为 $3^2=9$ 种组合。

$$\mathcal L(\theta)=\mathbb E\Big[\big(r+\gamma\,Q_{\theta'}(s',\arg\max_{a'}Q_\theta(s',a'))-Q_\theta(s,a)\big)^2\Big]$$

形式上这是 Double DQN（用 online 网络选动作、target 网络评估），$\theta'$ 由 Polyak averaging 得到。训练和推理时都先从 $\pi$ 取连续动作，再对 grasp critic 的输出取 argmax 得到离散夹爪动作，两者拼接后执行。

**6. 下游控制与表征**（Supp. D，Eq.5–7）。

- **ego-centric 本体感知**：每个 episode 开始时，末端初始位姿 $\{b_0\}$ 在一个区域内均匀随机采样。之后的位姿都用相对量 $T_{b_0b_t}=T_{b_0}^{-1}T_{b_t}$ 表示。twist 动作定义在当前末端坐标系 $\{b_t\}$ 下，下发前用 adjoint 映射转到基座系：$\mathcal V'=[\mathrm{Ad}_{T_{b_t}}]\mathcal V$。这等价于把随机化施加在目标物体的相对位置上，所以目标在执行中移动时策略仍然有效（Fig.6A）。
- **impedance controller（1000 Hz）加 reference limiting**：$F=k_p e+k_d\dot e+F_{ff}+F_{cor}$，并把误差裁剪到 $|e|\le\Delta$，使接触力上界为 $k_p|\Delta|+2k_d|\Delta|f$。这样既不需要降低增益，也能保证随机探索时的安全。两个动态任务改为直接下发前馈 wrench，经 $J^\top$ 转成关节力矩。

**训练流程**（Fig.3）：选相机、裁剪 → 采 classifier 数据并训练 → 采 20–30 条 demo 填入 demo buffer → 在线 RLPD 训练，人适时干预，干预率逐步降到 0 → 收敛。

## 关键实验与证据

- **设置**：共 7 类任务、13 个评测项，包括 RAM/SSD/USB/线缆卡扣、IKEA 三个子任务及整机组装、汽车仪表盘双臂装配、双臂 handover、timing belt、Jenga whipping、平底锅翻物。除 IKEA 整机只测 10 次外，每项评测 100 次。所有机载计算在单张 RTX 4090 上完成（`sections/results.tex`）。
- **对比模仿学习**（Table 1a）：BC 列实际是 HG-DAgger，与 RL 用相同数量的 episode，干预量"尽量相当"。Jenga 和 flipping 两项例外，用的是 flat BC，分别给了 50 和 200 条 demo。HIL-SERL 在全部任务上都是 100%，基线平均 49.7%（+101%）。cycle time 从平均 9.6 s 降到 5.4 s（1.8x）。差距最大的几项：timing belt 2→100（训练 6 h），Jenga 8→100，USB 26→100，RAM 29→100，IKEA 整机 1/10→10/10。
- **对比 RL 基线与消融**（Table 1b，只做了 RAM / Dashboard / Flipping 三个任务；DP、BC 用 200 条 demo 训练，IBRL、Residual RL、DAPG 用 200 条 demo 初始化，HG-DAgger 与 RL 用相同数量的 episode 和干预；两个消融分别是"无 demo 无干预"和"200 条 demo、无干预"）：

| | DP | HG-DAgger | BC | IBRL | Residual RL | DAPG | 无demo无干预 | 无干预(200 demo) | HIL-SERL |
|---|---|---|---|---|---|---|---|---|---|
| RAM | 27 | 29 | 12 | 75 | 0 | 8 | 0 | 48 | 100 |
| Dashboard | 18 | 41 | 35 | 0 | 0 | 18 | 0 | 0 | 100 |
| Flipping | 56 | 46 | 46 | 95 | 97 | 72 | 0 | 100 | 100 |

  从表中能读出三点：(i) 从零开始的 RL（既无 demo 也无干预）在三个任务上全是 0，说明人类先验数据必不可少；但这个消融同时去掉了 demo 和干预，并没有单独隔离 demo 的作用。(ii) 在线干预对 Dashboard（0→100）和 RAM（48→100）是决定性的，对以开环为主的 flipping 则不需要。(iii) BC 的 demo 从 20 条加到 200 条，flipping 成功率反而从 47% 变成 46%，说明加 demo 解决不了问题。
- **学习曲线**（Fig.5，RAM / USB / IKEA top panel / Dashboard 四个任务，按 20 个 episode 滑动平均）：HIL-SERL 的干预率降到 0%，cycle time 持续下降。按 caption 的说法，HG-DAgger 的干预率没有持续下降，cycle time 也没有改善（不过从图上看，Dashboard 的 HG-DAgger cycle time 其实也有下降）。
- **机制分析**（`sections/analysis.tex`）：以 RAM 任务为例，状态访问热力图逐渐收成一个 "funnel"（Fig.7）。作者把 "critical states" 定义为 Q 对动作扰动方差大的状态（Eq.4）：$\mathrm{Var}[Q]=\mathbb E_{\epsilon\sim\mathcal U[-c,c]}[(Q(s,a+\epsilon)-\mathbb E_\epsilon Q(s,a+\epsilon))^2]$，取 $c=0.2$，蒙特卡洛采样 100 次。funnel 内部的状态 Q 值高、方差也大。HG-DAgger 的访问分布更弥散。另外，Gaussian 策略的 std 能区分两类控制方式（Fig.8）：RAM 起始 std 约 0.6，接近目标时快速下降，对应 reactive 策略；Jenga 的 std 始终接近 0，对应 predictive 的开环策略。
- **鲁棒性**（Fig.6）：推动主板、强行打开夹爪、扰动皮带时，策略会重新抓取或重试。这部分只有定性展示，没有数字。

## 局限与论文没说的

- **作者承认的**：更长 horizon 的任务不确定能否扩展；随机化程度有限，没有测非结构化环境里的泛化；每个任务都要从头训练（`sections/discussion.tex`）。
- **收益来源有混淆**：Table 1b 中 IBRL、Residual RL、DAPG 都拿不到在线干预。只比较"无干预"版本的话，HIL-SERL 在 RAM 上是 48，而 IBRL 是 75；Dashboard 两者都是 0。可见相对 RL 基线的优势主要来自人类干预这一额外信号，而不是 RLPD 本身。论文强调 "careful integration" 的作用，但 pretrained backbone、grasp DQN、ego-centric 表征、reference limiting 这些组件都没有单独做消融。
- **人力成本没有量化**：按正文定义，"1–2.5 h" 的训练时间包括 scripted motion、policy rollout、有意暂停和机载计算，没有提到 classifier 数据和 demo 的采集时间，也不反映人全程盯守的注意力成本。干预的总步数和次数没有列表给出，与 HG-DAgger 的干预量也只是 "aim to provide a comparable number"。另外 timing belt 实际用了 6 h（Table 1a），不在摘要所说的范围内。
- **评测口径**：正文说评测协议见补充材料，但 `supp.tex` 里没有这一节。成功由谁判定（classifier 还是人）没有说明。论文没有报告多次训练（多 seed）的方差，也没有误差条。随机化范围很窄（handover、flipping、Jenga 为 None）。
- **reward classifier 的风险**：准确率 95–99%，剩下的误判可能被策略利用（reward hacking）。论文的应对是人工补采 FP/FN 样本，这一步依赖经验、需要反复迭代。
- **文本不一致**：Diffusion Policy 在 Dashboard 上的成功率，正文写 28%，Table 1b 写 18（平均值 34 与 18 一致）。Eq.1 的 target network 引用 mnih2013playing，但那篇并没有 target network（见 [DQN 笔记](../dqn/notes.md)）。Eq.3 说是"standard DQN"，实际形式是 Double DQN。
- **规模**：每个任务一个小网络（ResNet-10 加 MLP），单任务、无语言条件。它能否扩展到 VLA 这类大模型，本文没有回答。

## 在脉络中的位置

- **前驱**：[SAC](../sac/notes.md)（actor-critic 与熵正则）；RLPD（Ball et al. 2023，本项目未收录；把 prior data 和在线数据对半采样）；SERL（Luo et al. 2024，本项目未收录；同一系统的前作，只用 demo）；[DQN](../dqn/notes.md)（grasp critic、target network）；HG-DAgger（人接管纠错，数据用于监督学习）和 RLIF（把干预数据交给 RL 而不是监督学习）；真机 model-based RL 的 [DayDreamer](../../05_intersections/daydreamer/notes.md)，在 related work 中作为真机 model-based RL 的例子被引用。
- **后继 / 影响**：[WMPO](../../05_intersections/wmpo/notes.md) 在引言里把 HIL-SERL 归为"靠人类干预引导学习，费人力、难扩展"的一类，转而在 world model 里做 VLA 的 RL（`05_intersections/wmpo/source/sections/01introduction.tex`）。[π*0.6](../pi-star-06/notes.md) 把"demo + 专家干预 + 自主 rollout + 结果 reward"这套数据配方扩展到 VLA，policy extraction 改成 advantage conditioning；它引用的是 SERL，没有引用本文。[π_RL](../pi-rl/notes.md) 和 [SimpleVLA-RL](../simplevla-rl/notes.md) 则把 VLA 的 RL 放到仿真里，与本文的真机路线相对。[SimDist](../../05_intersections/simdist/notes.md) 指出 RLPD/SERL 这类高频 critic 更新容易造成 value overestimation，改用"仿真预训练 world model 再到真机快速适应"的做法。
- **与本项目其他主题的联系**：
  - **World model**：HIL-SERL 是纯 model-free 方法，每个任务要 1–10.8 万条真机转移（Table 2–12）。这给 [World-Gymnast](../../05_intersections/world-gymnast/notes.md)、[Ctrl-World](../../05_intersections/ctrl-world/notes.md)、[RWM](../../05_intersections/robotic-world-model/notes.md) 这类"用 world model 替代真机交互"的工作提供了一个真实样本预算的参照。
  - **TTT**：本文是在部署环境里用 reward 和人类信号做完整的 online RL。它和 [PAD](../../05_intersections/pad/notes.md) 那种部署时靠 self-supervised 目标做适应的思路方向相同，只是监督信号更强、代价也更高。作者提出的未来方向"先预训练 value function 再快速 fine-tune"属于 pretrain-then-adapt 范式。
  - **Real2sim2real**：本文完全绕开了 sim-to-real gap，代价是几乎不做随机化。可与 [RialTo](../../03_real2sim2real/rialto/notes.md)、[Domain Randomization](../../03_real2sim2real/domain-randomization/notes.md)、[ASAP](../../03_real2sim2real/asap/notes.md) 对照阅读。ego-centric 相对坐标可以看作一种廉价的空间泛化手段。
  - **RL**：它和 [legged-gym](../../03_real2sim2real/legged-gym/notes.md) 那种"仿真里海量并行跑 PPO"的路线相对。与 [IQL](../iql/notes.md)（offline RL）和 [DPPO](../dppo/notes.md)（微调 diffusion policy）的区别在于：本文是 off-policy online RL 加 prior data，策略用 Gaussian。

## 前置知识

1. SAC：最大熵 actor-critic、自适应温度 $\alpha$、target network 与 Polyak averaging
2. RLPD / offline-to-online RL：prior data 与在线数据的对称采样
3. DQN / Double DQN：离散动作的 Q-learning
4. DAgger / HG-DAgger：BC 的 compounding error，以及交互式纠错
5. 稀疏 reward 与 success classifier，reward hacking
6. 机器人学基础：齐次变换、twist、adjoint 映射、impedance control
7. 视觉 RL 技巧：预训练 encoder、random crop 数据增强

## 复现可行性

- **代码**：官方开源（上面的 GitHub）。按 README，代码基于 JAX 和 Python 3.10，分为 `serl_launcher`（训练）和 `serl_robot_infra`（硬件接口），支持 Franka arm（依赖 `serl_franka_controllers`）。示例任务有 RAM insertion、USB pickup-insertion、object handover、egg flip。
- **硬件**：论文正文没写机械臂和相机的型号，Franka 这一信息来自 README。还需要 SpaceMouse、腕部相机和侧面相机，双臂任务需要两台机械臂。**算力**：单张 RTX 4090（`sections/results.tex`）。每个任务的训练时长见 Table 1a，超参见 Table 2–12：γ 取 0.96–0.985，lr 3e-4，episode 长 20–200 步。
- **数据**：不需要公开数据集。每个任务要准备 20–30 条 demo，以及约 200 个正样本、1000 个负样本的 classifier 帧。
- **最小复现建议**（我的建议）：(a) 有 Franka 的话，复现 RAM insertion：20 条 demo、10 Hz、100 步 episode、约 3.2 万条转移、1.5 h，目标是复现 Table 1a 的 100% 以及 Fig.5 中干预率降到 0。(b) 没有硬件的话，在 MuJoCo 仿真的插孔任务里用脚本专家模拟"人类干预"：策略偏离时由专家接管 $N$ 步。对比四种设置：RLPD 从零开始；加 20 条 demo；加 demo 和干预；干预数据只写入 RL buffer。这样可以检验数据去向的设计，以及"长时间干预导致 Q 高估"的说法。
- **难点**：真机时间与人力；每个任务都要定制相机裁剪、reset 脚本和 classifier；impedance 参数和 $\Delta$ 的整定；干预策略凭经验，论文无法完全传授。

## 自测问题

1. 为什么干预数据要同时写入两个 buffer，而策略转移只写入 RL buffer？在 50/50 采样下，这会怎样改变有效的数据分布？如果干预数据也只写入 RL buffer，你预期会发生什么？
2. 同样的纠错数据，HG-DAgger 拿去做监督回归，HIL-SERL 拿去做 RL。为什么后者能突破人类水平的 cycle time？$\gamma<1$ 在其中起什么作用？
3. 为什么"反复长时间接管直到成功"会导致 value 高估？结合 Eq.1 的 bootstrapped target 和策略自身实际能到达的状态来解释。
4. 推导 ego-centric 表征：用 $T_{b_0b_t}$ 和 adjoint 映射说明，为什么主板在执行中平移后 RAM 策略依然有效？在什么情况下这种不变性会失效？
5. 看 Table 1b：去掉干预后，HIL-SERL 在 RAM 上不如 IBRL。这对"HIL-SERL 的 RL 部分优于其他 demo-RL 方法"这一说法意味着什么？要把它说清楚，还缺哪个实验？
