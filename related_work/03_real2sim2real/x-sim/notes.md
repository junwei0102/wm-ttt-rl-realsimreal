# X-Sim: Cross-Embodiment Learning via Real-to-Sim-to-Real

> arXiv [2505.07096](https://arxiv.org/abs/2505.07096) · CoRL 2025（Oral；PDF 页脚标注 CoRL 2025，Oral 已在 OpenReview 核实；PMLR 305:816–833） · 主题：03_real2sim2real · 层级：frontier
> 代码：https://github.com/portal-cornell/X-Sim（已确认存在；项目页 https://portal-cornell.github.io/X-Sim/ ，见摘要） · 笔记依据：`source/*.tex`（主文件 `main.tex`，正文 `paper/{introduction,approach,experiments,limitations,appendix}.tex`，表在 `tables/`；数字/公式）+ `paper.pdf`（图）

## 一句话

从一段没有动作标签的 RGBD 人类视频里只取**物体的 6D 位姿轨迹**，用它在 2D Gaussian Splatting 重建出的照片级仿真中定义 object-centric reward，让机器人用 PPO 自己学出可行的动作；再把 privileged-state 策略蒸馏成 RGB Diffusion Policy 零样本部署到真实世界，最后用"真实 rollout 在仿真里 replay 得到的 sim/real 成对图像"做对比学习，校准观测编码器。整个流程不需要任何机器人 teleoperation 数据。最关键的想法是：人和机器人的动作不同，但它们对物体造成的运动是一样的。

## 问题设定

- **目标**：从 action-less 的人类视频训练真实世界的 image-conditioned 操作策略 $\pi_{\rm img}(a|o)$，输入单张 RGB，输出 7 维末端 delta action（3 平移 + 3 旋转 + 1 夹爪，`appendix.tex` Image-Conditioned Diffusion Policy Training）。
- **输入**：每个任务一段 RGBD 人类视频（ZED 2 立体相机，不限制动作和抓取方式；相机是否固定原文未说明）；被操作物体的 3D mesh（Polycam 手机扫描，每个物体 <1 分钟）；环境的多视角视频（<2 分钟）用于重建（`appendix.tex` Real-to-Sim Scans）。
- **假设**：刚体物体；物理参数用 ManiSkill 默认值，不做 system identification；测试时相机视角大致已知（`experiments.tex` Experimental Setup）；环境与物体尺度靠人工缩放对齐。
- **之前卡在哪**（`introduction.tex`、`related_work.tex`）：(1) 手部 retargeting（HaMeR 估手 → IK）假设人手的运动机器人能复现，实际常常在运动学或动力学上不可行；(2) 在人类视频上盖黑色 mask 或叠加机械臂（PHANTOM）解决的是视觉差距，动作映射仍然依赖 IK；(3) 需要 human-robot 成对数据的方法难以扩展；(4) 已有的 human video → sim → real 工作中，Human2Sim2Robot 部署时仍要在线跟踪物体位姿，而且需要人手轨迹引导 RL，只适用于与人手接近的灵巧手；Video2Policy 只用起止两帧的物体状态，部署还依赖分割 mask。

## 核心方法

三阶段流水线（`approach.tex`，Fig. 1–3）。

**① Real-to-sim：物体轨迹 + 场景重建**（Sec. 3.1，Fig. 2）。第一帧用 SAM 生成物体 mask，FoundationPose 根据 mesh 和 mask 逐帧跟踪 6D 位姿：
$$\mathbf{v_H}=\lbrace v_H^t\rbrace_{t=1}^T,\ v_H^t\in\mathbb{R}^{H\times W\times 4}\ \longrightarrow\ \mathbf{s_H}=\lbrace s_H^t\rbrace_{t=1}^T,\ s_H^t\in\mathrm{SE}(3)^K$$
其中 $K$ 是被跟踪物体数。环境用 2D Gaussian Splatting 重建出几何准确、外观逼真的 mesh，与物体一起导入 ManiSkill。

**② 仿真中 RL + 合成数据**（Sec. 3.2）。从 $\mathbf{s_H}$ 中采样 $N$ 个物体位姿 waypoint 作为子目标序列。对当前子目标 $s_H^B$（`approach.tex` Eq. 1）：
$$r_{\rm goal}\propto -d_{pos}(s_H^B,s_R^t)-d_{rot}(s_H^B,s_R^t)$$
附录给出实际实现（`appendix.tex` Reward Function Formulation）：
$$r_{\rm approach}=1-\tanh(k\thinspace d_{\rm obj}),\qquad r_{\rm goal}=\big(1-\tanh(\alpha_d d_{\rm pos})\big)+\big(1-\tanh(\alpha_\theta d_{\rm rot})\big)+2\thinspace i_{\rm waypoint}$$
$$r_{\rm obj}=r_{\rm approach}+r_{\rm goal}$$
$d_{\rm obj}$ 是末端到目标物体的距离， $d_{\rm pos}$ 是欧氏距离， $d_{\rm rot}$ 是四元数角度差； $\alpha_d,\alpha_\theta$ 按每个 waypoint 从示范中自动计算； $i_{\rm waypoint}$ 是当前 waypoint 序号，用作进度奖励。物体进入当前目标的 $\epsilon$ 邻域后，观测中的 `desired_goal` 切换到下一个 waypoint，所以多步任务能用同一套 reward 表达。另外还有 $r_{\rm static}$（物体到位后让机器人保持稳定）、 $r_{\rm success}$（完成 +1）和可选的 binary grasp 奖励。除了 $r_{\rm approach}$（末端靠近物体）和 $r_{\rm static}$（机器人保持稳定）这类通用引导项，任务目标项只由物体状态定义，与人手怎么动无关，机器人可以用完全不同的抓法和轨迹，这就是它能绕开 embodiment gap 的原因。

Privileged state（`appendix.tex` Table 2）：`ee_pose`、`gripper_width`、`achieved_goal`（当前物体位姿）、`desired_goal`、位置差、角度差、`is_grasped`。用 PPO 训练 $\pi_{\rm RL}(a|s)$，训练时随机化物体初始位姿。PPO 超参（Table 1）：lr $3\times10^{-4}$， $\gamma=0.8$，GAE $\lambda=0.9$，clip $\epsilon=0.2$，entropy 系数 0，target KL 0.1，minibatch 9,600，**1,024 个并行环境**，actor/critic 都是 3 层 256 的 Tanh MLP。

训练好之后 rollout $\pi_{\rm RL}$，**只保留成功轨迹**，渲染时随机化物体初始位姿（XY $\pm0.025$ m，旋转 $\pm\pi/8$）、机器人关节（ $\pm0.02$ rad）、相机位置与朝向目标点（各 $\pm0.03$ m）和 4 种光照预设（Table 3），每个任务收 500 条 $\xi_R=\lbrace(o_R^t,a_R^t)\rbrace_{t=1}^N$，组成 $D_{\rm synthetic}$。

**③ Sim-to-real：蒸馏 + 自动校准**（Sec. 3.3，Fig. 3）。在 $D_{\rm synthetic}$ 上用 BC 训练 Diffusion Policy $\pi_{\rm img}(a|o)$：只输入当前 RGB，预测动作序列 $\lbrace a_R^h\rbrace_{h=1}^H$。配置（Table 4）：ResNet18，图像 960×720 缩到 96×96，训练 100 个 / 推理 10 个 diffusion step，action horizon 2，prediction horizon 8，batch 64，lr $1\times10^{-4}$。

零样本部署后开始校准。在真实世界跑策略（实验里跑 10 条，**失败的也保留**），用 FoundationPose 从第一帧估出物体初始位姿，按这个状态初始化仿真，然后**原样 replay 真实机器人轨迹**，得到逐帧对应的 $D_{\rm paired}=\lbrace(o_R^{\rm sim},o_R^{\rm real})\rbrace_{i=1}^D$。接下来训练时在 $D_{\rm synthetic}$ 上用 BC loss，同时在 $D_{\rm paired}$ 上加 InfoNCE（`approach.tex` Eq. 2，原文如此）：
$$\mathcal{L}_ {\rm calibration}=-\negthinspace\negthinspace\sum_{(o_R^{\rm sim},o_R^{\rm real})\in D_{\rm paired}}\frac{\exp\big(s(\phi(o_R^{\rm sim}),\phi(o_R^{\rm real}))/\tau\big)}{\sum_{({\char"5F},o_R^{'\rm real})\in D_{\rm paired}}\exp\big(s(\phi(o_R^{\rm sim}),\phi(o_R^{'\rm real}))/\tau\big)}$$
$\phi$ 是策略的图像编码器， $s$ 是余弦相似度， $\tau$ 是温度。标准 InfoNCE（CPC）在分式外还有一个 $\log$，Eq. 2 写漏了，按常规应读作 $-\sum\log(\cdot)$。总损失 = 1 × action prediction loss + 0.1 × calibration loss（Table 4）。正样本是同一状态下的 sim/real 图像，负样本是 $D_{\rm paired}$ 中其他配对的真实图像 $o_R^{'\rm real}$（原文："images corresponding to different environment states"）。这样编码器被迫忽略渲染伪影这类只属于某个域的特征，BC 项则保证编码器仍保留预测动作需要的信息。配对靠 replay 得到，不需要动作标签，也不需要 rollout 成功，所以失败轨迹同样能用。

**训练 / 推理流程**：训练依次是扫描 → 跟踪 → PPO → 渲染 500 条 → 训 DP；部署时先零样本跑，收集 10 条 rollout，replay 配对，带 InfoNCE 重新训练，再部署。推理时只用真实 RGB 图像，不需要物体位姿。

## 关键实验与证据

平台：7-DOF Franka。两个环境：*Kitchen*（Mustard Place、Corn in Basket）和 *Tabletop*（Shoe on Rack、Letter Arrange、Mug Insert）。指标是 Average Task Progress，分三段计分（接近 → 抓取/旋转 → 达到人类视频中的目标配置），每个方法每个任务 10 次试验，物体初始位置只在示范附近小幅变化（`experiments.tex`）。

- **主结果（Fig. 4，从图中柱上标注读取）**：

| 任务 | Hand Mask | Object-Aware IK | X-Sim | X-Sim (calibrated) |
|---|---|---|---|---|
| Mustard Place | 0.23 | 0.80 | 0.93 | 0.93 |
| Corn in Basket | 0.27 | 0.87 | 1.00 | 1.00 |
| Shoe on Rack | 0.30 | 0.67 | 0.83 | 0.97 |
| Letter Arrange | 0.27 | 0.37 | 0.83 | 0.93 |
| Mug Insert | 0.27 | 0.30 | 0.57 | 0.70 |

我按图中数字算了五个任务的均值：Hand Mask 0.268、IK 0.602、X-Sim 0.832、Calibrated 0.906。摘要里的"平均提升 30%"（原文 "over hand-tracking and sim-to-real baselines"）论文没有给出计算口径；与之最吻合的是 Calibrated 相对 IK（约 30.4 个百分点），这是我的推断。未校准版本相对 IK 约 23 个百分点；正文 Sec. 4.1 的说法是未校准的 X-Sim 在最不匹配的设置中提升超过 30%。IK 在 Kitchen 里表现不错（0.80/0.87），因为这两个任务人和机器人的执行方式接近；到了 Tabletop 就掉下来，Fig. 5 给出的原因是找不到 IK 解，或者照搬人手动力学会导致碰撞。Hand Mask 基本停在"接近"阶段，因为训练时看到的人手和测试时的机械臂之间视觉差距太大。

- **校准的作用**：论文称平均 +8%，Mug Insert +13%（0.57→0.70）；按 Fig. 4 两位小数算平均是 7.4 个百分点，Mustard 和 Corn 上没有变化。Fig. 7 的 t-SNE 显示，同一条 rollout 的 sim/real embedding 在校准后对齐了，这是定性证据。
- **图像输入 vs 位姿输入（Fig. 6 / `tables/sim2real_table.tex`）**：Letter Arrange 上 Human2Sim2Robot（H2S2R，以物体状态为观测）43.3%，X-Sim 83.3%。作者的解释是 FoundationPose 在测试时的跟踪误差会把 state-based 策略推到 OOD。
- **数据效率（Fig. 8，Mustard Place 扩大初始分布，指标换成 success rate）**：X-Sim 用 3 段人类视频（每段 20 s，共 1 分钟）达到 90%；Diffusion Policy 用 10 条遥操作轨迹（每条 60 s，共 10 分钟）是 70%。从图上读，1 段视频约 60%，1/5 条机器人轨迹约 10%/40%。这里人类视频的初始状态比机器人数据稀疏得多，缺的覆盖靠仿真里扰动物体位姿来补（Fig. 11）。
- **视角泛化（Fig. 9 / `tables/viewpoint_table.tex`，Shoe on Rack）**：只用 Side 训练时 Side/Frontal/Novel 分别为 83.3/23.3/33.3%，只用 Frontal 为 23.3/76.7/30.0%，Side+Frontal 为 96.7/80.0/53.5%。单一视角的数据换到另一个视角基本失效，而在仿真里加视角几乎没有成本。

## 局限与论文没说的

**作者承认的**（`limitations.tex`）：需要物体 mesh（FoundationPose 依赖它）；只处理 6D 位姿能描述的刚体，铰接物体和可形变物体都做不了；需要显式扫描环境；物理参数取默认值，没有从视觉估计；离 in-the-wild 的互联网视频还很远。作者说论文的定位是"在理想 real-to-sim 前提下验证图像策略迁移"。

**我读出来的**：
- **评测口径**：每个任务 10 次试验，分三段计分，所以分数粒度是 1/30≈3.3%，而且没有误差棒或置信区间。校准在有变化的三个任务上带来 +10~14 个百分点（五任务平均 7.4），很可能在噪声范围内。Novel 视角的 53.5% 不是 1/30 的整数倍（53.3% 才是），疑似笔误。测试分布也窄：只比示范初始位置稍有扰动，仿真随机化范围只有 ±2.5 cm / ±π/8。
- **基线偏弱**：Hand Mask 没有叠加机械臂，所以不是完整的 PHANTOM；Object-Aware IK 只是用 IK 重放（replay）相对物体的手部轨迹，是否有闭环修正原文未说明。H2S2R 只比了一个任务，复现细节也没交代。
- **"online / test-time" 的实际含义**：做法是收 10 条 rollout 后离线再训练一次。论文没说迭代了几轮，没说是 fine-tune 还是从头训，也没有对 rollout 数量、loss 权重、 $\tau$ 做消融（ $\tau$ 的取值本身也未报告）。更关键的是 replay 配对的前提：机器人碰到物体之后，默认物理下仿真里的物体运动可能与真实分叉，后半段的"正样本对"其实并不对应。论文没有分析这一点。
- **只对齐了视觉**：dynamics gap 完全没处理。Mug Insert 这种需要精度的任务最差（0.70），可能就是接触动力学或跟踪误差造成的上限。
- **蒸馏是纯 BC**：只用成功轨迹，没有 DAgger，student 学不到从 teacher 状态分布之外恢复的能力（对比 [RialTo](../rialto/notes.md) 的 DAgger + co-training）。
- **reward 的细节没有公开数值**： $k,\alpha_d,\alpha_\theta,\epsilon,N$ 以及 $r_{\rm static}$ 的形式都没给； $\gamma=0.8$ 很短视，要靠稠密的 waypoint 奖励撑住。
- **算力未报告**：GPU 型号、PPO 训练时长、DP 训练时长都没写。
- **"cross-embodiment" 的范围**：只测了人 → Franka 一种迁移，没有换别的机器人。

## 在脉络中的位置

- **前驱**：[Domain Randomization](../domain-randomization/notes.md)（视角和光照随机化）、[SimOpt](../simopt/notes.md)（被引为可兼容的随机化方法）；[RialTo](../rialto/notes.md)（real-to-sim-to-real + RL，但需要真实示范，策略用点云）；[Digital Cousins](../acdc-digital-cousins/notes.md)（在 limitations 中被引为 mesh 检索的思路）；[PPO](../../04_rl/ppo/notes.md)；privileged teacher → 视觉 student 的做法与 [RMA](../rma/notes.md) / [legged-gym](../legged-gym/notes.md) 同源（X-Sim 未引用这两篇）；此外还有 FoundationPose、2DGS、ManiSkill、Diffusion Policy、InfoNCE，以及人类视频方向的 PHANTOM、OKAMI、Human2Sim2Robot、Video2Policy。
- **后继 / 影响**：[reality gap 综述](../survey-reality-gap/notes.md) 把它归在 "Real-to-Sim First" 类（`chapters/solutions.tex`）。与 [SplatSim](../splatsim/notes.md) 相比，它在 GS 重建场景中补上了 RL 这一环。作者在 Discussion 中提出下一步是把这条流水线用于 fine-tune π0.5 / OpenVLA 等 foundation model。
- **与本项目其他主题的联系**：
  - *World model*：X-Sim 的"simulator"由物理引擎和 2DGS 外观拼成，能直接给出物体位姿作为 reward，接触由引擎计算；代价是需要 mesh 和扫描。另一条路是用 learned world model 当 simulator 训练 / 改进策略：[RWM](../../05_intersections/robotic-world-model/notes.md)、[World-Gymnast](../../05_intersections/world-gymnast/notes.md)、[WMPO](../../05_intersections/wmpo/notes.md)、[Ctrl-World](../../05_intersections/ctrl-world/notes.md)，这类方法不需要 mesh，但 reward 要么从预测的观测里计算（RWM），要么依赖另训的 reward model（WMPO、World-Gymnast）或人工筛选 rollout（Ctrl-World），还有幻觉问题。[SimDist](../../05_intersections/simdist/notes.md) 同样是"先用仿真大量造数据，再用少量真实数据适应"的结构。从人类视频迁移这一点，可以对照 [DreamZero](../../01_world_model/dreamzero/notes.md)。
  - *TTT / TTA*：校准本质是"部署后用无标签真实观测加自监督信号更新共享编码器，同时保留主任务 loss"，结构上接近 [TTT](../../02_test_time_training/ttt-self-supervised/notes.md) 的 joint training，也接近 [PAD](../../05_intersections/pad/notes.md) 的部署期自监督适应。区别在于它的自监督信号来自仿真 replay 构造的跨域正样本对，而且是批量再训练，不是逐样本在线更新；与 [Tent](../../02_test_time_training/tent/notes.md) 的熵最小化也不同。
  - *Real2sim2real*：[SimOpt](../simopt/notes.md) 用真实 rollout 闭环修正**物理参数**，X-Sim 用真实 rollout 闭环修正**视觉编码器**，两者互补。作者也在 limitations 里提出把同一框架扩展到物理参数；这方面可以参考 [ASAP](../asap/notes.md) 的 delta action 和 [PhysTwin](../phystwin/notes.md) 的可形变物体。[SIMPLER](../simpler/notes.md) 则是把 real-to-sim 用于评测。
  - *RL*：用 GPU 并行 PPO 当"动作标签生成器"。student 是 Diffusion Policy，所以后续可以接 [DPPO](../../04_rl/dppo/notes.md) 直接对 student 做 RL fine-tune；VLA 在仿真里做 RL 的方向可以参考 [SimpleVLA-RL](../../04_rl/simplevla-rl/notes.md) 和 [π_RL](../../04_rl/pi-rl/notes.md)。

## 前置知识

1. PPO + GAE、reward shaping（稠密奖励、带子目标切换的 goal-conditioned 设定）
2. Privileged teacher → vision student 蒸馏（behavior cloning）
3. Diffusion Policy（DDPM 生成 action chunk，action / prediction horizon 的区别）
4. Domain randomization，以及 visual gap 与 dynamics gap 的区分
5. 对比学习 InfoNCE（正负样本构造、温度 $\tau$）
6. 6D 位姿跟踪（FoundationPose，需要 mesh）与 SAM 分割
7. Gaussian Splatting / 2DGS 的 3D 重建
8. 手部 retargeting / IK 基线的做法（HaMeR → IK），用于理解作者在对比什么

## 复现可行性

- **代码**：https://github.com/portal-cornell/X-Sim 已开源。README 覆盖 real-to-sim、RL 训练、合成数据生成、DP 训练、auto-calibration 数据生成、校准训练六个步骤，内置修改过的 ManiSkill（任务在 `simulation/ManiSkill/mani_skill/envs/tasks/xsim_envs`），包含全部 5 个任务，README 徽章标注 Python 3.10，**没有写算力和训练时长**（WebFetch 读取 README）。
- **算力**：论文和 README 都没给 GPU 型号、数量或时长。已知的只有 1,024 个并行环境（Table 1）、每任务 500 条渲染轨迹（`appendix.tex`），以及 96×96 输入的 ResNet18 DP。按这个规模估计，单卡应该能跑，但这是我的推断，论文没有给出。
- **硬件与数据**：真实实验需要 Franka、ZED 2（RGBD）和手机扫描；README 要求用户自己扫描环境（2DGS）和物体（Polycam）；setup 脚本会下载 assets，但 README 没说明其中是否包含这 5 个任务的扫描场景和物体 mesh。
- **最小可复现实验**（纯仿真，不需要机器人）：(1) 在 repo 自带的 Mustard Place 或 Shoe on Rack 环境里用 object-centric reward 训练 PPO，检查成功率曲线，并去掉 $2i_{\rm waypoint}$ 或 $r_{\rm approach}$ 看 reward 各项的作用；(2) 渲染 500 条轨迹训练 DP，在仿真中复现 Fig. 9 的 Side / Frontal / 两者都用 → Novel 视角的趋势；(3) 另造一个"伪真实"域（换光照、纹理或渲染器）来模拟 sim-to-real，比较有无 InfoNCE 校准时的成功率。第 (3) 步是代理实验，不是论文原设定。
- **预计难点**：FoundationPose 在手遮挡下的跟踪质量；2DGS mesh 导入 ManiSkill 时的人工缩放和相机外参对齐；未公开的 reward 系数（ $k,\alpha,\epsilon,N$）需要照代码抄；replay 配对要求 sim 和 real 的相机与机器人控制严格一致。

## 自测问题

1. object-centric reward 为什么能绕开 embodiment gap？在 Tabletop 任务上 Object-Aware IK 会在哪几种情况下失败，而 PPO 在仿真里是怎么避开这些失败的？
2. 既然仿真里已经有 privileged-state PPO 策略，为什么不在真实世界配上 FoundationPose 直接部署它，而是要蒸馏成 RGB 策略？请用 H2S2R 43.3% vs 83.3% 的对比来解释。
3. 校准时正样本和负样本分别怎么构造？为什么失败的 rollout 也能用？如果机器人接触物体后仿真与真实的物体运动开始分叉，InfoNCE 会学到什么？为什么还要同时保留 $D_{\rm synthetic}$ 上的 BC loss？
4. $r_{\rm goal}$ 里的 $2i_{\rm waypoint}$ 和"物体到达 $\epsilon$ 邻域就切换 desired_goal"这两个设计分别解决什么问题？ $\epsilon$ 设得太大或太小会各自出现什么行为？
5. 数据效率实验里，3 段人类视频为什么能胜过 10 条遥操作轨迹？这个优势依赖哪些假设（物体 mesh、场景扫描、默认物理足够准）？
