# SplatSim: Zero-Shot Sim2Real Transfer of RGB Manipulation Policies Using Gaussian Splatting

> arXiv [2409.10161](https://arxiv.org/abs/2409.10161) · ICRA 2025（已由 Crossref 核实，DOI 10.1109/ICRA55743.2025.11128339；PDF v3（2024-10-07）正文没有标 venue。另在 CoRL 2024 MRM-D workshop 发表，OpenReview 记为 oral，项目页写作 spotlight） · 主题：03_real2sim2real · 层级：milestone
> 代码：https://github.com/qureshinomaan/SplatSim（项目页给的官方仓库，2026-09-27 已确认存在） · 笔记依据：`source/sections/*.tex`（数字/公式）+ `paper.pdf`（Fig. 1–5）

## 一句话

SplatSim 让 PyBullet 只管物理，把真实场景的 3D Gaussian Splatting 重建按刚体（机器人各 link、夹爪各 link）切开，每个物体另有自己的 splat $\mathcal{S}^k_{obj}$。每一步用仿真给出的位姿对这些 Gaussian 做刚体变换（ $\mu'=R\mu+t,\ \Sigma'=R\Sigma R^\top$），再渲染成照片级 RGB。这样生成的数据只用来训练 Diffusion Policy，训完不做任何微调，直接部署到真实机器人。4 个任务的平均成功率是 86.25%，用真实数据训练的 policy 是 97.5%。

## 问题设定

- **目标**：RGB-only 的操作 policy 完全在仿真数据上训练，然后 zero-shot 部署到真实世界。作者把 vision sim-to-real 看成一个 out-of-domain generalization 问题：仿真图像和真实图像的分布差得太远（`1_intro.tex`）。
- **输入**：
  - $\mathcal{S}_ {real}$：用多视角 RGB（"一段静态场景视频"）重建的真实场景 Gaussian Splat，场景里包含处于 home pose 的机器人（Fig. 3 caption）。
  - $\mathcal{S}^k_{obj}$：第 $k$ 个物体单独多视角扫描得到的 splat。
  - 仿真器（PyBullet）：提供机器人 URDF/CAD 和物体的仿真点云。
  - 专家 $\mathcal{E}$：在仿真里用 Gello 遥操作，或者用读取特权状态的 motion planner。
- **形式化**（`3_method.tex` §Problem Statement）：状态 $s_t=(q_t,x^1_t,\dots,x^n_t)$，其中 $q_t\in\mathbb{R}^m$ 是关节角， $x^k_t=(p^k_t,R^k_t)\in\mathbb{R}^3\times SO(3)$ 是物体位姿。动作 $a_t=(p^e_t,R^e_t)$ 是末端执行器位姿。仿真轨迹 $\tau_\mathcal{E}=\lbrace(s_t,a_t)\rbrace$ 被渲染成 $\tau_\mathcal{G}=\lbrace(I^{sim}_ t,a_t)\rbrace$，用来训练 $\pi_\mathcal{I}$。测试时 policy 只看真实 RGB $I^{real}$（按 Fig. 2，还输入末端位姿）。
- **隐含假设**：所有交互物体都是刚体；部署场景就是扫描过的那个场景（背景、光照不变）；仿真坐标系、真实世界坐标系和机器人基座坐标系对齐（ $\mathcal{F}_ {sim}$、 $\mathcal{F}_ {robot}$ 都与 $\mathcal{F}_ {real}$ 对齐）；不需要真实世界的 reward 或演示。
- **之前的方法卡在哪**（`2_related_works.tex`）：
  - 深度、点云、触觉这类 sim-to-real gap 小的模态已经能 zero-shot 迁移，RGB 却很少作为主模态。
  - RGB 路线用 domain randomization，需要逐任务调参和环境工程。
  - Domain adaptation（RL-CycleGAN）需要大量真实交互数据。
  - RialTo、Maniwhere 部署时仍然需要深度或点云。
  - Embodied Gaussians 从真实数据学 forward model，每换一个机器人或物体都要重新采数据。
  - Robo-GS（正文称 RoboStudio，引用的是 Robo-GS 论文）做的是 system identification。
  - Quach et al. 把 GS 和仿真器结合，但做的是导航，agent 不和物体交互。

## 核心方法

核心前提（§Method 开头）：只要真实场景 splat 里的**每个刚体都能被分割出来**，并且知道它**相对仿真器的齐次变换**，就可以在任意新位姿下渲染它。做法就是把仿真器里的 mesh 渲染整个换成 Gaussian 渲染。

**1. 刚体变换下的 Gaussian**（`3_method.tex` Eq. 1–2）。给定 $T=(R,t)$，对刚体上每个 Gaussian 做：

$$\mu'=R\mu+t,\qquad \Sigma'=R\thinspace\Sigma\thinspace R^\top$$

论文只给出 $\mu$ 和 $\Sigma$ 的变换，没有提 opacity 和颜色（SH 系数）怎么处理。刚体上所有 Gaussian 共用同一个 $T$。

**2. 坐标系**（§IV-B）。 $\mathcal{F}_ {splat}$ 是 splat 里机器人基座所在的坐标系，与仿真坐标系不同。 $T^{\mathcal{F}_ {splat}}_ {\mathcal{F}_ {robot}}$ 把 splat 坐标变到仿真坐标。每个物体还有两个局部坐标系： $\mathcal{F}_ {k\text{-}obj,sim}$（物体在仿真中初始化于原点，无旋转）和 $\mathcal{F}_ {k\text{-}obj,splat}$。

**3. Robot Splat Model**（§IV-C，Fig. 3），分三步：

- (a) 在 $\mathcal{S}_ {real}$ 里**手动**分割出属于机器人的 Gaussian，把它们的均值当点云，与仿真器给出的 ground-truth 点云做 **ICP**，得到 $T^{\mathcal{F}_ {splat}}_ {\mathcal{F}_ {robot}}$。
- (b) 用 CAD 模型给出的各 link bounding box，把 Gaussian 分到各个 link，得到 $\mathcal{S}^l_{real}$。
- (c) 用 PyBullet 的 forward kinematics 求出 link $l$ 在 $q_t$ 下的变换 $T^l_{fk}$，再按下式（`3_method.tex` Eq. 3）变换该 link 的 Gaussian：

$$T=\big(T^{\mathcal{F}_ {splat}}_ {\mathcal{F}_ {robot}}\big)^{-1}\cdot T^{l}_ {fk}\cdot T^{\mathcal{F}_ {splat}}_ {\mathcal{F}_ {robot}}$$

白板上的读法：这是一个**共轭**。先把 Gaussian 从 splat 系搬到仿真系，在仿真系里施加 link 的运动，再搬回 splat 系，这样新位姿的 link 仍然嵌在原来的场景 splat 里渲染。我的理解（论文记号里省略了）：扫描时机器人处在 home pose，所以这里的 $T^l_{fk}$ 实际上应当是相对 home 构型的运动，即 $FK_l(q_t)\thinspace FK_l(q_{home})^{-1}$；否则 $q_t=q_{home}$ 时得不到恒等变换。

**4. Object Splat Model**（§IV-D，Eq. 4）。物体单独扫描得到 $\mathcal{S}^k_{obj}$，与仿真物体点云做 ICP，得到 $T^{\mathcal{F}_ {k\text{-}obj,splat}}_ {\mathcal{F}_ {k\text{-}obj,sim}}$（§IV-B 里同一矩阵的上下标写反了，这里按 §IV-D 和 Eq. 4 的写法）。再由仿真状态 $(p^k_t,R^k_t)$ 构造 $T^{k\text{-}obj}_ {fk}$：

$$T=\big(T^{\mathcal{F}_ {splat}}_ {\mathcal{F}_ {robot}}\big)^{-1}\cdot T^{k\text{-}obj}_ {fk}\cdot T^{\mathcal{F}_ {k\text{-}obj,splat}}_ {\mathcal{F}_ {k\text{-}obj,sim}}$$

这条链从右往左读：物体 splat 系 → 物体在仿真中的规范系 → 仿真世界中第 $t$ 步的位姿 → 场景 splat 系。

**5. 铰接体（夹爪）**（§IV-E，Fig. 4）。平行夹爪的各 link 和坐标轴不对齐，用 bounding box 分不干净。作者用 URDF 标好的仿真点云训练一个 **KNN classifier**，对对齐后 splat 中的每个 Gaussian 预测所属 link。

**6. 渲染整条轨迹**（§IV-F）。对每个 $t$，把所有刚体按上面的公式变换，再用标准 3DGS rasterizer 渲染，得到 $\tau_\mathcal{G}$。

**7. Policy 训练与部署**（§IV-G，Fig. 2）。使用 Diffusion Policy，输入是两路相机的 RGB 加末端位置和姿态，末端状态也做增强。图像增强参照 NeRF2Real：高斯噪声、random erasing、亮度/对比度（color jitter）。增强针对的是残余的外观差异：仿真场景没有阴影；机器人线缆这类柔性部件按刚体渲染会出错；反射会变化。部署时 policy 冻结，直接接收真实 RGB。

**流程总结**：
- 离线，每个场景一次：扫描 → 训练 3DGS → 手动分割 + ICP 对齐 + link 分割。
- 数据生成：专家在 PyBullet 里产生状态轨迹 → 逐帧变换 Gaussian 并渲染。
- 训练：纯仿真数据的 behavior cloning。
- 推理：真实 RGB + 末端位姿 → 动作。

整个方法没有 RL，不对渲染求梯度，3DGS 本身也不重新优化，只对它做刚体搬运。

## 关键实验与证据

- **硬件**（§V-B）：UR5 + Robotiq 2F-85 夹爪 + 2 台 Intel RealSense D455。Diffusion Policy 部署在 NVIDIA RTX 3080Ti 上。
- **数据**（§V-B 各小节）：
  - T-Push：人在仿真里用 Gello 采了 160 条演示。
  - Pick-Up-Apple：privileged motion planner 生成 400 条，末端位置和姿态随机化。
  - Orange-On-Plate：planner 生成 400 条，末端位置和初始夹爪状态随机化，测试时总是从 home 出发。
  - Assembly：机器人从 home 出发且已经抓着绿块，要叠到红块上。论文**没给演示条数**。
  - 真实世界的演示条数没有报告。
- **主结果**（Table I，每个设置 40 次试验）：

| 任务 | Sim2Sim | Real2Real | Sim2Real (SplatSim) | 人工时（仿真 / 真实，h） |
|---|---|---|---|---|
| T-Push | 100% | 100% | 90% | 3.0 / 3.5 |
| Pick-Up-Apple | 100% | 100% | 95% | 0.0* / 3.5 |
| Orange-On-Plate | 97.5% | 95% | 90% | 0.0* / 6.0 |
| Assembly | 85% | 90% | 70% | 0.0* / 7.5 |
| Total | 95.62% | 97.5% | **86.25%** | 3.0 / 20.5 |

  \* 表示自动生成（motion planner）。Sim2Sim 指在 SplatSim 里训练，也在 SplatSim 里评测，用来量化迁移时掉了多少。
- **读法**：
  - T-Push、Pick-Up-Apple、Orange-On-Plate 从 Sim2Sim 到 Sim2Real 只掉 5–10 个点（分别掉 10、5、7.5）。
  - 对放置精度敏感的 Assembly 从 85% 掉到 70%，掉得最多。
  - 数据成本上，仿真约 3 h，真实约 20.5 h；仿真里只有 T-Push 需要人参与。
- **渲染保真度**（§V-C）：在 300 组关节构型下，把渲染的机器人图像和真实图像对比，平均 PSNR 22.62，SSIM 0.7845。**没有**报告 mesh 渲染的对照数字。
- **消融：数据增强**（§V-D）：4 个任务平均，不加增强时 Sim2Real 成功率是 21%，加上后是 86.25%，没有逐任务的拆分。这说明**光有 3DGS 不够**：阴影、反射、线缆这类残余外观差异，必须靠训练时的增强来吸收。
- **论文内部数字不一致**（我读出来的，需要注意）：
  - Assembly 的 Sim2Sim 在 Table I 里是 85%，§V-B 正文写的是 95%。表中 Total 95.62% 与 85% 一致，所以正文的 95% 应该是笔误。
  - Real2Real 四项按表内数字平均是 $(100+100+95+90)/4=96.25$%，和表中以及摘要里的 97.5% 对不上。

## 局限与论文没说的

- **作者承认的**：只能处理刚体，处理不了布料、液体、植物（§VI）；仿真场景没有阴影，线缆按刚体处理（§IV-G）；渲染表达不了动态的反射和阴影（§V-D）。
- **缺少关键 baseline**：没有"PyBullet mesh 渲染 + domain randomization"的 Sim2Real 对照，所以论文并没有直接分离出"3DGS 比 mesh 渲染好多少"。唯一的对照是作为上界的 Real2Real。
- **"zero-shot"的口径**：不需要真实演示，但需要目标场景的扫描。所有测试都在同一个扫描过的场景里做：没有测光照变化、背景变化、相机位移或新物体，本质上是一个特定场景的 digital twin。
- **视觉之外的差距没处理**：论文没有说明 PyBullet 的物理参数怎么设定（原文未报告），没有 system identification，也没有对动力学做 domain randomization。Assembly 掉 15 个点，外观原因和接触/摩擦原因分不开，论文也没有设计实验去分。
- **人工步骤没算进成本**：手动分割机器人、ICP 对齐（README 用 CloudCompare 辅助）、逐个扫描物体，这些时间都不在 Table I 的"Human Effort"里。
- **未说明的细节**：
  - 虚拟相机和两台真实相机的外参怎么对齐，没写。
  - Gaussian 旋转后，spherical harmonics（view-dependent color）是否也跟着旋转，没写；论文只写了 $\mu$ 和 $\Sigma$ 的变换。
  - 扫描时 home pose 下被遮挡的 link 表面，转到新构型后会露出来，那里可能重建得不好。PSNR 22.62 不算高，Fig. 4(b) 里夹爪的 KNN 分割也明显有噪声。
- **评测口径**：
  - 每个设置只有 40 次试验，没有置信区间，也没有多随机种子。
  - 各任务初始状态的随机化范围描述得很粗。
  - 真实和仿真的演示数量不一样（真实的条数未知），所以 Real2Real 与 Sim2Real 不是等数据量的比较。
- **算力**：policy 训练用的 GPU 和时长、3DGS 重建时间、渲染吞吐都没报告。渲染吞吐决定了它能不能用来做大规模 RL，作者把 RL 列为 future work。

## 在脉络中的位置

- **前驱**：
  - 3D Gaussian Splatting（Kerbl et al. 2023）。
  - [Domain Randomization](../domain-randomization/notes.md)：RGB sim-to-real 的"鲁棒化"路线，SplatSim 走的是相反方向，追求"逼真化"。
  - NeRF2Real：用 NeRF 当视觉仿真器，SplatSim 沿用了它的增强方案。
  - [RialTo](../rialto/notes.md)：同样是 real2sim2real，但 policy 输入是点云。
  - Diffusion Policy；Gello 遥操作。
  - 同期的 GS + 机器人工作：Embodied Gaussians、Robo-GS、Quach et al.。
- **后继 / 影响**：
  - `index.json` 把 GaussGym、PolaRiS、TwinRL 列为以它为基础的工作（不在本项目，本笔记未核实）。
  - 本项目内，[X-Sim](../x-sim/notes.md) 用 2D Gaussian Splatting 重建环境导入 ManiSkill，在重建场景里训练 RL（`x-sim/source/paper/approach.tex`），正好接上了 SplatSim 留给 future work 的"加 RL"。
  - [PhysTwin](../phystwin/notes.md) 用 spring-mass 加 Gaussian 渲染处理可形变物体，补的是 SplatSim 的"仅限刚体"这一局限。
  - [World model 综述笔记](../../01_world_model/survey-wm-robot-learning/notes.md) 把 DreMa 归为同一思路（GS 场景 + 物理仿真器）。
- **与本项目其他主题的联系**：
  - **World model**：SplatSim 可以看成一个"结构化、手工指定动力学"的 world model：外观来自重建，动力学来自物理引擎。学出来的 simulator（[UniSim](../../05_intersections/unisim/notes.md)、[Cosmos](../../01_world_model/cosmos-wfm/notes.md)、[Ctrl-World](../../05_intersections/ctrl-world/notes.md)）则把两者一起学。两边的取舍是：SplatSim 物理上一致、支持任意状态，但只能处理刚体和固定场景；学出来的模型覆盖面广，但会产生幻觉。
  - **real2sim 评测**：Sim2Sim 与 Real2Real 在各任务上的高低关系大体一致，可以和 [SIMPLER](../simpler/notes.md) 的 visual matching 评测对照；[ACDC](../acdc-digital-cousins/notes.md) 代表"要 cousin 不要 twin"的另一极。
  - **TTT / TTA**：残余外观差异目前靠训练时的增强处理。另一条路是部署期用无标签目标调视觉编码器（[PAD](../../05_intersections/pad/notes.md) 用 self-supervised auxiliary task，[Tent](../../02_test_time_training/tent/notes.md) 用熵最小化优化 BatchNorm 的仿射参数，统计量用 target batch 重估）。[LaCT](../../02_test_time_training/lact/notes.md) 的前馈 novel view synthesis 可以看作逐场景优化 3DGS 的替代。
  - **RL**：现在是纯 BC。如果把 SplatSim 渲染接到 [PPO](../../04_rl/ppo/notes.md) 这类在线 RL，或者接到 VLA 的 RL 微调（[SimpleVLA-RL](../../04_rl/simplevla-rl/notes.md)、[π_RL](../../04_rl/pi-rl/notes.md)）上，瓶颈会是渲染吞吐，而论文没有报告这个数。

## 前置知识

1. 3D Gaussian Splatting：Gaussian 的参数（ $\mu$、 $\Sigma=RSS^\top R^\top$、opacity、SH 颜色）和 tile-based rasterization。
2. 齐次变换、坐标系链式变换、共轭变换 $A^{-1}BA$ 的几何含义。
3. 机器人 forward kinematics 和 URDF。
4. ICP 点云配准。
5. Behavior cloning 和 Diffusion Policy（action chunk、观测条件化）。
6. sim-to-real gap 的分类（视觉 vs 动力学）；domain randomization 和 system identification。
7. PSNR、SSIM；KNN 分类。

## 复现可行性

- **代码**：[qureshinomaan/SplatSim](https://github.com/qureshinomaan/SplatSim) 已开源。以下依赖信息来自 README，不是论文：Python 3.12、PyTorch 2.7.1 + CUDA 12.6（README 给的一个示例配置）、PyBullet、Gaussian splatting 封装、simple-KNN、GELLO、COLMAP、CloudCompare；README 提供 COLMAP 数据、训练好的 Gaussian splat（.ply）和 GELLO 演示轨迹的下载，以及轨迹录制和回放脚本。没有提供 Diffusion Policy 的 checkpoint，"在 SplatSim 里可视化训练好的 policy"一节还写着 TODO。
- **算力**：论文只写了部署用 RTX 3080Ti（§V-B）。训练 GPU、训练时长、3DGS 重建时间都**未报告**。单场景 3DGS 加一个 Diffusion Policy，单张消费级 GPU 应该可以承担，但这是推测，不是论文里的数。
- **硬件与数据**：完整的 Sim2Real 复现需要 UR5 + Robotiq 2F-85 + 2×D455，还要自己扫描场景。没有真机时只能复现 Sim2Sim 和渲染部分。
- **最小可复现实验**：
  1. 用 README 提供的训练好的 Gaussian splat，在随机关节角下渲染机器人，检查 Eq. 3 的共轭变换是否正确（特别是 $q=q_{home}$ 时应该与原 splat 重合），并肉眼检查夹爪的 KNN 分割。
  2. Pick-Up-Apple 的 Sim2Sim：用 planner 生成 400 条演示，渲染后训练 Diffusion Policy，在 SplatSim 里评测 40 次，与 Table I 的 100% 对比。
  3. 补作者缺的对照（我的建议）：同一套演示分别用 PyBullet mesh 渲染和 SplatSim 渲染来训练，都在 SplatSim 渲染上评测（把 SplatSim 当作"伪真实"），同时做加/不加增强两组，粗略拆开"3DGS 带来的收益"和"增强带来的收益"。
- **预计难点**：
  - ICP 需要好的初值，README 用 CloudCompare 手动辅助。
  - 虚拟相机和真实相机外参的对齐，论文没写。
  - 线缆、阴影这类伪影。
  - Diffusion Policy 的超参数（观测步长、action horizon、训练轮数）论文里都没写，只能去代码里找。

## 自测问题

1. Eq. 3 为什么写成 $(T^{\mathcal{F}_ {splat}}_ {\mathcal{F}_ {robot}})^{-1}T^l_{fk}T^{\mathcal{F}_ {splat}}_ {\mathcal{F}_ {robot}}$ 这种共轭形式？扫描时机器人处在 home pose，这对 $T^l_{fk}$ 的实际含义意味着什么？
2. Eq. 4 的三段变换分别把 Gaussian 从哪个坐标系搬到哪个坐标系？为什么物体要单独扫描，而不直接从 $\mathcal{S}_ {real}$ 里分割出来？
3. 为什么只变换 $\mu$ 和 $\Sigma$ 就能在新位姿下渲染？哪些外观效应（阴影、反射、view-dependent color、柔性线缆）会因此出错？"21% → 86.25%"这个消融说明了什么？
4. 机械臂的 link 用 CAD bounding box 分割，夹爪却要用 KNN classifier，原因是什么？
5. SplatSim 缩小的是哪一类 sim-to-real gap？Assembly 从 Sim2Sim 的 85% 掉到 Sim2Real 的 70%，你会设计什么实验来判断这是外观问题还是物理（接触、摩擦）问题？
