# Reconciling Reality through Simulation: A Real-to-Sim-to-Real Approach for Robust Manipulation (RialTo)

> arXiv [2403.03949](https://arxiv.org/abs/2403.03949) · RSS 2024 · 主题：03_real2sim2real · 层级：milestone
> 代码：https://github.com/real-to-sim-to-real/RialToPolicyLearning（项目页 https://real-to-sim-to-real.github.io/RialTo/ ；场景构建 GUI 在独立的 RialToGUI repo，据 README）· 笔记依据：`source/*.tex`（数字/公式）+ `paper.pdf`（图）
> 源码结构：主文件 `source/paper_template.tex`，正文在 `source/texs/{intro2,method_as,experiments,ablations,userstudies,conclusion,appendix}.tex`（`texs/rebuttal.tex` 没有被 `\input`，不属于论文正文，本笔记不引用）。

## 一句话

用手机扫描 + GUI 在 15 分钟左右的人工操作内把目标场景做成带铰接的 digital twin，然后把"约 15 条真实示范学到的点云模仿策略"放进孪生里跑出带 privileged state 的仿真示范，以此为 BC 数据，用 **PPO + BC loss** 训练 state-based 策略（论文称之为对模仿策略做 RL fine-tuning / robustify；被 RL 训练的是新的 state-based 网络，不是点云策略本身），再用 teacher-student 蒸馏（DAgger + 真实示范 co-training）回到点云策略部署到真实世界。最关键的想法是 **"inverse distillation"**：把只有点云的真实示范，借"在仿真里跑真实策略"变成带 privileged state 的仿真示范，从而能在低维 state 上高效做 RL。

## 问题设定

- **目标**：为**一个特定部署场景**（某个厨房、某个书架）得到对物体位姿变化、视觉干扰物（distractors）、物理扰动（disturbances）都鲁棒的 visuomotor 操作策略，且人力成本低。
- **输入/输出**：真实世界观测 $o$ 是单个标定深度相机的 3D 点云（+ 机器人末端状态），动作 $a$ 是离散化的末端 delta pose：14 个动作 = 6 个平移（每轴 $\pm0.03$ m）+ 6 个旋转（每轴 $\pm0.2$ rad）+ 2 个开/合夹爪动作（`appendix.tex` Task details）。
- **假设**：只有少量真实示范（ $\sim$ 15 条，`method_as.tex` System Overview）；任务可在刚体/铰接刚体仿真中表达；有一个人工写的**稀疏成功判据**（如 `drawer_joint > 0.1 && gripper_open`）同时用作成功筛选和 RL 奖励；物理参数**不做 system identification**，所有物体统一默认值（摩擦 0.5、关节摩擦 0.1、质量 0.41 kg，`appendix.tex` Simulation details）。
- **之前卡在哪**：(1) 纯 imitation learning（BC）少量示范下对扰动很脆，且学不到示范里没出现的恢复行为（重抓、纠偏）；(2) 真实世界 RL 慢、不安全、需要 reset 和奖励设计；(3) sim-to-real 需要手工搭几何/视觉/运动学都逼真的场景，成本高；而在大量多样化合成资产上训 generalist 又难又不一定对某个具体场景好；(4) 真实示范没有 Lagrangian state（物体位姿），不能直接拿来引导 state-based RL。

## 核心方法

四步流水线（`method_as.tex` Sec. III-A，Fig. 1）：

**① Real-to-sim 场景构建**（`method_as.tex` Sec. III-B，Fig. 2）。用 Polycam（大场景，iPhone LiDAR）、AR Code（360° 单物体）或 NeRFStudio `nerfacto` + Poisson 表面重建（细薄结构如碗架）得到整体网格 $\mathcal G$；在 GUI 里切分物体、添加关节，得到
$$\mathcal S=\lbrace\lbrace\mathcal G_i\rbrace_{i=1}^{M},\ \mathcal K,\ \mathcal P\rbrace$$
（几何、运动学、物理参数），导出为 USD/URDF 进 Isaac Sim。碰撞体默认 convex decomposition（64 hull vertices、32 convex hulls），碗架用 SDF（分辨率 256）。物理参数刻意不辨识：dynamics gap 交给后面"靠近真实示范"的约束去补偿。

**② Inverse distillation：real→sim 的示范迁移**（`method_as.tex` Sec. III-C1，Fig. 3）。
1. 在 $\mathcal D_{\text{real}}=\lbrace(o_1^i,a_1^i),\dots,(o_H^i,a_H^i)\rbrace_{i=1}^N$ 上用 BC 训点云策略 $\pi_{\text{real}}(a|o)$；
2. 把 $\pi_{\text{real}}$ 放进孪生里，用**仿真渲染的点云**跑 rollout，只保留满足成功判据的轨迹，得到 $\mathcal D_{\text{sim}}=\lbrace(o_t^i,a_t^i,s_t^i)\rbrace$，其中 $s$ 是仿真直接给出的 privileged state（所有物体位姿 + 关节状态 + 机器人状态）。
关键点：真实世界拿不到 $s$，但仿真里 $o\leftrightarrow s$ 的对应是已知的，所以"让感知策略在仿真里演一遍"就把示范"升维"成带状态的示范。叫 *inverse* 大概是因为方向与常规 teacher-student 相反：常规是 state 策略 → 感知策略，这里先是感知策略 → state 示范（论文未解释命名，此为笔者理解）。论文实际收集 15 条 $\mathcal D_{\text{sim}}$ 示范（Fig. 3）。

**③ 仿真中的 demo-augmented RL fine-tuning**（`method_as.tex` Eq. 1）。state-based 策略 $\pi_\theta(a|s)$（MLP 256-256，输出 14 维 Categorical；value 与 actor 共享第一层），在 PPO 目标上加 BC 项：
$$\max_{\theta,\phi}\ \alpha\negthinspace\negthinspace\sum_{(s_t,a_t,r_t)\in\tau_{\pi_{\theta_{\text{old}}}}}\negthinspace\negthinspace\min\negthinspace\Big(\rho_t\hat A_t,\ \mathrm{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t\Big)+\beta\negthinspace\negthinspace\sum_{(s_t,V_t^{\text{targ}})\in\tau_{\pi_{\theta_{\text{old}}}}}\negthinspace\negthinspace\big(V_\phi(s_t)-V_t^{\text{targ}}\big)^2+\gamma\negthinspace\negthinspace\sum_{(s_i,a_i)\in\mathcal D_{\text{sim}}}\negthinspace\negthinspace\frac{\pi_\theta(a_i|s_i)}{\sum_{a_c}\pi_\theta(a_c|s_i)}$$
其中 $\rho_t=\pi_\theta(a_t|s_t)/\pi_{\theta_{\text{old}}}(a_t|s_t)$。第一项是 PPO clipped surrogate，第二项是 value 回归（按 PPO 惯例应被最小化，式中写在 max 里的 $+\beta$ 应理解为带负号），第三项是在 $\mathcal D_{\text{sim}}$ 上的 BC（对 14 类离散动作就是分类似然）。注意式中 $\alpha,\beta,\gamma$ 都是各项权重， $\gamma$ 不是折扣因子。奖励只有稀疏成功信号；训练时随机化机器人/物体/目标初始位姿（各任务范围见 `appendix.tex` Table IV）。BC 项有两个作用：**解决稀疏奖励下的探索**，以及**把策略拉向物理上合理、能迁移回真实的解**（补偿默认物理参数带来的误差）。超参（`appendix.tex` Table VI）：基于 Stable Baselines 3，n_steps = episode length，PPO batch size 31257，BC batch size 32，BC weight 0.1，gradient clipping 5。

**④ Teacher-student 蒸馏 + 真实数据 co-training**（`method_as.tex` Eq. 2）。teacher 是 $\pi^{\ast}_ {\text{sim}}(a|s)$，student 是点云策略：
$$\max_\theta\ \alpha\negthinspace\negthinspace\sum_{(s_i,o_i,a_i)\sim\tau_{\pi_\theta}}\negthinspace\negthinspace\frac{\pi_\theta(\pi_{\text{teacher}}(s_i)\thinspace|\thinspace o_i)}{\sum_{a_c}\pi_\theta(a_c|o_i)}+\beta\negthinspace\negthinspace\sum_{(o_i,a_i)\in\mathcal D_{\text{real}}}\negthinspace\negthinspace\frac{\pi_\theta(a_i|o_i)}{\sum_{a_c}\pi_\theta(a_c|o_i)}$$
第一项是仿真中的 DAgger（student 自己 rollout，teacher 用 privileged state 重标动作），第二项在原始真实示范上 co-train，用来缩小点云视觉分布差异和 dynamics 差异。实际做法分两步（`appendix.tex` Teacher-student distillation）：先 BC，数据等概率（各 1/4）混合 15000 条"全可见"点云轨迹（直接从 mesh 采点）、5000 条真实相机视角渲染轨迹、带 distractor 的轨迹（附录文字写 2000 条，Table VIII 写 1000 条）、15 条真实轨迹；再做一轮 DAgger，DAgger 数据 / 仿真 distractor 数据 / 真实数据各 1/3。点云增强：6000 点（其中 3000 点从机械臂 mesh 采样）、dropout 比例 [0.1,0.3]、jitter $\mathcal N(0,0.01)$ 等（Table VII）。编码器是 Convolutional Occupancy Networks 的体素编码器（local PointNet + 3D U-Net， $32^3$ 网格）→ max/avg pooling → 128 维，与 9 维机器人状态拼接 → MLP 256-256；AdamW，lr 3e-4（Table VIII）。

**推理**：真实世界只运行最终的点云 student 策略，闭环执行；部署时不再更新参数。

## 关键实验与证据

平台：Franka Panda + 单个 RealSense（D455/D435）、Polymetis 控制；每个设置至少 10 次 rollout，报告 bootstrapped std，并"使用每个方法得到的最好策略"（`experiments.tex`）。三档难度：随机位姿 → + 视觉干扰物 → + 物理扰动（推物体、改目标、关抽屉、移动底座）。

- **主结果（Fig. 5，6 个任务，RialTo vs BC，均 15 条示范）**：RialTo 平均 91% / 77% / 75%（位姿 / 干扰物 / 扰动），BC 为 25% / 11% / 5%。按图中各柱计算，这三档平均差约 67 个百分点，对应摘要的 "over 67%"。**注意**：主结果这里跑的是"15 条**仿真中直接采集**的示范 + 15 条真实示范 co-training"，不是 inverse distillation 路线（`experiments.tex` Sec. IV-A 原文如此）。
- **更多 BC 数据也不够（Table I，book on shelf）**：BC 15 条 10/0/0%，BC 50 条 40/30/20%，RialTo 90/70/60%。50 条示范耗时 1 小时 45 分钟；RialTo 采 15 条示范用 30 分钟，建场景约 15 分钟人工操作时间。
- **真实数据 co-training 的作用（Fig. 6）**：在 plate on rack / book on shelf 这类较难任务上，"real demos + real co-training"高于"sim demos + sim co-training"和"only sim demos"（例如 disturbances 下 plate 60 vs 30 vs 20，book 70 vs 30 vs 40）；open drawer 上各变体都在 80–100%。从真实示范（inverse distillation）起步和从仿真示范起步结果接近，前提是蒸馏时保留真实 co-training。定性观察：co-training 后的策略抓取前留出更大空隙，更接近示范、更保守。
- **Digital twin vs 多样化合成资产（Fig. 7）**：在 4 个 Objaverse 抽屉上训练的多任务策略（20 条示范）在仿真 Objaverse 上 81%，但在真实目标抽屉只有 10%；在孪生上训的 RialTo 真实 90%（在 Objaverse 仿真上 17%）。结论：对特定部署场景，有针对性的 real-to-sim 比泛泛的场景多样化更有效。把合成资产与目标孪生混训也没有提升（Table XI：90/90% vs 90/80%）。
- **RL from scratch（Table III，仿真成功率）**：0 条示范时 5 个任务里 3 个是 0%，toaster 62%，plate 2%；从 15 条真实或仿真示范 fine-tune 则 81–96%。toaster 上 62% 的策略是利用关节位置偏差从底部推开烤面包机的"钻仿真漏洞"行为（Fig. 15），不可迁移。
- **需要多少真实示范（Table X）**：book on shelf 在 0/5/10 条时都是 0%，15 条时 90%；open drawer 5 条就到 89%。呈阶跃：真实示范太少时 $\pi_{\text{real}}$ 在仿真里一条都成功不了，inverse distillation 采不到 $\mathcal D_{\text{sim}}$，退化为 RL from scratch。
- **RL from vision（Sec. V-C，Fig. 14）**：drawer 任务上 state-based RL 12 小时达到 96%，点云 RL 35 小时仅 1%；原因是视觉策略同卡 batch size 小约 100 倍、渲染点云慢约 10 倍（`appendix.tex` RL from vision）。这是 inverse distillation 存在的理由。
- **干扰物训练（Table II，mug on shelf）**：蒸馏时加 distractor 将干扰场景成功率从 30% 提到 70%，无干扰时从 60% 到 100%。反之，只在真实点云里贴入物体点云的 IL 增强没有用（Table IX），作者认为原因是贴入的点云不产生遮挡。
- **In-the-wild（Fig. 8）**：open toaster 90 vs 30，cup in trash 90 vs 30，dish in rack 50 vs 0，平均比 IL 高 57%。
- **Sim-to-real gap（Table XIII）**：最终点云策略仿真 / 真实：toaster 90/90，book 84/90，plate 80/90，mug 72/100，drawer 95/90，cabinet 92/85；作者解释是故意让仿真比真实更难。
- **用户研究（Fig. 9，Table XIV）**：6 人（5 名新手 + 1 名专家）平均总时间 25 分 12 秒，其中人工操作时间 14 分 40 秒；作者给出一个线性关系（不是回归拟合，系数直接取用户研究各步平均时间，`appendix.tex` Scaling laws of the RialTo GUI）：总 active time = $t_{\text{scan scene}}+t_{\text{scan object}}N_{\text{objects}}+t_{\text{cut}}N_{\text{cut}}+t_{\text{joint}}N_{\text{joints}}$（均值 3:14、4:50、3:40、2:54）。

## 局限与论文没说的

**作者承认的**（`conclusion.tex`）：依赖精确深度传感器，薄、透明、反光物体会出问题；只能处理能仿真、能数字化的物体，目前主要是铰接刚体；控制器仍偏慢（以缩小 dynamics gap），需要更快控制器的任务有待研究；所做任务相对准静态，因此不做精确物理参数辨识也能工作，但面对更复杂的环境时参数辨识会变得重要；单任务端到端约 2 天，无法做持续学习。

**我读出来的**：
1. **主结果并不是完整的"real-only"流水线**：Fig. 5 用的是仿真直接采集的示范；inverse distillation 起步的真实世界结果只在 Fig. 6 的 3 个任务上出现。"只需要真实示范"这一点的证据面比摘要给人的印象要窄。
2. **统计量很小**：每格约 10 次试验，非零结果的 bootstrapped std 多在 ±9–17%（Table I/II/IX/XI/XII）；"使用每个方法最好的策略"没说明在什么数据上选 checkpoint；物理扰动由人手施加，没有标准化协议。
3. **正文与图表数字对不上**：正文称 co-training 在 book / plate 的 disturbances 下带来 3.5× / 2× 提升，但 Fig. 6 中 book 的数（70 vs 30/40）算不出 3.5×；正文说 co-training 实验包含 mug on shelf，Fig. 6 里却没有；distractor 轨迹数附录文字 2000、Table VIII 1000；主贡献写"eight tasks 提升 67%"，而 67% 实际上只来自 Fig. 5 的 6 个任务。
4. **BC baseline 偏弱**：15 条键盘遥操作示范，点云预处理与增强同蒸馏阶段（`appendix.tex` Imitation Learning Baseline）；没有和 Diffusion Policy 等更强的 IL 方法、也没有和真实世界 RL 方法比较（[HIL-SERL](../../04_rl/hil-serl/notes.md) 晚于本文，只作今天的参照）。
5. **Dynamics gap 只靠 BC 正则化和 co-training 兜底**：统一的质量 0.41 kg、摩擦 0.5 对准静态抓放够用，但对重物、滑动、接触丰富的任务没有检验；BC 权重（0.1）对"探索 vs 保守"的权衡没有消融。
6. **人力并非只有 15 分钟**：每个任务还要手写成功判据、设定随机化范围和相机位姿（Table IV/V），部分场景要选碰撞近似方式（SDF vs 凸分解），这些人工成本没有计入用户研究。
7. **每个场景都要重新训练**：约 2 天 / 任务 / 场景（RTX 2080 或 3090），所以这是"部署前特化"（在仿真里做 online RL，不是 offline RL），而不是部署时适应；Fig. 7 也显示这种特化几乎不迁移到别的实例（在 Objaverse 仿真上只有 17%）。

## 在脉络中的位置

- **前驱**：[Domain Randomization](../domain-randomization/notes.md)（初始位姿与干扰物随机化）；privileged teacher → sensor student 蒸馏（Learning by Cheating、[RMA](../rma/notes.md)、visual dexterity 这一线）与 DAgger；demo-augmented RL（DAPG、Nair et al. 2018）与 [PPO](../../04_rl/ppo/notes.md)；GPU 并行仿真 RL（Isaac Sim / Orbit，与 [legged-gym](../legged-gym/notes.md) 同一范式）；NeRFStudio / Polycam 等 3D reconstruction；真实数据 co-training（PoCo）。
- **后继 / 影响**：[ACDC digital cousins](../acdc-digital-cousins/notes.md) 把 RialTo 归为"human-assisted"的 digital twin 构建，指出 twin 覆盖面窄，改为自动生成一组"cousins"扩大训练分布（正好对应本文 Fig. 7 的反面论点）；[X-Sim](../x-sim/notes.md) 指出 RialTo 仍需要真实机器人数据，改用人类视频做 real-to-sim-to-real；[DPPO](../../04_rl/dppo/notes.md) 把它列为"从示范预训练的策略出发做 policy gradient fine-tuning"的代表；[Reality Gap 综述](../survey-reality-gap/notes.md) 将其归入"test time 利用真实数据即时构建仿真环境"的一类。补 visual gap 的 [SplatSim](../splatsim/notes.md)（Gaussian splatting 渲染 RGB）、补 physics gap 的 [PhysTwin](../phystwin/notes.md) / [ASAP](../asap/notes.md) / [SimOpt](../simopt/notes.md)，恰好对应本文自认的两处短板（只用点云、不辨识物理）。
- **与本项目其他主题的联系**：
  - *World model*：RialTo 的"simulator"是显式重建 + 物理引擎，能直接提供 privileged state 和稀疏奖励判据；与之相对，[Robotic World Model](../../05_intersections/robotic-world-model/notes.md)、[World-Gymnast](../../05_intersections/world-gymnast/notes.md)、[WMPO](../../05_intersections/wmpo/notes.md)、[Ctrl-World](../../05_intersections/ctrl-world/notes.md) 用学到的 world model 当 simulator。RL from scratch 钻关节误差漏洞，与 model-based RL 里 policy 利用 model error 是同一类失败，本文的解法是 BC 正则化。[SimDist](../../05_intersections/simdist/notes.md) 走的是另一个方向：在仿真里预训练 world model，再在真实世界快速适应。
  - *TTT*：作者自己把 real-to-sim 称为"test time specialization"，即针对部署环境特化策略；但它是部署前在孪生仿真里做的 on-policy RL + 蒸馏（约 2 天；不是 offline RL），部署时参数不变。对比 [PAD](../../05_intersections/pad/notes.md) 在部署中用 self-supervised 目标在线更新、[RMA](../rma/notes.md) 在线估计 extrinsics。"更快的部署时特化"正是作者在 limitation 里留下的问题。
  - *RL*：属于"IL 预训练 + RL fine-tuning"范式，与 [DPPO](../../04_rl/dppo/notes.md)、[SimpleVLA-RL](../../04_rl/simplevla-rl/notes.md)、[π_RL](../../04_rl/pi-rl/notes.md) 一样在仿真中做 on-policy PG；区别在于 RialTo 先降到 state 空间做 RL，再蒸馏回感知策略。

## 前置知识

1. PPO（clipped surrogate、GAE advantage、value loss）与 BC：理解 Eq. 1 三项各自的梯度含义。
2. Teacher-student / privileged distillation 与 DAgger：理解 Eq. 2 以及为什么要 on-policy 重标动作。
3. Sim-to-real gap 的来源（visual gap vs dynamics gap）以及 domain randomization / system identification 的分工。
4. 稀疏奖励下的探索问题，以及 demo-augmented RL（DAPG 一类）。
5. 点云策略编码器（PointNet、Convolutional Occupancy Networks 的体素化 + 3D U-Net）。
6. 3D reconstruction 到仿真资产：NeRF / 网格提取、URDF/USD 铰接、碰撞体凸分解。

## 复现可行性

- **代码**：官方 repo https://github.com/real-to-sim-to-real/RialToPolicyLearning 存在（已访问确认）；README 写明依赖 Isaac Sim 2022.2.1 + Omniverse + Orbit；仿真环境按 torch-1.13.0+cu117 安装 torch-scatter / torch-cluster，真实机器人环境装 pytorch 1.13.1 + pytorch-cuda 11.7（含 Polymetis），场景构建用独立的 RialToGUI repo；README 中没有看到现成的预训练模型或数据（此项来自 README，论文里没写）。论文侧：仿真是 Isaac Sim，代码参考 Orbit，PPO+BC 基于 Stable Baselines 3（`appendix.tex`）。
- **算力**（`appendix.tex` Compute resources）：单张 RTX 2080 或 RTX 3090；真实策略训练 + 在仿真采 15 条示范平均 7 小时，RL fine-tuning 平均 20 小时，teacher-student（采轨迹 + 蒸馏 + DAgger）24 小时，合计平均约 2 天 3 小时 / 任务。
- **硬件与数据**：完整复现需要 Franka Panda、标定深度相机（D455/D435）、Polymetis，以及手机扫描（Polycam / AR Code）或 NeRFStudio。
- **最小可复现实验建议**（不需要真实机器人）：
  1. 仿真部分：选 open drawer（episode 80 步，成功判据 `drawer_joint > 0.1 && gripper_open`），在孪生或任一抽屉 USD 上用键盘采 15 条示范，训 state-based PPO+BC 和 PPO from scratch，看能否复现 Table III 的对比（约 95% vs 0%）；再扫 BC weight（论文固定 0.1）。
  2. 用"sim-as-real"验证 inverse distillation：在扰动过的仿真（不同相机外参、摩擦、质量）中采点云示范训 $\pi_{\text{real}}$，在标称仿真里 rollout 收集 $\mathcal D_{\text{sim}}$；改变示范条数，复现 Table X 的阶跃现象。
  3. 蒸馏阶段对比有无"源域"数据 co-training（模拟 Eq. 2 的 $\beta$ 项），在目标域仿真中测成功率。
- **预计难点**：旧版 Isaac Sim 2022.2.1 + Orbit 的环境配置；点云渲染慢（论文称比不渲染慢约 10 倍），蒸馏阶段要 1.5 万量级的轨迹；碰撞网格质量（碗架需要很精确的碰撞，论文为此改用 SDF）可能直接影响能否学成；真实实验里扰动施加和评测口径难以对齐。

## 自测问题

1. "Inverse distillation"反在哪里？它依赖仿真的哪个性质，什么时候会失效？（结合 Table X 的阶跃现象回答）
2. Eq. 1 中的 BC 项同时解决了哪两个问题？为什么它能在一定程度上替代物理参数辨识？去掉它会发生什么（Table III 与 Fig. 15 的 toaster 例子）？
3. 为什么不直接在点云上做 RL？请给出论文给的效率证据及其原因（batch size、渲染开销）。
4. 蒸馏时已经有 DAgger 了，为什么还要在真实示范上 co-training？Fig. 6 里"sim demos + sim co-training"和"sim demos + real co-training"的差异说明了什么？
5. Fig. 7 中 Objaverse 训练的策略仿真 81%、真实 10%，而孪生训练的策略真实 90%、Objaverse 仿真 17%。这组对比支持什么主张？ACDC 又是从哪个角度反驳 digital twin 路线的？
