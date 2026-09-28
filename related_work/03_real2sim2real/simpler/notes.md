# Evaluating Real-World Robot Manipulation Policies in Simulation

> arXiv [2405.05941](https://arxiv.org/abs/2405.05941) · CoRL 2024（已在 PMLR v270 核实；本地 arXiv v1 PDF 用的是 IEEEtran 模板，没有标注会议）· 主题：03_real2sim2real · 层级：milestone
> 代码：https://github.com/simpler-env/SimplerEnv（论文项目页 https://simpler-env.github.io 给出的仓库，已确认存在）；texture matching 脚本在 https://github.com/Jiayuan-Gu/GeTex（附录 C-B 给出）· 笔记依据：`source/*.tex`（数字/公式）+ `paper.pdf`（图）

## 一句话

SIMPLER 不去搭 full-fidelity digital twin，只做一个"够真"的仿真：用离线 system identification 对齐控制器，用 green screening 加 texture matching 对齐视觉。然后拿它评测**在真实数据上训练**的 generalist manipulation policy，使策略在仿真里的**相对排名**和真实世界强相关。论文还提出 MMRV 指标，按真实成功率的差距给排名错误加权。

## 问题设定

- **目标**（`source/sections/03_problem_statement.tex` §III-A）：给定策略 $\pi_1\dots\pi_N$ 和它们的真实成功率 $R_i$，构造仿真器 $\mathcal S$，使仿真成功率 $R_{\mathcal S,i}$ 与 $R_i$ 在**相对表现**上强相关。作者明确说不追求 1:1 复现绝对成功率，也不打算取代真实评测，目的是给研究者一个可靠的"改进信号"。
- **输入/输出**：被评测的是 RT-1（Converged / 15% / Begin 三个 checkpoint）、RT-1-X、RT-2-X、Octo-Base、Octo-Small（§VI-A）。策略的输入是真实相机视角的 RGB 图像加语言指令：Google Robot 用头部 egocentric 相机，Bridge 用 Logitech C920 第三人称相机（附录 C-B）。输出是末端执行器的 delta pose 加 gripper 动作。系统最终输出的是每个策略在每个任务上的仿真成功率。
- **假设**：任务只涉及 rigid object；相机固定；有现成的真实 demonstration 数据可以拿来做 system identification，不需要采新数据；有真实场景的首帧图像可以做 green screening；搭资产和场景允许少量人工。
- **之前卡在哪**：
  - 真实评测贵，也难复现。RB2、TOTO 这类方案需要长期维护硬件（§II）。
  - ManiSkill2、RLBench 等仿真 benchmark 在仿真里训练、也在仿真里测，得到的分数不反映真实世界的行为。
  - 导航领域已有 sim2real predictivity 研究（Kadian et al. 2019、RoboTHOR），但它们面对的是静态场景。manipulation 的交互闭环紧，动作序列上一点小偏差就可能决定成败。
  - full-fidelity digital twin 需要艺术家手工建模材质和几何，还要辨识质心、摩擦等物理参数，不可扩展（§I）。
  - 常用的 validation action MSE 不能代表真实表现（Table I）。

## 核心方法

### 1. 评测指标：Pearson $r$ + MMRV

Pearson $r$ 有两个缺陷（§III-B，Fig. 3）：
- 它只衡量线性关系。即使排名完全正确，只要不是线性关系，$r$ 也会被压低。
- 它对取值范围不敏感。几个真实成功率很接近的策略，只要被真实评测的噪声扰动一下，$r$ 就可能剧烈变化。

Spearman 只看名次，忽略了排错的两个策略在真实成功率上差多少。于是作者提出 **Mean Maximum Rank Violation**（`source/sections/03_problem_statement.tex` Eq. 1–2）：

$$\text{RankViolation}(i,j)=|R_i-R_j|\cdot\mathbf 1\big[(R_{\mathcal S,i}<R_{\mathcal S,j})\neq(R_i<R_j)\big]$$

$$\text{MMRV}(R,R_{\mathcal S})=\frac1N\sum_{i=1}^N\max_{1\le j\le N}\text{RankViolation}(i,j)\in[0,1]$$

排错的惩罚等于两个策略的真实成功率之差，所以把两个本来就差不多的策略排反，代价很小。对每个策略只取最坏的那次违例再求平均。

**白板例子**（我用 Table IV 的 Pick Coke Can 平均成功率重算，结果与论文一致）：

| 策略 | Conv | 15% | 1-X | 2-X | Octo-B | Begin |
|---|---|---|---|---|---|---|
| 真实成功率 | 0.853 | 0.920 | 0.760 | 0.907 | 0.293 | 0.133 |
| 仿真成功率（VisMatch） | 0.857 | 0.710 | 0.567 | 0.787 | 0.170 | 0.027 |

- 排反的有三对：(15%, Conv) 惩罚 0.067，(15%, 2-X) 惩罚 0.013，(2-X, Conv) 惩罚 0.054。
- 每个策略取最大违例：15% 和 Conv 都是 0.067，2-X 是 0.054，其余三个是 0。
- MMRV $=0.188/6\approx0.031$，与表中数值一致。

### 2. 缩小控制 gap：离线 system identification（§IV-A）

做法是把 demonstration 的动作序列 $\{\mathbf a_i\}$ 在仿真里 **open-loop 回放**，然后比较仿真末端位姿 $(\mathbf x'_i,R'_i)$ 和真实记录的 $(\mathbf x_i,R_i)$。待辨识的是关节 PD 参数，即刚度 $\mathbf p$ 和阻尼 $\mathbf d$（`source/sections/04_approach.tex` Eq. 3–5）：

$$\mathcal L_{\text{transl}}=\frac1T\sum_i\|\mathbf x_i-\mathbf x'_i\|_2,\quad
\mathcal L_{\text{rot}}=\frac1T\sum_i\arcsin\!\Big(\tfrac{1}{2\sqrt2}\|R_i-R'_i\|_F\Big),\quad
\mathcal L_{\text{sysid}}=\mathcal L_{\text{transl}}+\mathcal L_{\text{rot}}$$

旋转项可以这样理解（我的推导，论文没写）：设 $\theta$ 是 $R^\top R'$ 的旋转角，则 $\|R-R'\|_F^2=6-2\,\mathrm{tr}(R^\top R')=4-4\cos\theta=8\sin^2(\theta/2)$。代回去得到 $\arcsin(\cdot)=\theta/2$，也就是说 $\mathcal L_{\text{rot}}$ 等于 geodesic 角误差的一半，单位是弧度。

优化方式：先把搜索范围归一化到 $[0,1]$，跑 simulated annealing；取 loss 最低的参数，缩小搜索范围再跑一轮，共 3 轮。数据只用开源数据集里"一小部分"轨迹，论文没给具体条数和迭代数。

仿真控制器要尽量模仿真实控制器（附录 Alg. 1/2）：
- **Google Robot**：delta EE 动作经 IK 得到目标关节角，再用 Ruckig 做带速度/加速度/jerk 约束的时间最优规划。arm 的约束是 1.5/2.0/50.0，gripper 是 1.0/7.0/50.0；$H_{\text{sim}}=501$，$H_{\text{ctrl}}=3$ Hz。
- **WidowX**：delta EE 动作经 IK 后直接设关节位置目标（不经 Ruckig 规划）；$H_{\text{sim}}=500$，$H_{\text{ctrl}}=5$ Hz。
- 模拟 Google Robot 时，PhysX 的 PGS（Projected Gauss-Seidel）solver 在抓取时会出现穿模，所以在 SAPIEN 和 Isaac Sim 的后端都改用 TGS（Temporal Gauss-Seidel）solver（附录 C-B）。

### 3. 缩小视觉 gap：Visual Matching 和 Variant Aggregation（§IV-B）

**Visual Matching** 由两部分组成：
- **Green screening**：先用 inpainting 从真实首帧里抹掉机器人和前景物体，得到背景 $I_{\text{real}}$。再用仿真的 ground-truth segmentation 得到前景 mask $M$（机器人和可交互物体），合成
  $$I'=M\odot I_{\text{sim}}+(1-M)\odot I_{\text{real}}$$
- **Texture matching**，流程在附录 C-B：
  1. 用 SAM 在真实图像里分割出物体；
  2. 粗略对齐物体位姿；
  3. 用 nvdiffrast 做可微渲染，精修位姿，让仿真物体与真实 mask 对齐；
  4. 把真实 RGB unproject 到 mesh 上；
  5. 看不到的面可以选择用 Zero123++ 生成视角后补全。

  机器人 link 的贴图本来就比较接近真实，所以只用 GIMP 做颜色替换。真实机械臂在运动中颜色会变，作者因此调了**多套 arm 颜色**（Google Robot 用 4 套，§VI-A）并对结果取平均。

**Variant Aggregation** 受 domain randomization 启发，不去缩小 gap，而是先建一个 base 环境，再在 background、lighting、distractors、table texture 四个轴上各做 2 个变体，把该环境所有变体的成功率取平均（附录 C-C，Fig. 11，场景改自 ReplicaCAD；平均时是否计入 base 本身，原文未明确说明）。

### 4. 环境构建与评测流程（§V）

- **资产来源**：URDF 从 GitHub 获取或用 ROS 导出；物体来自 Objaverse、3D 扫描或 One-2-3-45++；在 Blender 里把尺寸改成真实尺寸；用 CoACD 生成凸碰撞体。
- **物理参数**：密度查 GPT-4 或 Google，摩擦按材质设定。
- **位姿对齐**：通过对齐桌子或柜子的边缘来调相机和机器人位姿。
- **最耗人工的一步**：articulated 柜子需要手工建模。
- **速度**：用 SAPIEN 仿真，单张 RTX 4090、$640\times512$ 分辨率下每秒 3.5k 仿真步。按 500 Hz 仿真频率换算，比真实评测快 7×。
- **评测流程**：`env.reset()` 后，策略读取合成图像和指令，输出 delta 动作；控制器执行后由环境判定是否成功（Fig. 2 给出 Gym 接口）。Octo 带 diffusion head，结果对 3 个 seed 取平均。

## 关键实验与证据

**实验规模**（附录 B）：

| 平台 | 任务 | 真实 trial 数 |
|---|---|---|
| Google Robot | Pick Coke Can（3 种朝向 × 25 位置） | 75 |
| Google Robot | Move Near | 60 |
| Google Robot | Open/Close Drawer | 54 |
| Google Robot | Open Drawer & Place Apple | 27 |
| WidowX/Bridge | 4 个任务 | 各 24 |

注：附录 B 写 eggplant 任务真实评测为 24 次，但 Table V 中 eggplant 的真实成功率（0.033、0.233、0.433 等）是 1/30 的倍数，与 24 次对不上；其余任务的数值与所写次数一致。

真实评测共约 1500 个 episode（Fig. 1）。

**主结果**（Table IV，Fig. 6，Fig. 7）：

| Google Robot 任务 | VisMatch MMRV↓ / $r$↑ | VarAgg MMRV↓ / $r$↑ |
|---|---|---|
| Pick Coke Can | 0.031 / 0.976 | 0.084 / 0.960 |
| Move Near | 0.111 / 0.855 | 0.111 / 0.887 |
| Open/Close Drawer | 0.055 / 0.915 | 0.321 / 0.737 |
| Drawer + Apple | 0.000 / 0.969 | 0.148 / 0.235 |

- 在 Bridge 上，VisMatch 对 8 列（grasp 和 success 各 4 列）取平均，得到 MMRV 0.014、$r=0.890$。唯一排错的是 carrot 任务的 success 列，MMRV 0.111（Table V）。
- Table I 把 3 个 Google 任务平均：validation MSE 为 MMRV 0.375、$r=0.308$；VarAgg 为 0.143 / 0.778；VisMatch 为 0.056 / 0.924。
- 在 Bridge 上，用 validation MSE 排序的 Pearson 甚至是负的。例如 spoon 任务 $-0.951$，eggplant 任务 $-1.000$（Table XII）。
- 结论：**VisMatch 优于 VarAgg**。VarAgg 的视觉分布与真实差得更远，对视觉敏感的策略（Octo-Base、RT-1 15%）会被过度惩罚。

**分布偏移**（§VI-C，Fig. 8，Table VI）：
- 设置：选两个 RT-1 策略（训练时有/无 data augmentation），沿 5 个轴施加偏移：background、lighting、distractors、table texture、camera pose。
- 指标：计算 $\Delta\text{Success}=\frac12\sum_{k=1}^2(\text{Success}(\text{shift},k)-\text{Success}(\text{base}))$（`source/sections/06_experiments.tex` Eq. 6）。
- 对照：真实侧的数据来自 Xie et al. 2023（Table VI 最后一列）。
- 结果：两个任务平均后，MMRV/$r$ 在无 aug 策略上为 0.000/0.831，在有 aug 策略上为 0.041/0.970。
- 在仿真和真实中，camera pose 与 table texture 的影响都最大。
- 纯色桌面和花纹桌面带来的性能下降：真实为 4% vs 25%，仿真为 2% vs 24%。

**预测新的偏移**（Table VIII，Fig. 9）：
- 仿真中的发现：Octo-Base 用未调色的机械臂时成功率为 0%，换成调过色的机械臂后为 29.3%。
- 真实验证：作者用礼品包装纸包住真实机械臂。RT-1-X 的成功率从 0.760 降到 0.520，Octo-Base 从 0.293 降到 0.000。仿真里的预测成立。

**消融**：
- **SysID**（Table II，Pick Coke Can）：两组扰动后的 PD 参数得到 control loss 0.267 / 0.432、MMRV 0.070 / 0.100；SysID 参数得到 0.131 / **0.031**。
- **视觉**（Table III/IX，drawer 任务，3 个策略；正文写的是 RT-1 (Converged)、RT-1 (Begin)、RT-1-X，但 Table IX 实际列出的是 RT-1 (Converged)、RT-1 (15%)、RT-1-X）：
  - 什么都不做：MMRV 0.087，真实与仿真成功率差 0.272。
  - 只做 green screen：MMRV 0.087，差距 0.198。
  - green screen + 只匹配 drawer：MMRV 反而升到 0.142。
  - 三者都做：MMRV 0.050，差距 0.136。
  - 作者的假设是：场景各部分外观**不一致**比整体都不匹配更糟。
- **物理参数**（Table X）：
  - coke can 质量取 10–80 g、gripper 摩擦取 0.25–2.0 时，MMRV 恒为 0.031，$r$ 在 0.957–0.990 之间。
  - 柜子关节摩擦取 0.0125–0.20 时，$r$ 在 0.903–0.941 之间。
  - 这些参数改变成功率的幅度 ≤15%。
- **换仿真器**（Table XI，Fig. 10）：换成 Isaac Sim + VarAgg，Coke Can 为 0.064 / 0.973，Move Near 为 0.053 / 0.865。
- **单任务策略**（Table XIII）：加入只用 coke can 数据训练的 RT-1（真实 0.680、仿真 0.403）后，MMRV 为 0.027、$r=0.959$。
- **绝对值对齐**（Table XIV）：对 VisMatch 做 Kruskal-Wallis 检验。Google 的 coke-can 平均列和 Move Near 列里，6 个策略中各有 3 个的仿真与真实差异显著（$p<0.05$）；Bridge 只有 1 列出现 1 个显著。

## 局限与论文没说的

**作者承认的**（§VII）：
- 只做了 rigid object 任务。
- green screening 只适用于固定相机，也还原不了阴影。
- 搭资产和场景仍要人工，articulated 物体最费事。

**我读出来的**：
1. **样本量太小，统计意义存疑。** Google 有 6 个策略，Bridge 只有 3 个。3 个点的 Pearson 天然容易到 ±1，比如 Stack Block 的 $r=1.000$，eggplant 任务上 validation MSE 的 $-1.000$（Table XII）。论文全文没有给 MMRV 或 $r$ 的置信区间，也没做 bootstrap。
2. **论文内部数字不一致。** Table I 的 Drawer 列（VisMatch 0.027 / 0.942、VarAgg 0.235 / 0.486、MSE 0.306 / 0.231），与 Table IV、Fig. 6 和 Table XII 对应的值（0.055 / 0.915、0.321 / 0.737、0.346 / 0.264）对不上。我用 Table IV 的成功率重算 VisMatch 和 VarAgg 两组（validation MSE 无法从表中重算），得到的是后一组。因此 Table I 的 Average 列（0.056 / 0.924）偏乐观，按 Table IV 算应约为 0.066 / 0.915。
3. **只保证相对排名，不保证绝对值。** Isaac 上 RT-1 (Converged) 的 coke can 仿真成功率只有 0.410，真实是 0.853（Table XI）。Kruskal 检验也显示，Google 的 Pick Coke Can 平均列和 Move Near 上各有 3/6 个策略的绝对成功率差异显著（Table XIV；作者的表述则是"在许多任务和策略上差异不显著"）。
4. **被测策略同质。** 全部是用 RT-1/OXE/Bridge 真实数据做 imitation learning 训练的 transformer 策略（其中 Octo 带 diffusion head，RT-2-X 是 VLA）。在仿真里训练过、或经 RL 微调过的策略都没测。
5. **没验证拿它当训练环境是否还成立。** 论文只验证了"评测"：被测策略从没见过这个仿真器。后续工作（如 π_RL）直接在 SIMPLER 里做 RL，一旦对它做优化，相关性就可能失效（Goodhart）。Kadian et al. 已经指出，策略会利用仿真器的缺陷。
6. **仿真结果依赖人为的视觉选择。** 同一个 Octo-Base，换一套 arm 颜色，成功率就在 0% 和 29.3% 之间跳。平均 4 套颜色能缓解，但说明结论对资产调参敏感。
7. **分布偏移实验的对照有局限。** 真实侧数据借自 Xie et al.；仿真侧用的是 VarAgg 风格的 base 环境；只测了 2 个策略；MMRV 在 5 个偏移因子上计算。
8. **SysID 只辨识关节 PD（刚度、阻尼），loss 只看末端位姿。** 没建模延迟、gripper 力学或接触细节。任务也都是对精度要求不高的 pick/place/drawer，没有插孔这类 contact-rich 任务。
9. **环境绑定特定实验室。** 每个环境只对应一套具体的实验室布置，换一个实验室就要重新建环境。
10. **没报告总算力。** 论文只给了渲染速度，没有总 GPU 时长。

## 在脉络中的位置

- **前驱**：
  - 导航领域的 sim2real predictivity：Kadian et al. 2019 用 Pearson 评价仿真的预测力，RoboTHOR 的结论是视觉差异影响很大。
  - SAPIEN / ManiSkill2 仿真栈（作者同组）。
  - 被测对象与数据：RT-1、RT-1-X/RT-2-X（Open X-Embodiment）、Octo、BridgeData V2。
  - 分布偏移的分解方式来自 Xie et al. 2023。
  - Variant Aggregation 来自 [domain randomization](../domain-randomization/notes.md) 的思路。
  - 用真实轨迹校准仿真参数的思路来自 system identification 传统，参考 [SimOpt](../simopt/notes.md) 用真实 rollout 调整仿真参数分布。SIMPLER 做得更简单：只做离线 open-loop 回放、只辨识 PD。
- **后继 / 影响**：
  - SimplerEnv 成了 VLA 的常用 benchmark。[π_RL](../../04_rl/pi-rl/notes.md) 在 SIMPLER 的 WidowX 4 任务上做 RL fine-tuning，同时指出"明显的视觉 domain shift 仍未解决"，转而用 Gaussian splatting 做 real2sim2real。
  - [World-Gymnast](../../05_intersections/world-gymnast/notes.md) 把 SIMPLER 当"software simulator"基线，对比 world model 里的 RL，报告后者在 4 个任务中的 3 个上真实成功率更高。
  - [Ctrl-World](../../05_intersections/ctrl-world/notes.md) 引用 SIMPLER 的目标，改用 learned world model 在 DROID 上给策略排名。
  - [World model 综述](../../01_world_model/survey-wm-robot-learning/notes.md) 在结果汇总表中把"SIMPLER-style"结果（S-G / S-W / S-O，即 Google Robot、WidowX 及其他设置）与 RoboTwin、CALVIN 并列为对比口径之一。
- **与本项目其他主题的联系**：
  - **real2sim2real**：
    - SIMPLER 的 "realistic enough" 思路与 [ACDC digital cousins](../acdc-digital-cousins/notes.md) 的"不求精确复刻"相通，但一个用于评测，一个用于训练。[RIALTO](../rialto/notes.md) 则是用 digital twin 训练 RL。
    - [SplatSim](../splatsim/notes.md) 的 Gaussian splatting 渲染可以直接补上 green screening 只能用固定相机的短板。
    - [PhysTwin](../phystwin/notes.md) 针对的正是 SIMPLER 没覆盖的 deformable 物体。
    - [ASAP](../asap/notes.md) 用 delta action model 对齐物理，SIMPLER 只辨识 PD，两者可以对比。
    - 全局框架见 [reality gap 综述](../survey-reality-gap/notes.md)。
  - **world model**：UniSim、Ctrl-World 这类 learned simulator 在做同一件事，即"用仿真代替真实评测"。MMRV 和 Pearson 可以直接用来给 world-model 评测器打分，这是比较两条路线的现成口径。
  - **TTT**：SIMPLER 暴露出的策略脆弱点（arm texture、camera pose、table texture）正是 test-time adaptation 要处理的 shift。Visual Matching 的思路是"改环境去贴合策略"，[PAD](../../05_intersections/pad/notes.md) 是"改策略去贴合环境"。SIMPLER 的可控偏移轴可以作为 TTA 策略的评测床（这是我的建议，不是论文内容）。
  - **RL**：它本来是评测工具，却被后续工作当成 RL 训练环境用，见局限第 5 条。

## 前置知识

1. **generalist manipulation policy**：RT-1 / Octo 的输入输出（图像 + 语言 → delta EE 动作）、Open X-Embodiment 和 BridgeData V2。
2. **sim-to-real gap 的两根轴**：dynamics（控制）和 visual（外观）；domain randomization 与 system identification 的区别。
3. **机器人控制基础**：PD 控制器的刚度和阻尼、IK、带 jerk 约束的轨迹规划、$SO(3)$ 上的旋转误差。
4. **排名与相关性统计**：Pearson、Spearman、Kruskal-Wallis 检验。
5. **仿真与渲染工具**：SAPIEN/PhysX 的基本用法、分割 mask 合成，以及可微渲染（nvdiffrast）做位姿对齐。

## 复现可行性

- **代码**：SimplerEnv 开源，可用 pip 安装，提供 Gym 接口（Fig. 2）。据仓库 README（WebFetch 确认），需要：
  - Python 3.10/3.11，以及单独安装的 ManiSkill2_real2sim；
  - RT-1 系列需要 TensorFlow 2.15，Octo 需要 JAX；
  - 建议用 RTX 显卡，因为部分环境依赖 ray tracing，在 A100 这类非 RTX 卡上会很慢。
- **公开的策略**：RT-1 的几个 checkpoint、RT-1-X、Octo 都有提供。RT-2-X **没有公开**，所以 Google 那一组 6 个策略只能复现其中 5 个。
- **算力**：论文只报告单张 RTX 4090 每秒 3.5k 仿真步（§V），没给整套评测的 GPU 时长。
- **真实侧**：没有 Google Robot 或 WidowX 就无法复现。可以直接用 Table IV/V 里的真实成功率作参照。
- **最小实验**：
  1. 在 Bridge VisMatch 的 4 个任务上跑 RT-1-X、Octo-Base、Octo-Small。Octo 取 3 个 seed，每个任务 24 个初始配置。
  2. 用 Table V 的真实成功率，按 Eq. 1–2 计算 MMRV 和 Pearson，目标是复现 MMRV ≈ 0.014、$r\approx0.890$ 的量级。
  3. 在 Google Pick Coke Can 上做两个消融：把背景换成 ReplicaCAD 场景（去掉 green screen），或者把 PD 参数换回非 SysID 的值，观察 MMRV 是否像 Table II/III 那样变差（注意 Table III 原本是在 drawer 任务上做的，coke can 上的视觉消融论文没报告）。
- **预计难点**：
  - TF 和 JAX 的 CUDA 版本冲突。
  - 渲染结果对 GPU 和驱动敏感。
  - 要给自己的实验室搭新环境，就得做 texture matching、articulated 建模和 SysID，这部分人工成本最高。

## 自测问题

1. 给出 4 个策略的真实和仿真成功率，手算 MMRV。为什么要用 $|R_i-R_j|$ 加权，并对每个 $i$ 取 $\max_j$？Spearman 在 Fig. 3 最左两幅图上为什么给出同样的分数？
2. SysID 的训练信号从哪里来？为什么用 open-loop 回放 demonstration 动作，而不是闭环跑策略？证明 $\arcsin(\|R-R'\|_F/2\sqrt2)$ 等于半个旋转角。这一步辨识了哪些量，又刻意没辨识哪些？
3. domain randomization 在 sim-to-real 训练中很有效，可为什么在 real-to-sim 评测里，Variant Aggregation 输给了 Visual Matching？请从"被测策略没见过随机化分布"这个角度解释。
4. Table III 中，只做 green screen 加 drawer 纹理匹配，MMRV 反而从 0.087 升到 0.142。作者给出的解释是什么？你会设计什么实验来验证？
5. 如果直接在 SIMPLER 里对某个 VLA 做 RL，把仿真成功率刷高，论文的哪些证据还能支持"真实成功率也会提高"，哪些不能？
