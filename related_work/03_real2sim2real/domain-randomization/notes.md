# Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World

> arXiv [1703.06907](https://arxiv.org/abs/1703.06907) · IROS 2017（arXiv 注释为 "Submitted to IROS 2017"；Crossref 核实已收录于 IROS 2017 论文集，pp. 23–30）· 主题：03_real2sim2real · 层级：foundational
> 代码：none（论文只给了视频主页 https://sites.google.com/view/domainrandomization/ ；2026-09 核查，arXiv 页面和主页上都没有代码链接）· 笔记依据：`source/root.tex`（数字/公式）+ `paper.pdf`（图）

## 一句话

在 MuJoCo 里用**非真实感的随机纹理**、随机光照、随机相机和随机 distractor 渲染大量图像，只用这些合成数据监督训练一个 VGG-16 回归器，让它定位桌面物体。训练数据里没有任何真实图像（多数实验的卷积层仍用 ImageNet 预训练权重初始化，但作者发现从零训练在大多数情况下同样可行），在真实 webcam 图像上仍能达到约 1.5 cm 的定位精度，并足以支撑抓取。最关键的想法是：不去让仿真逼近真实，而是让仿真足够多样，使真实世界在模型看来只是训练分布里的 "just another variation"。

## 问题设定

- **任务**（`source/root.tex` §III 首段、§III-A、§IV-A）：给定目标物体 $s_i$，其形状和尺寸已知、颜色未知。输入一张**未标定的单目 RGB 图** $I_0$，要求回归该物体质心在世界坐标系下的位置 $(x_i,y_i,z_i)$；场景中还可能有需要忽略的 distractor。8 种几何体各自单独训练一个 detector（§IV-A）。
- **数据假设**：训练时只有仿真渲染图，以及仿真中免费得到的精确标签。训练期间**完全没有真实数据**，因此也无法在真实数据上做 validation（§IV-B 脚注）。
- **简化**：仿真中桌面高度固定，任务实际上退化成 2D 位置估计（§III-A）。
- **Reality gap 的三个来源**（§I）：仿真参数与真实系统不匹配（靠 system identification 调参，但它耗时且容易出错）；即使 system identification 做得好，仍存在 unmodeled physical effects（非刚性、齿轮间隙、磨损、流体动力学等）；low-fidelity simulated sensors。本文**只处理第三个来源**，即渲染图与相机图之间的差距，完全不涉及动力学。
- **之前的方法卡在哪**（§II）：
  1. 高保真路线（realistic rendering + system identification）：只靠写实 RGB 渲染，迁移效果有限。
  2. domain adaptation、fine-tuning、progressive nets：都需要真实数据，或者需要真实环境里的标签或 reward。
  3. 最接近的工作是 CAD2RL（Sadeghi & Levine）。它从约 200 种预生成、大多写实的材质里挑纹理，并使用 ImageNet 预训练；任务是走廊避障，对精度要求不高。
  4. 物理参数的 randomization（Ensemble-CIO、EPOpt、UP-OSI）当时要么没有做真实迁移，要么只针对动力学。

## 核心方法

**1. 训练目标。** 原文没有编号公式，只写了用 Adam 最小化预测位置与真实位置之间的 $L_2$ loss（§III-B）。按 §III 的文字，可以形式化为：

$$\theta^\star=\arg\min_\theta\thickspace\mathbb{E}_ {s\sim p(s)}\thickspace\mathbb{E}_ {\xi\sim P(\xi)}\Big[\big\Vert d_\theta\big(\mathcal{R}(s,\xi)\big)-y(s)\big\Vert_2^2\Big]$$

- $s$：与任务相关的场景状态，包括目标位置以及 distractor 的数量、形状和位置（笔记自设符号，不同于原文中表示目标物体的 $s_i$）。
- $y(s)$：目标质心的世界坐标。
- $\xi$：与任务无关的 nuisance 参数，包括纹理、光照、相机和噪声。
- $\mathcal{R}$：MuJoCo 自带的非真实感渲染器，外加图像噪声。

核心假设见 §I："if the variability in simulation is significant enough, models trained in simulation will generalize to the real world with no additional training." 在白板上可以这样推：

- 对同一个 $s$，不管 $\xi$ 怎么变，网络都必须输出同一个 $y(s)$。因此它只能依赖对 $\xi$ 不变的线索，也就是形状、尺寸和几何位置。
- 如果真实图像大致相当于 $\mathcal{R}(s,\xi_{\text{real}})$，并且 $\xi_{\text{real}}$ 被 $P(\xi)$ 覆盖，那么真实图就不是分布外样本。

**2. $P(\xi)$ 具体随机化了什么**（§III-A）：

| 维度 | 做法 |
|---|---|
| distractor | 每个场景在桌面上放 0–10 个，从同一几何体集合里选；地面和背景不放 |
| 物体 | 桌面上所有物体的位置和纹理 |
| 环境纹理 | 桌子、地面、skybox、机器人 |
| 纹理类型 | 三选一：(a) 随机 RGB 纯色；(b) 两个随机 RGB 之间的渐变；(c) 两个随机 RGB 组成的棋盘格 |
| 相机 | 先手动放一个视角和 FOV 与真实相机大致相同的相机。每个样本在它周围 $10\times5\times10$ cm 的盒子里随机平移；朝向先解析地对准桌面上的一个固定点，再在各方向上偏移至多 0.1 rad；FOV 缩放至多 5% |
| 光照 | 灯的数量，以及每盏灯的位置、朝向和 specular 特性 |
| 噪声 | 加到图像上的噪声类型和幅度 |

两个关键设计：

- **目标物体本身的纹理也随机**。所以 detector 看不到目标的颜色，只能依靠形状和尺寸。
- **纹理刻意不追求写实，而是追求数量**。纹理由程序生成，因此可以得到 "hundreds of thousands (or more)" 种不同的场景纹理组合（§II-C，与 CAD2RL 的对比）。

**3. 网络与训练**（§III-B，Fig. 2）：

- **结构**：修改版 VGG-16。保留标准 VGG 卷积层，特征图从 224×224×64 一路下采样到 14×14×512；全连接层换成 256 和 64 两层，不用 dropout；最后输出 $(x,y,z)$。
- **初始化**：多数实验中卷积层用 ImageNet 预训练权重，作者起初假设这是迁移成功的关键。但实际发现，随机初始化在大多数情况下同样可行。
- **优化**：Adam，学习率约 1e-4，而不是常用的 1e-3。作者发现这样收敛更好，也能避开一个常见的局部最优：把所有物体都映射到桌子中心（笔者解读：这相当于退化成输出标签均值）。
- **超参搜索**：学习率取 {1e-4, 2e-4}，batch size 取 {25, 50, 100}，报告其中最好的网络（§IV-A）。
- **图像噪声**：对迁移精度的影响可以忽略，但能改善收敛，让训练更不容易陷入局部最优（§IV-C）。

**4. 训练和推理怎么走。**

- **训练**：每个样本采样 $s$ 和 $\xi$ → MuJoCo 渲染 → 加噪声 → 做 $L_2$ 回归。这是纯监督学习，不涉及 RL。
- **推理**：真实 webcam 单帧缩放到 224×224 → detector 输出坐标 → 交给 MoveIt 做运动规划，执行预设的抓取动作（§IV-D）。部署时没有任何微调或适应。

## 关键实验与证据

**测试集**（§IV-B）：

- 共 480 张真实 webcam 图，物体距相机 70–105 cm。
- 所有测试图的相机位置都相同；光照不控制；画面里有部分机器人，地面上有胶带和线缆。
- 每种物体 60 张：20 张单独放置、20 张有 distractor、20 张部分遮挡。
- GT 的获得方式：把物体对齐到桌面网格上。

**主结果：Table I（按物体，单位 cm）**

- 平均定位误差约 1.5 cm（摘要、§IV-B）。
- 最好的物体是 Hexagonal Prism：单独 0.7±0.5，有 distractor 0.6±0.3，遮挡 1.0±1.0。
- 最差的格子：Tetrahedron 遮挡 3.2±5.8；其次是 1.9 cm 的三格：Cylinder 有 distractor 1.9±2.8、遮挡 1.9±2.9，Triangular Prism 遮挡 1.9±2.2。
- 表中许多格子的最佳结果来自从零训练的 detector，用脚注 1 标出。
- 仿真训练数据上的误差是 0.3–0.5 cm，作者称之为对仿真数据的 "over-fitting"，也就是说 sim-to-real gap 仍然存在。
- 作者称，这个精度与传统单目方法 MOPED 在相近距离下的平移误差相当。但论文没有给出 MOPED 的具体数字，也没有在同一测试集上跑对比。

**消融**

- **训练数据量（Fig. 4）**：
  - 用预训练模型时，5,000 个样本就能达到 "relatively accurate" 的水平；样本数增加到约 50,000 之前，性能一直在提升。
  - 从零训练在小数据时差很多。图上目测：约 1k 样本时，从零训练约 14 cm，预训练约 4 cm。
  - 数据量足够大时两者基本持平。结论是 ImageNet 预训练不是必需的，只在小数据时有明显帮助。
- **纹理数量（Fig. 5，训练样本固定为 10,000）**：
  - 纹理少于 1,000 种时，性能显著下降。图上目测：10 种约 15 cm，100 种约 12 cm，1,000 种约 5.5 cm，10,000 种约 2 cm。
  - 作者指出，只用 1,000 种纹理时，用 1 万张图训练的效果和只用 1,000 张图差不多。由此得出：在低数据区间，纹理随机化比物体位置随机化更重要。
  - 脚注 3：纹理总数可能超过样本数，因为每个场景有很多表面，各自有独立纹理。
- **组件消融（Table II，每个模型都用 20,000 个样本，结果为几何体平均，单位 cm）**：

| 设置 | 单独 | Distractors | 遮挡 |
|---|---|---|---|
| Full method | 1.3±0.6 | 1.8±1.7 | 2.4±3.0 |
| No noise added | 1.4±0.7 | 1.9±2.0 | 2.4±2.8 |
| No camera randomization | 2.0±2.1 | 2.4±2.3 | 2.9±3.5 |
| No distractors in training | 1.5±0.6 | **7.2±4.5** | **7.4±5.3** |

  解读：
  - 训练时不放 distractor，单独放置的误差几乎不变（1.3 → 1.5），但有干扰物时误差从 1.8 升到 7.2。可见训练时的 distractor 是真实场景抗干扰能力的来源。
  - 相机随机化带来稳定但较小的提升。
  - 噪声对精度几乎没有影响。

**机器人实验（§IV-D）**

- 平台：Fetch 机器人 + MoveIt。
- 挑选两个 "most consistently accurate" 的 detector，各在 20 个越来越杂乱的场景中抓取，共 **38/40** 次成功。
- 部分测试场景里，distractor 的摆放朝向在训练中没出现过，例如侧放的六棱柱。
- detector 不知道目标颜色，仍能区分紧挨在一起的同色物体。
- 对纹理不均匀的真实物体，用 YCB 的 Spam 罐测试：训练时配几何体 distractor，测试时换成 YCB 里其他没见过的食品，抓取 **9/10** 次成功。

## 局限与论文没说的

**作者承认的**（§IV-B、§V）：

- 真实误差（约 1.5 cm）明显高于仿真误差（0.3–0.5 cm）。
- 无法用真实数据做 validation。
- 未来工作包括：接触丰富或更高精度的任务、更高分辨率的输入、更好的网络结构、更多种类的纹理/光照/渲染随机化和更多数据、多视角/双目/深度输入、与 domain adaptation 结合。

**我读出来的问题：**

1. **没有 baseline。** 论文没有与写实渲染、domain adaptation、真实数据训练或 CAD2RL 式纹理库做对照。唯一的外部比较（MOPED）也只是定性描述。所以论文证明的是"足够多的随机纹理就够用"，并**没有**证明随机纹理比写实渲染更好。
2. **模型选择口径不清。** Table I 报告的是 6 组超参 ×（预训练 / 从零训练）中的 "best network"，但论文没有说明 best 按什么标准选。训练期间又没有真实验证集，如果是按真实测试集挑的，Table I 就偏乐观。对照 Table II（每个模型固定用 20,000 个训练样本；§IV-A 说"每个实验"都做了同样的小规模超参搜索并报告 best network，所以 Table II 同样受这个选择口径影响）：full method 在遮挡条件下为 2.4±3.0 cm，比摘要中的 1.5 cm 差不少。
3. **任务被大幅简化。**
   - 桌面高度固定，每个物体单独一个 detector。
   - 形状和尺寸已知，只估计位置、不估计姿态。
   - 测试时相机固定，距离 70–105 cm。
   - 相机随机化只覆盖手动放置点附近的 10×5×10 cm、0.1 rad、5% FOV。它的作用是容忍仿真相机与真实相机"大致对齐"，而不是泛化到任意视角。
4. **只处理视觉 gap。** 论文没有做动力学随机化。抓取是开环的"感知 + 运动规划 + 预设抓取"，不是闭环 policy；摘要和 §I 里的 "for the purpose of robotic control" 实际只落在感知这一环。
5. **长尾失败被平均值掩盖。** 好几格的标准差接近甚至超过均值，例如 Tetrahedron 遮挡 3.2±5.8，说明误差分布里有少数大误差。我推测原因可能是锁定到了错误的物体。抓取实验一共只有 40 + 10 次，而且挑的是最稳定的两个 detector。
6. **$P(\xi)$ 是手工设计的。** 随机化哪些维度、范围取多大，都凭经验决定。论文没有研究随机化过度的代价（任务变难、需要更多数据和模型容量），也没有研究真实域落在分布之外时会发生什么。后续工作正是从这里切入：SimOpt 指出 "design of the appropriate simulation parameter distributions remains a tedious task"；RMA 指出 DR "trades optimality for robustness leading to an over conservative policy"。
7. **算力没有报告。** 论文没写 GPU、训练时长和渲染吞吐，只在 Fig. 1 caption 里说用了 "hundreds of thousands of low-fidelity rendered images"。
8. **测试物体覆盖面窄。** 测试物体几乎都是纯色几何体，唯一带纹理的真实物体是 Spam 罐；透明、反光、可形变的物体都没有测。

## 在脉络中的位置

- **前驱**：
  - CAD2RL（Sadeghi & Levine 2016）：四旋翼避障，从约 200 种预生成、大多写实的材质中选纹理，并使用 ImageNet 预训练。
  - 物理参数随机化：Ensemble-CIO（Mordatch 2015）、EPOpt（Rajeswaran 2016）、Antonova 2017、UP-OSI（Yu 2017）。
  - CV 中的合成数据：Render for CNN、Peng et al. 2015。
  - 基础组件：VGG-16、MuJoCo。
  - 本文在 §II-B 把自己明确定位为"不做 domain adaptation"的迁移方法，并在 §II-C 与 progressive nets（Rusu 2016）、Tzeng 2016 那条路线形成对照。
- **后继 / 影响**：
  - [SimOpt](../simopt/notes.md)：用少量真实 rollout 反过来优化 $P(\xi)$，把手工设定随机化范围改成一个闭环。
  - [RMA](../rma/notes.md)：把纯 DR 训练的策略当作 "Robust" baseline，改为在部署时在线估计 extrinsics。
  - [legged_gym](../legged-gym/notes.md)：DR 成为 sim-to-real RL 配方的标配，具体做法包括摩擦系数在 [0.5, 1.25] 内均匀采样、加观测噪声、随机推机器人。
  - [PAD](../../05_intersections/pad/notes.md)：把 DR 归入"学一个对环境不变的鲁棒策略"一类，认为测试环境真正未知时，这种期望很难实现，于是转向部署时的 test-time training。
  - [SplatSim](../splatsim/notes.md)：走相反方向。它认为 DR 类方法需要针对任务调参和做环境工程，于是改用 Gaussian splatting 做写实渲染。
  - [ACDC](../acdc-digital-cousins/notes.md)：发现在原 digital twin 场景上测试时，用全部可行资产训练（即 "naive domain randomization"）反而明显不如 digital twin，因此提出 digital cousins 作为一种 "conditional" randomization。
  - [RialTo](../rialto/notes.md)：在 teacher-student 蒸馏中加入视觉 distractor 随机化来提升抗干扰能力，与本文 Table II 的结论一致。
  - [SIMPLER](../simpler/notes.md)：把 DR 的思路搬到 real-to-sim 策略评测上，提出 "Variant Aggregation"，与 Visual Matching 并列。
  - [X-Sim](../x-sim/notes.md)：说明自己的方法与 DR 兼容。
  - [Reality-gap survey](../survey-reality-gap/notes.md)：把 DR 列为最常用的方法之一。
- **与本项目其他主题的联系**：
  - **real2sim2real 内部**：DR 是"鲁棒化"路线的起点，思路是让 $P(\xi)$ 覆盖真实；system identification 和 digital twin 是"校准"路线，思路是让仿真逼近真实。SimOpt 和 ACDC 位于两者之间。
  - **TTT**：
    - DR 是训练时的不变性，部署时参数冻结；[TTT](../../02_test_time_training/ttt-self-supervised/notes.md) 和 [Tent](../../02_test_time_training/tent/notes.md) 则在测试时用无标签的目标域数据更新参数。
    - 本文 §IV-B 脚注的困境——训练时没有真实数据，也无法在真实数据上验证——正是 TTA 的出发点。
    - RMA 的 adaptation module 从最近的状态-动作历史在线估计 extrinsics（环境参数的 latent 编码），可以看作两条路线之间的折中。
  - **World model**：
    - [SimDist](../../05_intersections/simdist/notes.md) 在仿真中预训练 world model 时对感知观测做了大量随机化。它明确指出，如果用 pixel reconstruction 目标，latent 会被迫编码这些刻意随机化的纹理、光照和渲染伪影。
    - 这说明 DR 与 [Dreamer](../../01_world_model/dreamer/notes.md) 这类重建式 world model 之间存在张力。
    - [UniSim](../../05_intersections/unisim/notes.md) 和 [Cosmos](../../01_world_model/cosmos-wfm/notes.md) 这类生成式 world model 能否提供 DR 所需的受控多样性，是一个开放问题（笔者观点）。
  - **RL**：本文本身是监督学习，但结论明确把 DR 定位为"让仿真中 deep RL 学到的策略在真机上可用"的工具。此后，[PPO](../../04_rl/ppo/notes.md) + DR + 大规模并行成为 locomotion 的主流配方。

## 前置知识

1. **Sim-to-real gap 的来源**，以及 system identification 和 domain adaptation 分别在解决什么问题。
2. **CNN 与 ImageNet 预训练 / 迁移学习**（VGG-16），以及监督回归中的 $L_2$ loss 和 Adam。
3. **分布偏移与不变性**：data augmentation 为什么能带来对 nuisance factor 的不变性；OOD 泛化和支撑集（support）覆盖的直觉。
4. **渲染和相机基础**：纹理、光照（specular）、相机内参和外参、FOV；为什么单目未标定相机需要固定桌面高度这类先验。
5. （可选）MuJoCo 场景建模与渲染 API；MoveIt 运动规划。

## 复现可行性

- **代码**：没有官方代码；网上的非官方实现未核实。需要自己搭建。
- **依赖**：
  - MuJoCo 及其自带渲染器（原文所用）。
  - VGG-16 + ImageNet 权重（torchvision 就有）。
  - 真实实验需要：webcam、一张桌子、8 种几何体积木（论文没给来源和尺寸）、网格纸（用于测 GT）。
  - 抓取部分需要 Fetch 机器人 + MoveIt，可以换成任意带运动规划的机械臂。
- **算力**：论文未报告。笔者估计（非论文数字）：VGG-16 在 224×224 输入、5 万张图的规模上，单张消费级 GPU 训练在小时级即可完成；渲染成本很低。
- **最小可复现实验**：
  1. **只用仿真的替代实验（无需硬件）**：
     - 用 MuJoCo 搭建"桌面 + 8 种几何体 + 0–10 个 distractor"的场景，按 §III-A 的随机化方案生成训练数据。
     - "伪真实"测试域用一组固定的写实纹理 + 固定光照 + 固定相机来渲染，或者直接换一个渲染器来制造 renderer gap。
     - 复现 Fig. 5 的纹理数量曲线：固定 1 万个样本，纹理数分别取 10、100、1k、10k。
     - 复现 Table II 的 "No distractors in training" 消融：检查有干扰物时误差是否从约 2 cm 跳到 7 cm 量级。
  2. **低成本真实验证**：一个 webcam + 桌面网格，按论文口径每种条件拍 20 张（单独 / distractor / 遮挡），比较真实误差与仿真误差之间的 gap。
- **预计难点**：
  - 让仿真相机的初始位置与真实相机大致对齐（论文靠手动放置）。
  - 世界坐标系原点与真实网格对齐。
  - 避开"输出桌子中心"的平凡解（用 1e-4 学习率、加噪声）。
  - 纹理数量的计数口径（脚注 3）。
  - "best network" 的选择口径不明（Table I、Table II 都受影响），因此复现时应自己固定一个模型选择规则（例如只在仿真 hold-out 集上选），并以 Table II（20,000 个样本）的数字作为主要对照。

## 自测问题

1. 为什么**目标物体自身的纹理**也必须随机？如果只随机化背景，网络最可能学到什么捷径？这和"能区分紧挨着的同色物体"有什么关系？
2. Fig. 5 中，只用 1,000 种纹理时，1 万张图的效果约等于 1,000 张图。从 $\mathbb{E}_ {s}\mathbb{E}_ {\xi}$ 两个期望的角度解释：为什么低数据区间里 $\xi$ 的多样性比 $s$ 的多样性更重要？
3. Table II 中，去掉训练 distractor 后，单独放置的误差几乎不变，而有干扰物时误差从 1.8 升到 7.2。请从回归目标 $y(s)$ 的角度，推测网络在真实杂乱场景中的失败模式。
4. 为什么学习率过大时，网络容易陷入"把所有物体都映射到桌子中心"的局部最优？这个解在 $L_2$ loss 下对应什么统计量？
5. 如果真实环境的某个参数（例如相机偏离手动放置点 30 cm）落在 $P(\xi)$ 之外，DR 会怎样？SimOpt（调整 $P(\xi)$）、RMA（在线推断环境参数）、PAD/TTT（测试时自监督更新）各自从哪个环节解决这个问题？
