# PhysTwin: Physics-Informed Reconstruction and Simulation of Deformable Objects from Videos

> arXiv [2503.17973](https://arxiv.org/abs/2503.17973) · ICCV 2025（arXiv v1 2025-03；已在 CVF Open Access 的 ICCV 2025 主会论文列表核实） · 主题：03_real2sim2real · 层级：frontier
> 代码：https://github.com/Jianghanxiao/PhysTwin（MIT 许可；项目页 https://jianghanxiao.github.io/phystwin-web/） · 笔记依据：`source/*.tex`（数字/公式）+ `paper.pdf`（图）

## 一句话

PhysTwin 从一段三视角 RGB-D 交互视频里重建可形变物体的 digital twin，由三部分组成：spring-mass 物理模型、生成式补全的几何、Gaussian splats 外观。它把 system identification 写成"仿真与观测对齐"的逆问题，分两步优化：先用 CMA-ES（zero-order）确定不可微的拓扑和少量低维参数，再用可微 spring-mass 仿真器（first-order）细化上万根弹簧的稠密刚度。结果是一个能在新交互下实时仿真、可以用于 model-based planning 的 action-conditioned 动力学模型。

## 问题设定

- **输入**：3 台 RealSense-D455 拍摄的 RGB-D 视频 $\mathbf O_{t,i}=(\mathbf I,\mathbf D)$，内容是人手与物体的交互（快速提起、拉伸、推、挤压，单手或双手），每段 1–10 s（`experiments.tex` §5.1）。**action 是手**：先用 Grounded-SAM2 分割手的 mask，再用 CoTracker3 跟踪，反投影到 3D 后做 farthest-point sampling，得到一组控制点， $a_t$ 就是这些控制点的位置（`method.tex` §4.2.1）。
- **输出**：一个 PhysTwin，包括几何与拓扑 $\mathcal G_0$、物理参数 $\alpha$（稠密弹簧刚度、碰撞参数、阻尼、控制点连接参数）和外观 $\theta$（3D Gaussians）。给定任意新的控制点轨迹，它都能前向仿真并渲染。
- **假设**：物体可以用 spring-mass 表示；首帧状态就是 canonical/rest 状态（`problem_statement.tex` 脚注）；手与物体的接触用"控制点和附近物体点之间连弹簧"来建模；只用每段视频前 70% 的帧（训练段）构建 twin。
- **之前卡在哪**（§1–2）：(1) dynamic NeRF 和 dynamic 3DGS 只能复现观测到的运动，不含物理，所以做不了 action-conditioned 预测。(2) 学习式 neural dynamics（GNN，例如 GS-Dynamics）需要大量数据，而且只适用于特定物体和特定动作。(3) physics-based 方法（PhysGaussian、PhysDreamer、PAC-NeRF、Spring-Gaus）要么手动设定参数，要么需要预扫描形状或稠密视角。作者还指出，最接近的 Spring-Gaus "过度正则"、缺少动量守恒和真实重力，而且不支持 action 输入。

## 核心方法

**1. 物理模型**（`source/sections/problem_statement.tex` Eq. 1 及其后的更新式；原文 Eq. 1 没加括号，求和范围有歧义；这里按物理含义把 spring 与 dashpot 两项括进求和号， $\mathbf F^{\text{ext}}$ 放在外面）

$$\mathbf F_i=\sum_{(i,j)\in\mathcal E}\Big[k_{ij}\big(\Vert\mathbf x_j-\mathbf x_i\Vert-l_{ij}\big)\frac{\mathbf x_j-\mathbf x_i}{\Vert\mathbf x_j-\mathbf x_i\Vert}-\gamma(\mathbf v_i-\mathbf v_j)\Big]+\mathbf F_i^{\text{ext}}$$

$$\mathbf v_i^{t+1}=\delta\Big(\mathbf v_i^t+\Delta t\thinspace\frac{\mathbf F_i}{m_i}\Big),\qquad \mathbf x_i^{t+1}=\mathbf x_i^t+\Delta t\thinspace\mathbf v_i^{t+1}$$

各符号含义： $k_{ij}$ 是每根弹簧各自的刚度（稠密参数）； $l_{ij}$ 是 rest length，取自首帧； $\gamma$ 是 dashpot 阻尼； $\delta$ 是 drag damping； $\mathbf F^{\text{ext}}$ 包括重力、impulse-based 碰撞（物体与 collider 之间、物体自身各点之间）和控制交互。整体记作 $\mathbf X_{t+1}=f_{\alpha,\mathcal G_0}(\mathbf X_t,a_t)$。另外，用 $\mathbf v^{t+1}$ 去更新 $\mathbf x$ 其实是 semi-implicit（symplectic）Euler，论文把它叫作 explicit Euler。

**2. 总目标**（`source/sections/method.tex` Eq. 2）

$$\min_{\alpha,\mathcal G_0,\theta}\sum_{t,i}C(\hat{\mathbf O}_ {t,i},\mathbf O_{t,i})\quad\text{s.t.}\quad \hat{\mathbf O}_ {t,i}=g_\theta(\hat{\mathbf X}_ t,i),\ \ \hat{\mathbf X}_ {t+1}=f_{\alpha,\mathcal G}(\hat{\mathbf X}_ t,a_t)$$

（原文 Eq. 2 约束里的下标写作 $\mathcal G$，而 Eq. 3 和 §3 写作 $\mathcal G_0$，应是同一个量。）其中 $C=C_{\text{geometry}}+C_{\text{motion}}+C_{\text{render}}$。从结构上看， $f$ 是 transition， $g_\theta$ 是 observation model，和 latent world model 的形式一样，区别在于这里的 state 是有物理意义的质点。整个问题拆成两个阶段优化（Fig. 2）。

**3. 第一阶段：物理与几何**（`method.tex` Eq. 3）

$$\min_{\alpha,\mathcal G_0}\sum_t\Big(C_{\text{geometry}}(\hat{\mathbf X}_ t,\mathbf X_t)+C_{\text{motion}}(\hat{\mathbf X}_ t,\mathbf X_t)\Big)\quad\text{s.t.}\ \hat{\mathbf X}_ {t+1}=f_{\alpha,\mathcal G_0}(\hat{\mathbf X}_ t,a_t)$$

- $C_{\text{geometry}}$ 是单向 Chamfer distance，方向是从部分观测点云到完整的预测状态，所以看不见的部分不会被惩罚。
- $C_{\text{motion}}$ 是预测点与 3D pseudo-GT 轨迹之间的误差；pseudo-GT 由 CoTracker3 的 2D 跟踪结果经深度反投影得到。论文没有细说跟踪点和质点之间怎么对应。

作者列出三个难点：视角稀疏导致只能看到部分物体；离散拓扑要和连续参数一起优化；接触让动力学不连续，再加上时间长、参数稠密，连续优化很难做。对应的做法有两块：

(a) **Generative shape prior**（§4.2.1 + 附录 A）。流程是：Grounded-SAM2 分割前景，超分辨率模型（LDM）放大，TRELLIS 从单张 RGB 生成完整 mesh；然后配准。配准时先在物体周围的球面上放虚拟相机渲染，用 SuperGlue 做 2D 匹配，取匹配数最多的视角作为粗略旋转，再用 PnP 精修；PnP 之后，匹配点对与相机原点共线，所以尺度和平移只需优化一个尺度量；接着做 as-rigid-as-possible（ARAP）非刚性配准，最后做 ray-casting 对齐。

(b) **Sparse-to-dense 分层优化**。参数空间包括：物体内部拓扑（连接半径、最大邻居数）、控制点与物体的连接（半径、最大邻居数）、碰撞参数、阻尼、刚度。
- **Stage 1：zero-order（CMA-ES）**。假设刚度处处相同，只搜索低维参数，其中包括不可微的半径和邻居数。
- **Stage 2：first-order**。用自建的可微 spring-mass 仿真器（基于 Warp）做梯度下降，同时优化"上万根弹簧"（原文 tens of thousands）的稠密 $k_{ij}$ 和碰撞参数。
- 为什么要这样拆：zero-order 方法在高维空间里效率太低；first-order 方法处理不了离散拓扑，而且需要一个好的初始化。

**4. 第二阶段：外观**（`method.tex` Eq. 4）

$\min_\theta\sum_{t,i}C_{\text{render}}(\hat{\mathbf I}_ {i,t},\mathbf I_{i,t})$，其中 $C_{\text{render}}$ 是 L1 加 D-SSIM。为了简单，只在 $t=0$ 优化，并把 Gaussians 限制为各向同性，防止形变时出现尖刺伪影。动态渲染用 LBS 实现（`main.tex` 附录 B，Eq. 5–8）：

$$R_i^t=\arg\min_{R\in SO(3)}\sum_{j\in\mathcal N(i)}\big\Vert R(\hat\mu_j^t-\hat\mu_i^t)-(\hat\mu_j^{t+1}-\hat\mu_i^{t+1})\big\Vert^2$$

$$\mu_j^{t+1}=\sum_{k\in\mathcal N(j)}w_{jk}^t\big(R_k^t(\mu_j^t-\hat\mu_k^t)+\hat\mu_k^t+T_k^t\big),\quad q_j^{t+1}=\Big(\sum_k w_{jk}^t r_k^t\Big)\otimes q_j^t,\quad w_{jk}^t\propto\Vert\mu_j^t-\hat\mu_k\Vert^{-1}$$

**5. 使用方式**。给出一条新的控制点轨迹（来自键盘、遥操作，或 planner 给出的机械臂动作），控制点用弹簧连到物体上，Warp 做前向仿真，再由 LBS 驱动 Gaussians 渲染。遇到新交互时，先用配准流程把 twin 对齐到目标序列的首帧（§5.2）。

## 关键实验与证据

**设定**（§5.1、附录 C）：共 22 个场景，物体有绳子、毛绒玩具、布、快递包裹，每个场景按时间 7:3 切分。每段视频用 TAPIR 的半自动工具手工标注 9 个 GT 跟踪点。泛化集由 11 对交互组成，每对可以双向迁移，一共 22 个实验。3D 指标是单向 CD 和 tracking error（论文没写单位）；2D 指标是 PSNR、SSIM、LPIPS 和 IoU，**只在中间视角计算**。两个 baseline 都经过改造：Spring-Gaus 加上了控制点弹簧，并用本文的 shape prior 初始化，**只报告它成功的 case**；GS-Dynamics 改用本文基于 CoTracker3 的 3D 跟踪做监督，并加上本文的 Gaussian blending。

**Table 1（重建与重仿真 / Future Prediction）**：

| 方法 | CD | Track | IoU% | PSNR | SSIM | LPIPS ‖ CD | Track | IoU% | PSNR | SSIM | LPIPS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Spring-Gaus | 0.041 | 0.050 | 57.6 | 23.445 | 0.928 | 0.102 ‖ 0.062 | 0.094 | 46.4 | 22.488 | 0.924 | 0.113 |
| GS-Dynamics | 0.014 | 0.022 | 72.1 | 26.260 | 0.940 | 0.052 ‖ 0.041 | 0.070 | 49.8 | 22.540 | 0.924 | 0.097 |
| PhysTwin | **0.005** | **0.009** | **84.4** | **28.214** | **0.945** | **0.034** ‖ **0.012** | **0.022** | **72.5** | **25.617** | **0.941** | **0.055** |

**Table 2（未见过的交互）**：GS-Dynamics 为 CD 0.029、Track 0.038、IoU 63.4、PSNR 25.053；PhysTwin 为 0.013、0.018、72.18、26.199。PhysTwin 在新交互上的 IoU（72.18）和它在 future prediction 上的 IoU（72.5）相当，作者据此强调 PhysTwin 对不同交互的鲁棒性和泛化能力（§5.2）。

**Table 3（消融，`tables/tab_ablation.tex`）**：

| 变体 | 重仿真 CD / Track / IoU | Future CD / Track / IoU |
|---|---|---|
| Zero-order only | 0.007 / 0.012 / 80.2 | 0.014 / 0.025 / 69.2 |
| First-order only | 0.008 / 0.012 / 82.7 | 0.019 / 0.034 / 65.7 |
| 完整 | 0.005 / 0.009 / 84.4 | 0.012 / 0.022 / 72.5 |

值得注意的是：只用 first-order 时，重建 IoU（以及 PSNR/SSIM/LPIPS）比只用 zero-order 好，但重建 CD 略差（0.008 vs 0.007），而 future prediction 的全部 6 项指标都是所有变体里最差的。笔者的推断（论文未这样表述）：在默认拓扑上拟合稠密刚度，更像是对训练段过拟合。作者的解释是：默认的连接方式表达不了真实结构，控制点与物体之间的连接距离用固定初值也不合适。只用 zero-order 时，又捕捉不到细粒度的材料差异。

**其他证据**：Fig. 6 表明，物理约束优化之后得到的轨迹，比用作训练监督的 CoTracker3 原始结果更平滑（定性）。Fig. 3 中，GS-Dynamics 往往几乎不动，Spring-Gaus 经常数值崩溃。附录 C 的数据效率实验里，GS-Dynamics 在同一动作上用了 30 段视频，PhysTwin 只用 1 段，作者称 GS-Dynamics"没有提升"，但**没有给出数字**。应用部分（§5.3、Fig. 5）展示了键盘实时交互、遥操作时叠加 twin 的预测、用 model-based planning 把绳子摆成目标形状，但**全部是定性展示**：没有成功率，没有说明用的是哪个 planner，也没有报告 FPS。

## 局限与论文没说的

**作者承认的**（附录 D Future Work）：需要三视角 RGB-D，希望将来只用单段 RGB 视频；物理参数只从一种交互里估计，用多种交互可能估得更准；用于 RL（减少对 domain randomization 的依赖）只是展望，planning 在正文里也只有定性展示。

**我读出来的**：
1. **模型类的限制**：spring-mass 加"首帧即 rest state"的设定，表达不了塑性（如橡皮泥）、断裂、流体和体积保持。如果首帧时物体已经受力（被压扁或悬挂着），rest length 本身就有偏差。论文列出的物体也只有绳子、毛绒玩具、布、快递包裹这几类以弹性为主的物体。
2. **接触模型**：控制点靠弹簧"粘"住附近的物体点，相当于假设抓取不打滑；摩擦、滑移、松手再抓的切换都没有建模。推（push）类交互具体怎么处理，正文没讲清楚。
3. **参数可辨识性**：论文没写质量 $m_i$ 怎么设定。运动只依赖 $\mathbf F/m$，所以能辨识出来的只有类似 $k/m$ 的比值（这是我的推断）。因此得到的"刚度"更接近等效参数，论文也没有拿真实材料测量来对照验证。
4. **评测口径**：只有 22 个场景，没有方差，也没有多次重复；2D 指标只在中间视角算，而这个视角也是训练视角之一，所以没有 novel view 的定量结果。Spring-Gaus 只统计成功的 case，失败率没有报告。两个 baseline 都是作者自己改造的。tracking 只用 9 个手工标注点。
5. **泛化范围**：只测了"同一物体、换一种人手交互"，而且新交互要先配准到目标首帧。没有测不同物体实例、不同环境（如桌面材质）、长时间 rollout 下的误差累积，也没有定量测机械臂执行的交互（只有 Fig. 5 的定性展示）。
6. **外观**：只在 $t=0$ 优化，Gaussians 是各向同性的，不建模光照和阴影随时间的变化。首帧看不到的那一面只能靠 shape prior 的几何，外观质量没有保证。
7. **算力与流水线**：论文没有报告优化耗时、仿真速度和 GPU 型号（repo 里的数字见"复现可行性"）。整个流水线依赖 TRELLIS、Grounded-SAM2、CoTracker3、SuperGlue、超分模型等多个组件，遇到透明或无纹理的物体、严重遮挡时，任何一环出错都会传下去，论文没有分析失败率。
8. **"2real"这一步缺证据**：论文既没有在 twin 里训练 policy 再部署到真机，也没有量化 planning 的效果。作为 real2sim2real 工作，从仿真回到真实世界这一段的证据最弱。

## 在脉络中的位置

- **前驱**：Spring-Gaus（spring-mass 加 3DGS，从视频学物理，是本文最直接的比较对象）；GS-Dynamics（同组的 Kaifeng Zhang、Yunzhu Li，用 GNN 从真实视频学动力学、用 Gaussian LBS 渲染，本文沿用了它的 LBS）；PhysGaussian、PhysDreamer、PAC-NeRF；可微软体多体仿真（Qiao et al. 2021）和 DiffPD 这类基于可微仿真的 system identification。用到的工具有 TRELLIS、CoTracker3、Grounded-SAM2、CMA-ES 和 NVIDIA Warp。思路上，[SimOpt](../simopt/notes.md) 同样根据真实轨迹与仿真的 discrepancy 反推仿真参数，也用 zero-order 采样来更新；PhysTwin 把对象换成可形变物体，参数从少量几维扩展到上万维，高维部分靠可微仿真做 first-order 优化。
- **后继 / 影响**：[Reality Gap 综述](../survey-reality-gap/notes.md) 把它归入 "Real-to-Sim First"（见 `survey-reality-gap/source/chapters/solutions.tex`）。以下信息来自官方 repo README，不是论文内容：2026-03 发布了 PhysTwin-MPC（基于 MPPI 的 planning 参考实现）；后续工作 Boba（ECCV 2026）做批量仿真，README 称其 demo 在 3 分钟内训出一个绳子操作的 RL policy。在本项目 related_work 的源码里，只有这篇综述引用了 PhysTwin。
- **与本项目其他主题的联系**：
  - **World model**：PhysTwin 可以看成一个显式的、带物理结构的 action-conditioned world model。 $f$ 对应 transition， $g_\theta$ 对应 decoder，Eq. 2 是"通过最小化 observation 重建误差来拟合 transition"，这和 [Dreamer](../../01_world_model/dreamer/notes.md) / [DreamerV3](../../01_world_model/dreamerv3/notes.md) 中 RSSM 的训练目标同构；不同的是，PhysTwin 的状态是可解释的质点，只从 1 段视频拟合，而且每个物体要单独构建。与学习式视频 world model（[UniSim](../../05_intersections/unisim/notes.md)、[Ctrl-World](../../05_intersections/ctrl-world/notes.md)、[Cosmos](../../01_world_model/cosmos-wfm/notes.md)）相比，两边的取舍正好相反：后者覆盖面广但不保证物理正确，前者数据效率高、物理一致，但能表达的物体类型受限。[Robotic World Model](../../05_intersections/robotic-world-model/notes.md) 处在神经网络 simulator 这一端；[SimDist](../../05_intersections/simdist/notes.md) 的方向相反，先在仿真里预训练 world model，再到真实世界快速适配。
  - **Real2sim2real**：[RialTo](../rialto/notes.md) 重建的是刚体和铰接物体的几何，PhysTwin 补上的是 physics gap；[SplatSim](../splatsim/notes.md) 同样用 Gaussian splatting，但目的是缩小 visual gap；[ASAP](../asap/notes.md) 不去辨识物理参数，而是用真实数据学一个 delta（residual）action model 来补偿动力学失配（SysID 是它的对比 baseline），PhysTwin 则是显式辨识物理参数、没有残差项；[RMA](../rma/notes.md) 做的是隐式的在线 system identification，PhysTwin 是显式的离线 system identification。附录 D 声称 PhysTwin 可以减少对 [domain randomization](../domain-randomization/notes.md) 的依赖，但没有做实验。[ACDC](../acdc-digital-cousins/notes.md) 主张用一组"cousins"扩大训练分布，与 PhysTwin 精确拟合单个 twin 是两种不同思路；[X-Sim](../x-sim/notes.md) 同样从人类视频做 real-to-sim。
  - **TTT**：构建一个 PhysTwin，就是针对单个物体做一次 per-instance 优化，结构上接近 [TTT-ARC](../../02_test_time_training/ttt-arc-fewshot/notes.md) 的 per-task fine-tune 和 [TTT-Discover](../../02_test_time_training/ttt-discover/notes.md) 讨论的单实例搜索，只是被更新的是物理参数 $\alpha$，不是网络权重。部署时用新观测持续更新 $\alpha$（online system identification）是一个自然的延伸，论文没有做。
  - **RL**：twin 可以直接作为 [PPO](../../04_rl/ppo/notes.md) 或 [SAC](../../04_rl/sac/notes.md) 的训练环境，也可以像 [World-Gymnast](../../05_intersections/world-gymnast/notes.md) 那样"在模型里"做 policy optimization。但本文只展示了定性的 planning。

## 前置知识

1. spring-mass-damper 系统与数值积分：explicit / semi-implicit Euler，以及时间步长与稳定性的关系。
2. system identification 与逆问题：可微仿真（让梯度穿过仿真器反传），以及接触带来的梯度不连续。
3. zero-order 黑箱优化：CMA-ES。
4. 3D Gaussian splatting 与 L1 + D-SSIM 损失；Linear Blend Skinning；ARAP 非刚性配准。
5. 视觉基础模型：CoTracker3（点跟踪）、Grounded-SAM2（开放词汇分割）、TRELLIS（image-to-3D 生成）、SuperGlue 与 PnP（匹配与位姿估计）。
6. 评测指标：单向 Chamfer distance、PSNR / SSIM / LPIPS、IoU；应用部分还需要了解 MPC / MPPI。

## 复现可行性

- **代码**：官方已开源，https://github.com/Jianghanxiao/PhysTwin（MIT 许可）。包含数据处理 `script_process_data.py`、zero-order 优化 `script_optimize.py`、first-order 优化 `script_train.py`、推理 `script_inference.py`、Gaussian 训练 `gs_run.sh`、评测 `evaluate.sh` 和 `interactive_playground.py`。环境是 Python 3.10 + CUDA 12.1（另有 RTX 5090 + CUDA 12.8 的安装路径和 Docker），依赖 Warp、TRELLIS、Grounded-SAM-2 和 Gaussian rasterizer；README 提醒 TRELLIS 与 Gaussian splatting 的 rasterizer 之间有冲突。原始数据、处理后的数据、zero-order 和 first-order 的优化结果、静态 Gaussians 都放在 Hugging Face 的 `Jianghanxiao/PhysTwin`。
- **算力**：**论文没有报告** GPU 型号、优化时长和仿真速度。以下来自 README，不是论文数字：first-order 优化约 5 min，CMA-ES 约 12 min（毛绒玩具场景，未开自碰撞）；在 RTX 4090 上，playground 的 sloth 场景约 37 FPS。一张消费级 GPU 就够用。
- **数据与硬件**：只复现论文数字的话，用官方数据即可，不需要相机；如果自己采数据，需要 3 台标定好的 D455 和人手交互。
- **最小可复现实验**：
  1. 下载官方的 data、experiments 和 gaussian_output，只跑 inference 和 `evaluate.sh`（README 的顺序是 `script_inference.py` → `gs_run_simulate.sh` → `export_render_eval_data.py` → `evaluate.sh`），对上 Table 1 中 PhysTwin 那一行。这一步先确认评测管线没问题。
  2. 选一个 case（例如 `double_stretch_sloth`），从头依次跑 CMA-ES 和 first-order 优化，复现 Table 3 的三种设置：只用 zero-order、只用 first-order、两阶段完整版。这是最小、也最能说明机制的消融。
  3. 进阶：取一对交互（在 A 上构建、在 B 上测试）复现 Table 2；或者把 twin 包装成一个 gym 环境，用 PPO 训练把绳子摆成目标形状的 policy。论文没做这件事，它正好是本项目几条线的交叉点。
- **预计难点**：多个依赖的 CUDA 编译；自采数据时的相机标定，以及 TRELLIS 配准失败；开启自碰撞的场景速度很慢（README 为此单独做了加速分支 `release_collision_accelerate`）；README 中未见 baseline 改造版本的代码，附录 C 只用文字描述了改法。

## 自测问题

1. 为什么拓扑参数（连接半径、最大邻居数）不能用 first-order 优化？如果去掉 zero-order 阶段，直接对稠密 $k_{ij}$ 做梯度下降，Table 3 里哪个任务退化最明显？为什么说这更像过拟合，而不是欠拟合？
2. $C_{\text{geometry}}$ 为什么用单向 Chamfer distance，而且是"从观测到预测"这个方向？如果改成双向，在部分观测下会出什么问题？
3. 把 Eq. 2 与 RSSM world model 的训练目标逐项对照：哪些部分对应 transition 和 decoder？PhysTwin 换来的数据效率，是用什么假设换的？
4. 手的 action 是怎么进入 $f_{\alpha,\mathcal G_0}$ 的？这种建模方式会让哪类操作（例如滑移、松手再抓）仿真失真？
5. 只在 $t=0$ 优化 Gaussians，渲染后续帧时靠什么让外观跟着形变？ $w_{jk}$ 和 $R_i^t$ 分别是怎么算出来的？
