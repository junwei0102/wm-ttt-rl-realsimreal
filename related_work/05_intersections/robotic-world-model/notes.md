# Robotic World Model: A Neural Network Simulator for Robust Policy Optimization in Robotics

> arXiv [2501.10100](https://arxiv.org/abs/2501.10100) · NeurIPS 2025 Workshop on Embodied World Models for Decision Making（Oral，已在 OpenReview 核实；workshop 论文，非主会。Outstanding Paper Award 只见于第一作者主页，证据较弱。arXiv v5 2025-12；PDF 页脚为 "Preprint. Under review."，NeurIPS 2025 模板，论文正文没有写录用信息）· 主题：05_intersections · 层级：milestone
> 代码：https://github.com/leggedrobotics/robotic_world_model（论文正文只给了项目页 https://sites.google.com/view/roboticworldmodel；仓库已用 WebFetch 确认存在；GitHub 识别的 LICENSE 为 Apache-2.0，但 README 徽章写的是 MIT）· 笔记依据：`source/*.tex`（数字/公式）+ `paper.pdf`（图）

## 一句话

RWM 用一个 GRU 做 world model，输入是低维本体观测，训练时不用 teacher forcing，而是在 $M$ 步真实历史之后把自己的预测接回输入、连续自回归 $N$ 步，并对整段 multi-step prediction error 做反传。这样训出来的神经网络模拟器能稳定 rollout 100 步以上，足以直接跑 PPO（MBPO-PPO）；在其中训出的速度跟踪策略 zero-shot 部署到了 ANYmal D 和 Unitree G1 上。

## 问题设定

- **环境**：POMDP $(\mathcal S,\mathcal A,\mathcal O,T,R,O,\gamma)$，policy $\pi_\theta:\mathcal O\to\mathcal A$，目标是最大化 $\mathbb E_{\pi_\theta}[\sum_t\gamma^t r_t]$（`source/sections/approach.tex` §3.1）。
- **World model 的输入/输出**（Table S2–S4）：
  - 观测 $o$：base 线速度 $v$、角速度 $\omega$、projected gravity $g$、关节位置 $q$、关节速度 $\dot q$、关节力矩 $\tau$。ANYmal D 共 45 维，G1 共 96 维。
  - 动作 $a$：关节位置目标 $q^*$，ANYmal 12 维，G1 29 维。
  - privileged information $c$ 是额外的预测目标。ANYmal 为膝/足接触，共 8 维；G1 为 body contact、足高、足速，共 30 维（Table S3）。此外还预测 termination，即 base 触地（§A.4.3）。
  - 控制频率 50 Hz（$\Delta t=0.02$）。
- **Reward**：不学 reward model。用手写的速度跟踪 reward（§A.1.2 给出 12 项公式，权重见 Table S6；是否直接沿用 Isaac Lab 默认任务配置，原文未说明），直接在**预测出的** $o'$ 和 $c'$ 上计算（"Rewards are computed from imagined observations and privileged information"）。
- **之前的方法卡在哪**（Intro / Related work；第 4 条来自 §4.4）：
  1. 带 domain-specific inductive bias 的模型（足端落点、刚体、Lagrangian 等）难以推广到新任务。
  2. 用 teacher forcing 训练的通用模型（RSSM、Transformer、MLP）在长程自回归时有 compounding error。
  3. MBPO、Dreamer、TD-MPC 只用短 rollout，没法撑起 PPO 这类需要长 on-policy 轨迹的算法。
  4. 一阶方法（SHAC）在接触不连续的腿足动力学上梯度不可靠。

## 核心方法

**1. 自回归训练目标**（`source/sections/approach.tex` Eq. 1）。每条训练样本是轨迹上一个长度为 $M+N$ 的滑窗。前 $M$ 步用真实 $(o,a)$ 做上下文，之后第 $k$ 步的预测为

$$o'_{t+k}\sim p_\phi\!\left(\cdot\mid o_{t-M+k:t},\;o'_{t+1:t+k-1},\;a_{t-M+k:t+k-1}\right)$$

也就是说，第 $k$ 步看到的"历史"里已经混进了前 $k-1$ 步的**自己的预测**。privileged information $c'_{t+k}$ 也按同样方式预测。损失函数（Eq. 2）：

$$\mathcal L=\frac1N\sum_{k=1}^{N}\alpha^k\Big[L_o(o'_{t+k},o_{t+k})+L_c(c'_{t+k},c_{t+k})\Big]$$

其中 $\alpha$ 是 forecast decay，实际取 $\alpha=1.0$（Table S10），所以各步等权。$L_o$、$L_c$ 的具体形式论文没写。网络输出 Gaussian 的均值和标准差，采样时用 reparameterization trick，让梯度能穿过整条自回归链。teacher forcing 就是 $N=1$ 的特例（Fig. 2b）：它可以高度并行（原文 "highly parallelized"），但训练时从没见过自己的误差，也就是训练分布与推理分布之间存在 mismatch。

**2. "dual-autoregressive" 结构**（§3.2 末段，Fig. S6）。下面是笔记按 Fig. S6 整理的写法，不是论文公式：

- **inner autoregression**：在 $M$ 步历史上逐步更新 GRU 隐状态，$h_\tau=\mathrm{GRU}(h_{\tau-1},[o_\tau,a_\tau])$。这就是普通 RNN 的 recurrence，作用是从历史里隐式捕捉不可观测的动力学（原文 "captures unobservable dynamics"）。
- **outer autoregression**：由 $(\mu,\sigma)=\mathrm{MLP}(h_\tau)$ 得到 $o'_{\tau+1}=\mu+\sigma\odot\epsilon$，再把 $[o'_{\tau+1},a_{\tau+1}]$ 喂回 GRU，重复 $N$ 步。

网络结构（Table S7）是 GRU 256-256，外加 MLP 128 的输出 heads。注意：这里的 inner/outer 与 TTT 里的 inner loop/outer loop 是两回事，不涉及测试时的梯度更新。

**3. MBPO-PPO**（Algorithm 1，Eq. 3，Fig. S7）。循环执行：

1. 用 $\pi_\theta$ 在环境中收集数据，存入 $\mathcal D$（单个 agent）。
2. 用 Eq. 2 更新 $p_\phi$。
3. 从 $\mathcal D$ 采样观测，初始化 4096 个 imagination 环境。
4. 按 $a'_{t+k}\sim\pi_\theta(\cdot\mid o'_{t+k})$ 与 $p_\phi$ 交替 rollout $T$ 步，每次迭代 100 步（Table S11）。
5. 在预测观测上算 reward，把预测出的 termination 当作 episode 结束（影响 return 和 value），然后做 PPO 更新。

policy 和 value 都是 MLP 128×3 + ELU（Table S9），policy 无记忆。PPO 超参见 Table S11：lr 1e-3、5 epochs、4 mini-batches、KL target 0.01、$\gamma=0.99$、clip 0.2、entropy 0.005。

**4. 预训练 + 在线微调**（§A.4.3–A.4.4）。locomotion 任务中，RWM 先用"为相似任务训练、在不同动力学下运行的策略"产生的**仿真数据**预训练。之后 policy 完全在 imagination 里从零学起，RWM 只用**单个**仿真环境的在线数据微调（这个环境相对预训练环境有 domain shift，用来近似真实世界的数据约束）。manipulation 任务不需要预训练。

## 关键实验与证据

- **长程预测**：用 ANYmal D 硬件轨迹评估（§4.1，Fig. 3a）。$M=32$、$N=8$，从 $t=32$ 开始自回归到 $t=200$，远超训练时的 $N$，预测曲线仍贴合真值。
- **噪声鲁棒性**（Fig. 3b）：在观测和动作上加 Gaussian 噪声，噪声尺度 0–0.8。同样用 AR 训练、同样 $M,N$ 的 MLP 误差随步数发散得明显更快。
- **跨环境泛化**（Fig. 4）：共 15 个 Isaac Lab 任务，包括 Reach-UR10/Franka、Lift/Open-Drawer-Franka、Repose-Cube-Allegro，以及 A1、Go1、Go2、ANYmal B/C/D、Spot、Cassie、H1、G1 的速度跟踪。RWM-AR 在所有环境上误差 $e$ 都最低（目测均 <1），MLP 在 Velocity-G1 上接近 10。**关键对照**：RWM-TF 与 RSSM、Transformer 同量级，说明大头来自 AR 训练而不是 GRU 结构。作者自己也写到，RSSM 换成 AR 训练后与 RWM 相当；Transformer 做 AR 训练时，多步反传的显存开销太大（§4.3 末段）。注意这里的 baseline 都按惯例用 teacher forcing 训练。
- **$M/N$ 消融**（Fig. S8 热图，数值从 PDF 读出；完整网格是 $M,N\in\{1,2,8,16,32\}$，下表只节选其中 12 格）：

| $e$ / 训练小时 | $M=1$ | $M=8$ | $M=32$ |
|---|---|---|---|
| $N=1$（TF） | 79.03 / 0.52 | 14.88 / 0.58 | 3.99 / 0.62 |
| $N=2$ | 21.78 / 0.70 | 5.44 / 0.78 | 1.91 / 1.05 |
| $N=8$ | 10.58 / 1.00 | 0.54 / 1.06 | **0.47 / 1.07** |
| $N=32$ | 1.50 / 1.97 | 0.49 / 2.04 | 0.47 / 2.27 |

  可以读出两点。第一，$N$ 的作用最大：$M=1$ 时只要把 $N$ 提到 32，$e$ 就从 79 降到 1.5。第二，$M=32$ 时 $N=8$ 与 $N=32$ 误差一样，训练时间却只有约一半，这就是默认设置的来源。
- **策略学习**（Fig. 5）：ANYmal D 和 G1 速度跟踪，对比 SHAC 与 DreamerV3。
  - MBPO-PPO 的 model error 很快下降。预测 reward 早期高于 ground-truth reward（policy 在利用模型的乐观误差），随训练逐渐对齐（按正文描述；目测 G1 右图到 10000 次迭代时预测 reward 仍比 GT 高几分）。
  - SHAC 的 model error 高且震荡，训练不收敛。作者的解释：一阶梯度在接触不连续处不可靠，训出的混乱行为又污染了数据。
  - Dreamer 只部分收敛（目测 ANYmal 上 GT reward 约 12，MBPO-PPO 约 20）。
  - SHAC 和 Dreamer 都没得到可部署的策略。MBPO-PPO 的策略 zero-shot 部署到两台硬件上，能扛外部冲击和地形变化（Fig. 1 与视频，均为定性展示）。
- **与 model-free 的对照**（Table 1）：

| | RWM 预训练 | MBPO-PPO | PPO（高保真仿真） |
|---|---|---|---|
| state transitions | 6M | – | 250M |
| 训练时间（total training time） | 50 min | 5 min | 10 min |
| step inference time | – | 1 ms | 1 ms |
| 真机 tracking reward | – | 0.90 ± 0.04 | 0.90 ± 0.03 |

- **算力**：单张 RTX 4090，PyTorch 2.4.0，CUDA 12.6（§A.3）。RWM 训练约 1 h（Table S10），实验跑 5 个 seed。

## 局限与论文没说的

**作者承认的**（§5，§A.4.3–A.4.4）：
- 性能仍不及在高保真仿真器上调好的 model-free RL。
- 从零训练不稳定，policy 会利用模型误差，所以 locomotion 依赖仿真数据预训练。
- 没有做真机在线学习。仿真里的在线学习阶段，policy 利用模型误差导致的失败（collision）平均超过 20 次，还需要自动复位策略；privileged information（接触）在真机上需要测量或估计。
- 缺少不确定性估计。

**自己读出来的**：
1. **真实数据从未进入硬件实验的训练环路。** 预训练和"单环境在线微调"用的都是 Isaac Lab 数据，所以严格说它是"把物理仿真器蒸馏成 GRU，再在 GRU 里跑 PPO"，sim-to-real 的鲁棒性很可能继承自预训练数据的动力学多样性（类似 domain randomization）。论文没有消融这一点。
2. **样本效率的口径不对等。** Table 1 的 6M 对 250M 没有计入"生成预训练数据的那些策略"本身消耗的仿真交互，MBPO-PPO 那一栏的在线数据量也写的是"–"。真机 reward 与 PPO 持平（0.90），这和 §5 说的"仍不及 model-free"也对不上；Table 1 也没有说明是哪台机器人、多少次试验。
3. **误差指标 $e$ 没有定义。** 正文只叫它 "relative prediction error"，没给公式，也没说统计了多长的 horizon、如何归一化。Fig. S8 用的是哪个环境也没写。
4. Fig. 4 的比较混合了"架构"和"训练方式"两个变量，baseline 全用 TF。真正公平的比较（全部 AR）只有一句文字描述，没有给数字。
5. **超参有出入。** Table S11 的 max iterations 是 2500，但 Fig. 5 中 G1 的横轴到了 10000。
6. **适用范围窄。** 只做了低维本体观测的速度跟踪策略（policy 观测不含地形感知，Table S5；训练地形原文没有明说，Fig. 1 的仿真画面是平地，仓库也主要是 ANYmal D flat），没有报告视觉、崎岖地形或 manipulation 的策略学习（manipulation 只出现在 Fig. 4 的预测误差比较和 Fig. 1 / Fig. S9 的 imagination 可视化里；§A.4.3 提到 manipulation 实验不需要预训练，但没有给出 manipulation 策略的结果）。reward 必须能从预测量算出来，G1 就不得不额外预测足高、足速。
7. **imagination 怎么初始化没交代。** 初始状态从 $\mathcal D$ 采样，但 GRU 的 $M$ 步历史在 imagination 起点如何 warm-up，论文没有细说。

## 在脉络中的位置

- **前驱**：
  - Dyna/MBPO 的"真实数据学模型 + 模型里做 model-free RL"框架（MBPO 不在本项目里）。
  - [World Models](../../01_world_model/world-models/notes.md) 与 [DIAMOND](../../01_world_model/diamond/notes.md) 的"完全在 world model 里训 agent"。
  - [Dreamer](../../01_world_model/dreamer/notes.md) / [DreamerV3](../../01_world_model/dreamerv3/notes.md) 的 RSSM，这里既是模型 baseline，也是策略 baseline。
  - [PPO](../../04_rl/ppo/notes.md)。
  - 同一组（ETH RSL）的 [legged gym](../../03_real2sim2real/legged-gym/notes.md)：4096 个并行环境跑 PPO，以及速度跟踪 reward 的设计。RWM 等于把其中的物理引擎换成了 GRU。
- **后继 / 影响**：
  - 官方仓库还包含同一作者的 RWM-U（Uncertainty-Aware RWM），即 offline MBRL、不访问仿真器，对应 arXiv 2504.16680（[SimDist](../simdist/notes.md) 的 `references.bib` 以旧标题 "Offline Robotic World Model: Learning Robotic Policies without a Physics Simulator" 引用它；仓库 README 里的标题是 "Uncertainty-Aware Robotic World Model Makes Offline Model-Based Reinforcement Learning Work on Real Robots"，策略算法为 MOPO-PPO）。
  - [SimDist](../simdist/notes.md) 在 related work 里批评这类方法依赖低维状态。它的做法是在仿真里预训练 world model，到真实世界只更新 latent dynamics。
  - [Reality Gap 综述](../../03_real2sim2real/survey-reality-gap/notes.md) 在 open problems 中把 RWM 作为 learnable simulator 的例子。
- **与本项目其他主题的联系**：
  - **World model**：与 [TD-MPC2](../../01_world_model/tdmpc2/notes.md) 走 MPC 短程规划的路线相反，RWM 押注长程 rollout 的保真度。它与 [DayDreamer](../daydreamer/notes.md) 对照最有意思：DayDreamer 真的在真机上在线学习，RWM 因为安全原因没有做。
  - **Real2sim2real**：GRU 从 $M=32$ 步历史里推断动力学，作用类似 [RMA](../../03_real2sim2real/rma/notes.md) 的 adaptation module。[ASAP](../../03_real2sim2real/asap/notes.md) 用真实数据修正仿真器，RWM 则是在原理上可以用真实数据学整个仿真器，但本文没有这么做。
  - **TTT**：RWM 部署时既不更新模型也不更新策略。"单环境在线微调 world model"近似于 system identification。如果要在部署时做 TTT 式适应，可以参考 [PAD](../pad/notes.md) 的自监督在线更新。
  - **RL on video world models**：[WMPO](../wmpo/notes.md)、[World-Gymnast](../world-gymnast/notes.md)、[Ctrl-World](../ctrl-world/notes.md) 在像素级 world model 里给 VLA 做 RL。RWM 是低维本体观测版本，rollout 只要毫秒级（Table 1 中 step inference 1 ms）。

## 前置知识

1. Model-based RL 基础：Dyna/MBPO、compounding error、model exploitation（[RL overview](../../04_rl/survey-rl-overview/notes.md)）。
2. [PPO](../../04_rl/ppo/notes.md)：clip、GAE、adaptive KL learning rate。另外要理解为什么 PPO 需要长 on-policy rollout。
3. RNN/GRU 序列建模中的 teacher forcing 与自回归训练，以及训练/推理分布的 mismatch。
4. Gaussian 输出头与 reparameterization trick。
5. 腿足 locomotion RL 的惯例：速度跟踪 reward、projected gravity、privileged information、termination（[legged gym](../../03_real2sim2real/legged-gym/notes.md)）。
6. 对照用：RSSM（[DreamerV3](../../01_world_model/dreamerv3/notes.md)）；SHAC 这类一阶可微 policy optimization。

## 复现可行性

- **代码**：官方仓库 `leggedrobotics/robotic_world_model` 已开源。据仓库 README 徽章，它基于 Isaac Lab 2.1.0、Isaac Sim 4.5.0、Python 3.10，并需要用 model-based RSL-RL fork（`rsl_rl_rwm`）替换 Isaac Lab 自带的 `rsl_rl_lib`；平台徽章只有 linux-64 和 windows-64（README 没写 GPU 要求，但 Isaac Sim 本身需要 NVIDIA RTX GPU），**macOS 跑不了**。提供 init、pretrain、finetune、visualize、play 等任务变体，参考 pipeline 只针对 ANYmal D（flat）。
- **算力**（论文）：单张 RTX 4090。RWM 约 1 h（Table S10），或者按 Table 1 的 50 min；MBPO-PPO 5 min；在 5 个 seed 上汇报。
- **最小复现实验**：在 Isaac Lab 里用现成的 ANYmal-D flat 速度策略采集轨迹，只训练 world model，比较 $M=32$ 下 $N\in\{1,2,8\}$ 三个设置。评估从第 32 步起自回归约 170 步的误差，看是否复现 Fig. S8 这一列的趋势（3.99 → 1.91 → 0.47），以及 N=8 约 1 h 的训练时间。三个模型在一张 4090 上大约 3 小时。第二步再跑官方 pretrain → MBPO-PPO，把 ground-truth reward 曲线和 Fig. 5 左图对照。
- **无 GPU 的退路**：在 MuJoCo 的低维 locomotion 环境上自己写 GRU + AR 损失，只验证"TF 对 AR 的误差曲线"这个定性结论。这样复现不了论文里的数字。
- **难点**：
  - $e$ 的定义不明，数值不一定能直接对上。
  - 预训练数据的来源（哪些策略、哪些动力学范围）论文描述很粗。
  - 真机部署需要 ANYmal 或 G1 以及状态估计。

## 自测问题

1. 为什么 teacher forcing 是 $N=1$ 的特例？它在 imagination 里具体怎样导致 compounding error？Fig. S8 里 "$M=1,N=32$ 的 $e=1.50$" 与 "$M=32,N=1$ 的 $e=3.99$" 分别说明了 history 和 forecast 各自的作用是什么？
2. privileged information head（接触、termination）在 MBPO-PPO 里起哪两个作用？为什么它能在不做 reward model 的情况下让 reward 可算？
3. 作者为什么认为 SHAC 在腿足任务上会失败，而零阶的 PPO 在同一个 learned model 上能成功？数据质量又怎样反过来影响了 model error？
4. 为什么 locomotion 需要仿真数据预训练而 manipulation 不需要？这对"sim-to-real 到底是谁的功劳"这个问题意味着什么？
5. Table 1 的 6M 对 250M transitions 为什么不能直接解读为"样本效率高 40 倍"？要想在真机上做在线学习，还需要补哪些组件？
