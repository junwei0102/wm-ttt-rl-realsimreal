# π_RL: Online RL Fine-tuning for Flow-based Vision-Language-Action Models

> arXiv [2510.25889](https://arxiv.org/abs/2510.25889) · arXiv preprint, 2025（本地 source 为 v3，2026-01-29，ICML 2026 模板的 preprint 版式，未写明录用） · 主题：04_rl · 层级：frontier
> 代码：https://github.com/RLinf/RLinf（Apache-2.0；模型发布在 https://huggingface.co/RLinf） · 笔记依据：`source/*.tex`（数字/公式）+ `paper.pdf`（图）

## 一句话

flow matching VLA（π0 / π0.5）的动作由确定性 ODE 积分得到，log-likelihood 算不出、也没有探索。π_RL 的做法是把去噪的每一步改成一个高斯转移：Flow-Noise 用一个可学习的噪声网络，Flow-SDE 把 ODE 换成边缘分布相同的 SDE。这样 $\log\pi$ 就是若干个高斯 log-prob 之和，可以直接代进 PPO，在仿真中在线微调 action expert。

## 问题设定

- **输入/输出**：观测 $\mathbf{o}_ t$ 包括 RGB（第三视角/腕部）、语言指令和本体感知。按 openpi 官方设置，π0.5 不输入 state（App. B.2；但 Sec.4.3.2 又写 π0.5 的 state 并入 VLM 的 prompt embedding，论文前后表述不一致）。输出是 action chunk $\mathbf{A}_ t=[a_{t,0},\dots,a_{t,H-1}]$，每次只执行前 $H'$ 步（replan horizon，Table 11–13）。
- **环境与奖励**：只在仿真中训练，包括 LIBERO、ManiSkill、MetaWorld、CALVIN，另有 SIMPLER 和一个 3DGS 渲染的 Real2Sim2Real 场景。奖励基本都是稀疏的：LIBERO/MetaWorld 成功得 1；ManiSkill 放对得 1.0，抓住物体额外得 0.1；CALVIN 每完成一个子任务得 1.0（App. C）。假设状态完全可观测， $s_t=\mathbf{o}_ t$（`main.tex` Sec.3.1）。
- **训练范式**：pretrain → SFT（全参数，3.3B）→ RL。论文称为 few-shot SFT 的只有 LIBERO、Real2Sim2Real（20 条）和 GR00T 实验；ManiSkill 用 16,384 条合成轨迹，MetaWorld 用官方全部 2,500 条，CALVIN 用整个 ABC 数据集（App. C），SIMPLER 每个任务 144 条（App. E.1）。RL 阶段**冻结 VLM，只训练 300M 的 action expert**，理由是显存，以及 RL4VLA 的发现“RL 主要改善动作泛化”（App. B.2）。
- **之前的方法卡在哪**：SimpleVLA-RL、RL4VLA、RLinf-VLA 等 VLA-RL 工作针对的是 OpenVLA / OpenVLA-OFT。离散动作可以直接对 logits 做 softmax 得到 log-prob，连续动作可以接一个 Gaussian head。flow VLA 的动作则是 $K$ 步 Euler 积分的确定性输出，带来两个问题：(1) 没有随机性，无法探索；(2) 精确 likelihood 要做 Hutchinson trace 的 divergence 积分，在 $K$ 很小时算不准（Sec.4 开头）。

## 核心方法

**背景（flow VLA，`main.tex` Sec.3.2，Eq.3）**：路径为 $\mathbf{A}^\tau=\tau\mathbf{A}+(1-\tau)\epsilon$， $\tau=0$ 时是纯噪声。训练用 CFM loss
$$\mathcal{L}_ {\text{CFM}}=\mathbb{E}\big\Vert\mathbf{v}_ \theta(\mathbf{A}^\tau,\mathbf{o})-(\mathbf{A}-\epsilon)\big\Vert_2^2,$$
推理时做 Euler 积分 $\mathbf{A}^{\tau+\delta}=\mathbf{A}^\tau+\mathbf{v}_ \theta\thinspace\delta$，其中 $\delta=1/K$。

**共同思路**：把每个去噪步都写成 $p(\mathbf{A}^{\tau+\delta}\mid\mathbf{A}^\tau,\mathbf{o})=\mathcal{N}(\mu_\tau,\Sigma_\tau)$。于是 log-prob 有闭式，梯度可以经 $\mu_\tau$ 回传到 $\mathbf{v}_ \theta$。

**1. Flow-Noise（one-layer MDP，思路来自 ReinFlow；Sec.4.1，Eq.4–5）**
$$\mu_\tau=\mathbf{A}^\tau+\mathbf{v}^\tau\delta,\qquad \Sigma_\tau=\mathrm{diag}\big(\sigma^2_{\theta'}(\mathbf{A}^\tau,\mathbf{o})\big)$$
$$\log\pi(\mathcal{A}\mid\mathbf{o})=\log\Big(\pi(\mathbf{A}^0\mid\mathbf{o})\prod_{k=0}^{K-1}\pi(\mathbf{A}^{\tau_{k+1}}\mid\mathbf{A}^{\tau_k},\mathbf{o})\Big)$$
- 均值就是原来的 Euler 步，没有漂移修正，所以注入噪声后边缘分布会偏离原分布（笔者理解，论文没有这样论证）。噪声的上下界设得很小：LIBERO 上 π0 的 log-var 上下界为 [0.08, 0.16]，π0.5 为 [0.04, 0.10]，另加 entropy bonus 0.005（Table 11；其他 benchmark 见 Table 12–13，例如 MetaWorld 的 π0 用 [0.04, 0.10]）。
- 噪声网络 $\theta'$ 和 velocity 一起训练，**推理时丢掉**，退回确定性 ODE。
- MDP 仍是环境层的 one-layer MDP。PPO ratio 用的是**整条去噪链的联合 likelihood**，理论依据见 ReinFlow。 $\pi(\mathbf{A}^0\mid\mathbf{o})=\mathcal{N}(0,I)$ 与参数无关。
- 代价：每次 update 都要重算整条链的 log-prob。

**2. Flow-SDE（two-layer MDP，思路来自 Flow-GRPO + DPPO；Sec.4.2，Eq.6–12）**
- ODE $d\mathbf{A}^\tau=\mathbf{v}^\tau d\tau$（Eq.6）换成边缘分布相同的 SDE（Eq.7，依据 Song et al. 的 probability flow）： $d\mathbf{A}^\tau=\big(\mathbf{v}^\tau-\tfrac12 g^2(\tau)\nabla\log q_\tau\big)d\tau+g(\tau)d\mathbf{w}$。
- 代入 Flow-GRPO 的 score–velocity 关系 $\nabla\log q_\tau=-\frac{\mathbf{A}^\tau}{\tau}-\frac{1-\tau}{\tau}\mathbf{v}^\tau$，并取 $g=\sigma_\tau=a\sqrt{\tau/(1-\tau)}$，得到 Eq.8：
$$d\mathbf{A}^\tau=\Big[\mathbf{v}^\tau+\tfrac{\sigma_\tau^2}{2\tau}\big(\mathbf{A}^\tau+(1-\tau)\mathbf{v}^\tau\big)\Big]d\tau+\sigma_\tau d\mathbf{w}_ \tau$$
- 离散化后（Eq.9）： $\mu_\tau=\mathbf{A}^\tau+[\cdot]\thinspace\delta$， $\Sigma_\tau=\sigma_\tau^2\delta\thinspace\mathbf{I}$。这是各向同性的固定噪声，噪声水平只由一个标量 $a$ 控制（Table 11–13 中记作 noise level σ，一般取 0.5；π0.5 的 LIBERO-Object/Goal 取 0.3）。
- **推导注意（笔者推导，不是论文原文）**：Eq.7–8 的符号和 $\sigma_\tau$ 用的是 Flow-GRPO 的约定，即 $\tau=1$ 为噪声、去噪时 $\tau$ 递减。这与 Sec.3.2 的写法相反（ $\tau=0$ 为噪声、 $\delta>0$）。如果按 Sec.3.2 的约定自己推，score 应为 $-(\mathbf{A}^\tau-\tau\mathbf{v})/(1-\tau)$，对应的 SDE 是 $d\mathbf{A}=[\mathbf{v}-\frac{\sigma^2}{2(1-\tau)}(\mathbf{A}-\tau\mathbf{v})]d\tau+\sigma d\mathbf{w}$，其中 $\sigma=a\sqrt{(1-\tau)/\tau}$。在白板上推导前要先统一约定。
- **Two-layer MDP（Eq.10–12，Fig.3）**：
  - 状态 $\bar s_t^\tau=(\mathbf{o}_ t,\mathbf{A}_ t^\tau)$。
  - 内层（ $\tau<1$）：动作为 $\mathbf{A}_ t^{\tau+\delta}=\mu_\tau+\sigma_\tau\sqrt\delta\thinspace\epsilon$，观测不变，奖励为 0。
  - 外层（ $\tau=1$）：执行 $\mathbf{A}_ t^1$，得到 $R_{\text{ENV}}$ 和 $\mathbf{o}_ {t+1}$，并把 $\mathbf{A}_ {t+1}^0$ 重置为 $\mathcal{N}(0,I)$。
  - 于是 $\log\pi(a_t|s_t)$ 变成单步高斯的 $\log\pi(\bar a_t^\tau|\bar s_t^\tau)$。
- **Hybrid ODE-SDE（Sec.4.2.3）**：two-layer MDP 会把 horizon 放大约 $K$ 倍（按 MDP 定义推出；原文只说 significantly extends the horizon）。解决办法是在每个环境步随机抽**一个** $\tau_t$ 做 SDE 采样，其余步走确定性 ODE，由 env wrapper 执行。这样每个环境步只留下一个带 log-prob 的决策。

**3. 策略优化（Sec.4.3，Eq.13–16）**
- **Chunk-level**：整个 chunk 视为一个宏动作，奖励 $R_t=\sum_{j}r_{t,j}$。
- GAE： $\hat A_t=\sum_k(\gamma\lambda)^k\mathcal{T}_ {t+k}$，其中 $\mathcal{T}_ t=R_t+\gamma V(s_{t+1})-V(s_t)$。
- PPO clip： $\mathbb{E}[\min(\rho\hat A,\mathrm{clip}(\rho,1\pm\epsilon)\hat A)]$。ratio 在 Flow-Noise 中取 $\pi(a_t|s_t)$ 之比（整条去噪链），在 Flow-SDE 中取 $\pi(\bar a^\tau_t|\bar s^\tau_t)$ 之比。
- 超参： $\gamma=0.99$， $\lambda=0.95$， $\epsilon=0.2$，actor lr 5e-6，critic lr 1e-4（LIBERO，Table 11）。
- **Critic**：actor 和 critic 共享主干。
  - π0.5：critic 接在 VLM 输出上，得到 $V_{\text{vlm}}(\mathbf{o}_ t)$。
  - π0：state 只从 action expert 输入。为了让 value 估计包含 state，作者把 critic 挂在 expert 上（并非只能如此：Sec.5.3.1 的消融里挂在 VLM 上的 $V_{\text{vlm}}$ 在 π0 上反而略好，但作者仍保留 $V_{\text{expert}}$）。expert 的输入里有噪声动作，于是用 $V_{\text{expert}}(\mathbf{o}_ t)\approx\mathbb{E}_ {\tau\sim U[0,1]}[V_{\text{expert}}(\mathbf{o}_ t,\mathbf{A}_ t^\tau)]$ 近似（Eq.16）。
  - critic head 用四层 MLP。

**训练/推理流程**：rollout 时用随机策略（SDE 或带噪声网络），在 GPU 上并行跑仿真，记录 old log-prob 和奖励；接着用 GAE 算 advantage，再用 PPO 更新 action expert、critic（以及噪声网络）。环境、rollout 模型和 actor 共置在同一块 GPU 上串行执行（App. B.2）。**评测**一律用确定性 ODE（ $\sigma=0$）。

## 关键实验与证据

**主结果（Table 1；CALVIN 列就是 Table 5 的 Len-5 成功率）**
- π0：SFT 平均 51.1，Flow-SDE 78.7，Flow-Noise 80.3（+29.2）。
- π0.5：SFT 平均 55.6，Flow-SDE 86.6（+31.0），Flow-Noise 84.7。

**LIBERO（Table 3，每个 suite 500 个初始状态）**
- π0：从 1,692 条 demo 里只取 58 条做 SFT（Long 用 208 条），结果 57.6 → Flow-SDE 96.1 / Flow-Noise 97.6，超过全量 SFT 的 π0（94.2）。
- π0.5：只用 40 条（4 个 suite × 10 个子任务，每个子任务 1 条，Fig.1 称为 one-shot），结果 77.1 → 97.9 / 98.3，全量 SFT 为 96.9。其中 Long 从 43.9 提到 94.0（Flow-Noise；Flow-SDE 为 93.0）。

**ManiSkill（Table 4，4,352 种任务组合，SFT 数据为 16,384 条 MPLib 轨迹）**
- IND：π0 从 38.4 到 78.8 / 77.8；π0.5 从 40.1 到 90.9 / 89.7。
- OOD 平均（π0.5）：26.4 → 49.3 / 53.4。
  - 分项看（Flow-SDE / Flow-Noise）：Vision 40.2 → 68.0 / 69.9；Semantic 16.6 → 34.5 / 35.5；Execution 22.4 → 45.4 / 54.9。
  - 细分结果见 Table 7。

**CALVIN（Table 5，Scene D，1,000 episodes）**
- π0.5 + Flow-SDE：平均完成子任务数 3.838 → 4.717，Len-5 从 61.3 → 87.0。
- π0 提升很小：+0.178。
- 越长的序列提升越大：Len-1 +7.0，Len-5 +25.7。

**MetaWorld MT50（Table 6）**：π0 从 50.8 → 85.8（Flow-Noise）；π0.5 从 43.8 → 70.7（Flow-SDE）；SmolVLA 为 68.2。

**其他**
- SIMPLER（Table 8，Flow-Noise）：π0 67.2 → 86.7；π0.5 59.2 → 79.1。
- GR00T N1.5（Table 10，Flow-SDE）：52.5 → 89.9。需要把 expert 里的 dropout 换成 identity，否则 ratio 不稳定（App. H）。

**Real2Sim2Real（App. E.2）**
- 设置：Franka + RealSense D435，ManiSkill 负责物理、3DGS 负责渲染，手工标定外参。
- 流程：先用 20 条 motion planner 轨迹做 SFT，再做 100 次 RL 迭代，然后零样本部署到真机。
- 结果：SFT 无法完成任务，RL 后达到 40%。论文没有给出试验次数。

**OOD（Fig.5 / App. D）**
- CALVIN ABC→D：61.3 → 79.1。
- MetaWorld ML45 的 5 个 held-out 任务：成功率在训练中一直振荡，没有明显提升，但也没有遗忘 SFT 阶段学到的能力。
- 作者的结论：RL 改善的是动作层面（视觉、执行扰动），**不能泛化到新任务目标**。

**消融（全部用 Flow-SDE）**
- **PPO vs GRPO（Table 9）**：π0 为 96.0 vs 90.0，π0.5 为 97.9 vs 91.5。π0 的差距主要在 Goal（96.2 vs 83.2）和 Long（90.2 vs 81.4）；π0.5 的差距主要在 Long（93.0 vs 77.6）。
- **MDP 形式（Fig.7，LIBERO-Goal）**：最终成功率三者相近，one-layer 收敛最快。Update time 分别为：two-layer 814.2 s，hybrid 428.6 s，one-layer 821.4 s，即 hybrid 约快 2×。one-layer 不省时间，因为每次都要重算整条链。
- **噪声注入方式（Fig.8）**：固定噪声 0.5 与可学噪声（log-var 0.08–0.16，entropy bonus 置 0）效果相当。
- **Critic（Fig.6，π0，LIBERO-Long）**： $V_{\text{vlm}}$ 略好，value loss 更低，explained variance 更高；四层 MLP 优于一层。
- **超参（Table 2，LIBERO-Spatial，100 步）**
  - 噪声水平 $a$ = 0.2 / 0.5 / 0.8：SFT 的 train（带噪）成功率 62.3 / 56.0 / 46.6，eval 恒为 65.2；RL 后 eval 为 73.1 / 94.5 / 98.1。 $a=0.2$ 时 clip fraction 明显偏高，训练不稳（Fig.9）。
  - 去噪步数 $K$ = 1 / 2 / 4 / 8：SFT train 为 9.4 / 28.3 / 56.1 / 62.6，说明 $K$ 小时 ODE→SDE 的离散误差很大；RL 后 eval 为 88.5 / 97.0 / 94.5 / 86.7， $K$ 越大反而越难训。
  - chunk = 5 / 10 / 20：RL 后 eval 为 94.5 / 95.5 / 89.2。chunk 越大 explained variance 越低（Fig.11），RL 能提升的上限也越低。
- **VLM LoRA（Fig.13，r=32）**：在 LIBERO 上看不出收益，而且需要更保守的学习率（1e-6，每个 epoch 2 次 update）。
- **cosine lr scheduler（Fig.14）**：π0.5 在 LIBERO-Long 上 KL 会持续上升，加 scheduler 后得到抑制。

## 局限与论文没说的

**作者承认的（Sec.6，App. I）**
- online RL 的样本效率低，所以只能先在仿真里训，再做 sim-to-real。
- ODE→SDE 转换会掉点。试过 Flow-CPS，精度误差变小了，但 RL 收益有限。
- hybrid 采样的实现比较简单（每个环境步只随机取一步做 SDE）。
- 语义泛化有限，ManiSkill Semantic 这一项绝对值仍低（App. I）；MetaWorld 跨任务 OOD 没有提升（App. D.3 与 D.4 Summary）。

**笔者读出的**
1. **训练和评测的策略不是同一个**：PPO 优化的是带噪声的随机策略，部署的却是确定性 ODE。评测策略的单调改进没有理论保证，只能靠 Table 2 的 train/eval 差距去经验调参（附录 G 也把这一点列为主要调参信号）。
2. **两种 likelihood 各有近似**：
   - Flow-Noise 优化的是整条链的联合概率，不是 $\mathbf{A}^1$ 的边缘概率；而且均值没有漂移修正，注入噪声会改变动作分布。
   - Flow-SDE 只在连续时间下保持边缘分布，而实际用的是 $K=3\sim5$ 步（Table 11–13）， $K=1$ 时 train 成功率只有 9.4%。
3. **评测口径**
   - LIBERO 的基线是 few-shot SFT，提升幅度因此偏大；没有报告“全量 SFT + RL”这个上限。
   - 没有报告 seed 数和方差/置信区间。
   - 论文没有说明 LIBERO 上 RL 训练用的初始状态是否与评测的 500 个初始状态分开，作者也承认 LIBERO 没有 OOD 测试接口（App. D）。所以 LIBERO 上的提升只能算 ID 提升，即在训练分布上变得更熟练。
   - CALVIN OOD 的训练/评测协议写得含糊（“identical D→D training settings”）。
4. **真机证据很薄**：只有一个 pick-and-place 任务，成功率 40%，试验次数不详；仿真是手工标定的，其他场景要复刻成本不低。
5. **算力与交互量**
   - 只说在 8×H100 80GB 上跑，没有总的 wall-clock 或 GPU·h，只有 Fig.7c 的 update time。
   - LIBERO 每个 suite 训练 500 epochs × 64 个并行环境 × 8 个 rollout epochs（Table 11），交互量很大，在真机上不可行。
6. **结构性假设**
   - VLM 冻结，视觉 OOD 的提升因此偏小（作者在 App. D.1 也这么推测）。
   - π0 的 $V_{\text{expert}}$ 靠对噪声动作取平均来近似，是启发式做法。
   - chunk 被当作宏动作，丢掉了 chunk 内部的 credit assignment。
   - ManiSkill 的 “partial reset” 机制没有展开说明。
7. GRPO 的超参表（`tabs/hyperparams_ppo_grpo.tex`）在 source 和 PDF 里都是空的，PPO vs GRPO 的对比是否公平无法核查。

## 在脉络中的位置

- **前驱**
  - [PPO](../ppo/notes.md)：优化器。
  - [DPPO](../dppo/notes.md)：diffusion policy 的 two-layer MDP，Flow-SDE 的 MDP 直接沿用它。
  - ReinFlow：可学噪声 + 离散马尔可夫链，是 Flow-Noise 的来源。
  - Flow-GRPO / Mix-GRPO / TempFlow-GRPO：图像生成里的 ODE→SDE 与 hybrid 采样。
  - Song et al. 的 score SDE。
  - [SimpleVLA-RL](../simplevla-rl/notes.md)、RL4VLA、RLinf-VLA：面向自回归/OFT 类 VLA 的 RL，chunk-level 奖励来自 RLinf-VLA。
  - GRPO 的对照线：[DeepSeek-R1](../deepseek-r1/notes.md)。
- **后继 / 影响**：它是 RLinf 框架里 flow VLA 在线 RL 的参考实现。本项目中与它针对同一问题的后续路线：
  - [π\*0.6](../pi-star-06/notes.md)：从真实经验里学，走 advantage-conditioned 路线，不做在线 PPO。
  - [WMPO](../../05_intersections/wmpo/notes.md)、[World-Gymnast](../../05_intersections/world-gymnast/notes.md)：用 world model 代替物理仿真器做 VLA 的 RL，恰好针对本文承认的“依赖仿真”这一局限。
  - 注意：以上是同一问题上的不同路线，并不是说它们都基于本文。
- **与本项目其他主题的联系**
  - **World model**：本文的 RL 信号全部来自物理仿真器。换成学出来的 simulator，就对应 [Robotic World Model](../../05_intersections/robotic-world-model/notes.md)、[Ctrl-World](../../05_intersections/ctrl-world/notes.md)、[DreamerV3](../../01_world_model/dreamerv3/notes.md) 的 imagination 路线。
  - **Real2sim2real**：它的 Real2Sim2Real 实验用 3DGS 缩小视觉 gap，与 [SplatSim](../../03_real2sim2real/splatsim/notes.md)、[RialTo](../../03_real2sim2real/rialto/notes.md) 同一思路；SIMPLER 评测见 [SIMPLER](../../03_real2sim2real/simpler/notes.md)；系统性讨论见 [Reality Gap 综述](../../03_real2sim2real/survey-reality-gap/notes.md)。
  - **RL 的真机路线**：对照 [HIL-SERL](../hil-serl/notes.md)。
  - **TTT**：π_RL 是训练时微调，需要奖励和仿真器。[PAD](../../05_intersections/pad/notes.md) 在部署时用自监督信号适应，[TTRL](../../02_test_time_training/ttrl/notes.md) 在测试时做无标签 RL，两者可作对照。一个开放问题是：能否把 π_RL 式的 flow-policy 策略梯度搬到部署时，用少量真机交互做适应。

## 前置知识

1. **Flow matching / Rectified Flow**：CFM loss、线性插值路径、Euler 采样、action chunk。
2. **Score-based SDE**：probability flow ODE 和 SDE 的边缘分布等价；score 与 velocity 的换算（Flow-GRPO 的推导）。
3. **策略梯度基础**：policy gradient 定理、PPO clip、GAE、actor-critic、explained variance、clip fraction、KL 监控。
4. **扩散/flow 策略上的 RL**：DPPO 的 two-layer MDP、ReinFlow 的噪声注入、Flow-GRPO。
5. **π0 / π0.5 架构**：PaliGemma VLM + action expert（KV-cache 连接），以及两者输入 state 的方式不同。
6. **GRPO**（用于对照）；LIBERO / ManiSkill / CALVIN / MetaWorld 的评测协议。

## 复现可行性

- **代码**：官方 RLinf 仓库开源（已确认存在，Apache-2.0），文档中有 OpenPI（π0/π0.5）示例，模型发布在 HF 的 RLinf 组织页（具体 checkpoint 名以仓库为准）。据仓库 README（论文未提），RLinf 用 PyTorch 重写了 π0/π0.5，数值行为与 JAX 参考实现对齐。
- **算力**：论文只写了 8×NVIDIA H100 80GB（App. B.2），环境、rollout、actor 三者共置在同一 GPU 上。**没有报告总训练时长或 GPU·h**。
- **数据/仿真**：
  - LIBERO：HF `physical-intelligence/libero`，共 1,692 条。
  - ManiSkill：16,384 条 MPLib 合成轨迹。
  - CALVIN：ABC（InternData-Calvin_ABC）。
  - MetaWorld：`lerobot/metaworld_mt50`，共 2,500 条。
- **最小可复现实验**
  1. **不训练，先验证 SDE 实现**：用 π0 few-shot SFT checkpoint 在 LIBERO-Spatial 上扫 $a\in\lbrace0.2,0.5,0.8\rbrace$ 和 $K\in\lbrace1,2,4,8\rbrace$，比较随机 rollout 与确定性 eval 的成功率，看能否复现 Table 2 SFT 行的趋势： $K=1$ 时 train 约 9.4， $a$ 增大时 train 下降。
  2. **正式训练**：π0 + Flow-SDE（hybrid）， $a=0.5$， $K=4$， $H'=5$，64 个并行环境，actor/critic lr 为 5e-6 / 1e-4，训练 100 步。目标是 Table 2 中 eval 65.2 → 94.5。
  3. **加分项**：对比 hybrid 与完整 two-layer 的 update time（Fig.7c，约 2×）。
- **预计难点**：
  - 3.3B 模型、仿真和 PPO 放在一起的显存压力（论文靠 8×H100）。
  - few-shot SFT 用的 58 / 40 条轨迹具体怎么抽样没有公开，起点难以对齐。
  - 噪声水平、去噪步数、学习率三者耦合， $a$ 小时需要更小的 lr。
  - ManiSkill 上 critic 预热期 eval 会先掉后升（Fig.15，App. G）。
  - 结果没有方差，单 seed 结果很难判断是否复现成功。
  - 总体难度：中等。

## 自测问题

1. $K=4$ 步 Euler 生成的 flow 策略，为什么不能直接算 $\log\pi(\mathbf{A}^1|\mathbf{o})$？Flow-Noise 和 Flow-SDE 各自用什么来代替它？两者 PPO ratio 里的“策略”是否是同一个概率对象？
2. 从 Eq.7 出发，代入 score–velocity 关系推出 Eq.8 和单步方差 $\sigma_\tau^2\delta$，并说明为什么 $K=1$ 时 rollout 成功率会崩到 9.4%（Table 2）。先把 $\tau$ 的方向约定写清楚。
3. 写出 two-layer MDP 在内层和外层的 state / action / transition / reward。hybrid ODE-SDE 是怎样把 horizon 缩短的？为什么 one-layer 的 Flow-Noise 在 wall-clock 上并不比完整 two-layer 快（Fig.7c）？
4. 噪声水平 $a$ 为什么是一个 trade-off？ $a$ 太小时 clip fraction 为什么会升高（从 Gaussian log-prob 对均值的梯度 $\propto 1/\sigma^2$ 的角度解释）？
5. π0 的 critic 为什么要对 $\tau$ 取平均，而 π0.5 可以直接挂在 VLM 上？这对 advantage 的方差和 explained variance 有什么影响（Fig.6）？
