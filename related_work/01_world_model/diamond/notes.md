# Diffusion for World Modeling: Visual Details Matter in Atari（DIAMOND）

> arXiv [2405.12399](https://arxiv.org/abs/2405.12399) · NeurIPS 2024（Spotlight，已在 NeurIPS 2024 proceedings 和 OpenReview 核实） · 主题：01_world_model · 层级：milestone
> 代码：https://github.com/eloialonso/diamond（MIT；`main` = Atari agent，`csgo` 分支 = CS:GO world model，`ddpm` 分支 = Sec 5.1 的 DDPM 对照） · 笔记依据：`source/*.tex`（数字/公式）+ `paper.pdf`（图）

## 一句话

DIAMOND 用 EDM 形式的扩散模型直接在像素空间做 world model，推理时只用 3 步 Euler 去噪，RL agent 完全在这个"扩散梦境"里训练。论文的核心论点是：不对观测做离散化压缩，能保住对决策关键的视觉细节。在 Atari 100k 上，它的 mean HNS 达到 1.46，是完全在 world model 内训练的 agent 中最好的。

## 问题设定

- **环境**：POMDP，只能看到图像观测 $\mathbf{x}_t$，动作离散。Sec 2.1 把 world model 定义为 $p(s_{t+1}, r_t \mid s_t, a_t)$ 的生成模型；Sec 2.3 中扩散部分建模的是 $p(\mathbf{x}_{t+1} \mid \mathbf{x}_{\le t}, a_{\le t})$，reward $r_t$ 和终止 $d_t$ 另由单独的 $R_\psi$ 预测（Sec 3.2）（`source/2_framework.tex`、`source/3_method.tex`）。
- **Benchmark**：Atari 100k，共 26 个游戏，每个游戏只允许 100k 次真实交互（约 2 小时人类游戏时长）。设置为 64×64×3 观测、最多 18 个动作、frameskip 4、max noop 30、life loss 即终止、reward clip 到 $\{-1,0,1\}$（App E，`tables/table_hyperparameters.tex`）。
- **训练范式**：Dyna / Dreamer 式三步循环：真实环境采数据 → 用全部数据训 world model → 在 imagination 里训 policy。agent 只在采集数据时接触真实环境。
- **之前卡在哪**：DreamerV2/V3、IRIS、TWM、STORM 都把观测压成离散 latent 或 token，这能减轻 compounding error，但压缩有损，可能丢掉几个像素大小却影响决策的细节。增加 token 数又贵：IRIS 的采样速度受 token 数 $K$ 限制，$K=64$ 比 $K=16$ 还慢 2.8×（App M）。扩散模型生成图像质量高，但 DDPM 常要上百到上千步，放进 RL 的 imagination 循环里太贵。

## 核心方法

**1. Score-based diffusion 背景**（`source/2_framework.tex` Eq.1–4）。前向 SDE 和逆向 SDE 分别是

$$d\mathbf{x}=\mathbf{f}(\mathbf{x},\tau)d\tau+g(\tau)d\mathbf{w},\qquad d\mathbf{x}=[\mathbf{f}-g^2\nabla_\mathbf{x}\log p^\tau(\mathbf{x})]d\tau+g\,d\bar{\mathbf{w}}.$$

用 denoising score matching 训练 score。由于扰动核是已知高斯，目标可以化成 L2 重建损失 $\mathcal{L}=\mathbb{E}\|\mathbf{D}_\theta(\mathbf{x}^\tau,\tau)-\mathbf{x}^0\|^2$，其中 $\mathbf{D}_\theta=\mathbf{S}_\theta\sigma^2+\mathbf{x}^\tau$。这里用 $\tau$ 表示扩散时间，$t$ 表示环境时间（Figure 1 的纵轴和横轴）。

**2. 条件化成 world model**（Eq.5）：

$$\mathcal{L}(\theta)=\mathbb{E}\big[\|\mathbf{D}_\theta(\mathbf{x}^\tau_{t+1},\tau,\mathbf{x}^0_{\le t},a_{\le t})-\mathbf{x}^0_{t+1}\|^2\big]$$

训练时从 replay buffer 取一段轨迹，只对下一帧加噪；历史帧保持干净（teacher forcing）。

**3. EDM 参数化**（`source/3_method.tex` Eq.6–7，App C）。扰动核取 $\mathcal{N}(\mathbf{x}^0,\sigma^2(\tau)\mathbf{I})$，对应 $\mathbf{f}=0$、$g=\sqrt{2\dot\sigma\sigma}$；Algorithm 1 中取 $\tau:=\sigma$。

$$\mathbf{D}_\theta=c_\text{skip}^\tau\mathbf{x}^\tau_{t+1}+c_\text{out}^\tau\mathbf{F}_\theta(c_\text{in}^\tau\mathbf{x}^\tau_{t+1},y_t^\tau),\quad y_t^\tau=(c_\text{noise}^\tau,\mathbf{x}^0_{\le t},a_{\le t})$$

各系数为 $c_\text{in}=1/\sqrt{\sigma^2+\sigma_\text{data}^2}$，$c_\text{out}=\sigma\sigma_\text{data}/\sqrt{\sigma^2+\sigma_\text{data}^2}$，$c_\text{skip}=\sigma_\text{data}^2/(\sigma_\text{data}^2+\sigma^2)$，$c_\text{noise}=\tfrac14\log\sigma$，其中 $\sigma_\text{data}=0.5$。训练时噪声水平按 $\log\sigma\sim\mathcal{N}(-0.4,1.2^2)$ 采样。代入后，网络 $\mathbf{F}_\theta$ 的实际回归目标是 $\frac{1}{c_\text{out}}(\mathbf{x}^0_{t+1}-c_\text{skip}\mathbf{x}^\tau_{t+1})$（Eq.7）。

- 当 $\sigma\gg\sigma_\text{data}$ 时，$c_\text{skip}\to0$，目标约等于干净图像。
- 当 $\sigma\to0$ 时，$c_\text{skip}\to1$，目标约等于噪声。

**为什么 EDM 比 DDPM 稳**（Sec 5.1）：DDPM 做 ε-prediction，在噪声占主导时网络会退化成恒等映射（$\xi_\theta\to\mathbf{x}^\tau$），采样一开始的 score 估计就很差。这个误差在自回归 rollout 中逐帧累积，导致画面漂出分布。EDM 在高噪声区直接预测干净图像，所以步数很少时依然稳定。

**4. 采样**：用 Euler 解 ODE，$n=3$ 步，即每帧 3 NFE。下面的推导由我根据 Sec 3.1 的定义自行得出，论文未写出显式形式：由 $\mathbf{S}=(\mathbf{D}-\mathbf{x})/\sigma^2$ 可得 probability-flow ODE $d\mathbf{x}/d\sigma=(\mathbf{x}-\mathbf{D}_\theta)/\sigma$，Euler 步为

$$\mathbf{x}_{i+1}=\mathbf{x}_i+(\sigma_{i+1}-\sigma_i)\frac{\mathbf{x}_i-\mathbf{D}_\theta(\mathbf{x}_i,\sigma_i)}{\sigma_i}.$$

若一步从 $\sigma_\max$ 直接走到 0，输出就是 $\mathbf{D}_\theta$ 本身，也就是 L2 意义下的**条件后验均值**。这正是 Sec 5.2 的论点：后验多峰时单步会给出模糊的平均图（Figure 4，Boxing 里对手的移动不可预测）。agent 自己控制的白色选手因为动作已知，不存在歧义，所以单步也清晰。论文没有给出 3 步采样时具体的 $\sigma_i$ 序列。

**5. 网络**（App D，`tables/table_architecture.tex`）：
- **world model**：U-Net 2D，residual blocks 为 [2,2,2,2]、channels 为 [64,64,64,64]。过去 $L=4$ 帧**按通道拼接**（frame stacking）到噪声帧上；动作和 $c_\text{noise}$ 通过 adaptive group norm 注入。
- **Reward/termination 模型 $R_\psi$**：独立的 CNN+LSTM（LSTM 维度 512），用 CE 分别拟合 $\text{sign}(r)$ 和 $d$（Algorithm 1）。
- **Actor-critic**：共享 CNN-LSTM 主干。

**6. 在 imagination 里做 RL**（App F，`source/appendix/rl_objectives.tex`）。先用 $L=4$ 帧真实数据做 burn-in，然后 rollout $H=15$ 步。λ-return 和两个损失为

$$\Lambda_t=r_t+\gamma(1-d_t)[(1-\lambda)V_\phi(\mathbf{x}_{t+1})+\lambda\Lambda_{t+1}],\ \Lambda_H=V_\phi(\mathbf{x}_H)$$

$$\mathcal{L}_V=\mathbb{E}\sum_t(V_\phi(\mathbf{x}_t)-\text{sg}(\Lambda_t))^2,\quad \mathcal{L}_\pi=-\mathbb{E}\sum_t[\log\pi_\phi(a_t|\mathbf{x}_{\le t})\,\text{sg}(\Lambda_t-V_\phi)+\eta\mathcal{H}(\pi_\phi)]$$

policy 用 REINFORCE + value baseline 训练，没有用 PPO。超参为 $\gamma=0.985$、$\lambda=0.95$、$\eta=0.001$。

**7. 外层循环**（App E/G）：共 1000 个 epoch。每个 epoch 先用 ε=0.01 采集 100 个真实环境步，再对扩散模型、$R_\psi$、actor-critic 各做 400 次更新（batch 32，AdamW，lr 1e-4）。

## 关键实验与证据

- **主结果**（Table 1，`tables/table_results.tex`，5 seeds）：

  | 方法 | Mean HNS | IQM | 超人类游戏数 |
  |---|---|---|---|
  | DIAMOND | **1.459** | **0.641** | 11 |
  | STORM | 1.266 | 0.636 | 10 |
  | DreamerV3 | 1.097 | 0.497 | 9 |
  | IRIS | 1.046 | 0.501 | 10 |
  | TWM | 0.956 | 0.459 | 8 |
  | SimPLe | 0.332 | 0.130 | 1 |

  - IQM 与 STORM 基本持平（Figure 2 给出 stratified bootstrap CI）。
  - 大幅领先的游戏：Breakout 132.5（IRIS 83.7，STORM 15.9）、Asterix 3698.5（第二名 1128.3）、RoadRunner 20673.2（STORM 17564.0）。
  - 明显落后的游戏：BankHeist 19.7（Random 14.2，DreamerV3 649.0）、PrivateEye 114.3（STORM 7781.0）、BattleZone 4702.0、Frostbite 274.1。
- **与 model-free / search 方法比**（App J Table）：BBF 的 mean/IQM 为 2.247/1.139，EfficientZero 为 1.943/1.047，都高于 DIAMOND。作者认为这两者用了 reset 或 MCTS，属于正交技巧。
- **EDM vs DDPM**（Sec 5.1 Figure 3；App K 为 drift 定量图）：两者用同一网络，在同一份 Breakout 专家静态数据（100k 帧）上训练，rollout 1000 步，在 400 条 held-out 轨迹上比较。DDPM 在 $n\le10$ 时迅速漂移，并出现色彩崩坏；即使 DDPM 用 10 步，也不如 EDM 用 1 步稳定。
- **去噪步数消融**（App L Table，$n=3$ vs $n=1$，只测 DIAMOND 表现最好的 10 个游戏，且 $n=1$ 只跑了 1 个 seed）：这 10 个游戏上的 mean HNS 为 3.052 vs 1.962。Boxing 为 86.9 vs 41.9，RoadRunner 为 20673.2 vs 5084.0；但有 3 个游戏在 $n=1$ 下反而更高：Asterix（6687.0 vs 3698.5）、Krull（9105.1 vs 8610.1）、Pong（20.9 vs 20.4）。
- **"视觉细节重要"的证据**（Sec 5.3 Figure 5）：与 IRIS 在同一静态数据上训练并对比。IRIS 的 rollout 出现"敌人和奖励来回互变"、砖块和分数不一致等问题，DIAMOND 没有，Breakout 中打掉红砖后分数也正确地 +7。算力方面，两者分辨率都是 64×64，DIAMOND 每帧 3 NFE，IRIS 每帧 16 NFE。App H Table 的参数量与训练天数为：IRIS 30M / 4.1 天，DreamerV3 18M / <1 天，DIAMOND 13M / 2.9 天。
- **3D 环境、视觉质量**（App M Table，每条轨迹 16 帧，1024 条，以 $L=6$ 帧真实数据为条件；这里的 CS:GO 用的是 *Clean* 子集 190k 帧、分辨率 64×64，与 Sec 6 的 5M 帧 *Online* 数据不同，且 DIAMOND frame-stack 此处用 20 步去噪）：

  | 方法 | CS:GO FID / FVD / LPIPS | Driving FID / FVD / LPIPS |
  |---|---|---|
  | DIAMOND frame-stack | 9.6 / 34.8 / 0.107 | 16.7 / 80.3 / 0.058 |
  | IRIS ($K=64$) | 22.8 / 85.7 / 0.116 | 44.3 / 276.9 / 0.148 |
  | DreamerV3 | 106.8 / 509.1 / 0.173 | 167.5 / 733.7 / 0.160 |
  | DIAMOND cross-attention | 11.6 / 81.4 / 0.125 | 35.2 / 299.9 / 0.119 |

  cross-attention 版本全面不如 frame-stack。
- **CS:GO 神经游戏引擎**（Sec 6）：训练数据为 87h（5M 帧）。分辨率先降到 56×30 建模，再用一个扩散 upsampler 放大。模型共 381M 参数（其中 upsampler 51M；Sec 6 称 Atari 模型为 4M，与 App H 表中 DIAMOND 的 13M 口径不同，论文未解释），在一张 RTX 4090 上训练 12 天，在 RTX 3090 上以 10Hz 运行。这一节只有定性结果。

## 局限与论文没说的

**作者承认的**：
- 只测了离散动作（Sec 8）。
- frame stacking 只记得 $L=4$ 帧，记忆很短。cross-attention 在早期实验里效果更差（App M）。
- 奖励和终止没有并入扩散模型。
- CS:GO 中，模型在少见区域会漂移，靠近墙或丢失视野后会"忘掉"当前状态，还会错误地允许空中连跳（Sec 6）。
- 用离线数据训练会产生因果混淆：驾驶场景中输入 slow down，模型让前车一起减速（App M）。

**我读出来的**：
- **因果归因偏弱**："视觉细节导致分数提升"主要靠 3 个游戏的定性图（Figure 5），再加上与 Table 1 的相关性。论文缺少受控实验，例如固定 agent 只换 world model，或人为降低 DIAMOND 的保真度看分数怎么变。失败的游戏也没有分析：BankHeist 仅略高于 Random，PrivateEye 这类需要长程探索或记忆的游戏明显落后。我推测这和 $H=15$、$L=4$ 有关，但论文没有检验。
- **消融口径**：$n=1$ 消融只跑了单 seed，而且只测 DIAMOND 表现最好的 10 个游戏，有选择偏差，Asterix、Krull、Pong 的结果还是反向的。EDM vs DDPM 只在 Breakout 上测，DDPM 只用了 ε-prediction，没有测 DDPM + x0/v-prediction。因此"稳定性来自参数化"这一解释只由论文的机制分析支撑。EDM 的几个组件（preconditioning、log-normal σ 采样）各自贡献多少也没有拆开。
- **算力口径**：Atari 100k 只约束样本数，不约束 wall-clock。每个 epoch 100 个环境步，对应三个组件各 400 次更新。单个 run 需 2.9 天，而 DreamerV3 不到 1 天；DIAMOND 在 Table 1 上的全部 run（26 游戏 × 5 seeds）共耗费 1.03 GPU-years。
- **像素空间的代价**：每个 imagination step 耗时 20.4ms，其中去噪 3×4.2ms、奖励/终止模型 7.0ms、动作预测 0.7ms（App I）。分辨率一升，代价就明显上升；CS:GO 为降低算力先降到 56×30 建模，再接 upsampler。多相机、高分辨率的机器人观测不适合直接照搬这条路线，后续的 latent diffusion / transformer 路线（Cosmos、Dreamer 4）走的是另一条路。
- **缺少的分析**：论文没有 world model 的不确定性估计，对 policy 利用模型错误（model exploitation）的问题也只在 Broader Impact 中一句带过（world model 的缺陷可能导致次优或意外的 agent 行为），没有实验分析。$R_\psi$ 在真实帧上训练，却在生成帧上推理，这里的分布偏移没有被量化。

## 在脉络中的位置

- **前驱**：
  - world model 这条线：[World Models](../world-models/notes.md) 提出在 imagination 中训练 agent；SimPLe 引入 Atari 100k；[Dreamer](../dreamer/notes.md) 在 RSSM 的 latent 空间里做 RL，DreamerV2 起改用离散 latent 以减轻 compounding error，[DreamerV3](../dreamerv3/notes.md) 沿用这一路线（Sec 7）；IRIS 是同组前作，本文的 agent 设计沿用 IRIS，也是主要对照对象。
  - 生成模型这条线：score SDE（Song et al.）、DDPM、EDM（Karras et al. 2022，是本文扩散参数化的直接来源）。CS:GO 数据来自 Pearce & Zhu 2022。
- **同期工作**：[Genie](../genie/notes.md)（从图像 prompt 生成可玩环境）、GameNGen（DOOM，扩散，规模更大），两者都在 Sec 7 related work 中提到。
- **后继 / 影响**：
  - [Dreamer 4](../dreamer4/notes.md) 在引言中把 DIAMOND 与 DreamerV3、DayDreamer、TD-MPC2、MuZero 等并列，作为 world model agent 的代表，并继续推进"在可扩展 world model 里训 agent"。
  - [Cosmos WFM](../cosmos-wfm/notes.md) 把它归入用于 model-based RL 的视觉空间 world model。
  - [LaCT](../../02_test_time_training/lact/notes.md) 引用它作为自回归视频扩散中 teacher forcing 的例子。
  - [WMPO](../../05_intersections/wmpo/notes.md) 借它来论证"像素级保真度对 RL 至关重要"（WMPO 原句限定为 RL with Gaussian policies，而 DIAMOND 本身是离散动作）。
  - [Robotic World Model](../../05_intersections/robotic-world-model/notes.md) 把它列为"在 world model 中完全替代真实交互"的代表工作。
  - [RL Overview 综述](../../04_rl/survey-rl-overview/notes.md) 把它归为用条件图像生成学 $p(\mathbf{o}'|\mathbf{o},a)$ 的代表。
- **与本项目其他主题的联系**：
  - **RL**：imagination 中的 actor-critic（REINFORCE + λ-return）。论文的理由是 imagination 里能生成大量 on-policy 轨迹，所以用简单的 REINFORCE 即可；可以和 [PPO](../../04_rl/ppo/notes.md) 这类同为 on-policy、但带 clipped surrogate 的方法对照。机器人 VLA 在生成式 world model 里做 RL 的工作有 [WMPO](../../05_intersections/wmpo/notes.md)、[World-Gymnast](../../05_intersections/world-gymnast/notes.md)；[Ctrl-World](../../05_intersections/ctrl-world/notes.md) 则用 world model 做 policy 评估，并在 imagination 中合成成功轨迹做 supervised fine-tuning，不是 RL。
  - **real2sim2real**：从日志数据学出的扩散 world model 可以看作 learned simulator，但此时 sim-to-real gap 就变成了 world model 与真实环境之间的差距。App M 中的离线因果混淆，就是用离线真实数据训 simulator（如 [UniSim](../../05_intersections/unisim/notes.md)）时同样会碰到的问题。[SimDist](../../05_intersections/simdist/notes.md) 走的是另一条路：先在物理仿真器中预训练 world model，真实部署时只用真实数据更新 latent dynamics。
  - **TTT**：作者点出 frame stacking 的记忆太短，CS:GO 中也出现了"忘掉状态"。[TTT layers](../../02_test_time_training/ttt-layers/notes.md)、[Titans](../../02_test_time_training/titans/notes.md) 这类 fast-weight 长上下文记忆能否作为 world model 的历史编码器，是一个自然的交叉问题，但论文本身没有涉及。

## 前置知识

1. **Score-based diffusion 的 SDE 视角**（Song et al. 2020）：前向/逆向 SDE、denoising score matching、probability-flow ODE。
2. **EDM**（Karras et al. 2022）：σ 参数化、preconditioning（$c_\text{skip}, c_\text{out}, c_\text{in}, c_\text{noise}$）、log-normal 噪声采样、Euler/Heun 采样器；还要知道它与 DDPM ε-prediction 的区别。
3. **Model-based RL 的 Dyna/Dreamer 范式**：POMDP、imagination rollout、burn-in。
4. **Actor-critic 基础**：REINFORCE + baseline、λ-return、熵正则。
5. **Atari 100k 评测口径**：HNS、IQM、stratified bootstrap CI（Agarwal et al. 2021）。
6. **IRIS / DreamerV3 的离散 token 或 latent world model**：用于理解本文的对比对象。
7. **网络与评测指标**：U-Net、adaptive group norm 条件注入；FID/FVD/LPIPS（用于读 App M）。

## 复现可行性

- **代码**：官方开源，MIT 许可，Python 3.10，依赖见 `requirements.txt`，Atari ROM 随依赖自动下载。
  - 训练：`python src/main.py env.train.id=BreakoutNoFrameskip-v4 common.devices=0`
  - 试玩：`python src/play.py --pretrained` 会从 HF Hub 下载预训练的 world model 和 agent。
  - 采样步数、随机性等参数在 `config/trainer.yaml` 的 `world_model_env.diffusion_sampler` 中设置（据 repo README）。
- **算力**（全部为论文数字）：
  - Atari：单个 run 约 12GB 显存，在单张 RTX 4090 上约 2.9 天；26 游戏 × 5 seeds 共 1.03 GPU-years（Sec 4.1）。
  - 一次完整更新 543ms，其中扩散模型 88ms、$R_\psi$ 115ms、actor-critic 340ms（App I Table）。
  - App M：120k updates、batch 64，最多用 4×A6000，每个 run 1–2 天。
  - CS:GO：在 RTX 4090 上训练 12 天。
- **最小可复现实验建议**：
  1. **零训练**：用 `play.py --pretrained` 看 Breakout/Boxing 的 rollout，直观验证 Figure 4/5 的现象。
  2. **机制实验（首选）**：在 Boxing 上把 `diffusion_sampler` 的步数设为 1 和 3，对比 rollout 的清晰度和 agent 分数，复现 App L 中 86.9 vs 41.9 的趋势，最好补跑多个 seed；再用 `ddpm` 分支在 Breakout 上复现 Figure 3 / App K 的漂移曲线。按 App I 估算，只训扩散模型约 35s/epoch × 1000 epoch ≈ 9.7h（由表中数字推得）。论文没有给出 Sec 5.1/5.3 静态数据实验的训练步数，也没说明那份专家数据集是否发布，可能需要自己用训练好的 agent 去采集。
  3. **完整单游戏**：Breakout 跑 1 个 seed（约 2.9 天），对照 Table 1 中的 132.5；注意 Table 1 是 5 seeds 的均值，单 seed 方差可能很大。
- **难点**：wall-clock 太长，完整复现 26 个游戏不现实；gym/ale-py 的版本和评测协议需要对齐（noop、life loss、reward clip）；CS:GO 模型训练成本高（12 天），且只有定性评估，难以判断复现是否成功。

## 自测问题

1. 写出 EDM 参数化下 $\mathbf{F}_\theta$ 的回归目标，分析 $\sigma\to\infty$ 和 $\sigma\to0$ 两种极限下它退化成什么。为什么 DDPM 的 ε-prediction 在少步数自回归 rollout 中会累积误差，EDM 却不会？
2. 从 $\mathbf{D}_\theta=\mathbf{S}_\theta\sigma^2+\mathbf{x}$ 推导 probability-flow ODE 的 Euler 步，证明单步采样等于输出条件后验均值。据此解释 Boxing 中黑色选手会模糊、白色选手却清晰的原因。
3. DIAMOND 的一次 imagination step 依次调用了哪些网络？$L$、$H$、$n$ 分别控制什么，又分别影响哪部分算力（结合 App I 的时间分解）？
4. 历史帧和动作分别以什么方式进入 U-Net？frame stacking 相比 cross-attention 的优势和代价是什么？它和 CS:GO 中"靠近墙就忘掉状态"的现象有什么关系？
5. 论文主张"更好的视觉细节带来更高的 RL 分数"，目前证据属于哪一类（定性/相关/受控）？设计一个能在 DIAMOND 框架内检验这一因果关系的实验。
