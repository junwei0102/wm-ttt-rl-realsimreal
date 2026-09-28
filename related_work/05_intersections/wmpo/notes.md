# WMPO: World Model-based Policy Optimization for Vision-Language-Action Models

> arXiv [2511.09515](https://arxiv.org/abs/2511.09515) · ICLR 2026（Poster，已在 OpenReview 和 proceedings.iclr.cc 的 camera-ready 核实；arXiv 预印本 2025，源码目录里有 `iclr2026_conference.bib`，但 PDF 用的是 ByteDance Seed 模板，正文和项目页都未写录用信息） · 主题：05_intersections · 层级：milestone
> 代码：https://github.com/WM-PO/WMPO（项目页 https://wm-po.github.io/ 给出此链接；2026-09-27 用 WebFetch 确认存在：基于 OpenSora + verl + OpenVLA-OFT，提供 4 个 Mimicgen 任务在 P=128/1280 下的训练脚本和 checkpoint 下载脚本） · 笔记依据：`source/*.tex`（数字/公式）+ `paper.pdf`（图）

## 一句话

先在 OXE 上预训练一个动作条件视频扩散 world model，再用策略在真实环境中自己跑出的少量轨迹（含失败）微调它（policy behavior alignment）。之后让 VLA（OpenVLA-OFT）在这个 world model 里闭环"想象"完整轨迹，由学到的 0/1 成功判别器给奖励，做 on-policy GRPO，全程不需要在真实环境中做 RL 交互。关键想法有两点：world model 输出的是像素图像，所以 VLA 的预训练视觉表征可以直接用上；world model 见过失败，想象轨迹里才会有 GRPO 需要的负样本。

## 问题设定

- **目标**：对已经做过 IL/SFT 的 VLA 做 RL 后训练，让它学会从失败中恢复。难点有两个：真实机器人上 on-policy RL 的样本太贵；GRPO 需要从同一初始状态重复 rollout，这在物理世界里很难做到（§1）。
- **MDP**（`source/sections/03method.tex` §3.1）：
  - 状态 $\mathcal S=\mathcal I\times\mathcal G$（图像序列 × 语言指令）。论文**假设仅凭图像就能定义机器人状态**，并明确把 POMDP 留作未来工作。
  - 动作是长度为 $K$ 的 action chunk $a_t\in\mathbb R^{K\times D}$，每一维离散化为 256 个 bin。
  - 转移 $P$ 由 world model $p_\phi$ 学出。
  - 奖励是学出来的 trajectory 级二值信号 $R_\psi(\tau)\in\{0,1\}$。
  - 目标是 $\max_\theta \mathbb E_{\tau\sim\pi_\theta,p_\phi}[R_\psi(\tau)]$（Eq.1）。
- **数据假设**：有 $P$ 条真实轨迹（rollout budget），由 base policy 采集，用来微调 WM 和训练 reward model。想象的起点 $s_0$（前 $c$ 帧加指令）从真实环境中采样。
- **以往方法卡在哪**（§1、§2）：
  1. human-in-the-loop 方法（HIL-SERL 等）需要人持续干预，难以扩展。
  2. 在仿真里做 PPO/GRPO（SimpleVLA-RL、VLA-RL）需要为每个场景搭建精确的仿真器。
  3. 真机 RL 为了省样本多采用 off-policy 方法，会引入 value estimation 偏差（§3.4 引 Park et al. 2025）。
  4. Dreamer 系的 latent world model 工作在抽象 latent 空间，和在真实图像上预训练的 VLA 对不上。大规模视频 WM（Cosmos、Genie 3）用在机器人上时存在分布不匹配，难以忠实复现 policy rollout 和细粒度的机器人–物体交互（§2）。"模拟不出失败"是论文针对只用 OXE / 专家示范训练的 WM 说的（§3.2 Policy Behavior Alignment），不是针对 Cosmos、Genie 3。

## 核心方法

方法由三个学出来的组件构成：world model $p_\phi$、reward model $R_\psi$、policy $\pi_\theta$（整体流程见 Fig.2）。

**1. 想象 rollout（Eq.2）。** 记号约定：$i$ 为帧下标，$t$ 为状态（chunk）下标。
- 策略读取最近 $m$ 帧和指令，输出一个 chunk：$a_{i:i+K}\sim\pi_\theta(I_{i-m:i},g)$。
- WM 以最近 $c$ 帧和这个 chunk 为条件，生成接下来的 $K$ 帧：

$$I_{i:i+K}\sim p_\phi(I_{i-c:i},\,a_{i:i+K})\qquad(\texttt{03method.tex}\ \text{Eq.2})$$

- 按 clip 自回归，直到最大长度 $N$，得到 $\tau=\{I_{0:N},a_{0:N}\}$，由 $R_\psi$ 打 0/1 标签。实验中 $K=8$、$c=4$（§4.1）。
- 生成的是**完整 trial**，而不是短时域预测。原因是短时域下奖励难以定义，还容易出现 reward hacking（§1）。

**2. World model 结构（§3.2）。**
- **骨干**：OpenSora 的视频 diffusion transformer，把其中的 3D VAE 换成 SDXL 的 2D VAE，以免时间维压缩破坏细粒度运动。扩散本身仍在 VAE latent 中进行，生成后解码回像素再交给 VLA。所以文中的"pixel-space"指 policy 的输入接口是像素。RSSM 路线则是 policy 直接在 latent 上学，这是两者的区别。
- **Noisy-frame conditioning**：训练时给条件帧加扩散步 50/1000 的噪声，让模型能容忍自己生成的不完美历史帧，缓解自回归误差累积。作者称这样可以稳定生成"数百帧"。
- **Frame-level action control**（借鉴 IRASim）：每帧的动作 $a_i$ 经过 MLP 生成 AdaLN 调制系数，和扩散时间步嵌入一起在帧级注入，使动作与帧对齐：

$$\mathbf x^i=\mathbf x^i+(1+\alpha_1^i)\cdot\mathrm{Block}\big(\gamma_1^i\cdot\mathrm{LayerNorm}(\mathbf x^i)+\beta_1^i\big)$$

  （§3.2 正文公式，Block 为 MHA 或 FFN。）
- **Policy behavior alignment**：WM 先在 OXE 上预训练。OXE 和专家示范几乎全是成功轨迹，只用它们训练的 WM 模拟不出失败。所以还要用 base policy 在下游任务上自己跑出的真实 rollout（含失败）微调 WM，使其对齐 policy 的 (state, action) 分布。这是整个方法成立的前提：GRPO 的组内必须同时有成功和失败。

**3. Reward model（§3.3）。**
- 结构是 VideoMAE encoder 加线性头，用 BCE 训练。
- 取长度为 $L$ 的 clip $c_i=I_{i-L:i}$。正样本是成功轨迹的末尾 clip $c_N$。负样本有两类：成功轨迹中 $\{c_i:L\le i\le N-L\}$ 范围内的 clip，以及失败轨迹中的任意 clip。每个 batch 内正负样本数量平衡。
- 推理时用 stride $s$ 的滑窗，只要**任一 clip** 的概率超过阈值 $\tau_{thr}$（在验证集上选定）就判为成功。实验取 $L=8$、stride 1。

**4. On-policy GRPO（§3.4，Alg.1）。**
- 对每个真实的 $s_0$，用 $\pi_{\theta_{old}}$ 在 WM 里想象 $G$ 条轨迹。采用 **dynamic sampling**（DAPO）：一组轨迹若全部成功或全部失败，advantage 全为 0，就丢弃这组并补采，直到 batch 填满。
- 一个 chunk 的对数概率是 $K\times D$ 个离散 token 对数概率之和（Eq.3）：

$$\log\pi_{\theta_{old}}(a_t\mid s_t)=\sum_{i=1}^{K}\sum_{j=1}^{D}\log\pi_{\theta_{old}}(a_t^{i,j}\mid s_t)$$

- 目标函数（Eq.4–5）去掉了 KL 项和 reference model，并使用非对称 clip（即 DAPO 的 Clip-Higher）：

$$\mathcal J(\theta)=\mathbb E_{s_0\sim\mathcal D,\{\tau_i\}\sim\pi_{\theta_{old}}}\Big[\frac1G\sum_{i=1}^G\frac1T\sum_{t=0}^{T}\min\big(r_{i,t}(\theta)\hat A_i,\ \mathrm{clip}(r_{i,t}(\theta),1-\epsilon_{low},1+\epsilon_{high})\hat A_i\big)\Big]$$

$$r_{i,t}(\theta)=\frac{\pi_\theta(a_{i,t}\mid s_{i,t})}{\pi_{\theta_{old}}(a_{i,t}\mid s_{i,t})},\qquad \hat A_i=\frac{R_i-\mathrm{mean}(\{R_i\}_{i=1}^G)}{\mathrm{std}(\{R_i\}_{i=1}^G)}$$

- advantage 是 trajectory 级的：同一条轨迹的所有 chunk 共享同一个 $\hat A_i$，既没有 critic 也没有 value function。这之所以可行，是因为 WM 可以从同一个 $s_0$ 反复 reset。
- 超参见 Table 4（`06appendix.tex` 的 `tab:grpo`）：lr $5\times10^{-6}$，training batch 64，$G=8$，mini-batch 128，$\epsilon_{low}=0.20$，$\epsilon_{high}=0.28$，采样温度 1.6；其余设置沿用 SimpleVLA-RL。
- 表里 batch 64 小于 mini-batch 128，看上去矛盾。Alg.1 又把 batch size $B$ 标为"trajectories"，并要求 $M\mid B$，与表中数值对不上，论文内部不自洽。附录 B 写到 GRPO 基线"一次更新至少需要 64×8=512 条真实轨迹"，据此推断 64 是组数、128 是轨迹数。这只是推断，论文没有写明。

**训练与推理流程。**
- 训练：SFT base policy → 采 $P$ 条真实轨迹 → 用这些轨迹微调（已在 OXE 上预训练的）WM 并训练 RM → 在 WM 内做 GRPO。
- 推理：只部署 VLA。
- Lifelong 版本（§4.5）：用更新后的策略再采 $P$ 条真实轨迹 → 再微调 WM → 再做 GRPO，如此交替。

## 关键实验与证据

**设置**（§4.1、附录 B）：
- base policy 是 OpenVLA-OFT，每个任务用 300 条专家轨迹做 IL，去掉本体状态和腕部相机，改用并行预测离散 token 的输出头。
- 在 Mimicgen 的四个 D0 任务上，每个任务用 128 个初始状态评测成功率。
- 基线：online GRPO 直接用真实 rollout，batch 在 8 和 64 中取结果更好的一个；offline DPO 用成功/失败轨迹对训练。所有方法的真实 rollout budget $P$ 相同。

**主结果（Table 1）**：
- base 平均成功率 33.6%。
- $P=128$ 时，GRPO / DPO / WMPO 平均为 33.2 / 37.3 / **47.1**；$P=1280$ 时为 37.1 / 42.4 / **57.6**。WMPO 对最强基线的领先幅度从 +9.8 扩大到 +15.2。
- Coffee 从 43.8 升到 61.7（P=128），再到 75.0（P=1280）；Square 从 24.2 升到 45.3（P=1280）。
- GRPO 表现差的原因见附录 B：batch 取 64 时一次更新就要至少 512 条真实轨迹，即使 P=1280 也只够更新一两次。

**Reward model**：所有任务上 F1 都大于 0.95（§4.2），但正文没有说明在什么数据上测得。

**涌现行为**：
- Fig.3（Square 任务）：碰撞发生后，base policy 一直下压直到超时；WMPO 学会先抬起、重新对准、再插入，即 self-correction。
- Fig.5：四个任务上，WMPO 成功轨迹的平均长度都比 base（记为 100%）短，说明卡住的情况更少。

**泛化（Table 2）**：三类扰动分别是 Square 的杆位置随机、StackThree 的背景变灰、ThreePieceAssembly 的底座纹理改变。
- 平均成功率：base 23.7、GRPO 24.7、DPO 19.5、WMPO **29.6**。
- DPO 在背景和纹理扰动下低于 base（34.4 vs 46.1，7.8 vs 10.9）。作者解释为 DPO 依赖了伪视觉线索。

**Lifelong（Fig.6，只测了 StackThree）**：
- 每轮采 128 条真实轨迹。WMPO 的成功率单调上升，DPO 在第二轮反而下降。
- WMPO 也优于用 300/428/556 条**专家**示范做 IL 的参照策略。图上读数约为：budget=256 时 WMPO 约 63%，IL 参照约 57%。

**真机（§4.6）**：
- 平台是 Cobot Mobile ALOHA，任务为把方块插到杆上，间隙只有 5 mm。先用 200 条专家示范做 SFT，再采 128 条 policy rollout。
- 30 次试验的成功率：base 53%、DPO 60%、WMPO 70%。
- Fig.7/8：从同一初始状态出发，WM 能预测出真实的成功轨迹和失败轨迹。Fig.9 是 WM 的失败案例，它漏掉了最后时刻方块卡住的情形。

**消融**：**论文没有任何定量消融**。noisy-frame conditioning、frame-level action control、policy behavior alignment、OXE 预训练、像素 vs latent，这些设计都只有文字论证。

## 局限与论文没说的

- **作者承认的局限**：
  - 只做了离散动作表示，flow-matching 策略（π0）和 FlowGRPO 留作未来工作（附录 D Limitation）。
  - 假设完全可观测，没有处理 POMDP（§3.1）。
  - WM 偶尔会漏掉细微的失败瞬间（附录 C），论文只说这种情况"relatively rare"，没有给出比例。
- **缺消融**：三项创新各自贡献多少无法判断。核心论点 policy behavior alignment 尤其缺少"只用专家数据微调 WM"这一对照。
- **比较口径不对等**：
  - WMPO 额外用了 OXE 大规模预训练（Table 3 写的是 12,000,000 预训练步 + 3,000,000 微调步，没说单位是否为 sample 数），基线没有这份 world model 先验。"相同 rollout budget"只控制了下游真实数据量。
  - base policy 去掉了本体状态和腕部相机，基线成功率偏低（33.6%），提升空间被放大。
- **Model exploitation / reward hacking 没有直接测量**：
  - F1 > 0.95 大概率是在真实轨迹上测的。policy 更新后，想象轨迹的分布会偏离 WM 和 RM 的训练分布。
  - 论文没有报告"WM 内成功率 vs 真实成功率"的差距。
  - 单轮训练中 WM 是冻结的（Alg.1 只更新 $\theta$），只对齐了 base policy 的行为，policy 离 base 越远，WM 越不可信。lifelong 的交替更新只能部分缓解这个问题。
- **统计不足**：
  - 仿真实验每个任务 128 个初始状态，论文没报告 seed 数，也没有 std。
  - 真机只有一个任务、30 次试验，也没有真机 GRPO 基线。
  - lifelong 只在 StackThree 上跑了两轮。
- **初始状态来源不清**：论文说 $s_0$ "sample from real environment"，没说是否只取那 $P$ 条轨迹的起点。
- **算力与成本不透明**：像素级 diffusion 要自回归生成"数百帧"，若按上文对 batch 的推断，每次更新约 512 条想象轨迹（论文未写明）。去噪步数、单条 rollout 耗时、总训练时长都没有报告，而这正是它和 Dreamer 式 latent imagination 相比最大的代价。

## 在脉络中的位置

- **前驱**：
  - Model-based RL / imagination：[Dreamer](../../01_world_model/dreamer/notes.md)、[DreamerV3](../../01_world_model/dreamerv3/notes.md)（以及 PlaNet、DreamerV2）。WMPO 把它们当作反面参照：latent 太抽象，和 VLA 对不上。[World Models](../../01_world_model/world-models/notes.md) 属于同一脉络，但 WMPO 没有引用它。
  - [DIAMOND](../../01_world_model/diamond/notes.md)：被引来支持"像素保真度对 RL 很关键"。
  - [Cosmos](../../01_world_model/cosmos-wfm/notes.md) 和 Genie 3（本项目收录的是 [Genie](../../01_world_model/genie/notes.md)）：大规模视频 WM 的代表，WMPO 指出它们用在机器人上有分布不匹配。
  - IRASim：同一第一作者的工作，frame-level action control 来自这里。
  - RL 这条线：[PPO](../../04_rl/ppo/notes.md) 的 clipped surrogate → GRPO（由 DeepSeekMath，即 Shao et al. 2024 提出；WMPO 引 [DeepSeek-R1](../../04_rl/deepseek-r1/notes.md) 说明其稳健性和可扩展性）→ DAPO（dynamic sampling、Clip-Higher、去 KL）。
  - [SimpleVLA-RL](../../04_rl/simplevla-rl/notes.md)：在仿真器里对 OpenVLA-OFT 做 GRPO，WMPO 沿用了它的设置。可以把 WMPO 看成"把仿真器换成 WM"的版本。
  - [HIL-SERL](../../04_rl/hil-serl/notes.md)：作者对比的 human-in-the-loop 路线。
  - [UniSim](../unisim/notes.md)：最早提出"视频 WM 当 simulator 训 RL"，但 WMPO 没有引用它。
- **后继 / 影响**：
  - [World-Gymnast](../world-gymnast/notes.md) 在 related work 里自称"与 WMPO 最相似"（`world-gymnast/source/related.tex`）。它改用 VLM 当 reward，在真机 AutoEval 上评测，并扩展到 test-time training 和 Dyna 式的 WM 更新。
  - 同期同类工作：[Ctrl-World](../ctrl-world/notes.md)（在 WM 中想象并筛选成功轨迹，再做 SFT）、[Robotic World Model](../robotic-world-model/notes.md)。
- **与本项目其他主题的联系**：
  - *World model*：WMPO 代表"生成式、像素接口的 WM 当 environment"这条路线。可对照 [Dreamer 4](../../01_world_model/dreamer4/notes.md)（在可扩展 WM 里训练 agent）和 [DreamZero](../../01_world_model/dreamzero/notes.md)（WM 本身就是 policy）。
  - *RL*：on-policy、off-policy、offline 的取舍，可对照 [SAC](../../04_rl/sac/notes.md)、[IQL](../../04_rl/iql/notes.md) 和论文里的 DPO 基线。往 flow 策略扩展可以看 [π_RL](../../04_rl/pi-rl/notes.md) 和 [DPPO](../../04_rl/dppo/notes.md)。
  - *Real2sim2real*：
    - 用少量真实 rollout 微调 WM，本质上是**数据驱动的 real-to-sim**，和 [RialTo](../../03_real2sim2real/rialto/notes.md)、[SplatSim](../../03_real2sim2real/splatsim/notes.md) 的"显式重建 + 物理引擎"形成对照。
    - Table 2 的背景和纹理扰动，测的是 [Domain Randomization](../../03_real2sim2real/domain-randomization/notes.md) 关心的那类视觉 gap。
    - lifelong 交替更新，类似 [SimOpt](../../03_real2sim2real/simopt/notes.md) 用真实经验修正 simulator 的闭环。
  - *TTT*：WMPO 本身不是 TTT。但 policy behavior alignment（让 WM 适应当前 policy 的分布）和 lifelong 交替更新，都属于部署后持续用新数据更新模型。World-Gymnast 把这一思路推进到单个测试场景上的 test-time training。另可对照 [PAD](../pad/notes.md)，以及同样依赖自产奖励信号做 RL 的 [TTRL](../../02_test_time_training/ttrl/notes.md)。

## 前置知识

1. GRPO / PPO 的 clipped surrogate 和 group-normalized advantage，DAPO 的 dynamic sampling 与 Clip-Higher（见 [PPO](../../04_rl/ppo/notes.md)、[DeepSeek-R1](../../04_rl/deepseek-r1/notes.md)）。
2. VLA 的 action chunk 与动作离散化 token，OpenVLA-OFT 的并行解码（[SimpleVLA-RL](../../04_rl/simplevla-rl/notes.md) 笔记讲得最清楚）。
3. Model-based RL 中的 imagination rollout 和 model exploitation（见 [Dreamer](../../01_world_model/dreamer/notes.md)）。
4. Latent video diffusion：DiT、AdaLN 条件注入、VAE latent、ε-prediction，以及自回归视频生成中的误差累积。
5. on-policy 与 off-policy 的偏差来源，DPO 的偏好目标。
6. 视频分类器（VideoMAE），以及类别不平衡时的 BCE 训练。

## 复现可行性

- **代码**：官方仓库 https://github.com/WM-PO/WMPO 已开源（WebFetch 确认）。
  - 依赖：OpenSora（WM）、verl（RL 框架，多节点靠 Ray）、OpenVLA-OFT、VideoMAE；环境为 PyTorch 2.5.1 / Python 3.11。
  - 提供 4 个 Mimicgen 任务在 budget 128/1280 下的训练和评测脚本，数据与 checkpoint 通过 `download_hf.py` 获取。
- **算力**（附录 A）：OpenVLA-OFT 的 SFT 用 8×H100；WM 训练和策略优化用 32×H100；**训练时长没有报告**。最重的是 WM 的 OXE 预训练（Table 3：batch 128，lr 1e-4，EMA 0.9999），个人基本无法重做。
- **数据 / 环境**：OXE（预训练）；Mimicgen（robosuite）四个 D0 任务，每个任务 300 条专家轨迹。真机部分需要 Cobot Mobile ALOHA，无法复现。
- **最小可复现实验**：
  1. 下载官方的 SFT base policy、微调后的 WM 和 RM checkpoint，只跑 Coffee_D0、$P=128$ 的 WMPO 脚本，在 128 个初始状态上评测，对照 Table 1（43.8 → 61.7）。
  2. 补一个论文没做的诊断：对若干真实初始状态，把真实执行的动作序列放进 WM 回放，再用 RM 判定，与仿真真值对比得到混淆矩阵。训练过程中定期记录"WM 内成功率 vs Mimicgen 真实成功率"，直接量化 model exploitation。
  3. 如果还有余力，做 policy behavior alignment 的消融：WM 只用那 300 条专家轨迹微调，其余设置不变。
- **预计难点**：
  - 像素级自回归 rollout 很慢。若按上文推断，每次更新要 64 组 × G=8 条轨迹，吞吐必须靠多卡。
  - RM 阈值 $\tau_{thr}$ 的具体选法论文没给。
  - Table 3 中步数的单位不明，GPU 少时需要重新调 batch 和 lr。

## 自测问题

1. 只用专家示范（或 OXE）微调的 world model，为什么撑不起 GRPO？请从组内 advantage 的计算和 dynamic sampling 两方面解释。
2. WMPO 的扩散其实发生在 SDXL VAE 的 latent 里，那它说的"pixel-space"指什么？和 DreamerV3 中 actor 直接读 RSSM latent 相比，关键差别在哪？
3. Noisy-frame conditioning 在训练时给条件帧加 50/1000 步的噪声，它要弥合的是训练和推理之间的哪一种分布差异？
4. Eq.4 里 advantage 是 trajectory 级的，ratio 是 chunk 级的（K×D 个 token 的 log-prob 求和）。这样做在信用分配上有什么问题？为什么在 WM 里仍然能工作？
5. $P=128$ 时 online GRPO 基线为什么几乎没有提升？同样 128 条真实轨迹，WMPO 拿它们做了哪些不同的事？
