# Training Agents Inside of Scalable World Models

> arXiv [2509.24527](https://arxiv.org/abs/2509.24527) · arXiv preprint（Google DeepMind），2025 · 主题：01_world_model · 层级：frontier
> 代码：none。论文只给了项目页 danijar.com/dreamer4，2026-09-27 核查时该页没有代码或权重链接；社区复现未核实。 · 笔记依据：`source/*.tex`（数字/公式）+ `paper.pdf`（图）

## 一句话

Dreamer 4 把 Dreamer 系列里的 RSSM 换成了可扩展的 block-causal Transformer world model：causal tokenizer 负责压缩画面，dynamics 用 **shortcut forcing**（diffusion forcing + shortcut model + x-prediction）训练，每帧只需 4 次前向，单张 H100 就能实时交互。策略、reward 和 value 以 agent token 的形式插进这个模型，再完全在 imagination 里用 PMPO 做 RL。这样只用 2.5K 小时离线 VPT 数据，就第一次在 Minecraft 里拿到了钻石，全程没有环境交互。

## 问题设定

- **任务**：Minecraft 的 offline diamond challenge。输入是 360×640 原始像素，输出是底层键鼠动作（与 VPT 协议一致，合成也要走游戏 UI）。每个 episode 60 分钟，从随机世界、空背包开始（`main.tex` §4.1）。据论文说，人类平均要 20 分钟、约 24,000 个动作才能拿到钻石。
- **数据假设**：只能用 VPT contractor 数据（subsets 6–10，共 2541 小时，20 FPS，带键鼠动作和事件标注，按 90/10 切分，见 `appendix.tex`），**不许与环境交互**。reward 是从数据集事件里标出来的 20 个任务的稀疏二值 reward（Table 4），可观测性是部分可观测的第一人称画面。
- **之前卡在哪**：
  1. Dreamer 3 这类 world model agent 又快又准，但只适用于窄环境，结构拟合不了复杂分布。Dreamer 3 在 Minecraft 里用的是 64×64 图像、背包状态和抽象合成动作（Table 3）。
  2. Genie 3、Oasis 这类可控视频模型可扩展，但物体交互和游戏机制预测不准，常常要多卡才能实时，没法当 RL 训练场。
  3. VPT 拿到钻石靠的是 270K 小时 web 视频 BC 加 194K 小时在线 RL（Table 3）；纯离线的 VPT (BC) 最多做到木镐这一级。

## 核心方法

三阶段流程（`algos/agent.tex` Algorithm 1）：
1. 预训练 tokenizer（Eq.5）和 dynamics（Eq.7）。
2. 插入 agent token，微调出 policy 和 reward head（Eq.7 + Eq.9）。
3. 冻结 transformer，在 imagination 里训练 policy/value head（Eq.10–11）。

所有损失项都按各自的 running RMS 归一化（§3 开头），所以一个 transformer 可以同时承担多模态、多个输出头。

**0. 背景公式**（`main.tex` §2）

- Flow matching（Eq.1）。$\tau=0$ 是纯噪声，$\tau=1$ 是干净数据：
$$x_\tau=(1-\tau)x_0+\tau x_1,\quad x_0\sim\mathcal N(0,I),\quad \mathcal L=\|f_\theta(x_\tau,\tau)-(x_1-x_0)\|^2$$
- Shortcut model（Eq.3–4）额外以步长 $d$ 为条件。在最细步长 $d_{\min}=1/K_{\max}$ 上用 flow loss；更大的步长用"两个半步"蒸馏出的 bootstrap 目标 $v_{\rm target}=\mathrm{sg}(b'+b'')/2$，其中 $b'=f(x_\tau,\tau,d/2)$，$x'=x_\tau+b'd/2$，$b''=f(x',\tau+d/2,d/2)$。$d$ 从 $1/\mathrm U\{1,2,4,\dots,K_{\max}\}$ 采样，$\tau$ 在该步长能到达的网格上均匀采样。Eq.4 字面写的是 $\{0,1/d,\dots,1-1/d\}$，按上下文应理解为间隔为 $d$ 的网格。
- Diffusion forcing：序列里每个时间步各取一个独立的 $\tau_t$，所以每一帧既是去噪任务，也是后续帧的（带噪）上下文。

**1. Causal tokenizer**（§3.1）

- encoder 和 decoder 都是时间上因果的 block-causal transformer。每帧由 patch token 和可学习的 latent token 组成，latent 经线性降维加 `tanh` 形成 bottleneck。
- Minecraft 设置：帧补零到 384×640，16×16 patch 得到 960 个 token。bottleneck 为 $512\times16$，送进 dynamics 时 reshape 成 $N_z=256$ 个 32 维 token（`appendix.tex`）。
- 损失（Eq.5）：$\mathcal L=\mathcal L_{\rm MSE}+0.2\,\mathcal L_{\rm LPIPS}$。
- 训练时加 MAE 式 patch dropout，丢弃率 $p\sim U(0,0.9)$ 逐图采样，所以也会训练到推理时用的 $p=0$。作者说这能提升生成视频的空间一致性。

**2. Interactive dynamics 与 shortcut forcing**（§3.2）

- 输入是按时间交错的序列：动作 token、一个 $(\tau,d)$ token（两者离散，各做 embedding 后拼接）、$S_z$ 个空间 token，以及 $S_r$ 个 register token。
- 每个动作分量单独编码成 $S_a$ 个 token，再加上一个可学习 embedding。**没有动作标签的视频只用这个可学习 embedding**，这就是它能利用无标注视频的方式。
- 前向（Eq.6）：$\tilde z=(1-\tau)z_0+\tau z_1$，$\hat z_1=f_\theta(\tilde z,\tau,d,a)$，其中 $\tau,d\in[0,1]^T$ 逐帧不同。
- 关键改动是 **x-prediction**：网络直接预测干净 latent $\hat z_1$，不预测速度。作者认为 v-prediction 输出高频成分，逐帧自回归时会累积误差（§3.2）。bootstrap 项要在 v 空间里算，换算关系是 $\hat v=(\hat z_1-\tilde z)/(1-\tau)$，推导很直接：
$$z_1-\tilde z=(1-\tau)(z_1-z_0)\;\Rightarrow\;\|\hat z_1-z_1\|^2=(1-\tau)^2\|\hat v-v\|^2$$
  因此 bootstrap 损失要乘 $(1-\tau)^2$，拉回和 x 空间 flow loss 相同的量级（脚注）。
- Shortcut forcing 损失（Eq.7）：
$$b'=\frac{f_\theta(\tilde z,\tau,\tfrac d2,a)-\tilde z}{1-\tau},\quad z'=\tilde z+b'\tfrac d2,\quad b''=\frac{f_\theta(z',\tau+\tfrac d2,\tfrac d2,a)-z'}{1-(\tau+\tfrac d2)}$$
$$\mathcal L=\begin{cases}\|\hat z_1-z_1\|_2^2 & d=d_{\min}\\[2pt] (1-\tau)^2\left\|\dfrac{\hat z_1-\tilde z}{1-\tau}-\mathrm{sg}(b'+b'')/2\right\|_2^2 & \text{else}\end{cases}$$
  原文 loss 里写成了 $\mathrm{sg}(b_1+b_2)$，与上一行的 $b',b''$ 记号不一致，是笔误。
- **Ramp 权重**（Eq.8）：$w(\tau)=0.9\tau+0.1$。在低 $\tau$（接近纯噪声）处，x-pred 的最优解退化成"预测数据均值"，学习信号少，所以降低这部分的权重。
- **推理**：时间上自回归。每帧从噪声出发，按 $d=1/4$ 走 $K=4$ 步。过去帧加轻噪到 $\tau_{\rm ctx}=0.1$，让模型对自己生成时的小瑕疵更稳健。（注意：原文写的是 "slightly corrupt ... to signal level $\tau_{\rm ctx}=0.1$"，但按原文自己 $\tau=1$ 为干净数据的约定，字面上 signal level 0.1 是重噪；两处表述不一致，原文未澄清。）
- **训练技巧**（§4 开头、§3.4）：
  - batch 里 30% 的视频当单张图像训练，学会无上下文生成首帧。
  - 交替使用短 batch 和长 batch（Minecraft 为 $T_1=64$、$T_2=256$，context $C=192$），最后只用长 batch 微调。batch 长度要大于 context，否则模型会过拟合"context 开头永远是首帧"，影响长度泛化。

**3. Efficient transformer**（§3.4）

- 基础结构：2D（时间 × 空间）transformer，时间上 block-causal，加 pre-RMSNorm、RoPE、SwiGLU、QKNorm 和 attention logit soft-capping。
- 提速手段：
  - space-only 和 time-only 的分解注意力（axial）；
  - 每 4 层才放 1 层 temporal attention；
  - dynamics 的所有注意力层都用 GQA，以缩小 KV cache。

**4. Agent finetuning：BC + reward model**（§3.3，Eq.9）

- 在 transformer 中插入 **agent token**，输入为 task embedding（one-hot）。注意力是单向的：agent token 能看所有模态，其他模态看不到 agent token。这样 world model 的未来预测只能被动作影响，不会被当前任务影响，作者称之为避免 causal confusion。
- 在 agent token 的输出 $h_t$ 上用 multi-token prediction（$L=8$）训练 policy 和 reward head：
$$\mathcal L=-\sum_{n=0}^{L}\ln p_\theta(a_{t+n}\mid h_t)-\sum_{n=0}^{L}\ln p_\theta(r_{t+n}\mid h_t)$$
- 输出参数化：reward head 沿用 DreamerV3 的 symexp twohot。键盘动作是 23 个 Bernoulli；鼠标是 121 类 categorical（μ-law，每个坐标 11 bin，$11\times11$）。
- 同时继续施加 Eq.7 的视频预测损失，以保住已有能力。
- 数据混合：50% 均匀采样序列，50% 完成了某个任务的"相关"序列。BC 损失只加在相关部分，dynamics 损失只加在均匀部分，目的是避免 world model 学成"乐观生成"。

**5. Imagination training**（§3.3，Eq.10–11）

- 新建 value head，并把 policy head 复制一份冻结，作为 behavioral prior。**只更新 policy 和 value head，transformer 冻结。**脚注说全量微调收益小、代价高，而且需要同时保留 dynamics、prior 和 reward 损失。
- rollout 从数据集 context 出发，每个 context 只起 1 条（DreamerV3 是从每个状态都起）。transformer 自回归生成：latent 由 flow head 采样，动作由 policy head 采样，再由 reward head 和 value head 标注 $r_t, v_t$。
- Value 用 TD(λ) 和 symexp twohot 训练（Eq.10），$\gamma=0.997$：
$$\mathcal L=-\sum_t\ln p_\theta(R^\lambda_t\mid s_t),\qquad R^\lambda_t=r_t+\gamma c_t\big((1-\lambda)v_t+\lambda R^\lambda_{t+1}\big),\quad R^\lambda_T=v_T$$
- Policy 用 **PMPO**（Eq.11），只看 advantage $A=R^\lambda-v$ 的**符号**。全部想象状态按符号分成 $\mathcal D^+$（$A\ge0$）和 $\mathcal D^-$（$A<0$）：
$$\mathcal L=\frac{1-\alpha}{|\mathcal D^-|}\sum_{i\in\mathcal D^-}\ln\pi_\theta(a_i|s_i)-\frac{\alpha}{|\mathcal D^+|}\sum_{i\in\mathcal D^+}\ln\pi_\theta(a_i|s_i)+\frac{\beta}{N}\sum_{i=1}^N\mathrm{KL}[\pi_\theta(\cdot|s_i)\,\|\,\pi_{\rm prior}]$$
  - 前两项对正集做 maximum likelihood、对负集压低似然，两组各自取平均，所以正负样本数不平衡时不会偏向一边。
  - 因为忽略了 advantage 的量级，**不需要 return/advantage normalization**，20 个任务即使 return 尺度可能不同（reward 本身都是稀疏二值），也能得到同等关注。这正好取代了 DreamerV3 的 return normalization 加熵正则（`appendix.tex` "Previous Dreamer Generations"；DreamerV3 中具体是基于 percentile 的 return 归一化）。
  - 取 $\alpha=0.5$、$\beta=0.3$。prior KL 用反向（$\pi_\theta$ 在前），比原版 PMPO 更强地把策略约束在合理行为附近。三项都以 nats 计，作者说它们之间的比例很稳健。
- λ 的取值、imagination horizon、RL 更新步数，论文都没有给出。

**训练 / 推理各自怎么走**

- 训练：Phase 1 → 2 → 3，都在离线数据上完成。
- 部署：在真实 Minecraft 里，把观测编码后送进 transformer，由 agent token 上的 policy head 直接出动作，不做 planning。评测时用人工写的任务 prompt 序列（Table 6，例如 `mine_log`×10 → … → `mine_diamond`×∞）按计数依次切换 task embedding。

## 关键实验与证据

规模：tokenizer 400M、dynamics 1.6B，共 2B 参数。在 256–1024 块 TPU-v5p 上训练，每设备 batch 1，用 FSDP（§4 开头）。

**1. 离线钻石挑战**（Figure 3/4，Table 7 成功率，Table 8 用时；每个 agent 1000 局）

| 物品 | VPT (finetuned) | BC | VLA (Gemma 3) | WM+BC | Dreamer 4 |
|---|---|---|---|---|---|
| Stone pickaxe | 0.0 | 53.8 | 76.7 | 89.4 | **90.1** |
| Iron ore | 0.1 | 26.5 | 46.3 | 62.9 | **66.7** |
| Iron ingot | 0.1 | 4.3 | 22.5 | 27.8 | **39.5** |
| Iron pickaxe | 0.0 | 0.6 | 11.2 | 16.9 | **29.0** |
| Diamond | 0.0 | 0.0 | 0.0 | 0.0 | **0.7** |

- VPT (finetuned) 最远到 stick（52.6%）。
- Dreamer 4 的 iron pickaxe 成功率是 VLA 的约 2.6 倍（29.0 / 11.2；Figure 3 图注说"nearly tripling"）。
- 用时（Table 8，分钟）：iron pickaxe 为 Dreamer 4 13.3、WM+BC 17.0、VLA 31.1；diamond 为 20.7。
- **组件拆分**：
  - world model 预训练带来的表征收益最大。BC → WM+BC 时，stone pickaxe 从 53.8 升到 89.4，iron pickaxe 从 0.6 升到 16.9。无任务条件的版本也一样：BC (notask) → WM+BC (notask)，stone pickaxe 从 8.8 升到 38.4。
  - imagination RL 的增益集中在难的里程碑：WM+BC → Dreamer 4，iron ingot 从 27.8 升到 39.5，iron pickaxe 从 16.9 升到 29.0，diamond 从 0 升到 0.7；早期物品已经饱和。RL 还同时缩短了到达各里程碑的时间。
  - 在作为 BC 表征时，WM 的表征优于 Gemma 3 VLA（Figure 4）。

**2. World model 质量：人类实时交互**（Table 1，Figure 5、12–14）

- 评测方式：人类玩家在模型里完成 16 个任务（挖坑、砌墙、放船骑船、进传送门等）。

| 模型 | 参数 | context | FPS（单 H100） | 完成任务 |
|---|---|---|---|---|
| Dreamer 4 | 2B | 9.6 s | 21 | **14/16** |
| Oasis (large) | 未公开 | 1.6 s | ~5（估计） | 5/16 |
| Oasis (small) | 500M | 1.6 s | 20 | 0/16 |
| Lucid-v1 | 1.1B | 1.0 s | 44 | 0/16 |
| MineWorld | 1.2B | 0.8 s | 2 | 不支持实时，未评测 |

- Oasis 的典型失败是"autocompletion"：放了几块方块后，就幻觉出大片建筑。
- 另在 SOAR 机器人数据（180 小时，7D 末端相对动作）和 Epic Kitchens 上做了定性展示（Figure 6、9），没有定量指标。

**3. 动作条件的数据需求**（Figure 7，§4.3）

- 评测方式：给 320 帧 context，生成 16 步，算 PSNR/SSIM。结果做了归一化，没有动作 = 0%，全部 2541 小时都有动作 = 100%。
- 只有 10 小时带动作时：PSNR 53%、SSIM 75%。
- 有 100 小时带动作时：PSNR 85%、SSIM 100%。
- **OOD 外推**：动作只来自 Overworld、Nether/End 只有无标注视频时，在 Nether/End 上达到 PSNR 76%、SSIM 80%。

**4. 模型设计消融**（Table 2 级联，Figure 8）

- 评测方式：每个模型训练 48 小时，无 context 生成 1024 条 384 帧视频，动作由固定的 BC 策略给出，按 16 帧一段算 FVD。
- 级联结果：
  - 起点是朴素 diffusion forcing transformer（$N_z=64$，$K=64$），FVD 306、0.8 FPS。
  - 把 $K$ 降到 4 后 FVD 875、9.1 FPS。
  - 加 shortcut 后 329。
  - 加 X-prediction 后 326（单独几乎无用）。
  - 加 **X-loss 后 151**。
  - 加 **ramp weight 后 102**。
  - 加交替 batch 长度后 80，训练步从 9.8 s 降到 1.5 s。
  - 每 4 层 1 层时间注意力后 70、18.9 FPS。
  - 加 GQA 后 71、23.2 FPS。
  - 时间分解的长 context 后 91、30.1 FPS。
  - 加 register 后 91。
  - $N_z=128$ 时 66；$N_z=256$ 时 **57、21.4 FPS**。
- 同样架构、改用 v 空间预测和损失时，FVD 为 124（正文）。
- Figure 8：4 步 shortcut forcing 接近 64 步 diffusion forcing 的质量，快 16×。

## 局限与论文没说的

**作者承认的**

- world model 远不是游戏的完整克隆：记忆只有 9.6 s（192 帧 / 20 FPS），背包物品有时模糊，或随时间变化（§4.2、§6）。
- 钻石成功率只有 0.7%。
- 作者列出的未来方向：互联网视频预训练、长期记忆、语言、少量纠错性在线数据、自动发现子目标。

**我读出来的**

- **评测口径**：
  - 钻石挑战依赖人工写的 25 步 prompt 阶梯（Table 6）来切换任务，相当于人给了任务分解和课程，和 VPT/Dreamer 3 的无条件设置并不完全等价。
  - 0.7% × 1000 局约等于 7 局（我的换算）。论文没有报训练 seed 数或置信区间。
  - world model 的"14/16"来自人类玩家的主观完成判定，尝试次数、玩家人数都没有报。
- **消融缺口**：
  - Table 2 是级联消融，结论依赖添加顺序。
  - RL 部分没有消融 PMPO（例如与 DreamerV3 式 REINFORCE + retnorm 比较）、$\beta$、imagination horizon，也没有 reward model 的准确率。冻结与否只有脚注里的定性说法（全量微调"small additional benefits"），没有给数字。
  - 离线 MBRL 最大的风险是策略利用 model error 或 reward model 的漏洞。论文只靠 BC prior 的 KL 加冻结 transformer 来约束，没有做不确定性惩罚，也没有度量"想象 return 与真实 return 的相关性"。
- **收益归因**：从 Table 7 看，WM 预训练对 BC 的提升大于 imagination RL 的提升。"RL in imagination"更像锦上添花，只在长链条的后段显著。
- **算力门槛**：256–1024 块 TPU-v5p 训练 2B 模型，总训练时长和 RL 阶段的算力都没有报告，也没有开源代码或权重。
- **真实机器人**：只有 SOAR 上的视频生成定性图，没有在真实机器人上训练或部署策略。所谓"适用于机器人"是推测。动作外推实验只在同一游戏的不同 dimension 之间，离"web 视频 → 新 embodiment"还很远。
- 动作泛化实验给的是 320 帧 context，而训练 context 是 192 帧。推理时怎样截断，论文没有说明。

## 在脉络中的位置

- **前驱**：
  - [World Models](../world-models/notes.md)：在梦里训练 controller。
  - [Dreamer](../dreamer/notes.md) 和 [DreamerV3](../dreamerv3/notes.md)：latent imagination 中的 actor-critic、λ-return、symexp twohot 都直接继承自这里。
  - [Genie](../genie/notes.md)：用时空 transformer 从无标注视频学可交互环境。Dreamer 4 不推断 latent action，而是在无标注视频上给动作位置放一个可学习的空 embedding。
  - [DIAMOND](../diamond/notes.md)：用扩散模型做 world model 并在其中训练 agent（Atari）。Dreamer 4 的 related work 只提到它的 CS:GO 模拟器只覆盖单个关卡、只做到短期预测。
  - 论文外的方法组件：diffusion forcing、shortcut models、PMPO、VPT 数据与协议、MAE。
- **后继 / 影响**：
  - [DreamZero](../dreamzero/notes.md) 在 related work 中把 Dreamer 系列（含 Dreamer 4）归入 latent-space world models，说它们"建模 $p(s_{t+1}|s_t,a_t)$、测试时需要 goal-conditioned planning 或搜索"，与自己的 world action model 对照。这个刻画对 Dreamer 4 不完全准确：Dreamer 4 部署时由 policy head 直接出动作，不做 planning。
  - [Ctrl-World](../../05_intersections/ctrl-world/notes.md) 把它列为基于扩散的动作条件 world model 代表。
  - 同一范式、但源码中没有引用它的机器人工作：[WMPO](../../05_intersections/wmpo/notes.md) 和 [World-Gymnast](../../05_intersections/world-gymnast/notes.md) 在视频 world model 里对 VLA 做 RL；[Robotic World Model](../../05_intersections/robotic-world-model/notes.md) 和 [SimDist](../../05_intersections/simdist/notes.md) 把 learned simulator 用于机器人策略优化。
- **与本项目其他主题的联系**：
  - **World model 内部的对照**：它和 [Cosmos](../cosmos-wfm/notes.md) 同属"大规模生成式视频 world model"，但它的目标是单卡实时加 agent 训练。[V-JEPA 2](../vjepa2/notes.md) 和 [TD-MPC2](../tdmpc2/notes.md) 走不重建像素、靠 planning 的路线。
  - **RL**：
    - PMPO 只用 advantage 符号加对 BC prior 的反向 KL，可以和 [PPO](../../04_rl/ppo/notes.md) 的 clipping、[DeepSeek-R1](../../04_rl/deepseek-r1/notes.md) 中 GRPO 的组内归一化加 reference KL 对照。
    - "BC 预训练 → RL 微调"的结构与 [π_RL](../../04_rl/pi-rl/notes.md) 和 [SimpleVLA-RL](../../04_rl/simplevla-rl/notes.md) 相同，只是 RL 环境换成了 world model。
    - 它本质上是 offline model-based RL，可以和 [IQL](../../04_rl/iql/notes.md) 这类 model-free 的离线 RL 比"怎样对抗分布偏移"：这里用的是 BC prior 加冻结模型。
  - **TTT**：
    - Dreamer 4 的记忆完全靠 192 帧 KV context，作者把"长期记忆"列为未来方向。[TTT layers](../../02_test_time_training/ttt-layers/notes.md)、[Titans](../../02_test_time_training/titans/notes.md)、[One-Minute Video TTT](../../02_test_time_training/ttt-one-minute-video/notes.md) 用 fast weights 做长上下文，是自然的候选。这是我的推断，论文没有讨论。
    - 部署时 world model 和 policy 都冻结，没有任何 test-time adaptation。能否借鉴 [PAD](../../05_intersections/pad/notes.md)，用视频预测损失在部署时适应，是开放问题。
  - **Real2sim2real**：
    - "少量动作标注就能 grounding，而且能外推到只见过无标注视频的场景"这一结果，是"从真实视频学 simulator"（参见 [UniSim](../../05_intersections/unisim/notes.md)）的关键证据。
    - 这类 learned simulator 没有手工仿真器的 [reality gap](../../03_real2sim2real/survey-reality-gap/notes.md)，但代价是 model error。论文没有像 [SIMPLER](../../03_real2sim2real/simpler/notes.md) 那样验证"模型内评测与真实表现的相关性"。
    - "少量纠错性在线数据"这一未来方向对应 [HIL-SERL](../../04_rl/hil-serl/notes.md) 和 [DayDreamer](../../05_intersections/daydreamer/notes.md) 的在线真机学习。

## 前置知识

1. Flow matching / rectified flow，以及 x-prediction 和 v-prediction 之间的换算。
2. Diffusion forcing（逐 token 独立噪声水平）和 shortcut models（以步长为条件、bootstrap 自蒸馏），可对比 consistency models。
3. DreamerV3 的 imagination actor-critic、λ-return、symexp twohot（见 [DreamerV3](../dreamerv3/notes.md)）。
4. KL 正则的 policy optimization：MPO/PMPO，前向 KL 与反向 KL 的区别，behavior cloning prior。
5. Transformer 推理效率：KV cache、block-causal mask、axial（space/time 分解）attention、GQA、RoPE、QKNorm。
6. MAE、LPIPS，视频评测指标 FVD/PSNR/SSIM。
7. Offline RL 的基本困难：分布偏移、model exploitation。

## 复现可行性

- **代码**：官方未开源（项目页没有代码链接，已核查）。论文给出的超参数不完整：缺 $S_r$、$S_a$、层数和宽度、λ、imagination horizon、学习率、RL 步数。
- **算力**：主模型 2B，在 256–1024 块 TPU-v5p 上训练（§4）。总时长没有报告。消融中每个模型训练 48 小时，硬件规模没有单独说明。推理为单 H100 21 FPS（Table 1）。完整复现 Minecraft 结果，个人或小组条件下不可行。
- **数据/环境**：VPT contractor 数据（subsets 6–10），Minecraft 评测沿用 VPT 协议。机器人部分用 SOAR（180 小时），另有 Epic Kitchens 100。
- **最小可复现实验**（验证机制，不追数字）：
  1. **Shortcut forcing 消融**：选一个小型、带动作标注的视频数据集（例如自己从简单游戏或 SOAR 子集采集并降到 64×64），先训一个小 tokenizer 并冻结，再训小型 block-causal dynamics。对比四种设置：v-pred/v-loss、x-pred/v-loss、x-pred/x-loss、x-loss + ramp。每种都测 $K\in\{1,2,4,64\}$ 下长 rollout（远超训练长度）的 FVD 或 PSNR，检验 Table 2 第 3–6 行和 Figure 8 的趋势：X-loss 和 ramp 主要改善长 rollout，4 步 shortcut 接近 64 步。
  2. **动作标签稀疏度**：只给 0/1/10/100% 的视频动作标签（其余用可学习的空 embedding），画归一化 PSNR 曲线，对照 Figure 7 左。
  3. **Agent 阶段**：在一个有真实环境可以评测的小任务上，比较 BC、WM+BC、WM+BC+PMPO imagination RL 的真实成功率，外加一组不带 prior KL（$\beta=0$）的对照，观察 model exploitation。
- **预计难点**：
  - x 空间 bootstrap 损失的数值细节，例如 $\tau\to1$ 时除以 $(1-\tau)$。
  - 让 $(\tau,d)$ 网格采样与推理步数一致。
  - agent token 的单向 attention mask。
  - 多损失的 RMS 归一化。
  - 冻结 transformer 后长 rollout 的显存与速度开销。

## 自测问题

1. 推导 x-prediction 的 flow loss 与 v 空间 loss 之间差一个 $(1-\tau)^2$ 因子。说明 Eq.7 的 bootstrap 目标怎样由两个半步构造，为什么 $d=d_{\min}$ 时要退回 flow loss，以及 ramp 权重 $w(\tau)=0.9\tau+0.1$ 为什么要压低低 $\tau$ 的权重。
2. 从 Table 2 看，单独改 X-prediction（329→326）几乎没用，改 X-loss（→151）却大幅改善。这说明问题出在参数化上还是损失空间上？为什么 v-prediction 更容易在逐帧自回归中累积误差？
3. agent token 为什么只能"看别人"、不能"被看"？如果视频 token 能 attend 到 task embedding，在 imagination 中会出现什么 causal confusion？为什么 dynamics 损失只用均匀采样的序列？
4. PMPO 只用 advantage 的符号，并对正负集分别取平均。它相对 DreamerV3 的"return 除以 $\max(1,S)$ 加熵正则"解决了什么问题（提示：20 个 return 尺度可能不同的任务）？反向 KL $\mathrm{KL}[\pi_\theta\|\pi_{\rm prior}]$ 和前向 KL 对策略的约束有什么不同？
5. 无动作标签的视频在 dynamics 里是怎样被使用的？Overworld → Nether/End 的实验控制了什么变量，得到 76%/80% 的归一化分数说明了什么，又没能说明什么（例如能否推广到跨 embodiment 的 web 视频）？
