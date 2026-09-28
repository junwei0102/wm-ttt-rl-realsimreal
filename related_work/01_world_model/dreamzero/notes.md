# World Action Models are Zero-shot Policies

> arXiv [2602.15922](https://arxiv.org/abs/2602.15922) · arXiv preprint（NVIDIA 技术报告），2026；另被 ICLR 2026 Workshop on World Models 接收（非存档，据 OpenReview） · 主题：01_world_model · 层级：frontier
> 代码：https://github.com/dreamzero0/dreamzero（论文声明会开源模型权重、推理代码，以及 RoboArena / PolaRiS / Genie Sim 3.0 的评测代码；2026-09-27 核对过，仓库存在。仓库页面另外列出了训练代码、DROID 预处理数据和 DROID/AgiBot 两个 checkpoint，许可证 Apache-2.0） · 笔记依据：`source/*.tex`（数字/公式）+ `paper.pdf`（图）。`source/method_original_notation.tex` 是没有被 `\input` 的旧稿，本笔记不采用。

## 一句话

DreamZero 把 14B 的 image-to-video 扩散模型（Wan2.1-I2V-14B）改造成 World Action Model（WAM）：同一个 DiT、同一个 flow-matching 目标，联合去噪"未来视频 + 动作块"。这样一来，学动作就从"观测→动作的直接模仿"变成了"对模型自己预测的视觉未来做 inverse dynamics"。在此基础上，它用 chunk-wise 自回归，闭环执行时把真实观测写回 KV cache，再配一套推理加速（38×，7Hz）。结果是在未见环境和未见任务上，比 SOTA VLA 高出 2 倍以上。

## 问题设定

- **要解决的问题**：VLA 从 VLM 初始化（如 π0.5、GR00T N1.6），继承了语义先验，知道该"做什么"；但它缺少时空和动力学先验，不知道该"怎么做"。因此它对新环境、新动作（unseen verbs，比如解鞋带）泛化很差，而且通常要给每个任务收集大量重复示范（`main.tex` §1）。
- **输入/输出**（`main.tex` §3.1）：输入是语言指令 $\mathbf c$、本体状态 $\mathbf q_l$、历史观测 $\mathbf o_{0:l}$（多视角直接拼成一帧，不改主干）；输出是未来视频 $\mathbf o_{l:l+H}$ 和动作块 $\mathbf a_{l:l+H}$，动作表示为 relative joint positions（§4.1）。
- **假设**：纯模仿学习（flow matching 形式的行为克隆），没有 reward，也没有 RL。每个 embodiment 单独预训练：AgiBot G1 用约 500h 自采遥操作数据，Franka 用 DROID。部署是闭环的，每执行完一个动作块都能拿到真实观测。
- **之前卡在哪**：(1) VLA 的预训练数据是静态图文，要从噪声很大的异质 state-action 对里"隐式地学动力学"很难。(2) 已有的 WAM（UWM、Video Policy、mimic-video、Genie Envisioner、Cosmos Policy）多数仍在重复示范数据上训练；论文把"采用自回归架构"列为区别于先前 WAM 的一点（§2，原文未逐一说明各 WAM 的架构）。双向模型要对视频做子采样，才能让语言描述和视频区间对上，代价是破坏原生帧率和视频-动作对齐（Appendix B，Figure 13）。(3) 14B 视频扩散模型直接推理，一个 chunk 要约 5.7 s（`real_time.tex`），根本做不了闭环控制。

## 核心方法

**1. 分解视角**（`main.tex` Eq.1）：

$$\pi_0(\mathbf o_{l:l+H},\mathbf a_{l:l+H}\mid \mathbf o_{0:l},\mathbf c,\mathbf q_l)=\underbrace{\pi_0(\mathbf o_{l:l+H}\mid \mathbf o_{0:l},\mathbf c,\mathbf q_l)}_ {\text{video prediction}}\ \underbrace{\pi_0(\mathbf a_{l:l+H}\mid \mathbf o_{0:l+H},\mathbf q_l)}_ {\text{IDM}}$$

论文此处引用的 LingBot-VA（Causal World Modeling for Robot Control）和 mimic-video 用两个模型（视频预测模型 + IDM）分别拟合这两项；DreamZero 不这么做，而是用一个模型端到端学联合分布。作者的论点是：视频预测的能力大部分已经从 web 视频预训练里继承过来，机器人数据只需要教两件事——机器人本体长什么样、怎么动（视频侧），以及隐式的 IDM。IDM 要覆盖各种各样的 state-action 对应关系，所以数据多样性比重复次数更重要（这是 §5.2 Q1 给出的解释）。

**2. 架构**（Figure 4，§3.1、§4.1）：主干是 Wan2.1 的 DiT，在它上面只加了 state encoder、action encoder 和 action decoder，尽量少引入新参数。VAE、文本编码器、图像编码器都冻结，训练时更新全部 DiT block 和新增模块。作者在脚注里提到试过 LoRA，效果不如全量更新。

**3. Chunk-wise 自回归 + teacher forcing**（§3.1，Appendix C）
- 每个 chunk 含 $K=2$ 个 latent frames，默认最多 $M=4$ 个 chunk。AgiBot 的设置是视频 5 FPS、动作 30Hz、 $H=48$；DROID 是 5 FPS、15Hz、 $H=24$。两者一个 chunk 都覆盖 1.6 s。最大上下文是 8 个 latent frames，约等于 33 帧原始图像、6.6 s。
- 加噪方式（Eq.2）。注意这里的约定是 $t=1$ 为干净、 $t=0$ 为纯噪声：

$$\mathbf z^k_{t_k}=t_k\mathbf z^k_1+(1-t_k)\mathbf z^k_0,\qquad \mathbf a^k_{t_k}=t_k\mathbf a^k_1+(1-t_k)\mathbf a^k_0,\qquad \mathbf z^k_0,\mathbf a^k_0\sim\mathcal N(0,I)$$

  同一个 chunk 里，视频和动作**共享** $t_k$。作者说这样训练初期收敛更快；UWM 等工作则给两个模态各自独立的 timestep。不同 chunk 的 $t_k$ 独立采样，Algorithm 1 中是 $t_k\sim\mathcal U(0,1)$。
- 训练目标（Eq.3）：

$$\mathcal L(\theta)=\mathbb E\Big[\tfrac1K\sum_{k}w(t_k)\big\Vert\mathbf u_\theta([\mathbf z^k_{t_k},\mathbf a^k_{t_k}];\mathcal C_k,\mathbf c,\mathbf q_k,t_k)-\mathbf v^k\big\Vert^2\Big],\qquad \mathbf v^k=[\mathbf z^k_1,\mathbf a^k_1]-[\mathbf z^k_0,\mathbf a^k_0]$$

  正文说做 trajectory-level update，即一条轨迹只做一次前向（Algorithm 1 的伪代码却写成逐 chunk 循环、每个 chunk 更新一次 $\theta$，与正文不一致）。token 序列排成 `[C0, Z1, Y1, C1, Z2, Y2, C2, Z3, Y3]`，其中 C 是干净帧、Z 是带噪视频、Y 是带噪动作。block-causal mask 保证第 $k$ 个带噪 chunk 只能看到 $C_0,\dots,C_{k-1}$ 和它自己（Figure 14a）。这和 LLM 的 teacher forcing 一样，所有 chunk 的损失在一次前向里并行算完。
- **记号上的两处不一致**：Eq.3 的归一化写成 $1/K$，但附录里 $K$ 表示每个 chunk 的 latent 帧数，chunk 数其实是 $M$。另外， $\mathcal C_k$ 的定义（正文和 Algorithm 1）里包含干净动作 $\mathbf a^j_1$，可 Figure 14 的 mask 和 Algorithm 2 里的上下文只有干净视频帧（原文 "Pass clean video, no action/state"），正文也说"autoregressive modeling only for the video modality"。理解时以 Figure 14 和 Algorithm 2 为准：动作不做自回归。

**4. 闭环推理**（Algorithm 2，Figure 4 右）
1. 用初始观测 prefill KV cache。
2. 每个 chunk 从噪声出发，用 flow ODE（Flow UniPC scheduler）对视频和动作联合去噪 $N$ 步。
3. 动作平滑后**异步**执行。
4. 把真实观测经 VAE 编码后写入 KV cache，丢弃这一轮预测的视频 latent。

因为上下文里始终是真实帧，自回归视频生成常见的 exposure bias / compounding error 就不存在了。论文强调纯视频生成做不到这一点，称之为 WAM 独有的好处。（笔记补充：Dreamer 部署时同样用真实观测更新 posterior 状态；它的 compounding error 出现在训练 actor 用的 imagination rollout 里，那里本来就没有真实帧可替换，两者用途不同，不宜直接类比。）预测出的视频在这里只充当"隐式视觉规划"，用完即丢。

**5. 实时化**（`real_time.tex`，Appendix D，Table 1）
- **异步执行**：控制器总是执行最新的动作块，推理总是用最新的观测。于是时间约束从"推理完成之前机器人不能动"放宽为"推理要在当前这个 1.6 s 的动作块执行完之前结束"，目标延迟定在约 200 ms 以下。
- **系统级**：CFG parallelism 把条件和无条件两次前向分到 2 张 GPU 上，单步延迟降低 47%。DiT caching 在相邻两步 velocity 的余弦相似度超过阈值时直接复用缓存的 velocity，有效步数从 16 降到 4。
- **实现级**：torch.compile + CUDA Graphs；cuDNN attention；scheduler 从 CPU 搬到 GPU；NVFP4 量化（QKV 和 Softmax 保留 FP8）。
- **模型级 DreamZero-Flash**：共享 $t$ 训练时，模型只见过"视频和动作噪声水平相同"的样本；而 1 步推理要求它在视频上下文仍然很噪的情况下直接给出干净动作，这就是 train-test mismatch。Flash 把两个模态的 timestep 解耦（Appendix D Eq.4–6）：

$$t^{\text{video}}_ k=1-\eta,\quad \eta\sim\mathrm{Beta}(7,1),\qquad t^{\text{action}}_ k\sim\mathcal U(0,1)$$

  这样 $\mathbb E[t^{\text{video}}]=0.125$，训练样本里的视频大多是高噪状态，模型必须学会"从噪视频上下文里预测干净动作"。Flash 作为主训练之后的最后一个阶段来做。注意 Algorithm 1 里写成 $t_{vid}\sim\mathrm{Beta}(7,1)$，漏了 " $1-$"（照字面均值是 0.875，偏向干净），以 §3.2 和 Appendix D Eq.5 为准。
- **动作平滑**：先用 cubic 插值上采样到 2×，再过 Savitzky-Golay 滤波（窗口 21、3 阶），最后降采样回原分辨率。
- **累积加速**（Table 1）：H100 上最终 9.6×。GB200 上，系统级加实现级优化累计 16.6×，再加 Flash 达到 38×，单个 chunk 从 5.7 s 降到 150 ms。Discussion 里说 7Hz 需要 2 张 GB200。除 DiT caching 和量化外，其余优化都与基线数学等价。

## 关键实验与证据

**评测口径**（§4）：训练数据的采集地和评测地在不同地理位置，所以默认条件就是"未见环境 + 未见物体"。AgiBot 每个任务 8 次 rollout，分布在 4 台机器人上，每个 checkpoint 共 80 次。DROID 用 20 个 seen 任务和 20 个 unseen 任务（DROID 中没有出现过的动词），每个任务 2 次。评分是按部分完成度计的 task progress。

**基线**：GR00T N1.6 和 π0.5，各有两种初始化——from-scratch（只用 VLM 权重）和 from-pretrained（官方的跨本体机器人预训练 checkpoint）。所有基线都在同一份数据上训练，并对齐 batch size 和梯度步数。DROID 上对比的是公开发布的 π0.5-DROID 和内部训练的 GR00T N1.6-DROID checkpoint（§4.1）。DreamZero 自身只有视频预训练、没有机器人预训练，在图里标为 "Scratch"。

- **Seen 任务 + 新环境**（Figure 8）：AgiBot 平均 task progress，DreamZero 62.2%，最好的 pretrained VLA（π0.5）27.4%，from-scratch VLA 接近 0（GR00T 0.6，π0.5 0）。分项对比 DreamZero 与 π0.5-pretrained：PnP Easy 93.8 vs 52.1，PnP Hard 48.4 vs 22.7，Contact-Rich 49.0 vs 9.2。DROID-Franka 上，task progress 82 vs 69（π0.5）/ 62（GR00T），success rate 75 vs 42 / 42。
- **Unseen 任务**（Figure 9）：AgiBot 上 10 个训练中没出现过的任务，平均 39.5%，pretrained π0.5 为 16.3%，from-scratch VLA 不到 1%。任务间差异很大：Remove Hat 85.7%，Shake Hands 59.2%，Fold Map 只有 8.2%。DROID 上，DreamZero 的 task progress / success 是 49% / 22.5%，GR00T 是 31% / 12.5%，π0.5 是 33% / 7.5%。
- **Post-training**（Figure 10）：三个任务的数据量分别为 33h、12h、40h，每个任务训 50K 步，评 10 次 rollout。平均 90.5，π0.5-pretrained 79.8。分任务：叠 T 恤 92.5 vs 92.5（持平），水果装袋 96 vs 71，收餐桌 83 vs 76。
- **仅用视频的跨本体迁移**（Table 2）：针对 9 个 unseen 任务，每种设置采 72 条轨迹（YAM 共 20 分钟，人类第一视角共 12 分钟）。这些数据只参与视频预测损失，与预训练数据 1:1 混合继续训 10K 步。结果从 38.3%±7.6 提升到 Robot2Robot 55.4%±9.5 和 Human2Robot 54.3%±10.4。
- **少样本换本体**（Figure 12）：用 YAM 上 55 条轨迹、11 个任务（约 30 分钟 play data）post-train AgiBot checkpoint，新本体能对新物体执行语言指令。这部分只有定性展示，没有数字。
- **Flash**（Table 3，收餐桌任务）：4 步 83%、350 ms；直接减到 1 步只有 52%、150 ms；Flash 1 步 74%、150 ms。
- **消融**（Table 4；因算力所限，都是 50K 步、batch 32，且只在 PnP Easy 上评测）：
  - 数据：同为 500h，diverse 数据 50%，repetitive 数据（70 个任务、大量重复示范）33%。
  - 规模：14B 50%，5B 21%；作者说 5B 更容易出现视觉幻觉，并传导成错误动作。把 VLA 做大（从 8B/32B 的 Qwen3 VLM 截取前一半层，再接 GR00T 式的 DiT 动作头，表中记作 5B/14B），两个都是 0%。
  - 架构：AR 和 BD 都是 50%，但 BD 的标准误是 ±14.4，方差更大。作者称 AR 的动作更平滑（定性描述），且借助 KV cache 推理快 3–4×。
- **失败分析**（Figure 16，Appendix H）：失败主要来自视频预测本身出错，机器人会"忠实执行一个错误的视觉计划"。这一方面说明视频和动作对齐得很紧，另一方面说明瓶颈在视频生成质量。

## 局限与论文没说的

**作者承认的**：
- 推理仍然很贵：7Hz 需要 2 张 GB200，而现有 VLA 在消费级 GPU 上就能跑到 20Hz 以上。
- 视觉记忆只有约 6 s，本质上是 System 1 模型；也没有评测需要记忆才能完成的任务（§3.1 脚注）。
- 在需要亚厘米精度的任务上（比如插钥匙、精细装配）继承了行为克隆的通病；以广度为先的预训练数据可能缺少这类任务需要的密集示范。
- 没有给出 WAM 的 scaling law；人类视频只用了实验室里采的 12 分钟；没有做多本体联合预训练。

**我读出来的**：
- **算力并不对等**：论文只对齐了 batch size 和步数。14B DiT 每步还要生成视频，每步 FLOPs 远高于 GR00T 和 π0.5。Table 4 里放大的 VLA 是"截掉一半层的 VLM + 动作头"，算不上同等强度的基线。
- **跨本体结论要打折扣**：用来迁移的视频，恰恰是被评测的那 9 个 unseen 任务的示范，严格来说已经不是 zero-shot。另外缺一个"只用预训练数据再训 10K 步"的对照，增益有多少来自额外的训练步数无从拆分。标准误在 ±10 个点左右，样本也只有 9 个任务 × 8 次 rollout（按 unseen 评测协议推算，Table 2 本身没写 rollout 数）。
- **消融偏薄**：只在最简单的 PnP Easy 上做；AR 和 BD 得分一样，"更平滑"没有量化指标。最核心的"KV cache 用真实帧替换"没有定量消融；共享 timestep 和独立 timestep 的对比也只有一句"收敛更快"，没有给曲线。
- **评测本身**：task progress 按部分完成度打 0–1 分，论文没说明打分者和逐项评分细则（真机评测大概率是人工判分）。"unseen task"是按"动作 + 物体类别"界定的，而 AgiBot 训练集还没公开（论文只说计划开源），外部没法核实这些任务是否真的没见过。Figure 12 和 100 多个自由形式任务都只有定性展示；Genie Sim 3.0 只说表现 "non-trivial"，没给数字。
- **它不是 simulator**：WAM 建模的是 $p(\mathbf o,\mathbf a\mid\cdot)$，而不是以动作为条件的 $p(s'\mid s,a)$，因此不能像 Dreamer、UniSim、Ctrl-World 那样拿来当 simulator 做 imagination RL 或策略评估。Appendix A 只点明了建模对象的区别（其他 world model 建模 forward dynamics，WAM 建模 $p(\mathbf o_{t:t+H},\mathbf a_{t:t+H}\mid\mathbf o_{0:t},\mathbf c)$），"不能当 simulator"是笔记的推论。整个系统里也没有任何 RL 或 reward。
- **失败模式**：每步只采样一次，既不检验视觉计划，也不重采样，视频幻觉会直接变成错误动作。5B 模型明显掉点，说明这套方法很依赖超大的视频主干。
- **训练用的 $t$ 分布有出入**：未被 `\input` 的旧稿里写的是 $t_k\sim$ Logit-Normal，与正式稿（§3.2、Appendix D Eq.4、Algorithm 1）的 $\mathcal U(0,1)$ 不一致，实际用哪个要以代码为准。

## 在脉络中的位置

- **前驱**：
  - UniPi（du2023learning）：视频生成 + IDM 当策略。
  - 联合视频-动作扩散：UWM、Video Policy、mimic-video、Genie Envisioner、Cosmos Policy、VPP。
  - 同一团队的 DreamGen：用视频 world model 合成机器人数据。
  - 视频主干 Wan2.1；自回归视频扩散与 teacher forcing 方面的 MAGI-1、Self Forcing、Pyramidal Flow。
  - 对比的 VLA 基线：π0.5、GR00T N1.6（引用的是 GR00T N1 论文）。
  - 论文把 [UniSim](../../05_intersections/unisim/notes.md) 归在"用轨迹预测做高层规划"一类。视频基础模型这条线可以对照 [Cosmos WFM](../cosmos-wfm/notes.md)，DreamZero 引用的是同系列的 Cosmos-Predict2.5 和 Cosmos Policy。
- **与 latent world model 的分野**（Appendix A）：
  - [Dreamer](../dreamer/notes.md)、[DreamerV3](../dreamerv3/notes.md)、[Dreamer 4](../dreamer4/notes.md) 学的是 $p(s_{t+1}\mid s_t,a_t)$。Appendix A 说这类模型要靠测试时的 planning / search 才能产出轨迹；但 Dreamer 系列实际上是在 imagination 里训练 actor-critic，部署时由 actor 直接出动作（这一点是笔记的补充，论文的概括并不准确）。
  - [V-JEPA 2](../vjepa2/notes.md) 在表征空间里做 MPC。
  - DreamZero 则直接从联合分布里采出动作，不需要测试时优化。
  - [Genie](../genie/notes.md) 和 DreamZero 都利用无动作视频，但机制不同：Genie 从中学 latent action，DreamZero 只对跨本体视频计算视频预测损失，不推断任何动作。
  - [DIAMOND](../diamond/notes.md) 同样说明，像素级扩散世界模型里视觉细节很关键。
- **后继 / 影响**：本项目 related_work 里暂时没有直接基于 DreamZero 的工作。[World Model for Robot Learning 综述](../survey-wm-robot-learning/notes.md) 在 world model 作策略的表里把它归入 single-backbone（视频与动作共享主干）一类。平行的方向是 [Ctrl-World](../../05_intersections/ctrl-world/notes.md)、[WMPO](../../05_intersections/wmpo/notes.md)、[World-Gymnast](../../05_intersections/world-gymnast/notes.md)：它们把视频 world model 当 simulator，用来评测或做 RL 训练 VLA。区别在于，在那里 world model 是环境，在 DreamZero 里 world model 本身就是 policy。
- **与本项目其他主题的联系**：
  - **TTT**：部署时每执行完一个 chunk 都能拿到真实帧，"预测帧 vs 真实帧"的视频预测误差就是现成的 self-supervised 信号，很适合做 [TTT](../../02_test_time_training/ttt-self-supervised/notes.md) 或 [PAD](../../05_intersections/pad/notes.md) 式的部署时适应。但论文没有做任何权重更新，写回 KV cache 只是 in-context 条件。6.6 s 的上下文上限可以对照 [TTT layers](../../02_test_time_training/ttt-layers/notes.md) 和 [One-Minute Video Generation with TTT](../../02_test_time_training/ttt-one-minute-video/notes.md) 的长上下文视频建模。"30 分钟 play data 换本体"属于 fine-tuning 式的快速适应，不是 TTT。
  - **Real2sim2real**：论文开源了让 DROID 版本在 PolaRiS（real-to-sim 评测）和 RoboArena 上运行的代码，但正文没有报告这两个评测的数字。PolaRiS 的思路和 [SIMPLER](../../03_real2sim2real/simpler/notes.md) 相同。它靠真实世界的环境多样性换泛化：22 个场景，一个任务攒满 50 个 episode 就下线（Appendix E），可以类比 [domain randomization](../../03_real2sim2real/domain-randomization/notes.md)。从人类视频迁移这一点可以对照 [X-Sim](../../03_real2sim2real/x-sim/notes.md)。
  - **RL**：DreamZero 是纯 BC。它的动作由联合视频-动作 DiT 用 flow matching 生成（没有单独的 flow 动作头），原则上可以接上 [π_RL](../../04_rl/pi-rl/notes.md) 或 [DPPO](../../04_rl/dppo/notes.md) 这类 flow/diffusion policy 的 RL fine-tuning，但对 14B 的联合视频-动作模型做 RL 的代价和可行性还没有人验证过。

## 前置知识

1. **Flow matching / rectified flow**：线性插值路径、velocity 回归、ODE 采样，以及 $t$ 的方向约定（本文 $t=1$ 表示干净）。
2. **DiT 与 latent video diffusion**：视频 VAE、image-to-video 条件注入、classifier-free guidance。
3. **自回归视频扩散**：chunk-wise teacher forcing（每个 chunk 独立的噪声水平）、block-causal mask、KV cache、exposure bias（可参考 Self Forcing）。
4. **行为克隆相关**：action chunking、inverse dynamics model，以及 UniPi 的"视频即策略"。
5. **VLA 训练范式**（π0.5、GR00T N1）和跨本体预训练。
6. **对照概念**：latent world model（RSSM/Dreamer）、JEPA + MPC。弄清这两者，才能理解 WAM 为什么"不是 simulator"。
7. （选读）部署工程：异步执行、torch.compile / CUDA Graphs、低精度量化。

## 复现可行性

- **代码**：官方仓库已开源（见文首）。论文层面承诺的是权重、推理代码和评测代码；DreamZero-DROID checkpoint 可以直接在 PolaRiS 的 DROID-sim 上跑（§4.1 脚注）。
- **算力**：论文**没有报告**训练用的 GPU 型号、数量和时长，只给了几组步数：
  - 预训练：AgiBot 和 DROID 各 100K 步，global batch 128。
  - Post-train：每个任务 50K 步。
  - 消融：因算力限制降到 50K 步、batch 32。

  训练是 14B DiT 的全参更新（只冻结 VAE 和文本/图像编码器），需要多机多卡。推理方面，7Hz 用的是 2×GB200，而 CFG 并行本身就需要 2 张卡。H100 上最多加速 9.6×；NVFP4 和 Flash 在 Table 1 里只列了 GB200 的结果。
- **数据/环境**：DROID 公开。AgiBot 的 500h 数据还没开源；post-train 数据、YAM 数据、人类视频数据论文都没说会公开。真机复现需要 AgiBot G1，或者 DROID 标准的 Franka 平台。
- **最小可复现实验**：
  1. **不训练**：下载 DreamZero-DROID checkpoint，用 2 卡起推理服务，在 PolaRiS / DROID-sim 上评测。把"预测视频 vs 实际执行"可视化出来，验证 Figure 16 说的对齐现象。再分别开关 DiT caching，测延迟和成功率的变化。
  2. **机制级小实验**（这是我建议的设置，论文里没有）：在仿真操作数据上，用一个较小的开源视频扩散主干做三组对比：
     - (a) 联合视频-动作 flow matching vs 同一主干只预测动作；
     - (b) 共享 $t$ vs Flash 解耦 $t$，比较 1 步推理时的成功率；
     - (c) 推理时 KV cache 里放真实帧 vs 放模型自己生成的帧，量化 exposure bias。(c) 恰好补上论文缺失的那个消融。
- **预计难点**：
  - 14B 模型的显存和吞吐。
  - KV cache 形状在第一条轨迹里不断变化，会导致 torch.compile 反复重编译。
  - NVFP4 依赖 Blackwell 架构。
  - 真机评测方差大，且依赖人工打分。
  - 小主干未必能复现"视频先验带来泛化"：Table 4 里 5B 就已经从 50% 掉到 21%。

## 自测问题

1. 写出 Eq.1 的分解。DreamZero 为什么不分开训练视频模型和 IDM？按作者的解释，500h 机器人数据主要在教哪一项？这又怎么解释"diverse 数据优于 repetitive 数据"？
2. 画出 $M=3$ 时训练用的 attention mask： $Y_2$ 能 attend 到哪些 token？推理时把 $C_k$ 换成真实观测，为什么就能消除 exposure bias？纯视频生成为什么做不到这一点？Dreamer 的 imagination rollout 与此有何不同？
3. 用共享 timestep 训练的 DreamZero 做 1 步推理，train-test mismatch 具体在哪里？Flash 取 $\eta\sim\mathrm{Beta}(7,1)$、 $t^{\text{video}}=1-\eta$，求 $\mathbb E[t^{\text{video}}]$，并说明在" $t=1$ 为干净"的约定下，这个值意味着什么样的训练样本。
4. 用 AgiBot 的设置（5 FPS、30Hz、 $K=2$、 $H=48$）推出一个 chunk 覆盖多长时间。异步执行为什么能把延迟预算放宽到约 200 ms？DiT caching 为什么不属于"数学等价"的优化？
5. Table 2 的跨本体数据只参与视频损失，它改进的是分解中的哪一项？要把 38.3→55.4 的增益归因于 YAM 视频本身，还缺哪个对照实验？
