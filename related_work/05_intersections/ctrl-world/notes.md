# Ctrl-World: A Controllable Generative World Model for Robot Manipulation

> arXiv [2510.10125](https://arxiv.org/abs/2510.10125) · ICLR 2026（Poster） · 主题：05_intersections · 层级：milestone
> 代码：https://github.com/Robert-gyj/Ctrl-World（MIT；含训练/推理代码与 DROID 上训好的 checkpoint，已核实仓库存在）· 项目页 https://ctrl-world.github.io · 笔记依据：`source/*.tex`（数字/公式）+ `paper.pdf`（图）

## 一句话

把预训练的 Stable Video Diffusion（SVD, 1.5B）改造成一个**多视角（含腕部相机）、逐帧动作条件、带 pose-conditioned memory retrieval** 的动作条件视频 world model，让现成的 VLA（π0 / π0-FAST / π0.5-DROID）在"想象"中闭环 rollout，从而**在不跑真机的情况下给策略排名**，并**把想象中筛出的成功轨迹拿来 SFT**，使 π0.5 在新指令/新物体上的成功率从 38.7% 升到 83.4%。

## 问题设定

- **目标**：学一个 world model $W$，能接住通用策略输出的 action chunk，预测多视角未来观测，再把预测喂回策略，形成 policy-in-the-loop 的自回归 rollout（`source/sec/2Method.tex` §3）：
  $$a_{t+1},\dots,a_{t+H}\sim\pi(\cdot\mid o_t,l)\quad(\text{Eq.1}),\qquad o_{t+1},\dots,o_{t+H}\sim W(\cdot\mid o_t,A_t)\quad(\text{Eq.2})$$
  其中 $o_t=[I_t^1,\dots,I_t^n,q_t]$ 是 $n$ 个相机视角 + 机械臂位姿， $l$ 是语言指令； $o_{t+H}$ 再送回 $\pi$ 得到下一个 chunk。
- **数据/环境假设**：只用开源 DROID（Franka Panda + Robotiq 夹爪，1 个腕部相机 + 2 个随机摆放的第三视角相机）；95,599 条轨迹、564 个场景，约 76k 成功 / 19k 失败（`3Exp.tex` §5.1）。作者强调失败数据对"可控性"很关键——模型得见过"动作做错了会怎样"。
- **奖励**：没有学 reward model；成功/失败由**人工判定**（`2Method.tex` §4.2，附录 B 给了每个任务的判定标准）。
- **之前卡在哪**（`0Intro.tex`）：已有动作条件视频模型（IRASim、WPE、WorldEval）基本只生成**单一第三视角**，① 部分可观测导致幻觉（物体没被接触就"吸"进夹爪）；② 与需要第三视角 + 腕部视角输入的现代 VLA 不兼容；③ 缺乏对高频动作的细粒度控制；④ 长时程生成不一致。

## 核心方法

骨干是 SVD 的 latent video diffusion（VAE 8×8 下采样 + 交替的 spatial / temporal transformer）。**唯一新初始化的参数**是一个 3 层 MLP，把 7 维 Cartesian 动作投到 1024 维（附录 A，`5appendix.tex`）。三个改动（Fig. 2）：

**1. Multi-view joint prediction。** 3 个视角各 192×320，经 VAE 成 24×40 latent；把 $N$ 个视角的 token 沿 token 维拼接，每帧 $P=N\times H\times W=3\times24\times40$ 个 token。spatial transformer 在 $(B\cdot T, P, C)$ 上做注意力——同一时刻的三个视角互相可见；temporal transformer 在 $(B\cdot P, T, C)$ 上沿时间做注意力（Fig. 2）。总输入形状 $B\times(7+5)\times(3\times24\times40)$（附录 A）：7 帧历史 + 5 帧待预测。直觉：腕部视角直接看到接触与物体状态变化，第三视角提供全局；联合预测让两者互相约束（Fig. 5 中绿框内容来自其他视角推断）。

**2. Pose-conditioned memory retrieval。** 为抑制长 rollout 漂移，按步长 $m$ 稀疏采 $k$ 帧历史放进上下文（正文无编号式）：
$$o_{t+1:t+H}\sim W(\cdot\mid o_{t-km},\dots,o_{t-m},o_t,l)$$
实现中 $k=7$、历史帧间隔 1–2 s（§5.1），每个历史帧加**独立随机噪声**以提升鲁棒性（附录 A，相当于对上下文做 noise augmentation，缓解自回归时的误差累积）。关键在"pose-conditioned"：每个历史帧也通过逐帧 cross-attention 注入其对应的臂位姿 $q_{t-km},\dots,q_t$。这样 temporal attention 可以把"当前要去的位姿"与"历史上相同位姿的帧"匹配起来，从而取回那时看到的内容——没有显式检索模块，检索是注意力自发完成的。Fig. 4 左的注意力可视化：预测 $t=4$ s 帧时强烈关注位姿相同的 $t=0$ s 帧。

**3. Frame-level action conditioning。** 把策略的 action chunk 转成 Cartesian 位姿 $a'_ {t+1:t+H}$，与历史位姿拼成一条位姿序列 $[q_{t-km},\dots,q_{t-m},q_t,a'_ {t+1:t+H}]$，在 spatial transformer 里做**逐帧** cross-attention：第 $i$ 帧的视觉 token（ $B\cdot T, P, C$）只 attend 自己那一帧的位姿 embedding（ $B\cdot T, 1, C_a$）。沿用 IRASim（及 He et al. 2025）的 frame-wise cross-attention；与"整段动作作为一个全局条件"相比，每一帧都对齐到对应时刻的动作。15 步动作（DROID 下 1 s）为省显存在时间上降采样成 5 步（附录 A），因此每次交互实际生成 5 帧未来（由附录 A 的输入形状 $7+5$ 推断，正文未直说）。另有 CLIP 语义 token 作为全局条件（Fig. 2）。

**训练目标**（`2Method.tex` Eq.3）：目标 $x_0=o_{t+1:t+H}$ 加噪
$$x_{t'}=\sqrt{\bar\alpha_{t'}}\thinspace x_0+\sqrt{1-\bar\alpha_{t'}}\thinspace\epsilon,\qquad \epsilon\sim\mathcal N(0,I)$$
模型输入是 $[o_{t-km},\dots,o_{t-m},o_t,x_{t'}]$（历史帧 + 按扩散步 $t'$ 加噪的未来帧；历史帧不是去噪目标，但附录 A 说每个历史帧另加独立随机噪声做增强，所以并非完全"干净"），损失写成 $x_0$-prediction：
$$\mathcal L=\mathbb E_{x_0,\epsilon,t'}\big\Vert\hat x_0(x_{t'},t',c)-x_0\big\Vert^2,\quad c=[q_{t-km},\dots,q_t,\thickspace a'_ {t+1:t+H},\thickspace o_{t-km},\dots,o_t]$$
训练采样：随机取轨迹 → 均匀取当前帧 → 往回采 memory 帧 → 后续帧作目标（附录 A）。

**与 VLA 对接**（附录 B）：官方 openpi 的 DROID 策略输入关节角 + 两路图像，输出关节速度；而 world model 用末端 Cartesian 位姿。作者在 DROID 上训一个 2 层 MLP adapter，再做 Franka 正运动学：
$$a^{jv}_ {t+1:t+H}=\pi(q^{joint}_ t,o_t,l),\quad q^{joint}_ {t+1:t+H}=\mathrm{Adapter}(q^{joint}_ t,a^{jv}_ {t+1:t+H}),\quad q^{cart}_ {t+1:t+H}=FK(q^{joint}_ {t+1:t+H})$$
策略输出不足 15 步时用 dummy action 补齐，只取有效部分的预测（§5.1）。

**推理 / 使用**（Algorithm 1）：对每个任务 $l^i$ 与初始观测 $o^i_0$，循环 $N$ 次：扰动后的策略出 chunk → 取稀疏历史 $h$ → $W(h,o_t,a_{t+1:t+H})$ 生成下一段 → 拼到 $\tau$；结束后人工判成功，成功的放进 $D_s$；最后 $\mathcal L_\theta=\mathbb E_{D_s}\Vert\pi_\theta(o_t,l)-a_{t:t+H}\Vert^2$ 做 SFT（Algorithm 1 把它写成 MSE；实际用 openpi 官方代码微调 π0.5，附录 C）。

**为什么要"扰动"**：作者观察到固定初始观测与指令时，VLA 行为几乎确定（反复抓同一个物体），搜索空间太窄。两种结构化扰动：① 用 LLM（Gemini）改写指令；② 用线性插值运动规划把机械臂先移到随机初始位姿（§4.2、§5.4）。

## 关键实验与证据

**World model 质量**（Table 1，验证集留出 2%，256 段 10 s 片段，每轮输入 15 步 chunk、自回归 10 轮；仅第三视角指标）：

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | FID↓ | FVD↓ |
|---|---|---|---|---|---|
| WPE-Single-View | 20.33 | 0.772 | 0.131 | 25.50 | 156.4 |
| IRASim-Single-View | 21.36 | 0.774 | 0.117 | 26.46 | 138.1 |
| Ctrl-World-Single-View | 21.27 | 0.793 | 0.110 | **23.47** | 127.5 |
| Ctrl-World | **23.56** | **0.828** | **0.091** | 25.00 | **97.4** |

值得注意：把基线直接改成多视角并不自动变好（IRASim-Multiview PSNR 20.21 / FVD 165.4，比其单视角版本更差；WPE-Multiview 21.17 / 147.1，只比其单视角版本略好，仍远逊于 Ctrl-World），说明收益来自整套设计而不只是"多喂几个视角"。

**消融**（Table 2）：第三视角上，去掉 memory → FVD 97.4→105.5；去掉 frame-level cond → 122.7。腕部视角：full 19.18 PSNR / FVD 127.1；w/o memory 18.84 / 133.1；w/o frame-level cond 15.69 / 179.1；w/o joint pred 15.94 / 158.1（joint pred 消融只在腕部视角上报）。去掉 frame-level cond 在腕部视角上的影响明显更大（FVD +52.0，第三视角 +25.3）；去掉 memory 的影响两视角相近（FVD +6.0 vs +8.1）。结论：**逐帧动作条件与多视角联合预测是主要贡献者**，memory 在指标上的增益较小，但 Fig. 4 定性显示去掉 memory 会变糊；Fig. 4 还展示了几厘米级别的动作差异能生成不同 rollout。

**策略评估**（§5.3，Fig. 7，附录 Table 3）：作者自搭 DROID 平台、相机随机摆放，world model **zero-shot** 用在新场景/新相机位姿上。3 个策略 × 7 个任务（Pick-Place、Fold-Towel、Drawer、Wipe-table、Close-laptop、Pull-tissue、Stack），真机与想象用同一初始观测，每条想象轨迹 20 次交互（Fig. 6）。Fig. 7 的回归线：instruction following 为 $y=0.87x-0.04$，success rate 为 $y=0.81x-0.11$（ $x$ 为真机， $y$ 为 world model）。论文没有报相关系数；我用 Table 3 的 21 个点自算得 Pearson $r\approx0.97$（instruction following）和 $\approx0.83$（success rate），自算回归线（IF $y=0.87x-0.04$；SR $y=0.82x-0.11$）与 Fig. 7 基本一致。按策略对 7 个任务取平均（自算），真机与想象里都是 π0 < π0-FAST < π0.5，排名保持。但 world model **系统性低估执行成功率**，且在接触/铰接物体上偏差很大：Close-laptop 的 π0.5 真机 SR 0.70 vs 想象 0.05，Drawer 的 π0.5 0.80 vs 0.30（Table 3）。

**策略改进**（§5.4，Fig. 9，附录 Table 4–7）：基座 π0.5-DROID；每个任务在想象中生成 400 条轨迹，人工保留 25–50 条成功轨迹，微调 2k 步（4×H100，附录 C）。

| 类别 | Base | After post-training |
|---|---|---|
| Spatial（Table 4） | 0.2875 | 0.875 |
| Shape（Table 5） | 0.4374 | 0.9125 |
| Towel 方向（Table 6） | 0.575 | 0.80 |
| Novel object（Table 7） | 0.25 | 0.75 |
| 平均（Fig. 9） | 38.7% | 83.4% |

摘要里"提升 44.7%"是**绝对百分点**（83.4 − 38.7）。

## 局限与论文没说的

**作者承认的**（`4conclusion.tex`、§5.3）：精细交互和长时程推理会失败；对初始观测敏感；碰撞、物体滑走、旋转等复杂物理建模不准（笔记本电脑交互明显失真）；真实策略失败后会反复重试，world model 有时模拟不出来；很多失败模式在 DROID 分布外；实验只针对 instruction following，作者认为模型**不够准，无法提升已见指令上的低层成功率**；判定靠人工。

**我读出来的**：
- **"排名准确"的证据面较窄**：只有 3 个同家族、能力差距明显的策略（平均 IF 真机 0.37 / 0.52 / 0.82，自算）。能力接近的策略、不同训练范式的策略是否仍能排对，没有测；也没报 Kendall/Spearman 等排名指标。每格评测次数正文未写（数值都是 0.05 的倍数，推测每格约 20 次，推测而非原文）。
- **成功率是有偏估计，且偏差依赖任务**：截距 −0.11，Close-laptop 几乎判成全失败。用它估计绝对成功率或筛选"安全"策略会误判；偏差的方向和大小依赖物理类型。
- **改进算法本质是 filtered BC / rejection-sampling SFT**：只用正样本，没有 reward model、value 或 policy gradient，也没有迭代（作者列为 future work）。评测用的就是生成合成数据的同一批下游任务（Algorithm 1 以这些任务的指令与初始观测为输入），属于定向适配；论文未说明 Fig. 9 的评测环境（真机还是 world model）、每格试验次数，以及评测初始摆放是否与合成数据相同；没检查其他任务是否退化（遗忘），也没与"同等数量真机示教 SFT"做对照。
- **人工标注成本未量化**：每任务 400 条想象轨迹需要逐条人工判定（改进实验的轨迹长度原文未报告），论文没有给出标注耗时；换成 VLM 判定后是否还有同样效果，未验证。
- **推理成本未报告**：论文没给扩散采样步数和单步耗时。仓库 README 写的是"一次交互约 10 s（A100）/ 5 s（H100）"，据此一条 20 次交互的 rollout 约 100–200 s，400 条/任务需要大量 GPU 时间（此为我按 README 估算）。
- **接口误差源未消融**：joint velocity → 学习到的 MLP adapter → FK 这一链条的误差、15 步降采样到 5 帧（5 Hz 视觉）对精细动作的影响，都没有单独分析。
- **只测了一个 embodiment**（Franka/DROID）、桌面操作；"新场景"仍是同一平台，跨平台/跨机器人泛化没有测。Table 1 的多视角基线只给了 PSNR/FVD。

## 在脉络中的位置

- **前驱**：动作条件视频预测可追溯到 Oh et al. 2015 与 Visual Foresight（Finn & Levine 2017；Ebert et al. 2018），[UniSim](../unisim/notes.md) 把它推向"交互式 real-world simulator"（以上本文均有引用）；frame-level action conditioning 直接沿用 IRASim；world-model-based policy evaluation 沿 WPE（Quevedo et al. 2025）、WorldEval；骨干为 SVD（Blattmann et al. 2023）；多视角联合建模借鉴 VGGT / EnerVerse；数据与策略来自 DROID 与 π0 / π0-FAST / π0.5（openpi）。视频 world foundation model 背景见 [Cosmos](../../01_world_model/cosmos-wfm/notes.md)（本文引用）、[Genie](../../01_world_model/genie/notes.md)（本文引用的是 Genie 3，不是 2024 年的 Genie）。
- **后继 / 影响**：[World-Gymnast](../world-gymnast/notes.md) 把"按 Ctrl-World 做法在 world model 里 rollout + VLM 筛成功 + 迭代 SFT"作为 Iter-SFT 基线，与在 world model 里做 RL 对比；[WMPO](../wmpo/notes.md) 同期走"在视频 world model 里对 VLA 做 on-policy RL"的路线（其源码未引用本文）。[World Model for Robot Learning 综述](../../01_world_model/survey-wm-robot-learning/notes.md) 把它当作 world model 作为 policy evaluator 的代表。
- **与本项目其他主题的联系**：
  - **World model**：与 [Dreamer](../../01_world_model/dreamer/notes.md) / [DreamerV3](../../01_world_model/dreamerv3/notes.md) 的 latent imagination 对比：Dreamer 是任务专用 RSSM + reward/value head + actor-critic；Ctrl-World 是像素空间、多任务、不预测 reward 的生成式模型，策略是外部 VLA。[Dreamer 4](../../01_world_model/dreamer4/notes.md) 在可扩展 world model 里做 RL，是更"RL 化"的对照；[DIAMOND](../../01_world_model/diamond/notes.md) 同样论证了扩散 world model 中视觉细节的重要性。
  - **TTT**：memory retrieval 是**上下文内检索**（注意力读稀疏历史），不更新权重；"对初始观测敏感"、"需要 in-domain rollout 数据"正是 test-time adaptation 的切入点，例如用 [TTT layers](../../02_test_time_training/ttt-layers/notes.md) / [Titans](../../02_test_time_training/titans/notes.md) 替代稀疏历史帧作长时记忆，或借 [PAD](../pad/notes.md) 的 self-supervised 目标在新场景上在线适配 world model。
  - **Real2sim2real**：用真机快照当初始帧，是一种不做 3D 重建、不做 system identification 的轻量 real-to-sim。可与 [SIMPLER](../../03_real2sim2real/simpler/notes.md)（visual matching + 物理仿真评测）、[SplatSim](../../03_real2sim2real/splatsim/notes.md)、[ACDC](../../03_real2sim2real/acdc-digital-cousins/notes.md) 对照：sim-to-real gap 在这里变成"world model 与真实之间的 gap"，Table 3 的成功率低估就是它的量化表现。
  - **RL**：改进环节是人类二值奖励 + 过滤后监督学习，离真正的 RL 还差一步。在想象里做 PPO/GRPO 的版本见 [WMPO](../wmpo/notes.md)、[World-Gymnast](../world-gymnast/notes.md)；在真实/物理仿真里对 VLA 做 RL 见 [π_RL](../../04_rl/pi-rl/notes.md)、[SimpleVLA-RL](../../04_rl/simplevla-rl/notes.md)、[π*0.6](../../04_rl/pi-star-06/notes.md)；人工奖励信号可对照 [HIL-SERL](../../04_rl/hil-serl/notes.md)。

## 前置知识

1. 扩散模型训练：前向加噪、 $\epsilon/x_0$-prediction、EDM 预条件（SVD 用的就是这一套）。
2. Latent video diffusion 与 SVD 结构：VAE latent、spatial / temporal 分解注意力、cross-attention 条件注入。
3. 动作条件视频预测 / learned simulator：UniSim、IRASim 的 frame-level conditioning。
4. VLA 与 action chunking：π0 / π0.5（flow matching 动作头）、openpi 的 DROID 接口（关节速度、关节角）、正运动学 FK。
5. DROID 平台与数据（腕部 + 双外部相机、成功/失败轨迹）。
6. 视频生成指标 PSNR / SSIM / LPIPS / FID / FVD；策略评估中的相关性与排名一致性。
7. Filtered BC / rejection-sampling fine-tuning 与 policy gradient 的区别。

## 复现可行性

- **代码**：官方仓库 https://github.com/Robert-gyj/Ctrl-World（MIT），README 列出训练代码、三种推理模式（回放轨迹、键盘控制、π0.5 policy-in-the-loop）、Ctrl-World checkpoint（约 8 GB）、SVD 与 CLIP 权重；依赖 Python 3.11 + PyTorch；与 π0.5 联跑需另装 openpi（JAX），并设 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.4`（来自 README，不是论文）。
- **训练算力（论文）**：2×8 H100，batch 64，lr 1e-5，100k steps，约 2–3 天（附录 A / §5.1）；π0.5 微调 4×H100、2k steps（附录 C）。推理耗时论文未报告，README 写一次交互约 10 s（A100）/ 5 s（H100）；显存需求两处均未写。
- **数据**：全量 DROID（README 称约 370 GB，需先用 SVD VAE 预提取 latent）。
- **真机依赖**：Fig. 7 的"排名对齐"需要一套 DROID 硬件（Franka + Robotiq + 3 相机）做真机对照；Fig. 9"改进 +44.7 点"的评测环境原文未明说，若为真机评测同样需要这套硬件。
- **最小可复现实验**（单卡 A100/H100，只做推理）：
  1. 用发布的 checkpoint 在 DROID 验证子集（比如 32 段）上回放真实 action chunk 10 轮，计算第三视角和腕部视角的 PSNR / SSIM / LPIPS / FVD，核对 Table 1/2 的量级，以及"腕部比第三视角难"的趋势。
  2. 可控性检验：从同一初始帧出发，只改一个轴（如 z ±3–6 cm），看 rollout 是否按厘米级区分（复现 Fig. 4 的定性结论）；再把历史帧打乱或去掉位姿注入，观察一致性是否下降（近似 memory 消融）。
  3. 若能跑 openpi：在 DROID 验证帧上用 π0 与 π0.5 各做 20 次交互的 rollout，人工/VLM 判定，检查想象中的排名是否与 Table 3 的方向一致。
- **预计难点**：推理慢（按 README 速度，256 段 × 10 轮约需数个 H100 小时，为估算）；PyTorch 与 JAX 共用一张卡时的显存分配；adapter 与 FK 的接口对齐；成功判定的主观性；完整重训需要 16×H100 跑 2–3 天。

## 自测问题

1. 多视角联合预测在 token 层面怎么实现（spatial / temporal attention 分别在哪个维度上做）？为什么加入腕部视角能减少"物体未接触就进夹爪"这类幻觉？Table 2 里哪一行支持这一点？
2. "Pose-conditioned memory retrieval"没有显式检索模块，检索是怎么发生的？如果只给未来帧注入动作，不给历史帧注入 $q$，temporal attention 会失去什么能力？历史帧为什么要加独立噪声？
3. 从 π0.5-DROID 输出的关节速度到 world model 的条件输入，完整链条是什么？哪几步会引入与真实执行不一致的误差？
4. Fig. 7 中 success rate 的回归是 $y=0.81x-0.11$，instruction following 是 $y=0.87x-0.04$。据此，Ctrl-World 更适合做"排名"还是"绝对成功率估计"？为什么 Close-laptop / Drawer 的偏差特别大？
5. Algorithm 1 的改进环节在 RL 术语里属于哪类方法？为什么需要改写指令和随机重置初始状态？如果要改成 PPO/GRPO 式的 policy gradient，还缺哪些组件（奖励来源、策略随机性、对 world model 被 exploit 的防护）？
