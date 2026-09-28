# Learning Interactive Real-World Simulators（UniSim）

> arXiv [2310.06114](https://arxiv.org/abs/2310.06114) · ICLR 2024（Oral，Outstanding Paper Award） · 主题：05_intersections · 层级：foundational
> 代码：none（论文只给出视频 demo 页 https://universal-simulator.github.io，没有给出代码或权重链接）· 笔记依据：`source/*.tex`（数字/公式）+ `paper.pdf`（图）

## 一句话

把互联网图文、人类活动视频、机器人数据、全景扫描和仿真渲染统一成"动作（T5 文本嵌入 + 离散化低层控制）进、视频出"的格式，训练一个 5.6B 的视频扩散模型。它以最近 4 帧历史为条件做 observation prediction，可以自回归 rollout，当作"真实世界 simulator"：用它生成的 hindsight relabeling 数据训练高层 VLM policy，在它里面训练低层 VLA 的 RL 策略（REINFORCE + learned reward），并定性展示了 zero-shot 部署到真实 Language Table 机器人的效果。

## 问题设定

- **目标**：学一个 action-conditioned 的视频 world model $p(o_t\mid h_{t-1},a_{t-1})$。$h_{t-1}$ 是历史帧，$a_{t-1}$ 是一个 temporally extended action（语言指令、机器人低层控制、相机运动都可以），$o_t$ 是接下来的一段视频（帧数可变）。输出只有 RGB 像素，没有状态、奖励和 termination。
- **假设**：环境部分可观测，只用有限历史近似；环境随机性交给扩散采样去表达；奖励不放在 simulator 里，按任务另学 reward model（`source/method.tex` §2.2；随机性的说法见 §3.1）。
- **之前卡在哪**（`source/related.tex`, `source/introduction.tex`）：
  1. 经典 system identification 和 model-based RL 一般每个系统单独学一个低维状态 dynamics，知识不能跨系统共享。像素级 world model（Dreamer、IRIS 等）大多在游戏或仿真这类视觉简单、数据充足的域里。
  2. 可控视频生成大多做特定域（如自动驾驶），控制粒度有限，没有被当成 dynamics modeling 问题来做。
  3. 数据在不同维度上各有长处：图文数据场景和物体多但运动很少；人类视频有高层动作，但缺少机械运动细节；机器人数据有动作标签，但量少。单个数据集覆盖不全。
  4. UniPi 已经证明视频生成可以当 policy 用，但 policy learning 的瓶颈在于"真实环境访问受限"。UniSim 针对的正是这个瓶颈。
- 和传统 sim-to-real 相比，它不用 physics engine + domain randomization，也不做 3D reconstruction 式的 digital twin，而是一个纯数据驱动的像素空间 simulator。

## 核心方法

**1. 统一动作空间**（`source/method.tex` §2.1，附录 Datasets 段，Table 5）

| 数据类型 | 例子 | 动作怎么构造 |
|---|---|---|
| 仿真渲染 | Habitat HM3D、Language Table sim | 有文本描述就用文本；连续控制离散化后和文本嵌入拼接 |
| 真实机器人 | Bridge、RT-1、Language Table real | 任务描述作为高层动作，有低层控制就一起拼上 |
| 人类活动 | Ego4D、EPIC-KITCHENS、SSv2 | 分类标签转成文本动作；视频下采样到能体现动作的帧率 |
| 全景扫描 | Matterport R2R | 截取全景，并用两张图之间的相机位姿构造 "turn left" 这类动作 |
| 互联网图文 | LAION-400M、ALIGN | 图像当单帧视频，caption 当动作 |

文本统一编码成 T5 embedding。低层控制一方面写成文本做 T5 编码，另一方面先归一化、再离散成 4096 个 bin，和语言嵌入拼接。混合权重只取 0.1 或 0.05 两档，"without careful tuning"。低数据域（Habitat）在动作前加 domain identifier（数据集名）可以提升生成质量，但作者提醒这样做会损害跨域泛化（Figure 6）。

**2. Observation prediction model**（`source/method.tex` §2.2）
理想情况下应该通过 recurrent state 条件在全部历史 $(o_0,a_0,\dots,a_{t-2},o_{t-1})$ 上。实际做法是只条件在有限帧上（取自上一段视频的 4 帧，即 Table 1 的 "4 recent"），而且历史里**不放过去的动作**。长程模拟就是自回归：采出 $o_t$，取其末尾若干帧作为新的 $h_t$，再接下一个动作。对比 RSSM：UniSim 没有 latent state，也没有 reward/continue head，用一个固定历史窗口顶替 recurrent state。这样 simulator 与任务无关，任何 reward 都能接上去做 planning 或 RL。

**3. 扩散参数化**
训练损失（`source/method.tex`，未编号的 equation*）：
$$\mathcal{L}_{\text{MSE}}=\Big\|\epsilon-\epsilon_\theta\big(\sqrt{1-\beta^{(k)}}\,o_t+\sqrt{\beta^{(k)}}\,\epsilon,\;k\,\big|\,h_{t-1},a_{t-1}\big)\Big\|^2,\quad \epsilon\sim\mathcal N(0,I)$$
动作用 classifier-free guidance 注入（`source/method.tex` Eq.1）。这里只对动作做 guidance，历史始终作为输入：
$$\tilde\epsilon_\theta(o_t^{(k)},k\mid h_{t-1},a_{t-1})=(1+\eta)\,\epsilon_\theta(o_t^{(k)},k\mid h_{t-1},a_{t-1})-\eta\,\epsilon_\theta(o_t^{(k)},k\mid h_{t-1})$$
$\eta$ 控制动作条件强度。原文左右两边用的是同一个符号，第二项写成了 $o_t$，这是记号笔误，我这里改写成 $\tilde\epsilon$。反向采样（`source/method.tex` Eq.2）：
$$o_t^{(k-1)}=\alpha^{(k)}\big(o_t^{(k)}-\gamma^{(k)}\epsilon_\theta(o_t^{(k)},k\mid h_{t-1},a_{t-1})\big)+\xi,\quad \xi\sim\mathcal N(0,\sigma_k^2I)$$
初始 $o_t^{(K)}\sim\mathcal N(0,I)$，并且和历史帧**按通道拼接**。论文没写训练时丢弃动作条件的概率（这是 CFG 的无条件分支需要的）。

**4. 架构与训练**（`source/appendix.tex` Architecture 段 + Table 6）
- 采用 3D U-Net（video U-Net），空间和时间的 attention、conv 交替堆叠。整体是 Imagen Video 式的级联：base 模型在 $[16,24,40]$（16 帧 × 24×40）上做历史条件预测，两个空间超分模型分别做 $[24,40]\to[48,80]$ 和 $[48,80]\to[192,320]$。base 用 temporal attention，超分模型为了效率用 temporal conv。
- 历史条件的做法：取上一段的 4 帧，复制到每一个未来帧位置，与噪声按通道拼接。
- 5.6B 参数，512 TPU-v3 训练 20 天（`source/method.tex`），1M 步，batch 256，Adam lr 1e-4，cosine noise schedule，采样 256 步，预测目标为 $\epsilon$，条件嵌入维度 4096，条件 token 长度 64。
- 纯文本生成：历史帧位置放白图占位，同时加大文本的 CFG（附录 Video Captioning 段）。

**5. 三种下游用法**（`source/experiment.tex` + 附录 D）
- **Hindsight relabeling 训练长程 VLM 策略**：在 UniSim 里每条轨迹串 3–5 个脚本化语言指令做 rollout，共得到 10k 条长程轨迹。用末帧当 goal image，用串起来的指令当监督，微调 PaLM-E 12B（加 goal image 输入），耗时 64 TPUv3 × 1 天。执行时按 UniPi 方式：VLM 输出指令，UniSim 生成 16 帧视频，再由单独训练的 inverse dynamics model 反推低层动作。评测指标 RDG（`source/experiment.tex` Eq.3）：
$$\mathrm{RDG}=\frac{\|s_0-s_{\text{goal}}\|_2-\|s_T-s_{\text{goal}}\|_2}{\|s_0-s_{\text{goal}}\|_2}$$
- **RL 训练低层 VLA**：分两个阶段。Stage 1 用 BC 微调 PaLI 3B，同时训"剩余步数预测" $d(o,g)$ 和"指令预测"（lr 1e-3，batch 128，300k 步）。Stage 2 复制一份 Stage 1 checkpoint 并冻结，作为 reward model，奖励定义为
$$r(o_t,a_t,o_{t+1},g)=-\big[d(o_{t+1},g)-d(o_t,g)\big]\cdot\mathcal C,\quad \mathcal C=0.05$$
被训练的策略同样从 Stage 1 checkpoint 初始化（冻结的只是 reward model 那份副本），用 REINFORCE 更新：$\nabla_\pi\mathcal L=\nabla_\pi\log\pi(a_t\mid o_t,g)\sum_{i=t}^{T}\gamma^{i-t}r_i$，其中 $r_i=r(o_i,a_i,o_{i+1},g)$。环境这一侧，把 UniSim 推理包装成 RPC，再套进 DM Env API 的 `step()`。Table 7 的设置：64 个 actor，每个 episode 最多 100 步，$\gamma=0.9$，图像历史 2 帧，learner batch 64，框架为 ACME。actor 做了限速，避免数据太 off-policy。论文没提 baseline 或 critic。
- **合成数据训练 video captioning**：对 ActivityNet Captions 训练集的 30,740 条文本，每条生成 4 段视频，用来微调 PaLI-X 55B。

## 关键实验与证据

- **历史条件消融**（Table 1，Ego4D 验证集）：1 帧时 FVD 315.69、FID 59.47；4 帧间隔按指数增长的远历史（"4 distant"）FVD 237、FID 34.89；最近 4 帧 FVD 211.3、FID 34.63，CLIP 22.63 也最好。作者说超过 4 帧后在 Ego4D 上不再提升。
- **数据消融**（Table 8，1024 个测试样本）：只用 internet 数据 FVD 219.62；去掉 internet 数据 FVD 307.80；全量 211.30。去掉 internet 数据后 FVD 显著变差，作者据此强调 internet 数据的重要性（只消融了 internet 这一类数据，不能据此给各类数据的贡献排序）。附录 Figure 13/14 定性表明：只用 SSv2 或只用通用 internet 数据时，"uncover bottle" 模拟不出来；只用 RT-1 数据时，长程交互约一半失败。
- **模型规模**（Table 9）：500M / 1.6B / 5.6B 的 FVD 分别为 277.85 / 224.61 / 211.30。作者自己承认提升趋于饱和，"slightly disappointing"。另外，Table 1 的 "4 recent" 和 Table 8/9 的全量模型 FVD 都是 211.3（CLIP 都是 22.63），应当是同一个默认模型；但 Table 1 写的是 Ego4D validation split，Table 8/9 写的是 held-out test split，论文没说明两者的关系。
- **长程 VLM**（Table 2，5 个 seed，mean ± standard error）：VLM-BC 的 RDG(moved) 为 0.11±0.13，RDG(all) 为 0.07±0.11；Simulator-Hindsight 两项都是 0.34±0.13。作者称提高了 3–4 倍。评测时用 Language Table 仿真器的真实状态算距离。Figure 7/10 是真机的定性结果。
- **RL**（Table 3，48 个任务）：VLA-BC 总成功率 0.58，Simulator-RL 为 0.81；pointing 类任务从 0.12 提升到 0.71。但这个成功率是**在 UniSim 生成的视频里定性判定**的（原文 "assessed qualitatively using video rollouts in the simulator"）。真机 zero-shot 只有定性图：Figure 8 底行和 Figure 12（6 个任务的真机首末帧 + learned reward model 输出的 steps-to-completion 曲线，step 0 到 70）。Figure 8 上两行做了 sanity check：反复施加低层控制（例如 $\Delta x=0.05,\Delta y=0.05$）20–30 步，让末端左右、上下和沿对角线移动，末端运动和碰撞物理都能模拟出来。
- **Captioning**（Table 4，CIDEr）：在 ActivityNet 上，不微调 15.2，只用合成数据 46.23（约为真数据 54.90 的 84%）。迁移到 MSR-VTT / VATEX / SMIT 时，合成数据分别为 27.63 / 40.03 / 20.58，都高于真数据的 24.88 / 36.01 / 16.91。
- **哪个组件真正起作用**：最近多帧历史、internet 数据加上 action-rich 的操作数据（联合训练）有量化或定性证据。RL 相对 BC 的增益在 demo 稀少的 pointing 任务上最明显。

## 局限与论文没说的

**作者承认的**（`source/conclusion.tex`）：(1) 动作和场景不符时会 hallucination，比如对桌面机器人下 "wash hands"，桌子会变成水槽；(2) 记忆有限，抽屉里的苹果可能消失；(3) 域外泛化有限，训练主要覆盖 4 种机器人形态；(4) 只模拟视觉，动作不引起画面变化的情况（比如抓握力大小）无法建模。

**我读出来的**：
- **sim-to-real 证据偏弱**：真机结果全是定性图，没有真机成功率，也没有试验次数。Table 3 的训练和评测在同一个 learned simulator 里完成，而且由人看视频判断，无法排除 policy 利用模型误差（model exploitation）。
- **Table 2 的执行链路很长**（VLM → UniSim → IDM → 仿真器）。误差条和均值同量级（0.11±0.13），只有 5 个 seed，baseline 的执行链路也没写清楚。
- **指标偏画质**：FID、FVD、IS、CLIP 衡量的是视觉质量，不是动作可控性或动力学保真度。除了 FVD/FID 这类分布级指标，论文没有拿 action-conditioned 预测和真实轨迹做逐条的定量对比（如末端位置或物体位置误差），也没有和 physics simulator 或 Dreamer 类 latent world model 比较。
- **低层控制只验证了 Language Table 的 2D 末端位移**。高维关节控制、接触力、灵巧操作都没测。
- **reward 是 steps-to-completion 代理**，由 Stage 1 同一模型的冻结副本给出，存在被 hack 的风险。Figure 12 里真机执行时 reward model 输出的 steps-to-completion 曲线明显不单调。
- **算力和速度**：训练需要 512 TPU-v3 × 20 天；每段 256 步采样再加两级超分，但论文没报告 RL rollout 吞吐、RL 总步数和 wall-clock。
- **数据不可完全复现**：包含未公开的 Miscellaneous robot / human / internet video（分别为 133k / 50k / 13M，Table 5）。Habitat 的样本数在 Figure 6 写 700，在 Table 5 写 710。
- 作者引用 dichotomy of control，说 policy 可以学会只控制可控部分，但没有实验验证。

## 在脉络中的位置

- **前驱**：[World Models](../../01_world_model/world-models/notes.md) 和 [Dreamer](../../01_world_model/dreamer/notes.md)（在学到的模型里 imagination 训策略；UniSim 把它从单环境 latent 空间推到多域像素空间。注意 UniSim 原文引用的是 DreamerV2、[DreamerV3](../../01_world_model/dreamerv3/notes.md) 和 IRIS，没有引用 World Models 与 Dreamer v1）、Dyna、Video Diffusion Models / Imagen Video（级联 3D U-Net）、classifier-free guidance、UniPi（视频生成当 policy + IDM）、HER（hindsight relabeling）、RT-1/RT-2、PaLM-E、PaLI、Language Table。
- **后继 / 影响**：[Genie](../../01_world_model/genie/notes.md) 拿 UniSim 做对照，强调自己从无标注视频学 latent action，不需要文本或动作标签；[Ctrl-World](../ctrl-world/notes.md) 把 UniSim 列为它所基于的早期工作之一；[World-Gymnast](../world-gymnast/notes.md) 引用它作为"用生成模型模拟 agent–环境交互"的代表；[SimDist](../simdist/notes.md) 引用它作为"internet-scale 视频模型 + 仿真数据"的例子；[WMPO](../wmpo/notes.md) 同样走"在 world model 里对 VLA 做 RL"的路线（但 WMPO 的参考文献里没有 UniSim）；[Cosmos](../../01_world_model/cosmos-wfm/notes.md)、[DreamZero](../../01_world_model/dreamzero/notes.md)、[V-JEPA 2](../../01_world_model/vjepa2/notes.md) 的参考文献也包含它。同属像素空间扩散 world model 的还有 [DIAMOND](../../01_world_model/diamond/notes.md)（Atari），更大规模"在 world model 里训 agent"的是 [Dreamer 4](../../01_world_model/dreamer4/notes.md)。
- **与本项目其他主题的联系**：
  - **real2sim2real**：UniSim 是纯数据驱动的 learned simulator，从真实初始帧出发 rollout，和 [RialTo](../../03_real2sim2real/rialto/notes.md)、[ACDC](../../03_real2sim2real/acdc-digital-cousins/notes.md)、[SplatSim](../../03_real2sim2real/splatsim/notes.md) 那种 3D reconstruction / digital twin + physics engine 的路线相对。它和 SplatSim 都用"视觉逼真"来缩小 sim-to-real gap，但 UniSim 没有显式物理。用 world model 做策略评测，可以对照 [SIMPLER](../../03_real2sim2real/simpler/notes.md)。
  - **RL**：在 world model 里对 VLA 跑 REINFORCE，是 [SimpleVLA-RL](../../04_rl/simplevla-rl/notes.md) 和 [π_RL](../../04_rl/pi-rl/notes.md)（这两篇在物理仿真器里做 RL）在 world-model 侧的先例。learned progress reward 的做法可以和 [HIL-SERL](../../04_rl/hil-serl/notes.md) 的 reward classifier 对照。
  - **TTT**：UniSim 部署时不做任何适应。它承认的"新机器人形态泛化差"和"长程记忆缺失"，正好对应 TTT 的两个方向：用部署时的少量真实交互做 self-supervised 适应（类似 [PAD](../pad/notes.md)、[TTT](../../02_test_time_training/ttt-self-supervised/notes.md)），以及用 [TTT layers](../../02_test_time_training/ttt-layers/notes.md)、[Titans](../../02_test_time_training/titans/notes.md) 这类 test-time memory 替代固定的 4 帧窗口。

## 前置知识

1. DDPM（$\epsilon$-prediction、反向采样）与 classifier-free guidance
2. 视频扩散：3D U-Net、时空交替 attention、级联超分（Imagen Video）
3. World model / model-based RL（Dyna、Dreamer 的 imagination），以及 POMDP 下的历史条件
4. REINFORCE 策略梯度；基于势函数差的 reward shaping
5. goal-conditioned policy 与 hindsight relabeling（HER）
6. VLM/VLA（PaLM-E、PaLI、RT-2 的动作 token 化）与 inverse dynamics model（UniPi）
7. 评测指标 FID、FVD、IS、CLIP score、CIDEr

## 复现可行性

- **代码 / 权重**：论文没有给出代码或权重链接，只给了视频 demo 页。
- **算力**（论文数字）：UniSim 本体需要 512 TPU-v3 × 20 天（`source/method.tex`）；PaLM-E 12B 微调需要 64 TPUv3 × 1 天（附录 D.1）；captioning 用的是 PaLI-X 55B；RL 阶段的算力**论文没报告**。完整复现不现实。
- **数据**：Table 5 列出的大部分是公开数据（Language Table sim 160k / real 440k、RT-1 70k、Bridge 2k、Ego4D 3.5M、SSv2 160k、EPIC 25k、R2R 3.5M、LAION-400M），但 Miscellaneous 部分未公开。Language Table 的环境和数据可以从 https://github.com/google-research/language-table 获得（该链接原文未报告）。
- **最小可复现实验**（只用 Language Table sim）：
  1. 训一个小型 action-conditioned 视频扩散模型：4 帧历史 → 16 帧未来，条件为 T5 指令嵌入 + 离散化的 $(\Delta x,\Delta y)$ 序列，分辨率和论文 base 模型同为 24×40。先复现 Table 1 式的"1 帧 vs 4 帧历史"FVD 消融。
  2. 复现 Figure 8 上两行的 sanity check，并**量化**：同一初始状态、同一动作序列下，比较 UniSim rollout 与仿真器 ground truth 的末端位置误差（训一个像素→状态的回归器来读出位置）。
  3. 训 BC 策略和剩余步数回归器 $d(o,g)$，在 learned simulator 里跑 REINFORCE，最后**在 Language Table 物理仿真器里评测**（把仿真器当作"真实世界"）。这样能补上论文"在自己的 world model 里评测"的缺口，直接测 model exploitation。
  算力方面，我的个人估计（不是论文数字）是单节点多卡、天级别。
- **预计难点**：扩散采样放在 RL 循环里太慢（论文用 256 步 + 两级超分，小模型需要少步采样或蒸馏）；自回归 rollout 漂移；reward hacking；IDM 的训练质量。

## 自测问题

1. UniSim 为什么把 simulator 定义成不含 reward 的 $p(o_t\mid h_{t-1},a_{t-1})$？在 RL 和 hindsight 两个应用里，reward 或监督信号分别从哪来？
2. CFG 公式里只对 action 做 guidance，history 始终拼在输入里。增大 $\eta$ 的 trade-off 是什么？纯文本生成为什么要用白图占位并加大 guidance？
3. 结合 Table 1 与"抽屉里的苹果消失"这个局限：固定 4 帧历史窗口在 POMDP 意义下丢掉了什么？为什么历史里不放过去的动作也可能出问题？
4. 在 $\gamma=1$ 时，把 $r=-[d(o_{t+1},g)-d(o_t,g)]\mathcal C$ 沿轨迹求和会得到什么？这种差分形式的 reward 比直接用 $-d$ 好在哪？它可能被怎样 hack？
5. Table 3 和 Figure 8/12 分别证明了什么、没证明什么？要严谨地支持"在 UniSim 里训、zero-shot 上真机"，你会补哪些实验和指标？
