# SimpleVLA-RL: Scaling VLA Training via Reinforcement Learning

> arXiv [2509.09674](https://arxiv.org/abs/2509.09674) · ICLR 2026 poster（据 OpenReview；论文 PDF 与官方 README 均未标注会议。arXiv 上只有 v1，即 2025-09-12 的技术报告，本笔记依据该版本），2025 · 主题：04_rl · 层级：frontier
> 代码：https://github.com/PRIME-RL/SimpleVLA-RL（已访问确认：是官方 repo，基于 veRL，含 LIBERO 和 RoboTwin2.0 的训练脚本及 SFT checkpoint） · 笔记依据：`source/*.tex`（数字/公式）+ `paper.pdf`（图）

## 一句话

本文把 DeepSeek-R1 的"GRPO + 0/1 结果奖励"直接用到动作 token 化的 VLA（OpenVLA-OFT）上。做法是在 veRL 里把 LLM 的一次性生成改成与并行仿真环境的多步闭环交互，再加上从 LLM RL（DAPO、POLARIS 等）借来的三个探索技巧。这样每个任务只用 1 条示范冷启动，LIBERO-Long 就从 17.3% 升到 91.7%。

## 问题设定

- **目标**：在已经 SFT 过的 VLA 上继续做 online RL 后训练，缓解 SFT 的两个瓶颈：示范数据稀缺，以及分布外泛化差（§1）。
- **输入/输出**（`2_preliminary.tex` Eq.4–5）：状态是 $s_t=(o^{vis}_t,o^{prop}_t,l_{task})$。本文实现只用单视角图像、语言指令和本体状态（LIBERO 上连本体状态也不用），不用腕部相机。输出是长度为 $k$ 的 action chunk，每一维都离散成 action token，模型一共配置 256 个 action token（§4.1）。
- **假设**：
  1. 有能大规模并行、可重置的仿真器：LIBERO、RoboTwin 1.0/2.0。
  2. 仿真器能判定任务是否成功（`is_successful`）；论文不另设计任何过程奖励。
  3. SFT 后的初始策略在该任务上成功率必须大于 0（§6.2）。
- **之前的方法卡在哪**：
  - 传统机器人 RL 靠手工设计的过程奖励（Eq.6 中的 $\phi_i$），很难扩展。
  - VLA 的 rollout 要和环境交互多轮，比 LLM 慢，也更贵（§1）。
  - 很多 VLA 用扩散去噪或 MLP 回归来解码动作，拿不到一个既能采样、又能算 log-prob 的动作分布（§3.1）。

## 核心方法

整体流程见 Figure 2：从同一个 $s_0$ 采样 $G$ 条轨迹，每条轨迹得到 0/1 结果奖励，在组内归一化得到 advantage，再对所有 action token 做 clipped policy gradient。

**1. 可采样的动作分布（`3_method.tex` §3.1、§4.1 Backbones）。** VLA 解码动作有三种方式：token 分布、扩散去噪、MLP 确定性回归。论文认为 token 化与 PPO 类算法最兼容，所以选它。

具体改动是：用 LLaMA2 的输出头给出 action token 上的 softmax，训练用 cross-entropy；官方 OFT 用的是 MLP 输出连续动作、L1 回归。因为改了头，官方 checkpoint 不能直接用，作者按官方的数据和超参从头重做了 SFT。

保留了 OFT 的 parallel decoding 和 action chunking：一次前向就输出整个 chunk 的所有 token。据此可推断 $\log\pi(\text{chunk}\mid s)=\sum_j\log\pi(\text{token}_j\mid s)$，每个 token 各有一个 ratio（这是笔记的推断，论文没有写出这个式子；论文只说用"action token probabilities"算 GRPO loss，§3 开头）。探索方式和 LLM 相同，就是温度采样 $\mathrm{softmax}(f_\theta(s)/T)$（Eq.2 的 VLA 版本）。

**2. 交互式 rollout（Listing 1）。** veRL 原本一次调用 `policy.generate` 就完成生成。本文改成下面的循环：

1. 把 batch 复制 $G$ 份。
2. 用进程池并行初始化环境。
3. 策略对所有仍在运行的环境采样一个 action chunk，环境并行执行 step。
4. 把已经 done 的环境剔除，回到第 3 步。

循环在全部环境完成或达到最大步数时结束。最大步数在 LIBERO 上是 512，在 RoboTwin 上按任务取 200、400 或 800。训练、推理和渲染整合在同一个框架里。

**3. 结果奖励（Eq.9）。**
$$R(a_{i,t}\mid s_{i,t})=\begin{cases}1,&\text{traj}_i\ \text{成功}\\0,&\text{otherwise}\end{cases}$$
轨迹级的奖励原样分给这条轨迹里的每个 action token。没有折扣，也没有逐步的 credit assignment。

**4. 去掉 KL 的 GRPO（Eq.11–12；原版 GRPO 见 Eq.7–8）。**
$$\mathcal J(\theta)=\mathbb E\Big[\tfrac1G\sum_{i=1}^G\tfrac1{|a_i|}\sum_{t=1}^{|a_i|}\min\big(r_{i,t}\hat A_i,\ \mathrm{clip}(r_{i,t},1-\varepsilon_{low},1+\varepsilon_{high})\hat A_i\big)\Big]$$
$$r_{i,t}=\frac{\pi_\theta(a_{i,t}\mid s_{i,t})}{\pi_{\theta_{old}}(a_{i,t}\mid s_{i,t})},\qquad \hat A_i=\frac{R_i-\mathrm{mean}(\{R\})}{\mathrm{std}(\{R\})},\qquad \text{s.t. }0<\#\{\text{成功}\}<G$$

- 式中 $t$ 遍历轨迹 $i$ 的全部 action token（论文记号 $|a_i|$ 没有明确说是 token 数还是时间步数；这里按 §3.2"奖励均匀传到每个 action token"来理解）。
- 和 Eq.7 相比，这里删掉了 $-\beta D_{KL}(\pi_\theta\|\pi_{ref})$。论文给的理由有两条：不再需要 reference model，省显存、训练更快；也不再把策略拴在 SFT 分布附近。

**白板推导**（这是我的推导，论文里没有）：组内有 $k$ 条成功时，记 $p=k/G$。用总体标准差 $\sqrt{p(1-p)}$ 算，成功轨迹 $\hat A=\sqrt{(1-p)/p}$，失败轨迹 $\hat A=-\sqrt{p/(1-p)}$。

- $G=8$、$k=1$ 时，成功轨迹约 +2.65，失败轨迹约 −0.38。成功越少见，那一条成功轨迹被放大得越多。
- $k=0$ 或 $k=G$ 时 std 为 0，advantage 退化为 0，梯度消失。这就是 dynamic sampling 要处理的问题。

**5. 三个探索增强（§3.3，都是从 LLM RL 借来的）。**
- **Dynamic sampling**（DAPO、PRIME）：丢掉全成功或全失败的组，一直采样到 batch 里只剩结果有好有坏的组（Eq.10）。
- **Clip-higher**（DAPO）：clip 区间从 [0.8, 1.2] 改成 [0.8, 1.28]，让低概率 token 的概率可以涨得更多。
- **提高 rollout 温度**：$T$ 从 1.0 调到 1.6。

论文的动机是：示范轨迹很同质，VLA 容易收敛到很窄的几种解法；成功率越低，越需要探索。

**训练与推理的区别**：训练时 rollout 用 $T=1.6$ 随机采样；评测时用 greedy 解码，每个 benchmark 测 3 次（§4.1）。

**超参**（§4.1）：
- 优化：全参数训练，lr $5\times10^{-6}$，train batch 64，mini-batch 128。
- 采样：每个输入采 8 条（$G=8$），$T=1.6$。
- 裁剪：$\varepsilon_{low}=0.2$，$\varepsilon_{high}=0.28$。
- action chunk：LIBERO 为 8，RoboTwin 为 25。

## 关键实验与证据

- **LIBERO（Table 2）**：每个 suite 先用 500 条示范做 SFT，再在 500 个仿真场景上做 RL。自家的 OFT 从 91.0 升到 99.1，其中 Spatial 99.4、Object 99.1、Goal 99.2、Long 98.5，Long 提升了 12.0。这个结果超过 π0（94.2）和 UniVLA（95.2）。
- **RoboTwin1.0（Table 3）**：4 个双臂任务，用 50 条示范 SFT、100 个场景 RL，从 39.8 升到 70.4；DP3 是 58.1。
- **RoboTwin2.0（Table 1、4）**：
  - 设置：12 个任务按 horizon 分成 4 档（112–637 步），用 1000 条示范 SFT、1000 个场景 RL。
  - 平均结果：从 38.3 升到 68.8，对比 π0 的 49.2 和 RDT 的 33.3。
  - 分档提升：short +43.6，medium +25.4，long/extra-long +22.4。
- **数据稀缺（Table 5，本文的核心卖点）**：
  - 每个任务只给 1 条示范做 SFT（One-Trajectory SFT，每个 suite 共 10 条），平均成功率 48.9，RL 后到 96.9。
  - 这比用 500 条示范 SFT 的 91.0 还高。LIBERO-Long 从 17.3 升到 91.7。
  - 与 Full-SFT 再加 RL（99.1）相比只差 2.2。
- **探索技巧消融（Figure 3，LIBERO-Long，约 300 个 RL step）**：图上标注的收益是 dynamic sampling 约 15%，clip-higher 约 10%，提高温度约 15%。
- **泛化（Figure 4，§5.2）**：
  - 设置：每个 suite 用 9 个任务训练、留出 1 个任务，每个 suite 画了 3 个不同的留出任务。两边都从 One-Traj SFT 出发：SFT 组在 450 条示范上继续训练，RL 组在同一起点上做 RL。
  - 训练任务上两者都超过 90%。
  - unseen 任务上，SFT 大多掉到 0，LIBERO-Goal 的 3 个 unseen 任务一开始就归零；RL 则普遍上升，例如 Spatial Unseen 1 从 43.3 升到 71.8，Object Unseen 2 提升了 36.5。
- **Sim2real（Table 6）**：只用仿真数据，即 1000 条仿真示范 SFT 加 1000 个仿真场景 RL，部署到由两条 AgileX Piper 机械臂组成的真机上，在未见过背景的干净桌面上评测，每个任务试 50 次。平均从 17.5 升到 38.5，RDT 是 23.5；Pick Bottle 从 0 升到 14.0。
- **Pushcut（Figure 5，§6.1）**：在 move can pot 和 place a2b 两个任务上，示范全部是"抓取–移动–放置"，RL 之后策略学会了直接把物体推过去。作者把它类比为 R1 的 "aha moment"。这部分只有定性展示，没有统计出现频率。
- **失败条件（Table 7，§6.2，5 个 RoboTwin2.0 任务）**：
  - 0 条示范 SFT：RL 前后都是 0，因为所有轨迹奖励为 0，没有梯度。
  - 100 条示范：从 7.3 升到 25.4。
  - 1000 条示范：从 28.2 升到 50.4。
  - Pick dual bottles 在 100 条示范时只从 1.2 升到 4.3，说明存在一个初始能力阈值。

## 局限与论文没说的

- **作者承认的**：RL 的效果强烈依赖 SFT 先验。初始成功率接近 0 时，只靠结果奖励几乎学不动（Table 7）。
- **只验证了一种 backbone 和一种动作参数化**，即 token 化的改版 OFT。扩散或 flow 类 VLA（如 π0）拿不到 token 概率，本文没有处理，这正是 [π_RL](../pi-rl/notes.md) 的出发点。
  - 另外，π_RL 的 `main.tex` 说 SimpleVLA-RL 对连续动作用的是"Gaussian 分布 + 方差预测头"，和本文 v1 的 token 化描述不一致。arXiv 上只有 v1，所以这可能对应代码实现或会议版本，读代码时需要核对。
- **基线口径**：
  - 所有 Δ 都是相对"自家改版、重新 SFT 的 OFT"算的，不是相对官方 checkpoint。
  - π0、UniVLA 等基线的数字是否重新跑过、输入视角是否一致，论文没有说明。
  - 评测测了 3 次，但没有报告方差；Figure 3 的曲线看起来是单次运行。
- **训练和测试是同一批任务**：主结果只是换了初始场景。跨任务泛化只在 LIBERO 的"9 训 1 留"设定下测过，没有测新环境、新纹理或新 embodiment。泛化实验的 SFT 对照只是在 450 条示范上继续 SFT，没有和加正则的 SFT 或 SFT/RL 混合训练比较。
- **奖励依赖仿真里的成功判定**：真实世界没有 `is_successful`，所以本文没有做任何真机 RL。
- **sim2real 的依据有限**：迁移完全依赖 RoboTwin2.0 自带的 domain randomization，论文没有做"去掉 DR 后还能不能迁移"的消融。真机只有 4 个任务、每个 50 次，平均绝对成功率也只有 38.5%。
- **Pushcut 的另一面**：结果奖励只看终态。对易碎物体或要求稳定放置的任务，用推代替抓可能不可接受；这属于 reward 不约束过程带来的 specification gaming 风险，论文只讲了好的一面。
- **系统贡献没有量化**：并行多环境渲染和"训练–推理–渲染"一体化是论文主打的工程贡献，但没有给出吞吐、wall-clock 或 GPU·小时。
- **正文和表格对不上**（以表为准）：
  - 引言写 LIBERO-Long 从 17.1 到 91.7，Table 5 是 17.3。
  - §5.3 正文列出的真机任务包括 Handover Block，Table 6 中却是 Place Empty Cup。
  - 正文说 Stack Bowls "from 32% to 70%"，Table 6 是 38.0→70.0，32 其实是 Δ。
  - 正文说 Pick Bottle 达到 15%，表中是 14.0。
  - Table 7 表头写的 "Place A2B Lift" 应为 Left。
- **论文没交代的细节**：
  - importance ratio 里的 $\pi_{\theta_{old}}$ 是否按 $T=1.6$ 计算。
  - mini-batch 128 的单位是什么。
  - RoboTwin 双臂的动作维度，以及每个 chunk 的 token 数。

## 在脉络中的位置

- **前驱**：
  - 算法：GRPO（DeepSeekMath）与 [DeepSeek-R1](../deepseek-r1/notes.md) 的"rule-based 结果奖励 + online RL"；[PPO](../ppo/notes.md) 的 clipped surrogate；DAPO（clip-higher、dynamic sampling、去掉 KL）和 POLARIS 的温度调节。DAPO 和 POLARIS 不在本项目中。
  - 框架与 backbone：veRL；OpenVLA 和 OpenVLA-OFT。
  - benchmark：LIBERO、RoboTwin 1.0/2.0。
  - 同期与相关工作（§7）：RIPT-VLA（用 RLOO，论文明确称其为同期工作），以及 VLA-RL（用 PPO）、RLinf、TGRPO、RFTF。
- **后继 / 影响**：
  - [π_RL](../pi-rl/notes.md)：把 VLA 在线 RL 扩展到 flow 类的 π0 和 π0.5，并对比了 PPO 与 GRPO；文中把本文当作 GRPO 用于 VLA 的代表。
  - [WMPO](../../05_intersections/wmpo/notes.md)：把仿真器换成像素级 world model，在 imagination 中对 VLA 做 GRPO；附录注明其余设置 "inherited from" 本文。
  - [World-Gymnast](../../05_intersections/world-gymnast/notes.md)：在 video world model 中用 VLM 打分作奖励来训练 OFT，并明确沿用了本文的四个技巧（去 KL、dynamic sampling、clip-higher、提高温度）。
  - [π\*0.6](../pi-star-06/notes.md)：把本文这类在仿真中直接套用 PPO 类算法的做法，归为"难以高效扩展到真实世界 RL"的一类；它自己走真实世界 offline RL 加 advantage conditioning 的路线。
- **与本项目其他主题的联系**：
  - *World model*：本文最大的外部依赖，是一个可并行、可重置、能判定成功的仿真器。WMPO、World-Gymnast 和 [Ctrl-World](../../05_intersections/ctrl-world/notes.md) 用 learned world model 替代它，代价是成功信号要另找来源：WMPO 用学到的二值 reward model，World-Gymnast 用 VLM 打分，Ctrl-World 则由人工判定 imagination 轨迹是否成功（且它是筛选成功轨迹做 SFT，不是 RL）。[Dreamer 4](../../01_world_model/dreamer4/notes.md) 也是"先 BC 预训练，再在 world model 里做 RL"的结构。
  - *TTT*：[TTRL](../../02_test_time_training/ttrl/notes.md) 与本文作者高度重叠，本文 §7 也引用了它。TTRL 在无标签的测试数据上用 majority voting 当伪奖励来跑 GRPO。
    - 把本文的 RL 搬到部署任务上做，就成了 test-time RL，但前提是能拿到成功信号。
    - [PAD](../../05_intersections/pad/notes.md) 走的是另一条路：部署时做无奖励的自监督适应。
  - *Real2sim2real*：
    - 本文的 sim2real 全靠 [domain randomization](../../03_real2sim2real/domain-randomization/notes.md)，没有做 real2sim 校准。
    - 可以和 [RialTo](../../03_real2sim2real/rialto/notes.md)、[ACDC](../../03_real2sim2real/acdc-digital-cousins/notes.md)、[SplatSim](../../03_real2sim2real/splatsim/notes.md) 这类"先重建真实场景、再在 digital twin 里做 RL"的方法组合。
    - 评测方面可以对照 [SIMPLER](../../03_real2sim2real/simpler/notes.md)。
    - [Reality gap survey](../../03_real2sim2real/survey-reality-gap/notes.md) 列出的各类 gap，本文同样都会遇到。
  - *RL*：本文属于"大模型 + 稀疏结果奖励 + critic-free"这一支。可以和 [DPPO](../dppo/notes.md)（用 PPO 训练扩散策略）、[HIL-SERL](../hil-serl/notes.md)（真机 off-policy RL 加人工干预）对照着看。

## 前置知识

1. Policy gradient 与 importance ratio，以及 [PPO](../ppo/notes.md) 的 clipped objective 和 trust region 的直觉。
2. GRPO 用组内归一化的回报代替 value function；critic-free 方法在组内奖励全部相同时梯度为 0。
3. DAPO 的 clip-higher 与 dynamic sampling；温度、熵和探索之间的关系。
4. VLA 基础：OpenVLA 的动作离散化（action token），OFT 的 action chunking 和 parallel decoding。
5. [DeepSeek-R1](../deepseek-r1/notes.md) 的 rule-based reward，以及 KL 正则在 RLHF 里起的作用。
6. LIBERO 和 RoboTwin 的任务结构与成功率评测；sim-to-real 与 domain randomization。

## 复现可行性

- **代码**：官方 repo 已开源，基于 veRL（README 未标注会议；ICLR 2026 来自 OpenReview）。
  - 训练脚本：`examples/run_openvla_oft_rl_libero.sh`、`run_openvla_oft_rl_twin2.sh`。
  - HuggingFace 上放了 SFT checkpoint：libero-10 的 traj1 和 trajall，libero-goal/object/spatial 的 traj1，以及 RoboTwin2.0 的 traj1000。所以不需要自己做冷启动 SFT。
- **算力**：论文只写了用 8×A800 80GB 做全参数训练（§4.1），**没有报告训练时长或 GPU·小时**。README 给出的测试配置是单机 8×A800，或双机 16×A800。
- **数据与仿真**：LIBERO（robosuite/MuJoCo）或 RoboTwin 2.0。rollout 需要大量并行的环境进程，CPU 和渲染容易成为瓶颈。
- **最小可复现实验**：用官方的 libero-10（即 LIBERO-Long）traj1 SFT checkpoint，论文中它的成功率是 17.3%。
  1. 按 §4.1 的超参做 RL：$G=8$、$T=1.6$、clip [0.8, 1.28]、不加 KL、开 dynamic sampling。跑约 250–300 个 step（参照 Figure 1/3 的横轴），用 greedy 解码在 10 个任务 × 50 个场景上评测，目标是接近 Table 5 的 91.7。
  2. 做消融：把温度改回 $T=1.0$，或关掉 dynamic sampling，看能否复现 Figure 3 中约 15 个点的差距。
  3. 每 20 个 step 记录一次两个量：结果有好有坏的组所占比例，以及平均 token 熵。用它们检验探索技巧到底通过什么起作用。
- **预计难点**：
  - veRL 要和大量仿真进程一起调度，显存和 CPU 都吃紧。
  - 成功率很高或很低时，dynamic sampling 要反复重采，吞吐会不稳定。
  - 改版 OFT 与官方权重不兼容，不能混用。
  - 真机 sim2real 部分需要 AgileX Piper 双臂，基本无法复现。

## 自测问题

1. $G=8$、二值奖励时，组内 1 条成功和 7 条成功两种情况下，成功轨迹的 advantage 各是多少？这对低成功率任务的学习信号意味着什么？
2. 为什么必须把 OFT 的 MLP + L1 回归换成 token 头 + CE？如果是扩散或 flow VLA，要怎么改才能套用同一个目标（提示：[DPPO](../dppo/notes.md)、[π_RL](../pi-rl/notes.md)）？
3. 去掉 KL，和把 $\varepsilon_{high}$ 调到 0.28，分别放松了对策略更新的哪一种约束？各自可能带来什么风险？
4. Table 7 中 0 条示范 SFT 的模型为什么 RL 后仍停在 0？不加示范的话，你会用什么方法打破这个僵局？这个方法和 pushcut 所依赖的"只看结果"的奖励之间有什么张力？
5. 泛化实验里 SFT 在 unseen 任务上归零，而 RL 没有。请从"在示范上最小化 CE"和"在自身采样分布上做小步 clipped 更新"两个角度解释，并指出这个实验设计没有排除哪些混淆因素。
