# World-Gymnast: Training Robots with Reinforcement Learning in a World Model

> arXiv [2602.02454](https://arxiv.org/abs/2602.02454) · ICLR 2026 Workshop on World Models（workshop 论文，已在 OpenReview 核实；未见主会或期刊录用。arXiv 预印本用 ICML 2026 模板的 preprint 选项排版，未标注录用），2026 · 主题：05_intersections · 层级：frontier
> 代码：https://github.com/world-gymnast/world-gymnast（论文只给了项目页 world-gymnast.github.io，项目页链到这个 repo；2026-09-27 访问确认存在） · 笔记依据：`source/*.tex`（数字/公式）+ `paper.pdf`（图）

## 一句话

把 OpenVLA-OFT 放进一个在真实机器人视频上预训练的动作条件视频 world model（WorldGym）里 rollout，由 GPT-4o 对生成的视频给 0/1 任务完成 reward，再用 GRPO 做 RL 微调。关键想法是：world model 只需要"一张初始帧 + 一句指令"就能搭出一个训练环境，所以换帧、改图、改指令、用测试帧做 test-time training 都只是换一下输入；另外还可以用真机数据回灌 world model（Dyna，这一步需要真机采数据）。在 AutoEval 真机上，这样训出的策略成功率最高是 SFT 的 18 倍（72% vs 4%），最高约为 SIMPLER 里做 RL 的 2 倍（摘要原话 "as much as 18x / 2x"）。

## 问题设定

- **问题**：真机交互太贵。SFT 受限于示范数据的覆盖面，缺少 recovery 行为；软件仿真器（这里指 SIMPLER）每个新场景都要人工搭，而且有视觉上的 sim-to-real gap。作者不去单独测 world model 的物理是否可信，而是直接问端到端的问题：在 learned world model 里做 RL，真机成功率能不能超过 SFT 和仿真器 RL？
- **形式化**（`source/preliminaries.tex`）：multi-task、finite-horizon POMDP $\mathcal M=(S,A,O,G,R,T,\mathcal E,H)$。policy $\pi(\cdot\mid o_t,g)$ 只看图像和语言指令 $g$。环境每步都给 $r_t=R(s_t,g)$，但**policy 的价值只取终止状态的 reward** $R(s_H,g)$：
  $$\rho(\pi)=\mathbb E\big[R(s_H,g)\mid s_0,g\sim G,\ o_t\sim\mathcal E(s_t),\ a_t\sim\pi(o_t,g),\ s_{t+1}\sim T(s_t,a_t)\ \ \forall t\in[0,H]\big]\quad\text{(Eq.1)}$$
  model-based 版本把 $T$ 换成视频 world model $\hat T(\cdot\mid\mathbf o,\mathbf a)$，输入历史帧序列和未来动作序列；把 $R$ 换成读完整段帧序列的 VLM $\hat R$：
  $$\hat\rho(\pi)=\mathbb E\big[\hat R([o_0,\dots,o_H],g)\mid s_0,g\sim G,\ \mathbf a\sim\pi(\mathbf o,g),\ \mathbf o'\sim\hat T(\mathbf o,\mathbf a),\ \mathbf o=\mathbf o'\big]\quad\text{(Eq.2)}$$
- **输入 / 输出**：训练一个任务只需要 $(o_0, g)$，输出是微调后的 VLA。不需要真实 reward、物体 3D 资产或物理参数。
- **假设**： $o_0$ 要"足够接近" world model 的训练分布（WorldGym 在 Open X-Embodiment 上预训练）；VLM 能从生成的视频里判断任务成败；基座 policy 已有一定成功率（否则 GRPO 组内 reward 全是 0，没有方差）。

## 核心方法

**1. 三个组件（Figure 1）**
- **Policy**：OpenVLA-OFT，从 Open X-Embodiment 预训练的 OpenVLA checkpoint 出发，在 BridgeData V2 上 SFT 20k 步（Sec 4.2）。参照 SimpleVLA-RL 改了两处：关掉 proprioception 和第二路相机；动作头从默认的 L1 回归头换成 LLaMA-2 的 LM head（离散 action token），这样才有 RL 需要的 $\pi_\theta(a\mid o,g)$ 概率（Sec 4.1）。动作是 7 维（6 维末端位姿 + 二值夹爪），action chunk 长度为 5。
- **World model $\hat T$**：WorldGym，609M 参数（Appendix Table `tab:hyper`，即 PDF 的 Table 4；Sec 4.1 正文写作 "600M parameter variant"）。先用 Stable Diffusion 3 的 VAE 把 $256\times256$ RGB 帧编码到 latent（16 通道），再用 16 层 DiT（hidden 1024，16 heads，patch 2）逐帧自回归做 diffusion 生成： $v$-prediction，sigmoid noise schedule，采样 10 步。动作是 10 维，这里只用前 7 维。上下文固定为 20 帧，rollout 更长时用最近 20 帧的滑窗。每一步对"有动作条件 / 空动作"各跑一次 DiT，再做 classifier-free guidance 和 DDIM 更新（Algorithm 1）。照此推算，每生成 1 帧要做 10×2=20 次 DiT 前向。作者加了 temporal KV cache，在"20 条轨迹并行、40 帧"的设置下 rollout 时间缩短到约 1/10（Appendix A.1）。
- **Reward $\hat R$**：GPT-4o。rollout 视频按 stride 3 做时间下采样后送进去，prompt 要求写 2–3 句理由，然后输出 `Final Score: 0/1`，并提示"视频由视频预测模型生成，可能有伪影"（Appendix A.3）。评测时每条 rollout 取 5 次独立 VLM 判断做 majority vote（Appendix D.1）；Appendix A.3 另写了"每次 policy 评测在 world model 里做 5 次独立 rollout"，两处的"5"是否指同一件事，论文没交代清楚。训练时是否也投票，论文没写。

**2. Model-based GRPO**（`source/method.tex` Eq.4–6）
对每个 $(g,o_0)$，在 $\hat T$ 里 rollout $K$ 条轨迹： $a_{t,k}\sim\pi_\theta(\cdot\mid o_{t,k},g)$， $o_{t+1,k}\sim\hat T(o_{t,k},a_{t,k})$，一直到 $H$。然后 $r_k=\hat R(\tau_k,g)\in\lbrace0,1\rbrace$，组内归一化：
$$\mu=\tfrac1K\sum_k r_k,\quad \sigma=\sqrt{\tfrac{1}{K-1}\sum_k (r_k-\mu)^2},\quad \hat A_k=\frac{r_k-\mu}{\sigma+\epsilon}\quad\text{(Eq.4, 5)}$$
轨迹级 advantage 广播到每一步（ $\hat A_{t,k}=\hat A_k$），再代入非对称 clip 的 PPO 目标：
$$\mathcal J(\theta)=\mathbb E_{g,o_0\sim\mathcal D}\Big[\tfrac1K\sum_{k}\tfrac1H\sum_{t=0}^{H-1}\min\big(r_{t,k}(\theta)\hat A_k,\ \mathrm{clip}(r_{t,k}(\theta),1-\epsilon_{low},1+\epsilon_{high})\hat A_k\big)\Big],\quad r_{t,k}=\frac{\pi_\theta(a_{t,k}\mid o_{t,k},g)}{\pi_{\theta_{old}}(a_{t,k}\mid o_{t,k},g)}\quad\text{(Eq.6)}$$
沿用 SimpleVLA-RL 的四个技巧（Sec 3.1）：去掉 KL 项；dynamic sampling（丢掉组内 reward 无方差的组）；clip-higher（ $\epsilon_{high}=0.28,\epsilon_{low}=0.2$）；rollout 时用高温采样（temperature 1.6）。这里的 group 是"同一 $(g,o_0)$ 下 $K$ 条想象轨迹"，baseline 由组内均值给出，所以不需要 critic。代价是 reward 在一组内全 0 或全 1 时就没有梯度。

**3. 利用"只需一帧"衍生的训练方式**（Sec 3.2–3.4）
- **Training from any frame**：任何接近 world model 分布的帧都能当 $o_0$，训练数据不再受专家示范数量限制。
- **Novel language**：同一帧配新的指令，可以让 VLM 来提议。
- **Distractors**：用 Nano Banana 在初始帧里插入无关物体。
- **Test-time training**：部署时拿到一张新场景帧，直接在 world model 里从这张帧开始跑 RL，用算力换性能，不需要真机 rollout。
- **Dyna 式迭代**：把真机 rollout（帧 + 动作序列）拿来微调 world model，再用更新后的 world model 训练 policy，让 world model 跟上 policy 诱导出的状态分布。

**训练 vs 推理**：训练时全部在 WorldGym 内闭环（policy 前向 → world model 生成下一帧 → 结束后 VLM 打分 → GRPO 更新）。推理时 policy 直接在真机上跑，world model 和 VLM 都不参与。只有 TTT 和 Dyna 两种用法会在部署阶段再调用 world model。

## 关键实验与证据

设置（Sec 4.1）：RL 训练集是 OpenVLA 评测任务集（17 个任务，覆盖 visual/motion/physical/semantic 泛化和 language grounding），按初始帧 80/20 划分训练和评测（Appendix C.1）。rollout 最多 40 步。真机评测用 AutoEval（WidowX，2 个场景，4 个任务）：每个 policy–task 组合跑 10 次 trial，重复 5 轮来估计标准误。**AutoEval 的 4 个任务不在 World-Gymnast 的 RL 训练集中**；SIMPLER baseline 则直接在这些任务的 digital twin 上训练过（Sec 4.2）。

**Table 1 真机：SIMPLER-RL vs World-Gymnast**（两者都从同一个 SFT policy 起步）

| 任务 | SIMPLER | World-Gymnast |
|---|---|---|
| Open the drawer | 34±7% | 58±4% |
| Close the drawer | 74±5% | 62±6% |
| Eggplant → blue sink | 32±10% | 72±10% |
| Eggplant → yellow basket | 40±10% | 78±2% |

除 *close the drawer* 外，基座 SFT policy 在 SIMPLER 各任务上的完成率都很低（常常方向对但做不完），binary reward 的方差塌缩，所以 SIMPLER 这一侧改用了逐步的 partial reward：抓取、提起等每项条件给 0.1，成功给 1（Appendix B.2）。

**Table 2 真机：SFT / Iter-SFT / World-Gymnast**：open 40±10 / 30±10 / 58±4；close 62±6 / 60±8 / 62±6；blue sink 4±4 / 10±3 / 72±10；yellow basket 8±4 / 17±17 / 78±2（%）。摘要里的"18x"就是 72/4；Sec 4.2 正文说的"nearly 10-fold"是 78/8。Iter-SFT 按 Ctrl-World 的思路做：在同一个 world model 里 rollout SFT policy，rollout 步数与 RL 相同，用 VLM 筛出成功轨迹，再与 Bridge V2 按 1:1 混合做一轮 SFT。它在两个难任务上略有提升，在两个简单任务上反而下降。作者的解释是 on-policy RL 的主动探索比"筛选后模仿"更能泛化，而 Iter-SFT 更容易被 world model 幻觉和 VLM 误判带偏。

**Table 3（在 WorldGym 里评测，OpenVLA held-out split）**：SFT 58±4 → World-Gymnast 74±3 → -Distract 78±2（10% 的训练 episode 用加了干扰物的帧）/ -Language 81±1（4 条新指令，与原指令按帧 50/50 混合）/ -Scaled 81±4（额外 5 个 Bridge 任务）。Figure 2 定性展示了 SFT 在有干扰物时抓错杯子。

**Test-time training**（Sec 4.4）：只用 AutoEval 4 个场景的初始帧和指令，在 world model 里做 RL。*Close the drawer* 从 62±6% 升到 100±0%。但 policy 会过拟合到单个任务，其他任务"普遍下降"，这部分没有给具体数字。

**Dyna**（Sec 4.5）：每个任务约收集 100 条 AutoEval 真机轨迹，4 个任务合计，world model 共微调 120k 步。Figure 3 显示，同一动作序列在 WorldGym-Dyna 里的 rollout 比原 WorldGym 和 SIMPLER 更贴近真机。用它训练后，*close the drawer* 成功率到 95%（没给标准误）。

**消融现状**：论文没有消融 GRPO 的几个技巧、group size、horizon，也没有测 VLM reward 的准确率。能归因到组件的证据只有 Iter-SFT 与 RL 的对照（同一 world model、同样 rollout 预算，区别只在"模仿成功轨迹"还是"on-policy RL"），以及 Table 3 里数据扩增的对照。

## 局限与论文没说的

**作者承认的**：初始帧离 world model 训练分布太远时无法泛化；依赖预训练 VLM 判成功，VLM 幻觉会让 RL 训偏，dense reward 和防止 reward hacking 留作未来工作；TTT 会过拟合单个任务；Impact Statement 提到 policy 可能利用 world model 里幻觉出来的物理。

**我读出来的**：
- **真机证据很窄**：只有 WidowX、2 个场景、4 个任务，horizon 最多 40 步，没有可形变物体（AutoEval 的布料任务被明确排除）。Appendix C 的"Real-Robot Evaluation Dataset"写的是评测了 **3 个**任务（没有 yellow basket），和 Table 1/2 的 4 行对不上。
- **统计口径**：按"10 trial × 5 轮"= 50 次计算，成功率应是 2% 的整数倍，但 Iter-SFT 的 yellow basket 是 17±17%，Dyna 的 95% 也不是 2% 的整数倍，说明这些格的 trial 数不是 50（或数字有误）。Table 2 标题又说标准误是"每 10 次连续 rollout 分组"算的。论文没有逐格给出样本量。
- **Table 3 是在训练用的 world model 里测的**，而且也用 VLM 判分，与训练 reward 同类。这是"自己出题自己判"，一旦出现 reward hacking 反而可能被计成成功。Distract / Language / Scaled 三个变体都没有真机数字。
- **SIMPLER 对照存在混杂**：两边的 reward 设计不同（dense partial reward vs binary），训练任务集也不同。从 Table 1 和 Table 2 交叉看，SIMPLER-RL 在两个茄子任务上也比 SFT 好很多（4→32，8→40），open drawer 却比 SFT 还差（40→34）。所以结论应该是"learned WM 的 RL 优于这个 SIMPLER 设置下的 RL"，不能推广成"仿真器 RL 无效"。*Close the drawer* 上 RL 之后和 SFT 完全一样（62±6），也就是说，在 world model 里做 RL 对这个任务的真机表现没有任何帮助。
- **TTT 和 Dyna 实际上用到了测试环境的信息**：TTT 用的是 AutoEval 的真实测试帧；Dyna 的真机数据就在 AutoEval 的 4 个评测任务上采集。它们与"held-out 泛化"不是同一个口径，更接近在目标环境上做 real-to-sim 适配。Dyna 迭代了几轮、评测的是哪个 policy（"base RL model"），论文都没说清楚。这里的"TTT"是从测试帧出发、在 world model 里用 VLM reward 做 RL 微调，与 TTT（Sun et al.，自监督辅助任务）和 Tent（熵最小化的 test-time adaptation）那类"在无标签测试输入上用无监督目标更新模型"的做法不是一回事。
- **未报告的成本**：GPT-4o 调用次数和费用；单条 rollout 的绝对延迟（只给了"约 10 倍加速"）；TTT 的步数和耗时。Appendix 里 TTT 的"details"实际上没有单独的超参。
- **超参语义不清**："batch size 20"与"group size 8"是否对应每次更新 20 个 $(g,o_0)$ × 8 条 = 160 条轨迹；"40 steps"是 40 个动作还是 40 帧，与 action chunk 5 怎样对应；这些论文都没说明。
- **可能的失败模式**：VLM 被生成视频里的伪影"骗"成成功（例如物体瞬移）；world model 对 policy 新探索出的动作外推失真（compounding error）；实验只覆盖了能给出离散 token 概率的 VLA（作者在 related work 里称方法原则上可用于任何 policy），flow/diffusion 类 VLA 没有测。

## 在脉络中的位置

- **前驱**：WorldGym（arXiv 2506.00613，同组的 world model 策略评测工作，不在本项目里），提供本文的 world model 和评测协议；[UniSim](../unisim/notes.md)（intro 把它列为"从真机数据学出、能近似真机执行结果"的 world model 之一，related work 把它作为"用生成模型模拟 agent–环境交互"的代表）；Dyna（Sutton 1991）；GRPO（DeepSeekMath，参见 [DeepSeek-R1](../../04_rl/deepseek-r1/notes.md)）和 [PPO](../../04_rl/ppo/notes.md) 的 clip 目标；[SimpleVLA-RL](../../04_rl/simplevla-rl/notes.md)（沿用了它的四个 GRPO 技巧和对 OpenVLA-OFT 的两处改造）；[Ctrl-World](../ctrl-world/notes.md)（Iter-SFT baseline 的来源）；[SIMPLER](../../03_real2sim2real/simpler/notes.md)（仿真器 baseline）。
- **后继 / 影响**：本项目里只有 [World Model for Robot Learning 综述](../../01_world_model/survey-wm-robot-learning/notes.md) Sec 4.1 提到它（那篇笔记指出该处引用键错指向了 WorldGym）。最接近的平行工作是 [WMPO](../wmpo/notes.md)。作者自述的区别是：本文强调用开源 VLA + WorldGym + AutoEval 做真机评测，并探索任意帧训练、TTT、Dyna 这些用法。
- **与本项目其他主题的联系**：
  - *World model*：它把 [Dreamer](../../01_world_model/dreamer/notes.md) / [DreamerV3](../../01_world_model/dreamerv3/notes.md) 的"在 world model 里训 policy"从 latent imagination 换成了像素空间的 diffusion 视频生成，与 [DIAMOND](../../01_world_model/diamond/notes.md) 同属一路，但 world model 是在跨机器人的真实数据上预训练的，reward 由外部 VLM 给出，而不是作为 world model 的一个 head 一起学。rollout 成本问题可以对照 [Dreamer 4](../../01_world_model/dreamer4/notes.md) 和 [Cosmos](../../01_world_model/cosmos-wfm/notes.md)。
  - *TTT*：Sec 3.3 的 TTT 是"对单个测试实例做 RL"，思路上接近 [TTT-Discover](../../02_test_time_training/ttt-discover/notes.md)（单个问题上的 test-time RL，也接受过拟合）；VLM 多数投票判分可以对照 [TTRL](../../02_test_time_training/ttrl/notes.md) 用 majority voting 生成伪 reward（不过本文的多数投票只在 WorldGym 评测协议里写明，训练 reward 是否投票未说明）。它与 [PAD](../pad/notes.md)、[TTT with self-supervision](../../02_test_time_training/ttt-self-supervised/notes.md) 那种部署期自监督适应机制不同：PAD 在真实环境里用自监督损失更新，本文则把 world model 当部署期的"想象环境"。
  - *Real2sim2real*：它是"学出来的 real-to-sim"，与 [SIMPLER](../../03_real2sim2real/simpler/notes.md)（system identification + green screen）、[SplatSim](../../03_real2sim2real/splatsim/notes.md)（Gaussian splatting）、[RialTo](../../03_real2sim2real/rialto/notes.md) / [ACDC](../../03_real2sim2real/acdc-digital-cousins/notes.md)（重建 digital twin / cousin）走的是不同路线。Dyna 用真机轨迹修正 world model，结构上对应 [SimOpt](../../03_real2sim2real/simopt/notes.md) 用真机 rollout 修正仿真参数分布；Nano Banana 插干扰物相当于在像素层面做视觉 [domain randomization](../../03_real2sim2real/domain-randomization/notes.md)。[SimDist](../simdist/notes.md) 则反过来，用仿真预训练 world model、在真机上只适配 dynamics。
  - *RL*：属于 outcome reward + GRPO 的 VLA 后训练，与 [SimpleVLA-RL](../../04_rl/simplevla-rl/notes.md)（reward 来自仿真器规则）、[π_RL](../../04_rl/pi-rl/notes.md)（flow-based VLA 的 RL）对照。差别在于环境和 reward 都换成了学出来的模型，reward hacking 的风险从"钻规则漏洞"变成了"钻 world model 和 VLM 的幻觉"。

## 前置知识

1. Policy gradient / PPO clip 目标，以及 GRPO 的组内归一化 advantage（为什么不需要 critic，为什么组内无方差就没有梯度）。
2. Model-based RL 与 Dyna：用 learned model 生成数据来训 policy，以及 model exploitation 和 compounding error。
3. 动作条件视频 diffusion world model：VAE latent、DiT、自回归逐帧生成、classifier-free guidance、DDIM、KV cache。
4. VLA（OpenVLA / OpenVLA-OFT）：动作 token 化、action chunking；为什么把 L1 头换成 LM head 才能做 RL。
5. VLM-as-judge 与 reward hacking；binary outcome reward 与 dense reward 的取舍。
6. Sim-to-real gap、digital twin、SIMPLER 式 real-to-sim 评测；AutoEval 真机评测的协议。

## 复现可行性

- **代码**：官方 repo https://github.com/world-gymnast/world-gymnast 存在（2026-09-27 访问）。README 给了 `examples/run_openvla_oft_rl_worldgym.sh`；repo 是 SimpleVLA-RL 的 fork，带 `verl` 目录（README 本身没写"基于 veRL"），致谢 WorldGym、SimpleVLA-RL、OpenVLA-OFT。训练数据格式是"PNG 初始帧 + JSON 指令/判分标准"。WorldGym 和 SFT policy 的 checkpoint 是否一并发布，repo 页面上我没有确认到，需要自己检查。
- **算力**（论文原文）：RL 用 4×H200（140GB）做 7B 全参数微调，耗时 1–2 天（Sec 4.1，Table `tab:rl_hyper`）。world model 训练用 2×A100 80GB、580k 步、batch 16、lr 8e-5（Table `tab:hyper`，标题是"World-Gymnast 视频 world model 的训练超参"，没明说这就是 Open X-Embodiment 上的预训练）。Dyna 的 world model 微调共 120k 步，硬件未写。另外还有 GPT-4o API 的调用开销，论文没报。
- **超参**：lr $5\times10^{-6}$，group 8，batch 20，chunk 5，最长 40 步， $\epsilon_{high}/\epsilon_{low}=0.28/0.2$，temperature 1.6，entropy coef 0，grad clip 1.0，AdamW。
- **数据 / 环境**：OpenVLA 评测任务集的初始帧（17 个任务），BridgeData V2（用于 SFT），Open X-Embodiment（world model 预训练）。真机部分依赖 AutoEval 平台，论文称其对公众开放。
- **最小可复现实验**：直接用官方 WorldGym checkpoint 和 Bridge SFT 后的 OpenVLA-OFT，只挑 2–3 个 OpenVLA 任务（例如 put eggplant into pot），group 8，horizon 40，GPT-4o 判分，训几百步。然后在 held-out 初始帧上用"5 次 VLM 判断 + 多数票"评测，看 SFT → RL 的成功率是否上升，参照 Table 3 里 58 → 74 的量级。建议同时做一个论文没做的校验：人工标注约 100 条生成 rollout，测 GPT-4o 判分的 precision/recall，并看训练后期 VLM 判定为成功、人眼却判为失败的比例有没有升高（reward hacking 信号）。如果能用 AutoEval，再做一次 *close the drawer* 的单帧 TTT，复核 62 → 100。
- **预计难点**：world model rollout 很慢，每帧要 20 次 DiT 前向，KV cache 必须开；7B policy 和 600M world model 要同时占显存；GPT-4o 的延迟、费用和非确定性；基座成功率过低时组内无方差，会被 dynamic sampling 大量丢弃；自己场景的帧如果偏离 Open X-Embodiment 分布，world model 生成会失真（作者自承的局限）。

## 自测问题

1. 在 Eq.4–6 中，如果某个 $(g,o_0)$ 的 8 条想象 rollout 全被 VLM 判为成功，这组数据对梯度有什么贡献？dynamic sampling 为什么要丢掉它？这也解释了为什么 SIMPLER baseline 不得不改用 dense partial reward。
2. OpenVLA-OFT 默认的 L1 回归动作头为什么不能直接用于 Eq.6？换成 LM head 之后，概率比 $r_{t,k}(\theta)$ 是在什么粒度上计算的？
3. Iter-SFT 与 World-Gymnast 用的是同一个 world model、同样的 rollout 步数，为什么前者在 easy 任务上会退化？请从"只学成功轨迹"与"有正负 advantage 的 on-policy 更新"的区别，以及对 VLM 误判的敏感性来解释。
4. Table 3 的数字和 Table 1/2 的数字在证据强度上差在哪里？如果 policy 学会了利用 WorldGym 的生成伪影，哪张表会暴露问题，哪张不会？
5. 本文的 "test-time training" 与 TTT（Sun et al. 2019）/ Tent（test-time adaptation）在"更新信号从哪里来、适应的是什么、为什么会过拟合"三点上有什么不同？Dyna 式 world model 更新又对应 real2sim2real 里的哪一类做法？
