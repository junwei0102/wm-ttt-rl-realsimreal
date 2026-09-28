# World Models

> arXiv [1803.10122](https://arxiv.org/abs/1803.10122) · arXiv preprint, 2018（用 ICML 2018 模板排版；同一工作的同行评审版本是 NeurIPS 2018 的 *Recurrent World Models Facilitate Policy Evolution*，已在 NeurIPS 2018 proceedings 页核实） · 主题：01_world_model · 层级：foundational
> 代码：https://github.com/hardmaru/WorldModelsExperiments （MIT 许可；交互版论文见 https://worldmodels.github.io） · 笔记依据：`source/main.tex`（数字/公式）+ `paper.pdf`（图）

## 一句话

把 agent 拆成三部分：V（ConvVAE，把每帧压成 $z_t$）、M（MDN-RNN，预测 $p(z_{t+1}\mid a_t,z_t,h_t)$）、C（只有约一千个参数的线性控制器，用 CMA-ES 优化）。world model 用随机策略采到的数据做无监督训练，C 只看 $[z_t,h_t]$（Doom 中还拼上 LSTM 的 cell 向量 $c_t$，A.3）。这样做在 CarRacing-v0 上首次达到 solved 标准（作者原话是“据我们所知”），在 VizDoom 中则可以完全在 M 生成的“梦境”里训练 C 再迁回真实环境。最关键的一个想法是：调高 MDN 的采样温度 $\tau$，让 controller 难以 exploit 模型的误差。

## 问题设定

- **任务**：从像素出发的 RL，环境部分可观测。实验用了两个 OpenAI Gym 环境。
  - `CarRacing-v0`：俯视视角赛车，每局赛道随机生成，访问的 tile 越多、用时越短，奖励越高。动作是 3 维连续量：转向 $[-1,1]$、油门 $[0,1]$、刹车 $[0,1]$（A.3）。solved 的标准是连续 100 局平均分达到 900（Sec 3 脚注；A.4 写作 above 900）。
  - `DoomTakeCover-v0`：躲避房间对面怪物射来的火球。环境没有显式奖励，回报定义为存活步数，每局最多 2100 步（约 60 s）；100 局平均超过 750 步（约 20 s）即 solved（Sec 4.1）。动作原本是离散的“左 / 不动 / 右”，论文把它映射到 $[-1,1]$ 并三等分（A.3）。
- **数据**：随机策略采 10,000 条 rollout（Sec 3.2 与 4.2 的步骤 1），每帧 resize 成 64×64×3（A.1）。
- **假设**：
  1. 随机策略的探索足以覆盖任务所需的状态。作者在 Sec 5 承认这只对简单任务成立。
  2. CarRacing 中 world model 完全看不到 reward（Sec 3.1）。
  3. Doom 中 M 额外预测 done，而回报等于存活步数、可以直接由 done 推出，所以 M 本身就构成一个完整的 RL 环境，不需要 reward model。这是 Doom 能在梦里训练的前提。
- **之前的方法卡在哪**：
  1. model-free RL 受 credit assignment 限制，只能训小网络（ $10^3\text{–}10^6$ 参数，脚注 1），大网络的表征能力用不上（Sec 1）。
  2. PILCO 这类 GP 动力学模型扩展不到高维像素（Sec 6）。
  3. 以往学到的 dynamics model 很少被用来**完全替代**真实环境。作者推测（原文 “could be the reason”）原因可能是 agent 会找到骗过模型的 adversarial policy，确定性 RNN 模型尤其容易被 exploit。Nagabandi 2017 最终仍要靠 model-free 微调（Sec 4.5）。

## 核心方法

**总体结构**（Figure 4、Figure 8）：V 和 M 合起来构成 world model，C 负责决策。大部分参数放在能用反传高效训练的 V/M 里，C 故意做得很小，好让 credit assignment 只发生在很小的搜索空间里（Sec 1、2.4）。

**V：ConvVAE**（A.1，结构见 Figure 22）
- 编码器：4 层 stride-2 的 relu conv，通道数 32/64/128/256，kernel 4。张量尺寸依次为 64×64×3 → 31×31×32 → 14×14×64 → 6×6×128 → 2×2×256。之后接 dense 层输出 $\mu,\sigma\in\mathbb R^{N_z}$，再用 reparameterization 采样 $z=\mu+\sigma\odot\epsilon,\ \epsilon\sim\mathcal N(0,I)$。
- 解码器：dense 到 1×1×1024，接 4 层 deconv（kernel 5/5/6/6），最后一层用 sigmoid。
- 维度： $N_z=32$（CarRacing）， $N_z=64$（Doom）。
- 损失：A.1 只用文字写了“ $L^2$ 重构 + KL”，下式是按这段描述写出的标准形式：
$$\mathcal L_V=\lVert x_t-\hat x_t\rVert_2^2+D_{\mathrm{KL}}\negthinspace\big(\mathcal N(\mu,\operatorname{diag}\sigma^2)\thinspace\Vert\thinspace\mathcal N(0,I)\big)$$
- 训练：只训 1 epoch。
- 作者特别指出高斯先验的两个作用：限制每帧压缩的信息量；让 world model 对 M 生成的“不真实” $z$ 更鲁棒（A.1，原文主语是 world model 而不是 V）。

**M：MDN-RNN**（Sec 2.2、A.2，Figure 6 与 Figure 23）
- 主干：LSTM，CarRacing 用 256 个隐藏单元，Doom 用 512 个。
- 输出层：Mixture Density Network，建模
$$P(z_{t+1}\mid a_t,z_t,h_t)\quad\text{（Sec 2.2）};\qquad P(z_{t+1},d_{t+1}\mid a_t,z_t,h_t)\quad\text{（Doom，Sec 4.2 步骤 3）}$$
- 混合形式：5 个高斯分量，不建模相关系数 $\rho$，即对角、因子化的混合高斯（A.2）。论文这句话有歧义。官方代码 `doomrnn/doomrnn.py` 先把输出 reshape 成 `[-1, KMIX*3]`，也就是给 $z$ 的**每一维**单独配一个 5 分量的一维混合：
$$p(z_{t+1}\mid\cdot)=\prod_{i=1}^{N_z}\sum_{k=1}^{5}\pi_{i,k}\thinspace\mathcal N\negthinspace\big(z_{t+1,i};\mu_{i,k},\sigma_{i,k}^2\big),\qquad \mathcal L_M=-\log p(z_{t+1}\mid\cdot)\thickspace\big[+\thinspace\mathrm{BCE}(d_{t+1})\big]$$
  代码里 done 这一项用 sigmoid cross-entropy，并对正样本（死亡帧）加权（`restart_factor`）；论文正文没有写这一点。
- 训练：teacher forcing，20 epochs。每帧预先存好 V 输出的 $(\mu,\sigma)$，每次组 batch 时重新采样 $z\sim\mathcal N(\mu,\sigma)$ 作为输入，防止 M 过拟合到某一个固定的采样值（A.2）。
- **温度 $\tau$**：论文只说“参照 SketchRNN 调 $\tau$ 来控制不确定性”（Sec 2.2）。官方代码的实现是把混合权重的 logits 除以 $\tau$，高斯噪声乘以 $\sqrt\tau$：
$$\pi^{(\tau)}_ {i,k}\propto\exp(\log\pi_{i,k}/\tau),\qquad z_{t+1,i}=\mu_{i,k^{\ast}}+\sigma_{i,k^{\ast}}\sqrt{\tau}\thinspace\epsilon$$
  $\tau\to0$ 时总是选最大分量、没有噪声，退化成确定性 LSTM； $\tau>1$ 时分量选择和噪声都会变大。
- **梦境中的 done**：预测概率 >50% 就判定死亡，而不是按 Bernoulli 采样。理由是死亡属于低概率事件，用阈值判定更稳定（A.2）。

**C：线性控制器**（Eq. 1，`source/main.tex` Sec 2.3）
$$a_t=W_c\thinspace[z_t\thickspace h_t]+b_c$$
- 输出外套一层 tanh，裁到动作范围内（A.3）。
- CarRacing： $h$ 取 LSTM 的输出向量（256 维），参数量 $(32+256)\times3+3=867$。
- Doom：输入是 $z(64)+c(512)+h(512)=1088$ 维（A.3：Doom 中隐状态取 LSTM 的 cell 向量 $c$ 和输出向量 $h$），输出 1 维且不带 bias（Sec 4.2 步骤 4），参数量为 1,088。
- 这两个数都和论文中未编号的参数表一致（算式是我按论文给的维度推出来的）。完整参数表：
  - CarRacing：VAE 4,348,547 / MDN-RNN 422,368 / C 867
  - Doom：VAE 4,446,915 / MDN-RNN 1,678,785 / C 1,088

**优化：CMA-ES**（A.4）
- 设置：population 64，每个候选用不同随机种子跑 16 局，fitness 取这 16 局的平均回报。
- 选 ES 的理由：只需把最终的累积回报交给优化器，而不需要整条 rollout 的历史，而且容易并行（Sec 6）。
- CMA-ES 只适合最多几千个参数的解空间（Sec 2.4），这正是 C 必须做得很小的原因。

**训练与推理流程**
- **分阶段训练**：V、M、C 各自单独训练（Sec 2.3）。V 和 M 原则上可以端到端一起训，但作者认为分开训更实用，而且 V、M 各自在单 GPU 上训练不到 1 小时（Sec 3.1 脚注）；C 则是在多核 CPU 上用 CMA-ES 训练（Sec 2.4、A.4）。
- **CarRacing**（Sec 3.2）：
  1. 用随机策略采 rollout。
  2. 训练 V。
  3. 用 V 把帧编码成 $z$，再训练 M。
  4. 在**真实环境**中用 CMA-ES 优化 C。
  
  推理时每一步依次执行：`z=V(obs)` → `a=C([z,h])` → `env.step(a)` → `h=M([a,z,h])`（Sec 2.4 的伪代码）。
- **Doom：在梦里学**（Sec 4.1–4.4、A.5）：
  1. 把 M 包装成 `gym.Env` 接口，每一步按温度 $\tau$ 采样 $z_{t+1}$；done 不采样，而是按预测概率 >50% 判定（A.2）。
  2. 整个过程都在 latent 空间里运行，**不需要 V 去编码像素**，也不需要运行游戏引擎。作者把这个虚拟环境叫 DoomRNN。
  3. C 在 DoomRNN 中用 CMA-ES 最大化存活时间。
  4. 训练好的 C 直接部署到真实 VizDoom，此时由 V 编码真实帧。
  
  论文没有说明梦境 rollout 的初始状态是怎么得到的（官方代码 `doomrnn.py` 在 `reset()` 时从预存的初始 $\mu$、logvar 采样 $z$）。
- **为什么 $h_t$ 有用**：M 被训练来预测 $z_{t+1}$ 的分布，所以 $h_t$ 里编码了对未来的预期。C 不需要显式地 rollout 或 planning，就能“本能地”做出反应（Sec 3.3）。

**Iterative training**（Sec 5。作者说本文实验相当于这个循环只跑了一轮，对简单任务已经够用；多轮迭代没有做实验）
1. 随机初始化 M 和 C。
2. 在真实环境里 rollout $N$ 次，把 $(a_t,x_t)$ 存下来。
3. 训练 M 去建模 $P(x_{t+1},r_{t+1},a_{t+1},d_{t+1}\mid x_t,a_t,h_t)$，并在 M 内训练 C。
4. 如果任务还没解决，回到第 2 步。

作者还建议把 M 的损失取负号作为 curiosity 内在奖励，引导 agent 去探索模型不熟悉的区域。

## 关键实验与证据

**CarRacing-v0**（Table 1，列名 Avg. Score。V model、Full World Model 与 leaderboard 三行在正文中注明是 100 局随机 trial；V model with hidden layer 与 DQN/A3C 的评测局数原文未写明；± 的含义论文未注明，按惯例应为标准差）

| 方法 | 分数 |
|---|---|
| DQN | 343 ± 18 |
| A3C（连续动作） | 591 ± 45 |
| A3C（离散动作） | 652 ± 10 |
| Gym leaderboard 最佳（ceobillionaire） | 838 ± 11 |
| 只用 V（ $a_t=W_cz_t+b_c$） | 632 ± 251 |
| 只用 V + 40 个 tanh 单元的隐藏层（C 共 1443 参数，A.4） | 788 ± 141 |
| **Full World Model（V+M）** | **906 ± 21** |

- **消融**：只看 $z$ 的 C 即使加了隐藏层、参数量（1443）比完整模型的 C（867）还多，成绩依然明显更差。所以真正起作用的是 M 提供的 $h_t$，而不是 C 的容量。
  - z-only 版本的分数有很长的低分尾巴（Figure 26、27），表现为开车摇摆、急弯冲出赛道（Figure 11）。
- **训练过程**（A.4，Figure 24）：每 25 代取当前最优个体，在 1024 局上评估一次。大约 1800 代时达到 900.46；同一个 agent 在 100 局上的成绩是 906 ± 21（Figure 25）。
- **梦境演示**（Sec 3.4，Figure 13）：只是定性展示，说明在真实环境里训练出的 C 在梦境里也“大体能开”。

**VizDoom Take Cover**（Table 2：在温度 $\tau$ 下于梦境中训练，然后在真实环境跑 100 局随机 rollout）

| $\tau$ | 梦境分数 | 真实分数 |
|---|---|---|
| 0.10 | 2086 ± 140 | 193 ± 58 |
| 0.50 | 2060 ± 277 | 196 ± 50 |
| 1.00 | 1145 ± 690 | 868 ± 511 |
| 1.15 | 918 ± 546 | **1092 ± 556** |
| 1.30 | 732 ± 269 | 753 ± 139 |
| 随机策略 / Gym leader | — | 210 ± 108 / 820 ± 58 |

- **低 $\tau$ 下 C exploit 了模型**：梦境分数接近满分上限 2100，真实分数反而低于随机策略。机制是 $\tau=0.1$ 时 M 接近确定性 LSTM，因为 mode collapse 无法“跳”到“发射火球”那个混合分量，于是梦里的怪物永远不开火（Sec 4.5）。
  - 早期实验中，agent 甚至学会用特定的走位让已经发射的火球“自动熄灭”（Figure 18）。
- **$\tau$ 的取舍**： $\tau=1.15$ 效果最好。 $\tau=1.30$ 的均值更低，但方差小得多，对应一种更保守的策略（Sec 4.5）。
- **真实分数高于梦境分数**（ $\tau$=1.15、1.30 两行）：作者的解释是，能在更嘈杂的梦里活下来的 agent，到了更“干净”的真实环境会表现得更好（Sec 4.4）。
- **A.5 补充**：在 DoomRNN（ $\tau=1.15$）中，最佳个体在梦境 1024 局上平均得分 959；同一个 agent 在真实 `DoomTakeCover-v0` 上 100 局得分 1092 ± 556（Figure 29）。注意 959 与 Table 2 中 $\tau=1.15$ 的梦境分数 918 ± 546 并不相同（Sec 4.3 又只写“约 900”），论文没有解释两者的差别。作者明确说明**没有**在真实 VizDoom 中训练 agent，VizDoom 只用来采随机策略的训练数据，因此也就没有“真实环境训练”的对照。
- **为什么用混合高斯而不是单高斯**（Sec 4.5）：“怪物开不开火”这类离散随机事件需要多模态分布来表达。单个对角高斯足够编码单帧，但表达不了环境逻辑。

## 局限与论文没说的

**作者承认的**
1. V 单独做无监督训练，不知道哪些信息与任务相关（Sec 7）。例如在 Doom 中复现了与任务无关的墙砖纹理，在 CarRacing 中却没有复现路面上与任务相关的 tile。和预测 reward 的 M 联合训练可以缓解这个问题，代价是 V 不容易再复用到新任务上。
2. world model 可以被 exploit（Sec 4.5）。C 能读到 M 的全部隐状态，相当于拿到了“游戏引擎的内部状态”。调 $\tau$ 只能缓解，而且需要调参。
3. 随机策略采的数据只够应付简单任务。复杂环境需要 iterative training 加 curiosity，但论文只提出了方案，没有实现（Sec 5）。
4. LSTM 容量有限，还存在 catastrophic forgetting（Sec 7）。
5. 模型只是逐步模拟未来，没有分层规划或抽象推理（Sec 7）。

**我读出来的**
- **梦中训练只在 Doom 上验证过**：这个任务只有三选一的 1 维动作，奖励就是存活时间。M 不预测 reward，所以这套做法不能直接用到需要 reward model 的任务上。CarRacing 的 C 其实是在真实环境里训练的。
- **样本效率被掩盖**：CarRacing 的 CMA-ES 每代需要 64×16 = 1024 局真实 rollout，到 1800 代（A.4）时已约 1.8M 局，这还不含每 25 代一次的 1024 局评估和 10,000 局随机数据（Figure 24 的训练曲线画到 2000 代）。Table 1 与 DQN/A3C 只比了最终分数，没有比与环境的交互量。
- **评测口径**：
  - $\tau$ 是按真实环境的分数挑出来的（Table 2），等于用测试环境调超参。
  - 报告的是单个“最佳个体”在 100 局上的 mean ±（± 的含义论文未注明），没有多个训练种子。Doom 的 ± 高达 556。
  - 基线来源不统一：DQN 的数字引自 Prieur 2017（参考文献只给了一个短链接），两个 A3C 数字引自 Khan & Elibol 2016（Stanford CS221 课程项目报告）和 Jang et al. 2017（Scribd 上的报告），另有 leaderboard 分数，并不是同一个评测协议。
- **分阶段训练导致 V/M 冻结**：C 一旦进入随机数据没有覆盖到的状态（distribution shift），world model 就可能失真（Sec 4.5 只是定性提到 adversarial policy 往往访问远离训练分布的状态）。论文没有量化 M 在 on-policy 状态上的预测误差，也没有 V/M 联合训练的对照实验。
- **可扩展性**：CMA-ES 只适合几千个参数的线性 C，没法直接扩展到高维动作或更大的策略网络。M 是可微的，论文却没有利用这一点：Sec 7 提到可以反传，但没有做。
- **环境版本过时**：实验用的是 gym 0.9.x 的 `CarRacing-v0` 和 `DoomTakeCover-v0`（gym 版本出自官方仓库 README，论文本身没写），这些版本已经过时，今天的环境版本下这些数字无法一比一对照。

## 在脉络中的位置

- **前驱**：
  - Schmidhuber 1990–1991 年基于 RNN 的 C–M 系统（*Making the World Differentiable* 等），以及 2015 年的 *Learning to Think*。本文自述是对这些思想的一个简化实验版，符号也沿用了 *Learning to Think*。
  - 各组件的来源：VAE（Kingma & Welling 2013）；MDN（Bishop 1994）与 RNN 的结合（Graves 2013）；SketchRNN（Ha & Eck 2017，温度采样）；CMA-ES（Hansen 2016 的 tutorial；原始算法 Hansen & Ostermeier 2001），其设置沿用 *Evolving Stable Strategies*（Ha 2017，博客）。
  - model-based RL 方面：PILCO，以及 Dyna 式“用模型生成经验”的思路。
- **后继 / 影响**：
  - Dreamer 系列：[Dreamer](../dreamer/notes.md) 把分阶段的 V+M 换成联合训练、带 reward 预测的 RSSM，C 换成在 latent imagination 中训练的 actor-critic，并通过模型反传价值梯度。这恰好是本文 Sec 7 提到的“可反传”和“与预测 reward 的 M 联合训练”这两个方向。[DreamerV3](../dreamerv3/notes.md) 做到跨领域用同一套超参；[Dreamer 4](../dreamer4/notes.md) 在可扩展的 world model 中训练 agent。
  - [DIAMOND](../diamond/notes.md)：用 diffusion 做 world model，标题主张 *visual details matter*。它针对的是近期基于离散 latent 的 world model（DreamerV2/V3、IRIS 等）压缩时可能丢掉对 RL 重要的视觉细节，并不是直接针对本文；但与本文 Sec 7 提到的“V 没复现路面上与任务相关的 tile”属于同一类问题。DIAMOND 在 related work 中把“在 world model 的想象中训练 RL”这一思路归于本文。
  - 把“学到的可交互模拟器”扩大到视频规模：[Genie](../genie/notes.md)、[Cosmos WFM](../cosmos-wfm/notes.md)、[UniSim](../../05_intersections/unisim/notes.md)。[DreamZero](../dreamzero/notes.md) 则把 world model 与动作生成合并。
  - [TD-MPC2](../tdmpc2/notes.md)、[V-JEPA 2](../vjepa2/notes.md)：latent model 不重构像素，并配合 planning，与本文“重构式 V、C 不做规划”的路线形成对照。
  - 机器人与 VLA 方向延续了“在 world model 里训 policy，再上真机或真实环境”的范式，也都要面对本文 Sec 4.5 的 model exploitation 问题：[DayDreamer](../../05_intersections/daydreamer/notes.md)、[Robotic World Model](../../05_intersections/robotic-world-model/notes.md)、[Ctrl-World](../../05_intersections/ctrl-world/notes.md)、[WMPO](../../05_intersections/wmpo/notes.md)、[World-Gymnast](../../05_intersections/world-gymnast/notes.md)。
- **与本项目其他主题的联系**：
  - **real2sim2real**：
    - 本文把 learned world model 当作 simulator 来用，Sec 7 提到它可能（“may”）与 sim2real 方法互补。
    - 调高 $\tau$，让梦境比真实环境更随机更难，再迁回真实环境。机制上类似 [domain randomization](../../03_real2sim2real/domain-randomization/notes.md)：在更宽的分布上训练，从而覆盖真实环境。
    - Sec 5 的 iterative training（用真实数据更新模型、再重新训练策略）与 [SimOpt](../../03_real2sim2real/simopt/notes.md)（用真实经验修正仿真参数分布）是同一种闭环思路。
    - “C exploit 模型误差”可以看作 learned simulator 版本的 sim-to-real gap，可参照 [reality gap 综述](../../03_real2sim2real/survey-reality-gap/notes.md)。[SimDist](../../05_intersections/simdist/notes.md) 则反过来，先在仿真里预训练 world model，再快速适应真实环境。
  - **TTT**：
    - 本文的 V/M 在部署时是冻结的。[PAD](../../05_intersections/pad/notes.md) 在部署时用 self-supervised auxiliary task 更新编码器，正好针对这个短板。
    - Sec 5 把 M 的预测损失当作 curiosity 信号，Sec 7 讨论了 LSTM 容量不足与遗忘问题。这两点和 TTT 的做法相通：在部署时用自监督损失更新 fast weights，例如 [TTT layers](../../02_test_time_training/ttt-layers/notes.md)、[Titans](../../02_test_time_training/titans/notes.md)。
  - **RL**：
    - C 用黑箱 ES 训练，而不是 policy gradient。后来的 world model 工作大多改为在 imagination 中训练 actor-critic。
    - Table 1 的 DQN 基线对应 [DQN](../../04_rl/dqn/notes.md)。
    - 本文与 model-free RL 的对照可放在 [RL overview](../../04_rl/survey-rl-overview/notes.md) 的 model-based RL 框架下来理解。

## 前置知识

1. VAE：ELBO、reparameterization trick、KL 项的作用（限制信息量、正则化 latent 空间）。
2. RNN/LSTM：hidden state $h$ 与 cell $c$ 的区别，teacher forcing。
3. Mixture Density Network：混合高斯的负对数似然、logsumexp，以及 SketchRNN 式的温度采样。
4. POMDP 与 model-based RL：Dyna、learned dynamics、model exploitation 与 compounding error。
5. Evolution strategies / CMA-ES：只依赖 fitness 的黑箱优化，population 与协方差自适应。
6. OpenAI Gym 的 `reset/step/done` 接口，这是把 M 包装成虚拟环境的关键。

## 复现可行性

- **代码**：官方仓库 https://github.com/hardmaru/WorldModelsExperiments（MIT，包含 `carracing/` 和 `doomrnn/`），2026-09-27 通过 WebFetch 确认仓库存在。README 声明只支持 gym 0.9.x，不支持 0.10.x。
- **论文给出的算力**：
  - V、M 各自在单 GPU 上训练不到 1 小时（Sec 3.1 脚注）。
  - CMA-ES 在一台 64 核机器上并行跑 rollout（A.4）。
  - 实验跑在 Google Cloud 的 Ubuntu 虚拟机上（Acknowledgements）。
  - 论文**没有**给出 CMA-ES 的总墙钟时间或 GPU 型号。
- **作者博客给出的算力**（blog.otoro.net 2018-06-09 *World Models Experiments*，**不是论文内容**，以博客原文为准）：
  - V/M 在 P100 GPU 实例上训练，C 在 64 核 CPU 实例（n1-standard-64）上训练。
  - DoomTakeCover 实验整体复现不到 24 小时。
  - CarRacing 跑 150–200 代大约需要 3 天，此时平均分约 880（博客原文），还没到 900；论文达到 900.46 用了约 1800 代（A.4）。
  - 依赖版本：Python 3.5.4、TF 1.8.0、gym 0.9.4、cma 2.2.0。
- **数据与环境**：`CarRacing-v0`（依赖 Box2D）和 `DoomTakeCover-v0`（依赖 VizDoom 及其 gym 封装）。10,000 条随机 rollout 需要自己采集。
- **最小可复现实验**：选 DoomTakeCover 的梦境训练加温度扫描。这个实验直接对应论文的核心主张（在梦中训练，用 $\tau$ 抑制 exploit），而且训练 C 时不用跑游戏引擎，CPU 预算可控。
  1. 用随机策略采 10,000 局，训练 VAE（ $N_z=64$，1 epoch），再训练 MDN-RNN（512 个单元，5 个分量，20 epochs，同时预测 done）。
  2. 在 DoomRNN 中分别用 $\tau=0.5$ 和 $\tau=1.15$ 跑 CMA-ES（population 64 × 每候选 16 局），代数可以先减少。
  3. 在真实环境中各跑 100 局评估，检查两点：低 $\tau$ 是否出现“梦中接近 2100，现实约 200（随机水平）”； $\tau=1.15$ 是否越过 750 的 solved 线。
  
  如果还想做 CarRacing，建议只复现 V-only 对比 V+M 的消融（Table 1），减少代数，只看趋势。
- **难点**：
  - 依赖老旧：TF1、gym 0.9.x、VizDoom 需要编译。
  - 现在的 CarRacing 版本号和细节都已变化，分数不能直接对照。
  - CMA-ES 的结果方差很大，而论文只给了单次运行的结果，需要跑多个种子才能判断差异是否可靠。
  - 梦境的初始状态、done 损失的加权等细节论文没写，只能去看 `doomrnn.py`。

## 自测问题

1. CarRacing 中 C 的参数量为什么恰好是 867，Doom 中为什么是 1,088？由此能推断 C 的输入各包含哪些向量？
2. 只看 $z$ 的 C 加了 40 个单元的隐藏层（1443 参数）后，为什么仍然比不过看 $[z,h]$ 的线性 C（867 参数）？ $h_t$ 里存了什么信息，为什么不需要显式 planning？
3. 用 MDN 的温度采样公式和 mode collapse 解释：为什么 $\tau=0.1$ 时梦境分数接近满分，真实分数却低于随机策略？ $\tau$ 调到 1.30 为什么又会降分？
4. 为什么 Doom 可以全程在梦里训练，而 CarRacing 论文没有这样做？如果要在 CarRacing 的梦境里训练 C，M 还需要额外输出什么？
5. 训练 M 时为什么每次组 batch 都要从 $\mathcal N(\mu,\sigma)$ 重新采样 $z$，而不是用固定的 $z$？这与 A.1 所说“高斯先验让 world model 对 M 生成的不真实 $z$ 更鲁棒”有什么关系？
