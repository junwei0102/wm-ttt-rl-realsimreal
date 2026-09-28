# TD-MPC2: Scalable, Robust World Models for Continuous Control

> arXiv [2310.16828](https://arxiv.org/abs/2310.16828) · ICLR 2024（Spotlight） · 主题：01_world_model · 层级：milestone
> 代码：https://github.com/nicklashansen/tdmpc2（MIT；项目页 https://tdmpc2.com 提供 300+ checkpoints 和 30/80-task 数据集）· 笔记依据：`source/iclr2024_conference.tex`（数字/公式）+ `paper.pdf`（图）

## 一句话

在 TD-MPC 的框架上（decoder-free 的 implicit world model，加潜空间 MPPI 规划，再用 learned Q 做 bootstrap）做了一组鲁棒性改动：SimNorm 潜变量归一化、在 log 变换空间里用离散回归预测 reward/value、maximum-entropy policy prior、Q ensemble。结果是一套固定超参数能跑通 104 个连续控制任务，world model 从 1M 扩到 317M 参数时，80 任务 agent 的性能一直在涨。

## 问题设定

- **目标**：用大规模、质量参差（uncurated，从随机到专家都有）的多任务、多 embodiment、多 action space 数据训练 generalist 连续控制 agent。作者认为已有路线卡在两点（Sec 1）：behavior cloning 需要近专家的数据，而 RL 算法大多是单任务的，还要逐任务调超参。另一个直接动机：把 TD-MPC 的模型和数据直接放大，性能反而会**下降**（Fig 1 左、Fig 7 蓝线）。
- **形式化**：infinite-horizon MDP，连续动作（Sec 2）。所有环境的 episode 长度固定，没有 termination（App C）。输入以 proprioceptive state $\mathbf s$ 为主，另有 10 个 DMControl 图像任务。输出是连续动作。需要环境给 reward，dense 和 sparse 两种都有。
- **三种训练形态**：(a) 单任务 online RL，从零开始，数据进 replay buffer；(b) 多任务训练：在固定数据集上训练，数据是 240 个单任务 agent 的 replay buffer，共 545M transitions，覆盖 80 个任务（50 个 Meta-World + 30 个 DMControl）。作者在 App J 承认这本质上是 offline RL；(c) 预训练模型迁移到新任务，做 few-shot online finetune。
- **TD-MPC 的具体问题**：潜变量没有约束，会梯度爆炸，部分任务训练发散（App G，Fig 21）；reward/value 用连续回归，对 reward 的量级敏感（App A）；Dog/Humanoid 这类任务要逐任务调规划迭代数、latent 维度、batch size 和 lr（App H）。

## 核心方法

**1. 五个组件**（`source/iclr2024_conference.tex` Eq.2，下文省略任务嵌入 $\mathbf e$）

$$\mathbf z=h(\mathbf s,\mathbf e),\quad \mathbf z'=d(\mathbf z,\mathbf a,\mathbf e),\quad \hat r=R(\mathbf z,\mathbf a,\mathbf e),\quad \hat q=Q(\mathbf z,\mathbf a,\mathbf e),\quad \hat{\mathbf a}=p(\mathbf z,\mathbf e)$$

五个组件依次是 encoder、latent dynamics、reward、terminal value 和 policy prior，**没有 decoder**。模型不重建观测，只求能准确预测给定动作序列下的 *outcome*（也就是 return）。作者称之为 control-centric 的 implicit world model（Sec 3.1）。

**2. 模型目标**（Eq.3）。从 buffer 采长度为 $H$ 的子轨迹。令 $\mathbf z_0=h(\mathbf s_0)$，再用 $d$ 沿 replay 里的动作做 latent rollout，每一步都计算损失：

$$\mathcal L(\theta)=\mathbb E_{(\mathbf s,\mathbf a,r,\mathbf s')_{0:H}\sim\mathcal B}\Big[\sum_{t=0}^{H}\lambda^t\big(\|\mathbf z'_t-\mathrm{sg}(h(\mathbf s'_t))\|_2^2+\mathrm{CE}(\hat r_t,r_t)+\mathrm{CE}(\hat q_t,q_t)\big)\Big],\quad q_t=r_t+\gamma\bar Q(\mathbf z'_t,p(\mathbf z'_t))$$

- 第一项是 joint-embedding prediction（BYOL 式）：用 stop-grad 的真实下一状态编码作为预测目标。$h,d,R,Q$ 联合优化（Sec 3.1），第二、三项的梯度也会回传到 encoder 和 dynamics，所以潜空间由 latent 预测、reward、value 三项共同塑形（control-centric），而不是由观测重建塑形。
- $\bar Q$ 是 $Q$ 的 EMA（momentum 0.99）。$Q$ 是 5 个成员的 ensemble，第一层后加 1% Dropout。TD-target 取随机抽出的两个 $\bar Q$ 中较小的那个（REDQ 式）。
- reward 和 value 都在 **log 变换空间里做离散回归**：分 101 个 bin，以 $r_t,q_t$ 为 soft target 算 cross-entropy。这样损失量级就和任务的 reward 量级无关，这是跨任务共用一套超参数的关键之一。latent 一项仍用 MSE，因为 SimNorm 已经把它归一化了，而离散化高维 $\mathbf z$ 的代价太高（App A）。
- 系数（Table 8）：joint-embedding 20，reward 0.1，value 0.1。$\lambda=0.5$，$H=3$。论文训练子序列长度和规划 horizon 用的是同一个符号 $H$。按论文的记号，$\mathbf z'_t$ 指 Eq.2 中 dynamics 的输出；TD-target 具体用哪个 next latent 需要对照代码确认。

**3. Policy prior**（Eq.4）：maximum-entropy 的随机策略，梯度只更新 $p$，latent 沿 $d$ 展开：

$$\mathcal L_p(\theta)=\mathbb E_{(\mathbf s,\mathbf a)_{0:H}\sim\mathcal B}\Big[\sum_{t=0}^{H}\lambda^t\big[\alpha Q(\mathbf z_t,p(\mathbf z_t))-\beta\mathcal H(p(\cdot|\mathbf z_t))\big]\Big]$$

$\beta=10^{-4}$；$\alpha$ 通过 moving statistics 自动调节（Sec 3.1；Table 8 记为 policy prior loss norm. = moving (5%, 95%) percentiles），以便在不同数据集和训练阶段之间平衡 Q 项和熵项、防止熵过早坍缩。注意：论文说这个式子是要 maximize 的，但字面上 $-\beta\mathcal H$ 的符号和 max-entropy 的意图相反，应理解为鼓励熵，实现以代码为准。$p$ 有两个用途：给规划器提供候选动作序列，以及在 TD-target 中代替对动作的 max。

**4. SimNorm**（Eq.5）：把 $\mathbf z$ 切成 $L$ 段，每段 $V$ 维，每段各自做 softmax，投到单纯形上：

$$\mathbf z^\circ=[\mathbf g_1,\dots,\mathbf g_L],\quad \mathbf g_i=\frac{e^{\mathbf z_{i:i+V}/\tau}}{\sum_{j=1}^{V}e^{\mathbf z_{i:i+V}/\tau}}$$

（原文 Eq.5 左式写作 $[\mathbf g_i,\dots,\mathbf g_L]$，此处按含义写成 $\mathbf g_1$。）$V=8$，$\tau=1$。5M 模型的 latent 是 512 维，由 Table 8 推得 $L=64$。SimNorm 可以看作 VQ-VAE 那种"一组 categorical"表示的 soft 版本（App H）：范数有界，所以能抑制梯度爆炸；同时偏向稀疏，但不强制离散。注意 App H 说 $\tau\to\infty$ 得到 one-hot、$\tau=0$ 得到均匀分布，但按 Eq.5 的 $e^{\mathbf z/\tau}$ 写法，实际情况正好相反（$\tau\to0$ 趋于 one-hot，$\tau\to\infty$ 趋于均匀）。中间层统一用 Linear→LayerNorm→Mish。

**5. MPPI 规划**（Eq.6）：每个决策步求一个时变的对角高斯 $\mathcal N(\mu,\sigma^2)$，$\mu,\sigma\in\mathbb R^{H\times m}$：

$$\mu^*,\sigma^*=\arg\max_{(\mu,\sigma)}\mathbb E_{\mathbf a_{t:t+H}\sim\mathcal N(\mu,\sigma^2)}\Big[\gamma^H Q(\mathbf z_{t+H},\mathbf a_{t+H})+\sum_{h=t}^{H-1}\gamma^h R(\mathbf z_h,\mathbf a_h)\Big]$$

流程：population size 为 512，policy prior samples 为 24（Table 8）；Sec 3.2 只说"a fraction of action sequences"来自 $p$，没有明确这 24 条是否计入 512。在潜空间展开后，估计 return（前 $H$ 步 reward 加末端 Q bootstrap）。取 64 条 elite，按加权平均更新 $\mu,\sigma$（temperature 0.5，std 夹在 [0.05, 2] 之间）。迭代 6 次，$|\mathcal A|\ge 20$ 时迭代 8 次。最后从 $\mathcal N(\mu^*_t,\sigma^*_t)$ 采出第一个动作执行。下一步用上一步的解平移 1 格做 warm-start，不用 momentum（Table 8，Sec 3.2）。有了 Q 的 bootstrap，$H=3$ 的局部轨迹优化才能近似完整的 RL 目标，而不只是 $H$ 步内的最优。

**6. 多任务扩展**（Sec 3.3）：五个组件都以一个可学习的 task embedding $\mathbf e$ 为条件（96 维，$\ell_2$ 范数 $\le 1$）。观测和动作都 zero-pad 到最大维度。policy prior 的预测和熵只在有效动作维上计算，规划时也只在有效维上采样（action masking）。多任务训练时 batch 取 1024，其余超参数不变。

**7. 流程与"无需调参"的来源**：online 阶段先收集 $S=\max(5T,1000)$ 步数据（Eq.8）再开始更新。之后循环：MPPI 选动作 → 写入 buffer → 按 UTD=1、batch 256 均匀采样，更新 Eq.3 和 Eq.4 → 更新 EMA。推理时只需编码当前观测，然后做 MPPI。discount 按 episode 长度用启发式设定：$\gamma=\mathrm{clip}(\frac{T/5-1}{T/5},[0.95,0.995])$（Eq.7），DMControl 的 $T=500$ 时 $\gamma=0.99$。

## 关键实验与证据

- **设置**：104 个任务，其中 DMControl 39、Meta-World 50、ManiSkill2 5、MyoSuite 10（Tables 2–5）。每个配置 3 个 seed，报告 95% CI。环境步数（Table 6，细分见 Table 2/4 caption 和 Fig 12/14 caption）：DMControl 4M（Dog/Humanoid 为 14M），Meta-World 2M，ManiSkill2 4M（Pick YCB 为 14M），MyoSuite 2M。成功只按 episode **最后一步**判定，比部分文献的口径严。对比方法：SAC、DreamerV3（S 尺寸约 20M 参数，UTD 512）、TD-MPC（约 1M 参数）。SAC 和 TD-MPC 都用逐任务调好的超参数；TD-MPC2 只用 5M 模型、batch 256、UTD 1（Sec 4）。
- **单任务**（Fig 1 右、Fig 4）：TD-MPC2 在四个 domain 上都领先。在 Pick YCB 上成功率超过 60%，其他方法在给定预算内都学不会（Fig 14 caption）。在 Dog 任务上 SAC 和 DreamerV3 出现数值不稳定（Fig 5）；TD-MPC 在 Walker Stand/Walk 上发散（Fig 12 caption）；它的梯度范数在 Quadruped Run、Walker Stand 上持续攀升，在 Dog Trot 上一度升到 $10^{9}$ 以上（约 10M 步后又骤降），Fig 21 为对数坐标、每任务只画 1 个 seed；TD-MPC2 则一直保持平稳（Fig 12、Fig 21）。MyoSuite 是事先完全没跑过任何 TD-MPC2 实验就直接报告的结果（Sec 4.1）；在这个 domain 上 TD-MPC2 在所有任务上持平或更好，唯一例外是 Key Turn Hard，TD-MPC 在训练早期就已成功（Fig 16 caption）。
- **Scaling**（Table 1、Fig 7）：80 任务上 1M/5M/19M/48M/317M 模型的 normalized score 分别为 16.0/49.5/57.1/68.0/70.6。在单张 RTX 3090 上的训练开销分别是 3.7/4.2/5.3/12/33 GPU days。30 任务的 DMControl 子集（345M transitions）上分别为 18.9/28.3/54.2/59.4/71.4。TD-MPC 随参数增加反而下降（Fig 7 蓝线）。作者没有拟合 scaling law，只指出 score 大致与 log 参数量成线性。t-SNE 显示，task embedding 更多按 dynamics/embodiment 聚类，而不是按目标（walk/run）聚类（Fig 7 右）。
- **Few-shot**（Fig 8、Fig 17、App E）：19M 模型在 70 个任务上预训练，然后迁移到 10 个 held-out 任务，buffer 从空开始 online finetune 20k 步。得分从零训练的 24.0 提升到 47.0（"2×"）。逐任务看收益差异很大，例如 Hopper Hop 两种做法都接近 0。新任务的 embedding 用人工挑选的相似任务来初始化（Table 7）。
- **消融**（Fig 9；曲线是 Dog Run、Humanoid Walk、Pick YCB 三个任务的平均，柱状图是 19M 模型在 80 任务上的得分）：
  - Actor：只用 policy 42.2，只用 planning 53.7，planning+policy 54.2。规划贡献最大；单任务曲线上，只用 policy 也明显更低。
  - 归一化：不做归一化 46.8，SimNorm 51.0，LN+SimNorm 54.2。
  - 回归方式：连续回归 49.6，离散回归 54.2。
  - Q 的个数：2 个 53.5，5 个 54.2，10 个 57.0。
  - task embedding 不归一化 46.6，归一化 54.2（Fig 18）。
  - 激活函数：Mish 与 ELU/ReLU 性能相当，但梯度更平滑（Fig 20）。
  - 作者的结论：各项改动都有贡献，而且单任务和多任务场景下各项的相对重要性一致。
- **Visual RL**（Fig 10）：换成 4 层卷积 encoder，输入 64×64 图像，加 random shift 增强。在 10 个 DMControl 任务上，超参数不变，与 DrQ-v2、DreamerV3 相当。
- **Test-time regularization**（App J，Table 10）：规划时减去不确定性惩罚 $u_t=c\cdot\mathrm{avg}(\hat q_{1:N})\cdot\mathrm{std}(\hat q_{1:N})$（Eq.9）。对 19M 模型的 80 任务得分，不加正则 56.54，$c=0.001$ 为 58.14，$c=0.01$ 为 62.01，$c=0.1$ 为 44.13。正文的其他实验都没有使用这个正则。

## 局限与论文没说的

**作者承认的**：离散动作空间仍未解决，因为 MPC 本身针对连续动作，可能要换成 MCTS（Sec 5，App I）。方法依赖 reward，而怎样用成功标签、偏好等泛化的 reward 做大规模预训练还没解决。few-shot finetune 只是探索，效果随任务变化。多任务训练本质是 offline RL，但没加 conservatism（App J）。真正的泛化可能需要多几个数量级的任务。风险包括 reward misspecification、物理机器人的安全问题，以及数据成本带来的资源集中。

**我读出来的**：
1. **全部在仿真里完成，而且以 state 输入为主**：没有真机实验，没有 sim-to-real，视觉部分只有 10 个任务、64×64 分辨率。
2. **Markov 与确定性假设**：encoder 只看单帧 $\mathbf s$，没有历史或 recurrent state（不同于 RSSM）；$d$ 是确定性的。因此部分可观测的情况，比如未知的摩擦和质量，也就是 sim-to-real 关心的那类参数，无法靠历史在线推断；多模态的随机转移也表示不了。
3. **implicit 模型不能当 simulator 用**：没有 decoder，就生成不了观测，也不能像 UniSim/Ctrl-World 那样拿来评估或训练其他 policy。潜空间是为特定 reward 塑形的，reward 一换，表示可能就缺失需要的信息。
4. **"一套超参数"其实包含若干启发式规则**：$\gamma$ 和 seed steps 依赖已知的 episode 长度，$|\mathcal A|\ge20$ 时规划多迭代 2 次。另外，baseline 的配置（DreamerV3 固定用 S 尺寸和 UTD 512；SAC/TD-MPC 沿用原论文的超参数）是否对每个 domain 都公平，论文没有讨论。
5. **评测口径有不一致之处**：19M 模型、80 任务的默认配置，在 Table 1 中是 57.1，在 Fig 9/18 中是 54.2，在 Table 10 中是 56.54，论文没有解释（可能是训练时长或批次不同），所以只能在同一张图内部比较。Fig 5 caption 写 Humanoid 的动作空间是 $\mathbb R^{21}$，Table 2 写的是 24。结论都只基于 3 个 seed。
6. **成本没算全**：Table 1 只统计了多任务训练本身，不包括产生 545M transitions 所需的 240 次单任务训练；而且这些数据来自 TD-MPC2 自己的 buffer，分布对它有利。单任务训练的算力和墙钟时间，论文都没有给出数字（App A 只定性地说 wall-time 与 TD-MPC 相当）。
7. **推理开销**：每个控制步要做 6–8 轮、每轮约 512 条序列（population size，Table 8）的潜空间 rollout，外加 Q ensemble 计算。论文只说规划吞吐量提升了约 2×，没有给出控制频率或延迟，而这在真机上很关键。
8. **scaling 的证据有限**：只有 5 个尺寸、最多 80 个任务。在 1M 参数时 TD-MPC2 反而略低于 TD-MPC（Fig 7）；80 任务上从 48M 到 317M 只从 68.0 涨到 70.6，只有 30 任务的子集上还看得到明显增长。

## 在脉络中的位置

- **前驱**：TD-MPC（Hansen et al. 2022，不在本项目中），方法几乎完全继承自它。它和 MuZero 同属"不重建观测、只预测 reward/value"的路线。joint-embedding 目标来自 BYOL；policy prior 来自 [SAC](../../04_rl/sac/notes.md) 的 maximum-entropy；log 空间离散回归借鉴自 [DreamerV3](../dreamerv3/notes.md) 和 C51；Q ensemble 取两个的最小值来自 REDQ；规划器是 MPPI；SimNorm 来自 simplicial embeddings。作为对照，另一条路线是做重建的生成式 world model，在 imagination 里训练策略：[World Models](../world-models/notes.md)（用 CMA-ES 训练线性 controller）到 [Dreamer](../dreamer/notes.md)，再到 [DreamerV3](../dreamerv3/notes.md)（后两者训练 actor-critic）。
- **后继 / 影响**：
  - [SimDist](../../05_intersections/simdist/notes.md)：沿用 TD-MPC 系的 decoder-free 目标形式（sg latent 预测 + reward + value，但 reward/value 用 MSE 回归，value 是对 expert value 的纯回归、不做 TD learning，另加 BC 项）和 TD-MPC 实现的 MPPI；它引用的是 TD-MPC 原文，没有引用 TD-MPC2。它在仿真中预训练，上真机时只 finetune latent dynamics，是本项目里把这条路线用到 sim-to-real 最直接的一篇。
  - [Dreamer 4](../dreamer4/notes.md) 把 TD-MPC2 列为最强的 world model agent 之一；[V-JEPA 2](../vjepa2/notes.md) 也是"joint-embedding 潜空间加 MPC（CEM）"，但不用 reward，而是用到目标图像 embedding 的距离作为 energy；[Robotic World Model](../../05_intersections/robotic-world-model/notes.md) 在 related work 中把它归为 MPC 路线，而 RWM 自己选择用长程 autoregressive rollout 训练 PPO。两篇综述（[RL overview](../../04_rl/survey-rl-overview/notes.md) 有 TD-MPC2 专节；[WM for robot learning](../survey-wm-robot-learning/notes.md)）都把它作为代表方法。
- **与本项目其他主题的联系**：
  - **RL**：它是"decision-time planning + TD learning"的典型，本质是 actor-critic 加一个潜空间规划器。多任务部分是没有 conservatism 的 offline RL，可以对照 [IQL](../../04_rl/iql/notes.md) 的保守性处理；App J 则是把不确定性惩罚放到了规划阶段。
  - **TTT**：MPPI 是推理时在动作空间里做优化，不更新权重，属于 test-time compute，不算 TTT。few-shot finetune 是靠 reward 的在线适应。以下是我的推测，论文没有做：joint-embedding 一致性损失不需要 reward，可以像 [PAD](../../05_intersections/pad/notes.md) 和 [TTT](../../02_test_time_training/ttt-self-supervised/notes.md) 那样作为部署时的 self-supervised 辅助任务来更新 encoder/dynamics。
  - **Real2sim2real**：论文本身不涉及 domain randomization 或 system identification。与 [RMA](../../03_real2sim2real/rma/notes.md) 相比，它缺少一个从历史推断环境参数的模块；[DayDreamer](../../05_intersections/daydreamer/notes.md) 展示了 Dreamer 在真机上 online 学习，而 TD-MPC2 论文本身没有真机实验，这条路线（TD-MPC 式模型 + MPPI）的真机验证只能看后续工作（如 SimDist）。

## 前置知识

1. MDP、TD learning、target network（EMA）、Q ensemble 与 overestimation bias（REDQ）。
2. SAC / maximum-entropy RL，以及 actor-critic 中 actor 如何对 Q 求梯度。
3. MPC 与基于采样的轨迹优化（CEM / MPPI）：elite、warm-start、receding horizon。
4. latent dynamics model 和 multi-step latent rollout 损失；BYOL/SPR 式带 stop-grad 的 self-predictive 表示学习。
5. 把回归写成分类（C51 / two-hot / DreamerV3 的 symlog 思路），以及 LayerNorm、softmax 的数值性质。
6. 建议先读 TD-MPC 原文，再对照 Dreamer 系列，理解 generative 与 implicit world model 的区别。

## 复现可行性

- **代码**：官方仓库 https://github.com/nicklashansen/tdmpc2，MIT 许可，已确认存在。数据集、300+ checkpoints（含 12 个多任务模型）在 https://tdmpc2.com 。环境：DMControl（含作者新增的 11 个任务）、Meta-World、ManiSkill2、MyoSuite，全部是开源仿真。
- **算力**：论文只报告了多任务训练开销（Table 1，单张 RTX 3090）：1M 3.7、5M 4.2、19M 5.3、48M 12、317M 33 GPU days。单任务训练的时间没有报告。仓库 README（不是论文）写的要求是：单任务 online RL 需 ≥12GB 内存，推荐 ≥8GB 显存；在 80 任务数据集上做多任务 offline RL 需 128GB 内存；训练 317M 模型需 ≥24GB 显存。
- **最小可复现实验**（按成本从低到高）：
  1. 下载 80 任务的 19M checkpoint 并评估，核对 Table 1 的 57.1。在同一 checkpoint 上关闭 MPC、只用 policy prior 采样，检验 Fig 9 "规划 > 纯 policy" 的趋势（默认的 planning + policy 为 54.2，只用 policy 为 42.2；53.7 对应图中单独的 "Planning" 变体）。
  2. 单任务训练，按 README 的命令 `python train.py task=<task> steps=<N>`：选 walker-walk 和 cheetah-run，各 3 个 seed、1M 步，与 Fig 12 的曲线对比。再加一组去掉 SimNorm 或改成连续回归的消融，观察梯度范数（对照 Fig 21）和最终 return。
  3. 有余力的话，跑 dog-run 验证高维稳定性（Fig 5）。论文里 Dog 任务跑 14M 步，是单任务里预算最长的一档（与 Humanoid、Pick YCB 相同）。
- **预计难点**：MyoSuite 和 ManiSkill2 的环境版本依赖比较重；545M transitions 的数据集需要大内存；Fig 9、Table 1、Table 10 里同一配置的数字不一致，复现时要先确定对照哪一个；3 个 seed 的方差在难任务上很大（例如 Pick YCB 的曲线抖动明显）。

## 自测问题

1. TD-MPC2 没有 decoder，是什么阻止它的潜空间 collapse 成常数？三项损失各自给 encoder 提供了什么梯度信号？
2. 为什么 MPPI 只用 $H=3$ 就能近似无限时域的目标？如果 $Q$ 系统性高估，规划会出什么问题？App J 的惩罚项是怎么对冲这一点的？
3. 离散回归（log 空间、101 bins、soft CE）为什么能让同一套超参数跨任务通用？为什么 latent 预测那一项不也改成离散回归？
4. SimNorm 和 VQ-VAE 的离散 code 有什么关系？温度 $\tau\to\infty$ 和 $\tau\to0$ 分别退化成什么？它为什么能抑制梯度爆炸？
5. 多任务训练时，action masking 如果不屏蔽无效维度上的熵，会让 policy prior 出现什么错误行为？用 545M 条混合质量数据训练，为什么在原理上属于 offline RL？可能出现什么 extrapolation 问题？
