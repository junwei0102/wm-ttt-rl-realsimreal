# Simulation Distillation: Pretraining World Models in Simulation for Rapid Real-World Adaptation

> arXiv [2603.15759](https://arxiv.org/abs/2603.15759) · RSS 2026（已在 RSS XXII 论文集 p017 核实；arXiv v2 为 2026-05-12，源码里没写会议名）· 主题：05_intersections · 层级：frontier
> 代码：https://github.com/CLeARoboticsLab/simdist （论文摘要只给了项目页，仓库链接是从项目页找到的；已确认仓库存在，MIT 协议，README 写着目前只放出了 Go2 四足部分，manipulation 部分 "coming soon"）· 笔记依据：`source/*.tex`（数字/公式）+ `paper.pdf`（图）

## 一句话

SimDist 先在仿真里用 privileged expert、中间 checkpoint 和动作扰动生成大规模、带 dense reward 和 value 标签的数据，用这些数据预训练一个 TD-MPC 式的 latent world model（encoder、dynamics、reward、value、policy prior）。部署到真机时只 finetune latent dynamics，其余组件全部冻结，再用 MPPI 在线规划。这样真实世界里的适应就只剩一个监督学习问题：supervised system identification。

## 问题设定

- **形式化**（`source/sections/prelims.tex`）：真实系统是 POMDP $(\mathcal S,\mathcal A,\mathcal O,p,r,\gamma)$，机器人只看得到原始观测 $o_t$（图像、height map、本体感知），目标是最大化 $\mathbb E[\sum_t\gamma^t r(s_t,a_t,s_{t+1})]$。真机上很难算 dense reward，因为底层状态测不到。
- **唯一的额外假设**（Assumption 1）：有一个近似的物理仿真器 $p_{\mathtt{sim}}(\cdot|s_t,a_t)$，在仿真里能读到 privileged state $s_t$。
- **输入/输出**：输入是原始观测历史和候选动作序列。输出是 MPPI 规划出来的动作，manipulation 是 6 维末端相对位姿加夹爪，控制频率 5 Hz；四足是 12 个关节的位置目标，控制频率 50 Hz。
- **之前的方法卡在哪**（`intro.tex`、`related.tex`）：
  1. 在真实世界做 model-free finetune（RLPD、SAC 类方法）时，表征、reward、value、动力学和动作选择都搅在一个端到端更新里，还要从少量数据里解决长时程 credit assignment，所以经常 collapse 或出现 catastrophic forgetting。
  2. 直接在真实世界训练一个能用来规划的 action-conditioned world model，需要覆盖面很广的数据，量大到不现实。规划器会主动去找并利用模型出错的区域。
  3. 基于物理模型的 system identification 或 neural physics engine，通常需要物体位姿、接触标签等低维状态。在部分可观测、接触丰富的场景里，这些量拿不到。
  4. 视频生成类 world model 通常是在接近专家的窄动作分布上训练的，只能复现示范中已有的行为。

## 核心方法

**1. 模型结构**（`prelims.tex` Eq.1，Fig.3）

$$
\begin{aligned}
&z_t=E_\theta(o_t),\quad h_t=C_\theta(o_{t-H:t-1},a_{t-H:t-1}),\quad \hat z_{t+1:t+T}=f_\theta(z_t,a_{t:t+T-1},h_t)\cr
&\hat r_{t:t+T-1}=R_\theta(\hat z_{t:t+T},a_{t:t+T-1}),\quad \hat v_{t+1:t+T}=V_\theta(\hat z_{t:t+T}),\quad \hat a_{t:t+H}=\pi_\theta(z_t,h_t)
\end{aligned}
$$

与 TD-MPC / Dreamer 这类逐步 world model 相比有三处工程差异（`method.tex` §IV-C；论文引用的是 2022 年原版 TD-MPC，没有引用 TD-MPC2，本项目相关笔记见 [TD-MPC2](../../01_world_model/tdmpc2/notes.md)）：
- **Minimal history**：history encoder 只接收本体感知和动作的历史，外部感知（图像或 height map）只取最新一帧，目的是降低规划延迟。
- **Chunked prediction**： $f_\theta$ 是一个带 causal mask 的 transformer，候选动作序列与 history token 做 cross-attention，**一次前向就输出全部 $T$ 步**，不做逐步自回归 rollout。这个设计借鉴了 AnyCar。
- **Seq2seq reward/value**：reward 和 value 头也是 transformer，对整条预测轨迹 $\hat z_{t:t+T}$ 做 attention，不是逐步 MLP。因此 value 实际上以整条轨迹为条件，不是 Markov 的 $V(z)$。

**2. 仿真数据生成**（Alg.2，`appendix.tex` §A–C）
- 先用 privileged state 训练 expert $\pi^e$ 和它的 critic $V^e$。四足用 IsaacLab 里的 PPO 并加 domain randomization；manipulation 直接沿用 Yin et al. 2026 的训练流程。训练过程中的中间 checkpoint $\lbrace\pi^k\rbrace$ 也保存下来。
- 每个并行环境有自己的对角动作噪声 $\Sigma_j$， $\sigma_i\sim\mathcal U[\sigma_i^{\min},\sigma_i^{\max}]$。每次 reset 时以 0.5 的概率换成一个次优 checkpoint，并在**连续时间窗口**内加 Gaussian 扰动： $a_t=\pi^k(s_t)+\varepsilon_t$。
- 每一步记录 $(o_t,a_t,b^e_t,r_t,v_t)$。其中 $r_t$ 由 privileged state 算出， $v_t=V^e(s_t)$。**注意 value 标签始终来自 expert 的 critic，与当时实际执行的是哪个策略无关**。 $b^e_t=1$ 当且仅当这一步动作来自 expert 且没有加噪声。
- 这样做的目的是覆盖 expert manifold 之外的失误和恢复行为。规划器会在分布外采样，如果模型没在这些区域见过数据，就会被规划器利用。

**3. 预训练损失**（`method.tex` Eq.2）

$$
\mathcal L_t^{\mathtt{sim}}(\theta)=\sum_{i=0}^{T}\Big(\Vert\hat z_{t+i+1}-\mathtt{sg}(E_\theta(o_{t+i+1}))\Vert_2^2+c_1(\hat r_{t+i}-r_{t+i})^2+c_2(\hat v_{t+i+1}-v_{t+i+1})^2+c_3\thinspace\mathbb 1_e(a_{t+i})\Vert\hat a_{t+i}-a_{t+i}\Vert_2^2\Big)
$$

- 第 1 项是 latent consistency，与 TD-MPC 相同，目标加 stop-grad。第 2 项回归 reward。第 3 项把 $V^e$ 蒸馏到 latent 上，是**纯回归，不做 TD bootstrap**。第 4 项只在"干净 expert 动作"上做 BC，得到 policy prior。 $c_{1:3}$ 按各自目标的数值范围做归一化。
- 由于行为已经交给 privileged expert 生成，这个目标是**平稳**的监督学习，不需要 temporal-difference learning。这是它与 Dreamer / TD-MPC 在线 MBRL 的根本区别。
- **没有像素重建项**（§IV-A）。一是数据多样性已经足以逼出鲁棒表征；二是 sim-to-real 需要大量视觉随机化，重建目标会迫使 latent 去编码被随机化掉的纹理和光照。

**4. 真实世界适应**（`method.tex` Eq.3，Alg.1）

$$
\mathcal L_t^{\mathtt{real}}(\theta)=\sum_{i=0}^{T}\Vert\hat z_{t+i+1}-\mathtt{sg}(E_\theta(o_{t+i+1}))\Vert_2^2,\qquad C_\theta,E_\theta,R_\theta,V_\theta,\pi_\theta\ \text{冻结，只更新}\ f_\theta
$$

- 核心论点（§IV-B）：任务的全局结构，比如 peg 离孔有多远、是否对齐，基本不受低层动力学 gap 影响。reward 和 value 只需要在真实状态上给出**正确的排序**，不要求数值准确；真正对 sim-to-real gap 敏感的是把动作映射到未来状态的 $f_\theta$。
- 冻结 $E_\theta$ 有两个作用：它为 Eq.3 提供固定不变的回归目标；它也把新的 dynamics 钉在 $R_\theta$、 $V_\theta$ 训练时所用的 latent 空间里，不会漂走。
- Eq.3 的监督信号只来自未来 $T$ 步的真实观测（经冻结的 $E_\theta$ 编码成 latent 目标），**不需要真实 reward 标签**，也能用任意来源的轨迹训练（示教、play data、旧策略 rollout），所以作者称它为一种简单的 off-policy RL。
- 迭代流程：用 MPPI 采集 $M$ 条 rollout，加入 $\mathcal D_{\mathtt{real}}$，finetune $f_\theta$ 直到收敛，再重复。manipulation 每 20 个 episode 更新一次。

**5. 规划**（`prelims.tex`）：MPPI 候选序列的回报为

$$
\mathcal R(a_{t:t+T-1})=\gamma^T\hat v_{t+T}+\sum_{s=t}^{t+T-1}\gamma^{s-t}\hat r_s
$$

一部分候选序列用 $\pi_\theta$ 的输出加噪声来 warm-start（与 TD-MPC 一样）。实现直接用 TD-MPC 的 MPPI（elites / temperature / 多轮迭代），正文只说按预测回报做 importance weighting，没有写出权重公式（原文未报告）。manipulation 的参数是 250 条候选 + 100 条 policy 样本、3 轮迭代、64 elites、temperature 0.4（附录 Table III）；四足是 450 + 22、8 轮、64 elites、temperature 0.25（Table VIII）。规划窗口也很短：manipulation 是 $H=T=5$ 步、5 Hz，即 1 s；四足是 $H=T=25$ 步、50 Hz，即 0.5 s。更远的回报全靠冻结的 $\hat v$。

## 关键实验与证据

**真实世界主结果**（Fig.4）。manipulation 用 UR5e 做 Peg Insertion 和 Table Leg 两个任务，初始条件分 Narrow 2×2 cm 和 Wide 35×35 cm 两档，每个点测 20 次，45 s 内完成算成功。四足用 Go2 做 Slippery Slope（PTFE 斜面）和 Foam（5 cm 记忆海绵）两个任务，每个任务 3 个速度 × 5 次。下表 manipulation 的数字是**读图估计的近似值**：

| 任务（最终数据量） | SimDist 起点→终点 | SimDist+BC | SGFT | IQL | RLPD |
|---|---|---|---|---|---|
| Peg Wide（120 ep） | ≈0.30→0.85 | ≈0.90 | ≈0.35 | ≈0.35 | 20 ep 后降到 0 |
| Table Leg Wide（100 ep） | ≈0.10→0.80 | ≈0.90 | ≈0.40 | ≈0 | ≈0 |
| Peg Narrow（100 ep） | ≈0.25→0.85 | ≈0.90 | ≈0.40 | ≈0.25 | ≈0.35 |
| Table Leg Narrow（80 ep） | ≈0.20→0.80 | ≈0.95 | ≈0.40 | ≈0.25 | ≈0.15 |

Diffusion Policy 和 $\pi_{0.5}$ 各用 100 条真实示教训练，另有加入仿真数据 co-train 的版本，成功率大致都在 0–0.2 之间，图中是虚线。正文的总结是 SimDist "typically reaches scores around 2× higher than any baseline"；初始范围越宽，与基线的差距越大。Fig.6 显示成功吞吐量（success/min）比 zero-shot 提高约 1.5–2×。

**四足精确数字**（附录 Table IX，`tables/qped_all_results.tex`）。Slippery Slope 使用 35.7 分钟真实数据，Foam 使用 32.1 分钟：
- Slippery 三个速度 0.1/0.3/0.5 m/s：SimDist 成功 4/5、5/5、5/5；zero-shot 预训练模型全部 0/5；IQL 和 RLPD 全部 0/5；单步 BC 为 2/5、1/5、0/5。
- Foam 三个速度 0.2/0.7/1.2 m/s：SimDist 全部 5/5（前进 3.00±0.00 m）；预训练模型 3/5、2/5、0/5；IQL 1/5、2/5、3/5；RLPD 在适应过程中让机器人失稳，没有报告结果。

**机制证据**：
- Fig.5：同一初始条件下，成功轨迹的预测 value 基本持续上升（有小幅波动，并非严格单调）；失败轨迹在 peg 掉落时 value 陡降。说明冻结的 $E_\theta$ 和 $V_\theta$ 在真机上仍能区分好坏轨迹。
- Fig.8a：在 held-out 的真实 Slippery 轨迹上，latent dynamics loss 从 0.076 降到 0.019。
- Fig.8c/d：finetune 后的模型能预测前左脚打滑，MPPI 选出的 plan 也随之改变。
- Fig.9：用一个辅助 probe 从冻结的 $z_t$ 重建真实图像，结果可辨认。这个 probe 是在全部真实数据上单独训练的，只用于分析；world model 本身不用重建损失训练。

**消融**：
- Fig.10，只在真机 Peg 任务上做。解冻 encoder 后成功率直接跌到 0，因为 $R/V$ 收到的是分布外的 latent。解冻 value（按 Feng et al. 2023 的做法）会重新引入长时程 credit assignment，出现 catastrophic forgetting。
- Table I，在**仿真**中评估，指标为成功率 / 四足平均 reward：

| 配置 | Peg | Table Leg | Quadruped |
|---|---|---|---|
| SimDist | 0.90 | 0.85 | 22.78 |
| 50% data | 0.72 | 0.61 | 22.73 |
| 10% data | 0.06 | 0.02 | 19.38 |
| Expert Data Only（等量） | 0.10 | 0.05 | 16.68 |
| MLP Reward+Value | 0.82 | 0.60 | 19.47 |
| + 像素重建 | 0.32 | 0.21 | **23.34** |

从这张表看，最关键的因素是数据的**规模和多样性**：只用 expert 数据时，manipulation 几乎降到零（0.10 / 0.05），四足 reward 从 22.78 降到 16.68。seq2seq 的 reward/value 头次之。重建损失对视觉 manipulation 伤害很大，对四足反而略有帮助（height map 输入，没有做视觉随机化）。

## 局限与论文没说的

**作者承认的**（`conclusion.tex`）：
- 冻结 reward 和 value 会给性能设上限。value 饱和或无法区分高水平真实轨迹时，要逼近接近完美的成功率，可能需要在 dynamics 之外选择性地更新 value。
- 没有使用互联网视频或更丰富的传感模态。
- 仍依赖仿真覆盖足够广。

**我读出来的**：
1. **"只适应 dynamics"成立有前提**。gap 必须主要体现在动力学上，而任务结构（观测到 latent 的映射、奖励排序）在仿真里是对的。如果视觉 gap 大到 encoder 本身失效，或者真实任务的目标与仿真 reward 不一致，冻结方案就无路可退，而解冻 encoder 已被证明会崩（Fig.10）。
2. **history encoder $C_\theta$ 也被冻结了**。按 [RMA](../../03_real2sim2real/rma/notes.md) 的思路， $C_\theta$ 恰恰最适合做隐式 system identification。论文没有比较"只调 $C_\theta$"或"调 $f_\theta$ 和 $C_\theta$"，也没有比较 residual、LoRA 这类更轻的 dynamics 适应方式。它 finetune 的是整个 $f_\theta$，那么 $f_\theta$ 在真实数据没覆盖到的状态区域会不会遗忘仿真先验，论文没有分析。
3. **对比口径不完全对等**。manipulation 里的 RLPD/IQL 用 sparse reward，每个 episode 更新一次，而 SimDist 带着从仿真迁移来的 dense reward/value。SGFT 在一定程度上补上了这一点（同样迁移 sim value）。RL 基线用的是 MLP 单步策略头，不做 chunking，也不做规划，推理算力远小于 MPPI。BC 基线用 100 条示教，SimDist+BC 只用 20 条。
4. **统计量偏少**。Fig.4 每条真机曲线都没有误差带或置信区间，论文也没说明重复了几次运行。四足每条曲线只有 3–4 个点。Table I 的消融都在仿真中做，衡量的是规划性能，**没有测这些消融对真实世界适应速度的影响**。
5. **文中有几处数字前后不一致**：
   - Fig.1 图注和引言的贡献列表写"15–30 分钟真实数据"（摘要里没有这个数字），但四足实际用了 32.1 和 35.7 分钟。
   - 四足部分正文写"100M simulated trajectories"，附录是 4096 env × 25000 steps，约 100M 个 data points，应指 transition 数。
   - 正文提到的"Peg Hard"在 Fig.7 图注里叫 Peg Wide。
   - Fig.4 中 Foam 的 SimDist 起点读图约 2.1 m，而 Table IX 预训练模型三个速度的平均前进距离是 (2.98+2.39+1.70)/3≈2.36 m。Fig.4 中 Slippery 的 RLPD 终点为 0，而 Table 中是 0.32–0.35 m。Fig.4 画了 Foam 的 RLPD 曲线，Table 却标注"不报告"。
6. **算力和规划频率**。四足规划在笔记本 RTX 4090M 上达到 50 Hz。manipulation 的推理硬件和预训练耗时都没有报告。真实世界 finetune 的学习率、步数、"converged"的判据也都没给。
7. **没测的 setting**：更长时程的任务（1 s 的规划窗口完全依赖 $V^e$ 的排序）；可变形物体和多物体任务；新任务或新 reward（reward 头冻结，无法在真机上换任务）；在线逐步更新（真正 TTT 式）与每 20 个 episode 批量更新的对比。

## 在脉络中的位置

- **前驱**：
  - 模型结构和规划器几乎直接继承 TD-MPC（论文引用的是 2022 年原版；同系的后续工作见 [TD-MPC2](../../01_world_model/tdmpc2/notes.md)）：latent consistency、reward/value/policy prior、MPPI warm-start。
  - 对"无重建 latent"的辩护是针对 [Dreamer](../../01_world_model/dreamer/notes.md) / [DreamerV3](../../01_world_model/dreamerv3/notes.md) 式重建目标说的。
  - manipulation 的仿真 expert 与数据流程来自 Yin et al. 2026（ICLR, *Emergent Dexterity Via Diverse Resets and Large-Scale Reinforcement Learning*）；四足 expert 用 IsaacLab 里的 [PPO](../../04_rl/ppo/notes.md) + [domain randomization](../../03_real2sim2real/domain-randomization/notes.md) 训练，论文引用的是 Lee et al. 2020 和 IsaacLab，没有引用 [legged gym](../../03_real2sim2real/legged-gym/notes.md)，但两者属于同类的大规模并行 PPO 流程。
  - "把仿真 value 迁到真机"的思路来自 SGFT（Yin et al. 2025，arXiv 2502.02705），chunked dynamics 借鉴 AnyCar。
  - "只 finetune world model 的一部分"与 Feng et al. 2023 *Finetuning Offline World Models in the Real World* 直接对比（后者会更新 value）。
- **后继 / 影响**：论文较新（2026-03），本项目 related_work 里还没有建立在它之上的工作。
- **与本项目其他主题的联系**：
  - *World model 当 simulator/规划器*：[Robotic World Model](../robotic-world-model/notes.md) 用 world model 替代仿真器，在 imagination 中跑 PPO；SimDist 反过来**用仿真器训练 world model**，真机上做 MPC 而不做策略优化。[DayDreamer](../daydreamer/notes.md) 在真机上从零训练 Dreamer，SimDist 把表征、reward、value 的 bootstrap 全部挪到仿真里完成。[Ctrl-World](../ctrl-world/notes.md)、[WMPO](../wmpo/notes.md)、[World-Gymnast](../world-gymnast/notes.md) 是像素级生成式 world model 加 VLA 的路线，与 SimDist 的"不预测像素、在宽动作分布上训练"形成对照。
  - *TTT / TTA*：Eq.3 就是在部署期用 self-supervised 预测损失更新模型，监督信号来自未来观测，不需要标签，与 [TTT](../../02_test_time_training/ttt-self-supervised/notes.md) 和 [PAD](../pad/notes.md) 同源。区别在于，PAD 更新的是共享 encoder；SimDist 的消融（Fig.10，真机 Peg）显示在它的 world model 里**解冻 encoder 会崩**，应该只更新 dynamics。它的更新方式是批量迭代，不是逐样本在线更新。
  - *Real2sim2real*：它不做 real-to-sim 重建（对照 [RialTo](../../03_real2sim2real/rialto/notes.md)、[PhysTwin](../../03_real2sim2real/phystwin/notes.md)），而是在 **latent 空间里做 system identification**。对照 [SimOpt](../../03_real2sim2real/simopt/notes.md) 调仿真参数分布，[ASAP](../../03_real2sim2real/asap/notes.md) 学 delta action 再回到仿真训练，[RMA](../../03_real2sim2real/rma/notes.md) 用历史 encoder 在线隐式辨识、不做梯度更新。
  - *RL*：基线是 RLPD（SAC 系，参见 [SAC](../../04_rl/sac/notes.md)、[HIL-SERL](../../04_rl/hil-serl/notes.md)）和 [IQL](../../04_rl/iql/notes.md)。SimDist 的卖点是把真实世界里 TD 学习的不稳定性完全挪到仿真中。

## 前置知识

1. **TD-MPC 系 latent world model**：latent consistency loss 加 stop-grad、reward/value head、policy prior，以及用 MPPI/CEM 在 latent 空间规划（先读 [TD-MPC2](../../01_world_model/tdmpc2/notes.md)）。
2. **MPC 与 MPPI**：采样、按回报做指数加权、elites，以及短窗口加 terminal value 的回报估计。
3. **privileged teacher 的 sim-to-real 流程**：PPO + domain randomization + teacher-student（[legged gym](../../03_real2sim2real/legged-gym/notes.md)、[RMA](../../03_real2sim2real/rma/notes.md)）。
4. **Value function 的含义**： $V^{\pi^e}$ 与行为策略价值的区别，以及规划时为什么"排序正确"就够用。
5. **Offline-to-online RL 基线**：RLPD、[IQL](../../04_rl/iql/notes.md)，以及它们在 finetune 中出现 collapse 或 forgetting 的原因。
6. **system identification 与 TTT 的自监督适应**（[TTT](../../02_test_time_training/ttt-self-supervised/notes.md)、[PAD](../pad/notes.md)）。
7. Transformer 的 cross-attention 和 causal mask（用于理解 chunked dynamics）。

## 复现可行性

- **代码**：https://github.com/CLeARoboticsLab/simdist（MIT）。按 README，Go2 四足的 expert 训练、数据生成、预训练、真机部署和适应流程都已开源，IsaacLab 以 submodule 形式引入；manipulation 部分标注 "coming soon"。README 里没有写 GPU 要求。
- **算力**（仅四足有报告，`appendix.tex` §C）：
  - expert：PPO，4096 个并行环境 × 5000 iter × 24 steps，共 490M steps，耗时未报告。
  - 数据生成：4096 env × 25000 steps，约 100M 个数据点，其中 55.7% 是干净 expert 动作，单卡 RTX 4500 Ada 约 7 h。
  - 预训练：2 epoch，batch 512，约 3.69e5 步，Adam，lr 从 2e-4 cosine 退火到 1e-4，单卡 RTX 4500 Ada 约 28 h。
  - 部署：笔记本 RTX 4090M 上 50 Hz 规划。
  - manipulation：每个任务 100k 条轨迹，约 36% 是干净 expert 动作；2 epoch，batch 256，约 200k 步；ResNet-18 编码 3 路 224×224 图像，latent 64 维。耗时和 GPU 未报告。
- **硬件与数据**：真机结果需要 Unitree Go2（加 PTFE 斜面或记忆海绵），或者 UR5e 加三路相机；20 条遥操作示教只有 SimDist+BC 和在线 RL 基线用到，纯 SimDist 不用示教。
- **最小可复现实验（sim-to-sim，不需要真机）**：
  1. 在 IsaacLab Go2 上按官方流程训练 expert、生成数据并预训练。数据可以降到 50%：Table I 显示四足在 50% 数据下 reward 为 22.73，几乎不掉。
  2. 构造一个超出 DR 范围的"伪真实"环境作为目标域。训练时摩擦在 [0.2, 1.2]，基座附加质量在 −1 到 +3 kg（Table IV），目标域可以取摩擦 0.05–0.1 或附加质量 +6 kg。
  3. 在目标域里用 MPPI 采约 30 分钟（按 50 Hz 算约 9 万步）的数据，按 Eq.3 只 finetune $f_\theta$。
  4. 比较 zero-shot、只调 $f_\theta$、解冻 $E_\theta$、解冻 $V_\theta$ 四种设置的前进距离和 held-out latent loss，对照 Fig.8a 中 0.076→0.019 的趋势。
  5. 延伸：加一组"只调 $C_\theta$"的对照，检验 RMA 式适应能否替代。
- **预计难点**：数据生成要调噪声区间，四足是扰动 1–50 步、间隔 25–500 步；expert 与 checkpoint 的比例要调。MPPI 的实时性要求 chunked transformer 推理速度足够快。manipulation 在代码放出前需要自己复刻 Yin et al. 2026 的 expert。真实世界 finetune 的超参论文没有给，需要自己调。

## 自测问题

1. 为什么 reward、value、encoder 可以冻结，而 dynamics 必须 finetune？"value 只需要给出正确排序"这一点，在 MPPI 的回报公式 $\gamma^T\hat v_{t+T}+\sum\gamma^{s-t}\hat r_s$ 里是怎么体现的？
2. 如果在 Eq.3 里也更新 $E_\theta$，回归目标 $\mathtt{sg}(E_\theta(o_{t+1}))$ 和下游的 $R_\theta$、 $V_\theta$ 各会出什么问题？与 Fig.10 的结果对应起来解释。
3. 数据生成为什么要混入中间 checkpoint 和"连续时间窗口"的动作噪声？BC 项为什么要用 $\mathbb 1_e$ 做 mask？value 标签为什么用 $V^e(s_t)$，而不是行为策略自身的 Monte Carlo 回报？
4. 与 TD-MPC2 相比，SimDist 的预训练去掉了哪一类学习，为什么说它的目标是"平稳"的？它的真实世界适应又在什么意义上算 off-policy RL？
5. 在 Table I 中，重建损失对 manipulation 伤害很大（0.90→0.32），对四足却略有帮助（22.78→23.34）。用两个平台观测模态和随机化方式的差异来解释这个现象。
