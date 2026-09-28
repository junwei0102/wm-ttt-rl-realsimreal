# Intersections (cross-topic)

> 本线收录 8 篇：foundational 3 篇、milestone 3 篇、frontier 2 篇。收录标准是同时跨两条以上主线。
> 文中数字都取自各篇 `notes.md`，笔记里的数字能在对应论文的 `source/` 或 PDF 中找到出处；读图估计的值会单独注明。
> 链接约定：本线笔记写作 `./<slug>/notes.md`，其他主线写作 `../0x_<topic>/<slug>/notes.md`。

---

## 1. 这条线在解决什么问题

在真实机器人上做 RL，交互昂贵、有危险、难以 reset；手工搭建的物理仿真器又有 sim-to-real gap（reality gap），每换一个场景都要重新建模。这条交叉线要回答的是：能否用**从数据学出来的模拟器**（world model，即 learned simulator）来训练、评估和适应策略？能否在部署时、拿不到 reward 的情况下继续修正策略或模型？

四条主线在这里各自提供一块：

- world model：模拟器本身；
- RL：把模拟器转化为策略提升的优化器；
- real2sim2real：模拟器与真实世界之间存在 gap 这个问题框架，以及 domain randomization、system identification 这类校准手段；
- TTT：部署期用自监督信号更新参数的机制。

整条线绕不开的风险是 **model exploitation**：策略会去利用模拟器（以及学出来的 reward）里的错误。这就是 learned simulator 里的 "gap"。

---

## 2. 发展脉络

**前史（不在本线，见 01–04）**：Dyna 提出"真实数据学模型、模型里训策略"；[Domain Randomization](../03_real2sim2real/domain-randomization/notes.md)（2017）；[World Models](../01_world_model/world-models/notes.md)（2018）；[SimOpt](../03_real2sim2real/simopt/notes.md)（2018，用真实轨迹调仿真参数分布）；[TTT](../02_test_time_training/ttt-self-supervised/notes.md)（2019，测试时自监督更新）；[Dreamer](../01_world_model/dreamer/notes.md)（2019，latent imagination）；[RMA](../03_real2sim2real/rma/notes.md)（2021，部署时从历史推断 extrinsics，不做梯度更新）；[legged gym](../03_real2sim2real/legged-gym/notes.md)（2021，大规模并行 PPO）。

```
前史(01-04): Dyna · DR · World Models · SimOpt · TTT · Dreamer · RMA · legged gym
                                 │
[部署期适应]   PAD (2020) ────────┼──────────────────────────────────┐
                                 │                                  ├─→ SimDist (2026)
[latent/低维 WM] DayDreamer (2022) ──→ RWM (2025) ──────────────────┘   仿真预训练 WM + 真机只更新 dynamics
                                 │
[视频 WM]      UniSim (2023) ──→ Ctrl-World (2025) ──→ WMPO (2025) ──→ World-Gymnast (2026)
                                 评估 + filtered SFT     WM 内 GRPO       WM 内 GRPO + VLM reward
                                                                          + test-time RL + Dyna（接回[部署期适应]）
```

| 年份 | 工作 | 分支 | 核心做法 | 相对前一步补上了什么 | 留给下一步的问题 |
|---|---|---|---|---|---|
| 2020 | [PAD](./pad/notes.md) | 部署期适应 | 训练时 RL 目标与 inverse dynamics 自监督任务共享 encoder；部署时拿不到 reward，每步只用自监督损失更新 encoder，policy head 冻结 | [DR](../03_real2sim2real/domain-randomization/notes.md) 要提前预料变化；用 reward fine-tune 要在部署环境设计 reward；domain adaptation 要目标域数据；原版 [TTT](../02_test_time_training/ttt-self-supervised/notes.md) 只做了图像分类 | 自监督任务靠人挑；没有 reward 时察觉不到性能在变差（CRLMaze 上 IDM (PAD) 让训练环境从 585 掉到 −416） |
| 2022 | [DayDreamer](./daydreamer/notes.md) | latent WM | DreamerV2 加异步 actor/learner，不用仿真器，直接在 4 台真机上从零在线学习 | model-free 真机 RL 样本太贵（A1 上 SAC 学不会走）；仿真有 reality gap。A1 约 1 h 学会行走 | 每个任务从零开始；要做 reward 工程；"adaptation" 就是带 reward 继续训练，耗时 10 分钟到 5 小时；单次运行，没有消融 |
| 2023 | [UniSim](./unisim/notes.md) | 视频 WM | 用互联网、人类视频、机器人、仿真等异构数据训练 5.6B 动作条件视频扩散模型当通用 simulator；在其中用 REINFORCE 训练 VLA | 以往的 world model 每个系统单独学、知识不能跨系统共享，像素级 WM 多停留在游戏或仿真里 | 真机只有定性结果；RL 成功率是在 UniSim 自己生成的视频里人工判的，无法排除 model exploitation；512 TPU-v3 × 20 天；没有代码 |
| 2025 | [Robotic World Model](./robotic-world-model/notes.md) | 低维 WM | GRU world model 用自回归（AR）多步损失训练，rollout 100 步以上仍稳定，可以直接跑 PPO；策略 zero-shot 部署到 ANYmal D 和 G1 | teacher forcing 训练的模型有 compounding error；Dreamer/MBPO 只用短 rollout，撑不起 PPO 需要的长 on-policy 轨迹 | 硬件实验中真实数据没有进入训练环路（WM 只用了 Isaac Lab 数据）；只有低维本体观测；没有不确定性估计 |
| 2025 | [Ctrl-World](./ctrl-world/notes.md) | 视频 WM | 基于 SVD 的多视角、逐帧动作条件、带 pose-conditioned memory 的 WM；让现成 VLA 在想象中闭环 rollout，用来给策略排名，再从想象中筛出成功轨迹做 SFT | UniSim 类单视角模型与需要腕部视角的现代 VLA 不兼容，接触处有幻觉；并给出了 world model 评估结果与真机成功率的定量对齐（回归线） | 系统性低估成功率（截距 −0.11）；改进只是 filtered BC，没有负样本，成败靠人工判定 |
| 2025 | [WMPO](./wmpo/notes.md) | 视频 WM | 用 policy 自己的真实 rollout（含失败）对齐视频 WM（policy behavior alignment），在其中用学到的 0/1 判别器做 on-policy GRPO | （与 Ctrl-World 同期，不是直接继承）filtered SFT 只学正样本；仿真里做 GRPO 要为每个场景搭仿真器；真机做 GRPO 太贵，也难从同一初始状态 reset | 没有任何定量消融；单轮训练中 WM 冻结在 base policy 的分布上；没有测量 WM 内成功率与真实成功率的差距 |
| 2026 | [World-Gymnast](./world-gymnast/notes.md) | 视频 WM + 适应 | WorldGym + GPT-4o 0/1 reward + GRPO；只需"初始帧 + 指令"就能构成训练环境；衍生出从测试帧出发的 test-time RL 和 Dyna 式 WM 更新 | WMPO 的 reward 判别器要按任务训练，主要评测在 Mimicgen 仿真；本文在 AutoEval 真机上直接对照 SFT 和 [SIMPLER](../03_real2sim2real/simpler/notes.md) 中的 RL | GPT-4o 判分准确率没测；真机只有 4 个任务；TTT 会过拟合单任务 |
| 2026 | [SimDist](./simdist/notes.md) | latent WM + 适应 | 用仿真 privileged expert、中间 checkpoint 和动作扰动生成大规模数据，预训练 TD-MPC 式 latent WM；真机上只 finetune latent dynamics（supervised system identification），其余冻结，用 MPPI 规划 | DayDreamer 在真机从零学表征、reward、value 太慢；RWM 依赖低维状态；真机 model-free finetune（RLPD/IQL）会 collapse。同时把 PAD 式的部署期自监督更新落实为"只更新 dynamics" | 冻结的 reward/value 给性能设了上限；按批迭代更新而不是在线更新；真机只跑一次，文中数字有几处前后不一致 |

**三个阶段的主题变化**

- **2020–2022：分头证明可行。** PAD 证明部署期不用 reward 也能适应；DayDreamer 证明 latent imagination RL 能直接在真机上跑。
- **2023–2025：模拟器变大、变通用，也开始被当作评估器。** UniSim 把模拟器从单任务 latent 模型推到多域视频模型。RWM 证明低维 learned simulator 足以支撑 PPO，并能 zero-shot 上真机。Ctrl-World 把视频 WM 定量地用作 policy evaluator，WMPO 在视频 WM 里做 on-policy RL。
- **2026：拿 WM 内 RL 与仿真器 RL 做真机对照，并把 WM 带进部署期。** World-Gymnast 做了真机对照，并加入 TTT 和 Dyna 两种部署期用法。SimDist 在仿真里预训练 WM，到真实世界只做 system identification。

---

## 3. 技术分类（taxonomy）

子方向沿用 scouting 阶段的划分。"本项目收录"一列链接到笔记；未收录的工作给出 arXiv ID。

| 子方向 | 要回答的问题 | 本项目收录 | 未收录的代表工作 |
|---|---|---|---|
| **A1. latent / 低维 world model 当 learned simulator 训练策略**（控制类任务） | learned model 能否替代真实交互或物理仿真，撑起 actor-critic / PPO？ | [DayDreamer](./daydreamer/notes.md)、[RWM](./robotic-world-model/notes.md)；另见 01 的 [Dreamer](../01_world_model/dreamer/notes.md)、[DreamerV3](../01_world_model/dreamerv3/notes.md)、[Dreamer 4](../01_world_model/dreamer4/notes.md) | RWM-U（2504.16680，带 epistemic uncertainty 的 offline MBRL，MOPO-PPO）；ReDRAW（2504.02252） |
| **A2. 视频 world model 当环境，对 VLA 做 RL 后训练** | 像素级生成模型能否替代仿真器做 on-policy RL？reward 从哪来？ | [UniSim](./unisim/notes.md)、[WMPO](./wmpo/notes.md)、[World-Gymnast](./world-gymnast/notes.md) | World-Env（2509.24948）、VLA-RFT（2510.00406）、RISE（2602.11075）、VLAW（2602.12063）、PlayWorld（2603.09030） |
| **B. world model 当 policy evaluator** | 想象中的成功率 / 排名与真机一致吗？ | [Ctrl-World](./ctrl-world/notes.md)；物理仿真版的对照是 03 的 [SIMPLER](../03_real2sim2real/simpler/notes.md) | WorldEval（2505.19017）、WorldGym（2506.00613）、Veo World Simulator（2512.10675）、GigaWorld-1（2607.02642）、闭环 benchmark World-in-World（2510.18135） |
| **C. 用视频 world model 生成数据** | 合成轨迹能否替代真机示教或 rollout？ | [UniSim](./unisim/notes.md)（hindsight relabeling）、[Ctrl-World](./ctrl-world/notes.md)（filtered SFT）；01 的 [Cosmos](../01_world_model/cosmos-wfm/notes.md) | DreamGen（2505.12705）、DreamDojo（2602.06949） |
| **D. real-to-sim(-to-real) 重建，用于 RL 和评估** | 显式重建 + 物理引擎是否比 learned simulator 更可靠、更可扩展？ | 主要在 03：[RialTo](../03_real2sim2real/rialto/notes.md)、[SplatSim](../03_real2sim2real/splatsim/notes.md)、[ACDC](../03_real2sim2real/acdc-digital-cousins/notes.md)、[SIMPLER](../03_real2sim2real/simpler/notes.md)、[X-Sim](../03_real2sim2real/x-sim/notes.md)、[PhysTwin](../03_real2sim2real/phystwin/notes.md)。本线里 WMPO 的 policy behavior alignment 和 World-Gymnast 的 Dyna 属于数据驱动的 real-to-sim | GaussGym（2510.15352）、PolaRiS（2512.16881）、Scaling Sim-to-Real VLA RL with Generative 3D Worlds（2603.18532） |
| **E. 用 learned model 做 sim-to-real 动力学对齐** | 能否只修正动力学，把表征、reward、value 留在仿真里学？ | [RWM](./robotic-world-model/notes.md)、[SimDist](./simdist/notes.md)；03 的 [ASAP](../03_real2sim2real/asap/notes.md) | ReDRAW（2504.02252，latent dynamics residual）、SGFT（2502.02705，把仿真 value 迁到真机） |
| **F. 部署期 test-time training / adaptation**（对象可以是 policy，也可以是 world model） | 部署时用什么信号、更新哪部分参数？ | [PAD](./pad/notes.md)（自监督，更新 encoder）、[SimDist](./simdist/notes.md)（自监督 latent 预测，只更新 dynamics）、[World-Gymnast](./world-gymnast/notes.md)（在 WM 中做 test-time RL）；02 的 [TTT](../02_test_time_training/ttt-self-supervised/notes.md)、[Tent](../02_test_time_training/tent/notes.md)；03 的 [RMA](../03_real2sim2real/rma/notes.md)（不做梯度更新） | Learning to Adapt（1803.11347）、EVOLVE-VLA（2512.14666）、TT-VLA（2601.06748）、WAM-TTT（2607.06988）、TTT on Video Streams（2307.05014） |

**换个角度：按"gap 在哪里被处理"来划分**

| 处理方式 | 代表 | 代价 |
|---|---|---|
| 不修模拟器，部署时修策略 | PAD | 只能修 encoder 能表达的那部分变化；没有 reward 时无法判断是否在变好 |
| 完全不用模拟器，直接在真机上学 | DayDreamer | 真机时间、硬件磨损、reward 工程 |
| 用 learned model 替代模拟器 | RWM、UniSim、WMPO、World-Gymnast | gap 变成"world model 与真实之间的误差"，风险是 model exploitation |
| 用模拟器训练 learned model，再用真实数据校准其中一部分 | SimDist（ReDRAW 同类） | 依赖仿真覆盖面足够广；冻结的部分设定了性能上限 |
| 用 learned model 评估，不直接优化 | Ctrl-World（评估部分） | 成功率估计有偏，偏差大小依赖物理类型 |

---

## 4. 论文总表

阅读顺序按学习路线排（第 6 节），不按年份排。

| 阅读顺序 | 论文 | 年份 | 层级 | 一句话 | 代码 | 笔记 |
|---|---|---|---|---|---|---|
| 1 | PAD: Self-Supervised Policy Adaptation during Deployment（2007.04309，ICLR 2021） | 2020 | foundational | 把 TTT 搬进视觉 RL：部署期只用 inverse dynamics 自监督损失，每步更新共享 encoder、冻结 policy head，不需要 reward | [有](https://github.com/nicklashansen/policy-adaptation-during-deployment)（PyTorch，DMControl） | [notes](./pad/notes.md) |
| 2 | DayDreamer: World Models for Physical Robot Learning（2206.14176，CoRL 2022） | 2022 | foundational | DreamerV2 加异步 actor/learner，不用仿真器，在 4 台真机上从零在线学习；A1 约 1 h 学会行走 | [有](https://github.com/danijar/daydreamer)（TF2） | [notes](./daydreamer/notes.md) |
| 3 | Robotic World Model（2501.10100，NeurIPS 2025 EWM Workshop） | 2025 | milestone | AR 多步训练的 GRU 神经模拟器，rollout 稳定到能直接跑 PPO，策略 zero-shot 上 ANYmal D / G1 | [有](https://github.com/leggedrobotics/robotic_world_model)（Isaac Lab，不支持 macOS） | [notes](./robotic-world-model/notes.md) |
| 4 | SimDist: Simulation Distillation（2603.15759，RSS 2026） | 2026 | frontier | 仿真大规模数据预训练 TD-MPC 式 latent WM；真机只 finetune dynamics（supervised system identification），加 MPPI 规划 | [有](https://github.com/CLeARoboticsLab/simdist)（目前只有 Go2 四足） | [notes](./simdist/notes.md) |
| 5 | UniSim: Learning Interactive Real-World Simulators（2310.06114，ICLR 2024） | 2023 | foundational | 用异构数据统一成"动作进、视频出"，训练 5.6B 视频扩散模型当真实世界 simulator，在其中训练 VLM 规划器和 VLA 的 RL 策略 | 无 | [notes](./unisim/notes.md) |
| 6 | Ctrl-World（2510.10125，ICLR 2026） | 2025 | milestone | 多视角、逐帧动作条件、pose-conditioned memory 的视频 WM：给 VLA 排名，并用想象中的成功轨迹 SFT，π0.5 从 38.7% 升到 83.4% | [有](https://github.com/Robert-gyj/Ctrl-World)（含 DROID checkpoint） | [notes](./ctrl-world/notes.md) |
| 7 | WMPO（2511.09515，ICLR 2026） | 2025 | milestone | 用 policy 真实 rollout（含失败）对齐视频 WM，在其中用 0/1 判别器 reward 做 on-policy GRPO；Mimicgen 平均从 33.6% 升到 57.6%（P=1280） | [有](https://github.com/WM-PO/WMPO)（OpenSora + verl） | [notes](./wmpo/notes.md) |
| 8 | World-Gymnast（2602.02454，ICLR 2026 World Models Workshop） | 2026 | frontier | WorldGym + GPT-4o 0/1 reward + GRPO；AutoEval 真机上最高是 SFT 的 18 倍（4%→72%），并做了 test-time RL 和 Dyna | [有](https://github.com/world-gymnast/world-gymnast)（checkpoint 是否发布未确认） | [notes](./world-gymnast/notes.md) |

---

## 5. 四条线如何在这里交汇

### 5.1 每篇论文连接了哪几条线

图例：● 表示这条线是论文的核心贡献所在，○ 表示用到了或作为对照，— 表示无关。

| 论文 | 01 WM | 02 TTT | 03 Real2Sim2Real | 04 RL | 怎么连接 |
|---|---|---|---|---|---|
| PAD | — | ● | ● | ● | 把 [TTT](../02_test_time_training/ttt-self-supervised/notes.md) 的 Y 形结构、joint training 和 TTT-Online 原样搬进 [SAC](../04_rl/sac/notes.md)/A2C。部署期的 IDM 标签就是 agent 自己执行的动作，所以是免费的。它的主要对手是 [DR](../03_real2sim2real/domain-randomization/notes.md)，并在真机 Kinova 上验证 sim-to-real |
| DayDreamer | ● | ○ | ○ | ● | [Dreamer](../01_world_model/dreamer/notes.md) 的 RSSM 加 imagination actor-critic，直接跑在真机上，靠"完全不用仿真"绕开 reality gap。部署期继续带 reward 做 online RL，论文称之为 adaptation，但它不是 TTT 意义上的自监督适应 |
| UniSim | ● | ○ | ○ | ● | 视频扩散 WM（与 [DIAMOND](../01_world_model/diamond/notes.md) 同属像素空间路线）作为纯数据驱动的 simulator，与 [RialTo](../03_real2sim2real/rialto/notes.md) 这类重建 digital twin 的路线相对；在其中用 REINFORCE 训练 VLA。它承认的"长程记忆缺失"指向 02 的 [TTT layers](../02_test_time_training/ttt-layers/notes.md)、[Titans](../02_test_time_training/titans/notes.md) |
| RWM | ● | — | ● | ● | 把 [legged gym](../03_real2sim2real/legged-gym/notes.md) 的物理引擎换成 GRU，照样开 4096 个并行 imagination 环境跑 [PPO](../04_rl/ppo/notes.md)。GRU 从 32 步历史推断动力学，作用类似 [RMA](../03_real2sim2real/rma/notes.md) 的 adaptation module |
| SimDist | ● | ● | ● | ● | 四条线全部用到：仿真 privileged expert 用 PPO + [DR](../03_real2sim2real/domain-randomization/notes.md) 训练；world model 是 [TD-MPC2](../01_world_model/tdmpc2/notes.md) 式的 latent consistency 加 MPPI；真机上用自监督的 latent 预测损失更新 dynamics，与 [PAD](./pad/notes.md) 同源；对照的 RL 基线是 RLPD、[IQL](../04_rl/iql/notes.md) |
| Ctrl-World | ● | ○ | ● | ○ | 视频 WM 被当作 policy evaluator，对应 03 中 [SIMPLER](../03_real2sim2real/simpler/notes.md) 的角色。用真机快照当初始帧，是一种不做 3D 重建的轻量 real-to-sim。改进环节是 filtered BC，还不算 RL，可对照 [HIL-SERL](../04_rl/hil-serl/notes.md) 的人工 reward 信号 |
| WMPO | ● | ○ | ○ | ● | 可以看作把 [SimpleVLA-RL](../04_rl/simplevla-rl/notes.md) 里的仿真器换成视频 WM，GRPO 用法见 [DeepSeek-R1](../04_rl/deepseek-r1/notes.md)。用 policy 的真实 rollout 微调 WM，是数据驱动的 real-to-sim；lifelong 版本交替更新 WM 与策略，结构上类似 [SimOpt](../03_real2sim2real/simopt/notes.md) |
| World-Gymnast | ● | ● | ● | ● | 在 WM 里做 GRPO（沿用 SimpleVLA-RL 的四个技巧），真机上直接对照 SIMPLER。它所说的 "TTT" 是从测试帧出发做 test-time RL，思路接近 [TTT-Discover](../02_test_time_training/ttt-discover/notes.md)；Dyna 式更新对应 SimOpt；用 Nano Banana 在初始帧里插干扰物，相当于像素层面的视觉 DR |

### 5.2 每条线在这里贡献了什么

**01 World Model：提供模拟器，这条线里用到四种模型族。**

1. RSSM latent：[Dreamer](../01_world_model/dreamer/notes.md) → DayDreamer。
2. 低维 GRU：RWM。
3. TD-MPC 式 decoder-free latent：[TD-MPC2](../01_world_model/tdmpc2/notes.md) → SimDist。
4. 视频扩散：[DIAMOND](../01_world_model/diamond/notes.md)、[Cosmos](../01_world_model/cosmos-wfm/notes.md)、[Genie](../01_world_model/genie/notes.md) → UniSim、Ctrl-World、WMPO、World-Gymnast。

01 线内部的"像素还是 latent"之争，到了这里变成更具体的问题：policy 能不能直接读 WM 的输出（WMPO 的论点），以及重建目标会不会伤害 sim-to-real（SimDist 的 Table I）。

**04 RL：提供优化器，也提供"不用 WM 会怎样"的对照。**

- 优化器从 imagination actor-critic（DayDreamer）、PPO（RWM）、REINFORCE（UniSim），发展到 GRPO + DAPO 技巧（WMPO、World-Gymnast）。GRPO 用组内归一化代替 critic，前提是 WM 能从同一个 $s_0$ 反复 reset。这件事在真机上很难做到，在 WM 里是免费的。
- SimDist 走另一条路：不做策略梯度，用 MPPI 规划，把所有 TD 学习都留在仿真里完成。
- 对照组包括：真机 model-free 方法（SAC、RLPD、[HIL-SERL](../04_rl/hil-serl/notes.md)），物理仿真里的 VLA RL（[SimpleVLA-RL](../04_rl/simplevla-rl/notes.md)、[π_RL](../04_rl/pi-rl/notes.md)），以及 Ctrl-World 的 filtered SFT（不是 RL）。

**03 Real2Sim2Real：提供问题框架和校准工具。**

这条线里的每一种做法，在 03 都能找到对应物：

| 本线的做法 | 03 中的对应物 |
|---|---|
| PAD、SimDist expert 训练里的随机化 | [DR](../03_real2sim2real/domain-randomization/notes.md) |
| WMPO 的 lifelong 更新、World-Gymnast 的 Dyna | [SimOpt](../03_real2sim2real/simopt/notes.md) |
| RWM 的 GRU 历史、SimDist 的 history encoder $C_\theta$ | [RMA](../03_real2sim2real/rma/notes.md) |
| RWM、SimDist 用 learned model 修动力学 | [ASAP](../03_real2sim2real/asap/notes.md) |
| Ctrl-World 的评估、World-Gymnast 的 baseline | [SIMPLER](../03_real2sim2real/simpler/notes.md) |

需要记住的一点是：learned simulator 并没有消除 gap，只是把 sim-to-real gap 换成了 world model 与真实之间的误差。Ctrl-World 的成功率回归线（ $y=0.81x-0.11$）就是这种误差的一次定量测量。

**02 TTT：提供部署期更新参数的机制，但本线里的 "TTT" 有三种不同含义。**

1. **自监督辅助任务**：PAD 更新 encoder；SimDist 用 latent 预测损失更新 dynamics。这与 [TTT (Sun et al.)](../02_test_time_training/ttt-self-supervised/notes.md) 同源，都需要辅助任务的梯度与主任务相关。
2. **测试实例上的 RL**：World-Gymnast 的 TTT 属于这一类，接近 [TTT-Discover](../02_test_time_training/ttt-discover/notes.md)。它的 VLM 判分是 reward，不是自监督信号，而且接受对单个任务的过拟合。
3. **带 reward 的持续在线训练**：DayDreamer 属于这一类，严格说不是 TTT。

另外，RWM 的 "inner/outer autoregression" 与 TTT 的 inner loop / outer loop **不是一回事**。UniSim 的 4 帧窗口和 Ctrl-World 的 memory retrieval 都是上下文内的记忆，不更新权重；02 的 [TTT layers](../02_test_time_training/ttt-layers/notes.md)、[Titans](../02_test_time_training/titans/notes.md)、[LaCT](../02_test_time_training/lact/notes.md) 提供了用权重当记忆的替代方案，目前还没有被用到这些 world model 上。

### 5.3 适应发生在哪一层

"在部署或真实数据上适应"是这条线最有研究空间的交叉点。把 8 篇放在一起，可以看出适应发生在不同的层上：

| 适应的对象 | 信号 | 需要 reward？ | 代表 | 已知结果 |
|---|---|---|---|---|
| policy 的 encoder | 自监督 IDM | 否 | PAD | 36 个测试环境中 31 个提升；CRLMaze 上 IDM 版本反而崩溃 |
| world model 的 dynamics | 自监督 latent 预测 | 否 | SimDist | 真机约 30 分钟数据；解冻 encoder 后成功率跌到 0（Fig.10） |
| world model 整体 | 真实 rollout 的生成损失 | 否（reward 另有来源） | WMPO lifelong、World-Gymnast Dyna | Dyna 后 close the drawer 达到 95%；WMPO 只在 StackThree 上跑了两轮 |
| policy（在 WM 里） | VLM reward | 是（learned） | World-Gymnast TTT | close the drawer 从 62% 升到 100%，其他任务下降（论文没给数字） |
| 全部组件 | 真实 reward + 重建 | 是 | DayDreamer | 约 10 分钟到 5 小时 |

---

## 6. 推荐学习路线

### 阶段 0：先修（来自 01–04，没读过就先补）

- [ ] [PPO](../04_rl/ppo/notes.md)：clipped surrogate、GAE，以及为什么 PPO 需要长 on-policy rollout
- [ ] [SAC](../04_rl/sac/notes.md)：off-policy actor-critic、replay buffer
- [ ] GRPO 的组内归一化 advantage（[DeepSeek-R1](../04_rl/deepseek-r1/notes.md)）；DAPO 的 dynamic sampling / Clip-Higher 在 VLA 上的用法（[SimpleVLA-RL](../04_rl/simplevla-rl/notes.md)）
- [ ] [Dreamer](../01_world_model/dreamer/notes.md) / [DreamerV3](../01_world_model/dreamerv3/notes.md)：RSSM、latent imagination、λ-return
- [ ] [TD-MPC2](../01_world_model/tdmpc2/notes.md)：latent consistency + stop-grad、MPPI
- [ ] [V-JEPA 2](../01_world_model/vjepa2/notes.md)：表征空间预测，EMA teacher 加 stop-gradient 防坍缩（阶段 1 第 2 题要用）
- [ ] [DIAMOND](../01_world_model/diamond/notes.md)：扩散 world model 里的 RL
- [ ] [TTT (Sun et al.)](../02_test_time_training/ttt-self-supervised/notes.md)、[Tent](../02_test_time_training/tent/notes.md)：测试时自监督更新、gradient correlation 条件
- [ ] [Domain Randomization](../03_real2sim2real/domain-randomization/notes.md)、[RMA](../03_real2sim2real/rma/notes.md)、[SimOpt](../03_real2sim2real/simopt/notes.md)：三种应对 sim-to-real gap 的思路（鲁棒化、隐式辨识、校准仿真）
- [ ] [SIMPLER](../03_real2sim2real/simpler/notes.md)：物理仿真版的 real2sim 策略评估

### 阶段 1：部署期适应与真机 world model RL

- **读**：[PAD](./pad/notes.md) → [DayDreamer](./daydreamer/notes.md)
- **前置 checklist**
  - [ ] joint training 与 TTT-Online（参数不重置、在 episode 内累积）
  - [ ] inverse 与 forward dynamics 的区别，以及特征空间中的 representation collapse
  - [ ] 视觉 RL 配方：共享卷积 encoder、frame stacking、random crop
  - [ ] RSSM 中 prior 与 posterior 的区别、KL balancing
  - [ ] Reinforce 与 reparameterization 两种 policy gradient 估计器
  - [ ] 异步 actor/learner 与 replay ratio
- **读完应能回答**
  1. PAD 部署时更新哪些参数、冻结哪些？既然 $\pi_a$ 不变，为什么更新 $\pi_e$ 能提高 return？请用 gradient correlation 条件解释。
  2. 为什么 PAD 选 inverse dynamics 而不是特征空间里的 forward dynamics？TD-MPC2 / V-JEPA 2 又是靠什么避免同样的坍缩的？
  3. DayDreamer 的 "adaptation"（A1 被推、XArm 遇到日出）与 PAD 的适应，在更新信号、更新的参数、所需时间三方面有什么区别？
  4. DayDreamer 为什么不让 actor/critic 的梯度回传到 world model？如果允许回传，会出什么问题？
  5. 没有 reward 时，怎样察觉部署期适应正在伤害性能？（参考 PAD 在 CRLMaze 上的反例）

### 阶段 2：用 learned model 替代或蒸馏物理仿真器（低维 / latent，控制类任务）

- **读**：[Robotic World Model](./robotic-world-model/notes.md) → [SimDist](./simdist/notes.md)
- **前置 checklist**
  - [ ] Dyna / MBPO、compounding error、model exploitation（[RL overview](../04_rl/survey-rl-overview/notes.md)）
  - [ ] teacher forcing 与自回归训练，训练和推理之间的分布 mismatch
  - [ ] 腿足任务的 privileged information、termination、速度跟踪 reward（[legged gym](../03_real2sim2real/legged-gym/notes.md)）
  - [ ] MPPI：采样、指数加权、elites，以及短窗口加 terminal value
  - [ ] $V^{\pi^e}$ 与行为策略价值的区别
  - [ ] offline-to-online RL 基线（RLPD、[IQL](../04_rl/iql/notes.md)）在 finetune 时为什么会 collapse
- **读完应能回答**
  1. 为什么 teacher forcing 是 $N=1$ 的特例？RWM 的 Fig. S8 中，" $M=1,N=32$ 时 $e=1.50$"和" $M=32,N=1$ 时 $e=3.99$"分别说明了 history 和 forecast 的什么作用？
  2. RWM 的真机鲁棒性可能来自哪里？为什么说它严格上是"把物理仿真器蒸馏成 GRU"？
  3. SimDist 为什么可以冻结 encoder、reward、value，只 finetune dynamics？冻结 encoder 起了哪两个作用？
  4. SimDist 的 Eq.3 与 PAD 的部署期更新，在监督信号、更新对象和更新频率上有哪些相同、哪些不同？
  5. RWM 是"用 WM 替代仿真器"，SimDist 是"用仿真器训练 WM"。两者的真实数据分别放在流程的哪一步？

### 阶段 3：视频 world model 作为交互式模拟器和评估器

- **读**：[UniSim](./unisim/notes.md) → [Ctrl-World](./ctrl-world/notes.md)
- **前置 checklist**
  - [ ] DDPM（ $\epsilon$ / $x_0$-prediction）与 classifier-free guidance
  - [ ] latent video diffusion：VAE latent、spatial / temporal 分解注意力、cross-attention 条件注入（SVD、Imagen Video）
  - [ ] VLA 与 action chunking（π0 / π0.5、OpenVLA），inverse dynamics model（UniPi）
  - [ ] 视频生成指标 PSNR / SSIM / LPIPS / FID / FVD，以及它们衡量的是什么
  - [ ] filtered BC / rejection-sampling fine-tuning 与 policy gradient 的区别
  - [ ] hindsight relabeling：把实际到达的结果当作目标重新标注轨迹（UniSim 训练长程 VLM 策略时用）
- **读完应能回答**
  1. UniSim 为什么把 simulator 定义成不含 reward 的 $p(o_t\mid h_{t-1},a_{t-1})$？在 RL 和 hindsight relabeling 两种用法里，监督信号各从哪里来？
  2. UniSim 的 Table 3（BC 0.58 → RL 0.81）证明了什么、没证明什么？要严格支持"在 UniSim 里训练、zero-shot 上真机"，还需要补哪些实验？
  3. Ctrl-World 的多视角联合预测、frame-level action conditioning、pose-conditioned memory 各自解决什么问题？Table 2 中哪个消融的影响最大？
  4. 成功率回归线是 $y=0.81x-0.11$，那么 Ctrl-World 更适合做排名，还是估计绝对成功率？为什么 Close-laptop 上偏差特别大？
  5. 同样是做策略评估，learned simulator（Ctrl-World）与物理 real2sim（SIMPLER）各自的优缺点是什么？

### 阶段 4：在视频 world model 里对 VLA 做 RL

- **读**：[WMPO](./wmpo/notes.md) → [World-Gymnast](./world-gymnast/notes.md)
- **前置 checklist**
  - [ ] GRPO 与 DAPO：去掉 KL、dynamic sampling、非对称 clip
  - [ ] VLA 的动作 token 化：为什么 OpenVLA-OFT 默认的 L1 回归头要换成 LM head 才能做 RL
  - [ ] VLM-as-judge 与 reward hacking；binary outcome reward 与 dense reward 的取舍
  - [ ] Dyna，以及 world model 与 policy 交替更新的结构
- **读完应能回答**
  1. 为什么只用专家示范（或 OXE）训练的 WM 撑不起 GRPO？请从组内 advantage 和 dynamic sampling 两方面解释 policy behavior alignment 的必要性。
  2. 两篇论文都用 trajectory 级 advantage、不训练 critic。这对 WM 提出了什么要求（例如能 reset 到同一 $s_0$）？
  3. World-Gymnast 的 Iter-SFT 与 RL 对照说明了什么？为什么 Iter-SFT 在简单任务上反而退化？
  4. World-Gymnast 的 Table 3 和 Table 1/2 在证据强度上差在哪里？如果 policy 学会利用 WorldGym 的生成伪影，哪张表会暴露问题？
  5. World-Gymnast 的 "TTT" 与 PAD / TTT (Sun et al.) 有什么本质区别？它的 Dyna 对应 03 里的哪一类做法？
  6. 如果要量化 model exploitation，你会在 WMPO 或 World-Gymnast 的训练过程中记录哪些量？

### 阶段 5（选读）：批判视角和前沿

- 读第 9 节的 World-in-World（2510.18135）和 decision-making-centric 评测立场论文（2606.15032），回头审视本线 8 篇的证据口径。
- 读 EVOLVE-VLA（2512.14666）和 WAM-TTT（2607.06988），看 TTT 怎样进入 VLA 和 world action model。
- **读完应能回答**：如果你提出一个新的 world model，并声称它"对策略学习有用"，最少需要哪三类证据？（提示：闭环任务成功率；WM 内指标与真实指标的一致性；model exploitation 诊断）

---

## 7. 关键机制对比表

### 7.1 模拟器侧：world model 是什么、从哪来、reward 怎么给

| 方法 | world model 形式 | 表征 / 预测空间 | WM 训练数据 | reward 从哪来 | 长程误差怎么控制 |
|---|---|---|---|---|---|
| PAD | 没有 world model，只有一个 IDM 辅助 head | CNN 特征 | — | 训练期用环境 reward；部署期没有 reward | — |
| DayDreamer | RSSM（DreamerV2） | 离散 latent（32×32），有重建 decoder | 真机在线采集，从零开始 | 手工 reward，由 WM 的 reward head 学习预测 | imagination horizon 15–16 步，加 λ-return bootstrap |
| RWM | GRU + Gaussian 输出头 | 低维本体观测，外加 privileged info（接触、termination） | Isaac Lab 仿真：多动力学预训练 + 单环境微调 | 手写公式，直接在预测量上计算 | AR 训练（ $M=32,N=8$），每次迭代 rollout 100 步 |
| SimDist | TD-MPC 式 latent WM，chunked transformer dynamics | latent，没有重建项 | 仿真 privileged expert、中间 checkpoint、动作扰动（四足约 100M 数据点） | 仿真 privileged reward 回归，加 $V^e$ 蒸馏；真机上冻结 | 短窗口（0.5–1 s）加 terminal value；一次前向输出全部 $T$ 步 |
| UniSim | 级联 3D U-Net 视频扩散（5.6B） | 像素（24×40，超分到 192×320） | 互联网图文、人类视频、机器人数据、全景扫描、仿真渲染 | 另学 steps-to-completion，由 Stage 1 冻结副本提供 | 4 帧历史自回归，没有专门机制 |
| Ctrl-World | SVD 改造的 latent 视频扩散（1.5B） | 多视角像素（在 VAE latent 中扩散） | DROID（约 76k 成功 / 19k 失败） | 人工判定 | 7 帧稀疏历史 + 位姿（pose-conditioned memory）；历史帧加噪 |
| WMPO | OpenSora DiT + SDXL 2D VAE | 像素接口（在 VAE latent 中扩散） | OXE 预训练，加 base policy 的真实 rollout（含失败） | VideoMAE 0/1 判别器 | noisy-frame conditioning（条件帧加 50/1000 步噪声） |
| World-Gymnast | WorldGym DiT（609M）+ SD3 VAE | 像素接口（在 VAE latent 中扩散） | OXE 预训练（Dyna 阶段再用真机轨迹微调） | GPT-4o 0/1 | 20 帧滑窗，加 temporal KV cache |

### 7.2 优化与适应侧：怎么用、要多少真实数据、证据和成本

| 方法 | 策略 / 优化方式 | 部署期是否更新、更新什么 | 需要的真实数据 | 真机证据 | 算力（论文数字） |
|---|---|---|---|---|---|
| PAD | SAC（DMControl、机器人）或 A2C（CRLMaze），与 IDM 联合训练 | 是：每步一次梯度，更新 $\pi_e$（和 $\pi_s$）， $\pi_a$ 冻结，不用 reward | 训练在仿真中完成；部署期只用自身的 transition | Kinova Gen3，每个设置 25 次，Push cloth 从 64% 升到 88% | 未报告；每个任务训练 500k 步 |
| DayDreamer | imagination actor-critic（连续动作用 reparameterization，离散动作用 Reinforce） | 训练和部署不分开，WM 与策略持续更新（需要 reward） | 全部来自真机（1–10 h） | 4 台机器人，单次运行 | 未报告，只说 single GPU |
| RWM | MBPO-PPO，4096 个 imagination 环境 | 否 | 硬件实验中不用（WM 只用仿真数据） | ANYmal D、G1 zero-shot；tracking reward 0.90±0.04，PPO 为 0.90±0.03 | 单卡 RTX 4090；RWM 约 1 h，MBPO-PPO 5 min |
| SimDist | MPPI 规划（用 policy prior warm-start），没有策略梯度 | 是：分批迭代 finetune $f_\theta$，其余冻结，不需要真实 reward | 真机 rollout 约 30 分钟（四足 32.1 / 35.7 min） | UR5e 2 个任务、Go2 2 个任务；四足 Slippery 上 zero-shot 0/5，适应后 4–5/5 | 四足：数据生成约 7 h + 预训练约 28 h（单卡 RTX 4500 Ada） |
| UniSim | REINFORCE（PaLI 3B VLA）；hindsight relabeling 后 SFT（PaLM-E 12B） | 否 | 预训练用大规模真实视频 | 只有定性结果 | 512 TPU-v3 × 20 天 |
| Ctrl-World | 不做 RL：policy evaluation + filtered SFT（π0.5） | 否 | DROID 离线数据 | DROID 平台：3 个策略排名一致；SFT 后从 38.7% 升到 83.4% | 2×8 H100，2–3 天 |
| WMPO | GRPO（去 KL、dynamic sampling、Clip-Higher），策略为 OpenVLA-OFT | lifelong 版本交替更新 WM 与策略 | $P$=128 或 1280 条 base policy 真实 rollout | ALOHA 1 个任务 30 次：53% → 70% | 32×H100（SFT 用 8×H100），时长未报告 |
| World-Gymnast | GRPO（同上，沿用 SimpleVLA-RL），策略为 OpenVLA-OFT | 可选：TTT（从测试帧在 WM 里做 RL）；Dyna（用真机轨迹微调 WM） | 只要初始帧和指令（Dyna 每个任务另需约 100 条真机轨迹） | AutoEval WidowX 4 个任务；eggplant → blue sink 从 4% 升到 72% | RL：4×H200，1–2 天；WM：2×A100，580k 步 |

**从两张表里能读出的几条规律**

1. **reward 的来源决定了 reward hacking 的形式。**
   - 手写公式（RWM）：reward 必须能从预测量算出来，G1 因此不得不额外预测足高和足速。
   - learned reward（UniSim、WMPO）和 VLM reward（World-Gymnast）：会被 world model 的伪影欺骗。
   - 冻结的仿真 value（SimDist）：不会被 hack，但给性能设了上限。
2. **rollout 成本相差几个数量级。** RWM 单步推理 1 ms；World-Gymnast 每生成 1 帧要做 20 次 DiT 前向；Ctrl-World 的仓库 README 写一次交互约 5–10 s（论文本身没有报告）。这直接决定了能不能用 PPO 这类需要大量样本的算法。
3. **"需要多少真实数据"的口径各不相同。** RWM 的"6M 对 250M transitions"没有计入生成预训练数据时消耗的仿真交互；WMPO 的"相同 rollout budget"没有把 OXE 预训练先验算进去；World-Gymnast 的 TTT 和 Dyna 用到了评测环境本身的信息。

---

## 8. 开放问题与争议

**1. learned simulator 里测到的提升，有多少在真机上成立？（model exploitation / reward hacking）**

各篇的证据情况：

- [UniSim](./unisim/notes.md)：Table 3 的 0.58 → 0.81 是在 UniSim 自己生成的视频里人工判定的，真机只有定性图。
- [RWM](./robotic-world-model/notes.md)：Fig. 5 显示训练早期预测 reward 高于 ground-truth reward，说明 policy 在利用模型的乐观误差；locomotion 必须先用仿真数据预训练才稳定。
- [WMPO](./wmpo/notes.md)：单轮训练中 WM 冻结在 base policy 的分布上；论文既没有报告 WM 内成功率与真实成功率的差距，也没有说明 RM 的 F1>0.95 是在什么数据上测的。
- [World-Gymnast](./world-gymnast/notes.md)：Table 3 在训练所用的 WM 里、用同类 VLM 评测，相当于自己出题自己判。

已有的应对手段：

- DayDreamer 不让 actor/critic 的梯度回传到 world model。
- SimDist 在数据中混入中间 checkpoint 和动作扰动，覆盖 expert manifold 之外的区域。

还没有人做过的诊断：在同一批初始状态上，跟踪"WM 内成功率 − 真实（或物理仿真）成功率"随训练步数的变化。

**2. 视觉质量指标不等于决策有用性（claim/evidence mismatch）**

- [UniSim](./unisim/notes.md) 只报告了 FVD、FID、CLIP 这类画质指标。
- [Ctrl-World](./ctrl-world/notes.md) 在 PSNR 和 FVD 上领先，但成功率回归的截距是 −0.11。Close-laptop 任务上，π0.5 的真机成功率 0.70，想象中只有 0.05。"排名准确"只在 3 个同家族、能力差距明显的策略上验证过。
- World-in-World（2510.18135）和立场论文 2606.15032 都主张以闭环任务成功率为准。

开放问题：什么样的指标能衡量动作条件 world model 的动力学保真度和动作可控性？

**3. 像素还是 latent？**

- [WMPO](./wmpo/notes.md) 的论点：像素接口能让 VLA 的预训练视觉表征直接派上用场，而 Dreamer 的 latent 与 VLA 对不上。
- [SimDist](./simdist/notes.md) 的证据：去掉重建项更好。在仿真消融（Table I）中加入像素重建后，manipulation Peg 从 0.90 降到 0.32；四足从 22.78 升到 23.34（四足输入是 height map，没有做视觉随机化）。
- [DayDreamer](./daydreamer/notes.md) 的 decoder 只提供表示学习信号，behavior learning 不用它。
- 两种路线的 rollout 成本相差几个数量级（见 7.2）。

至今没有一篇论文在同一任务、同一算力下做过受控对比。

**4. 部署期适应应该更新哪一部分、用什么信号？**

- [PAD](./pad/notes.md) 用自监督信号更新 encoder。但在 CRLMaze 上，IDM (PAD) 让训练环境本身从 585 跌到 −416，而部署时没有 reward，察觉不到。
- [SimDist](./simdist/notes.md) 的结论与 PAD 相反：解冻 encoder 后成功率跌到 0（Fig.10），应该只更新 dynamics。它是分批迭代更新的，与逐步在线更新的比较还没有人做。
- [DayDreamer](./daydreamer/notes.md) 全部更新，需要 reward，耗时 10 分钟到 5 小时。

笔记里提出了三个可以直接检验的假设：

- world model 的预测损失能否直接充当 PAD 的自监督损失 $L$（PAD 笔记）？
- 只调 history encoder $C_\theta$，也就是 RMA 式的隐式辨识，能否替代 finetune $f_\theta$（SimDist 笔记）？
- 能否用 [TTT layers](../02_test_time_training/ttt-layers/notes.md) 或 [Titans](../02_test_time_training/titans/notes.md) 替代固定上下文窗口，解决 UniSim 的记忆丢失（UniSim、Ctrl-World 笔记）？

**5. learned simulator 与物理 real2sim，哪条路线更可扩展？**

- [World-Gymnast](./world-gymnast/notes.md) 在 3/4 个真机任务上优于 SIMPLER 中的 RL。但两边的 reward 设计不同（binary 对 dense partial），训练任务集也不同；在 close the drawer 上 SIMPLER 反而更好（74% 对 62%）。所以结论只能是"优于这个 SIMPLER 设置下的 RL"。
- [RWM](./robotic-world-model/notes.md) 的 WM 只用 Isaac Lab 数据训练，它的 sim-to-real 鲁棒性可能继承自预训练数据的动力学多样性（类似 DR），论文没有消融。
- [SimDist](./simdist/notes.md) 实际上把两条路线组合在一起：用物理仿真器生成数据来训练 WM。

对照阅读：03 的 [SIMPLER](../03_real2sim2real/simpler/notes.md)、[RialTo](../03_real2sim2real/rialto/notes.md)、[SplatSim](../03_real2sim2real/splatsim/notes.md)。

**6. world model 要不要和 policy 一起持续更新？**

- [WMPO](./wmpo/notes.md) 的 lifelong 版本只在 StackThree 一个任务上跑了两轮。
- [World-Gymnast](./world-gymnast/notes.md) 的 Dyna 用的是评测任务本身的真机数据，迭代了几轮也没有写清楚。
- [Ctrl-World](./ctrl-world/notes.md) 把这一步列为 future work。

开放问题：迭代多少轮后会收敛或者开始遗忘？数据取自评测任务时，结果还能算泛化吗？这与 [SimOpt](../03_real2sim2real/simopt/notes.md) 的闭环校准是什么关系？

**7. binary outcome reward 与 GRPO 的配合**

组内 reward 全为 0 或全为 1 时，GRPO 没有梯度。World-Gymnast 的 SIMPLER 基线就因为这个原因被迫改用 dense partial reward。WMPO 靠 policy behavior alignment 让想象中出现失败轨迹，才保证了组内方差。基座策略太弱（全失败）或太强（全成功）时，这套方法会怎样退化，都还没有研究。

**8. 术语歧义与证据口径**

- 本线里的 "TTT" 至少有三种含义（见 5.2）。引用时要说清楚：更新信号是自监督损失还是 reward，更新的对象是 policy 还是 world model。
- 多篇论文统计偏弱：
  - DayDreamer：没有消融，A1 是单次运行。
  - WMPO：没有定量消融。
  - SimDist：真机单次运行，文中有几处数字前后不一致。
  - World-Gymnast：各格样本量与"10 次 × 5 轮"的说法对不上。
- 复现和引用时，以各篇笔记"局限"一节列出的口径问题为准。

---

## 9. 扩展阅读

以下都选自 scouting 阶段的 dropped_notable 列表，arXiv ID 已于 2026-09-28 在 arxiv.org 核对标题。

| arXiv | 论文 | 为什么值得读 |
|---|---|---|
| 1803.11347 | Learning to Adapt in Dynamic, Real-World Environments Through Meta-RL | 用 meta-learning 训练动力学模型，让它在部署时用最近的经验在线适应。"部署期适应 world model"这一思路的早期代表，适合对照 PAD 和 SimDist |
| 2504.02252 | ReDRAW: Adapting World Models with Latent-State Dynamics Residuals | 在仿真里预训练 latent 自回归 WM，到目标域用 latent dynamics residual 校准，再在修正后的 WM 里做 imagination RL。与 SimDist 的"只 finetune dynamics"直接对照 |
| 2505.12705 | DreamGen | 用视频 WM 生成 neural trajectories，再用 latent action model 或 IDM 恢复伪动作来训练策略。数据生成子方向的代表，可与 Ctrl-World 的 filtered SFT 对比 |
| 2505.19017 | WorldEval | 用 WM 作为真实机器人策略的评估器（Policy2Vec），与 Ctrl-World 同一方向 |
| 2506.00613 | WorldGym | World-Gymnast 的前身：在 WM 中做 Monte Carlo rollout，由 VLM 给 reward 来评估策略，只需要一张初始帧；报告了与真机成功率的相关性和策略排名的一致性 |
| 2509.24948 | World-Env | 把 WM 当虚拟环境做 VLA 的 RL 后训练，带任务完成检测。WM 内 VLA RL 的早期工作 |
| 2510.00406 | VLA-RFT | 用数据驱动的 WM 做可控模拟器，以参考轨迹给出 dense 的 trajectory 级 reward。可与 WMPO / World-Gymnast 的 0/1 结果奖励对照 |
| 2510.18135 | World-in-World | 闭环 benchmark，按任务成功率而不是画质来评 WM。"视觉质量不等于决策有用性"的直接证据 |
| 2512.10675 | Evaluating Gemini Robotics Policies in a Veo World Simulator | 工业级 WM 策略评估，覆盖 nominal、OOD 和 safety 三类用途（没有代码） |
| 2512.14666 | EVOLVE-VLA | VLA 的 test-time training：用 learned progress 估计代替测试时拿不到的 oracle reward，在环境交互中持续适应。可与 PAD（自监督）和 World-Gymnast（WM 内 test-time RL）对照 |
| 2602.06949 | DreamDojo | 在 44k 小时人类第一视角视频上预训练的机器人 world model，用连续 latent action 作代理动作；有代码 |
| 2603.18532 | Scaling Sim-to-Real VLA RL with Generative 3D Worlds | 用 3D 生成模型批量构造仿真场景来做 VLA RL，走"生成式 real2sim + 物理仿真"路线，与视频 WM 路线（World-Gymnast）形成对照 |
| 2607.06988 | WAM-TTT | 冻结 world action model，只用无标注人类视频，通过自监督视频预测更新一个轻量的 adaptive memory。TTT × world action model 的最新方向（没有代码） |
| 2507.00917 | A Survey: Learning Embodied Intelligence from Physical Simulators and World Models | 同时覆盖物理仿真器和 world model 的综述，正好对应本线"两类模拟器"的比较 |
| 2606.15032 | How Should World Models Be Evaluated for Embodied Decision-Making? | 立场论文：梳理 world model 评测指标，提出 claim/evidence mismatch 的问题。读完再回头审视本线各篇的证据 |

---

## 10. 复现推荐

按 research-workflow "先复现再提假设"的原则，推荐按下面的顺序复现 3 个 baseline。三者都不需要真机，都能在单卡上完成最小实验，合起来覆盖了本线最核心的三个机制：部署期自监督适应、learned simulator 里的 RL、仿真预训练 WM 加真实域 dynamics 适应。

### 10.1 PAD：部署期自监督适应（难度 easy）

- **理由**：代码成熟、DMControl 单机就能跑，能直接检验 TTT × RL 最基本的机制：不用 reward 的部署期更新能否提高 return。它也是后面所有"部署期适应"假设最便宜的测试平台，比如把 IDM 换成 world model 的预测损失，或者比较逐步更新和按 chunk 更新。
- **算力**：论文没有报告 GPU。每个任务训练 500k 步 SAC（Table 11）；PAD 在评测时每步多做一次 batch=32 的反传。按个人估计单卡可跑（这不是论文数字）。
- **最小实验**：
  1. 在 DMControl Cartpole swingup（action repeat 8）上训练 SAC+IDM，至少 3 个 seed。
  2. 在随机颜色环境上比较四种设置：不做 PAD、PAD、O-PAD、冻结 $\pi_s$ 的 PAD。论文数字依次是 585±73、630±63、578±69、623±57（Table 1/3/9）。
  3. 在训练环境上评测，确认 PAD 几乎没有伤害（论文是 849 对 845，Table 7）。
- **注意**：2020 年版本的 dm_control 和 dmc2gym 与现在的环境不兼容；DMControl 的测试期学习率和系数 $\alpha$ 论文没写，要去代码里查；提升常常在一个 std 以内，seed 要够多。

### 10.2 Robotic World Model：learned simulator 里的 PPO（难度 medium）

- **理由**：在本线里，要跑通"训练 learned simulator → 在其中跑 PPO"的完整闭环，它的成本最低：单张 4090、约 1 小时。 $M/N$ 消融能直接量化 compounding error，这是之后任何"在 WM 里做 RL"研究都要先做的诊断。
- **算力**：单卡 RTX 4090。RWM 约 1 h（Table 1 写的是 50 min），MBPO-PPO 5 min。需要 Linux 或 Windows 加 NVIDIA RTX GPU，Isaac Sim 不支持 macOS。
- **最小实验**：
  1. 在 Isaac Lab 里用现成的 ANYmal-D flat 速度策略采集轨迹。
  2. 固定 $M=32$，比较 $N\in\lbrace1,2,8\rbrace$，评估从第 32 步起自回归约 170 步的误差，看能否复现 Fig. S8 的趋势（3.99 → 1.91 → 0.47）。三个模型大约 3 小时。
  3. 跑官方 pretrain → MBPO-PPO，把 ground-truth reward 曲线与 Fig. 5 左图对照。
  4. 顺带补一个论文缺的诊断：记录"预测 reward − 真实 reward"随训练的变化，作为 model exploitation 的量化指标。
- **没有 GPU 时的退路**：在 MuJoCo 低维 locomotion 环境上自己写 GRU 加 AR 损失，只验证 teacher forcing 与 AR 的定性差异。

### 10.3 SimDist：仿真预训练 WM + 伪真实域 dynamics 适应（难度 medium）

- **理由**：一篇论文就覆盖了四条线（privileged expert + DR、TD-MPC 式 latent WM、部署期自监督 dynamics 更新、与 RL finetune 基线对照）。用超出 DR 范围的仿真环境充当"伪真实域"，不需要真机就能检验它的核心论点："只更新 dynamics 就够，解冻 encoder 会崩"。
- **算力**：只有四足部分有报告。数据生成约 7 h，预训练约 28 h，都是单卡 RTX 4500 Ada；expert PPO 的耗时没有报告。数据量可以减到 50%，Table I 中四足 reward 为 22.73，全量是 22.78，几乎不变。
- **最小实验**：
  1. 在 IsaacLab Go2 上按官方流程训练 expert、生成数据、预训练 WM。
  2. 构造超出 DR 范围的目标域。训练时摩擦在 [0.2, 1.2]、附加质量在 −1 到 +3 kg；目标域可以取摩擦 0.05–0.1，或附加质量 +6 kg。
  3. 在目标域里用 MPPI 采约 30 分钟数据，按 Eq.3 只 finetune $f_\theta$。
  4. 比较 zero-shot、只调 $f_\theta$、解冻 $E_\theta$、解冻 $V_\theta$ 四种设置的前进距离和 held-out latent loss，对照 Fig.8a 中 0.076 → 0.019 的趋势。
  5. 再加两组论文没做的对照：只调 $C_\theta$（RMA 式），以及逐步在线更新与分批更新的对比（和 PAD 连起来）。

### 暂不推荐先复现的

| 论文 | 原因 |
|---|---|
| UniSim | 没有代码，训练需要 512 TPU-v3 × 20 天 |
| DayDreamer | 需要真机硬件；TF2 的旧依赖难装；异步训练的结果依赖 GPU 吞吐 |
| Ctrl-World | 只做推理可以（单卡 A100/H100），但排名对齐的实验需要 DROID 真机 |
| WMPO | 用官方 checkpoint 只跑 Coffee_D0、 $P=128$（对照 43.8 → 61.7）是可行的第 4 个选择，但论文用了 32×H100 |
| World-Gymnast | 需要 4×H200 和 GPT-4o 调用费用 |

WMPO 和 World-Gymnast 适合在前三个跑通、多卡资源到位之后，专门去补"WM 内成功率与真实成功率差距"这个诊断。
