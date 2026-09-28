# World Models

> 主题目录 `01_world_model` 的入口。共收录 11 篇：1 篇综述，3 篇 foundational，4 篇 milestone，3 篇 frontier。
> 每篇的 `notes.md` 都写了机制、证据、局限和复现可行性。本 README 只做"地图"：脉络、分类、阅读顺序、横向对比、开放问题和复现建议。
> 文中数字都取自各篇笔记，出处可在对应论文的 `source/` 或 `paper.pdf` 中找到。未收录论文只给 arXiv ID，这些 ID 已于 2026-09-28 按 arXiv 标题逐一核对。
> 最后更新：2026-09-28

---

## 1. 这条线在解决什么问题

World model 是一个学出来的预测模型：给定历史观测和动作，预测环境接下来会怎样变化，预测对象可以是观测、latent state，常常还包括 reward 和终止信号。它要回答的核心问题是：真实交互昂贵、慢、危险，能不能先从数据里学出环境的替代品，再在其中做 planning / MPC、在 imagination（latent rollout）里训练 policy、评估 policy 或合成数据，从而大幅减少真实交互。

围绕这个目标，这条线一直在处理三个子问题：

- **预测什么**：像素、离散 token、可重建的 latent state，还是只对决策有用的表征。
- **从什么数据学**：环境里自己采的、带动作的交互数据，还是海量无动作视频。
- **怎么用**：当 simulator 训练 policy、做在线规划，还是直接把 world model 变成 policy。

贯穿始终的难点是 model error。模型在 policy 实际访问的状态上预测不准时，policy 会利用模型的漏洞（model exploitation），长 rollout 中误差还会不断累积（compounding error）。这正是 world model 和本项目的 sim-to-real 线、TTT 线相交的地方：learned simulator 的 model error 就是它自己的 reality gap，而部署时能否在线修正这种误差，本身就是一个 test-time adaptation 问题。

---

## 2. 发展脉络

先看分支结构。带链接的是本项目收录的论文，其余给出年份，arXiv ID 见第 3 节：

```
2018 World Models（V-M-C；controller 在 M 生成的"梦境"里训练）
 ├─ Latent MBRL：PlaNet(2018) → Dreamer(2019) → DreamerV2(2020) → DreamerV3(2023) ──┐
 ├─ Decoder-free / value-equivalent：MuZero(2019) → TD-MPC(2022) → TD-MPC2(2023)     │
 ├─ 观测空间生成式 WM 用于 RL：IRIS(2022) → DIAMOND(2024)                             ├─→ Dreamer 4(2025)
 ├─ 生成式交互环境 / world foundation model：Genie(2024), GameNGen(2024) → Cosmos(2025) ┘   （可扩展视频 WM + 纯离线 imagination RL）
 ├─ JEPA 表征空间预测：DINO-WM(2024) → V-JEPA 2(2025) → LeWorldModel(2026)
 └─ Video policy / World Action Model：UniPi(2023) → DreamZero, Cosmos Policy(2026)
```

按时间逐步看。每一行都写明它解决了前一步的什么问题，又留下了什么问题：

| 年份 | 工作 | 解决了上一步的什么问题 | 留下的问题 |
|---|---|---|---|
| 2018 | [World Models](./world-models/notes.md) | model-free RL 受 credit assignment 限制只能训小网络；此前学到的 dynamics model 很少被用来**完全替代**真实环境。这篇用 V（ConvVAE）+ M（MDN-RNN）无监督地学环境，C 只有约一千个参数。在 Doom 中，C 全程在 M 生成的梦境里训练，然后迁回真实环境 | V、M、C 分阶段训练，V 不知道哪些信息与任务相关；CMA-ES 只能优化很小的 C；model exploitation 只能靠调 MDN 温度 τ 缓解，而 τ 是按真实环境分数挑出来的 |
| 2018 | PlaNet（未收录） | 把 V 和 M 合并成一个联合训练、带 reward 预测的 RSSM（确定性路径 + 随机 latent） | 每一步都在线做 CEM 规划，只看 horizon 内的 reward，所以行为短视，训练也慢（约 11 h / 1e6 步） |
| 2019 | [Dreamer](./dreamer/notes.md) | 在 RSSM latent 里 imagine H=15 步，用 λ-return 训练 value，再把 value 的解析梯度穿过 dynamics 传给 actor。这样学到的行为能超出 horizon，推理时不用在线规划，训练也降到约 3 h / 1e6 步 | 连续任务和离散任务要两套超参数；在 Atari/DMLab 上没有竞争力；没有不确定性估计，也没有分析 model exploitation |
| 2020 | DreamerV2（未收录） | 改用离散 latent 并加入 KL balancing，纯 world model agent 在 Atari 上达到人类水平 | 仍然需要按 domain 调参 |
| 2023 | [DreamerV3](./dreamerv3/notes.md) | symlog、symexp twohot、KL balancing + free bits、percentile return normalization 让各项损失的尺度与信号量级解耦。一套固定超参数跑通 8 个 domain、150 多个任务，并首次在不用人类数据的条件下从零拿到 Minecraft 钻石 | 缺少定向探索（BSuite Deep Sea 为 0.00）；representation 主要由重建塑形；每个任务都要在线从零训练 |
| 2023 | [TD-MPC2](./tdmpc2/notes.md) | 走另一条路：不重建观测（decoder-free），只用 latent consistency + reward + value 三项训练，决策时在潜空间做 MPPI。SimNorm 和离散回归修好了 TD-MPC 一放大就发散的问题，模型从 1M 扩到 317M 参数，一个 agent 覆盖 80 个任务 | 不能生成观测，所以不能当 simulator；encoder 只看单帧、dynamics 是确定性的，表达不了部分可观测和多模态转移 |
| 2024 | [Genie](./genie/notes.md) | 此前的 world model 都需要带动作的数据。Genie 用 latent action model（只有 8 个码字的 VQ 瓶颈），从 30k 小时无动作的互联网游戏视频里学出一个可逐帧控制的 11B 模型 | 没有 reward 和终止信号，本身不是 RL 环境；8 个离散 latent action 表达不了机器人的动作；约 1 FPS，只有 16 帧记忆 |
| 2024 | [DIAMOND](./diamond/notes.md) | IRIS、DreamerV2/V3 的离散化压缩可能丢掉对决策关键的视觉细节。DIAMOND 改在像素空间用 EDM 扩散做 world model，每帧只需 3 步去噪。在完全靠 world model 训练的 agent 里，它的 Atari 100k mean HNS 为 1.46，排名第一 | 只记得 4 帧；像素空间的成本随分辨率上升；只测了离散动作 |
| 2025 | [Cosmos WFM](./cosmos-wfm/notes.md) | 把"从视频学动力学先验"推到 foundation model 的规模：用约 20M 小时视频预训练扩散式（7B/14B）和自回归式（4B–13B）两族 WFM，再用小数据 post-train 出相机、机器人动作、驾驶轨迹条件的版本，并开源权重 | 没有任何闭环实验（RL、MPC、policy 评估都没做）；模型变大并不改善物理对齐；动作只在 post-training 阶段才接入，而且接入方式很"薄"；rollout 很贵 |
| 2025 | [V-JEPA 2](./vjepa2/notes.md) | 生成式 WM 用来规划太慢（论文中 Cosmos 每个动作要 4 min），像素生成还把容量花在不可预测的细节上。V-JEPA 2 改为在表征空间做预测：先用超过 1M 小时视频做 JEPA 预训练，再用不到 62 小时的 Droid 数据训练动作条件 predictor；部署时用 CEM 最小化到目标图像表征的距离，在两个未见过的实验室 zero-shot 完成抓取和拾放（每个动作 16 s） | 主实验的 planning horizon=1；对相机位置敏感；只接受 image goal；确定性的 L1 predictor 表达不了多模态的未来 |
| 2025 | [Dreamer 4](./dreamer4/notes.md) | 把 Dreamer 式的 agent 训练与视频 WM 的规模合到一起：用 shortcut forcing（diffusion forcing + shortcut model + x-prediction）训练 2B 的 block-causal Transformer，在单张 H100 上跑到 21 FPS。只用 2.5K 小时离线 VPT 数据，在 imagination 里用 PMPO 做 RL，首次纯离线拿到 Minecraft 钻石（成功率 0.7%） | 记忆只有 9.6 s；没有度量想象 return 与真实 return 的相关性；没有真机 policy；没有开源代码 |
| 2026 | [DreamZero](./dreamzero/notes.md) | 直接预测动作的 VLA 缺少"未来会怎样"的显式预测，到新环境泛化差。DreamZero 把 14B 视频扩散模型改造成 World Action Model，联合去噪未来视频和动作块，闭环执行时把真实帧写回 KV cache。在未见环境和未见任务上，它比 SOTA VLA 高 2 倍以上 | 它建模的是 $p(o,a\mid\cdot)$，不是 $p(s'\mid s,a)$，所以不能当 simulator；7 Hz 需要 2×GB200；上下文只有约 6.6 s |
| 2026 | [WM for Robot Learning 综述](./survey-wm-robot-learning/notes.md) | 把 2023–2026 年的机器人 WM 放进三套分类：与 policy 的耦合方式、当 simulator 的两种用途、video WM 的四个能力阶段 | 几乎不涉及 Dreamer 系 MBRL；没有量化 learned simulator 的 reality gap；没有 TTT 视角 |

三条趋势：

1. **数据来源**从"自己采、带动作的交互数据"转向"海量无动作视频 + 少量动作数据"。这条路径依次是：Genie 的 latent action；Cosmos 和 V-JEPA 2 的"无动作预训练 + 小数据动作后训练"；Dreamer 4 的实验（只有 10 h / 100 h 带动作时，归一化 PSNR 分别达到 53% / 85%）；DreamZero 的"视频预训练主干 + 机器人数据"。
2. **用途**从"在 imagination 里训练 policy"扩展到规划、交互环境、policy 评估，最后连 policy 本身也可以是 world model（WAM）。
3. **预测空间**上，生成式（像素 / token / 可解码 latent）和非生成式（decoder-free、JEPA）两派并行发展，至今没有定论（见第 7 节）。

---

## 3. 技术分类（taxonomy）

可以从两个正交的维度切分：**预测发生在什么空间**，以及**world model 拿来做什么**。本项目收录的论文用相对链接指向笔记，未收录的给出 arXiv ID。综述视角可参考本目录收录的 [2605.00080](./survey-wm-robot-learning/notes.md)，也可以看另一篇具身 WM 综述 2510.16732。

### 3.1 按预测空间分

| 子方向 | 核心思路 | 本项目收录 | 未收录的代表 |
|---|---|---|---|
| 重建式 latent state（RSSM 系） | 把观测压成 latent，用可重建的目标（ELBO / 重建 + KL）学 latent dynamics，通常同时预测 reward 和 continue | [World Models](./world-models/notes.md)（VAE + MDN-RNN）、[Dreamer](./dreamer/notes.md)、[DreamerV3](./dreamerv3/notes.md) | PlaNet 1811.04551、DreamerV2 2010.02193 |
| Decoder-free / value-equivalent | 不重建观测，只要求 latent 能预测 reward、value 和下一步 latent，表征完全为决策塑形 | [TD-MPC2](./tdmpc2/notes.md) | MuZero 1911.08265、TD-MPC 2203.04955 |
| 观测空间（像素）生成 | 直接在像素上用扩散模型生成下一帧 | [DIAMOND](./diamond/notes.md) | GameNGen 2408.14837 |
| 离散 token 生成 | 先用 VQ / FSQ 把视频 token 化，再做 masked 或自回归的 token 预测 | [Genie](./genie/notes.md)（VQ + MaskGIT）、[Cosmos](./cosmos-wfm/notes.md) 自回归族（FSQ） | IRIS 2209.00588 |
| 连续 latent 生成（可解码回图像） | 在 video tokenizer / VAE 的连续 latent 上做 diffusion 或 flow matching | [Cosmos](./cosmos-wfm/notes.md) 扩散族、[Dreamer 4](./dreamer4/notes.md)（causal tokenizer + shortcut forcing）、[DreamZero](./dreamzero/notes.md)（Wan2.1 VAE latent） | Cosmos-Predict2.5 2511.00062 |
| JEPA 表征空间预测 | 用 EMA teacher + stop-gradient 在表征空间里回归，不生成像素，也不用 reward | [V-JEPA 2](./vjepa2/notes.md) | DINO-WM 2411.04983、LeWorldModel 2603.19312、JEPA-WMs 2512.24497、V-JEPA 2.1 2603.14482 |

### 3.2 按用途分

| 用途 | 做法 | 本项目收录 | 未收录的代表 / 交叉线 |
|---|---|---|---|
| 在 imagination 中训练 policy（Dyna 式） | world model 当 learned simulator，policy 在 latent rollout 或生成帧上用 actor-critic / ES / PMPO 训练 | [World Models](./world-models/notes.md)（Doom）、[Dreamer](./dreamer/notes.md)、[DreamerV3](./dreamerv3/notes.md)、[DIAMOND](./diamond/notes.md)、[Dreamer 4](./dreamer4/notes.md) | 交叉线：[DayDreamer](../05_intersections/daydreamer/notes.md)、[Robotic World Model](../05_intersections/robotic-world-model/notes.md) |
| 决策时规划（MPC） | 每一步都在模型里搜索动作序列 | [TD-MPC2](./tdmpc2/notes.md)（MPPI + Q bootstrap）、[V-JEPA 2](./vjepa2/notes.md)（CEM + goal energy） | PlaNet 1811.04551（CEM）、MuZero 1911.08265（MCTS）、交叉线 [SimDist](../05_intersections/simdist/notes.md)（MPPI） |
| 可交互环境 / neural game engine | 人或 agent 逐帧输入动作，模型实时生成下一帧 | [Genie](./genie/notes.md)、[DIAMOND](./diamond/notes.md)（CS:GO）、[Dreamer 4](./dreamer4/notes.md)（人类实时交互评测） | GameNGen 2408.14837、Matrix-Game 2.0 2508.13009、WorldPlay 2512.14614；交叉线 [UniSim](../05_intersections/unisim/notes.md) |
| 直接作为 policy（video policy / WAM） | 预测未来和生成动作由同一个模型完成，或串联完成 | [DreamZero](./dreamzero/notes.md)；综述把这一类细分为 IDM-style、single-backbone、MoE/MoT、unified VLA、latent-space 五种 | UniPi 2302.00111、Cosmos Policy 2601.16163、LingBot-VA 2601.21998；Fast-WAM 2603.16666 质疑推理时想象未来是否必要 |
| Policy 评估与 VLA 的 RL post-training 环境 | 在视频 WM 里 rollout VLA，用于排序 / 打分，或在其中做 GRPO / PPO | [综述](./survey-wm-robot-learning/notes.md) Sec.4；[Cosmos](./cosmos-wfm/notes.md) 设想了这一用途，但没有做实验 | 交叉线 [Ctrl-World](../05_intersections/ctrl-world/notes.md)、[WMPO](../05_intersections/wmpo/notes.md)、[World-Gymnast](../05_intersections/world-gymnast/notes.md)；WorldEval 2505.19017、WorldGym 2506.00613、Veo World Simulator 2512.10675、World-in-World 2510.18135、World-Env 2509.24948、VLA-RFT 2510.00406、WoVR 2602.13977 |
| 数据合成 | 用 video WM 生成 neural trajectories，再用 latent action / IDM 补上动作标签 | 综述 Sec.5 | DreamGen 2505.12705、DreamDojo 2602.06949 |

### 3.3 本项目没有重点覆盖的子方向

- **驾驶 world model**：GAIA-1 2309.17080、GAIA-2 2503.20523。离本项目的机器人主线较远。
- **world model 的训练目标**：Diffusion Forcing 2407.01392（Dreamer 4 与 DreamZero 的技术前提）；shortcut forcing 见 [Dreamer 4](./dreamer4/notes.md)；RLVR-World 2505.13934 用 RL 训练 world model 本身。

---

## 4. 论文总表

"阅读顺序"即第 5 节学习路线的顺序，与 `index.json` 中的 `reading_order` 不完全相同：这里把 DreamerV3 提前，和 Dreamer 连读；把 DIAMOND 放在 Genie 之前，和 TD-MPC2 一起读，围绕"world model 该预测什么"做对照。

| 阅读顺序 | 论文 | 年份 | 层级 | 一句话 | 代码 | 笔记 |
|---|---|---|---|---|---|---|
| 1 | World Models（1803.10122） | 2018 | foundational | 由 V（ConvVAE）、M（MDN-RNN）、C（线性，用 CMA-ES 优化）组成。CarRacing 上 906±21，首次达到 solved；Doom 中 C 全程在梦境训练，真实环境得分 1092±556；用温度 τ 抑制 model exploitation | [hardmaru/WorldModelsExperiments](https://github.com/hardmaru/WorldModelsExperiments)（TF1、gym 0.9.x） | [notes](./world-models/notes.md) |
| 2 | Dream to Control / Dreamer（1912.01603） | 2019 | foundational | 在 RSSM 的 latent imagination 中用 λ-return 训练 value，把 value 的解析梯度反传给 actor；DMC 20 任务、5e6 步，平均分 823 | [danijar/dreamer](https://github.com/danijar/dreamer)（TF 2.2） | [notes](./dreamer/notes.md) |
| 3 | DreamerV3（2301.04104） | 2023 | milestone | 靠 symlog、twohot、KL balancing + free bits、percentile return normalization，一套固定超参数跑通 150 多个任务，并在 Minecraft 从零拿到钻石 | [danijar/dreamerv3](https://github.com/danijar/dreamerv3)（JAX） | [notes](./dreamerv3/notes.md) |
| 4 | TD-MPC2（2310.16828） | 2023 | milestone | decoder-free latent + MPPI + Q bootstrap，再加 SimNorm、离散回归、Q ensemble；104 个任务用固定超参数，80 任务 agent 扩展到 317M 参数 | [nicklashansen/tdmpc2](https://github.com/nicklashansen/tdmpc2)（含 checkpoints） | [notes](./tdmpc2/notes.md) |
| 5 | DIAMOND（2405.12399） | 2024 | milestone | 像素空间的 EDM 扩散 world model，每帧 3 步去噪，agent 完全在其中训练；Atari 100k mean HNS 1.46 | [eloialonso/diamond](https://github.com/eloialonso/diamond) | [notes](./diamond/notes.md) |
| 6 | Genie（2402.15391） | 2024 | foundational | 用 8 码字的 latent action model，从 30k 小时无动作游戏视频学出一个 11B、可逐帧控制的 world model | 无官方代码；社区 JAX 复现 [Jafar](https://github.com/flairox/jafar) | [notes](./genie/notes.md) |
| 7 | Cosmos WFM（2501.03575） | 2025 | milestone | 用约 20M 小时视频预训练扩散式和自回归式两族 WFM，再 post-train 出相机 / 机器人动作 / 驾驶条件的版本，开源权重 | [nvidia-cosmos/cosmos-predict1](https://github.com/nvidia-cosmos/cosmos-predict1)（已不再积极开发） | [notes](./cosmos-wfm/notes.md) |
| 8 | V-JEPA 2（2506.09985） | 2025 | frontier | 在表征空间做 JEPA 预训练，再冻结编码器、训练动作条件 predictor；用 CEM-MPC 逼近目标图像的表征，Franka 上 zero-shot 完成抓取和拾放 | [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2) | [notes](./vjepa2/notes.md) |
| 9 | Dreamer 4（2509.24527） | 2025 | frontier | 用 shortcut forcing 训练 2B Transformer WM（单张 H100 21 FPS），在 imagination 里用 PMPO 做 RL，纯离线拿到 Minecraft 钻石（0.7%） | 无官方代码 | [notes](./dreamer4/notes.md) |
| 10 | DreamZero（2602.15922） | 2026 | frontier | 以 14B 视频扩散模型为主干的 World Action Model，联合去噪视频和动作，闭环时把真实帧写回 KV cache，推理 7 Hz；在未见环境和未见任务上比 SOTA VLA 高 2 倍以上 | [dreamzero0/dreamzero](https://github.com/dreamzero0/dreamzero) | [notes](./dreamzero/notes.md) |
| 11 | World Model for Robot Learning 综述（2605.00080） | 2026 | survey | 以 VLA 为中心的三套分类，加上 benchmark、数据集和结果汇总。结论是照片级视频生成不是有效控制的必要条件，真正的瓶颈是动作条件下的因果一致性和长时程可靠性 | 无代码（论文列表：NTUMARS/Awesome-World-Model-for-Robotics-Policy） | [notes](./survey-wm-robot-learning/notes.md) |

> 综述可以在第一天先扫一遍 Fig.2 和 Sec.3.1 当作地图。但它几乎不覆盖 Dreamer 系 MBRL（参考文献里没有 Dreamer/DreamerV3/Dreamer 4/Genie/DIAMOND），所以精读放在最后。

---

## 5. 推荐学习路线

### 阶段 0：本线之外的前置

- [ ] MDP / POMDP、return 与 discount、actor-critic、TD(λ)：[RL Overview](../04_rl/survey-rl-overview/notes.md)（重点读 model-based RL 部分）
- [ ] REINFORCE 与 pathwise（reparameterization）两种梯度估计器；[PPO](../04_rl/ppo/notes.md) 的 clipped surrogate 与 GAE；[SAC](../04_rl/sac/notes.md) 的 tanh-Gaussian actor 与 maximum entropy
- [ ] VAE：ELBO、KL、reparameterization trick

### 阶段 1：经典骨架，latent MBRL（阅读顺序 1–3）

**前置 checklist**
- [ ] RNN / LSTM / GRU，teacher forcing
- [ ] Mixture Density Network 与温度采样；CMA-ES（只需知道它是只看 fitness 的黑箱优化）
- [ ] PlaNet 的 RSSM：确定性路径与随机 latent，posterior（filtering）与 prior（transition）的区别
- [ ] λ-return 的 bias–variance 权衡；free bits；straight-through estimator
- [ ] 分布式 value（C51）与 twohot 编码

**要读**：[World Models](./world-models/notes.md) → [Dreamer](./dreamer/notes.md) → [DreamerV3](./dreamerv3/notes.md)

**读完应能回答**
1. 为什么 τ=0.1 时 Doom 梦境分数接近满分（2086），真实分数（193）却低于随机策略？这说明 learned simulator 有什么根本风险？
2. 写出 Dreamer 中 actor 梯度从 $V_\lambda$ 流回策略参数的完整路径。为什么它只需要 state value，不需要 Q？
3. 为什么加了 value model 后，Dreamer 对 imagination horizon 更鲁棒，而 No value 和 PlaNet 在稀疏任务上学不会？
4. DreamerV3 的 symlog、symexp twohot、percentile return normalization、KL balancing + free bits 各自在解决哪一种"尺度"问题？所谓"一套固定超参数"，实际上还有哪些量是按 benchmark 选的？
5. DreamerV3 的 learning signal 消融说明 representation 主要由哪一项塑形？这对"部署时用自监督损失更新 world model"意味着什么？

### 阶段 2：world model 该预测什么？decoder-free 对比像素生成（阅读顺序 4–5）

**前置 checklist**
- [ ] MPC 与基于采样的轨迹优化（CEM / MPPI：elite、warm start、receding horizon）
- [ ] BYOL / SPR 式 self-predictive 表示学习；REDQ 式 Q ensemble
- [ ] Score-based diffusion 与 EDM（preconditioning、 $c_{\rm skip}$、log-normal σ 采样），以及它和 DDPM ε-prediction 的区别
- [ ] Atari 100k 的评测口径：HNS、IQM、stratified bootstrap CI
- [ ] （第 2 题用）test-time training 的基本含义：部署时用无标签信号更新参数，而不只是多花推理算力，见 02 线的 [TTT](../02_test_time_training/ttt-self-supervised/notes.md)

**要读**：[TD-MPC2](./tdmpc2/notes.md) → [DIAMOND](./diamond/notes.md)

**读完应能回答**
1. TD-MPC2 不重建观测，它的 latent 靠哪几项损失塑形？为什么它不能像 Dreamer 或 DIAMOND 那样当 simulator 用？
2. TD-MPC2 的消融中，只用 planning 得 53.7，只用 policy 得 42.2。这对"推理时多花计算"意味着什么？MPPI 与 TTT 有什么区别？
3. DIAMOND 为什么用 EDM 而不用 DDPM 的 ε-prediction？为什么 1 步去噪在 Boxing 上会变模糊，并导致分数下降？
4. "visual details matter"的证据有多强？要得到因果结论，还缺什么受控实验？
5. 把 Dreamer（重建式 latent）、TD-MPC2（decoder-free latent）、DIAMOND（像素）放在一起，说出它们各自的 model error 表现在哪里。

### 阶段 3：从视频规模学 world model（阅读顺序 6–8）

**前置 checklist**
- [ ] VQ-VAE（码本、commitment loss、以码本大小作信息瓶颈）与 FSQ
- [ ] ViT、tubelet embedding、时空分解注意力（ST-transformer）；MaskGIT
- [ ] DiT、latent diffusion、classifier-free guidance、AdaLN
- [ ] JEPA / I-JEPA / V-JEPA：EMA teacher + stop-gradient 如何防止 collapse
- [ ] inverse dynamics model 与 latent action；FVD / PSNR / SSIM

**要读**：[Genie](./genie/notes.md) → [Cosmos WFM](./cosmos-wfm/notes.md) → [V-JEPA 2](./vjepa2/notes.md)

**读完应能回答**
1. Genie 的 LAM 为什么要把码本做得很小（8 个）？它默认了场景的什么结构？镜头运动或其他角色的运动会被怎样吸收？
2. ΔtPSNR 衡量的是什么，又衡量不了什么？Cosmos 生成视频的 Sampson error（0.355）为什么会"优于"真实视频（0.431）？
3. Cosmos 的动作条件是在哪个阶段、以什么方式接入的？为什么论文没能把"视频预训练带来的收益"单独分离出来？
4. V-JEPA 2-AC 的规划目标（energy）是什么？主实验取 horizon=1 意味着什么？它与 Cosmos 在 Table 3 中的对比为什么不是等算力对比？
5. Genie、Cosmos、V-JEPA 2 分别对应"生成离散 token"、"生成连续 latent"、"只预测表征"。这三种选择分别怎样影响它们能否用于规划或 RL？

### 阶段 4：前沿汇合，WM 内 RL 与 WAM（阅读顺序 9–11）

**前置 checklist**
- [ ] Flow matching / rectified flow；x-prediction 与 v-prediction 的换算
- [ ] Diffusion forcing（逐帧独立噪声）与 shortcut model（以步长为条件的 bootstrap 自蒸馏）
- [ ] KV cache、block-causal mask、GQA；自回归视频生成中的 exposure bias
- [ ] KL 正则的 policy optimization（MPO / PMPO，前向与反向 KL，BC prior）；offline RL 中的分布偏移
- [ ] VLA 的基本形态（π0.5、GR00T）、action chunking、IDM

**要读**：[Dreamer 4](./dreamer4/notes.md) → [DreamZero](./dreamzero/notes.md) → [综述](./survey-wm-robot-learning/notes.md)

**读完应能回答**
1. Dreamer 4 为什么改用 x-prediction 加 x 空间损失，又加 ramp 权重？这些改动怎样让每帧 4 步采样的长 rollout 保持稳定（Table 2 中 FVD 从 875 降到 102）？
2. 按 Dreamer 4 的 Table 7，world model 预训练和 imagination RL 各贡献了多少？offline MBRL 中它靠什么抑制 model exploitation？
3. DreamZero 建模的 $p(o,a\mid\cdot)$ 和 Dreamer 建模的 $p(s'\mid s,a)$ 有什么区别？为什么 DreamZero 不能直接拿来做 imagination RL 或 policy 评估？
4. DreamZero 部署时把真实帧写回 KV cache，这和 Dreamer 部署时用真实观测更新 posterior 有什么异同？
5. 综述说的 causal conditioning gap 是什么？设计一个实验，检测一个视频 WM 的预测是否真的由动作决定。

### 阶段 5：接到交叉线

本线读完后，转到 [05_intersections](../05_intersections/)：

- [DayDreamer](../05_intersections/daydreamer/notes.md)：把 Dreamer 用到真机上在线学习。
- [Robotic World Model](../05_intersections/robotic-world-model/notes.md)：在 learned simulator 里用 PPO 训练腿足 policy。
- [SimDist](../05_intersections/simdist/notes.md)：在仿真里预训练 TD-MPC 式 WM，上真机时只更新 latent dynamics。
- [UniSim](../05_intersections/unisim/notes.md)、[Ctrl-World](../05_intersections/ctrl-world/notes.md)、[WMPO](../05_intersections/wmpo/notes.md)、[World-Gymnast](../05_intersections/world-gymnast/notes.md)：视频 WM 用作 simulator、evaluator 和 RL 环境。
- [PAD](../05_intersections/pad/notes.md)：部署时用 self-supervised auxiliary task 做适应。

长时程记忆这个问题，可以接着读 02 的 [TTT layers](../02_test_time_training/ttt-layers/notes.md)、[Titans](../02_test_time_training/titans/notes.md) 和 [One-Minute Video](../02_test_time_training/ttt-one-minute-video/notes.md)。

---

## 6. 关键机制对比表

### 6.1 World model 本身

| 方法 | 预测空间 | WM 训练目标 | 动作如何接入 | 随机性如何表达 | 上下文 / 记忆 |
|---|---|---|---|---|---|
| World Models | VAE latent $z$（32/64 维）+ LSTM 隐状态 | VAE 重建 + KL；MDN 负对数似然（Doom 中另预测 done） | 与 $z$ 拼接后输入 LSTM | 5 分量 MDN + 温度 τ | LSTM 状态 |
| Dreamer | RSSM（确定性 $h$ + 30 维高斯 $z$） | ELBO：重建 + reward + KL（β=1，free nats 3） | RSSM transition 的输入 | 高斯 latent | RNN 状态 |
| DreamerV3 | RSSM（categorical 离散 $z$） | 重建 / reward / continue 预测 + 两项 KL（dyn 1、rep 0.1，free bits 1 nat） | 同上 | categorical latent + 1% unimix | GRU 状态 |
| TD-MPC2 | decoder-free latent（SimNorm） | stop-grad latent consistency + reward 离散回归 + value 离散回归（TD） | MLP dynamics 的输入 | 确定性 | 只看单帧，没有历史 |
| DIAMOND | 像素（64×64） | EDM 去噪（L2） | AdaGN 注入 | 扩散采样（3 步） | 4 帧 frame stack |
| Genie | VQ token（ST-ViViT） | MaskGIT masked token 预测；LAM 重建下一帧 | latent action（8 码字）以加性嵌入接入 | 离散 token 采样 | 16 帧 |
| Cosmos | 连续 latent（CV8×8×8）或 FSQ token（DV8×16×16） | EDM denoising score matching；next-token NLL | 预训练阶段没有动作；post-train 时加到 timestep embedding 或 cross-attn | 扩散或自回归采样 | Video2World 以 1 帧或 9 帧为条件（Table 20） |
| V-JEPA 2(-AC) | 冻结的 ViT-g 表征 | 表征空间 L1：预训练为 mask-denoising；AC 阶段为 teacher forcing + 2 步 rollout loss | 与 $(s,z)$ 交错成 token，用 block-causal attention | 确定性 | 训练片段 T=15 |
| Dreamer 4 | causal tokenizer latent（MSE + LPIPS，MAE 式 dropout） | shortcut forcing：x-prediction + x 空间损失 + ramp 权重 | 动作 token；无标注视频用一个可学习的空 embedding | flow 采样，每帧 K=4 | 192 帧（9.6 s） |
| DreamZero | Wan2.1 VAE 视频 latent 与动作块 | 联合 flow matching（视频与动作共享 $t$） | 动作不是条件，而是与视频一起被生成 | flow 采样 | 约 6.6 s（8 个 latent 帧） |

### 6.2 怎么用、要什么数据、要多少算力

| 方法 | 决策方式 | 需要 reward？ | 数据 | 需要与环境交互？ | 训练算力（论文口径） | 推理 |
|---|---|---|---|---|---|---|
| World Models | 线性 C + CMA-ES；CarRacing 在真实环境训练，Doom 全在梦境 | CarRacing 用环境 reward；Doom 由 done 推出存活回报 | 随机策略采 10k 局 | 需要（CarRacing 约 1.8M 局真实 rollout，为推算值） | V、M 各不到 1 GPU·h；CMA-ES 用 64 核 CPU | 一次前向 |
| Dreamer | imagination 中的 actor-critic（解析 value 梯度，H=15） | 需要（reward head） | 在线收集 | 在线，5e6 步 | 1×V100，约 3 h / 1e6 步 | 一次前向，不做规划 |
| DreamerV3 | imagination 中的 actor-critic（REINFORCE + return normalization） | 需要 | 在线收集 | 在线 | 1×A100，每 run 0.1–16.1 GPU days | 一次前向 |
| TD-MPC2 | 潜空间 MPPI（H=3）+ Q bootstrap + policy prior | 需要 | 单任务在线；多任务用 545M transitions 离线训练 | 单任务需要 | 多任务 3.7–33 GPU days（RTX 3090）；单任务未报告 | 每步都做 MPPI |
| DIAMOND | imagination 中的 actor-critic（REINFORCE + λ-return，H=15） | 需要（独立的 CNN-LSTM） | Atari 100k 在线 | 在线，100k 步 | RTX 4090 每 run 2.9 天 | policy 一次前向；imagination 每步 20.4 ms |
| Genie | 用户选 latent action 交互；另有 LAM 标签 + BC 实验 | 不需要（也没有 reward / 终止） | 30k 小时无动作互联网视频 | 不需要 | 256 TPUv5p，6.6e22 FLOPs | 约 1 FPS |
| Cosmos | 论文没做闭环；定位为 simulator、数据生成、policy 评估 | 不需要 | 约 20M 小时视频 + 小规模 post-train 数据 | 不需要 | 全部 WFM 合计 10,000 张 H100 × 3 个月 | AR 4B 生成 32 帧需 31 s（H100） |
| V-JEPA 2 | 潜空间 CEM-MPC，最小化到目标图像表征的 L1 energy | 不需要（用 goal image） | 超过 1M 小时视频 + 不到 62 小时 Droid | 不需要（zero-shot 部署） | 预训练总量未报告（推算约 7–20 GPU-years） | 每个动作 16 s（RTX 4090） |
| Dreamer 4 | BC + reward model，然后在 imagination 中用 PMPO 做 RL（冻结 transformer） | 需要（reward head 从数据学） | 2541 小时 VPT contractor 离线数据（带键鼠动作）；Figure 7 另测了只给部分动作标签的情形 | 不需要（纯离线） | 256–1024 块 TPU-v5p，总时长未报告 | 21 FPS（单张 H100） |
| DreamZero | 本身就是 policy：联合生成视频和动作，闭环时写回真实帧 | 不需要（纯 BC） | Wan2.1 视频预训练 + 约 500h AgiBot 或 DROID | 不需要 | 未报告（14B 全参数训练） | 7 Hz（2×GB200） |

**从两张表能读出的规律**
- **"要不要 reward"对应"怎么用"**：凡是在 imagination 里训练 policy 的（Dreamer 系、DIAMOND、Dreamer 4）都需要 reward head；做 goal-conditioned 规划的（V-JEPA 2）和纯 BC 的（DreamZero）都不需要。Genie 和 Cosmos 缺 reward，这正是它们还不能直接当 RL 环境用的原因。
- **推理成本对比悬殊**：从 Dreamer、DreamerV3 的一次前向，到 V-JEPA 2 每个动作 16 s，再到 V-JEPA 2 论文中 Cosmos 每个动作 4 min。想用生成式 WM 做规划或 RL，推理速度是首先要过的门槛。Dreamer 4 的 shortcut forcing 和 DreamZero 的 38× 加速都在解决这个问题。
- **记忆都很短**：从 4 帧到 9.6 s 不等。表中所有方法部署时都冻结 world model，没有一个在部署时更新参数。

---

## 7. 开放问题与争议

1. **生成式还是表征空间预测？**
   - 支持生成的证据：DIAMOND 与 IRIS 在同一份数据上的对比中，IRIS 的 rollout 出现"敌人与奖励来回互变"等不一致（只有定性证据）；DreamerV3 的 learning signal 消融显示，去掉重建梯度后性能大幅下降。
   - 支持不生成的证据：TD-MPC2 不做重建，在 104 个任务上领先，但以 state 输入为主；V-JEPA 2 的 Table 3 中，V-JEPA 2-AC 每个动作 16 s，pick-and-place 成功率 80% / 50%，Cosmos 每个动作 4 min、成功率 0% / 0%；综述汇总后认为五种范式都能拿高分，照片级生成不是必要条件。
   - 两边的证据都不是受控比较。V-JEPA 2 只在 Lab 2 测了 Cosmos，而且 CEM 样本数少 10 倍；DIAMOND 的因果论证主要靠 3 个游戏的定性图。至今没有一个固定 agent、只改变 WM 保真度的实验。

2. **视觉质量指标与决策有用性脱节**
   - Cosmos 生成视频的 3D 一致性指标优于真实视频（Sampson error 0.355 对 0.431），说明这个指标本身有偏好。
   - Genie 的 ΔtPSNR 只衡量"对动作是否敏感"，不衡量"动作语义是否正确"。
   - 综述认为 rank consistency、value fidelity 比像素精度更有信息量。
   - 本线没有一篇论文量化过"WM 内的 policy 排序 / 成功率"与真实结果的相关性：Cosmos 没有闭环实验，Dreamer 4 没有度量想象 return 与真实 return 的相关性，综述只做了定性描述。这是 learned simulator 版本的 reality gap，可以与 03 线的 [SIMPLER](../03_real2sim2real/simpler/notes.md) 对照。

3. **Model exploitation 与 compounding error**
   - World Models 的 Table 2 是最直观的反例：τ=0.1 时梦境 2086 分、真实 193 分。
   - Dreamer 和 DreamerV3 都没有不确定性估计，也没有量化 imagination 中的 model error。
   - DIAMOND 的 reward 模型在真实帧上训练、在生成帧上推理，这里的分布偏移没有量化。
   - Dreamer 4 纯离线训练，只靠 BC prior 的反向 KL 加冻结 transformer 来约束。
   - 只有 TD-MPC2 的附录 J 试过在规划时加不确定性惩罚： $c=0.01$ 时从 56.54 升到 62.01，但正文实验没有采用。
   - 在大规模 WM 里，怎样做 uncertainty-aware 的 imagination，仍然没有答案。

4. **动作可控性，以及需要多少动作数据**
   - Genie 的 8 个离散 latent action 默认场景里只有一个可控主体。
   - Cosmos 预训练阶段完全没有动作，动作只在 post-train 时加到 timestep embedding 上。
   - Dreamer 4 的实验：10 h 带动作数据时，归一化 PSNR / SSIM 为 53% / 75%；100 h 时为 85% / 100%；在只有无标注视频的 Nether/End 上外推到 76% / 80%。
   - 综述把"预测的未来更多由历史和任务意图决定，而不是由待执行的动作决定"称为 causal conditioning gap，列为头号挑战。
   - 机器人连续高维动作要多少带动作数据才能 grounding，目前没有系统研究。

5. **World model 的价值主要在 simulator，还是在表征？**
   - Dreamer 4 的 Table 7：world model 预训练带来的 BC 提升（iron pickaxe 0.6→16.9）大于 imagination RL 的提升（16.9→29.0）。
   - DreamerV3 的 representation 主要靠无监督目标塑形。
   - Fast-WAM（2603.16666）认为 WAM 的收益主要来自训练时的视频 co-training，而不是推理时想象未来，这与 DreamZero、Cosmos Policy 的主张相左。
   - 如果 world model 的主要作用是表征学习，那"在 WM 里做 RL"的真正边际收益有多大？这直接影响 05 线（WMPO、World-Gymnast）的立论。

6. **WAM 能否取代 VLA，能否做 RL？**
   - DreamZero 在未见环境上比 π0.5 高 2 倍以上，但只对齐了 batch size 和步数，14B DiT 每步的 FLOPs 远高于基线。
   - 5B 版本明显掉点（21% 对 14B 的 50%）。
   - WAM 不是以动作为条件的 simulator，本身也没有 RL。能否对 14B 的联合视频-动作模型做 RL fine-tuning（参照 [π_RL](../04_rl/pi-rl/notes.md)、[DPPO](../04_rl/dppo/notes.md)），还没有人验证。

7. **长时程记忆**
   - 各方法的记忆：DIAMOND 4 帧，Genie 16 帧，Dreamer 4 为 9.6 s，DreamZero 约 6.6 s。
   - V-JEPA 2 的 horizon 为 1，pick-and-place 要靠人给 sub-goal 图。
   - fast-weight 类的长上下文机制（[TTT layers](../02_test_time_training/ttt-layers/notes.md)、[Titans](../02_test_time_training/titans/notes.md)）能否作为 world model 的历史编码器，本线没有论文检验过。

8. **部署时适应（与 TTT 线交叉）**
   - 本线所有模型部署时都冻结。现成的自监督信号其实很多：Dreamer 的重建与 KL 损失，TD-MPC2 不需要 reward 的 latent consistency 损失，V-JEPA 2 的 JEPA 损失，Genie 的 LAM 重建误差，DreamZero 执行完每个 chunk 后"预测帧对真实帧"的误差。
   - V-JEPA 2 在 App.B.4 提出了一种无监督相机标定，但明确说没有做。
   - 综述完全没有 TTT 章节。
   - [PAD](../05_intersections/pad/notes.md) 是最接近的先例。这是本项目最直接的切入点。

9. **"一套超参数"和 scaling 的前提**
   - DreamerV3 的模型规模和 replay ratio 仍按 benchmark 选；TD-MPC2 含若干依赖 episode 长度的启发式规则。
   - Scaling 的证据都偏弱：TD-MPC2 从 48M 到 317M 只涨 2.6 分；Cosmos 14B 与 7B 的物理对齐几乎相同（9 帧条件下 IoU 0.598 对 0.592）；Genie 的 scaling 只看 training loss。

10. **可复现性**：Genie 和 Dreamer 4 没有官方代码；Cosmos 和 Genie 的训练数据私有；DreamZero 的 AgiBot 500h 数据还没开源。本线的"前沿结论"很多只能在推理或小规模机制层面验证。

---

## 8. 扩展阅读

以下论文选自 scout 阶段因名额所限没有收录的工作，arXiv ID 已于 2026-09-28 按标题核对。

| 论文 | arXiv | 为什么值得读 |
|---|---|---|
| PlaNet: Learning Latent Dynamics for Planning from Pixels | 1811.04551 | RSSM 的出处。Dreamer 系列都复用这个结构，它的 CEM 在线规划也是 Dreamer 要替代的对象 |
| DreamerV2: Mastering Atari with Discrete World Models | 2010.02193 | 离散 latent 与 KL balancing 的出处，是读懂 DreamerV3 设计选择的中间一环 |
| MuZero | 1911.08265 | value-equivalent、decoder-free 路线的概念源头（MCTS 加 learned model），TD-MPC2 的思想前身 |
| IRIS: Transformers are Sample-Efficient World Models | 2209.00588 | 离散 token + Transformer 的 world model，是 DIAMOND 的同组前作和主要对照 |
| Diffusion Forcing | 2407.01392 | 逐帧独立噪声的训练方式，是 Dreamer 4 的 shortcut forcing 和各类自回归视频 WM 的共同技术前提 |
| GameNGen: Diffusion Models Are Real-Time Game Engines | 2408.14837 | 扩散式 neural game engine（DOOM），可以和 Genie、DIAMOND 的 CS:GO 模型对照 |
| Matrix-Game 2.0 | 2508.13009 | 开源的实时、流式交互 world model，代表交互式视频 WM 走向长时程和实时的方向 |
| DINO-WM | 2411.04983 | 在冻结的 DINOv2 特征上学 world model，做 zero-shot 规划，是 V-JEPA 2-AC 最接近的前作，规模小、适合轻量复现 |
| LeWorldModel | 2603.19312 | 从像素端到端稳定训练的 JEPA world model，参数量小、单卡可训，是 JEPA 线的复现备选 |
| UniPi: Learning Universal Policies via Text-Guided Video Generation | 2302.00111 | "video as policy"（生成视频 + IDM）的早期代表，是理解 DreamZero 和综述中 IDM-style 一类的起点 |
| Cosmos Policy | 2601.16163 | 把 action、未来 state、value 当作 latent 帧塞进视频扩散模型，可以和 DreamZero 同读，比较 WAM 的两种做法 |
| DreamGen | 2505.12705 | 用视频 WM 生成 neural trajectories，再用 latent action / IDM 补动作标签，代表"WM 用于数据合成" |
| World-in-World | 2510.18135 | 闭环评测平台，给出"视觉质量不等于决策有用性、可控性更关键"的批判性证据，对应第 7 节的问题 2 |
| WorldEval / Veo World Simulator | 2505.19017 / 2512.10675 | 把视频 WM 当 policy evaluator，前者关注 policy 排序一致性，后者是工业级的 Gemini Robotics 评估 |
| DreamDojo | 2602.06949 | 用大规模人类视频训练通用机器人 world model，有代码，是 Cosmos 之后机器人 WFM 的延续 |

---

## 9. 复现推荐

按项目"先复现、再提假设"的原则，本线最值得先复现的 baseline 如下。复现报告放在 [`../../复现/`](../../复现/)。选择标准有四条：有官方代码；单卡可做；能直接检验论文的核心主张；复现产物能被 05 线和 TTT 方向复用。

### ① DreamerV3（首选）

- **理由**：它是目前最常用的通用 MBRL baseline，官方 JAX 代码成熟。"RSSM + imagination actor-critic"是 DayDreamer、Dreamer 4 以及 05 线多篇工作的骨架。它的重建和 KL 损失又是"部署时用自监督损失更新 world model"最现成的信号，复现出来的 world model 可以直接当后续 TTT 实验的底座。
- **算力**：单卡 A100。DMC Visual 用 12M 模型，按论文 Table 2 约 0.1 GPU days / run；下面的最小实验共 12 个 run，约 1.2 GPU days。
- **最小实验**：在 DMC Visual Control 上用 12M 模型、1M 步、replay ratio 512，跑 Walker Walk（论文 Table 12 为 961）和 Cartpole Swingup Sparse（759），各 3 个 seed。然后在 Cartpole Swingup Sparse 上做两个消融，检验正文 "Actor learning" 部分"稀疏奖励下 advantage normalization 会放大噪声"的论点：把 retnorm 换成 advnorm；去掉 free bits。注意这是新实验，不是复现论文数字：Figure 6a/17 的 14 个消融任务不含 Cartpole Swingup Sparse，而且论文是把 KL balance 和 free bits 一起去掉的。
- **注意**：论文文本里 T=16 与 H=15、Eq.6 的熵项符号、reward 损失的描述互相矛盾，一律以代码为准。

### ② TD-MPC2

- **理由**：它是 decoder-free 加 planning 路线的代表，和 DreamerV3 形成最干净的对照。[SimDist](../05_intersections/simdist/notes.md) 的 sim-to-real world model 就沿用了 TD-MPC 式目标。它的 latent consistency 损失不需要 reward，是 TTT 辅助任务的候选。官方提供 checkpoint，第一步只做评估、不用训练，门槛最低。
- **算力**：论文没有报告单任务算力。仓库 README（不是论文）写单任务需要至少 8GB 显存；80 任务的 19M checkpoint 只需评估，不需要训练。
- **最小实验**：
  1. 下载 80 任务的 19M checkpoint 做评估，核对 Table 1 的 57.1。在同一 checkpoint 上关掉 MPC、只用 policy prior，检验 Fig 9 中"规划优于纯 policy"的趋势：默认的 planning + policy 为 54.2，只用 policy 为 42.2（53.7 是图中单独的 "Planning" 变体；同一配置在 Table 1、Fig 9、Table 10 里分别是 57.1、54.2、56.54，只能在同一张图内部比较）。
  2. 单任务在 walker-walk 和 cheetah-run 上各跑 3 个 seed、1M 步，与 Fig 12 对比。再加一组去掉 SimNorm 或改成连续回归的消融，观察梯度范数（Fig 21）和最终 return。

### ③ DIAMOND

- **理由**：它是算力最友好的生成式（像素扩散）world model 加 imagination RL，把 MBRL 和视频生成两条线接了起来。它能直接检验两个可证伪的主张：EDM 比 DDPM 稳定，以及去噪步数影响多峰转移。复现后可以作为理解 Cosmos 和 Dreamer 4 的低成本入口。
- **算力**：单张 RTX 4090（约 12GB 显存）。完整训练一个 Atari 游戏约 2.9 天；只训扩散模型做 drift 对照约 9.7 h（按 App I 推算）。
- **最小实验**：
  1. 用 `play.py --pretrained` 做零训练检查。
  2. 在 Boxing 上把去噪步数设为 n=1 和 n=3，跑多个 seed，复现 App L 中 86.9 对 41.9 的趋势和 Figure 4 的模糊现象。
  3. 用 `ddpm` 分支在 Breakout 上复现 Figure 3 / App K 的漂移曲线。
  4. 有余力再在 Breakout 上跑单 seed 完整训练，对照 Table 1 的 132.5。

### 备选与不建议

- **备选**
  - World Models 的 DoomTakeCover 梦境训练加温度扫描。这是最便宜、最直观的 model exploitation 演示，但依赖 TF1、gym 0.9.x 和需要编译的 VizDoom。
  - V-JEPA 2 用官方权重复现 energy landscape（Fig.9）和离线动作推断误差，不用训练。
  - DreamZero 下载 DROID checkpoint，只做推理评测，需要 2 张卡。
- **不建议从头复现**：Genie 的全规模（数据私有，需要 256 TPUv5p），Cosmos 的预训练（10,000 张 H100 × 3 个月），Dreamer 4（无代码，256–1024 块 TPU-v5p，笔记判断完整复现在个人或小组条件下不可行）。这三者只适合做机制级的小规模复现，具体方案见各自笔记的"复现可行性"一节。
