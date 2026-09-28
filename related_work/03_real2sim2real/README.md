# Real-to-Sim-to-Real

> 主题目录 `related_work/03_real2sim2real/`，收录 12 篇：1 篇综述、4 篇 foundational、4 篇 milestone、3 篇 frontier。入门这条线时先读本页，再按第 5 节的路线读各篇 `notes.md`。
> 本页的数字都取自各篇笔记，笔记里的数字又能在对应论文的 `source/` 或 PDF 中找到出处。标"未收录"的论文本项目没有笔记，只给 arXiv ID。

## 1. 这条线在解决什么问题

仿真便宜、安全，可以同时跑上千个环境，还能直接给出 privileged state 和 reward。问题是仿真的外观（perception & sensing）、动力学（dynamics）、执行器与底层控制（actuation）以及系统设计都和真实世界不一致，这种不一致就是 reality gap。它会让 policy 在仿真里拿高分、到真机上失效，policy 甚至会专门利用仿真的建模误差。[Reality Gap 综述](./survey-reality-gap/notes.md) 的形式化是：sim-to-real 真正要缩小的是 policy 的 performance gap $G_{\text{perf}}=|J_{\mathcal M_s}(\pi)-J_{\mathcal M_r}(\pi)|$（前提是真机回报本身要高），reality gap 既不可能也不需要清零。

经典 sim-to-real 主要靠 domain randomization 和在线适应，去"克服"剩下的 gap。Real-to-Sim-to-Real 的做法是先拿真实世界数据构建或校准仿真，这一步是 real→sim，数据可以是手机扫描、单张照片、人类视频、少量真机 rollout 或示范；然后在仿真里训练或评测 policy；最后回到真实世界部署（→real）。真机数据还能再回头修正仿真，形成闭环。整条线要回答三个问题：仿真里哪些部分必须对齐真实；用什么真实数据去对齐；对不齐的剩余 gap 靠什么克服。

```
 真实数据（扫描 / 照片 / 视频 / 真机 rollout / 示范）
        │ real→sim：辨识参数(SimOpt, SIMPLER) · 学残差(ASAP) · 重建场景(RialTo, ACDC)
        │           · 神经渲染(SplatSim, X-Sim) · 可形变物理(PhysTwin)
        ▼
 仿真 ──训练 policy（PPO / BC / 蒸馏）或 评测 policy（SIMPLER）
        │ sim→real：DR(DR, legged-gym) · 在线推断 extrinsics(RMA) · 真实数据 co-training(RialTo)
        ▼
 真机部署 ──真机 rollout 回流，再修一次仿真或 policy（SimOpt, ASAP, X-Sim）──┐
        ▲──────────────────────────────────────────────────────────────┘
```

## 2. 发展脉络

| 年份 | 工作 | 核心做法 | 解决了前一步的什么问题 | 留下的问题（由谁接手） |
|---|---|---|---|---|
| 2017 | [Domain Randomization](./domain-randomization/notes.md) | 在 MuJoCo 里对纹理、光照、相机和 distractor 做大范围的非写实随机化，只用合成图训练物体定位网络。真实定位误差约 1.5 cm，真机抓取 38/40 | 之前有两条路。一条是写实渲染加 system identification，费力，迁移效果也有限；另一条是 domain adaptation 或 fine-tuning，需要真实数据或真实标签。DR 完全不用真实数据 | $P(\xi)$ 靠手工设定，而且只处理视觉。随机化太宽会让任务无解，或者让 policy 变保守（见 SimOpt Fig.4/5） → SimOpt、RMA |
| 2017–2019（未收录） | Dynamics randomization（1710.06537）、actuator network（1901.08652）、ADR（1910.07113） | 把随机化扩展到质量、摩擦、延迟等动力学参数；用真实电机数据学执行器模型；ADR 按 policy 的表现自动扩大随机化范围 | 把 DR 从感知推广到动力学和执行器 | 随机化分布仍不以真实系统为条件 → SimOpt |
| 2018 | [SimOpt](./simopt/notes.md) | 每轮先在 $p_\phi(\xi)$ 上用 PPO 训练 policy，再在真机上跑 3 条 rollout，然后按仿真与真实观测轨迹的 discrepancy，用 REPS 在 KL 约束内更新 Gaussian 参数分布 | DR 分布靠手调，太宽有害；迭代式 sysID 要把仿真逐步设到真实状态，需要 full state，有的还要在真实世界估计 reward。SimOpt 只用部分观测、不用真实 reward，第一次把 real→sim→real 的回路闭合 | 目标会让分布方差塌缩；参数之间可以互相补偿，可辨识性差；参数化仿真修不了结构性误差；实验用了 64 块 GPU → ASAP |
| 2021 | [legged-gym](./legged-gym/notes.md) | 仿真、reward 和 PPO 全部放在单张 GPU 上，4096 个并行机器人、每个每次迭代 24 步，加游戏式地形课程。不到 20 min 训出 rough-terrain 策略，部署到 ANYmal C | 之前的 locomotion 训练要 12–120 h，sim-to-real 迭代一轮就是几天。分钟级训练让"修改仿真后当天重训"成为可能 | 部署时完全不做适应；DR 很窄，靠 actuator network 撑着（同年的 RMA 补上了在线适应这一半） |
| 2021 | [RMA](./rma/notes.md) | Phase 1 用 PPO 训练以 privileged extrinsics $z$ 为条件的 base policy；Phase 2 让 adaptation module 从 0.5 s 的 state-action 历史回归 $z$。部署时以 10 Hz 推断 $z$，不做梯度更新 | 纯 DR 的 policy 对环境参数不敏感，偏保守；显式 SysID 要把全部物理参数估准，Table II 里 SysID（56.5%）甚至不如 Robust（62.4%）；在真机上 fine-tune 又危险 | 能适应的范围被训练时随机化的支撑集限死，而且不修正仿真（部署时继续学习的做法见 05 线的 PAD、SimDist） |
| 2021（未收录） | Planar Robot Casting（2111.04814） | 平面 casting 任务上的自监督 real2sim2real | 据本项目的调研记录，"Real2Sim2Real"这一名称出自这篇 | 任务很窄 |
| 2024 | [RialTo](./rialto/notes.md) | 用手机扫描加 GUI 搭出带铰接的 digital twin。inverse distillation 把约 15 条真实点云示范变成带 privileged state 的仿真示范，在 state 空间做 PPO+BC，再用 DAgger 加真实示范 co-training 蒸馏回点云 policy | 操作任务里手工搭逼真场景成本高；少量示范训出的 BC 对扰动很脆；真实世界 RL 慢且不安全；直接在点云上做 RL 太慢（35 h 只到 1%） | 每个场景约要 2 天；物理参数用默认值；策略只针对单一实例（在 Objaverse 仿真上只有 17%） → ACDC、SplatSim |
| 2024 | [SplatSim](./splatsim/notes.md) | PyBullet 只负责物理。把真实场景的 3DGS 按刚体切开，每一步按仿真位姿做刚体变换后渲染 RGB，只用这些数据训练 Diffusion Policy | 深度、点云这类 gap 较小的模态已经能 zero-shot 迁移，RGB 却很少作为主模态：以前要靠 DR 逐任务调参，或者干脆改用深度或点云（RialTo 就是点云）。SplatSim 让 RGB policy zero-shot 达到 86.25%，用真实数据训练的是 97.5% | 只处理刚体和扫描过的那一个场景；缺 mesh+DR 的对照；没有接 RL → X-Sim、GaussGym |
| 2024 | [ACDC](./acdc-digital-cousins/notes.md) | 从单张 RGB 自动抽取物体，用 DINOv2 在 BEHAVIOR-1K 资产库里检索一组 digital cousins，在 cousins 上用脚本示范做 BC | digital twin 要人工搭（RialTo 的 GUI），而且只对单一实例优化；在全部资产上训练（naive DR）到了目标场景反而掉点。真实开柜门任务上 cousin 90%，twin 25% | 真实世界的定量结果只有 1 个任务、每个策略 20 次；只补几何和语义 gap，不补物理 → 2026 年的 SimFoundry |
| 2024 | [SIMPLER](./simpler/notes.md) | 离线把示范动作 open-loop 回放，辨识 PD 参数；用 green screening 加 texture matching 对齐视觉。拿这样的仿真评测在真实数据上训练的 generalist policy，并提出 MMRV 指标 | 真实评测贵，也难复现；仿真里训、仿真里测的 benchmark 反映不了真实表现；full-fidelity twin 不可扩展。SIMPLER 只要求仿真与真实的相对排名一致 | 只保证排名，不保证绝对成功率；每次只比 3–6 个策略，没有置信区间；被当作 RL 训练环境之后相关性是否还成立没有验证 → PolaRiS、05 的 Ctrl-World |
| 2025 | [ASAP](./asap/notes.md) | 用真机轨迹训练 delta action model $\pi^\Delta(s,a)$，训练方法是 PPO，奖励是仿真状态与真实状态的匹配程度。之后冻结它，在 $f^{sim}(s,a+\pi^\Delta)$ 里微调 policy，部署时去掉 $\pi^\Delta$ | SysID（例如 SimOpt）只能在预先定义的参数空间里调，参数化之外的动力学补不了；DR 会让高敏捷动作变保守 | 需要 MoCap，采数据时损坏了两台 G1；真机只验证了 2 个动作；动作空间的残差表达不了与动作无关的 gap |
| 2025 | [PhysTwin](./phystwin/notes.md) | 从一段三视角 RGB-D 人手交互视频重建可形变物体，模型由 spring-mass、TRELLIS 补全的几何和 Gaussians 组成。先用 CMA-ES 确定拓扑和低维参数，再用可微仿真优化上万根弹簧的刚度 | 之前的 real2sim（RialTo、SplatSim、SIMPLER）都假设刚体或铰接刚体；dynamic 3DGS 没有物理，做不了 action-conditioned 预测；GNN 学动力学需要大量数据 | 没有"在 twin 里训 policy 再上真机"的证据；只能辨识类似 $k/m$ 的比值 |
| 2025 | [X-Sim](./x-sim/notes.md) | 从没有动作标签的 RGBD 人类视频里只取物体的 6D 轨迹，作为 object-centric reward，在 2DGS 重建的 ManiSkill 场景里训练 PPO，再蒸馏成 RGB Diffusion Policy。部署后在仿真里 replay 真实 rollout，得到 sim/real 成对图像，用 InfoNCE 做校准 | RialTo 仍需要真实机器人示范；手部 retargeting 在运动学或动力学上常常不可行；SplatSim 没有 RL | 需要物体 mesh 和场景扫描；只校准视觉，不校准物理；每个任务只有 10 次试验 |
| 2025 | [Reality Gap 综述](./survey-reality-gap/notes.md) | 把 gap 拆成 dynamics、perception、actuation、system design 四类，把方法分成 reducing 和 overcoming 两类，并整理了 SRCC、offline replay error 等指标 | 此前的讨论常把 gap 当成"一个东西" | 没有给出从 $G_{\text{dyn}}$、$G_{\text{perc}}$ 到 $G_{\text{perf}}$ 的界；对 world model 和 TTT 覆盖很薄 |
| 2026（未收录） | TwinRL（2602.09023）、D-REX（2603.01151）、SimFoundry（2606.28276）、Agentic Real2Sim（2607.19190） | 据本项目的调研记录：TwinRL 用数字孪生引导真实世界 RL；D-REX 用可微 real2sim 辨识质量；SimFoundry 从单段视频生成数字孪生和 digital cousins；Agentic Real2Sim 由 VLM agent 驱动 real2sim | 趋势是全自动化和闭环 | 都很新，代码开源情况未确认 |

另有一条并行的路线：不用物理引擎，改用从数据（真实数据、仿真数据或两者）学出的 world model 当 simulator，例如 [UniSim](../05_intersections/unisim/notes.md)、[RWM](../05_intersections/robotic-world-model/notes.md)、[Ctrl-World](../05_intersections/ctrl-world/notes.md)、[World-Gymnast](../05_intersections/world-gymnast/notes.md)、[SimDist](../05_intersections/simdist/notes.md)。综述 Sec.6.3 把它列为开放问题，本项目把它放在 `05_intersections`。

## 3. 技术分类

上层分类用综述的 **reducing the gap**（用真实数据改仿真）和 **overcoming the gap**（不改仿真，让 policy 对剩余 gap 鲁棒或自适应），下层按调研阶段整理的子方向，另外按"用途"单列评测。综述 Fig.3 用色块标出每种方法针对 perception（P）、actuation（A）、dynamics（D）、system（S）中的哪几类 gap。下表"综述标注"一列照抄 Fig.3 的标注；综述笔记提醒过，Fig.3 和正文有几处不一致，这一列只能当粗略参考。

### 3.1 Overcoming the gap

| 子方向 | 机制 | 综述标注 | 本项目收录 | 未收录代表作 |
|---|---|---|---|---|
| Domain randomization | 训练时随机化 nuisance 参数，目标是让真实环境落进随机化分布的支撑集 | Domain randomization：D P A S | [Domain Randomization](./domain-randomization/notes.md)（视觉）、[legged-gym](./legged-gym/notes.md)（摩擦、推力、观测噪声） | Dynamics randomization（1710.06537） |
| 自动调整 DR 分布 | 按 policy 的表现，或者按 LLM 的建议，自动设定随机化范围 | 综述放在 DR 条目下（与 DROPO、SimOpt 并列） | —（[SimOpt](./simopt/notes.md) 用真实数据调分布，本页归在 3.2） | ADR（1910.07113）、DrEureka（2406.01967） |
| Privileged information 与在线适应 | teacher 能看到仿真的 privileged state，student 或 adaptation module 从历史或传感器推断；RMA 相当于隐式的 online system identification | Privileged information：D P A；Meta learning：D P A S（综述在这两处都以 RMA 为例） | [RMA](./rma/notes.md)；teacher-student 蒸馏也出现在 [RialTo](./rialto/notes.md) 和 [X-Sim](./x-sim/notes.md)，asymmetric actor-critic 出现在 [ASAP](./asap/notes.md) | — |
| 大规模并行仿真 RL | GPU 向量化仿真加 on-policy PPO，是几乎所有 locomotion sim-to-real 的训练底座 | 不是 Fig.3 中的方法；综述在 Sec.1 把它列为 locomotion 突破的推动因素 | [legged-gym](./legged-gym/notes.md) | — |
| 真实数据 co-training 与表征对齐 | 训练时混入真实示范，或者用 sim/real 成对图像对齐观测编码器 | Incorporating real-world data：D P A S；Representation learning：P | [RialTo](./rialto/notes.md)（Eq.2 真实示范 co-training）、[X-Sim](./x-sim/notes.md)（InfoNCE 校准） | Sim-and-Real Co-Training（2503.24361） |

### 3.2 Reducing the gap

综述在 "Improving simulation fidelity" 下列了三项：System Identification（Fig.3 标 D）、Learned Residual Models（D P A S）、Real-to-Sim First（D P A S）。下表前两行对应前两项，后四行都属于 Real-to-Sim First，也就是先用真实数据把几何、运动学与动力学、渲染构建出来。

| 子方向 | 从真实数据得到什么 | 本项目收录 | 未收录代表作 |
|---|---|---|---|
| 参数分布校准与 system identification（刚体、控制器） | 仿真参数或参数分布（摩擦、阻尼、PD 增益、action scaling 等） | [SimOpt](./simopt/notes.md)（closed-loop 轨迹 discrepancy 加 REPS）、[SIMPLER](./simpler/notes.md)（open-loop 回放加 simulated annealing，只辨识 PD） | BayesSim（1906.01728，求参数后验并用作 DR 分布）、ASID（2404.12308，主动探索式辨识）、D-REX（2603.01151） |
| Learned residual models | 一个网络，补上参数化仿真表达不了的部分。综述把它分成"改输出"（修正仿真预测的状态）和"改输入"（修正施加的动作） | [ASAP](./asap/notes.md)（改输入的 delta action model，延续 grounded action transformation 一脉；论文里的 DeltaDynamics 基线属于改输出） | Actuator network（1901.08652，用真实数据学执行器模型接进仿真） |
| 可形变物体的物理辨识 | 拓扑、稠密刚度、碰撞参数 | [PhysTwin](./phystwin/notes.md) | PhysGaussian（2311.12198，偏图形学） |
| 场景与资产重建：digital twin 和 digital cousins | 带铰接的 mesh 场景；或者从资产库检索出的一组相近资产 | [RialTo](./rialto/notes.md)（human-assisted digital twin）、[ACDC](./acdc-digital-cousins/notes.md)（全自动的 digital cousins） | URDFormer（2405.11656）、SimFoundry（2606.28276）、Agentic Real2Sim（2607.19190） |
| 神经渲染仿真（NeRF / 3DGS） | 照片级的外观模型，由物理引擎驱动它运动 | [SplatSim](./splatsim/notes.md)（3DGS）、[X-Sim](./x-sim/notes.md)（2DGS）、[PhysTwin](./phystwin/notes.md)（Gaussians 加 LBS） | NeRF2Real（2210.04932）、GaussGym（2510.15352） |
| Video-to-sim | 从人类或互联网视频中提取任务定义（物体轨迹）或物理参数 | [X-Sim](./x-sim/notes.md)、[PhysTwin](./phystwin/notes.md)（从人手交互视频辨识物理） | Video2Policy（2502.09886）、VideoMimic（2505.03729） |

reducing 一侧还有 "Choice of Modalities and Representations" 和 "Design choices" 两类手段：换成 gap 更小的传感模态，例如 depth 或点云；换动作空间；用高频 impedance 控制器；把任务限制在 quasi-static 区间。本线里 [RialTo](./rialto/notes.md) 和 [ACDC](./acdc-digital-cousins/notes.md) 用点云 policy，就是在利用前一类。

### 3.3 按用途：real-to-sim 评测

| 做法 | 本项目收录 | 未收录代表作 |
|---|---|---|
| 搭一个"够真"的仿真去评测在真实数据上训练的 policy，用排名一致性（Pearson、MMRV、SRCC）衡量仿真的预测力 | [SIMPLER](./simpler/notes.md) | PolaRiS（2512.16881）、RobotArena∞（2510.23571） |

### 3.4 这条线和本项目另外三条线的接口

- **World model**：物理引擎加重建出来的外观，可以看成一个结构化、动力学由人工指定的 world model。[PhysTwin](./phystwin/notes.md) 的 Eq.2（transition 加 observation model，靠最小化观测误差来拟合）和 RSSM 的训练目标同构。学出来的 simulator 覆盖面更广，但有 compounding error 和幻觉，[ASAP](./asap/notes.md) 的 DeltaDynamics 基线在 Fig.5 里就展示了这种误差放大。
- **TTT / TTA**：这条线里的适应都不是部署时的梯度更新。RMA 做前向推断，SimOpt 和 ASAP 在两次部署之间离线修正，X-Sim 在部署后离线再训练一次。部署时用自监督梯度更新的做法见 [PAD](../05_intersections/pad/notes.md)，对照 [TTT](../02_test_time_training/ttt-self-supervised/notes.md) 和 [Tent](../02_test_time_training/tent/notes.md)。
- **RL**：仿真一侧基本都是大规模并行的 on-policy [PPO](../04_rl/ppo/notes.md)。RialTo 走"IL 预训练 + RL fine-tuning"，和 [DPPO](../04_rl/dppo/notes.md)、[SimpleVLA-RL](../04_rl/simplevla-rl/notes.md)、[π_RL](../04_rl/pi-rl/notes.md) 同一范式；[HIL-SERL](../04_rl/hil-serl/notes.md) 则是绕开仿真、直接在真机上做 RL 的对照路线。

## 4. 论文总表

| 阅读顺序 | 论文 | 年份 | 层级 | 一句话 | 代码 | 笔记 |
|---|---|---|---|---|---|---|
| 1 | The Reality Gap in Robotics: Challenges, Solutions, and Best Practices（2510.20808） | 2025 | survey | 把 reality gap 拆成 dynamics、perception、actuation、system design 四类，方法分成 reducing 与 overcoming 两类；目标是缩小 performance gap，不是清零 reality gap | 无（综述） | [notes](./survey-reality-gap/notes.md) |
| 2 | Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World（1703.06907） | 2017 | foundational | 只用非写实的随机纹理、光照和相机渲染图训练定位网络，零真实数据就能在真机上达到约 1.5 cm 精度 | 无官方代码 | [notes](./domain-randomization/notes.md) |
| 3 | Learning to Walk in Minutes Using Massively Parallel Deep RL（2109.11978） | 2021 | foundational | 单 GPU 上 4096 个并行机器人跑 PPO，加地形课程和 time-out bootstrap，不到 20 min 训出能上真机的 rough-terrain 策略 | [leggedrobotics/legged_gym](https://github.com/leggedrobotics/legged_gym) | [notes](./legged-gym/notes.md) |
| 4 | RMA: Rapid Motor Adaptation for Legged Robots（2107.04034） | 2021 | foundational | PPO 训练以 privileged extrinsics 为条件的 base policy，再让 adaptation module 从 0.5 s 历史回归 extrinsics，部署时靠前向推断在线适应 | 无官方代码（项目页链到基于 RMA 代码的后续工作 antonilo/rl_locomotion） | [notes](./rma/notes.md) |
| 5 | Closing the Sim-to-Real Loop: Adapting Simulation Randomization with Real World Experience（1810.05687） | 2018 | foundational | 用每轮 3 条真机 rollout 的观测 discrepancy，通过 REPS 在 KL 约束下更新 DR 参数分布，不需要真实 reward | 无 | [notes](./simopt/notes.md) |
| 6 | Reconciling Reality through Simulation: A Real-to-Sim-to-Real Approach for Robust Manipulation（2403.03949） | 2024 | milestone | 手机扫描建 digital twin，用 inverse distillation 把少量真实示范转成仿真示范，PPO+BC 强化后再蒸馏回点云 policy | [RialToPolicyLearning](https://github.com/real-to-sim-to-real/RialToPolicyLearning) | [notes](./rialto/notes.md) |
| 7 | SplatSim: Zero-Shot Sim2Real Transfer of RGB Manipulation Policies Using Gaussian Splatting（2409.10161） | 2024 | milestone | 物理交给 PyBullet，外观用按刚体切分、随仿真位姿变换的 3DGS 渲染，纯仿真数据训出的 RGB policy 可 zero-shot 上真机 | [qureshinomaan/SplatSim](https://github.com/qureshinomaan/SplatSim)（项目页给的官方仓库） | [notes](./splatsim/notes.md) |
| 8 | Automated Creation of Digital Cousins for Robust Policy Learning（2410.07408） | 2024 | milestone | 从单张 RGB 自动检索生成一组 digital cousins，在上面做 BC，比只在 digital twin 上训练更鲁棒 | [cremebrule/digital-cousins](https://github.com/cremebrule/digital-cousins) | [notes](./acdc-digital-cousins/notes.md) |
| 9 | Evaluating Real-World Robot Manipulation Policies in Simulation（2405.05941） | 2024 | milestone | 用离线 SysID 对齐控制、用 green screening 加 texture matching 对齐视觉，让仿真里的策略排名和真实强相关，并提出 MMRV | [simpler-env/SimplerEnv](https://github.com/simpler-env/SimplerEnv) | [notes](./simpler/notes.md) |
| 10 | ASAP: Aligning Simulation and Real-World Physics for Learning Agile Humanoid Whole-Body Skills（2502.01143） | 2025 | frontier | 用 RL 从真机轨迹学 delta action model 来对齐仿真动力学，在对齐后的仿真里微调人形 motion tracking policy | [LeCAR-Lab/ASAP](https://github.com/LeCAR-Lab/ASAP) | [notes](./asap/notes.md) |
| 11 | PhysTwin: Physics-Informed Reconstruction and Simulation of Deformable Objects from Videos（2503.17973） | 2025 | frontier | 从三视角 RGB-D 交互视频重建可形变物体，由 spring-mass、生成式几何和 Gaussians 组成；CMA-ES 加可微仿真分层辨识物理参数 | [Jianghanxiao/PhysTwin](https://github.com/Jianghanxiao/PhysTwin) | [notes](./phystwin/notes.md) |
| 12 | X-Sim: Cross-Embodiment Learning via Real-to-Sim-to-Real（2505.07096） | 2025 | frontier | 把人类视频里的物体轨迹当作 object-centric reward，在重建场景里训 PPO，蒸馏成 RGB policy，部署后用 InfoNCE 校准，全程不需要机器人示范 | [portal-cornell/X-Sim](https://github.com/portal-cornell/X-Sim) | [notes](./x-sim/notes.md) |

## 5. 推荐学习路线

先读 overcoming 一侧，建立"仿真里训、真机上用"的基本配方；再读 reducing 一侧，看真实数据怎样进入仿真；最后读三篇前沿，它们各自补上一类前面没解决的 gap。

### 阶段 0：本线之外的前置

- [ ] MDP/POMDP、discounted return、belief：看 [RL 综述](../04_rl/survey-rl-overview/notes.md) 的相关章节
- [ ] PPO 的 clipped objective、GAE（$\gamma,\lambda$）、value bootstrapping：看 [PPO](../04_rl/ppo/notes.md)
- [ ] Behavior cloning，以及 DAgger 为什么能缓解 covariate shift

### 阶段 1：框架与 sim-to-real 基础（overcoming the gap）

**前置 checklist**
- [ ] domain randomization 和 system identification 的区别，以及它们各自的代价
- [ ] privileged information、teacher-student 蒸馏、asymmetric actor-critic
- [ ] MDP 里 termination 和 time-limit truncation 的区别
- [ ] 关节 PD 位置控制（action 是期望关节角，不是力矩）

**要读的笔记**：[Reality Gap 综述](./survey-reality-gap/notes.md)（这一阶段只读 Sec.2–4 的框架和分类）→ [Domain Randomization](./domain-randomization/notes.md) → [legged-gym](./legged-gym/notes.md) → [RMA](./rma/notes.md)

**读完应能回答**
1. reality gap（Eq.2）和 performance gap（Eq.4）分别依赖什么？为什么单独把 $\min_\pi G_{\text{perf}}$ 当训练目标会退化？
2. DR 论文里，为什么目标物体自身的纹理也必须随机化？去掉训练时的 distractor 后，有干扰物时的误差从 1.8 cm 升到 7.2 cm，这说明网络学到了什么捷径？
3. legged-gym 固定 batch $B=n_{robots}\cdot n_{steps}$ 时，$n_{robots}$ 过大和过小分别靠什么机制掉性能？time-out 时为什么要用 critic 做 bootstrap？
4. RMA 为什么回归 $z=\mu(e)$ 而不是物理参数 $e$？Table II 中 SysID 低于 Robust 说明了什么？部署时 RMA 一个权重都不更新，那它的"适应"具体是什么？
5. legged-gym 和 RMA 分别靠什么对抗 reality gap？遇到训练随机化范围之外的环境时，两者各自会怎样失效？

### 阶段 2：用真实数据改仿真（reducing the gap）——参数、场景、外观与评测

**前置 checklist**
- [ ] REPS 或 CEM、CMA-ES 这类 KL 约束或采样式的黑箱优化；多元 Gaussian 与 KL divergence
- [ ] 3D 重建基础：SfM/MVS、NeRF、3D Gaussian Splatting（$\mu$、$\Sigma$、opacity、SH）；URDF/USD 铰接、碰撞体凸分解
- [ ] 齐次变换、坐标系链式变换、ICP 配准
- [ ] Diffusion Policy、BC-RNN-GMM 这类 BC policy；点云编码器（PointNet）
- [ ] DINOv2、CLIP、Grounded-SAM 等视觉基础模型的用途
- [ ] Pearson、Spearman 相关系数

**要读的笔记**：[SimOpt](./simopt/notes.md) → [RialTo](./rialto/notes.md) → [SplatSim](./splatsim/notes.md) → [ACDC](./acdc-digital-cousins/notes.md) → [SIMPLER](./simpler/notes.md)

**读完应能回答**
1. SimOpt 的 Eq.2 为什么没法直接优化？Eq.3 做了哪两处近似？"最小化期望 discrepancy"的理论最优分布是什么，和 DR 的初衷有什么矛盾？
2. SimOpt 为什么在仿真里用 closed-loop policy rollout，而 SIMPLER 用 open-loop 回放示范动作？两者各自辨识了什么、需要什么真实数据？
3. RialTo 的 inverse distillation"反"在哪里？Table X 里示范条数和成功率呈阶跃关系，为什么？PPO 目标里的 BC 项同时解决了哪两个问题？
4. SplatSim 只对 Gaussian 的 $\mu,\Sigma$ 做刚体变换，哪些外观效应会因此出错？数据增强的消融（21%→86.25%）说明了什么？论文缺了哪个对照，所以分不清 3DGS 本身的贡献？
5. RialTo Fig.7（在 Objaverse 资产上训练的策略，真实只有 10%）和 ACDC Fig.5（twin 策略真实 25%，cousin 90%）看起来结论相反，能否调和？各自的实验条件差在哪里？
6. SIMPLER 为什么用 MMRV 而不只用 Pearson？domain randomization 在训练中有效，为什么 Variant Aggregation 在评测中输给 Visual Matching？

### 阶段 3：前沿——动力学残差、可形变物理、人类视频

**前置 checklist**
- [ ] Residual learning 与 grounded action transformation 的"改输入 / 改输出"之分（综述 Sec.4.1.1）
- [ ] DeepMimic 式的 motion tracking（phase、RSI、early termination）和 MPJPE 指标
- [ ] spring-mass-damper、semi-implicit Euler、可微仿真，以及接触为什么会让梯度不连续
- [ ] 6D 物体位姿跟踪（FoundationPose）；对比学习 InfoNCE

**要读的笔记**：[ASAP](./asap/notes.md) → [PhysTwin](./phystwin/notes.md) → [X-Sim](./x-sim/notes.md)

**读完应能回答**
1. ASAP 的 delta action model 在 RL 意义下的"环境""动作""奖励"分别是什么？为什么说它能绕开"对不可微仿真器求导"？为什么 fixed-point 和梯度法这种单步匹配在 Fig.11 里比不微调还差？
2. DeltaDynamics（residual state）和 ASAP（residual action）在 open-loop 1.0 s 时，$E_{mpjpe}$ 和 $E_{g\text{-}mpjpe}$ 的排序为什么相反？这对"把 learned model 嵌进仿真器"有什么启示？
3. PhysTwin 为什么要先用 zero-order 再用 first-order？只用 first-order 时，重建指标更好、future prediction 最差，说明了什么？
4. X-Sim 的 object-centric reward 为什么能绕开 embodiment gap？InfoNCE 校准的正负样本怎么构造？机器人接触物体之后，仿真和真实的物体运动开始分叉，这时正样本对还成立吗？
5. SimOpt、ASAP、PhysTwin 都用真实数据修仿真的动力学，它们分别修的是什么表示（参数分布、动作残差、稠密物理参数）？各自表达不了哪类 gap？

### 阶段 4：收尾与跨线

**前置 checklist**
- [ ] world model（learned simulator）的基本概念，以及 model exploitation、compounding error：看 [01 README](../01_world_model/README.md) 第 1 节
- [ ] test-time training 的基本设定：部署时用自监督辅助任务更新参数，看 [TTT](../02_test_time_training/ttt-self-supervised/notes.md)
- [ ] （读 SimDist 用）MPC / MPPI，以及 TD-MPC 式 latent world model，看 [TD-MPC2](../01_world_model/tdmpc2/notes.md)

**要读的笔记**：回到 [Reality Gap 综述](./survey-reality-gap/notes.md) 的 Sec.5（指标）和 Sec.6（开放问题），然后按兴趣读 05 交叉线：[PAD](../05_intersections/pad/notes.md)（部署时用自监督做适应）、[RWM](../05_intersections/robotic-world-model/notes.md)（神经网络 simulator 上跑 PPO）、[SimDist](../05_intersections/simdist/notes.md)（在仿真里预训练 world model，到真实世界只 finetune latent dynamics）、[Ctrl-World](../05_intersections/ctrl-world/notes.md)（用 world model 评测策略）。

**读完应能回答**
1. 构造一个 offline replay error 很大、SRCC 却很高的场景，再构造一个 replay error 很小、$G_{\text{perf}}$ 却很大的场景。
2. 把本线每篇论文放到"适应在何处发生（仿真侧或 policy 侧）× 何时发生（部署前、两次部署之间、部署时）"的二维表里。哪一格还空着？
3. 物理引擎加重建的 simulator 和 learned world model simulator 在"能否给出 privileged state 和 reward""compounding error""能覆盖的物体类型"三点上各有什么取舍？

## 6. 关键机制对比表

### 6.1 真实数据怎样进入仿真（real→sim 一侧）

| 方法 | 主要针对的 gap | 从真实数据估计或构建什么 | 所需真实数据 | 优化方式 / 训练信号 | 发生时机 |
|---|---|---|---|---|---|
| [DR](./domain-randomization/notes.md) | 视觉（P） | 什么都不估计，$P(\xi)$ 手工设定 | 无 | — | 训练前 |
| [legged-gym](./legged-gym/notes.md) | 动力学、执行器、观测噪声 | 执行器模型：沿用 Hwangbo 2019 用真实电机数据训练的 LSTM actuator network | 执行器数据（仓库自带训练好的网络） | — | 训练前 |
| [RMA](./rma/notes.md) | 动力学（摩擦、负载、电机强度、地形） | 不修改仿真；部署时从 0.5 s 历史推断 8 维 $z$ | 无 | 在仿真里监督回归 $z$ | 部署时（10 Hz 前向推断） |
| [SimOpt](./simopt/notes.md) | 动力学 | Gaussian 参数分布 $p_\phi(\xi)$，full covariance | 每轮 3 条真机 rollout 的部分观测（关节角，加上 DART 跟踪的物体位置），不需要 reward | 加权 $\ell_1+\ell_2$ 轨迹 discrepancy 加 REPS（KL ≤ ε） | 两次部署之间，迭代进行 |
| [RialTo](./rialto/notes.md) | 几何 / 运动学、点云观测 | 带铰接的 mesh 场景 $\mathcal S=\{\mathcal G,\mathcal K,\mathcal P\}$，物理参数取默认值 | 手机扫描，加约 15 条真实示范 | 3D reconstruction 加 GUI 人工切分；inverse distillation | 训练前（每场景约 2 天） |
| [SplatSim](./splatsim/notes.md) | 视觉（RGB） | 按刚体分割的 3DGS，以及 splat 坐标系到仿真坐标系的变换 | 场景的多视角视频，加每个物体的扫描 | 3DGS 重建，加手动分割、ICP 对齐、KNN 分割 link | 训练前 |
| [ACDC](./acdc-digital-cousins/notes.md) | 几何 / 语义 | 从 BEHAVIOR-1K 检索出的一组 cousins 及其位姿、尺度 | 单张 RGB（加相机内参） | CLIP 加 DINOv2 最近邻的层级检索，GPT 精修 | 训练前（抽取约 7 s/物体，匹配约 20 s/cousin，生成不到 30 s/场景） |
| [SIMPLER](./simpler/notes.md) | 控制 + 视觉 | PD 刚度和阻尼；green screen 背景；物体纹理 | 现有数据集里的示范轨迹，加真实首帧图像 | open-loop 回放加 simulated annealing；nvdiffrast 位姿精修 | 评测前 |
| [ASAP](./asap/notes.md) | 动力学 / 执行器 | delta action model $\pi^\Delta(s,a)$；真机上只做了 4-DoF 踝关节 | 真机 rollout（MoCap 加本体感知）；真机用了 100 条 clip 加 10 分钟 locomotion 数据 | 以仿真与真实状态匹配为 reward 的 PPO | 两次部署之间（只做一轮） |
| [PhysTwin](./phystwin/notes.md) | 可形变物体的动力学 + 外观 | 拓扑、上万根弹簧的刚度、碰撞参数、Gaussians | 一段 1–10 s 的三视角 RGB-D 人手交互视频 | CMA-ES 加 Warp 可微仿真；几何、运动、渲染三项 loss | 训练前（离线） |
| [X-Sim](./x-sim/notes.md) | 视觉 + embodiment | 2DGS 场景、物体 mesh、物体 6D 轨迹（用作 reward）；部署后再对齐观测编码器 | 一段 RGBD 人类视频加手机扫描；部署后 10 条真实 rollout（失败的也用） | FoundationPose 跟踪；InfoNCE（权重 0.1） | 训练前，加部署后离线再训练一次 |

### 6.2 Policy 一侧与证据规模

| 方法 | Policy 学习方式 | 观测模态 | 部署时是否更新 | 用途 | 论文报告的算力 | 真实世界证据 |
|---|---|---|---|---|---|---|
| [DR](./domain-randomization/notes.md) | 监督回归（VGG-16） | 单目 RGB | 否 | 训练（感知模块） | 未报告 | 480 张真实图；抓取 38/40 |
| [legged-gym](./legged-gym/notes.md) | PPO（4096 个并行环境） | 本体感知加 108 个高度点 | 否 | 训练 | 1×RTX A6000，rough 地形不到 20 min | ANYmal C，只有定性展示 |
| [RMA](./rma/notes.md) | PPO，再监督训练 adaptation module | 本体感知 | 不更新权重，只推断 $\hat z$ | 训练 | 1 GPU 台式机，Phase 1 约 24 h，Phase 2 约 3 h | A1 室内每任务 5 次；户外未报告次数 |
| [SimOpt](./simopt/notes.md) | PPO | 关节角加物体位置 | 否 | 训练 | 64 GPU；每轮 RL 约 7 或 22 min | 2 个任务，各 20 次：peg 90%，drawer 20/20；真机没有 baseline |
| [RialTo](./rialto/notes.md) | state 空间 PPO+BC，再用 DAgger 加真实 co-training 蒸馏成点云 policy | 点云 | 否 | 训练 | RTX 2080 或 3090，每个任务约 2 天 3 h | 6 个任务加 in-the-wild，每格约 10 次；平均 91/77/75% 对 BC 的 25/11/5% |
| [SplatSim](./splatsim/notes.md) | Diffusion Policy（BC） | 2 路 RGB 加末端位姿 | 否 | 训练 | 只报告了部署用的 3080Ti | 4 个任务 × 40 次，86.25% |
| [ACDC](./acdc-digital-cousins/notes.md) | BC-RNN-GMM | 点云 | 否 | 训练 | 未报告 | 1 个任务（开柜门）20 次：cousin 90%，twin 25% |
| [SIMPLER](./simpler/notes.md) | 不训练，评测现成策略 | RGB（被测策略的输入） | — | 评测 | 1×RTX 4090，每秒 3.5k 仿真步 | 约 1500 个真实 episode 作对照；Bridge MMRV 0.014，$r=0.890$ |
| [ASAP](./asap/notes.md) | PPO 预训练，再在修正后的仿真里 RL 微调 | 本体感知加 phase | 否（部署时去掉 $\pi^\Delta$） | 训练 | 未报告 | G1 上 2 个动作，只和 Vanilla 比 |
| [PhysTwin](./phystwin/notes.md) | 不训练 policy，只有定性的 model-based planning | — | — | 仿真 / 规划 | 未报告（README：CMA-ES 约 12 min，first-order 约 5 min） | 没有 sim→real 的定量结果 |
| [X-Sim](./x-sim/notes.md) | privileged-state PPO，蒸馏成 RGB Diffusion Policy，再加 InfoNCE | 单张 RGB | 部署后离线再训练一次 | 训练 | 未报告（1,024 个并行环境） | 5 个任务 × 10 次，平均 task progress 0.906（未校准 0.832） |

**从这两张表能看出的规律**
- **对真实数据的要求分成几档**：真机 rollout（SimOpt、ASAP，ASAP 还要 MoCap）、真实示范（RialTo）、现成数据集（SIMPLER）、扫描或视频（SplatSim、PhysTwin、X-Sim）、单张照片（ACDC）。越往后越不需要机器人本身，但依赖的基础模型和人工对齐步骤越多，例如 mesh、ICP、FoundationPose、TRELLIS；而且越往后越只对齐几何和外观，动力学用默认值，PhysTwin 是例外。
- **没有一种方法同时补齐视觉和动力学**。SplatSim、ACDC、X-Sim 不做物理辨识；SimOpt、ASAP 不管视觉；RialTo 用默认物理，靠 BC 正则和 co-training 兜底。
- **部署时的适应几乎都是"不更新权重"**。唯一例外是 X-Sim 的离线再训练，而 RMA 的适应是前向推断。部署时的梯度式更新在这条线里还没人做，这正是和 TTT 线的交叉口。

## 7. 开放问题与争议

1. **追求高保真的 digital twin，还是追求多样的分布？** 支持 twin 的证据：[RialTo](./rialto/notes.md) Fig.7 里，在 4 个 Objaverse 抽屉上训练的策略到真实目标抽屉只有 10%，twin 策略是 90%，混入合成资产也没有提升（Table XI）；[SIMPLER](./simpler/notes.md) 的评测中，Visual Matching 也优于 DR 风格的 Variant Aggregation。支持多样性的证据：[ACDC](./acdc-digital-cousins/notes.md) Fig.5 里 twin 25%、cousin 90%；而在全部资产上训练（naive DR）在分布内反而最差。一个可能的调和是训练和评测的需求不同：评测时被测策略没见过随机化分布，训练时则需要覆盖。但两边的真实证据都薄，ACDC 只有 1 个任务、每策略 20 次，RialTo 每格约 10 次，目前还缺一个在同一任务、同一真实场景上对比 twin、cousins 和 DR 的受控实验。
2. **调参数、学残差，还是学整个模型？** [SimOpt](./simopt/notes.md) 的参数化仿真修不了结构性误差；它的目标在理论上的最优解是 Dirac 分布，最终保留多少随机化宽度只取决于 KL 步长和迭代次数。[RMA](./rma/notes.md) Table II 里，估物理参数的 SysID（56.5%）不如不做估计的 Robust（62.4%），这和综述 Sec.6.1 的 objective mismatch 是一致的：模型准不准，和它对控制是否有用，是两回事。[ASAP](./asap/notes.md) 在 open-loop 1.0 s 上，$E_{g\text{-}mpjpe}$ 为 37.9，远低于 SysID 的 77.6 和 DeltaDynamics 的 68.1；但它的 SysID 基线只在每个参数 3 个离散值中搜索，偏弱，而且动作空间的残差表达不了外部扰动和接触差异。[PhysTwin](./phystwin/notes.md) 显式辨识物理参数，可只能辨识类似 $k/m$ 的比值，也没验证辨识对了能不能带来更好的迁移。再往前一步，就是 05 线的 [RWM](../05_intersections/robotic-world-model/notes.md) 和 [SimDist](../05_intersections/simdist/notes.md)，它们把整个 dynamics 换成神经网络，于是 compounding error 成了主要问题。
3. **用什么指标衡量"仿真修好了"？** SimOpt 附录 A 说 closed-loop 匹配能给出有意义的参数更新，open-loop 重放则不能，但只有定性描述。SIMPLER 用 open-loop 回放辨识 PD。ASAP 发现单步匹配不如 RL 做的多步匹配。综述指出，offline replay error 在 chaotic 或接触丰富的系统里即使参数完全正确也会发散。什么样的 gap 指标既便宜、又能预测闭环的 $G_{\text{perf}}$，目前没有答案。综述也没有给出从 $G_{\text{dyn}}$、$G_{\text{perc}}$ 到 $G_{\text{perf}}$ 的界，所以没法判断先缩小哪一类 gap 最划算。
4. **视觉 gap 应该写实化、随机化，还是换模态？** [DR](./domain-randomization/notes.md) 从没和写实渲染做过对照。[SplatSim](./splatsim/notes.md) 缺一个 mesh 渲染加 DR 的 Sim2Real 对照，而不加增强时成功率只有 21%，可见光靠 3DGS 不够。RialTo 和 ACDC 用点云绕开了纹理问题。[X-Sim](./x-sim/notes.md) 的 InfoNCE 校准平均带来 7.4 个百分点，但每任务只有 10 次试验，分数粒度是 3.3%，这个提升是否显著存疑。
5. **real-to-sim 评测环境能不能当训练环境用？** SIMPLER 只验证了"被测策略从没见过这个仿真器"的情形。[π_RL](../04_rl/pi-rl/notes.md) 直接在 SIMPLER 上做 RL，同时承认明显的视觉 domain shift 仍没解决。[World-Gymnast](../05_intersections/world-gymnast/notes.md) 报告，在 world model 里做 RL，4 个任务中有 3 个的真实成功率高于在 SIMPLER 里做 RL。RialTo 里从零开始的 RL 会钻关节误差的漏洞，toaster 任务上的 62% 迁移不到真机，说明 policy 一旦对仿真做优化就会去 exploit 它的误差。这就是 Goodhart 问题。learned world model 评测器（[Ctrl-World](../05_intersections/ctrl-world/notes.md)）能否在 MMRV 这个口径下达到或超过 SIMPLER，也还没有人比过。
6. **适应应该发生在何处、何时？** 离线构建 twin 的有 RialTo（约 2 天）和 PhysTwin；在两次部署之间修正的有 SimOpt 和 ASAP；部署时做前向推断的是 RMA；部署后离线再训练的是 X-Sim。没有一篇在部署时对仿真参数、$\pi^\Delta$ 或 policy 做梯度更新。RMA 一旦超出训练支撑集就无法继续学习。ASAP 的作者把 delta action model 的 few-shot 适应列为未来工作，但没说在部署时做；PhysTwin 的笔记指出，部署时用新观测持续更新物理参数 $\alpha$（online system identification）是自然的延伸，论文没有做。TTT 式的在线更新能否扩大适应范围，是本项目最直接的交叉问题，可参照 [PAD](../05_intersections/pad/notes.md) 和 [SimDist](../05_intersections/simdist/notes.md)。
7. **真实数据的获取成本与安全**。ASAP 需要 MoCap，采数据时损坏了两台 G1；SimOpt 中间轮次的 policy 直接上真机；X-Sim 需要物体 mesh 和场景扫描，作者也承认离 in-the-wild 的互联网视频还很远；PhysTwin 需要 3 台标定过的 RGB-D 相机；RialTo 的用户研究没有把手写成功判据、设定随机化范围这些人工成本算进去。2026 年的 SimFoundry、Agentic Real2Sim 在推进全自动化，效果还需要跟进。
8. **证据规模与论文内部的一致性**。这条线的真实世界实验普遍很小，每格多在 5–40 次，基本没有置信区间，而且好几篇的正文和图表数字对不上：RMA Fig.3 的 caption 与图内表格，像是把两个任务对调了；SIMPLER Table I 的 Drawer 列与 Table IV 不一致；SplatSim 的 Assembly 一处写 85%、一处写 95%，Real2Real 按表平均是 96.25% 而不是 97.5%；ASAP 的噪声基线正文写 150、图里是 173；RialTo 文中的 3.5× 从 Fig.6 算不出来。读数字时以表格和源码为准，详见各篇笔记的"局限"一节。

## 8. 扩展阅读

以下都是调研时列入 `dropped_notable` 的论文。本项目没有为它们写笔记，arXiv ID 在 2026-09-28 核对过标题。

| 论文 | arXiv | 为什么值得读 |
|---|---|---|
| Sim-to-Real Transfer of Robotic Control with Dynamics Randomization | 1710.06537 | 把 DR 从视觉扩展到动力学参数。RMA、legged-gym 里的动力学随机化都属于这一脉，读完 DR 论文后接着读 |
| Solving Rubik's Cube with a Robot Hand | 1910.07113 | Automatic Domain Randomization 按 policy 的表现自动扩大随机化范围，方向正好和 SimOpt 用真实数据收窄分布相反；是综述反复引用的里程碑（同线还有 1808.00177） |
| BayesSim: adaptive domain randomization via probabilistic inference for robotics simulators | 1906.01728 | 用 likelihood-free inference 求仿真参数的后验，并直接拿后验当 DR 分布，对应综述 Sec.6.4 的 simulation-based inference；适合和 SimOpt 的迭代式 REPS 对照 |
| Learning agile and dynamic motor skills for legged robots | 1901.08652 | 用真实电机数据学 actuator network 并接进仿真。legged-gym 的 LSTM actuator net 源于此，它也是 ASAP delta action 之前"学习式 real2sim 组件"的代表 |
| Planar Robot Casting with Real2Sim2Real Self-Supervised Learning | 2111.04814 | 据调研记录，"Real2Sim2Real"这一名称出自这里，可以用来了解这个词最初指什么 |
| ASID: Active Exploration for System Identification in Robotic Manipulation | 2404.12308 | 在仿真里学探索策略，到真机上采集信息量大的数据再做辨识，对应综述里的 sim-to-real-driven exploration；可以缓解 SimOpt 那种"只在 $\pi_i$ 激发出的行为下做匹配"的可辨识性问题 |
| NeRF2Real: Sim2real Transfer of Vision-guided Bipedal Motion Skills using Neural Radiance Fields | 2210.04932 | 把神经渲染当作视觉仿真器的早期工作，SplatSim 的图像增强方案沿用了它 |
| URDFormer: A Pipeline for Constructing Articulated Simulation Environments from Real-World Images | 2405.11656 | 从真实图像生成铰接仿真场景，是资产重建子方向的代表；ACDC 附录和它做了定性对比 |
| GaussGym: An open-source real-to-sim framework for learning locomotion from pixels | 2510.15352 | 把 3DGS 渲染和大规模并行 RL 结合，从像素学 locomotion，正好回应了 SplatSim"渲染吞吐能否支撑 RL"的问题；调研记录把它列为本线补充阅读的首选 |
| Video2Policy: Scaling up Manipulation Tasks in Simulation through Internet Videos | 2502.09886 | video-to-sim 的另一种做法：从互联网视频批量构建仿真任务。X-Sim 指出它只用了起止两帧的物体状态，两篇适合对照 |
| Sim-and-Real Co-Training: A Simple Recipe for Vision-Based Robotic Manipulation | 2503.24361 | 系统研究视觉操作中的 sim-real co-training，综述引用了它。可以用来理解 RialTo 为什么要在真实示范上做 co-training |
| PolaRiS: Scalable Real-to-Sim Evaluations for Generalist Robot Policies | 2512.16881 | 调研记录认为它是 real-to-sim 评测目前的标杆，强烈建议作为 SIMPLER 的续读（同子方向还有 RobotArena∞，2510.23571） |
| TwinRL: Digital Twin-Driven Reinforcement Learning for Real-World Robotic Manipulation | 2602.09023 | 用数字孪生引导真实世界 RL，是 2026 年"闭环"趋势的代表，适合和 [HIL-SERL](../04_rl/hil-serl/notes.md) 这种直接在真机上做 RL 的路线对照。无代码 |
| D-REX: Differentiable Real-to-Sim-to-Real Engine for Learning Dexterous Grasping | 2603.01151 | 用可微仿真做 real2sim 的物理辨识（调研记录称是质量辨识），和 PhysTwin 的 first-order 阶段同一思路，但对象是灵巧抓取。无代码 |
| SimFoundry: Modular and Automated Scene Generation for Policy Learning and Evaluation | 2606.28276 | 据调研记录，它从单段视频自动生成数字孪生和 digital cousins，同时服务训练和评测，可以看成 ACDC 的后继。很新，代码未确认 |

综述备选：A Survey of Sim-to-Real Methods in RL: Progress, Prospects and Challenges with Foundation Models（2502.13187）。它从 foundation model 的角度整理 sim-to-real RL，可以补足本线综述对 world model 覆盖薄的问题。

## 9. 复现推荐

按 research-workflow"先复现再提假设"的原则，挑选标准有四条：有官方代码、不需要真机也能复现核心结论、单卡可行、能直接成为本项目交叉假设（world model、TTT）的对照基线。下面三个按优先级排列。三个都需要 **Linux 加 NVIDIA GPU**，本机 macOS 跑不了。

### ① legged-gym，并在其上做 sim-to-sim gap 审计

- **为什么先复现它**：官方代码完整，单卡 20 分钟出结果，成本最低。它的配方（4096 个环境、每次迭代 24 步、地形课程）被 [ASAP](./asap/notes.md)、[RWM](../05_intersections/robotic-world-model/notes.md)、[SimDist](../05_intersections/simdist/notes.md) 沿用或对应。它部署时完全不做适应，所以是 RMA、PAD/TTT、SimDist 这类"适应"假设最自然的零适应基线。综述笔记里设计的 gap 审计实验（$G_{\text{perf}}$、offline replay error、SRCC）也可以直接搭在它上面。
- **算力**：论文用 1 块 RTX A6000 加 i9-11900k；rough 地形 4096 个机器人 headless 时占 6 GB 显存，不到 20 min 训完，平地不到 4 min。依赖 Isaac Gym Preview 3、rsl_rl v1.0.2、Python 3.6–3.8。README 说环境已迁移到 Isaac Lab，换过去的话数字不一定对得上。
- **最小实验**：
  1. `anymal_c_flat` headless 训练，核对"平地不到 4 min"和 reward 曲线的形状。
  2. 复现 Fig.10：time-out bootstrap 开和关各跑一组，预期 total reward 差 10%–20%，开的那组 critic loss 更低。
  3. gap 审计：把标称环境当作 $\mathcal M_s$；再做一个伪真实环境 $\mathcal M_r$，加执行器一阶低通和延迟、质量与摩擦偏置、有时间相关性的观测噪声。分别用标称参数、DR、先 sysID 再训练三种方式训练 PPO，计算各组的 $G_{\text{perf}}$、$J_{\mathcal M_r}$、offline replay error，以及多个 checkpoint 上的 SRCC。
  4. 可选：在同一底座上实现 RMA 的 Phase 1 和 Phase 2（原论文没有官方代码），比较 Expert、RMA、Robust 的相对排序。

### ② ASAP sim-to-sim（IsaacGym → Genesis）

- **为什么**：在本线里，针对机器人自身动力学、用真机轨迹做对齐、而且三个阶段（motion tracking、delta action 训练、微调）都开源的方法只有它；SimOpt 没有代码，PhysTwin 针对的是被操作的可形变物体。sim-to-sim 设定不需要真机。论文自带 SysID（调参数）和 DeltaDynamics（学 residual state，相当于一个嵌进仿真器的 world model）两个对照，正好能让"调参数、学残差动作、学残差状态"三条路线同台比较，第 7 节第 2 条的争议可以在这里做实验。论文正文和图表有多处数字不一致，基线数字本身也需要复现来确认。
- **算力**：论文没有报告。README 示例中，motion tracking 和微调用 `num_envs=4096`，delta action 训练用 `num_envs=5000`。需要 IsaacGym Preview 4（Python 3.8）和 Genesis 0.2.1。
- **最小实验**：选 1–3 个 medium 或 hard 动作，在 IsaacGym 里预训练 tracking policy；把 Genesis 当作"真实"环境 rollout 约 430 条轨迹（Fig.10a 显示 430 条时 closed-loop MPJPE 已经到 104.95，不过那张图用的是 IsaacSim 数据）；用 horizon 1.0 s、action norm 权重 0.1 训练 $\pi^\Delta$；冻结后微调 policy，回到 Genesis 评测。对比 Vanilla、随机动作噪声微调（$\beta=0.1$）和 ASAP，并复现 Table III 在 0.25、0.5、1.0 s 的 open-loop 误差排序。验收标准：closed-loop MPJPE 明显低于 Vanilla 和噪声微调。

### ③ SIMPLER（real-to-sim 评测）

- **为什么**：难度 easy，是本线唯一的评测基线。MMRV 和 Pearson 可以原样用来给 learned world model 评测器（[Ctrl-World](../05_intersections/ctrl-world/notes.md)、[World-Gymnast](../05_intersections/world-gymnast/notes.md)）打分，是比较"物理 real2sim"和"learned simulator"两条路线的现成口径。SimplerEnv 也已经是 VLA-RL 论文的常用 benchmark，例如 [π_RL](../04_rl/pi-rl/notes.md)。
- **算力**：论文报告单张 RTX 4090 每秒 3.5k 仿真步。README 建议用 RTX 卡，因为部分环境依赖 ray tracing。需要 Python 3.10/3.11 和 ManiSkill2_real2sim；RT-1 要 TF 2.15，Octo 要 JAX，注意两者的 CUDA 版本冲突。RT-2-X 没有公开。真实侧没有机器人，直接用 Table IV/V 的真实成功率。
- **最小实验**：在 Bridge VisMatch 的 4 个任务上跑 RT-1-X、Octo-Base、Octo-Small，Octo 取 3 个 seed，每个任务 24 个初始配置；按 Eq.1–2 计算 MMRV 和 Pearson，目标是复现 MMRV≈0.014、$r\approx0.890$ 的量级。再做两个消融：去掉 green screen（背景换成 ReplicaCAD），或者把 PD 参数换回非 SysID 的值，看 MMRV 是否像 Table II/III 那样变差。注意 Table I 和 Table IV 的 Drawer 列不一致，以 Table IV 为准。

**备选**：[PhysTwin](./phystwin/notes.md) 用官方数据只跑 inference 加 `evaluate.sh`，一张消费级 GPU 就能对上 Table 1，适合往可形变方向走的人。[X-Sim](./x-sim/notes.md) 可以纯仿真复现 PPO、Diffusion Policy，再在伪真实域上做 InfoNCE 校准，适合操作方向。[RialTo](./rialto/notes.md) 依赖旧版 Isaac Sim 2022.2.1，每个任务约 2 天，难度 hard，不建议作为第一个复现。
