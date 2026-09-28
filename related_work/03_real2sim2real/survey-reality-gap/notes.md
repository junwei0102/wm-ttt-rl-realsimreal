# The Reality Gap in Robotics: Challenges, Solutions, and Best Practices

> arXiv [2510.20808](https://arxiv.org/abs/2510.20808) · Annual Review of Control, Robotics, and Autonomous Systems Vol. 9, pp. 403–432, 2026（原文标注 "Preprint to appear"，现已出版，Crossref 核实；arXiv 预印本 2025） · 主题：03_real2sim2real · 层级：survey
> 代码：none（综述，无代码）。项目页 https://robotics-reality-gap.github.io/ 已确认存在，除论文 PDF 和 arXiv 链接外只有按章节整理的介绍子页面（Sources of Gaps、Evaluation Metrics、Best Practices、Open Problems），没有代码、benchmark 或工具。· 笔记依据：`source/chapters/*.tex`（公式/分类）+ `paper.pdf`（图）

覆盖范围：PDF 共 31 页，正文在 p.1–23（Sec.1–7），后面是参考文献（编译后的 `annualreviews.bbl` 有 178 条）。正文全部读完：`introduction / preliminaries / gaps / solutions / metrics / openproblems.tex` 和 `figures/gap_list.tex`。Fig.1–3 在 PDF 上看，Fig.3 的色块放大到 400 dpi 逐个核对过。作者来自 UZH RPG（Scaramuzza 组）、NVIDIA（Ramos、Narang、Hermans、Akinola、Garrett、Heiden；其中 Ramos 兼 Sydney，Hermans 兼 Utah）和 UW（Gupta、Fox），所以举例偏向 drone racing 和 NVIDIA 的仿真栈。公式编号按 PDF。

## 一句话

综述的做法是先拆后分。先把 reality gap 从"一个笼统的 gap"拆成四类可以逐项诊断的子来源：dynamics、perception & sensing、actuation & control、system design，大多数子项都附一个 margin 里的 *Symptom*，说明它在真机上会表现成什么现象。再把 sim-to-real 方法分成两大类：**reducing the gap**（改进仿真、换接口、改系统设计）和 **overcoming the gap**（让 policy 对剩下的 gap 保持鲁棒或能自适应）。核心主张是：sim-to-real 要缩小的是 policy 的 **performance gap**，不需要消灭 reality gap。

## 问题设定

框架来自 Sec.2，文件 `preliminaries.tex`。

- **POMDP**：$\mathcal M=(\mathcal S,\mathcal A,\mathcal T,\mathcal R,\mathcal Z,\mathcal O,\gamma)$，agent 维护一个 belief $\mathbf b_t\in\mathcal B(\mathcal S)$。目标是 Eq.1：
  $$\pi^*=\arg\max_\pi\ \mathbb E\Big[\sum_{t=0}^{\infty}\gamma^t\mathcal R(\mathbf s_t,\pi(\mathbf b_t))\Big]$$
  实际做 sim-to-real 时，$\mathbf b_t$ 一般就是从图像和本体感知里提出来的特征向量。仿真的定义是 $\mathcal T_s\approx\mathcal T_r$、$\mathcal O_s\approx\mathcal O_r$。
- **Reality gap**（Eq.2）描述的是环境本身，和 policy 无关：
  $$G_{\text{dyn}}=\mathbb E_{(\mathbf s,\mathbf a)\sim\mathcal M_r}\big[D(T_{\text{sim}}(\cdot|\mathbf s,\mathbf a)\,\|\,T_{\text{real}}(\cdot|\mathbf s,\mathbf a))\big],\quad G_{\text{perc}}=\mathbb E_{\mathbf s\sim\mathcal S_r}\big[D(O_{\text{sim}}(\cdot|\mathbf s)\,\|\,O_{\text{real}}(\cdot|\mathbf s))\big]$$
- **Performance gap**（Eq.3–4）依赖具体的 policy：
  $$J_{\mathcal M}(\pi)=\mathbb E_{\tau\sim p(\tau|\pi,\mathcal M)}\Big[\sum_{t=0}^{T}\gamma^tR(\mathbf s_t,\pi(\mathbf b_t))\Big],\qquad G_{\text{perf}}(\mathcal M_s,\mathcal M_r,\pi)=|J_{\mathcal M_s}(\pi)-J_{\mathcal M_r}(\pi)|$$
- **Sim-to-real 的目标**（Eq.5）：$\pi^*=\min_\pi G_{\text{perf}}$。论点是 $G_{\text{dyn}}$ 和 $G_{\text{obs}}$ 既不可能清零，也没必要清零，只要 policy 对两个环境之间的差异足够鲁棒，就能迁移成功。
- **之前卡在哪**（Sec.1）：大家习惯把 gap 当成"一个东西"来谈，其实它由大量子 gap 组成。policy 还会**利用**仿真里的建模误差和 simulator 特有的 corner case，在仿真里拿高分，到了真机上失效，甚至有危险。

## 核心方法

这篇综述没有提出方法，这一节整理它的分类框架。

### A. Reality gap 的四类来源（Sec.3，Fig.2，`gaps.tex`）

Fig.2 用同一个机械臂小车画了左右对照：左半是真实世界，右半是网格化的仿真。每类来源下面的子项如下，括号里是对应的 margin Symptom：

| 大类 | 子项（小节号） | 典型 Symptom |
|---|---|---|
| **Dynamics** $G_{\text{dyn}}$（3.1） | 3.1.1 Modeling：rigid body（真实的连杆和机架受载会弯曲、振动，关节有阻尼、内部弹簧、backlash）、chaotic nature、stochasticity（地面摩擦变化和碎屑，仿真里要么忽略、要么简化成高斯噪声）、battery（电压随负载、温度、老化下降，最大力矩跟着变小）、contact dynamics（point contact、线性化摩擦锥、spring-damper）；3.1.2 Parameterization；3.1.3 Numerical integrators；3.1.4 Human-robot interaction；3.1.5 Unmodeled effects（磨损、热效应）；3.1.6 Asset fidelity | 抓取打滑；overshoot 或 undershoot；长任务里积分误差累积；随时间漂移 |
| **Perception & sensing** $G_{\text{perc}}$（3.2） | 3.2.1 Sensor model：RGB 相机（指 Gazebo Classic、早期 MuJoCo 这类简化 OpenGL 渲染）缺镜头光晕、色差、flying pixels、rolling shutter；depth 缺量化噪声、depth shadow；LiDAR 缺 beam divergence、材质反射率；IMU 漂移、GPS multipath、传感器延迟。3.2.2 Sensor noise：真实噪声非高斯、依赖状态、有时间相关性，仿真里多用固定方差高斯。3.2.3 Environment representation（HDRI、纹理）；3.2.4 Robot model（URDF/USD 省掉了制造公差）；3.2.5 Collision sensing（凸分解、球分解，只在离散时刻检测） | 观测分布大幅偏移；换一个真实房间就认不出物体 |
| **Actuation & control**，文中叫 actuation gap（3.3） | 3.3.1 Actuator models：仿真把电机当成理想的一阶系统，真实电机是高阶系统，还有 dead-zone、backlash、hysteresis、slew rate；3.3.2 Low-level control：厂商固件不开放，里面藏着滤波、饱和、anti-windup、rate limit；3.3.3 Power electronics：ESC 和驱动器的内环会带来"hundreds of microseconds"的延迟，PWM 量化在零点附近形成死区，另有限流、限压 | 高速动作时抖动或延迟；零速附近 jitter |
| **System design**（3.4） | 3.4.1 Communication（延迟、丢包、安全停车逻辑）；3.4.2 Safety mechanisms（virtual walls）；3.4.3 POMDP formulation：reward 和终止条件用到了真机拿不到的特权信息（例如精确的碰撞），reset 把物体直接放到精确位置，由此导致 state-action 分布不一致；3.4.4 Implementation details（离散化粒度、控制频率不一致） | 真机评测和仿真评测差距很大；控制频率不匹配导致失稳 |

### B. 解决方案分类（Sec.4，Fig.3，`solutions.tex`）

**Sim-to-Real Recipe**（Sec.4 开头的方框）共六步：(1) 仿真要覆盖与任务相关的所有变量；(2) 尽量缩小各项 gap；(3) 设计训练方法去克服剩下的 gap；(4) 在仿真里训练，最好大规模并行；(5) 在真实环境评测；(6) 根据真机表现调仿真参数，然后重训。第 6 步可以看成 SimOpt 式的闭环（这是笔记的类比，原文在这里没有点名 SimOpt）。

Fig.3 用色块标出每种方法针对哪类 gap。图例：粉 = Perception，浅蓝 = Actuation，灰 = Dynamics，绿 = System。下表的 P/A/D/S 按这个对应。

| 大类 | 子类 | 方法（Fig.3 标注） | 机制要点 |
|---|---|---|---|
| **Reducing** | Improving simulation fidelity（4.1.1） | System identification（D） | 物理参数之外还要辨识系统级参数：latency、控制频率、actuation delay。近来的趋势是拿每轮真机评测的数据做在线、迭代的 sys-ID，例如 [SimOpt](../simopt/notes.md)。 |
| | | Residual learning（D P A S） | **改输出**：学一个网络修正 simulator 预测的状态。Golemo 用 LSTM 拟合 sim 与 real 轨迹之差；Ajay 用随机 RNN 补接触的不确定性；NeuroBEM 补四旋翼的气动残差；还有 FEM 软体的修正力。**改输入**：Christiano 学一个 inverse dynamics，把仿真动作映射成真实动作；GAT 更进一步，在训练 policy 之前就把 action transformation 接进 simulator。适用于 sys-ID 失效的情况，比如用刚体去近似柔体。 |
| | | Real-to-sim（D P A S） | 从 $\mathcal D_{\text{real}}=\{(z,a,z')_i\}_{i=1}^N$ 反推出仿真环境，分三部分：几何（SfM/MVS，或单图生成 mesh）、运动学与动力学（articulation 推断，如 Ditto、URDFormer、Real2Code；作者评价是离真实"仍有相当差距"）、渲染（NeRF、3DGS）。引到的例子有 [RialTo](../rialto/notes.md)、[X-Sim](../x-sim/notes.md)、[PhysTwin](../phystwin/notes.md)、ASID。 |
| | Modalities & representations（4.1.2） | Sensor modality（P）；State repr.（P）；Action repr.（D A） | 用 depth 或点云时的 gap 比 RGB 小。keypoint、visual feature tracking、foundation-model embedding 都可以作为抽象层。操作任务里用 joint velocity 作为动作空间更有利于迁移（Aljalbout 2024）。 |
| | Design choices（4.1.3） | Low-level controller（A）；Software alignment（D P A S）；Hardware design（D A）；System constraints（D P） | 高频 impedance 控制器；仿真和真机的软件栈保持一致（频率、滤波、安全检查）；选低延迟的执行器和噪声特性明确的传感器；先把动作限制在 quasi-static 区间。 |
| **Overcoming** | Domain generalization & adaptation（4.2.1） | Domain randomization（D P A S） | 训练时随机化纹理、光照、观测与动作噪声、质量、惯量、延迟等参数，减少了对费力的 sys-ID 和 simulator 标定的需求（Peng 2018）。自动选分布的方法有：ADR（policy 成功后逐步扩大随机化范围）、DROPO（用离线真实数据估计 DR 范围）、SimOpt。 |
| | | Meta learning（D P A S） | meta-RL 让 policy 在内部识别当前的参数（Arndt 2020、AdaptSim）。思路来自 adaptive control；文中把 [RMA](../rma/notes.md) 作为 explicit meta learning 之外的一个例子：encoder 利用仿真里的特权信息推断出环境 dynamics 的 latent，policy 再以这个 latent 为条件。 |
| | | Domain adaptation（P） | 重点是让 policy 适应**观测**的分布偏移，不一定处理 dynamics 的变化。 |
| | | Adversarial methods（Fig.3 只标 P） | RARL：额外训练一个对手 agent 施加扰动。 |
| | Data & exploration（4.2.2） | Incorporating real-world data（D P A S） | H2O：对和真实 dynamics 差距大的仿真 (s,a) 加惩罚。RialTo：先用真实数据训一个 vision-based policy，再用它帮助在仿真里训 teacher；teacher 再用来训 vision student，student 用"仿真 rollout 上的 RL + 真实数据上的 BC"混合训练。也可以用 residual RL（Ankile 2024），或者做 sim-real co-training（Maddukuri 2025、RoboCasa、GR00T）。 |
| | | Sim-to-real-driven exploration（D P A） | 在真机上执行探索 policy 去辨识关键参数，再做轨迹优化（Liang 2020）；或者在仿真里学探索策略，拿到真机上收集数据来微调（ASID、Wagenmaker 2024）。 |
| | Policy architecture & regularization（4.2.3） | Modularity（D P A）；Privileged information（D P A）；Representation learning（P）；Policy regularization（D） | 感知和控制解耦、分层技能；asymmetric actor-critic（critic 用特权状态）；teacher-student 蒸馏，RL loss 和蒸馏 loss 的权重可以固定（Nguyen）或按 schedule 调（Radosavovic）；Krinner 2025 把特权状态用在 state-space world model 里；DIRL 做特征分布对齐；Yoneda 在遇到真实样本时用对抗方式调整 encoder，同时加 dynamics consistency loss；用对比学习去掉背景；在 reward 里惩罚动作幅值和相邻动作差（action rate），或者用 Lipschitz 平滑 loss。 |

**白板推导（补充）**：DR 和 residual 在综述里只有文字描述，下面两式是我按 Eq.3 的记号写出来的标准形式，不是论文原文：
$$\text{DR：}\ \max_\pi\ \mathbb E_{\xi\sim p(\xi)}\big[J_{\mathcal M_s(\xi)}(\pi)\big];\qquad \text{output residual：}\ \hat{\mathbf s}_{t+1}=f_{\text{sim}}(\mathbf s_t,\mathbf a_t)+g_\phi(\mathbf s_t,\mathbf a_t)$$
两者分别落在两大类里。DR 属于 "overcoming"，靠加宽训练分布，让 $\mathcal M_r$ 落进 $\{\mathcal M_s(\xi)\}$ 的支撑集。Residual 属于 "reducing"，直接缩小 $G_{\text{dyn}}$。

### C. 评测指标（Sec.5，`metrics.tex`）

指标分两组：一组量 **gap 本身**，一组量**迁移效果**。
- **SRCC**（Eq.6，Kadian 2020）：$N$ 个 policy 在仿真和真机上的任务表现（例如成功率）$(x_i,y_i)$ 的 Pearson 相关系数。作者强调，一个 simulator 就算真机平均表现高，只要 SRCC 低就仍然有问题，因为在仿真里做的改进无法预测真机上的效果。
- **Offline replay error**（Eq.7）：把真实轨迹的动作 open-loop 地在仿真里回放一遍，
  $$\mathcal E_{\text{replay}}=\frac1T\sum_{t=1}^T\|\mathbf s^{\text{sim}}_t-\mathbf s^{\text{real}}_t\|^2$$
  好处是便宜，只要有日志就能离线算。
- **Visual fidelity**：分布级指标有 IS、FID、KID、t-SNE，单图指标有 SSIM、PSNR、IPD。
- **迁移效果**：success rate $N_{\text{success}}/N_{\text{total}}$、cumulative reward $\sum_t r_t$、任务特定指标（路径效率、物体到目标的距离）。作者指出成功率是二值或聚合量，看不出**在哪里、为什么**失败；cumulative reward 要求仿真和真机的 reward 定义一致。

### D. 开放问题（Sec.6，`openproblems.tex`）

6.1 **Wrong models, better controllers**：Lambert 的 objective mismatch 表明，模型只要在高回报区域准确，效果就比在整个空间都均匀准确更好；Guzman 用 Bayesian optimization 调 MPC，结果"按控制性能优化"比"用最可能的模型做规划"更好。
6.2 **Differentiable simulators**：Warp、Taichi、JAX、PyTorch compile。
6.3 **Video & world models**：world model 可以用真实数据训练，"potentially creating a smaller reality gap"；但存在 compounding error、泛化差、抽象程度与真实感难以平衡、数据需求大等问题。作者预期 simulator 以后会给 world model 提供 bootstrap 数据。
6.4 **Simulation-based inference**：
$$p(\theta\mid\{\mathbf s^r_i\},\{\mathbf s^s_j\})\propto p(\{\mathbf s^r_i\}\mid\theta)\,p(\theta),\qquad p(\{\mathbf s^r_i\}\mid\theta)=\int p(\{\mathbf s^r_i\},\mathbf s^s\mid\theta)\,d\mathbf s^s$$
这个似然算不出来，所以用 ABC、变分推断或 neural posterior estimation（如 BayesSim）去近似。后验通常是多峰的：机械臂推方块时遇到的阻力，既可以归因于质量大，也可以归因于摩擦大，或两者都有。得到的后验可以直接作为 DR 的分布。
6.5 **Simulation for large robotics models**：用仿真大规模生成数据（MimicGen、DexScale），以及用仿真评测真实世界的 policy（[SIMPLER](../simpler/notes.md)）。

## 关键实验与证据

综述**没有自己的实验，也没有定量对比表**，全文不报任何成功率或误差数字。出现的数量级描述只有两处：Sec.1 说 GPU 并行的 simulator 能同时仿真"thousands of robots"；Sec.3.3.3 说功率电子会带来"hundreds of microseconds"的延迟。它引用的定性结论如下：
1. Sec.4.2.1 把几个里程碑归功于 DR：魔方灵巧手（Akkaya 2019）、四足 locomotion（Tan 2018、Hwangbo 2019）、冠军级无人机竞速（Kaufmann 2023）；并说 DR 最早的应用之一是无人机导航（CAD2RL）。Sec.1 另外把四足越野（Lee 2020）和双足控制（Grandia 2024、Radosavovic 2024）归功于 GPU 大规模并行仿真。Fig.1 的配图取自 Akkaya 2019、Tan 2018、Tang 2023（IndustReal）、Lee 2020、Radosavovic 2024、Kaufmann 2023。
2. Sys-ID 在导航、locomotion、操作等多个领域被"repeatedly shown"是迁移成功的关键；除物理参数外，辨识 latency、控制频率、actuation delay 这类系统级参数也是常见做法（Sec.4.1.1）。
3. 用 depth 或点云时的 gap 小于 RGB，这是领域惯例（Sec.4.1.2）。
4. 用 loss 做正则（Lipschitz 平滑 loss）"clear advantage"，优于调 reward（Mysore 2021，Sec.4.2.3）；Chen 2024 的 LCP 在人形机器人上得到类似结论。
5. RialTo 表明，混入真实数据做 co-training 比只用仿真数据好（Sec.4.2.2）。
6. Lambert：模型在高回报区域准确，比均匀准确更好。Guzman：以控制性能为目标调参，比用最大似然模型做 MPC 更好（Sec.6.1）。
7. 限制在 quasi-static 区间能让迁移更可靠（Chen 2021）；高频 impedance 控制器能明显改善迁移（Zhang 2024）。见 Sec.4.1.3。

**综述梳理的工具**：GPU 并行 simulator（Sec.1：Orbit、Isaac Gym、MuJoCo，以及 Liang 2018 的 GPU 加速方案）；渲染和传感器仿真（Sec.3.2：Isaac Sim 用 ray tracing 和 PBR，CARLA、AirSim 都基于 Unreal，Habitat-Sim 提供合成深度；Gazebo Classic 和早期 MuJoCo 被当作简化 OpenGL 渲染的反例）；Bullet 出现在碰撞检测（Sec.3.2.5）和可微仿真（Sec.6.2）部分；真实数据集（Open X-Embodiment、DROID）；评测基准（SIMPLER）。综述没有把数据集和 benchmark 整理成表。

## 局限与论文没说的

**作者明说的**（都在 Sec.6）：world model 和 video model 有 compounding error、物理不可信、可控性差、数据需求大等问题；SBI 是 ill-posed 的反问题，后验多峰；为了缩小 gap 所做的方法和假设，可能反过来限制仿真数据能扩展到的规模；把仿真当 policy 评测器用时，怎样校准和克服 gap、使仿真表现与真机表现一致，仍是开放问题。

**我读出来的**：
1. **只有定性分类，没有量化**。给定一个平台，读者没法判断哪类 gap 占主导，综述也没有给出任何类似"gap 预算"的分解或 meta-analysis。
2. **形式化有漏洞**：
   - Eq.5 $\min_\pi G_{\text{perf}}$ 单独拿来当目标会退化：一个在仿真和真机上都失败的 policy，$G_{\text{perf}}=0$。真正想要的是 $J_{\mathcal M_r}$ 高。
   - Eq.2 里的 $(\mathbf s,\mathbf a)\sim\mathcal M_r$ 没说明由哪个 policy 诱导出来。
   - 综述没有给出任何从 $G_{\text{dyn}}$、$G_{\text{perc}}$ 到 $G_{\text{perf}}$ 的上界（例如 model-based RL 里 simulation lemma 那类结果）。
   - Eq.2 写的是 $G_{\text{perc}}$，正文后面却变成 $\mathcal G_{\text{obs}}$，记号前后不一致。
   - Actuation gap 和 system gap 都没有形式化；其中 reward 不一致、reset 分布不一致这两项根本不在 Eq.2 里。
3. **Fig.3 与正文不一致**：adversarial methods 在 Fig.3 里只标了 P，但 margin note 写的是 A、P、D。Meta learning 的 margin note 写的是"adapts the simulation parameters"，和正文把它描述成 policy 自适应相矛盾。DR 段落里还有一句孤立的 "lowest-return dynamics from an ensemble"，指的应该是 EPOpt 一类方法，但没有给引用。引用也有错配，例如 SSIM/PSNR 被引到了 IS 的论文 `salimans2016improved`。所以色块只能当粗略参考。
4. **对 world model 和 TTT 覆盖很薄**：
   - world model 只有 Sec.6.3 一小节，具体模型只引了 Genie、GEM、Cosmos 三篇（另有 future frame synthesis 和 world model 各一篇综述）；方法部分只有 Sec.4.2.3 的 Krinner 2025 用到 world model。
   - 源码里有一段更长的 world model / video model / learnable simulator 开放问题清单，点名了 Dreamer 和 JEPA 类模型，引用了 RWM，但被 `\begin{comment}` 注释掉了，编译出的 PDF 里没有。
   - ASAP 在 `references.bib` 里有条目，正文却没有引用。
   - SplatSim、ACDC、Dreamer 系列、PAD 在编译出的 PDF 里都没有出现（Dreamer 只在上面那段被注释掉的文字里出现过）。
   - "test-time adaptation" 这个词全文没有出现，自适应只落在 RMA、meta-RL、domain adaptation，以及 representation learning 里 Yoneda 2022 的 encoder 适应这几种方法上。
   - RL 算法怎么选（PPO 还是 SAC）全文没有讨论。
5. **指标本身的局限没展开**：
   - SRCC 要求很多个 policy 都在真机上跑评测，成本高。
   - Offline replay error 是 open-loop 回放：在 chaotic 或接触丰富的系统里，就算参数完全正确，误差也会发散；它还需要真实状态（mocap），也量不到 perception gap。
   - FID 这类视觉指标和 policy 的实际表现是否相关，综述没有讨论。
6. **领域偏向**：举例集中在 drone racing、四足、NVIDIA 仿真栈；可变形物体、流体、HRI 都只有一两句。

## 在脉络中的位置

- **前驱**：[Domain Randomization](../domain-randomization/notes.md)（Tobin 2017）、[SimOpt](../simopt/notes.md)（真实数据闭环调 DR 分布）、[RMA](../rma/notes.md)（特权 latent + 在线推断）、[legged-gym](../legged-gym/notes.md)（大规模并行仿真）；Muratore 2022 的 DR 综述；Kadian 2020 的 SRCC。
- **后继 / 影响**：本项目里有几篇比它晚、正好对上它 Sec.6.3/6.5 开放问题的工作。它们是否引用了这篇综述，我没有核实。
  - [SimDist](../../05_intersections/simdist/notes.md)：在仿真里预训练 world model，再到真实世界快速适应，对应"simulators providing data to bootstrap world models"。
  - [World-Gymnast](../../05_intersections/world-gymnast/notes.md)、[Ctrl-World](../../05_intersections/ctrl-world/notes.md)：把 world model 当成 simulator 或评测器，对应 Sec.6.5 的"用仿真评测 policy"。
- **与本项目其他主题的联系**：
  - **World model**：Sec.6.3 把 world model 定位成"可以用真实数据学的、白盒 simulator 的替代品"；引了 [Genie](../../01_world_model/genie/notes.md) 和 [Cosmos](../../01_world_model/cosmos-wfm/notes.md)。
    - Residual learning（Golemo、Ajay）本质上是"解析 simulator + 学出来的 dynamics"的混合模型，[Robotic World Model](../../05_intersections/robotic-world-model/notes.md) 和 [UniSim](../../05_intersections/unisim/notes.md) 是往"完全学出来"那一端走。
    - Sec.6.1 的 objective mismatch 为 [TD-MPC2](../../01_world_model/tdmpc2/notes.md) 这类不重建观测、只做面向价值的 latent model 提供了动机。
    - [DayDreamer](../../05_intersections/daydreamer/notes.md) 直接在真实世界学 world model，从根本上绕开了 reality gap。
  - **TTT**：综述 "overcoming" 下那些部署时在线适应的方法（RMA、Yoneda 2022），相当于机器人领域的 test-time adaptation，只是没用这个名字；综述没有区分 adaptation 发生在训练时还是部署时，DR 本身则不做适应。
    - RMA 靠在线推断 latent 来适应，不做梯度更新。
    - Yoneda 2022 部署时用对抗对齐加 dynamics consistency 来更新 encoder，和 [PAD](../../05_intersections/pad/notes.md)、[TTT self-supervised](../../02_test_time_training/ttt-self-supervised/notes.md) 同属"部署时用自监督 loss 更新表征"这一类。
    - 只处理观测偏移的 domain adaptation 可以和 [Tent](../../02_test_time_training/tent/notes.md) 对照着看。
  - **Real2sim2real**：[RialTo](../rialto/notes.md)、[X-Sim](../x-sim/notes.md)、[PhysTwin](../phystwin/notes.md) 归在 "Real-to-sim"，[SIMPLER](../simpler/notes.md) 归在 Sec.6.5。
    - 按这套分类，[ASAP](../asap/notes.md) 学真实与仿真之间的动作差，属于"改输入"的 residual，是 GAT 这条线的延续。
    - [SplatSim](../splatsim/notes.md) 属于 real-to-sim 的渲染分支（3DGS）。
    - [ACDC](../acdc-digital-cousins/notes.md) 介于 real-to-sim 和 DR 之间。
  - **RL**：成功案例的底座基本都是大规模并行仿真里的 on-policy RL，见 [PPO](../../04_rl/ppo/notes.md)（这是笔记的判断，综述本身没有讨论具体用哪种 RL 算法）。其他相关点：asymmetric actor-critic、reward 里的 action rate 惩罚、H2O 的 offline + online 混合 RL。[HIL-SERL](../../04_rl/hil-serl/notes.md) 走的是"直接在真机上做 RL"的替代路线。[SimpleVLA-RL](../../04_rl/simplevla-rl/notes.md) 和 [π_RL](../../04_rl/pi-rl/notes.md) 在仿真里对 VLA 做 RL，同样会面临这里列出的所有 gap。

## 前置知识

1. POMDP、MDP、belief、discounted return（读懂 Eq.1–5 需要）。
2. 物理仿真基础：刚体动力学、接触模型（point contact、摩擦锥、spring-damper）、数值积分（Euler、RK）、碰撞代理几何。
3. 执行器与底层控制：PD 和 impedance 控制、电机的高阶响应、PWM、ESC。
4. System identification 与贝叶斯推断：ABC、变分推断、neural posterior estimation。
5. Domain randomization、teacher-student 与 privileged information、asymmetric actor-critic。
6. 深度 RL（PPO，大规模并行训练）；domain adaptation（对抗式特征对齐）；对比学习。
7. 3D 重建与新视角合成：SfM/MVS、NeRF、3DGS。

## 复现可行性

- **代码**：没有，综述本身不附代码，项目页也没有 repo。算力：论文没有报告，作为综述也不涉及。
- **综述点名的工具**：训练用 GPU 并行仿真（Isaac Gym、Orbit、MuJoCo）；渲染较真实的 simulator 有 Isaac Sim、CARLA、AirSim、Habitat-Sim；可微仿真用 Warp、Taichi、JAX；policy 评测用 SIMPLER。诊断指标推荐 SRCC（Eq.6）、offline replay error（Eq.7）、FID/KID/SSIM/PSNR。
- **最小可复现实验**（sim-to-sim 形式的"gap 审计"）：
  - **搭环境**：在 [legged-gym](../legged-gym/notes.md) 或 MuJoCo 里准备两份环境，一份标称环境 $\mathcal M_s$，一份作为"伪真实"的 $\mathcal M_r$。$\mathcal M_r$ 相对 $\mathcal M_s$ 做以下改动：执行器加一阶低通和延迟，质量和摩擦加偏置，观测噪声改成有时间相关性的。这些改动分别对应 Sec.3.3.1、3.1.2、3.2.2。
  - **训练**：用 PPO 训三组 policy：(a) 标称参数，(b) DR，(c) 先用 $\mathcal M_r$ 的日志做 sys-ID 再训。
  - **测量**：每组都算 $G_{\text{perf}}$（Eq.4）和 $J_{\mathcal M_r}$，用 $\mathcal M_r$ 的动作日志算 $\mathcal E_{\text{replay}}$，再在 $N$ 个 checkpoint 上算 SRCC。
  - **要检验的问题**：replay error 能不能预测 $G_{\text{perf}}$；DR 是否降低了 $G_{\text{perf}}$，代价是不是 $J_{\mathcal M_s}$ 下降。
  - **规模**：legged-gym 这类设置的开销不大。论文没有给出任何算力数字，这里也不做估计。
- **难点**：sim-to-sim 只能模拟"已知的未建模项"，真正的 unmodeled effects（磨损、厂商固件）必须上真机；SRCC 需要大量真机 rollout。

## 综述强调、但本项目未收录的重要工作

`references.bib` 里大多数条目没有 arXiv 号，下面有 ID 的就写 ID，没有的写 venue。

1. **Akkaya et al. 2019, Solving Rubik's Cube with a Robot Hand**（arXiv 1910.07113）：Automatic DR，按 policy 的表现逐步扩大随机化范围，是综述反复引用的里程碑。Peng et al. 2018 *Sim-to-real transfer of robotic control with dynamics randomization*（ICRA 2018）：综述引它说明 DR 能减少费力的 sys-ID 和 simulator 标定。
2. **Tan et al. 2018, Sim-to-Real: Learning Agile Locomotion for Quadruped Robots**（RSS 2018）和 **Hwangbo et al. 2019, Learning agile and dynamic motor skills for legged robots**（Science Robotics）：对 latency 和执行器做 sys-ID 的典范。
3. **Pinto et al. 2017, Robust Adversarial RL**（ICML 2017）和 **Pinto et al. 2018, Asymmetric Actor Critic**（RSS 2018）：对抗训练、privileged critic 的来源。
4. **Residual simulation**：Christiano et al. 2016 inverse dynamics transfer（arXiv 1610.03518）；Hanna & Stone 2017 GAT（AAAI 2017）；Golemo et al. 2018 neural-augmented simulation（CoRL 2018）。
5. **Kadian et al. 2020, Sim2Real Predictivity**（IEEE RA-L）：SRCC 指标的出处。
6. **Ramos et al. 2019, BayesSim**（RSS 2019）和 **Muratore et al. 2022, Robot Learning from Randomized Simulations: A Review**（Frontiers in Robotics and AI）：用 SBI 求后验，再拿后验当 DR 分布。
7. **Niu et al. 2022, When to Trust Your Simulator (H2O)**（NeurIPS 2022）：dynamics-aware 的 offline + online 混合 RL。
8. **Maddukuri et al. 2025, Sim-and-Real Co-Training**（arXiv 2503.24361）：视觉操作任务的 sim-real co-training 方法。
9. **Krinner et al. 2025, Accelerating MBRL with State-Space World Models**（arXiv 2502.20168）：把特权状态用在 world model 里，在基于视觉的高速无人机竞速上验证。综述正文里 world model × sim-to-real 的方法论文很少，这是其中一篇。
10. **Lambert et al., Objective Mismatch in Model-based RL**（L4DC）：dynamics 模型的精度和控制性能不一致，是 Sec.6.1 的核心依据。

## 自测问题

1. Reality gap 和 performance gap 分别依赖什么？为什么 $G_{\text{dyn}}>0$ 时 policy 仍然可以迁移成功？为什么单独拿 Eq.5 $\min_\pi G_{\text{perf}}$ 当训练目标是不够的？
2. 真机上零速附近 jitter、细微动作做不出来，应该归到哪个子来源？请分别给出一个 "reducing" 手段和一个 "overcoming" 手段，并说明各自改变的是 Eq.2/Eq.3 中的哪一项。
3. Output residual（Golemo）、input residual（Christiano、GAT）、system identification 三者分别学什么、需要什么数据？在什么情况下 sys-ID 会失效，而 residual 仍然有效？
4. 构造两个场景：一个 offline replay error 很大，但 SRCC 很高（提示：chaotic 或接触丰富的系统）；另一个 replay error 很小，但 $G_{\text{perf}}$ 很大（提示：gap 在感知一侧，或者在 reward、reset 的定义上）。
5. 在 SBI 里，推方块任务的后验为什么是多峰的？把后验直接当作 DR 分布，和 SimOpt 那种迭代调分布相比，各自的优缺点是什么？
