# World Model for Robot Learning: A Comprehensive Survey

> arXiv [2605.00080](https://arxiv.org/abs/2605.00080) · arXiv preprint, 2026 · 主题：01_world_model · 层级：survey
> 代码：none（综述，无代码）；配套论文列表 https://github.com/NTUMARS/Awesome-World-Model-for-Robotics-Policy （已确认存在，是 paper list），主页 https://ntumars.github.io/wm-robot-survey/ · 笔记依据：`source/sec/*.tex`（数字/公式）+ `paper.pdf`（图）

覆盖范围：全文 43 页，Sec.1–8 都读了。精读 Sec.3（world model for policy）、Sec.4（simulator：RL + evaluation）、Sec.5（robotic video world model）、Sec.7（benchmark / 数据集 / 结果汇总）、Sec.8（挑战）；Sec.6（导航 / 自动驾驶）只略读。公式编号按 PDF。

## 一句话

这篇综述以 VLA / 机器人 policy 为中心，把 2023–2026 年的机器人 world model 工作放进三套分类里：**和 policy 怎么耦合**（IDM → single-backbone → MoE/MoT → unified VLA → latent-space）、**当 simulator 用来做什么**（RL 训练 / 评估），以及 **robotic video world model 到了哪个能力阶段**（imagination → action-controllable → structure-aware → foundation-scale）。Sec.3 开头给了一个概率视角，用来串起第一套分类（policy 耦合方式）：policy、passive world model、controllable world model 和 IDM，可以看成对同一个联合分布 $p(o_{t+1:t+k},a_{t+1:t+k}\mid o_t,l)$ 的不同查询（边缘分布或条件分布）。这个视角只在 Sec.3 使用，没有用来统一 simulator 和 video WM 那两套分类。

## 问题设定

- **动机**（Sec.1）：纯 reactive 的 VLA（RT-2、OpenVLA、π0）在长时程推理、temporal credit assignment 和 compounding error 上都吃力。作者认为原因之一是模型里没有显式结构去预测"世界在我的动作下会怎么变"。按作者的说法，actionable 的 world model 要有三种能力：**foresight**（执行前预判后果）、**imagination-driven planning**（用想象出的 rollout 比较候选动作）、**data amplification**（合成额外的示范 / 交互轨迹）。
- **World model 的定义**（`source/sec/2_Pre-knowledge.tex`，Eq.1）： $p(x_{t+1:t+H}\mid x_t,a_{t:t+H-1},l)$。这里把 **action 的含义放宽了**：低层电机指令 $a$ 算 action，高层语言 / 任务指令 $l$ 也算 action（语言指定"要实现哪种未来"）。 $x_t$ 可以是像素、latent、结构化物理状态，也可以是符号谓词。
- **功能性判据**：只能生成"看着合理"的未来视频，不算 world model。它必须刻画机器人相关的动作会怎样改变未来，并且能服务于 control / planning / simulation / evaluation / data generation。最重要的子类是 action-conditioned world model。
- **实际讨论的对象**是视觉 world model，也就是 video generation model $p(v_{t+1:t+H}\mid o_t,a_{t:t+H-1},l)$（Eq.2）。作者专门声明：聚焦视频不代表认为像素是最优的抽象。
- **Policy 背景**（Sec.2.2，Eq.3）： $p(a_{t+1:t+k}\mid o_t,l)$，其中 $k$ 是 action chunk 长度。分两类：visuomotor policy（Diffusion Policy 等）和 VLA（离散 token 的 RT-2 / OpenVLA / FAST，连续动作的 π 系列 flow matching）。**这部分明确只讲 imitation learning 范式。**
- **已有文献的问题**：按架构、功能角色和应用域都很碎片化。作者说自己比先前的综述（`worldmodel-survey`，arXiv 2511.02097）分得更细，也更以 policy 为中心。

## 核心方法

这篇综述没有提出方法，这一节整理它的分类框架。

### 0. 统一的概率视角（Sec.3.1，`source/sec/3_World_Model_Policy.tex`，Eq.4–8）

$$p(o_{t+1:t+k},a_{t+1:t+k}\mid o_t,l)$$

| 查询 | 形式 |
|---|---|
| Policy | $p(a_{t+1:t+k}\mid o_t,l)=\int p(o,a\mid o_t,l)\thinspace d_o$ |
| Passive WM（视频生成） | $p(o_{t+1:t+k}\mid o_t,l)=\int p(o,a\mid o_t,l)\thinspace d_a$ |
| Controllable WM | $p(o_{t+1:t+k}\mid o_t,a_{t+1:t+k})$ |
| Inverse Dynamics Model | $p(a_{t+1:t+k}\mid o_{t:t+k})$ |

在白板上可以这样推（这一步是我从 Eq.4–8 推出来的，原文只用文字说"policy 可以把 world model 生成的未来当作中间隐变量"）：
$p(a\mid o_t,l)=\int p(a\mid o_{t:t+k},l)\thinspace p(o_{t+1:t+k}\mid o_t,l)\thinspace do$。如果再假设 $a\perp l\mid o_{t:t+k}$（看到未来就不再需要指令），第一项就退化成 IDM。所以 "passive WM × IDM" 正好就是 IDM-style policy 的分解。

### 1. World Model for Policy：按"预测和动作生成怎么交互"分五类（Sec.3，Table 1，Fig.3/4）

1. **IDM-style（decoupled，Sec.3.2，Eq.9–12）**：先生成未来 $\hat{\mathbf o}_ {t+1:t+H}=\mathcal W(o_t,l)$（或 latent $\hat{\mathbf z}$），再用 $\pi=P(a_t\mid E_{img}(o_t),E_{text}(l),\Phi(\hat{\mathbf o}_ {t+1:t+H}))$ 解码 chunk 长度为 $H'$ 的动作。WM 通常是先预训练，再冻结或轻量适配。这条线的演化方向是让"交给 policy 的未来"越来越紧凑：
   - 显式像素 rollout：UniPi；VidMan / Vidar 加了 masked IDM；Gen2Act 生成的是人类视频。
   - video diffusion 的中间特征：VPP、Video2Act。
   - 部分去噪的 latent visual plan：MimicVideo。
   - 面向执行的中间表示：TC-IDM（以工具为中心的轨迹）、LVP。
   - 把视频当 in-context 提示：Say-Dream-ACT。
   - 另有一支用 3D 结构作中间量：AVDC（dense correspondence）、VidBot、NovaFlow（3D object flow）。

   **弱点**：效果被生成未来的保真度卡住；"看着合理但与动作不一致"的预测会累积误差。
2. **Single-backbone（Sec.3.3，Eq.13–14）**：把未来视觉和动作的表示拼成 $\mathbf x=[z^v;z^a]$，由共享 backbone 去噪： $\hat y=f_\theta(\tilde{\mathbf x}_ \tau,o_t,l,\tau)$， $\mathcal L=\mathbb E[\ell(\hat y,y)]$，其中 $y$ 可以是 diffusion noise、flow velocity 或 masked token。代表作：
   - UVA：学联合 latent，推理时可以跳过视频解码。
   - UWA：两个模态各用一个 timestep，靠控制 timestep 把视觉边缘化掉。
   - VideoVLA / VideoPolicy：直接把预训练视频生成器改造成 policy。
   - Cosmos Policy：把 action、未来 state 和 value 当作额外的 latent "帧"塞进原视频 diffusion 序列。policy 模式只取 action；planning 模式用 state + value 给候选轨迹排序。
   - DreamZero：自回归 flow-matching 的 video-action DiT，按 chunk 闭环联合去噪，以减少 compounding error。
   - UD-VLA（离散同步去噪）、GigaWorld-Policy（视觉分支在推理时可选）。

   **动机**：视频预训练带有运动连续性和时间因果先验，而 VLM 主要做的是图文对齐。但作者明确承认：**video backbone 是否稳定优于同规模的 VLM backbone，仍是开放问题**。
3. **MoE/MoT（Sec.3.4，Eq.15）**： $(\mathbf h^v_{\ell+1},\mathbf h^a_{\ell+1})=\mathcal F^{\mathrm{mix}}_ \ell(\mathbf h^v_\ell,\mathbf h^a_\ell;o_t,l)$。视频 expert 和动作 expert 参数分开，每层通过 joint / cross / shared attention 交互。结构上类似 π0 的 action expert，只是主干换成了视频生成器。三种做法：
   - 并行 expert：GE-Act 用深层 cross-attention 把视频 latent 注入 flow-matching 动作分支。
   - MoT 深度交互：Motus 有理解 / 视频 / 动作三个 expert；LingBot-VA 把视频和动作 token 交织成自回归序列；BagelVLA 用单步去噪的 Residual Flow Guidance；DiT4DiT。
   - latent expert：LDA-1B 在 DINO latent 里做预测；FRAPPE 对齐视觉基础模型的表示。

   **Fast-WAM 的结论**：收益可能主要来自训练时的视频 co-training，而不是推理时的未来想象。
4. **Unified VLA（Sec.3.5，Fig.4a）**：在 MLLM / UMM 主干里加入预测未来的目标。三个子类：
   - 显式预测未来图像：GR-1、UP-VLA、WorldVLA / RynnVLA-002（主要当训练信号用）。
   - 隐式 / latent 未来：DreamVLA 预测动态、空间、语义三类世界知识；UniVLA；CoWVLA 预测 latent motion。
   - 多 expert / 多系统：F1、InternVLA-A1、HALO、TriVLA。
5. **Latent-space WM（Sec.3.6，Fig.4b）**：完全在表示空间里预测，不解码图像，属于 JEPA 思路。
   - FLARE：把动作去噪网络的 hidden 与未来观测的 embedding 对齐。
   - VLA-JEPA：leakage-free，未来帧只用来生成 latent target。
   - JEPA-VLA：直接拿 V-JEPA 2 的预测性 embedding 当 backbone。
   - WoG：在 condition space 里只预测对控制最有用的那部分未来信息。
   - DIAL。
   - 另附 symbolic / planner-facing WM：在谓词、关系、operator 上建转移模型，交给 TAMP 规划器。

**总趋势**（Fig.2）：从 decoupled 走向更紧的耦合和内化。作者强调这是 "dominant directions rather than strictly sequential replacements"，不是依次替代。

### 2. World Model as Simulator（Sec.4，Fig.5）

**(a) RL（Sec.4.1，Eq.16–19）**

$$(\hat o_{t+1},\hat r_t,\hat d_t)\sim p_\phi(\cdot\mid o_{\le t},a_{\le t},l),\qquad J(\theta)=\mathbb E_{\hat\tau\sim(\pi_\theta,p_\phi)}\Big[\sum_t\gamma^t\hat r_t\Big]$$

实际训练用 Eq.18： $\mathcal L_{\mathrm{RL}}=-\mathbb E_t[\min(r_t\hat A_t,\mathrm{clip}(r_t,1-\epsilon,1+\epsilon)\hat A_t)]$， $r_t=\pi_\theta/\pi_{\theta_{old}}$。注意这个式子本身就是 [PPO](../../04_rl/ppo/notes.md) 的 clipped surrogate。综述叫它 "GRPO-style"，但没有解释为什么这样叫，也没有定义 $\hat A_t$。我的理解是：按 GRPO 的做法， $\hat A_t$ 取组内相对归一化的回报，不需要 critic。组相对优势的定义见 [DeepSeek-R1](../../04_rl/deepseek-r1/notes.md) 里的 GRPO。

- **一级范式：WM 就是环境。**
  - 奠定"动作条件模拟 + 奖励生成"配方：UniSim、World-Env、VLA-RFT。
  - DiWA：用冻结的、从 play data 学到的 WM，完全离线地适配 diffusion policy。
  - World4RL；World-Gymnast；PlayWorld。
  - RehearseVLA：用 instant reflector 给出奖励和终止信号。
  - WMPO：在像素空间想象，做 on-policy GRPO。
  - ProphRL：用 FA-GRPO / FlowScale 适配 flow 动作头。
  - RISE：组合式动力学 + progress value 估计。
  - GigaBrain-0.5M\*。
- **二级范式：承认 WM 本身不可靠，让它和 policy 共同进化。**
  - World-VLA-Loop：用 policy 失败的 rollout 修 WM。
  - VLAW：交替进行"用真实数据修 WM"和"用合成数据改 VLA"。
  - WoVR：Keyframe-Initialized Rollouts + 共进化：

$$\phi^{k+1}\leftarrow\mathrm{UpdateWM}\big(\phi^k,D_{\mathrm{real}}\cup D_{\mathrm{policy}}(\pi_{\theta^k})\big),\qquad\theta^{k+1}\leftarrow\mathrm{UpdatePolicy}\big(\theta^k,\hat D(\phi^{k+1})\big)$$

**(b) Evaluation（Sec.4.2）**

- **对候选做 rollout 再排序**：
  - GPC：部署时给冻结的 policy 外挂一个 action-conditioned WM，做 look-ahead 排序和精修。
  - IRASim。
  - World-in-World：闭环地在想象中规划并修正。
  - DreamPlan：用 WM rollout 构造偏好对，再拿来训练。
- **MPC**：TD-MPC2、LeWorldModel 在 latent 空间做 MPC。
- **WM 直接当 policy evaluator**：
  - 用 Veo world simulator 评估 Gemini Robotics policy（离线评估、OOD、safety）。
  - WorldEval：在想象中给不同 policy 和 checkpoint 排序，同时当 safety detector。
  - WorldArena。
- **带反馈头的 simulator**：World-Env（连续奖励 + 终止预测）、VLA-RFT（verified reward）、World-VLA-Loop、RISE（progress value model）。
- **JEPA 线**（V-JEPA 2 / 2.1、LeWorldModel）只被定位为 adjacent direction，还不算成熟的 evaluator。
- **核心判据**：只有当想象出的未来保留了候选动作的因果后果，evaluator 才有用。WoVR 指出，幻觉和长时程误差会直接污染评估信号。

### 3. Robotic Video World Model 的能力阶段（Sec.5，Table 2，Fig.6）

Table 2 按四组列出，每个方法标 Task-cond. / Action-cond. / Structure-aware / Foundation-scale / Main use：

| 组 | 代表作与要点 |
|---|---|
| Imagination-based | UniPi；VLP；Dreamitate；RoboDreamer（组合式）；ManipDreamer（action tree）；DreMa（Gaussian Splatting + 物理仿真器，把想象变成可学习的 digital twin）；PhysWorld（从生成视频重建物理世界 + object-centric residual RL）；DreamGen（neural trajectories，再用 latent action / IDM 补动作标签） |
| Action-controllable | IRASim（每个 transformer block 内做 frame-level action conditioning）；RoboEnvision；RoboMaster；Ctrl-World（多视角联合预测 + frame-level 控制 + memory 支撑长时程）；EnerVerse-AC；Interactive World Simulator；EVA（用 inverse-dynamics reward 做 post-training，对齐可执行性） |
| Structure-aware | Mask2IV；TesserAct（RGB + depth + normal 的 4D 表示）；RoboVIP（多视角 identity prompting） |
| Foundation video WM | Vid2World（把预训练 video diffusion 改成交互式 WM）；Genie Envisioner；DreamDojo（人类第一视角视频 + 连续 latent action）；WoW；UnifoLM-WMA-0；Cosmos Predict 2.5；GigaWorld-0（视频分支 + 3D 物理分支）；ABot-PhysWorld |

Sec.5.6 对瓶颈的总结：难点已经不是生成逼真的未来，而是生成**与动作因果对齐、长时程物理自洽、跨视角和跨本体一致、能被执行**的未来。

### 4. 其他域（Sec.6，略读）

- 导航：Pathdreamer、VISTA/VISTAv2、NWM、SparseVideoNav、EgoWM。
- 驾驶：MILE、OccWorld、GAIA-1、DriveDreamer、Drive-WM、UniDWM、DriveWorld-VLA、DriveVLA-W0、SteerVLA。

## 关键实验与证据

**Benchmark 的三层框架（Sec.7.1）**

1. **Open-loop 的动作条件生成质量**：RBench、EWMBench（拆成场景一致 / 运动正确 / 语义对齐三项）、DreamGen Bench、EVA-Bench。
2. **Closed-loop 的任务效用 / policy 评估**：
   - WorldArena：感知 + 功能（数据生成、policy 评估、规划）。
   - WorldEval：看 policy 排序是否一致。
   - WorldGym：Monte Carlo 估计 policy value。
   - World-in-World：把 WM 嵌进在线规划，直接测任务成功率。
3. **物理一致性 / 可控性 / 可执行性诊断**：
   - WorldSimBench：perceptual + manipulative 两部分，看 IDM 能否从生成视频恢复动作。
   - WoW-World-Eval：IDM-based Turing Test。
   - DrivingGen；WM-ABench（按原子能力拆解）。

作者的结论：rank consistency、value fidelity、decision reliability 比像素精度更有信息量；视觉逼真度只是 control utility 的弱代理。

**数据集（Sec.7.2，Table 3/4）**

- 共 27 个资源。除 Table 3 的 Year、Source 两列外，其余列都只用 ✓ / – / ✗ 做定性标注。两张表分别看：
  - 数据属性（Table 3）：X-Emb.、Act.、Obs./3D、Lang.、M/C。
  - 支持哪种建模能力（Table 4）：General Traj.、Long-Horizon、X-Emb. Scaling、Human Prior、Contact / Physics、Synth. / Recipe。
- 包括 OXE、DROID、BridgeData V2、AgiBot World、RoboMIND 2.0、UMI 系列、EgoMimic、RH20T、RoboTwin 2.0、Action100M 等。
- **表里没有任何规模数字。**
- 结论：失败恢复、决策敏感的变化、密集的物理监督，这几类数据都远少于成功示范。

**汇总结果（Sec.7.3，Table 5/6；数字全部摘自各方法原论文，不是统一复跑）**

- **LIBERO**（Table 5，Spatial / Object / Goal / Long / Avg）：

  | 方法 | Spatial | Object | Goal | Long | Avg |
  |---|---|---|---|---|---|
  | Cosmos Policy | 98.1 | 100.0 | 98.2 | 97.6 | 98.5 |
  | LingBot-VA | 98.5 | 99.6 | 97.2 | 98.5 | 98.5 |
  | Say-Dream-ACT | 99.4 | 99.2 | 98.6 | 95.4 | 98.1 |
  | TriVLA | 91.2 | 93.8 | 89.8 | 73.2 | 87.0 |

  另外：Motus Avg 97.7，RynnVLA-002 97.4，VLA-JEPA 97.2，JEPA-VLA 96.4；UniPi 只报了 Long，为 0.0。
- **RoboTwin**（Table 6，RT-A 非随机 / RT-B 随机化环境）：
  - 在随机化下保持得住：LingBot-VA 92.9 / 91.6，InternVLA-A1 89.4 / 87.0，Motus 88.7 / 87.0。
  - 在随机化下大幅下滑：Vidar 65.8 / 17.5，BagelVLA 75.3 / 20.9，HALO 80.5 / 26.4，JEPA-VLA 73.5 / 17.7。
- **SIMPLER-style**（S-G = Google Robot，S-W = WidowX，S-O = 其他设置）：VideoVLA S-G 73.1/62.8（"a/b" 表示原论文的两种协议变体），WoG S-G 69.4、S-W 63.5，CoWVLA S-W 76.0。
- **CALVIN**（C-A / C-D 列，表注写的是 "CALVIN ABCD and ABCDD protocols"，原文如此）：数值从 0.92（UniPi）到 4.64（UD-VLA，C-D）。表中没有注明指标是什么。

作者从这些表得出三点：

1. WM 类方法在标准操作 benchmark 上已经很实用。
2. 五种范式里都有高分，说明照片级视频生成不是有效控制的必要条件。
3. 剩下的难点是长时程鲁棒性（Goal / Long suite 掉分更常见）、跨 benchmark 泛化、报告口径不统一。

## 局限与论文没说的

**作者承认的（Sec.8，以及散见各处）**

- **Causal conditioning gap**：预测的未来更多由历史和任务意图决定，而不是由待执行的动作决定。结果是语义上合理，但不忠实于动作后果（8.1）。
- **效率**：WM-based policy 的训练和推理都远比 VLA 贵，diffusion 的迭代去噪带来延迟。现有缓解办法：partial denoising（MimicVideo、LingBot-VA）、latent 模型（LeWorldModel）、训练时用 WM 而推理时去掉（Fast-WAM）（8.2）。
- **多模态**：只靠视觉 + 本体感觉，看不到摩擦、刚度、接触稳定性；触觉信号高频但低维，在联合 latent 里容易被视觉淹没（8.3）。
- **与经典控制结合**：MPC 要反复做 world model rollout，计算开销大，限制了大模型的实时部署；如何让 learned dynamics 与 Lyapunov 稳定性、robust control 这类形式化保证相容，仍是前沿问题（8.4）。
- **Symbolic 结构**：像素 rollout 的长时程误差会累积，symbolic 表示能缓解，但需要合适的抽象和把感知对齐到符号上（8.5）。
- **评测**：没有公认的统一指标（8.6）。
- video backbone 能否胜过同规模 VLM，作者说仍是开放问题（3.3）。

**我读出来的**

1. **Table 5/6 不是受控比较**。数字直接取自各原论文，预训练数据、参数量、微调协议、试验次数都不一样。更关键的是，表里**没有 π0、OpenVLA 这类标准 VLA 作对照**（唯一看着像普通 VLA 的是 Table 6 的 LingBot-VLA，bib 标题为 "A pragmatic VLA foundation model"，正文没有介绍，被归在 MoE/MoT 组），所以从这些表推不出"加 world model 带来增益"。"每种范式都能高分"也可能只是 LIBERO 已经饱和（多个方法 Avg ≥ 97）。RoboTwin RT-B 上的大幅掉点反而更能区分方法。
2. **经典 latent MBRL 线几乎缺席**。参考文献（`paper.bbl`）里没有 Dreamer / DreamerV3 / Dreamer 4 / RSSM / Genie / DIAMOND（只有 Genie Envisioner，不是 Genie），MBRL 一侧只引了 Ha & Schmidhuber、DayDreamer、TD-MPC 和 TD-MPC2（TD-MPC 只在 Sec.8.4 出现）。Sec.4.1 讲的 "RL in world model" 只覆盖 VLA / diffusion policy 的 post-training，没碰 imagination actor-critic、model bias、rollout 长度与 value bootstrapping 这些 MBRL 的核心问题。
3. **没有量化 learned simulator 的 reality gap**。Sec.4 列了很多"WM 当 simulator"的方法，却没有汇总"WM 内成功率 vs 真实成功率"的相关性，也没有 policy 排序一致性的数字（WorldEval、WorldArena 都只是定性描述）。
4. **没有 sim-to-real / TTT 视角**。全文没有 domain randomization、system identification、test-time adaptation 的讨论。WM 和现实之间的偏差，只在"共进化时用真实数据修 WM"（WoVR / VLAW / World-VLA-Loop）里一笔带过。
5. **Policy 背景只讲 imitation learning**（Sec.2.2 明说了）。reward model 的设计、on-policy 与 off-policy 的取舍都没有展开；Eq.18 叫 "GRPO-style"，却没写出组相对优势。
6. **引用错位，查原文时以 bib 标题为准**：
   - Sec.4.1 的 "World-Gymnast" 引的是 `quevedo2025worldgym`，那是 WorldGym（arXiv 2506.00613，一篇 policy 评估论文）；bib 里真正的 World-Gymnast 条目 `sharma2026world`（arXiv 2602.02454）没有被引用。
   - Table 1 的 GE-Act 引的是 `gen2act`（Sec.3.4 正文里 GE-Act 引的是 `liao2026genie`，即 Genie Envisioner）。
   - "UniVLA" 在 Sec.3.5 / Table 1（`unifiedvla`）和 Table 5（`bu2025univla`）指向不同的论文；Table 5 另把 `unifiedvla` 单列为 "Unified VLA"。
7. **算力完全没报告**。single-backbone 和 MoT 类方法基本都建在大视频 DiT 上（Wan、Cosmos 等），门槛显著高于 VLA 微调，但综述没给任何算力数字。
8. **时效**：Fig.2 最后一栏是 2026.1–2026.3，之后的工作只能靠 GitHub 列表跟进。

## 在脉络中的位置

- **前驱**：
  - [World Models](../world-models/notes.md)：和 Craik (1943)、Bryson & Ho (1975) 一起，被引为 world model 这个概念的思想来源之一（bib 条目是它的 NeurIPS 2018 版本 "Recurrent world models facilitate policy evolution"）。
  - UniPi（文本引导的视频 → IDM）是 decoupled 路线的早期代表；GR-1 是 unified VLA 里"在同一个 GPT 式 transformer 中联合预测动作和未来图像"的早期代表。
  - [UniSim](../../05_intersections/unisim/notes.md)、[DayDreamer](../../05_intersections/daydreamer/notes.md)：WM 当 simulator 训 policy 的早期代表。
  - [TD-MPC2](../tdmpc2/notes.md)：被引为 latent MPC 的代表。
  - [V-JEPA 2](../vjepa2/notes.md)：latent-space 路线，也是 JEPA-VLA 的 backbone。
  - [Cosmos WFM](../cosmos-wfm/notes.md)：综述没有引用这篇（arXiv 2501.03575），引的是后续的 Cosmos Predict 2.5（`CosmosPredict`，arXiv 2511.00062）和 Cosmos Policy。把 Cosmos WFM 看作它们的前身平台，是我补的脉络，不是综述的说法。
- **综述收录的本项目论文**：
  - [DreamZero](../dreamzero/notes.md)：Table 1 的 single-backbone 类。
  - [Ctrl-World](../../05_intersections/ctrl-world/notes.md)：Sec.4.2 / 5.3，action-controllable + 评估。
  - [WMPO](../../05_intersections/wmpo/notes.md)：Sec.4.1，像素空间想象 + on-policy GRPO。
  - [World-Gymnast](../../05_intersections/world-gymnast/notes.md)：Sec.4.1（注意上面说的引用键错误）。
  - [SIMPLER](../../03_real2sim2real/simpler/notes.md)：Table 6 的评测协议。
  - [π\*0.6](../../04_rl/pi-star-06/notes.md)：只在 Sec.2.2 作为 π 系列被提到。
- **后继 / 影响**：作为 2026 年的综述，它最直接的价值是"WM × VLA"文献的入口索引，GitHub 列表会持续更新。
- **与本项目其他主题的联系**：
  - **world model**：本项目 01 目录偏 MBRL 和生成式 WM（Dreamer 系、Genie、DIAMOND），这篇补上了"VLA + 视频 WM"那一侧。建议和 [DreamerV3](../dreamerv3/notes.md)、[Dreamer 4](../dreamer4/notes.md) 对照读，能看清"在 latent imagination 里做 actor-critic"和"VLA 在像素 WM 里做 GRPO"到底差在哪。
  - **RL**：
    - Eq.16–18 本质上就是把 PPO / GRPO 搬进 learned simulator。
    - ProphRL 的 FA-GRPO 与 [π_RL](../../04_rl/pi-rl/notes.md)、[SimpleVLA-RL](../../04_rl/simplevla-rl/notes.md) 同属 flow / VLA 的 RL post-training。
    - [Robotic World Model](../../05_intersections/robotic-world-model/notes.md)、[SimDist](../../05_intersections/simdist/notes.md) 属于同一类工作，但综述没收录。
  - **real2sim2real**：
    - DreMa 与 [SplatSim](../../03_real2sim2real/splatsim/notes.md)（都是 Gaussian Splatting 场景 + 物理仿真器）、PhysWorld 与 [PhysTwin](../../03_real2sim2real/phystwin/notes.md)（都是从视频重建物理世界）同一思路。
    - WoVR / VLAW 的"用真实数据修 WM"，本质上是给 learned simulator 做 real-to-sim 校准，可以类比 [SimOpt](../../03_real2sim2real/simopt/notes.md) 在闭环里调仿真参数。
    - RoboTwin 2.0 的 bib 标题里写着 strong domain randomization，对应 [Domain Randomization](../../03_real2sim2real/domain-randomization/notes.md)。
  - **TTT**：综述没有 TTT 章节。GPC（部署时用 WM 做 look-ahead）和 Fast-WAM（"要不要 test-time future imagination"）讨论的都是**推理时多花计算**，不是**推理时更新参数**。"部署时用自监督信号更新 WM 或 policy"（[PAD](../../05_intersections/pad/notes.md)、[TTT](../../02_test_time_training/ttt-self-supervised/notes.md)、[Tent](../../02_test_time_training/tent/notes.md)）在这篇综述里是空白，这正是本项目可以切入的交叉点。

## 前置知识

1. Imitation learning 下的 policy 形式与 action chunking；Diffusion Policy 和 flow matching 动作头。
2. VLA 的基本架构：RT-2 / OpenVLA 的离散 token，对比 π0 的 flow-matching action expert。
3. Video diffusion、DiT、latent diffusion（VAE latent、timestep、三种去噪目标：noise / velocity / masked token）。
4. Inverse dynamics model、latent action。
5. Mixture-of-Transformers，以及共享 attention 的多 expert 结构。
6. JEPA：在 embedding 空间里做预测（V-JEPA 2）。
7. MBRL 基础：learned dynamics、imagined rollout、MPC；PPO 的 clip 目标，GRPO 的组相对优势。
8. 评测协议：LIBERO、CALVIN、SIMPLER、RoboTwin 2.0。

## 复现可行性

- **代码**：综述本身没有代码。GitHub 仓库是 paper list，按五个 Policy 子类 / Simulator / Video Generation / Benchmarks / Datasets 组织。
- **算力**：综述没有报告任何方法的算力。
- **综述汇总结果时用的下游 benchmark**（Sec.7.3，综述只是汇总，没有做推荐）：
  - LIBERO 4-suite（Table 5）。综述认为 RoboTwin / CALVIN / SIMPLER 在本体和协议上更异质、不适合严格排名，LIBERO 相对更好比；"已接近饱和"是我的判断（多个方法 Avg ≥ 97）。
  - RoboTwin 2.0（Table 6）：随机化的 RT-B 区分度更高，这也是我从表里读出来的，不是综述的结论。
  - CALVIN；SIMPLER（Google Robot / WidowX）。
- **WM 本身的评测**（Sec.7.1）：
  - open-loop：EWMBench / RBench / DreamGen Bench。
  - closed-loop：WorldArena / WorldEval / WorldGym / World-in-World。
  - 诊断：WorldSimBench / WoW-World-Eval / WM-ABench。
- **训练数据**（Table 3/4）：
  - 通用轨迹：OXE、DROID、BridgeData V2。
  - 跨本体 + 接触：RoboMIND 2.0。
  - 接触丰富：RH20T。
  - 仿真合成：RoboTwin 2.0。
- **最小可复现实验建议**（这是我的设计，直接测综述没量化的 learned simulator reality gap）：
  1. 在 LIBERO 的一个 suite 上，用示范数据微调一个开源的 action-conditioned video WM（Ctrl-World / IRASim 这类）。
  2. 把 LIBERO 的物理仿真器当作"真实世界"，按 WorldEval 的思路，比较几个 policy checkpoint 在 WM 内的排序和在仿真器里真实成功率的排序，算 Spearman 相关。
  3. 再做一次小规模的 WMPO / VLA-RFT 式 GRPO post-training，对比"WM 内的成功率提升"和"仿真器内的真实提升"差多少。这一步需要自己训一个奖励 / 终止判别器，做法参考 World-Env / VLA-RFT。

  全程不需要真机。
- **预计难点**：
  - 视频 WM 微调的显存和算力门槛（综述没给数字，要去查所选方法的原论文）。
  - action faithfulness 不足会直接让排序相关性失真。
  - 奖励 / 终止判别器需要额外训练。

## 综述强调、但本项目未收录的重要工作

1. **UniPi**：Learning Universal Policies via Text-Guided Video Generation（NeurIPS 2023；bib 没给 arXiv 号）。综述称它是 IDM-style 的代表性早期工作，排在 Fig.2 最早一栏（2023.1–2025.1）；LIBERO-Long 0.0（Table 5）。
2. **Cosmos Policy**（arXiv 2601.16163）：把 action / state / value 当作 latent 帧塞进视频 diffusion；LIBERO Avg 98.5（Table 5）。
3. **LingBot-VA**，即 Causal World Modeling for Robot Control（arXiv 2601.21998）：MoT，视频和动作 token 交织自回归；LIBERO 98.5，RoboTwin 92.9 / 91.6（Table 5/6）。
4. **Fast-WAM**（arXiv 2603.16666）：质疑推理时想象的必要性，认为收益来自训练时的视频 co-training。
5. **VLA-RFT**（arXiv 2510.00406）与 **World-Env**（arXiv 2509.24948）：在 WM 里做 VLA 的 RL post-training，前者用 verified reward，后者预测奖励和终止。
6. **WoVR**（arXiv 2602.13977）：把 simulator 可靠性当作核心瓶颈；Keyframe-Initialized Rollouts + WM–policy 共进化（Eq.19）。
7. **WorldEval**（arXiv 2505.19017）与 **Evaluating Gemini Robotics Policies in a Veo World Simulator**（arXiv 2512.10675）：把 WM 当 policy evaluator。
8. **DreamGen**（CoRL 2025，bib 没给完整 arXiv 号）：用 neural trajectories 放大数据，并提出 DreamGen Bench。
9. **LeWorldModel**（arXiv 2603.19312）：端到端 JEPA，做 latent MPC，还能检测物理上不合理的事件。
10. **WorldArena**（arXiv 2602.08971）：同时测感知和功能效用的统一 embodied WM benchmark。

## 自测问题

1. 从联合分布 Eq.4 出发，写出 policy / passive WM / controllable WM / IDM 四种查询。IDM-style policy 等于哪两项的组合？需要什么条件独立假设？
2. Cosmos Policy 和 Fast-WAM 都和视频模型联合训练，但对"推理时要不要生成未来"给出了不同答案。结合 Table 1 的 "Future Generation at Inference" 列，说明这对延迟和性能分别意味着什么。
3. 在 learned simulator 里做 GRPO（Eq.16–18）时，哪些误差来源会让"WM 内回报提升"兑现不到真实环境？WoVR 的共进化（Eq.19）针对的是其中哪一个？
4. 作者为什么说 visual realism "neither necessary nor sufficient"？请构造一个 open-loop 指标很高、closed-loop 却失败的机制。
5. 为什么 Table 5 不能证明"加 world model 让 VLA 更强"？要得出这个结论，需要什么样的对照实验？
