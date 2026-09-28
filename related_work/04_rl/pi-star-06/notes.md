# π*0.6: a VLA That Learns From Experience

> arXiv [2511.14759](https://arxiv.org/abs/2511.14759) · Physical Intelligence 技术报告（arXiv preprint）, 2025 · 主题：04_rl（扩散/flow 策略与 VLA 的 RL 微调） · 层级：frontier
> 代码：none（论文只给博客 https://pi.website/blog/pistar06；官方 [openpi](https://github.com/Physical-Intelligence/openpi) 只放出 π0 / π0-FAST / π0.5，没有 π0.6、π*0.6 权重，也没有 RECAP 训练代码） · 笔记依据：`source/main.tex`（数字/公式）+ `paper.pdf`（图）+ `source/figures/experiment*/`（柱状图原图）

## 一句话

RECAP（RL with Experience and Corrections via Advantage-conditioned Policies）先用 Monte Carlo 回报训练一个 distributional value function，再把每个动作的 n-step advantage 二值化成文本 "Advantage: positive/negative"，作为条件去监督训练 flow-matching VLA。推理时把条件设成 positive，就等价于从一个有理论保证的改进策略里采样。这样示范、人类纠正和自主 rollout 就都能放进同一个可迭代的真实世界 offline RL 流程。

## 问题设定

- **目标**：让已经很强的通用 VLA（π0.6）在真实部署中"练习"，同时提高成功率和速度（throughput）。模仿学习有两个上限：示范质量，以及 compounding error（DAgger 指出的问题）。
- **输入/输出**（`main.tex` Sec V-A，l.337-352）：观测 $\mathbf o_t=[\mathbf X^1_t,\dots,\mathbf X^n_t,\mathbf q_t]$，即多路相机图像加关节/夹爪位置；语言 $\ell=\ell_t+s$，即任务 prompt 加 metadata。模型输出三样东西：子任务文本 $\hat\ell$（如 "pick up the coffee cup"）；FAST 离散化动作 token $a^\ell_{t:t+H}$，只在训练时用；action expert 用 flow matching 生成的连续 action chunk $\mathbf a_{t:t+H}$，内容是 50 Hz 的关节角加夹爪指令。
- **假设**：把观测当作 Markov 状态（脚注承认这一般不成立）；不打折扣；奖励只来自人工给的 episode 级成功标签，是稀疏的；允许专家遥操作实时接管（human-gated DAgger）；场景复位和打标签都靠人。
- **之前的方法卡在哪**（Sec II、IV-B）：
  1. 直接对 VLA 跑 PPO（SimpleVLA-RL、π_RL 等）需要 on-policy 数据，很难在真实世界高效扩展。
  2. flow matching 没有 tractable log-likelihood，policy gradient 和 reparameterized gradient 都难套上去。
  3. AWR 这类加权回归会丢弃或大幅降权大部分数据，效果接近 filtered imitation。
  4. residual policy、只微调动作头、noise-space steering（DSRL）这些做法只训练模型的一部分，不是端到端训练整个 VLA。
  5. 以前用 value function 训 VLA 的工作，多是离散动作模型配短而简单的任务。

## 核心方法

**0. 理论出发点**（Sec III，l.212-215）。正则化 RL 的目标是 $\max_\pi \mathbb E[\sum r]-\beta\thinspace\mathbb E[D(\pi\Vert\pi_{\text{ref}})]$。当 $D$ 取 KL 时，解是熟知的 $\hat\pi\propto\pi_{\text{ref}}\exp(A^{\pi_{\text{ref}}}/\beta)$，这也是 AWR/REPS 的来源。RECAP 用的是一个不那么常见的形式。先定义"改进事件" $I$ 的概率 $p(I\mid A)=g(A)/\int g(A(\mathbf o,\mathbf a'))d\mathbf a'$，其中 $g$ 单调递增。那么

$$\hat\pi(\mathbf a\mid\mathbf o)\propto\pi_{\text{ref}}(\mathbf a\mid\mathbf o)\thinspace p\big(I\mid A^{\pi_{\text{ref}}}(\mathbf o,\mathbf a)\big)^\beta$$

保证 $\mathcal J(\hat\pi)\ge\mathcal J(\pi_{\text{ref}})$（论文引 MARWIL、CFGRL 作为依据）。

**1. Distributional value function**（Sec IV-A Eq.1，l.233-241；Sec V-C Eq.5，l.377-385）。奖励定义为

$$r_t=\begin{cases}0 & t=T\ \text{且成功}\cr -C_{\text{fail}} & t=T\ \text{且失败}\cr -1 & \text{otherwise}\end{cases}$$

所以成功 episode 的 value 等于"负的剩余步数"，失败 episode 的 value 是一个很大的负数。value 按每个任务的最大 episode 长度归一化到 $(-1,0)$。论文没给 $C_{\text{fail}}$ 的具体值。

value function $p_\phi(V\mid\mathbf o_t,\ell)\in\Delta_B$ 输出 $B=201$ 个 bin 上的分布。训练时把经验回报离散化成 $R^B_t(\tau)$，再做交叉熵：

$$\min_\phi\ \mathbb E_{\tau\in\mathcal D}\Big[\sum_{\mathbf o_t\in\tau}H\big(R^B_t(\tau),\thinspace p_\phi(V\mid\mathbf o_t,\ell)\big)\Big],\qquad V^{\pi_{\text{ref}}}(\mathbf o_t,\ell)=\sum_b p_\phi(V{=}b\mid\mathbf o_t)\thinspace v(b)$$

这里的关键是：这是 **Monte Carlo 估计**，估的是数据集所代表的行为策略 $\pi_{\text{ref}}$ 的 value。 $\pi_{\text{ref}}$ 是示范、历代策略和人类纠正混在一起的策略。这里不做 TD bootstrap，也不学 Q。作者承认它不如 off-policy Q 估计，但简单、可靠（l.251）。value function 的架构与 VLA 相同，只是 backbone 更小：670M 的 VLM（Fig.3 标注为 SigLIP 400M + Gemma 270M），从 Gemma 3 初始化，并混入少量 web 多模态数据来防止过拟合（l.387）。

**2. Advantage 估计**（App F，l.752）。post-training 阶段用 $N=50$ 的 n-step 估计：

$$A(\mathbf o_t,\mathbf a_t)=\sum_{t'=t}^{t+N-1}r_{t'}+V(\mathbf o_{t+N})-V(\mathbf o_t)$$

预训练阶段改用 $N=T$，即整条回报减 $V(\mathbf o_t)$。这样每个样本只需调用一次 value function，可以在 VLA 训练中 on-the-fly 计算。代价是方差更大，作者称在大规模多任务数据上已经够用。

**3. Advantage-conditioned policy extraction**（Sec IV-B Eq.2-3，l.275-295）。由 Bayes 公式， $p(I\mid A)=\pi_{\text{ref}}(\mathbf a\mid I,\mathbf o)/\pi_{\text{ref}}(\mathbf a\mid\mathbf o)$。代入上式得到

$$\hat\pi(\mathbf a\mid\mathbf o,\ell)\propto\pi_{\text{ref}}(\mathbf a\mid\mathbf o,\ell)\left(\frac{\pi_{\text{ref}}(\mathbf a\mid I,\mathbf o,\ell)}{\pi_{\text{ref}}(\mathbf a\mid\mathbf o,\ell)}\right)^{\beta}\quad\text{(Eq.2)}$$

当 $\beta=1$ 时， $\hat\pi=\pi_{\text{ref}}(\mathbf a\mid I,\mathbf o,\ell)$。也就是说，**改进策略就是"以 I=True 为条件的行为策略"**。因此不需要显式学习 $p(I\mid A)$，只要一个模型能同时表示条件分布和无条件分布，这与 classifier-free guidance 的思路相同，即 CFGRL。

$I$ 用阈值指示函数定义： $I_t=\mathbb 1\big(A^{\pi_{\text{ref}}}(\mathbf o_t,\mathbf a_t,\ell)>\epsilon_\ell\big)$。训练目标为

$$\min_\theta\ \mathbb E_{\mathcal D_{\pi_{\text{ref}}}}\big[-\log\pi_\theta(\mathbf a_t\mid\mathbf o_t,\ell)-\alpha\log\pi_\theta(\mathbf a_t\mid I_t,\mathbf o_t,\ell)\big]\quad\text{(Eq.3)}$$

它比 AWR 好用，原因有三点：
- 所有样本都走普通监督学习，没有 importance weight，也不需要 likelihood ratio，所以可以直接用在 flow matching 上。
- 负样本不会被丢掉：它们教模型"negative 长什么样"，同时支撑无条件分布。
- 人类纠正片段一律强制 $I_t=\text{True}$，前提是假设专家纠正总是好的。

**4. 在 π0.6 上的实现**（Sec V-A/B Eq.4，App C）。π0.6 在 π0.5 基础上做了三处改动：加入更多机器人平台的数据，VLM 换成 Gemma 3 4B，action expert 扩到 860M。训练按 Knowledge Insulation（KI）：VLM 部分做 next-token 预测（子任务文本、FAST 动作 token、web 数据）；action expert 用 flow matching，并用 stop gradient 阻止 action expert 的梯度回传进模型其余部分（VLM）。

π*0.6 只多做一件事：在序列中 $\hat\ell$ 之后、动作之前插入文本 "Advantage: positive / negative"，所以它只影响动作部分的似然。连续动作部分用 flow-matching loss 近似似然下界：

$$\log\pi_\theta(\mathbf a_{t:t+H},a^\ell_{t:t+H}\mid I_t,\mathbf o_t,\ell,\hat\ell)\ \ge\ \mathbb E_{\eta,\omega}\Big[\log p_\theta(a^\ell_{t:t+H}\mid I_t,\cdot)-\alpha_\eta\big\Vert\omega-\mathbf a_{t:t+H}-f_\theta(\mathbf a^{\eta,\omega}_ {t:t+H},I_t,\cdot)\big\Vert^2\Big]$$

其中 $\mathbf a^{\eta,\omega}=\eta\mathbf a+(1-\eta)\omega$， $\omega\sim\mathcal N(0,\mathbf I)$（Eq.4）。App C 按 Kingma 的 diffusion ELBO 推导，取权重 $w(\eta)=e^{-\eta/2}$。实际训练中不调 $\alpha$，而是以 30% 的概率丢掉 indicator（App F，l.755）。这样同一个模型既是条件模型，也是无条件模型。

**5. 阈值设定与推理**（App E/F）。 $\epsilon_\ell$ 按比例来设：
- 预训练时，让约 30% 的示范数据为 positive（在 10k 个随机样本上估计）。
- 微调时，让每轮约 40% 的 rollout 为 positive。
- T-shirt/shorts 任务上，用高质量示范训练出的策略"慢但成功率高"，所以把 positive 比例压到约 10%。

正文 l.395 的写法是"value 的 30% 分位"，和附录"约 30% 为 positive"的说法对不上，按附录理解更自洽。

推理默认 $\beta=1$，即直接以 positive 为条件采样。也可以用 CFG 取 $\beta>1$，在 flow 推理时沿下式积分：

$$\nabla_{\mathbf a}\log\pi_\theta(\mathbf a\mid\mathbf o,\ell)+\beta\big(\nabla_{\mathbf a}\log\pi_\theta(\mathbf a\mid I,\mathbf o,\ell)-\nabla_{\mathbf a}\log\pi_\theta(\mathbf a\mid\mathbf o,\ell)\big)$$

但大 $\beta$ 会把动作推向学到的支撑集边缘，动作变得过激，而且对自回归部分不起作用。所以作者主要靠训练时的 $\epsilon_\ell$ 在"正则化"和"最优性"之间取舍，CFG 只取中等值 $\beta\in[1.5,2.5]$（l.744）。

**6. 整体流程**（Algorithm 1，l.309-327）。
- **预训练**：在数万小时的多机器人示范上训练 $V_{\text{pre}}$（Eq.1），再用它 on-the-fly 算 $I_t$ 来训练 $\pi_{\text{pre}}$（Eq.3），得到"RL pre-trained π*0.6"。
- **任务 $\ell$ 的初始化**：用该任务的示范微调， $I_t$ 固定为 True，相当于 SFT，得到 $\pi^0_\ell$（Algorithm 1 里同时也从 $V_{\text{pre}}$ 微调出 $V^0_\ell$）。
- **迭代 $k=1..K$**：
  1. 用 $\pi^{k-1}_ \ell$ 采集数据，一部分全自主，一部分有专家接管，全部并入 $\mathcal D_\ell$。
  2. **从预训练 checkpoint** 重新微调 $V^k_\ell$ 和 $\pi^k_\ell$，而不是在上一轮基础上继续训；作者称这样可以避免多轮漂移（l.404）。

这是批式的 iterated offline RL，不是实时在线更新。

## 关键实验与证据

**平台**（Fig.5）：静态双臂，两条 6-DoF 机械臂加平行夹爪，50 Hz 关节位置控制，3 路相机（底座 1 路、腕部 2 路）。

**任务**（Sec VI-A）：

| 任务 | 说明 | 时限 |
|---|---|---|
| Laundry T-shirt/shorts | 叠一件 T 恤或短裤并放好 | 200 s |
| Diverse laundry | 训练覆盖 11 类衣物，只在最难的 button-up shirt 上计分 | 500 s |
| Targeted failure removal | 从固定平铺的橙色 T 恤开始，必须领口朝上 | 200 s |
| Cafe | 双份 espresso，不能有严重失误 | 200 s |
| Box assembly | 在工厂现场，从平板纸板到组装、贴标签、放进箱子 | 600 s |

**指标**：throughput（每小时成功次数，同时反映速度和成功率）；success rate（由人工多维打分汇总）。误差棒为标准误。

**论文没有数值表。** 下面带"≈"的数字都是从 Fig.7-12 的柱状图（`source/figures/experiment*/`）目测读出的，精度约 ±1 个单位；不带"≈"的是正文原文数字。

**主结果**（Fig.7 throughput / Fig.8 success）：

| 任务 | π0.6 pretrain | π*0.6 offline RL pretrain | + SFT | π*0.6（RECAP） |
|---|---|---|---|---|
| T-shirt/shorts throughput（/h） | ≈17 | ≈17 | ≈33 | ≈60 |
| T-shirt/shorts success | ≈64% | ≈77% | ≈90% | ≈96% |
| Diverse laundry throughput | ≈3.9 | ≈5.0 | ≈3.9 | ≈8.4 |
| Diverse laundry success | ≈38% | ≈49% | ≈45% | ≈75% |
| Espresso throughput | ≈4 | ≈9.6 | ≈14.7 | ≈29 |
| Espresso success | ≈11% | ≈27% | ≈40% | ≈92% |
| Box throughput | ≈2.6 | ≈1.5 | ≈9.9 | ≈13.3 |

- π0.5 pretrain 在 T-shirt 上约 ≈3/h、成功率 ≈17%，在 diverse laundry 上为 0。
- 正文结论（l.528-530）：从 offline RL+SFT 到最终模型，diverse laundry 和 espresso 的 throughput "more than doubles"，失败率大约减半；除 diverse laundry 外，最终成功率都在 90% 以上。按图读数，diverse laundry 的失败率从 ≈55% 降到 ≈25%，与"减半"相符；espresso 从 ≈60% 降到 ≈8%，降幅远大于减半。
- Box 的分阶段成功率（retrieval / folding / labeling / stacking）在最终模型上都约为 91-96%，失败多半是超时。
- 部署展示（Sec I）：连续做 espresso 13 小时；在一个新家庭里连续叠衣 2 小时以上没有中断。

**多轮迭代**（Fig.9/10，正文 l.551-566）：
- T-shirt 只用自主数据（没有人类纠正），每轮在 4 台机器人上收集 300 条轨迹。throughput 从 ≈33 到 ≈42 再到 ≈60，success 从 ≈90% 到 ≈93% 再到 ≈96%。正文称第一轮后成功率已超过 90%、第二轮主要提升速度（按图 i=0 已约 90%）。
- Box 每轮 600 条自主加 360 条带纠正。throughput 从 ≈9.9 先降到 ≈8.5（i=1），再升到 ≈13.3。
- 注意：正文写 laundry 迭代后 throughput"提升 50%"、box 第二轮后"2×"（l.566），和按图读数以 i=0 为基准算出的约 1.8× 和约 1.3× 对不上（laundry 若以 i=1 为基准约 1.44×，才接近 50%）。

**Policy extraction 对比**（Fig.11，T-shirt 任务）。所有方法使用与最终模型完全相同的数据，这一点对 baseline 略有利。AWR 和 PPO 都从 π0.6（无 advantage 条件、非 RL 预训练）checkpoint 起训（l.514、l.727）。

| 方法 | throughput | success | 备注 |
|---|---|---|---|
| RECAP | ≈60/h | ≈96% | |
| AWR | ≈30/h | ≈91% | 成功率尚可，但动作慢 |
| PPO | ≈22/h | ≈75% | DPPO/FPO 式单步扩散似然 + SPO 式 trust region（App D，l.704-727）；约束要设到 $\eta=0.01$ 这么小才稳定 |
| offline RL+SFT（参照） | ≈33/h | ≈90% | |

AWR 和 PPO 都没有明显超过 offline RL+SFT。这是全文最接近消融的证据：它说明提升来自 policy extraction 的方式，而不只是来自更多数据。但两者的起点 checkpoint 和 RECAP 不同，所以这组对比同时混入了"是否有 offline RL 预训练"这个因素。

**Failure mode removal**（Fig.12）：标准是领口必须朝上。两轮，每轮 600 条轨迹，最终成功率为 97%（正文数字）。成功率从 i=0 的 ≈23% 到 i=1 的 ≈77% 再到 97%；throughput 从 ≈17 到 ≈21 再到 ≈54/h。

**各任务数据量**（App F，l.759）：

| 任务 | 数据 |
|---|---|
| Diverse laundry | 450 条评测 + 287 条纠正 |
| Failure removal | 约 1000 条自主 + 280+378 条纠正，3 台机器人 |
| Box | 每轮 600 条 + 360 条纠正，3 台机器人 |
| Cafe | 只做 1 轮，429 条纠正 + 414 条自主 |

## 局限与论文没说的

**作者承认的**（Sec VII，l.617；Sec IV-A，l.251）：
1. 不是全自主：奖励标签、人工接管、场景复位都靠人。
2. 探索很朴素，基本靠策略自身的随机性加人类接管，前提是模仿学习得到的初始策略已经不错。
3. 采用批式 iterated offline 更新，不是并发的在线 RL。
4. value 用 MC 估计，不如 off-policy Q 估计。

**我读出来的**：
- **消融严重不足**：
  - 下面这些设计都没有单独消融：distributional value（对比标量回归）、 $N=50$、阈值 $\epsilon_\ell$ 的正样本比例、indicator dropout、CFG 的 $\beta$、纠正片段强制 $I=\text{True}$、每轮从预训练 checkpoint 重训。
  - 缺一个关键 baseline：同样数据上只用成功 episode 做 filtered BC。有了它才能区分提升究竟来自"多了数据"还是来自"advantage 条件化"。
  - 与 AWR/PPO 的对比只在一个任务上做过。
- **理论和实践之间有距离**：
  - 改进保证只是相对 $\pi_{\text{ref}}$ 的一步改进，而且要求 $A^{\pi_{\text{ref}}}$ 估得准。
  - 实际上 $\pi_{\text{ref}}$ 是人类和历代策略的混合，advantage 来自 MC value 加硬阈值，观测也不是 Markov 的。
  - 多轮迭代能否持续改进、而不是停在行为策略附近，只有两轮的证据。
- **数字口径问题**：
  - 没有数值表；评测依赖人工打分；并非每个柱子都给出了评测条数。
  - T-shirt 的评测数据同时又是下一轮的训练数据。
  - 正文、附录和图之间有多处对不上：一是迭代提升的倍数；二是 box 正文写"600 autonomous"，附录写"600 demonstrations"；三是 failure removal 正文称"without any intervention data"，附录却写收集了 280+378 条纠正。
- **没测的设置**：
  - 没有仿真 benchmark，因此无法和 SimpleVLA-RL、π_RL 在同等条件下比较。
  - 迭代实验只在一种静态双臂上做过。
  - specialist 微调后是否遗忘了其他任务，没有报告。
  - value function 在 OOD 状态下是否可靠，没有测。
- **算力与门槛**：GPU/TPU 型号、训练时长、训练步数都没报。预训练数据是数万小时的私有示范。每个任务要几百到上千条真实 episode 加人工标注。chunk 长度 $H$、 $C_{\text{fail}}$、以及一个"step"对应多长时间，也都没给。
- **可能的失败模式**：
  - 阈值按比例设定，意味着总有固定比例的数据被标成 positive。如果整批数据都差，模型仍会把"没那么差"的动作当作目标。
  - 人类纠正的质量参差不齐（作者自己也承认），强制标 positive 可能把次优动作也学进去。
  - 奖励只有稀疏的成功/失败标签，长任务中段的 credit assignment 完全依赖 value 的泛化。

## 在脉络中的位置

- **前驱**：
  - VLA 本体：π0 → π0.5 → π0.6（KI 训练、FAST tokenizer、flow-matching action expert）。
  - 条件化策略 / RvS：Reward-Conditioned Policies、Upside-Down RL、Decision Transformer、RCSL。最直接的来源是 CFGRL（Frans et al. 2025，Diffusion Guidance Is a Controllable Policy Improvement Operator）。
  - offline RL 中的加权回归式 policy extraction：AWR、CRR、[IQL](../iql/notes.md)（l.272）。论文把这一类归为"会丢数据"的方法，拿来作对照。MARWIL 则是和 CFGRL 一起被引作 $p(I\mid A)^\beta$ 形式改进保证的来源（l.214）。value 部分借鉴 distributional RL（Bellemare 2017）。
  - 人类在环：DAgger / HG-DAgger。真实机器人 RL 系统方面引了 SERL，与本项目的 [HIL-SERL](../hil-serl/notes.md) 同属"人类纠正 + 真实世界 RL"路线；区别在于 HIL-SERL 用的是小模型加 off-policy actor-critic。
  - PPO baseline 的实现基于 [PPO](../ppo/notes.md)、[DPPO](../dppo/notes.md)、FPO、SPO。
- **后继 / 影响**：
  - 本项目中的 [SimDist](../../05_intersections/simdist/notes.md)（2026）把 π*0.6 当作"端到端单体策略"的代表，主张用 world model 把决策模块化，以便更快适应新环境。
  - [World Model for Robot Learning survey](../../01_world_model/survey-wm-robot-learning/notes.md) 把它列为 π 系列中 flow-matching 连续动作 VLA 的一员。
- **与本项目其他主题的联系**：
  - **RL 线内部的对照**：[SimpleVLA-RL](../simplevla-rl/notes.md) 和 [π_RL](../pi-rl/notes.md) 走的是"在仿真里做 on-policy PPO/GRPO"的路线；RECAP 走的是"真实世界 + offline + advantage 条件化"的路线。[DeepSeek-R1](../deepseek-r1/notes.md) 这类 LLM RL 依赖可验证奖励和海量 on-policy 采样，这正是真实机器人负担不起的。
  - **World model**：RECAP 完全是 model-free 的，所有经验都来自真实机器人，采样成本是主要瓶颈。[WMPO](../../05_intersections/wmpo/notes.md)、[World-Gymnast](../../05_intersections/world-gymnast/notes.md)、[Ctrl-World](../../05_intersections/ctrl-world/notes.md) 用生成式 world model 替代真实 rollout，来做 VLA 的 policy optimization 或评估。一个自然的问题是：能否把 RECAP 的 value function 和 advantage 条件化用到 world model 的 imagination rollout 上。
  - **TTT / TTA**：作者把 CFG 的 $\beta>1$ 称为"test-time policy improvement"，但它不更新参数，只改变推理时的采样。RECAP 的迭代是部署后的批式重训，不同于 [TTT](../../02_test_time_training/ttt-self-supervised/notes.md) 和 [PAD](../../05_intersections/pad/notes.md) 那种在线自监督适应。[TTRL](../../02_test_time_training/ttrl/notes.md) 同样在没有真值的情况下从自己的 rollout 中学习，可以拿来类比。
  - **Real2sim2real**：RECAP 直接在真实世界（甚至工厂现场）收集经验，从而绕开了 sim-to-real gap，代价是需要人工复位和标注。这与 [RIALTO](../../03_real2sim2real/rialto/notes.md) 那种"先重建 digital twin、再在仿真里做 RL"的做法正好相反。

## 前置知识

1. 优势函数、n-step advantage、Monte Carlo 与 TD 两种 value 估计的区别（[RL overview](../survey-rl-overview/notes.md)）。
2. KL 正则化 RL 与 AWR/REPS 的闭式解 $\pi\propto\pi_{\text{ref}}\exp(A/\beta)$；offline RL 中的 one-step policy improvement（[IQL](../iql/notes.md)）。
3. flow matching / diffusion policy、diffusion ELBO、classifier-free guidance 的"得分相减"形式。
4. distributional value function（离散 bin 加交叉熵）。
5. 条件化策略：reward/return-conditioned policies、Decision Transformer、CFGRL。
6. VLA 架构：π0/π0.5 的 VLM 加 action expert、action chunk、FAST tokenization、Knowledge Insulation。
7. DAgger / HG-DAgger 式的人类在环纠正。
8. PPO 及其在扩散策略上的变体（[PPO](../ppo/notes.md)、[DPPO](../dppo/notes.md)），用来理解 PPO baseline 为什么难做。

## 复现可行性

- **代码**：没有官方的 RECAP / π*0.6 代码或权重。可用的开源底座是 [openpi](https://github.com/Physical-Intelligence/openpi)，提供 π0 / π0-FAST / π0.5 的实现和 checkpoint。
- **算力**：论文没有报告 GPU/TPU 型号、数量或训练时长。唯一相关的信息是：value function 只有 670M，在 VLA 训练中 on-the-fly 推理的额外开销很小（l.395）。
- **数据/环境**：预训练用的是数万小时私有多机器人示范；下游在真实双臂机器人上做，每个任务每轮要数百条 episode，外加人工成功标注和专家接管。完整复现不可行。
- **最小可复现实验建议**（目的是验证机制，不是追论文数字）：
  1. 在仿真里（例如 LIBERO，或 [SIMPLER](../../03_real2sim2real/simpler/notes.md) 的环境）拿 openpi 的 π0/π0.5，或者一个小的 flow-matching policy，先在示范上做 SFT。
  2. 滚出几百条 episode，按 Eq.5 的稀疏奖励打标签，训练一个 201-bin 的 distributional value function（小 ViT 即可）。
  3. 用 n-step advantage 加比例阈值（先试 30% 和 40%）得到 $I_t$，以 30% 的 indicator dropout 做条件训练。
  4. 设置对照组：在全部数据上 SFT、只用成功 episode 的 filtered BC、AWR、同样数据上的 DPPO 式 PPO。报告成功率和平均完成时长（后者对应 throughput）。
  5. 扫描阈值比例和 CFG 的 $\beta\in\lbrace1,1.5,2.5\rbrace$，迭代 2-3 轮，观察是持续改进还是出现漂移。
- **预计难点**：
  - MC value 在数据少时噪声大，阈值附近的标签会来回抖动。
  - 仿真很难复现"人类纠正"和"throughput 提升"这类真实世界现象。
  - 如果 positive 和 negative 条件下模型输出几乎一样，要检查 indicator 在序列中的位置以及 dropout 设置。

## 自测问题

1. 为什么在 $\beta=1$ 时，"以 $I=\text{True}$ 为条件的行为策略"恰好就是正则化改进策略 $\hat\pi$？请从 $\hat\pi\propto\pi_{\text{ref}}\thinspace p(I\mid A)^\beta$ 出发，用 Bayes 公式推出 Eq.2。
2. RECAP 的 value function 估的是哪个策略的 value？论文为什么用 MC 回归而不用 Q-learning？这对"多轮迭代能否超越行为策略"意味着什么？
3. 和 AWR 相比，advantage conditioning 在数据利用上有什么不同？负 advantage 的样本在 Eq.3 中起什么作用？
4. 阈值 $\epsilon_\ell$ 和 CFG 的 $\beta$ 都能锐化策略，作者为什么主要用 $\epsilon_\ell$？大 $\beta$ 有什么风险？为什么 CFG 对自回归部分不起作用？
5. 奖励只有 $r_t\in\lbrace-1,0,-C_{\text{fail}}\rbrace$ 三种取值。这样一个由稀疏成功标签导出的奖励，为什么能同时推高成功率和速度？在 T-shirt 任务里，为什么要把 positive 比例压到约 10%？
