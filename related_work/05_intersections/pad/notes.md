# Self-Supervised Policy Adaptation during Deployment

> arXiv [2007.04309](https://arxiv.org/abs/2007.04309) · ICLR 2021（Spotlight；arXiv 初版 2020） · 主题：05_intersections · 层级：foundational
> 代码：https://github.com/nicklashansen/policy-adaptation-during-deployment（论文只给了项目页 https://nicklashansen.github.io/PAD/，见 `source/abstract.tex` 脚注；仓库已确认存在）· 笔记依据：`source/*.tex`（数字/公式）+ `paper.pdf`（图）

## 一句话

PAD（Policy Adaptation during Deployment）把 [TTT](../../02_test_time_training/ttt-self-supervised/notes.md) 搬到了 vision-based RL 上。训练时，RL 目标和一个 self-supervised auxiliary task（控制任务用 inverse dynamics，导航任务用 rotation prediction）共享一个 feature extractor。部署时拿不到 reward，于是只用自监督损失，每个时间步对共享 encoder 在线做一步梯度更新。policy head 不动，policy 借此适应没见过的视觉变化和动力学变化。

## 问题设定

- **输入/输出**：像素观测 $\mathbf s$。DMControl 和 CRLMaze 上是 $k=3$ 帧堆叠，机器人任务上 $k=1$；每帧渲染成 $100\times100$，再 crop 到 $84\times84$（`source/appendix.tex` §F）。输出是连续动作（SAC）或离散动作（A2C）。评测指标是测试环境里的 episodic return 或 success rate。
- **假设**：训练环境有 reward。测试环境的变化事先未知，包括颜色、视频背景、干扰物、纹理/光照，以及真机上的桌布、disco 灯光和动力学差异。测试时**没有 reward**，也不允许预先收集目标域数据，不做 reset，不做 pre-training（§F "without pre-training or resetting"）。唯一能免费拿到的监督，是 agent 自己产生的 transition $(\mathbf s_t,\mathbf a_t,\mathbf s_{t+1})$。
- **之前的方法卡在哪**（1–2 见 `source/introduction.tex`，3–4 见 `source/related-works.tex`）：
  1. [domain randomization](../../03_real2sim2real/domain-randomization/notes.md) 要求人提前预料变化。随机化太多，模型容量和样本复杂度跟不上；随机化太少，又覆盖不到真实变化，反而可能有害。作者说这两种现象在实验里都出现了。
  2. 用 reward fine-tune 需要在部署环境里设计 reward。
  3. domain adaptation 需要目标域数据。
  4. 原版 TTT 只做了图像分类。

## 核心方法

**1. 网络切分**（`source/method.tex` §3.1，Fig.1）。policy 参数 $\theta=(\theta_e,\theta_a)$，$\pi(\mathbf s;\theta)=\pi_a(\pi_e(\mathbf s))$，其中 $\pi_e$ 是 feature extractor，$\pi_a$ 是基于这些特征输出动作分布的 controller。自监督 head $\pi_s(\theta_s)$ 接在 $\pi_e$ 的输出上，整体呈 Y 形。DMControl 和机器人任务中，$\pi_e$ 是 8 层卷积；CRLMaze 中是 6 层；所有卷积层都是 32 个 filter（`appendix.tex` §F）。两个 head 结构相同，都是 3 层卷积加若干全连接层。全连接层的数量正文写 4 层（`experiments.tex` "Network details"），附录写 3 层（`appendix.tex` §F），前后不一致。

**2. 自监督任务**。inverse dynamics model（IDM）在特征空间里预测两帧之间的动作（`method.tex` Eq.1）：

$$L(\theta_s,\theta_e)=\ell\big(\mathbf a_t,\ \pi_s(\pi_e(\mathbf s_t),\pi_e(\mathbf s_{t+1}))\big)$$

连续动作时 $\ell$ 取 MSE，离散动作时取 cross-entropy。**为什么用 inverse 而不用 forward**（脚注 2）：forward dynamics 在特征空间里有平凡解，把所有状态映射到常向量 $\mathbf 0$ 就能让 loss 为 0，也就是 representation collapse；放到像素空间做 forward 预测又太难。inverse 不存在这种平凡解。备选任务是 rotation prediction：把图像转 0/90/180/270°，做 4 分类。RL 场景的特殊之处在于，IDM 的标签 $\mathbf a_t$ 是 agent 自己选的动作，部署时同样是免费的。

**3. 训练：multi-task joint training**（`method.tex` §3.3，未编号公式）

$$\min_{\theta_a,\theta_s,\theta_e}\ J(\theta_a,\theta_e)+\alpha L(\theta_s,\theta_e),\quad \alpha>0$$

$J$ 是任意 RL 目标，off-policy 或 on-policy 都可以；DMControl 和机器人任务用 SAC，CRLMaze 用 A2C。CRLMaze 的损失系数在训练时为 0.5，测试时为 1（Table 12）。DMControl 的 Table 11 里有一个 $\alpha$ 的 optimizer/lr，但那应是 SAC 的 temperature：论文没有明说，官方代码里与之对应的是 `--init_temperature`、`--alpha_lr 1e-4`、`--alpha_beta 0.5`，数值与 Table 11 一致。DMControl 上 trade-off 系数 $\alpha$ 的具体取值论文没有写。

**4. 部署：在线自监督更新**（`method.tex` Eq.2–5）。对 $t=1\dots T$：

$$\mathbf s_t\sim p(\mathbf s_t\mid\mathbf a_{t-1},\mathbf s_{t-1})$$
$$\theta_s(t)=\theta_s(t-1)-\nabla_{\theta_s}L(\mathbf s_t;\theta_s(t-1),\theta_e(t-1))$$
$$\theta_e(t)=\theta_e(t-1)-\nabla_{\theta_e}L(\mathbf s_t;\theta_s(t-1),\theta_e(t-1))$$
$$\mathbf a_t=\pi(\mathbf s_t;\theta(t)),\quad \theta(t)=(\theta_e(t),\theta_a)$$

要点：
- $\theta_a$ 始终冻结，因为没有 $J$ 可以优化。
- 参数不在每步之后重置，会一直累积到 episode 结束，属于 TTT-Online 式的做法。
- IDM 需要相邻两帧，所以 Eq.3–4 里的 $L(\mathbf s_t;\cdot)$ 实际用的应该是最近一步 transition $(\mathbf s_{t-1},\mathbf a_{t-1},\mathbf s_t)$。这是我根据 IDM 的定义推出来的，论文只写了 "most-recent observation"。
- 实现细节（§F）：没有 memory，只用最新观测；每步只做 1 次梯度更新；batch 由最新观测的 32 个 random crop 副本组成，这类增强同时起正则作用。IDM 的 $\mathbf s_t$ 和 $\mathbf s_{t+1}$ 各自用不同的 crop。执行动作时输入用 center crop。CRLMaze 的测试期学习率是 1e-5，训练期是 1e-4（Table 12）。DMControl 的测试期学习率论文没有单独列出。
- 测试时冻结 $\pi_s$、只更新 $\pi_e$，效果差别可以忽略（Table 9）。默认两者一起更新，因为反正要算梯度。

**5. 为什么这样能帮到 RL**。在白板上可以这么推：$\pi_a$ 冻结，所以只有在"最小化 $L$ 让 $\pi_e$ 往回调"的同时 $J$ 也下降时，这一步才有用。作者自己的说法是，辅助任务的"features (and consequently gradients) need to be sufficiently correlated with the primary RL task"（§4.2）。这正是 TTT 论文里 gradient correlation 条件 $\langle\nabla l_m,\nabla l_s\rangle>0$ 在 RL 中的版本。IDM 直接把观测和动作连起来，所以更贴近 motor control 的需要；rotation 只看观测，更贴近 scene understanding。

## 关键实验与证据

总账：在 36 个测试环境里，PAD 在 31 个上提升了泛化（`abstract.tex`）；其中 DMControl 占 19/22（`introduction.tex`）。我逐表核对了与"不加 PAD 的同一模型"的比较，结果如下：颜色 9/9，视频 7/8，干扰物 3/5，CRLMaze 4/4，真机 5/6，仿真动力学 3/4，合计正好 31/36。

- **DMControl 随机颜色**（Table 1）：9 个任务，每个跑 10 个 seed、100 个随机初始化。对比对象有 SAC、+DR（在 100 种颜色上随机化，与测试同分布但方差更小；作者说直接在测试分布上训练不收敛）、+IDM（joint training，但部署时不做 PAD）。
  - PAD 在 9/9 个任务上优于 +IDM，例如 Walker stand 从 743±37 升到 797±46，Ball in cup 从 471±75 升到 563±50，Cheetah run 从 121±38 升到 159±28。
  - 但 DR 在 Walker walk（594 对 468）、Cartpole swingup（647 对 630）和 Cartpole balance（867 对 848）上仍然最好。
  - 把 episode 拉长到 10 倍，PAD 在 6/7 个任务上仍有提升，例如 Walker walk 从 3830 升到 5505。唯一的例外是 Cartpole balance，从 7746 降到 7670。作者据此认为 embedding 没有漂移失控。
  - Fig.3（Walker walk/stand、Cartpole swingup/balance 四个任务）显示，单个 episode 内的相对提升主要在前几百步随步数上升，之后大致持平，说明收益来自在线累积。
- **视频背景 / 干扰物**（Table 2）：视频背景下 7/8 个任务最好，Finger spin 上 PAD 691、DR 338，这就是正文 §4.1 里的 "104%"（摘要里没有这个数）。干扰物下 3/5 个任务最好，Ball in cup 上 SAC 177、PAD 545，对应 "208%"。在两个 cartpole 任务上，vanilla SAC 最好。
- **消融**（Table 3）：
  - CURL (PAD) 在多数任务上优于 CURL。
  - Rot (PAD) 只在 4/9 个任务上优于 Rot。
  - IDM 整体最稳。
  - **offline PAD（O-PAD）**每步更新后就丢弃参数，只能适应单张观测。它明显弱于 online 版，但在 Finger turn_easy（321 对 304）和 Reacher easy（241 对 214）上反而更好。
- **与带 reward 的 fine-tune 对比**（Table 10，3 个任务；fine-tune 用部署前在目标域收集的 1/10/100 个 episode，同时更新 $\pi_e$ 和 $\pi_a$）：PAD 的水平落在 1 到 100 个 episode 的 fine-tune 之间，因任务而异。Walker stand 上 PAD 797，高于 100 个 episode 的 784；Cartpole swingup 上 PAD 630，介于 10 个（605）和 100 个（650）之间；Walker walk 上 PAD 468，介于 1 个（395）和 10 个（489）之间。
- **训练环境本身**（Table 7）：PAD 在 9/9 个任务上都略低于 +IDM，例如 Reacher 从 390 降到 365，Walker walk 从 911 降到 895。作者称之为 "hurts minimally"。DMControl 训练 500k frames（§B）。
- **CRLMaze**（Table 4，10 个 seed，20 个起点，报告 std. error）：Rot (PAD) 在 4 个环境上全部优于 Rot，例如 Lights 从 −84 升到 312，Walls 从 −206 升到 −74。**反例同样关键**：IDM (PAD) 比 IDM 差很多，Floor 从 −47 跌到 −530，Ceiling 从 166 跌到 −508；在训练环境里也从 585 跌到 −416（Table 8）。
- **真机 Kinova Gen3**（Table 5，每个设置 25 次试验，覆盖 5 个目标位置，相机未标定）：Push cloth 从 64% 升到 88%，Push disco 从 72% 升到 84%，Reach cloth 从 56% 升到 80%，与 DR 持平。Reach default 上所有方法都是 100%。正文说的 "24%" 是 Push cloth 的百分点差。
- **仿真动力学变化，push 任务**（Table 6，50 种配置）：PAD 在 all 设置上从 48% 升到 76%，object 设置上从 72% 升到 82%；mount 设置上从 86% 降到 84%，略有下降。

## 局限与论文没说的

**作者承认的**：
1. 选哪个自监督任务依赖先验（`conclusion.tex`），作者把自动选择留作未来工作（§4.2）。
2. 没有显式机制阻止 embedding 偏离 RL 任务（§4.1）。
3. IDM 在连续动作下最有效；在小的离散动作空间里，作者猜测 inverse dynamics 预测"不够有挑战性"，效果不如连续动作（§3.2）。

**我读出来的**：
1. **选错辅助任务会造成灾难性后果，而且没有检测机制**。CRLMaze 上 IDM (PAD) 连训练环境都从 585 掉到 −416，这和 DMControl 上 "hurts minimally" 的叙述相矛盾。部署时没有 reward，你无法知道自己正在变差。
2. **提升幅度常常落在方差以内**。例如 Cartpole balance 从 835±40 到 848±29，CRLMaze（Table 4）上非 Random 方法的 std. error 在 ±53 到 ±270 之间，24 个格子里有 19 个超过 ±100。真机每格只有 25 次，一次成功就是 4 个百分点，也没有给置信区间。
3. **DR baseline 很窄**。DMControl 上它只随机化了 100 种颜色；视频背景下 DR 输给 PAD，有一部分是训练分布根本没覆盖视频造成的，作者也承认这一点。真机 DR 只说"as in Section 4.1"，具体随机化了什么没有写清。CRLMaze 的 DR 用了 56 种纹理组合，但 DMControl 上没有和 texture/video 级别的 DR 比，也没有和同期其他 TTA 方法比（如 [Tent](../../02_test_time_training/tent/notes.md) 或 BN-adapt）。
4. **动力学适应的机制没有分析**。只更新 $\pi_e$，却能改善 push 在质量、摩擦变化下的表现，这件事很有意思，但论文只给了成功率，没有解释 encoder 的变化是怎样补偿动力学差异的。动力学实验也只做了 push 一个任务。
5. **算力和延迟完全没有报告**。训练时长、GPU 型号都没写；真机上每个控制步都要对 8 层卷积加 3 层卷积的网络做一次 batch=32 的反传，控制频率和是否满足实时要求也没写。
6. 长时部署只测到 10 倍 episode 长度，没有测跨 episode 的持续漂移，也没有测 reward 或目标本身发生变化的情形。SAC 执行的动作（也就是 IDM 的标签）是采样值还是均值，论文没有交代。

## 在脉络中的位置

- **前驱**：
  - [TTT (Sun et al.)](../../02_test_time_training/ttt-self-supervised/notes.md)：Y 形结构、joint training、TTT-Online 和 rotation prediction 基本原样继承。
  - [SAC](../../04_rl/sac/notes.md)，以及 SAC-AE（Yarats et al. 2019）的架构和超参数。
  - CURL 的 random crop。
  - 在 RL 里用 inverse dynamics 做自监督的 curiosity/ICM（Pathak et al. 2017）。
  - 主要对手是 [domain randomization](../../03_real2sim2real/domain-randomization/notes.md)。
- **后继 / 影响**：
  - [TTT-Layers](../../02_test_time_training/ttt-layers/notes.md) 把 PAD 列为 TTT 在机器人操作上的后续工作，并指出这类方法的自监督任务是人工设计的，而 TTT-Layers 改为在 outer loop 里学出自监督任务。
  - [TTT-E2E](../../02_test_time_training/ttt-e2e/notes.md) 引用了 PAD "no reset 带来大得多的提升"这一发现，对应本文的 online 与 O-PAD 对比。
  - [Survey on Test-Time Intelligence](../../02_test_time_training/survey-test-time-intelligence/notes.md) 把 PAD 归入 environment-interaction feedback 中的自监督信号，以及 direct policy adaptation。
  - [Titans](../../02_test_time_training/titans/notes.md) 的笔记拿 PAD 当部署期梯度更新的对照：两者都在测试时用梯度写参数，区别在于 Titans 的写入规则是在 outer loop 中端到端学出来的，而 PAD 的自监督任务是人工选定的。[LaCT](../../02_test_time_training/lact/notes.md) 的笔记则把 PAD 的部署期自监督更新列为思路相近的做法。
- **与本项目其他主题的联系**：
  - *real2sim2real*：PAD 是"不修 simulator，直接在部署时修 policy"的路线。与之对照：[SimOpt](../../03_real2sim2real/simopt/notes.md) 用真实轨迹去调仿真参数；[RMA](../../03_real2sim2real/rma/notes.md) 在部署时不做梯度更新，而是用在仿真里监督训练的 adaptation module 从历史中估计 extrinsics。
  - *world model*：脚注 2 讲的 forward 模型坍缩问题，正是 latent world model 必须处理的。[TD-MPC2](../../01_world_model/tdmpc2/notes.md) 和 [V-JEPA 2](../../01_world_model/vjepa2/notes.md) 用 stop-grad/EMA target 防坍缩，[Dreamer](../../01_world_model/dreamer/notes.md) 系列用像素重建。这意味着 world model 自带一个可在测试时计算的自监督 loss（预测误差），可以直接充当 PAD 的 $L$。[SimDist](../simdist/notes.md)（仿真预训练 world model、真机只 finetune dynamics）和 [DayDreamer](../daydreamer/notes.md)（真机在线学 world model，但需要 reward）是这条线的延续。
  - *RL*：PAD 的 IDM 标签来自 agent 自己的动作。[TTRL](../../02_test_time_training/ttrl/notes.md) 则用多数投票当伪奖励；两者都是在"部署时没有 reward"条件下的替代信号。

## 前置知识

1. vision-based RL 的标准配方：[SAC](../../04_rl/sac/notes.md)、共享卷积 encoder、frame stacking、action repeat、random crop 数据增强（CURL/RAD）。
2. [TTT](../../02_test_time_training/ttt-self-supervised/notes.md)：joint training、测试时的自监督更新、TTT-Online，以及梯度相关性条件。
3. 自监督辅助任务：inverse dynamics 与 forward dynamics 的区别、representation collapse、rotation prediction。
4. sim-to-real gap 与 [domain randomization](../../03_real2sim2real/domain-randomization/notes.md)。
5. A2C 和 DMControl 基准（action repeat、episode 长度 1000）。

## 复现可行性

- **代码**：官方 PyTorch 实现在 https://github.com/nicklashansen/policy-adaptation-during-deployment 。按 README，依赖 dm_control/MuJoCo（通过 dmc2gym）、SAC、CUDA ≥ 9.2；用 `setup/conda.yml` 和 `setup/install_envs.sh` 安装；提供 `scripts/train.sh` 和 `scripts/eval.sh`；评测模式里有 `color_hard` 等。README 只展示了 DMControl，我没有在里面看到 CRLMaze 和真机相关代码。
- **算力**：论文**完全没有报告** GPU 型号、数量和时长。训练量是每个任务 500k 步、replay buffer 500k、batch 128（Table 11）。注意 §B 写的是 "500,000 frames"，Table 11 写的是 "Number of training steps 500,000"，有 action repeat 时两者不等价，论文没有说清。完整复现 Table 1 需要 9 个任务 × 10 个 seed × 3 种训练变体（SAC/+DR/+IDM），开销主要在训练；PAD 本身只在评测时增加每步一次、batch=32 的反传。
- **最小可复现实验**：选 Cartpole swingup（action repeat 8），训练 SAC+IDM 500k 步，至少 3 个 seed。然后在随机颜色环境上比较四种设置：不做 PAD、PAD、O-PAD、冻结 $\pi_s$ 的 PAD。论文里对应的数字是 585±73、630±63、578±69、623±57（Table 1/3/9）。再加一组训练环境评测，检验"hurts minimally"（论文是 849 对 845，Table 7）。
- **难点**：
  1. 2020 年的 dm_control/MuJoCo 版本和 dmc2gym 与现在的环境不兼容。
  2. 视频背景要另外准备视频数据。
  3. DMControl 的测试期学习率和 trade-off 系数 $\alpha$ 论文里都没写，只能去代码里查。
  4. 效果差距常在一个 std 以内，seed 太少就得不出结论。
  5. 真机部分无法复现。能替代的是 Table 6 的仿真动力学变化，但 README 里也没有这个仿真 push 环境，需要自己搭。

## 自测问题

1. 为什么 PAD 选 inverse dynamics 而不是特征空间里的 forward dynamics？forward 版本的平凡解是什么？TD-MPC2 或 V-JEPA 2 这类 latent world model 又靠什么避免同样的坍缩？
2. 部署时哪些参数被更新，哪些冻结？为什么 $\theta_a$ 必须冻结？既然 $\pi_a$ 不变，更新 $\pi_e$ 为什么能提高 return？请用 TTT 的 gradient correlation 条件解释。
3. IDM 的标签在部署时从哪里来？为什么这在 RL 里是"免费"的，而在图像分类的 TTT 里没有对应物？
4. CRLMaze 上 Rot (PAD) 一律提升，IDM (PAD) 却连训练环境都崩掉（Table 4/8）。结合离散动作空间和任务性质（scene understanding 与 motor control 的差别）解释原因，并设想一个无 reward 条件下能提前发现"PAD 在变差"的信号。
5. online PAD 和 O-PAD 的区别是什么？为什么观测相关性强反而对 online 有利？这和 TTT-Online、TTT-E2E 里 "no reset" 的结论是什么关系？
