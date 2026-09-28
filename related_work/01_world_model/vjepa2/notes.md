# V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning

> arXiv [2506.09985](https://arxiv.org/abs/2506.09985) · arXiv preprint（FAIR at Meta 技术报告）, 2025 · 主题：01_world_model · 层级：frontier
> 代码：https://github.com/facebookresearch/vjepa2 · 笔记依据：`source/main.tex`（数字/公式）+ `paper.pdf`（图）

## 一句话

两阶段训练。先在 >1M 小时互联网视频上做无动作的 JEPA 式 mask-denoising 预训练，预测发生在表征空间而不是像素空间，得到 V-JEPA 2 编码器；然后冻结编码器，只用 <62 小时无奖励的 Droid 机器人视频后训练一个动作条件 predictor（V-JEPA 2-AC）。部署时在 latent 空间用 CEM 做 MPC，最小化"想象的未来表征"与"目标图像表征"之间的 L1 距离。这样在两个未出现在训练数据里的实验室，Franka 机械臂 zero-shot 完成了抓取和拾放。

## 问题设定

- **目标**：从大量"只有观测、没有动作"的视频，加上少量交互数据，学一个同时能做 understanding、prediction、planning 的 world model（`main.tex` Sec.1）。
- **Stage 1 输入/输出**：视频 clip（图像复制成 16 帧当视频用），输出每个 tubelet token 的表征。没有任何标签。
- **Stage 2 输入**：Droid 左侧 exocentric 相机的单目视频，256×256、4 fps、4 秒一段，即 16 帧 $(x_k)$。另有 7D 末端状态 $s_k$：3 维位置、3 维 extrinsic Euler 角、1 维 gripper。动作取相邻帧的状态差 $a_k$（7D）。**不用** reward、任务标签，也不用"是否成功"的标注，成功和失败轨迹都用上了（Sec.3.1）。
- **部署假设**：目标用 goal image 给出（pick-and-place 另外人工提供 2 张 sub-goal 图）。不用 reward，也不采部署环境的任何数据。相机未标定，位置固定。控制方式是末端笛卡尔 delta 加 blocking control（Sec.4.1）。
- **前人卡在哪**：
  1. 从交互数据学 world model 的一类（Sec.1 引的是 [World Models](../world-models/notes.md)、PlaNet、TD-MPC；Sec.8 另列了 [Dreamer](../dreamer/notes.md) 等）受限于真实交互数据的规模，常依赖 reward（Sec.1）；其中做过真机的工作都是 task-specific 的，要用部署环境里采的交互数据（Sec.8）；
  2. 用互联网视频训练的动作条件视频生成（[Genie](../genie/notes.md)、[Cosmos](../cosmos-wfm/notes.md)、[UniSim](../../05_intersections/unisim/notes.md)）主要评估生成质量，逐像素生成让规划很贵，几乎没有闭环控制真机的结果；
  3. VLA 靠 behavior cloning，需要高质量的成功示教，用不上失败数据（Sec.8 Related Work）。

## 核心方法

**Stage 1：V-JEPA 2 预训练（mask-denoising in representation space）**（`source/main.tex` Eq.1）

$$\min_{\theta,\phi,\Delta_y}\ \big\| P_\phi(\Delta_y, E_\theta(x)) - \mathrm{sg}\big(E_{\bar\theta}(y)\big)\big\|_1$$

- 整段视频记作 $y$，按 multiblock masking 丢掉一部分 token 后得到 $x$。编码器 $E_\theta$ 只处理可见 token。predictor $P_\phi$ 在被丢位置放入可学习的 mask token $\Delta_y$，预测这些位置的表征。target 由 EMA 编码器 $E_{\bar\theta}$ 加 stop-gradient 给出，loss 只算 masked patch。防止 representation collapse 靠 EMA 加 sg，既不用负样本，也不做像素重建。
- **为什么不预测像素**：JEPA 只需要预测"可预测的部分"（比如物体运动轨迹），可以忽略不可预测的细节（比如每片草叶的位置）（Sec.1）。
- **架构**：ViT，tubelet 为 2×16×16。位置编码用 3D-RoPE：把特征维分成 T/H/W 三段，每段各做 1D RoPE。作者发现它能让最大的模型训练更稳（Sec.2.1）。编码器有 ViT-L 300M、ViT-H 600M、ViT-g 1B（width 1408、depth 40）三档；predictor 固定为 ViT-s 22M（Table 12）。
- **四个 scaling 要素**（Fig.3，以 VM2M 上训练的 ViT-L/16 为基线，看 6 个任务的平均精度）：
  - 数据从 2M 增加到 22M 视频：+1.0；
  - 模型从 300M 增加到 1B：+1.5；
  - 训练从 90K 延长到 252K iter：+0.8；
  - 分辨率 256→384、帧数 16→64：平均精度到 88.2，累计 +4.0。
- **数据 VM22M**（Table 1，样本数 / 采样权重）：SSv2 168K/0.056，Kinetics 733K/0.188，HowTo100M 1.1M/0.318，YT-Temporal-1B 19M/0.188（经过 curation），ImageNet 1M/0.25。
- **YT1B 的 curation**：用 PySceneDetect 切 scene，丢掉短于 4 s 的后剩 316M 个；取每个 scene 中间帧的 DINOv2 ViT-L embedding，聚成 1.5M 个 cluster；只保留被 K710/SSv2/COIN/EpicKitchen 检索命中的约 210k 个 cluster，共 115M 个 scene（App.A.2，Table 11）。curation 带来 +1.4（Fig.4 右）。
- **Progressive resolution 训练**：学习率先 warmup 12K iter，再 constant 228K iter，都用 16 帧、256²；最后 cooldown 12K iter，学习率线性衰减，同时换成 64 帧和更高分辨率。这样高分辨率长 clip 的开销只花在 cooldown 阶段。
  - 其他超参：global batch 3072，EMA 固定 0.99925，weight decay 0.04（Table 9）。
  - 若 ViT-g 全程用 64×384×384 训练，约需 60 A100 GPU-years；正文称 progressive 方案省 8.4× GPU 时间（Sec.2.4），Fig.5 图注写的是 "up to 8×"。但 Fig.5 中图 64 帧柱子读图约为 0.72×10⁴ vs 2.2×10⁴ GPU-days，只差约 3×，与正文的 8.4× 对不上（论文内部不一致）。

**Stage 2：V-JEPA 2-AC（动作条件 latent world model）**

- 冻结 ViT-g 编码器，每帧**单独**编码：$z_k=E(x_k)\in\mathbb{R}^{16\times16\times1408}$。
- predictor 约 300M，是 24 层、16 头、hidden 1024 的 transformer。$a$、$s$、$z$ 各自经过一个 affine 层映射进来。patch token 用 3D-RoPE，action/pose token 只加 temporal RoPE。
- 输入按时间交错排成 $(a_k,s_k,z_k)$，用 **block-causal attention**：第 $k$ 步的 patch token 能看到同一步和之前各步的 action、state、patch（Sec.3.1 Architecture）。
- **Teacher-forcing loss**（Eq.2，$T=15$）：

$$\mathcal{L}_{\text{tf}}(\phi)=\frac{1}{T}\sum_{k=1}^{T}\Big\|P_\phi\big((a_t,s_t,E(x_t))_{t\le k}\big)-E(x_{k+1})\Big\|_1$$

- **Rollout loss**（Eq.3）：把 predictor 的输出回填为下一步输入，自回归 rollout 之后再和真实表征对齐。实际取 $T=2$，也就是只对一步 recurrent 反传：

$$\mathcal{L}_{\text{rollout}}(\phi)=\big\|P_\phi(a_{1:T},s_1,z_1)-z_{T+1}\big\|_1,\qquad L(\phi)=\mathcal{L}_{\text{tf}}+\mathcal{L}_{\text{rollout}}\ \text{(Eq.4)}$$

  编码器冻结、只训 predictor，所以这一阶段不需要 EMA target。
- **训练超参**（App.B.1）：AdamW，weight decay 0.04，batch 256。学习率从 7.5e-5 warmup 到 4.25e-4（4500 iter），保持 85500 iter，再用 4500 iter 衰减到 0。只用左视角；左右视角混训而不加相机条件时性能会下降。

**推理：goal-conditioned energy 加 CEM-MPC**（Eq.5）

$$\mathcal{E}(\hat a_{1:T};z_k,s_k,z_g)=\big\|P(\hat a_{1:T};s_k,z_k)-z_g\big\|_1,\qquad a^\star_{1:T}=\arg\min_{\hat a_{1:T}}\mathcal{E}$$

- CEM 的做法：
  1. 每个 horizon 步的动作坐标从 $\mathcal{N}(0,I)$ 初始化的高斯里采样；
  2. 用 top-k 样本的统计量更新均值和方差，重复若干轮；
  3. 取最后的均值序列，只执行第一个动作，然后重新规划（receding horizon）（Fig.7）。
- 实际配置（App.B.2）：每轮 **800 个样本、10 轮迭代、top-10，planning horizon $T=1$**。动作限制在半径 0.075 的 L1-ball 内（单步末端位移约 13 cm），在单张 RTX 4090 上**每个动作约 16 s**（Table 3）。
- Pick-and-place 按固定步数切换目标：前 4 步对准 sub-goal 1（抓住物体），接下来 10 步对准 sub-goal 2（物体到目标附近），最后 4 步对准 final goal（Sec.4.2）。

## 关键实验与证据

**真机规划**（Sec.4）

- **对比 Octo**（Table 2）：每个任务在每个 lab 做 10 次 trial，下面是两个 lab 的平均。Octo 由作者从 octo-base-1.5 出发，在**全部** Droid 上用 hindsight relabeling 做 BC fine-tune。V-JEPA 2-AC 只用了 23k 条轨迹（脚注 1）。

  | 任务 | V-JEPA 2-AC | Octo |
  |---|---|---|
  | Reach | 100% | 100% |
  | Grasp Cup / Box | 65% / 25% | 15% / 0% |
  | Reach w/ Obj Cup / Box | 75% / 75% | 15% / 70% |
  | Pick-&-Place Cup / Box | 80% / 65% | 15% / 10% |

- **对比 Cosmos**（Table 3，只在 Lab 2 测）：Cosmos 是 latent diffusion-7B，在 Droid 上做了动作条件 fine-tune，作者额外降低了学习率、去掉 video conditioning 里的 dropout、把 noise level 调大 $e^2$ 倍。

  | 模型 | CEM 样本数 | 每个动作耗时 | Reach | Grasp Cup / Box | P&P Cup / Box |
  |---|---|---|---|---|---|
  | Cosmos | 80 | 4 min | 80% | 0% / 20% | 0% / 0% |
  | V-JEPA 2-AC | 800 | 16 s | 100% | 60% / 20% | 80% / 50% |

- **Single-goal reaching**（Fig.8）：Δx、Δy、Δz 三个任务的末端误差都单调下降，最终都在 4 cm 以内。作者把这看作一种 visual servoing。
- **Energy landscape**（Fig.9）：真值动作在 $(0,-0.1)$，能量最小值落在约 $(0,-0.05)$。曲面平滑，局部近似凸。
- **可视化**（App.B.3，Fig.15）：另训了一个 ViT-L 帧 decoder。解码 rollout 后看到：机械臂动了，背景没变；最后一帧里杯子位置略偏低，说明有误差累积。gripper 张开时杯子留在原地，闭合时杯子随臂移动。

**编码器质量**（它决定了 AC 能用到哪些状态信息）

- **分类**（Table 4，attentive probe）：
  - ViT-g 平均 87.5，ViT-g$_{384}$ 平均 88.2。
  - SSv2 达到 77.3；同协议下 InternVideo2$_{s2}$-1B 69.7，PE$_{core}$G 55.4，DINOv2 50.7。
  - 外观类任务只是有竞争力，不是最好：IN1K 85.1，SigLIP2 为 88.0。
- **EK100 action anticipation**（Table 5）：Recall@5 从 ViT-L 32.7 升到 ViT-g 38.0，ViT-g$_{384}$ 为 39.7。相比 PlausiVL 的 27.6 高 12.1，相对提升 44%。
- **关键消融**（Table 20）：只用 encoder 39.1，只用 predictor 20.2，两者一起 39.7。作者的结论是 EK100 主要考语义理解，而不是 forecasting。
- **VidQA**：
  - Table 6：冻结编码器、Qwen2-7B、18M 对齐数据，平均 52.3，高于 PE 49.1、SigLIP2 48.1、DINOv2 45.7。
  - Table 8：Llama 3.1 8B、88.5M 对齐数据，PerceptionTest 84.0、TempCompass 76.9、TOMATO 40.3；但 TVBench 60.6、MVBench 73.5 不如 PLM 8B（63.5、77.1）。

**预训练消融**

- 不做 curation 的数据在约 600 epoch（1 epoch = 300 step）之后不再提升（Fig.12）。
- cooldown 带来的收益（Table 14，ViT-g）：IN1K 83.8→84.6，K400 85.8→86.6。
- 评测时帧数从 16 增加到 64，平均 +9.7（Fig.13，single-clip 协议）。

## 局限与论文没说的

**作者承认的**（Sec.4.3、Sec.6 Limitations、Sec.9 Future work）

1. **对相机位置敏感**。模型要从单目图像里隐式推断动作坐标轴。App.B.4 的扫描显示，推断坐标轴的旋转误差几乎是相机角度的线性函数（Fig.16 读图约 5°–33°），平均绝对误差约 1.6 cm，而真值位移约 5 cm。实验用的相机位置是人工试出来的。
2. **长 horizon 规划受限**。自回归会累积误差，搜索空间随 horizon 指数增长，所以 pick-and-place 需要 sub-goal 图。
3. **只接受 image goal**，不支持语言目标。
4. **规模只到 1B**（Sec.9）。另外 EK100 上预测时距拉长后性能明显下降（Sec.6 Limitations，Fig.18）。

**我读出来的**

- **主结果用的是 $T=1$**。实际每步是一步贪心：在"单步动作后的预测表征"与目标表征之间做能量最小化，本质接近学出来的 visual servoing。多步 rollout 能力在真机实验里几乎没有被用到。pick-and-place 的任务分解（3 张目标图，4/10/4 的步数分配）是人给的。
- **AC 阶段没有定量消融**。rollout loss、编码器选择（比如换成 DINOv2 或 SigLIP 作冻结编码器）、Droid 数据量、horizon、CEM 超参对成功率的影响都没有给出数字（只定性提到：更长 horizon 也能用但更慢，App.B.2；左右视角混训且不加相机条件会变差，App.B.1），所以"真机成功来自 V-JEPA 2 表征"这个论断缺少直接对照。
- **评测口径偏弱**：
  - 每个格子只有 10 次 trial，没有置信区间；
  - Cosmos 只在 Lab 2 测，样本数比 V-JEPA 2-AC 少 10 倍，不是等算力对比；
  - Octo 和 Cosmos 两个 baseline 都是作者自己 fine-tune 的；
  - 任务只有杯子、盒子两种物体的桌面操作，没有可变形物体、接触丰富或动态任务，也没有其他机器人本体。
- **速度**：每个动作 16 s 且用 blocking control，离实时闭环还很远。
- **表征能力的上限**：predictor 是确定性的 L1 回归，不建模多模态的未来。能量用的是整张 16×16×1408 特征图的 L1 距离，目标图必须在同一场景、同一视角下拍摄，对背景变化和干扰物可能敏感（论文没测）。编码器冻结，所以 gripper 开合这类细粒度信息如果没编码进去，AC 也补不回来；Box 抓取只有 25%，可能与此有关（这是推测）。
- **算力**：论文正文没给最终 ViT-g 预训练的总 GPU 数和时长。Fig.5 中图的 progressive（252K iter 16 帧/256² + 12K cooldown 64 帧/384²）柱子读图约 0.72×10⁴ A100 GPU-days，约 20 GPU-years；若按正文 60 GPU-years ÷ 8.4 则只有约 7 GPU-years。两者差近 3 倍，是论文内部不一致，这里都只是读图/推算，不是原文数字。VidQA 阶段用了 128 或 512 张 H100（App.E）。AC 后训练的算力没有报告。
- **rollout 时 $s_{k+1}$ 怎么得到**，正文没写（猜测是 $s_k+a_k$，需要查代码确认）。

## 在脉络中的位置

- **前驱**：
  - JEPA 路线：LeCun 2022 "A path towards autonomous machine intelligence"、I-JEPA（Assran 2023）、V-JEPA（Bardes 2024）；
  - latent world model 加规划：[World Models](../world-models/notes.md)、[Dreamer](../dreamer/notes.md)、[TD-MPC2](../tdmpc2/notes.md)；
  - 最接近的前作是 DINO-WM（Zhou 2024）和 Sobal 2025，它们用冻结或端到端的表征做 zero-shot 规划，但只在小规模环境里评测；
  - 数据用 Droid，优化器用 CEM（Rubinstein 1997）加 receding-horizon MPC；
  - 对比对象是 [Cosmos](../cosmos-wfm/notes.md) 和 Octo。
- **后继 / 影响**：
  - [DreamZero](../dreamzero/notes.md) 把 V-JEPA 2 当作"latent-space world model"的代表来对照：这类模型只学 $p(s_{t+1}\mid s_t,a_t)$，测试时还要做规划或搜索；
  - [WM for Robot Learning 综述](../survey-wm-robot-learning/notes.md) 把 V-JEPA 2 和 V-JEPA 2.1（arXiv 2603.14482）、LeWorldModel 归为 JEPA 线，定位是"embedding 空间里的预测性规划"，认为它还算不上成熟的 policy evaluator；
  - 官方仓库已经放出 V-JEPA 2.1 的权重（论文之外的信息，2026-09-27 查仓库 README 确认）。
- **与本项目其他主题的联系**：
  - **World model**：
    - 和 [DreamerV3](../dreamerv3/notes.md)（RSSM，带重建和 reward 头，在 imagination 里用 actor-critic 学 policy）正好相反：V-JEPA 2-AC 训练时没有 decoder（App.B.3 的帧 decoder 只用于可视化），没有 reward，也没有 policy，只有 energy 加 CEM；
    - 和 [TD-MPC2](../tdmpc2/notes.md) 一样用 latent MPC，但 TD-MPC2 有 reward 和 value，能做长 horizon；
    - 和 [Genie](../genie/notes.md)、[DIAMOND](../diamond/notes.md)、[Cosmos](../cosmos-wfm/notes.md) 构成"表征空间预测还是像素生成"的对照，Table 3 是这一对照的直接证据。
  - **TTT**：
    - 规划本身就是 test-time optimization；
    - App.B.4 提出一种无监督标定：让机器人执行随机动作，用最小二乘拟合"推断动作→真实动作"的 $W^\star$，再旋转输出动作。作者称之为 unsupervised calibration，并明确说**没做**；按本项目的术语可以把它看作一种部署时的 test-time adaptation（这是我们的解读）。它和 [PAD](../../05_intersections/pad/notes.md)、[TTT](../../02_test_time_training/ttt-self-supervised/notes.md) 的思路直接相通：部署时可以用 JEPA 的预测 loss 作为 self-supervised auxiliary task 来更新 predictor。
  - **Real2sim2real**：
    - V-JEPA 2-AC 相当于一个从真实数据学出来的"神经 simulator"，不做 [domain randomization](../../03_real2sim2real/domain-randomization/notes.md)，也不做 system identification，就能迁移到新实验室；
    - 但相机视角偏移是一种视觉上的 reality gap，可以考虑用相机位姿随机化或 viewpoint 条件化来缓解；
    - 可以和 [Robotic World Model](../../05_intersections/robotic-world-model/notes.md)、[Ctrl-World](../../05_intersections/ctrl-world/notes.md) 对照看。
  - **RL**：
    - 本文没有 RL。作者在 Sec.4.2 把"在 world model 的 imagination 里训一个前馈 policy 来初始化规划"列为降低规划耗时的方向之一；进一步把 Eq.5 的 energy 当作 goal-conditioned reward 来训这个 policy，是我们的延伸，论文没有提；
    - 在 world model 里训 policy 正是 [WMPO](../../05_intersections/wmpo/notes.md) 和 [World-Gymnast](../../05_intersections/world-gymnast/notes.md) 走的路线，不过它们用的是像素生成式 WM，reward 来自学到的成功判别器或 VLM，而不是 latent energy。

## 前置知识

1. **JEPA / 自监督表征学习**：I-JEPA、V-JEPA、BYOL 式 EMA teacher 加 stop-gradient 防 collapse，masked modeling（对比 MAE / VideoMAE 的像素重建）。
2. **ViT 和位置编码**：patch 或 tubelet embedding、RoPE（以及 3D 扩展）、block-causal attention。
3. **Model-based control**：MPC / receding horizon、Cross-Entropy Method、teacher forcing 与自回归 rollout 的误差累积。
4. **机器人基础**：末端笛卡尔控制、operational space control、goal-conditioned BC 与 hindsight relabeling（理解 Octo baseline 要用）。
5. **评测协议**：frozen encoder 加 attentive probe、EK100 的 mean-class Recall@5、LLaVA 式视觉指令微调（VidQA 部分）。

## 复现可行性

**已开源的部分**

- 仓库 https://github.com/facebookresearch/vjepa2 可访问（2026-09-27 查 README 确认；本小节的仓库信息都来自 README，不来自论文），大部分代码是 MIT 许可，部分文件另有 Apache 2.0 等许可。
- 放出了 ViT-L/H/g/g$_{384}$ 的预训练权重，以及 V-JEPA 2-AC 权重（torch hub 名 `vjepa2_ac_vit_giant`）。
- 有预训练和后训练代码。AC 后训练的配置是 `configs/train/vitg16/droid-256px-8f.yaml`，另有 `notebooks/energy_landscape_example.ipynb`。
- 依赖 PyTorch、timm、einops、decord（README 写明 decord 不支持 macOS，建议换 eva-decord 或 decord2）。
- 仓库 README 里没看到 CEM 真机规划脚本，也没有 Droid 预处理说明。

**算力**

- 从头预训练 ViT-g 属于 A100 GPU-years 量级（见上文推算），个人基本不可能从头复现。
- AC 后训练的算力论文没有给出。已知规模：300M predictor、batch 256、共 94.5K iter（4.5K+85.5K+4.5K），而且每帧都要过一遍冻结的 1B 编码器（在线算还是预先缓存，论文没说）。
- 规划阶段：单张 RTX 4090，每个动作约 16 s。

**最小可复现实验**（不需要真机）

1. 加载官方 V-JEPA 2 编码器和 AC predictor 权重。
2. 从 Droid 里取出不在训练集里的左视角轨迹（或自己录一段带末端位姿的桌面视频），复现 Fig.9 的 **energy landscape**：扫描 $(\Delta x,\Delta y)$，检查最小值是否靠近真值动作。
3. 做 **离线 action inference**：对相邻两帧用 $T=1$ 的 CEM 解出 $a^\star$（App.B.4 没写这一步的 CEM 配置，可以先沿用 App.B.2 的 800 样本、10 轮、top-10，以及 Sec.4.1 的 L1-ball 0.075），只取 $\Delta x,\Delta y$，和真实 $a$ 做最小二乘，复现 App.B.4 的旋转误差分析和约 1.6 cm 的误差。
4. 进一步：在 Droid 上重训 AC predictor，做论文缺失的消融（去掉 rollout loss，或把编码器换成 DINOv2），指标用离线 action-inference 误差。

**预计难点**

- Droid raw 数据体量很大，需要自己筛出 ≥4 s 的片段并对齐末端状态；
- 1B 编码器的前向显存占用；
- 真机结果依赖相机摆位和 blocking controller，离线复现无法验证成功率。

## 自测问题

1. Stage 1 靠什么防止 representation collapse？Stage 2 为什么不再需要 EMA target？如果 Stage 2 同时解冻编码器，会出现什么风险？
2. 写出 teacher-forcing loss 和 rollout loss。rollout loss 为什么只取 $T=2$？它想解决 train/test 之间的什么不一致？
3. 主实验的 planning horizon 是 1。在这种设定下，"world model 的多步预测能力"在多大程度上真正被检验了？pick-and-place 为什么必须提供 sub-goal？
4. 为什么相机位置变化会表现为推断动作坐标轴的**旋转**误差？App.B.4 的最小二乘标定在 TTT 框架里对应什么？
5. Table 20 中 predictor-only 只有 20.2、encoder-only 有 39.1。这说明 EK100 "prediction" 结果的主要来源是什么？这对理解 V-JEPA 2-AC 的能力边界有什么启示？
