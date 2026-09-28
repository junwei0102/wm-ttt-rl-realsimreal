# Tent: Fully Test-Time Adaptation by Entropy Minimization

> arXiv [2006.10726](https://arxiv.org/abs/2006.10726) · ICLR 2021（Spotlight） · 主题：02_test_time_training · 层级：foundational
> 代码：https://github.com/DequanWang/tent（摘要脚注给出；2025-02 起仓库已 archived、只读）· 笔记依据：`source/*.tex`（数字/公式）+ `paper.pdf`（图）

## 一句话

提出 **fully test-time adaptation** 设定：部署时手里只有训练好的模型参数 $\theta$ 和无标签的 target 数据 $x^t$，不碰 source 数据、也不改训练流程。Tent 在测试时逐 batch 地**最小化模型自身预测的 Shannon entropy**，只更新 BatchNorm 的 target 统计量 $\mu,\sigma$ 和 channel-wise 仿射参数 $\gamma,\beta$（占模型参数 <1%）。只做一轮前向加一次梯度，ImageNet-C 的误差就从 59.5% 降到 44.0%。

## 问题设定

- **输入/输出**：一个在 source 数据 $(x^s,y^s)$ 上训好的分类器 $f_\theta$（需要满足三点：已针对监督任务训好、输出概率分布、可微，见 `source/03-method.tex` 开头），以及一串无标签的 target batch $x^t$。输出是对 target 的预测，模型边预测边更新。
- **与已有设定的区别**（`source/02-setting.tex` Table 1）：

| setting | source data | target data | train loss | test loss |
|---|---|---|---|---|
| fine-tuning | – | $x^t,y^t$ | $L(x^t,y^t)$ | – |
| domain adaptation | $x^s,y^s$ | $x^t$ | $L(x^s,y^s)+L(x^s,x^t)$ | – |
| test-time training (TTT) | $x^s,y^s$ | $x^t$ | $L(x^s,y^s)+L(x^s)$ | $L(x^t)$ |
| **fully test-time adaptation** | – | $x^t$ | – | $L(x^t)$ |

- **动机**（`source/01-intro.tex`）分三点：availability（出于带宽、隐私或商业原因，模型发布时不附带 source 数据）、efficiency（测试时重新处理 source 数据的算力不现实）、accuracy（不做适应，模型准确率不够用）。
- **之前卡在哪**：DA 需要 source 和 target 联合训练；[TTT](../ttt-self-supervised/notes.md) 必须在训练时就把 rotation 这类 self-supervised auxiliary task 和主任务一起训，而且 proxy task 要选得和领域对得上，proxy 与主任务的优化也要平衡（`source/03-method.tex` §3.1）。拿到一个现成的预训练模型之后，这两条路都走不通。
- **隐含假设**：target 以 batch 形式到达（BS=64/128），顺序经过 shuffle，接近 i.i.d.；shift 属于 covariate shift，标签空间不变；source 模型在 target 上"大致对"，自信的预测多半是正确的。

## 核心方法

**1. 为什么选 entropy。** 目标来自 `source/03-method.tex` §3.1（正文公式，无编号）：

$$H(\hat y)=-\sum_c p(\hat y_c)\log p(\hat y_c),\qquad \hat y=f_\theta(x^t)$$

作者给出两条经验依据。Fig.1（CIFAR-100-C）：entropy 越低，error 越低，按 entropy 分桶后 error 单调上升。Fig.2：corruption 越重，loss 和 entropy 一起升高，二者的 rank correlation 是 $\rho=0.61$。所以 entropy 既能间接反映 error，又能在无标签时估计 shift 有多大。和 rotation 这类 proxy task 相比，entropy 直接定义在监督任务的输出上，不需要额外设计 head，也不需要调多个 loss 的权重。

**2. 平凡解与 batch。** 如果只对单个样本优化，把全部概率压到 argmax 类就能让 entropy 归零。Tent 的做法是在**整个 batch 共享的**参数上联合优化 batch 内所有预测。实现里 loss 是 batch 平均 $\mathcal L=\frac1B\sum_i H(f(x_i^t))$（官方 `tent.py` 中 `softmax_entropy(...).mean(0)`，正文没有写成公式）。

**白板推导（笔者补充，非论文内容）**：设 logits 为 $z$， $p=\mathrm{softmax}(z)$，则

$$\frac{\partial H}{\partial z_j}=-p_j\left(\log p_j+H\right)$$

梯度下降时， $z_j$ 的变化正比于 $p_j(\log p_j+H)$。满足 $\log p_j>-H$ 的类（概率高于"典型值"）被推高，其余被压低，效果就是 **sharpening，强者愈强**，相当于用 soft pseudo-label 做 self-training（论文的 PL baseline 是它的 hard 版本）。这也说明了失败方式：source 模型一旦自信地错了，entropy minimization 会把这个错误继续放大。

**3. 只更新什么：feature modulation**（`source/03-method.tex` §3.2，Fig.4）。直接更新整个 $\theta$ 有两个问题： $\theta$ 是 source 数据唯一的"记忆"，改动它会让模型偏离训练时的状态；而且 $\theta$ 维度高、 $f$ 非线性，测试时优化既敏感又慢。因此 Tent 只更新线性、低维（channel-wise）的调制，直接复用 BN 层：

$$\bar x=\frac{x-\mu}{\sigma},\qquad x'=\gamma\bar x+\beta$$

- **normalization**： $\mu\leftarrow\mathbb E[x^t],\ \sigma^2\leftarrow\mathbb E[(\mu-x^t)^2]$，在前向时用当前 target batch 估计，**由数据统计得到、不由 loss 优化**（论文："statistics are estimated from the data while the parameters are optimized by the loss"）；source 的 running statistics 直接丢弃。这一步单独使用就是 baseline "test-time normalization (BN)"（Schneider et al. 2020 / Nado et al. 2020）。
- **transformation**： $\gamma,\beta$ 由 $\nabla_{\gamma,\beta}H$ 更新。Fig.4 把更新写成 $\gamma\leftarrow\gamma+\partial H/\partial\gamma$，符号不严谨，实际是用 Adam 或 SGD 做梯度**下降**。

**4. 算法**（`source/03-method.tex` §3.3）：
- *Initialization*：收集每层 $l$、每个 channel $k$ 的 $\lbrace\gamma_{l,k},\beta_{l,k}\rbrace$，其余参数冻结，source 统计量丢弃。
- *Iteration*：前向时逐层用当前 batch 估计 $\mu,\sigma$ 并输出预测；反向时用 entropy 梯度更新 $\gamma,\beta$。**本 batch 的预测在更新之前就已给出，更新只作用于下一个 batch**，所以每个测试点只多花一次梯度计算。
- *Termination*：online 模式下只要有数据就持续更新；offline 模式先在整个 target 集上更新（可跑多个 epoch），再重新推理。另有 episodic 模式（§3.3 未列出，见 §4.2 分割实验、附录 Fig.9 及官方代码的 `episodic` 选项）：每次从头 reset，只针对单个输入迭代若干步，分割实验里就把一张图当作"一个 batch 的像素"。
- 官方实现中，`configure_model` 把模型设为 `train()`（让 BN 使用 batch 统计），设置 `track_running_stats=False`，并只对 BN 层打开 `requires_grad`。

**训练 vs 推理**：训练完全不改动，就是标准的监督训练；所有额外工作都发生在推理阶段。超参数沿用 source 训练的配置：ImageNet 用 SGD+momentum，BS=64，LR=0.00025；其余数据集用 Adam，BS=128，LR=0.001。batch 调小时按比例降低 LR（`source/04-experiments.tex` Optimization）。

## 关键实验与证据

模型：CIFAR 用 R-26，ImageNet 用 R-50，数字任务用 R-26，分割用 HRNet-W18，VisDA-C 用 ResNet-50（ImageNet 预训练）。所有方法共用同一架构，数据顺序经过 shuffle 且各方法相同。

**CIFAR-10-C / 100-C，severity 5，offline**（Table 2，`source/04-experiments.tex`），error %：

| Method | 用 source | 用 target | C10-C | C100-C |
|---|---|---|---|---|
| Source | train | | 40.8 | 67.2 |
| RG (DANN) | train | train | 18.3 | 38.9 |
| UDA-SS | train | train | 16.7 | 47.0 |
| TTT | train | test | 17.5 | 45.0 |
| BN | | test | 17.3 | 42.6 |
| PL | | test | 15.7 | 41.2 |
| **Tent** | | test | **14.3** | **37.3** |

不用 source，只做一个 epoch 的 test-time 优化（offline），Tent 就胜过需要多 epoch 联合训练的 DA 和 TTT。

**ImageNet-C，online，15 种 corruption × 5 个 level 取平均**（Fig.5 图例 + §4.1）：source 59.5%，BN 49.9%，Tent **44.0%**，offline 为 42.3%。对比的 SOTA 鲁棒训练方法：ANT 50.2%，AugMix 51.7%，ANT+SIN 47.4%。除 noise 类（ANT 正是在 noise 上训练的）外，Tent 在各 corruption 上都优于 ANT；在原始数据上误差没有升高。论文称相对 BN 有 18% 的相对改进、相对 Schneider et al. 报告的 49.3% 有 14% 的相对改进（注意：直接按 49.9→44.0、49.3→44.0 算只有约 12%、11%，论文未说明口径）。

**Source-free digit DA**（Table 3），error %，MNIST / MNIST-M / USPS：
- Source：18.2 / 39.7 / 19.3；RG（10+10 epochs）：15.0 / 33.4 / 18.9；UDA-SS（10+10）：11.1 / **22.2** / 18.4
- BN（0+1）：15.7 / 39.7 / 18.0；Tent（0+1）：10.0 / 37.0 / 16.3；Tent（0+10）：**8.2** / 36.8 / **14.4**
- SVHN→MNIST：跑 100 epochs 可到 6.5%；online 模式 1 epoch 为 12.5%，10 epochs 为 8.4%。RG/UDA-SS 用到的 source 数据约为 target 的 7 倍，Tent 的计算量约为它们的 1/80。
- **失败案例**：MNIST→SVHN 的 source-only 误差 71.3%，Tent 反而**升到 79.8%**；DIRT-T 为 45.5%，UDA-SS 为 38.7%。

**Sim-to-real**：GTA→Cityscapes 分割（HRNet-W18），mIoU source 28.8%，BN 31.4%，Tent 35.8%（offline，Adam）。单张图像 episodic 迭代 10 次为 36.4%，定性结果见 Fig.9：路面噪声被抑制，漏掉的 motorcycle/rider 类被找回。VisDA-C（合成渲染→真实，12 类）：source 56.1%，Tent 45.6%；改用 SHOT 的参数化（更新除最后分类器外的所有层）后为 39.6%。

**分析 / 消融**（§4.3）：
- Fig.6：在 CIFAR-100-C 全部 75 个 type/level 组合上， $\Delta H$ 与 $\Delta L$ 的 rank correlation 为 0.22，大多数点的 entropy 和 loss 都下降了。
- **参数选择消融**（只有文字，没有给表格数字）：不更新 normalization 时误差上升，可能连 BN、PL 都比不过；不更新 $\gamma,\beta$ 就退化成 BN；只更新最后一层会先变好、继续优化后变差；**更新全部 $\theta$ 从未优于未适应的 source 模型**（"never improves over"）。由此可见，"低维调制"这个约束是方法能奏效的前提。
- 泛化性：在 target train 上适应、在 target test 上评测，CIFAR-100-C 从 37.3% 降到 34.2%，SVHN→MNIST 从 8.2% 降到 6.5%（子采样到相同规模后差异 <0.1%），说明学到的调制不是针对单个点的。
- Fig.7 / Fig.10 的特征分布：BN 把特征拉回无 corruption 时的 reference；Tent 并不向 reference 靠拢，而是更接近在 target 标签上优化得到的 oracle。可见 Tent 的作用是 task-specific 的，不只是把分布对齐回去。
- 其他架构（Table 4，CIFAR-100-C，Source/BN/Tent）：SAN-10 pair 55.3/39.7/**36.7**，SAN-10 patch 48.0/31.8/**29.2**，MDEQ large 53.3/44.9/**41.7**，均未重新调参。

## 局限与论文没说的

**作者自己承认的**（`source/06-conclusion.tex`、§4.2）：
- 对 CIFAR-10.1、ImageNetV2 这类"自然但未知"的 shift，Tent **没有带来提升**。
- MNIST→SVHN 这种大 shift 会失败，误差反而升高；要达到最高精度、应对更难的 shift，仍需要 source 与 target 联合优化的 UDA。
- entropy 需要 batch，无法像 TTT 那样对单个点做 episodic 更新（分割任务能做到，是因为一张图本身就是一个像素 batch）。
- 可更新的参数有限，VisDA 这种大 shift 用 SHOT 的参数化效果更好；每次更新都要完整地前向加反向；和 calibration 的关系尚未研究。

**笔者读出来的**：
1. **i.i.d. + 大 batch 假设很强**。实验固定 BS=64/128，而且显式 shuffle，没有做 batch size 扫描，也没有测试非 i.i.d. 或类别不均衡的数据流。BN 统计量在小 batch 或时间相关数据上会严重失真，而机器人部署时的观测恰恰是时间强相关的单帧流。
2. **没有测试连续变化的 shift 或长时程 online 稳定性**。75 个 corruption 组合各自作为独立 target（正文没有写明组合之间是否 reset）。按上面推导的"强者愈强"动力学，长时间运行后有塌缩到少数类的风险，论文没有讨论。
3. **消融只有文字，没有数字**。"更新全部 $\theta$ 从未优于 source"这样的关键结论缺少表格支撑。
4. **模型选择口径模糊**。LR 等超参数"沿用 source 训练配置"，PL 的阈值是 "tuned"，但论文没说用什么数据来调；在严格的 source-free、无标签设定下，调参本身就是个问题。
5. **任务形式受限**。方法要求输出是 categorical 分布，所以只验证了分类和逐像素分类。对于回归、连续动作 policy（Gaussian）、生成式 world model，entropy 要么没有良好定义，要么与目标冲突。
6. **没有报告算力**：没有 GPU 型号或耗时，只说"每个测试点一次梯度"，以及数字任务上约 80× 的计算节省。

## 在脉络中的位置

- **前驱**：semi-supervised 里的 entropy regularization（Grandvalet & Bengio 2005）和 pseudo-labeling（Lee 2013）；DA 中以 entropy 为正则的方法，如 DIRT-T、AutoDIAL；BatchNorm 与 FiLM 式 feature modulation；test-time normalization（Schneider et al. 2020、Nado et al. 2020）；[TTT (Sun et al. 2019)](../ttt-self-supervised/notes.md) 提出了"测试时用无监督 loss 更新模型"的范式，Tent 去掉了它对训练流程的改动和对 proxy task 的依赖。同期的 SHOT 同样用了 entropy（information maximization），但 loss 多、参数化不同、只做 offline。
- **后继 / 影响**：Tent 成为 TTA 分支的标准 baseline，MEMO、CoTTA、EATA、SAR 等后续工作分别针对单样本、连续 shift、效率与遗忘、小 batch/非 i.i.d. 下的塌缩等问题改进（这些不在本项目内，属论文之外的背景）。在本项目内，[TTRL](../ttrl/notes.md) 把"用模型自身输出（majority voting 共识）作为无标签监督信号"搬到了 LLM 的 test-time RL 上，思路与 Tent 的 confidence-as-supervision 一脉相承；[综述](../survey-test-time-intelligence/notes.md) 把这类方法归入 feedback-driven 的测试时自我改进。[TTT layers](../ttt-layers/notes.md)、[Titans](../titans/notes.md)、[LaCT](../lact/notes.md) 走的是另一条路：把"测试时更新 fast weights"做成架构内的 inner loop，用自监督重建 loss，而不是做分布偏移下的 TTA。两条路共同面对的问题都是"测试时更新哪些参数、用什么 loss"。
- **与本项目其他主题的联系**：
  - *real2sim2real*：GTA→Cityscapes 和 VisDA-C 本身就是 perception 的 sim-to-real gap。[domain randomization](../../03_real2sim2real/domain-randomization/notes.md) 在训练端扩大分布，Tent 在部署端做无源适应，两者可以叠加。[RMA](../../03_real2sim2real/rma/notes.md) 则用训练好的 adaptation module 做前向式的在线隐式 system ID，不需要测试时梯度，和 Tent 的"梯度式 TTA"是互补的两类思路。参见 [reality gap 综述](../../03_real2sim2real/survey-reality-gap/notes.md)。
  - *RL*：[PAD](../../05_intersections/pad/notes.md) 与 Tent 是同一届 ICLR（2021）的工作，它把 TTT 的自监督辅助任务（控制任务用 inverse dynamics）用在部署时的 policy encoder 上。Tent 式的 entropy minimization 直接用在 policy 上会有问题：[SAC](../../04_rl/sac/notes.md) 的 maximum entropy 目标、[PPO](../../04_rl/ppo/notes.md) 的 entropy bonus 都是在**鼓励**高 entropy，最小化 policy entropy 会与之冲突，也会削弱 exploration。
  - *world model*：world model 本身就带有可在测试时计算的自监督 loss（重建 / 下一步预测 / KL，见 [Dreamer](../../01_world_model/dreamer/notes.md)、[DreamerV3](../../01_world_model/dreamerv3/notes.md)），因此它更自然的 TTA 信号是 TTT/PAD 那样的预测误差，而不是 Tent 的 entropy。Tent 给这条线留下的主要启示是**参数化的选择**：测试时只动低维的 normalization/affine 参数，是稳定性的关键（参见"更新全部 $\theta$ 从未优于 source"）。

## 前置知识

1. **Batch Normalization**：train/eval 两种模式下统计量的差别，running statistics 与 batch statistics，仿射参数 $\gamma,\beta$ 的作用。这是理解 BN baseline 与 Tent 区别的前提。
2. **Softmax、Shannon entropy、cross-entropy**，以及 entropy 对 logits 的梯度。
3. **Dataset shift / covariate shift**，以及 corruption benchmark 协议（CIFAR-C / ImageNet-C：15 种 corruption × 5 个 severity，Hendrycks & Dietterich 2019）。
4. **Unsupervised DA 与 source-free DA**：DANN 的 gradient reversal（RG），以及 SHOT 的思路。
5. [**Test-time training (Sun et al.)**](../ttt-self-supervised/notes.md)：self-supervised auxiliary task 加共享 encoder。
6. **Self-training / pseudo-labeling / entropy regularization**：低密度分离假设。
7. online、offline、episodic 三种适应协议的区别。

## 复现可行性

- **代码**：官方仓库 https://github.com/DequanWang/tent，MIT 许可，2025-02 起 archived。据 README，仓库包含 `tent.py`（核心逻辑）、`norm.py`（BN baseline）以及基于 RobustBench v0.1 预训练 WRN 的 CIFAR-10-C 示例（`python cifar10c.py --cfg cfgs/{source,norm,tent}.yaml`）。README 明确说明这是"explanation, not reproduction"，模型和优化细节可能与论文不同；精确复现 ImageNet-C 的代码没有放出。依赖为 PyTorch、RobustBench、yacs。
- **算力**：论文**没有报告** GPU 型号、数量和时长。方法本身每个测试点只多一次梯度（反向）计算，前向与预测共用，只更新 BN 仿射参数（<1% 参数），CIFAR 规模单卡即可。ImageNet-C 需要准备 75 份 50k 验证集的 corruption 版本，存储和 I/O 是主要开销。
- **数据**：CIFAR-10/100-C、ImageNet-C（公开）；SVHN、MNIST、MNIST-M、USPS；GTA5、Cityscapes；VisDA-C。
- **最小可复现实验**：
  1. 用官方 CIFAR-10-C 示例在 severity 5 的 15 种 corruption 上跑 source / norm / tent，确认 Tent < BN < Source 的排序。注意论文 Table 2 用的是 R-26，与仓库里的 WRN 不同，**数字不会一一对上**，只看相对趋势。
  2. 参数化消融：只更新统计量、统计量加仿射参数、全部参数三种设置，验证"全参数更新变差"。
  3. 论文没做的压力测试：batch size 扫描（128→16→1），按类排序的非 i.i.d. 数据流，15 种 corruption 串成一条流、中间不 reset 的连续适应，观察长时程下是否塌缩。
- **预计难点**：BN 在 `train()` 模式下的细节（`track_running_stats=False`、丢弃 running stats）容易写错；数据顺序和 batch size 对结果影响很大；在无标签设定下调 LR 缺少公认的规范。

## 自测问题

1. 对单个样本最小化 entropy 为什么有平凡解？Tent 靠什么避免它？batch size 降到 1 时，normalization 和 transformation 这两步分别会出什么问题？
2. 推导 $\partial H/\partial z_j=-p_j(\log p_j+H)$，并据此解释为什么 Tent 会放大 source 模型"自信的错误"。MNIST→SVHN 从 71.3% 升到 79.8% 这个结果怎样与此呼应？
3. BN baseline 与 Tent 分别"估计"了什么、"优化"了什么？Fig.7 中两者特征偏移的方向不同，说明了什么？
4. 按 Table 1 的四列比较 TTT 与 Tent：为什么 TTT 必须改训练，而 Tent 不用？各自需要付出什么代价？
5. 如果要把 Tent 的思路用到机器人部署（时间相关的单帧观测、连续动作 policy 或 world model）上，loss 和参数化分别该怎么改？哪些假设会被破坏？
