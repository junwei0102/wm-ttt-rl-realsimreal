# Test-Time Training with Self-Supervision for Generalization under Distribution Shifts

> arXiv [1909.13231](https://arxiv.org/abs/1909.13231) · ICML 2020（arXiv 初版 2019） · 主题：02_test_time_training · 层级：foundational
> 代码：https://github.com/yueatsprograms/ttt_cifar_release（CIFAR-10-C / CIFAR-10.1），https://github.com/yueatsprograms/ttt_imagenet_release（ImageNet-C / VID-Robust）；论文里只给了项目页 https://test-time-training.github.io/（`source/tex/intro.tex` 脚注）· 笔记依据：`source/*.tex`（数字/公式）+ `paper.pdf`（图）

## 一句话

训练时让监督主任务和 rotation prediction 这个 self-supervised auxiliary task 共享一个 feature extractor 做 joint training；测试时拿到一个无标签样本 $x$，先用 $x$ 上的自监督损失对共享 extractor 做几步梯度更新，再用更新后的参数预测。这样模型参数就成了测试输入的函数 $\boldsymbol\theta(x)$，也就是论文说的 variable decision boundary。

## 问题设定

- **任务**：图像分类。训练集 $\{(x_i,y_i)\}_{i=1}^n \overset{iid}{\sim} P$，测试样本来自未知的 $Q\neq P$。测试时只拿得到 $x$，拿不到 $y$。
- **假设**：训练阶段看不到任何目标分布的数据，这一点和 unsupervised domain adaptation（UDA）不同；也不预先假设 shift 属于哪一类，这一点和 adversarial robustness 预设扰动集 $\Delta$ 不同。online 版本多一个假设：测试流 $x_1,x_2,\dots$ 来自同一个 $Q$，或者来自缓慢变化的 $Q_t$（`source/tex/intro.tex`、`results.tex` "Gradually changing"）。
- **之前的方法卡在哪**（`source/tex/related.tex`）：
  1. adversarial training、randomized smoothing 等都在"平滑"一个固定的 decision boundary，属于 one-boundary-fits-all，天然存在 accuracy 与 robustness 的 tradeoff，而且依赖 $\Delta$ 的数学结构。
  2. UDA 需要在训练时就拿到目标域的无标签数据，泛化只针对那一个目标分布。
  3. domain generalization 受限于 meta distribution，出了训练时见过的环境族就失效。
  4. 在 joint training 里加入 self-supervision 已知能提高 robustness（Hendrycks et al. 2019），但测试时模型仍是固定的。本文把它当作主要的强 baseline（原文 "a strong baseline"，即 joint training）。

## 核心方法

**1. Y 形结构。** 把 $K$ 层网络切在第 $\kappa$ 层：前面的 $\boldsymbol\theta_e=(\theta_1,\dots,\theta_\kappa)$ 是 shared feature extractor；后面分成两支，一支是 main task branch $\boldsymbol\theta_m$，另一支是 self-supervised task branch $\boldsymbol\theta_s$。$\boldsymbol\theta_s$ 与 $\boldsymbol\theta_m$ 结构相同，只有最后一层的输出维度不同。自监督任务是 rotation prediction：把图像旋转 0/90/180/270°，做 4 分类（`source/tex/method.tex`）。CIFAR 上用 ResNet-26（共 3 个 group），切在第 2 个 group 末尾；ImageNet 上用 ResNet-18（共 4 个 group），切在第 3 个 group 末尾（`results.tex` "Network details"）。

**2. 训练：multi-task joint training**（`source/tex/method.tex` Eq.2）

$$\min_{\boldsymbol\theta_e,\boldsymbol\theta_m,\boldsymbol\theta_s}\ \frac1n\sum_{i=1}^n l_m(x_i,y_i;\boldsymbol\theta_m,\boldsymbol\theta_e)+l_s(x_i;\boldsymbol\theta_s,\boldsymbol\theta_e)$$

两项损失直接相加，没有加权系数。

**3. 测试：standard TTT**（`method.tex` Eq.3）。对单个测试样本 $x$：

$$\min_{\boldsymbol\theta_e}\ l_s(x;\boldsymbol\theta_s,\boldsymbol\theta_e)$$

记 $\boldsymbol\theta_e^\star$ 为它的（近似）最优解，模型再用更新后的参数 $\boldsymbol\theta(x)=(\boldsymbol\theta_e^\star,\boldsymbol\theta_m)$ 做预测。

- 这里的 batch 只包含 $x$ 的若干增强副本（random crop 与 horizontal flip，和训练时相同）。用 SGD，learning rate 取训练最后一个 epoch 的 0.001，weight decay 和 momentum 设为 0，一共走 **10 步**（`results.tex` "Optimization details"）。
- 预测完就丢掉 $\boldsymbol\theta_e^\star$，下一个样本从 joint training 得到的参数重新开始。
- 只更新 $\boldsymbol\theta_e$ 还是同时更新 $\boldsymbol\theta_e$ 与 $\boldsymbol\theta_s$，实验上差别可以忽略；理论上只在走不止一步时才有差别。
- 推理时 $\boldsymbol\theta_m$ 不动，所以主任务能否受益，完全取决于"在 $l_s$ 上改动 $\boldsymbol\theta_e$"是否也有利于 $l_m$。

**4. TTT-Online。** 处理 $x_t$ 时，从上一个样本更新后的 $\boldsymbol\theta(x_{t-1})$ 出发，而不是从初始参数出发，每个样本只走 **1 步**。这样参数会累积 $x_1,\dots,x_t$ 的分布信息（`method.tex`）。`related.tex` 把 TTT 的更新规则整体看作 on the fly 的 one-sample unsupervised domain adaptation，并提醒只有一个目标域样本时标准 domain adaptation 技术可能失去定义。

**5. 为什么用 GN 不用 BN。** 一个 batch 里全是同一张图的副本，BN 估出的统计量不准，所以主结果都用 Group Normalization（`results.tex`；GN 不依赖 batch size）。BN 的结果见附录 A4.1：standard 版只对 $l_s$ 大的样本做更新（hard example mining），仍然有效；online 版在 BN 下不可用，因为统计量误差会在上万步里不断累积。

**6. 理论：gradient correlation**（`source/tex/theory.tex` Eq.4–9，证明在 `source/tex/proofs.tex` 附录 A3）。先看 toy model，一个两层线性网络。$\mathbf A\in\mathbb R^{h\times d}$ 是共享层，$\mathbf v$ 和 $\mathbf w$ 是两个 head。记 $r_1=y_1-\mathbf v^\top\mathbf A x$，$r_2=y_2-\mathbf w^\top\mathbf A x$，则 $l_m=\tfrac12 r_1^2$，$l_s=\tfrac12 r_2^2$。

- 在 $l_s$ 上走一步（Eq.6）：$\mathbf A'=\mathbf A+\eta\, r_2\,\mathbf w x^\top$，于是 $\mathbf v^\top\mathbf A'x=\mathbf v^\top\mathbf A x+\eta\, r_2\,(\mathbf v^\top\mathbf w)(x^\top x)$。
- 令它等于 $y_1$，解出使主任务损失为 0 的步长（Eq.7）：
$$\eta^*=\frac{r_1}{r_2\,\mathbf v^\top\mathbf w\,x^\top x}$$
- 两个梯度都是秩 1：$\nabla_{\mathbf A}l_m=-r_1\mathbf v x^\top$，$\nabla_{\mathbf A}l_s=-r_2\mathbf w x^\top$。它们的 Frobenius 内积是 $r_1r_2\,\mathbf v^\top\mathbf w\,x^\top x$，因此 $\mathrm{sign}(\eta^*)=\mathrm{sign}\langle\nabla l_m,\nabla l_s\rangle$（附录 Lemma 2）。只要 $\eta^*>0$，由凸性可知任何小的正步长都会降低 $l_m$（Lemma 1）。
- 由此得到 $\eta^*>0$ 的一个充分条件（Eq.8–9，要求 $x\neq0$ 且两个损失都不为 0）：两个任务的残差同号，即"错在同一个方向"；并且 $\mathbf v^\top\mathbf w>0$，即两个 head 在特征空间里的决策方向相关。

**Theorem 1（一般情形）。** 假设 $l_m$ 对 $\boldsymbol\theta$ 可微、凸、$\beta$-smooth，且 $\|\nabla l_m\|,\|\nabla l_s\|\le G$。取 $\eta=\varepsilon/(\beta G^2)$，那么只要

$$\langle\nabla l_m(x,y;\boldsymbol\theta),\nabla l_s(x;\boldsymbol\theta)\rangle>\varepsilon\ \Rightarrow\ l_m(x,y;\boldsymbol\theta-\eta\nabla l_s)<l_m(x,y;\boldsymbol\theta)$$

证明思路：先用 smoothness 得到降幅至少是 $\varepsilon^2/(2\beta G^2)$，此时步长取 $\eta^\star=\langle\cdot,\cdot\rangle/(\beta\|\nabla l_s\|^2)$；然后因为 $\eta\le\eta^\star$，再用凸性插值，说明固定步长 $\eta$ 也能让 $l_m$ 下降。

## 关键实验与证据

对比对象有四个：object recognition task only（B，plain ResNet+GN），joint training（JT，即 Hendrycks et al. 2019 的改进实现），TTT，TTT-Online。TTT-Online 评测时先把整个测试集打乱，当作一条流，报告整条流上的平均结果（`results.tex`）。

- **CIFAR-10-C，level 5**（Figure 1 / Table A2，test error %）：
  - original：B 8.9、JT 8.1、TTT 7.9、TTT-Online 8.2。
  - Gaussian noise：50.5 → 49.4 → 45.6 → **25.8**。
  - pixelation：55.8 → 51.6 → 47.2 → **18.1**。
  - 论文的说法（`results.tex`）：在全部 5 个 level、15 种 corruption 上，standard 版总是优于 JT；online 版经常比 JT 好 10% 以上，最多只差 0.2%；在 3 种 noise 上带来超过 24% 的提升，pixelation 上带来 38%。
  - TTT 在原始测试集上也比 JT 好 0.2%，并且"在多次独立试验中一致"。据此作者认为这种方法不需要在 specificity 和 generality 之间取舍。
- **ImageNet-C，level 5**（Figure 2 / Table A3，accuracy %）：
  - original：B 68.9、JT 69.1、TTT 69.0、TTT-Online 68.8。
  - standard TTT 的提升幅度小，例如 Gaussian noise 从 2.1（JT）到 3.1。
  - TTT-Online 提升很大：Gaussian noise 26.3，fog 47.6（JT 为 17.0），brightness 58.3。但 glass blur 上 B 6.6、JT 6.7、TTT 6.8、TTT-Online 6.6，基本没有提升（TTT-Online 甚至略低于 JT）。
  - Figure 2 下半部分画的是 100 样本滑动窗口的 accuracy：在原始测试集上 5 万个样本后没有明显变化；在 corruption 上 1 万个样本内就明显上升，到第 5 万个样本仍在上升。
- **与 UDA-SS 对比**（Table 1）：UDA-SS 在训练时就能看到整个无标签测试集，作者因此把它当作 oracle。结果 TTT-Online 在 15 种 corruption 中的 13 种以及原始测试集上都优于 UDA-SS，只在 impulse（30.6 vs 20.8）和 contrast（15.6 vs 11.6）上输。作者的解释是 TTT-Online 可以"忘掉"训练分布，而 UDA-SS 必须找到一个对两个分布都成立的 invariant representation。
- **缓慢变化的 shift**（Figure 3）：三种 noise 的 $\sigma_1,\dots,\sigma_n$ 在 level 1 与 level 5 的标准差之间插值（插值方式原文未报告）。TTT-Online 的误差曲线斜率比 JT 平缓（caption），正文称它仍优于 JT 和 standard TTT，在 Gaussian 和 shot 两种 noise 上还优于 UDA-SS。
- **VID-Robust**（Table 3，accuracy，1109 组、30 类，只在 anchor frame 上评测）：B 62.7、JT 63.5、TTT 63.8、TTT-Online 64.3。
- **CIFAR 类别子集**（Table 2）：取 VID-Robust 中与 CIFAR-10 重合的 7 类，平均 accuracy 为 41.4 / 42.4 / 45.2 / 45.4。ship 从 66.7 升到 77.8，dog 从 15.5 升到 21.6（TTT）和 22.4（TTT-Online）。airplane 在 JT、TTT、TTT-Online 下都是 70.2（B 为 67.9），TTT 没有带来提升：大多数图两侧有黑边，给旋转预测提供了平凡的线索；天空中的飞机连人也很难判断是否被旋转过。这是论文里最直接的失败分析，说明自监督任务必须 well-defined，且不能过于简单。
- **CIFAR-10.1**（Table 4，error，测试集 2000 张）：B 17.4、JT 16.7、TTT 15.9。作者称这是第一个在该测试集上改进现有模型的方法，但也承认 0.8% 的提升（相对 4.8%）远小于整体性能下降。
- **机制证据**（Figure 4）：取 15 种 corruption × 5 个 level，共 75 个测试集。每个测试集算一次平均梯度内积，再和误差改善量做回归：standard 版相关系数 0.93，online 版 0.89。梯度取在 shared feature extractor $\boldsymbol\theta_e$ 上。作者用它说明 Theorem 1 的 insight（梯度相关性决定 TTT 是否有效）在非凸深网上同样成立。
- **消融**：
  - BN 对 GN（Table A1）：换成 BN 后 baseline 在几乎所有 corruption 上都更差，作者由此顺带指出 GN 本身能提高 robustness。
  - 切分点：附录 A4.5 说切在第 2 个 group 比切在第 3 个 group 稳定好约 0.5–1%，但没有给出分项表。
  - JT baseline 在全部 corruption 和 level 上的平均误差是 22.8%，Hendrycks 原版是 28.6%（A4.5）。
  - ALP（adversarial logit pairing）在 noise 上很强，但原始测试集误差 16.5，fog 64.8、contrast 73.6（Table A2），印证了前面说的 tradeoff。

## 局限与论文没说的

**作者承认的：**
- 推理开销约为普通测试的 $2\times$`batch_size`$\times$`iterations` 倍（附录 A2）。作者给了两个缓解办法：
  - 只对 $l_s$ 超过阈值的样本做 TTT，约 20% 的样本需要更新；
  - standard 版改成 1 步、lr 0.01，效果"几乎一样好"。
  - 这两种办法都"未充分验证"。
- 自监督任务必须 well-defined 且不平凡（airplane 的例子）。
- CIFAR-10.1 上的提升很小。
- BN 下 online 版不可用。
- 理论部分只覆盖凸模型；variable decision boundary 的讨论是非形式化的（附录 A1）。

**我读出来的：**
1. **需要改训练流程。** 必须事先按 Y 形结构做 joint training，不能直接用在现成的预训练模型上。这正是后来 [Tent](../tent/notes.md) 提出 fully test-time adaptation 时要解决的问题。
2. **部分超参的选择没有说明来源。** lr 有交代（取训练最后一个 epoch 的 lr，即 0.001），weight decay 与 momentum 置 0 也注明参考了 He et al. 2018、Liu et al. 2018；但 10 步是怎么定的没有说明，切分点 $\kappa$ 只有附录 A4.5 一句"比切在第 3 个 group 稳定好约 0.5–1%"，有没有在目标 corruption 上调过，论文都没交代。除了原始测试集那 0.2%，其他结果都没有 error bar 或 seed 数。
3. **online 版的评测口径偏乐观。** 每个 corruption×level 单独构成一条 1 万张（CIFAR）或 5 万张（ImageNet）的 i.i.d. 流，每条流都从 JT 参数重新开始（论文未明说，按 method 的初始化描述推断）。以下情况都没测：多种 shift 混在一条流里、shift 突变、长时间不 reset 导致的 collapse 或遗忘。论文说"forgetting 无害"，这个结论依赖"整条流是同一个分布"这一前提。
4. **Figure 4 的相关性有混杂。** 它是测试集层面的平均相关，不是逐样本相关。corruption 越重，梯度内积和可改进空间可能同时变大，所以相关不等于因果。另外，Theorem 1 的条件依赖未知的 $y$，测试时无法检验，也就不能用来决定对哪个样本做更新。
5. **理论与实践有落差。** 定理对全部 $\boldsymbol\theta$ 做一步更新，步长依赖未知的 $\varepsilon,\beta,G$；实践中只更新 $\boldsymbol\theta_e$，走 10 步，lr 固定。附录 A3 中 smoothness 那一行把 $-\eta\langle\cdot,\cdot\rangle$ 写成了 $+$，是笔误，不影响结论。此外，附录 A2 说阈值技巧"在 method 一节介绍过"，但终版 method 里并没有这部分，只在附录 A4.1 的 BN 部分出现。
6. **适用范围窄。** 只测了图像分类和 rotation 这一种自监督任务，没有检测、分割或控制任务。而对 rotation 不敏感的场景（俯视、天空、医学图像）很常见。
7. **缺少 test-time BN 统计量重估这类简单 baseline。** 这是时代原因，当时这类方法还没成体系。

## 在脉络中的位置

- **前驱**：
  - 在单个测试实例上学习的思路：ZSSR（Shocher et al. 2018）、GAN 适配（Bau et al. 2019）。
  - rotation prediction（Gidaris et al. 2018）。
  - 训练时的 self-supervision 能提升 robustness（Hendrycks et al. 2019），这就是 JT baseline。
  - UDA-SS（Sun et al. 2019），作者自称本文是它的 follow-up。
  - online 版受 Mullapudi et al. 2018 启发；精神上接近 transductive learning（Vapnik）。
  - 以上都不在本项目 related_work 中。
- **后继 / 影响**：
  - TTA 分支：[Tent](../tent/notes.md) 去掉了对训练流程的要求，把学习信号从自监督损失换成预测熵，把更新对象从 $\boldsymbol\theta_e$ 换成 BN 仿射参数。
  - "TTT 即架构"分支：[TTT layers](../ttt-layers/notes.md) 把"隐状态是权重、每读一个 token 就在自监督损失上走一步"做成序列层，后续有 [LaCT](../lact/notes.md)、[One-Minute Video](../ttt-one-minute-video/notes.md)、[TTT-E2E](../ttt-e2e/notes.md)、[TTT-KV ≈ linear attention](../ttt-kv-linear-attention/notes.md)。[Titans](../titans/notes.md) 同样用测试时的梯度更新来实现记忆。
  - LLM 方向：[TTT for ARC](../ttt-arc-fewshot/notes.md)、[TTRL](../ttrl/notes.md)（学习信号换成 majority vote 伪奖励）、[TTT-Discover](../ttt-discover/notes.md)。
  - 综述：[Test-Time Intelligence survey](../survey-test-time-intelligence/notes.md)。
  - RL 方向：[PAD](../../05_intersections/pad/notes.md)（Yu Sun、Efros、Xiaolong Wang 都是作者）把同一范式用到 policy 上，自监督任务换成 inverse dynamics 等，部署时不需要 reward。
- **与本项目其他主题的联系**：
  - **real2sim2real**：本文的出发点就是"不预判 shift，在测试时从 shift 里学"，正好和 [domain randomization](../../03_real2sim2real/domain-randomization/notes.md) 那种"训练时预判 shift"的思路相对。[RMA](../../03_real2sim2real/rma/notes.md) 也在部署时适应，但它不做梯度更新，而是用 adaptation module 前向估计 extrinsics。TTT 则是用无标签的真实观测做梯度更新。两者是 sim-to-real gap 上 test-time adaptation 的两种典型做法（参见 [reality gap survey](../../03_real2sim2real/survey-reality-gap/notes.md)）。
  - **world model**：world model 的训练目标本来就是自监督的（重建或 latent prediction），天然可以当作 $l_s$。Theorem 1 给了一个判断标准：用真实数据在 world model 损失上更新共享 encoder，对下游 policy 或 value 是否有帮助，要看两者的梯度是否相关。[SimDist](../../05_intersections/simdist/notes.md) 在部署时只更新 dynamics，[DayDreamer](../../05_intersections/daydreamer/notes.md) 在真机上在线学 world model，这些都可以放在这个框架下对照。
  - **RL**：TTT-Online 本质上是没有 reward 的 online 更新。它依赖的"测试流近似同分布"假设，在 RL 部署中对应的是环境参数缓慢变化。

## 前置知识

1. multi-task learning 与共享 encoder（hard parameter sharing），以及"辅助任务梯度会不会帮到主任务"这个问题。
2. self-supervised pretext task，尤其是 rotation prediction（Gidaris et al. 2018）。
3. distribution shift 的几种设定之间的区别：covariate shift、UDA、domain generalization、adversarial robustness。
4. BatchNorm 与 GroupNorm 在统计量上的差别，以及小 batch 或单样本 batch 为什么会让 BN 失效。
5. 凸优化基础：$\beta$-smooth 的 descent lemma、凸组合不等式。有了这些才能推导 Theorem 1。
6. benchmark：CIFAR-10-C / ImageNet-C（15 种 corruption × 5 个 level）、CIFAR-10.1、ImageNet-VID-Robust。

## 复现可行性

- **代码**：已开源，WebFetch 确认两个仓库都存在。
  - `ttt_cifar_release` 负责 CIFAR-10-C 与 CIFAR-10.1，入口是 `script.sh` 和 `script_baseline.sh`。
  - `ttt_imagenet_release` 负责 ImageNet-C 与 VID-Robust。
  - 依赖：PyTorch ≥ 1.0、Python 3.7、tqdm、colorama。README 只写了"至少一块有足够显存的现代 GPU"。
- **算力**：论文没有报告 GPU 型号、数量或训练时长。joint training 的 epoch 数也只写了"standard hyper-parameters"（引用 Huang et al. 2016、He et al. 2016）。测试阶段每个样本的开销是普通推理的 $2\times$`bs`$\times$`iters` 倍（附录 A2），而 TTT 用的 batch size 论文正文没有给出。按实验规模看，CIFAR 部分单卡可行，这是个人判断，不是论文给的数字。真正的瓶颈在 standard TTT 的评测：每个样本要走 10 步，CIFAR-10-C 一个 level 就有 15 × 1 万个样本。ImageNet 部分需要完整的 ImageNet 训练和 ImageNet-C，门槛高得多。
- **数据**：CIFAR-10、CIFAR-10-C（Hendrycks & Dietterich 的 robustness 仓库）、CIFAR-10.1（Recht et al.）都可以公开下载，也不需要仿真环境。
- **最小可复现实验**：
  1. 在 CIFAR-10 上训练 ResNet-26+GN，用 Y 形结构（rotation head 接在第 2 个 group 后）做 joint training，同时训练一个 B baseline。
  2. 只在 CIFAR-10-C level 5 的 Gaussian noise 上评测 JT、TTT（10 步，lr 0.001，无 momentum 和 wd）、TTT-Online（1 步，测试集打乱）。
  3. 目标是对上 Table A2 的 49.4 / 45.6 / 25.8，并确认原始测试集约为 8.1 / 7.9 / 8.2。
  4. 加分项：计算每个样本的 $\langle\nabla_{\boldsymbol\theta_e}l_m,\nabla_{\boldsymbol\theta_e}l_s\rangle$，检验 Figure 4 的相关性在逐样本层面是否还成立。
- **预计难点**：
  - GN 与 BN 的实现细节，以及 BN 版的 hard example mining 阈值（论文没有给出数值）。
  - 增强副本的 batch 大小。
  - online 版对测试集顺序敏感，要多跑几个 shuffle 取平均。
  - standard TTT 逐样本评测很慢，最好把对多个样本的独立更新向量化，比如用 `torch.func` 按样本并行。

## 自测问题

1. 在两层线性 toy model 中，推导出使 $l_m(\mathbf A')=0$ 的 $\eta^*$，并证明 $\mathrm{sign}(\eta^*)=\mathrm{sign}\langle\nabla l_m,\nabla l_s\rangle$。为什么 Theorem 1 的条件在测试时无法检验？它在实践中还能提供什么指导？
2. standard TTT 和 TTT-Online 的初始化方式、每个样本的步数有什么不同？为什么 online 版能在 ImageNet-C 上把 Gaussian noise 的 accuracy 从 2.1% 提到 26.3%，而 standard 版只到 3.1%？这个优势依赖测试流的什么假设？
3. 为什么 TTT 的实验必须用 GN？在 BN 下，standard 版和 online 版分别会遇到什么问题，论文各自是怎么处理的？
4. TTT-Online 只能看到已经评测过的样本，为什么还能在 13/15 种 corruption 上胜过能看到整个测试集的 UDA-SS？作者的解释在混合 shift 或 shift 突变的测试流下还成立吗？
5. airplane 类为什么没有提升？如果把 TTT 用到机器人视觉 policy 上（参考 PAD），你会选什么自监督任务？切分点 $\kappa$ 放在哪？怎么验证它与控制目标的梯度是正相关的？
