# Test-Time Training

> 主题目录 `related_work/02_test_time_training/`，共收录 12 篇：1 篇综述、3 篇 foundational、5 篇 milestone、3 篇 frontier。
> 与本线直接相关的交叉工作不在这个目录里：[PAD](../05_intersections/pad/notes.md)、[SimDist](../05_intersections/simdist/notes.md)、[World-Gymnast](../05_intersections/world-gymnast/notes.md) 在 05_intersections，作为对照的 [RMA](../03_real2sim2real/rma/notes.md) 在 03_real2sim2real。
> 本文所有数字都来自各篇 `notes.md`，出处（表号、公式号、`source/` 文件）以笔记为准。标"笔者"的内容是笔记作者或本文的推断，不是论文结论。

## 1. 这条线在解决什么问题

参数冻结的模型在部署时会碰到两类用不上的信息。第一类是偏离训练分布的输入，例如图像 corruption，或 sim-to-real 带来的视觉与动力学差异。第二类是训练时不存在的新信息，例如一段很长的上下文、一个新任务的几条示例，或者一个只要求找到最优解的具体问题。Test-Time Training（TTT）把每个测试实例（也可以是一条测试流、一条序列、一个问题）当成一个新的学习问题：推理时先用不需要真实标签的信号更新模型的一部分状态，再做预测。这些信号可以是自监督损失、预测 entropy、示例上的 LM loss、多数投票得到的伪奖励，或者可执行 evaluator 给出的 reward。这条线反复要回答四个问题：用什么信号；更新哪些参数或状态；更新后的状态保留多久；什么时候更新有益，什么时候会 collapse。放到本项目里看，应对 sim-to-real gap 大致有三种思路：TTT 是 world model 或 policy 在部署时用真实数据**显式**修改自己；[domain randomization](../03_real2sim2real/domain-randomization/notes.md) 是在训练端预先把分布扩大；[RMA](../03_real2sim2real/rma/notes.md) 不做梯度更新，而是从历史中隐式推断环境参数。

**先分清本线里 "TTT" 的三种用法**，这是最容易混淆的地方：

| 用法 | 测试时更新什么 | 状态保留多久 | 本项目中的代表 |
|---|---|---|---|
| 分布偏移下的 TTT / TTA | 模型参数的一部分，例如共享 encoder、BN 仿射参数 | 每个样本 reset（episodic），或沿测试流累积（online / continual） | [TTT](./ttt-self-supervised/notes.md)、[Tent](./tent/notes.md)、[PAD](../05_intersections/pad/notes.md) |
| TTT 作为序列建模层（TTT layers） | 层内的 fast weights $W$；slow weights $\theta$ 全部冻结 | 每条序列都从 $W_0$ 重新开始，不跨序列积累 | [TTT layers](./ttt-layers/notes.md)、[Titans](./titans/notes.md)、[LaCT](./lact/notes.md)、[One-Minute Video](./ttt-one-minute-video/notes.md)、[TTT-E2E](./ttt-e2e/notes.md) |
| LLM 的 TTT 与 test-time RL | per-task LoRA，或整个模型、LoRA 的参数 | 每个任务、每个 benchmark 或每个问题单独训练，用完即丢 | [ARC TTT](./ttt-arc-fewshot/notes.md)、[TTRL](./ttrl/notes.md)、[TTT-Discover](./ttt-discover/notes.md) |

第二种用法里的 "test-time training"，本质上是 outer loop 学出来的 in-context learning 机制，见 [TTT layers 笔记](./ttt-layers/notes.md)"我读出来的"第 7 条。[KV-binding 一文](./ttt-kv-linear-attention/notes.md)更进一步，论证这类层在代数上就是 learned linear attention。所以读具身方向的论文时，先确认它说的是哪一种 TTT。

## 2. 发展脉络

```
                    TTT（Sun et al. 2019）：测试时用单个无标签样本的自监督损失更新参数
                                           │
        ┌──────────────────────────────────┼──────────────────────────────────┐
        ▼                                  ▼                                  ▼
 A. 分布偏移下的 TTT / TTA          B. TTT 作为序列建模层               C. LLM 的 TTT 与 test-time RL
 Tent（2020）                      TTT layers（2024）                  ARC TTT（2024）
  ├─ PAD（2020，05 交叉线）          ├─ Titans（2025）                    ├─ TTRL（2025）
  └─ MEMO / CoTTA / SAR /           ├─ One-Minute Video（2025）          └─ TTT-Discover（2026）
     TTT-MAE / video streams        ├─ LaCT（2025）
     （2021–2023，未收录）           ├─ TTT-E2E（2025）
                                    └─ KV-binding ≈ linear attention（2026，反思）
        └──────────── Test-Time Intelligence 综述（2026）：用 Update × Compute 两根轴统一 ────────────┘
```

下表先按分支、再按时间排列。每一行说明它解决了上一步的什么问题，又留下了什么问题。

| 分支 | 时间 | 工作 | 解决了上一步的什么问题 | 留下的问题 |
|---|---|---|---|---|
| A | 2019 | [TTT](./ttt-self-supervised/notes.md)（ICML 2020） | 此前的 robustness 方法在训练端固定一个 decision boundary：adversarial training 依赖预设的扰动集，UDA 要预先拿到目标域数据。TTT 在测试时用单个无标签样本的 rotation 自监督损失更新共享 extractor，并由 Theorem 1 给出充分条件：主任务与自监督任务的梯度内积为正。CIFAR-10-C Gaussian noise 上，JT 49.4 → TTT 45.6 → TTT-Online 25.8 | 训练必须改成 Y 形 joint training；自监督任务要手工挑，还可能平凡（airplane 类）；online 版假设测试流同分布；BN 下 online 版无法使用 |
| A | 2020 | [Tent](./tent/notes.md)（ICLR 2021） | 去掉了对训练流程和 proxy task 的依赖，即 fully test-time adaptation。只用预测 entropy，只更新 BN 的统计量和 $\gamma/\beta$（不到 1% 的参数）。ImageNet-C 上 59.5% → 44.0% | 依赖大 batch 和 i.i.d. 数据；会放大自信的错误（MNIST→SVHN 从 71.3% 升到 79.8%）；长时程连续适应没有测 |
| A | 2020 | [PAD](../05_intersections/pad/notes.md)（ICLR 2021，05 交叉线） | 把 TTT 搬到 vision-based RL。部署时没有 reward，就用 inverse dynamics 自监督损失每步在线更新 policy encoder，全程不 reset；36 个测试环境中 31 个有提升 | 自监督任务与控制目标的对齐仍靠手工选择（forward dynamics 在特征空间有塌缩的平凡解，所以选了 inverse） |
| A | 2021–2023 | MEMO（2110.09506）、CoTTA（2203.13591）、SAR（2302.12400）、TTT-MAE（2209.07522）、TTT on video streams（2307.05014），本项目未收录 | 依次处理：单样本 TTA；目标域持续变化时的 error accumulation 与 forgetting；wild 场景（小 batch、混合 shift）下的 collapse；用 MAE 重建代替 rotation；把 TTT 扩展到按时间到达的视频帧 | 研究对象仍以分类和分割为主，与序列建模、LLM 两条线基本隔开 |
| B | 2023–2024 | Learning to (Learn at Test Time)（2310.13807，未收录）→ [TTT layers](./ttt-layers/notes.md)（ICML 2025） | 把"测试时在自监督损失上走梯度步"做成序列建模层：RNN 的隐状态就是模型权重 $W$；自监督任务（$\theta_K/\theta_V/\theta_Q$）在 outer loop 里端到端学出；mini-batch TTT 加 dual form 让计算可以并行。长上下文、FLOPs 对齐时，perplexity 优于 Mamba | 只评测了 perplexity；小 mini-batch（$b=16$）的硬件利用率低；TTT-MLP 的 wall-clock 代价高 |
| B | 2025 | [Titans](./titans/notes.md)（v1 于 2024-12 提交） | 给 fast-weight memory 加上 momentum（past surprise）和 weight decay 形式的 forget gate，memory 改用深层 MLP；与 attention 的组合给出 MAC、MAG、MAL 三种方式 | 没有官方代码，chunk 大小等超参没有报告；deep memory 是否必要存在争议 |
| B | 2025 | [One-Minute Video](./ttt-one-minute-video/notes.md)（CVPR 2025） | 第一次把 TTT-MLP 插进预训练的 DiT：attention 只处理 3 秒片段内，TTT 层负责跨片段的全局记忆，生成 63 秒的多场景视频，人评 Elo 平均比第二名高 34 分 | 生成是 non-causal 的，也没有 action 条件；训练折合 256 张 H100 × 50 h；没有组件级消融 |
| B | 2025 | [LaCT](./lact/notes.md) | 16–64 token 的 mini-batch 让 GPU 利用率很低（H100 上 TTT-MLP 只有 31 TFLOPS）。LaCT 改用 2K–1M token 的大 chunk，纯 PyTorch 就能进入 compute-bound（515 TFLOPS），于是可以用大的 SwiGLU fast weight（最多占参数的 40%）并用 Muon 更新；覆盖 NVS、LM、AR 视频扩散 | 观测逐步到达（流式、闭环控制）时 chunk 天然很小，效率论证不再成立；检索仍明显落后 full attention |
| B | 2025 | [TTT-E2E](./ttt-e2e/notes.md) | TTT-KVB 的逐层重建损失和任务损失不一致。TTT-E2E 在测试时直接用 next-token loss 做 TTT，训练时用 meta-learning（gradients of gradients）优化"TTT 之后的 loss"；3B 模型的 loss 随上下文长度的变化与 full attention 一致，prefill 延迟保持恒定 | 压缩式记忆的召回很差（NIAH@128K 只有 0.03–0.06）；8K 上下文训练时比 full attention 慢 3.4× |
| B | 2026 | [KV-binding ≈ linear attention](./ttt-kv-linear-attention/notes.md)（ICML 2026） | 对前面几篇的反思：只要最后一层线性、无 bias，TTT-KVB（包括 MLP fast weight 和 momentum）展开后就是 learned linear attention。据此做化简（只更新最后一层，三个任务上都最好），并改写成并行形式 | 实证只覆盖 Frobenius / 点积损失；MSE 损失（Titans、TTT-MLP）下的"反记忆"证据还没有 |
| C | 2024 | [ARC TTT](./ttt-arc-fewshot/notes.md)（ICML 2025） | 把 TTT 用到 LLM 的 few-shot 上：用 demo 构造 leave-one-out 的 ICL 任务，加可逆增广，训练 per-task LoRA。8B 模型在 ARC 公开验证集上从 FT 的 18.3% 提到 47.1%，与程序合成集成后 61.9% | 增益很大一部分来自 ARC 特有的可逆变换先验；没有 compute-matched 对照 |
| C | 2025 | [TTRL](./ttrl/notes.md)（NeurIPS 2025） | 解决没有标签时怎么做 RL：用 majority voting 的结果当伪标签，给 0/1 reward 跑 GRPO，test-time scaling（投票）由此变成 test-time training 的奖励来源。AIME 2024 上 12.9 → 40.2 | 设定是 transductive 的，在同一批测试题上反复训练；多数但错误的答案会自我强化 |
| C | 2026 | [TTT-Discover](./ttt-discover/notes.md) | 面向单个问题、只需要一个最好的解：用 entropic objective（自适应 $\beta$）加 PUCT state reuse，在 Erdős minimum overlap、GPU kernel、AtCoder 上超过已知 SOTA | 依赖精确的 evaluator；单次运行，没有方差；每个 run 约 \$500 |
| 统一 | 2026 | [Test-Time Intelligence 综述](./survey-test-time-intelligence/notes.md) | 用 Update × Compute 两根轴统一 TTA、TTL（test-time learning）、TTS（test-time scaling），并给出 feedback signal、update target、temporal horizon、objective scope 四个维度的分类 | 没有统一的算力度量；没有收录 ARC TTT、LaCT、One-Minute Video，也没有讨论 KV-binding 争议 |

## 3. 技术分类（taxonomy）

有两种互相正交的切法：

- **按测试时构造的是什么学习问题**切，就是下面的子方向 A–F。
- **按综述的四个设计维度**切：feedback signal（用什么反馈）、update target $\phi$（改哪部分状态）、temporal horizon（又分成 supervision horizon 和 persistence horizon 两根独立的轴）、objective scope（是适应分布偏移，还是超越适应）。第 6 节的对比表就是沿这几个维度展开的。

本项目收录的工作链接到笔记；没有收录的给出 arXiv ID（均已通过 arXiv API 核对标题）。

**A. 分布偏移下的 TTT / TTA**
- *A1 自监督辅助任务*（训练端需要事先准备辅助头）：[TTT](./ttt-self-supervised/notes.md)（rotation）；TTT-MAE（2209.07522，MAE 重建）；[PAD](../05_intersections/pad/notes.md)（inverse dynamics，用于 RL policy）。
- *A2 fully test-time adaptation：entropy 与置信度*（训练不变）：[Tent](./tent/notes.md)；MEMO（2110.09506，单样本加增广）；SAR（2302.12400，wild 场景下的稳定化）。
- *A3 continual / online*：TTT-Online（见 [TTT 笔记](./ttt-self-supervised/notes.md)）；CoTTA（2203.13591）；TTT on video streams（2307.05014）。
- 经典 TTA 综述：2303.15361。

**B. TTT 作为序列建模层**（fast weights；inner loop 更新 $W$，outer loop 学 slow weights）
- *B1 TTT-KVB*（inner loss 是 key→value 的重建或绑定损失）：[TTT layers](./ttt-layers/notes.md)、[Titans](./titans/notes.md)、[LaCT](./lact/notes.md)、[One-Minute Video](./ttt-one-minute-video/notes.md)；MesaNet（2506.05233）。
- *B2 端到端的 inner loss*（inner loop 直接优化 next-token loss）：[TTT-E2E](./ttt-e2e/notes.md)。In-Place TTT（2604.06169）把现有 LLM 中 MLP 的最终投影矩阵当作 fast weights，目标换成与 next-token prediction 对齐的损失。
- *B3 统一视角与理论*：[KV-binding ≈ linear attention](./ttt-kv-linear-attention/notes.md)；Test-time regression（2501.12352）；MIRAS（2504.13173）；Nested Learning（2512.24695）。
- *B4 视觉、视频与 3D*：[One-Minute Video](./ttt-one-minute-video/notes.md)、[LaCT](./lact/notes.md)（NVS 与 AR 视频扩散）；TTT3R（2509.26645）；tttLRM（2602.20160）。

**C. LLM 的测试时训练**（参数级，按任务或按上下文）
- per-task 或 per-instance 微调：[ARC TTT](./ttt-arc-fewshot/notes.md)；TTT on nearest neighbors（2305.18466）。
- 把长上下文写进权重：[TTT-E2E](./ttt-e2e/notes.md)（它同时属于 B2）；In-Place TTT（2604.06169）。
- 模型自己生成更新数据：SEAL（2506.10943）。
- 理论：Specialization after Generalization（2509.24510）。

**D. Test-time RL**（综述里 TTL 与 TTS 的交叉区）
- consensus reward：[TTRL](./ttrl/notes.md)；TEMPO（2604.19295）。
- task-grounded reward（可执行的 evaluator）：[TTT-Discover](./ttt-discover/notes.md)。
- 对照组，只加推理算力、不改参数的 TTS：Scaling LLM Test-Time Compute Optimally（2408.03314）。

**E. 机器人与 world model 中的 TTT**（与 05 交叉线对接）
- 部署时用自监督损失更新 policy：[PAD](../05_intersections/pad/notes.md)。
- 部署时用真实转移更新 world model：[SimDist](../05_intersections/simdist/notes.md)（只 finetune latent dynamics，本质是 supervised system identification）；WorldAgen（2609.08162，world model head 与 agent head 共享 backbone，测试时采集转移、更新 world model）。
- 以测试帧为起点、在 world model 里做 per-instance RL：[World-Gymnast](../05_intersections/world-gymnast/notes.md)。它说的 TTT 和 Sun et al. 的自监督 TTT 不是一回事，见该笔记。
- VLA 的 test-time RL：EVOLVE-VLA（2512.14666）、TT-VLA（2601.06748）。
- World action model 的 TTT：WAM-TTT（2607.06988）。
- 对照：[RMA](../03_real2sim2real/rma/notes.md) 不做梯度更新，靠隐式的在线 system identification，按综述的判据不算 TTL（笔记按判据做的归类，综述没有讨论 RMA）；Learning to Adapt via Meta-RL（1803.11347）用 meta-learning 训练 dynamics model，使它能在线快速适应。

**F. 综述**：[Test-Time Intelligence 综述](./survey-test-time-intelligence/notes.md)（2609.01679）；经典 TTA 综述（2303.15361）。

## 4. 论文总表

"阅读顺序"是第 5 节学习路线里的顺序。综述建议读两遍：第一遍只读 Sec.2 和 Sec.3.1，最后再回读 Sec.5、6.4、7。

| 阅读顺序 | 论文 | 年份 | 层级 | 一句话 | 代码 | 笔记 |
|---|---|---|---|---|---|---|
| 0 / 12 | A Survey on Self-Improving Test-Time Intelligence（2609.01679） | 2026（Machine Intelligence Research） | survey | 用 Update（是否改写可写状态）和 Compute（额外推理算力）两根轴统一 TTA、TTL、TTS；给出两条边界判据：KV cache 和 RNN hidden state 的演化不算 TTL，只在离线 pipeline 里生效的算 training-time amortization | 无；paper list：mr-eggplant/awesome_test_time_intelligence | [notes](./survey-test-time-intelligence/notes.md) |
| 1 | Test-Time Training with Self-Supervision（1909.13231） | 2019（ICML 2020） | foundational | 监督任务与 rotation prediction 共享 extractor，做 joint training；测试时先在单个样本的自监督损失上更新 extractor，再预测；梯度内积为正时这一步有益 | [ttt_cifar_release](https://github.com/yueatsprograms/ttt_cifar_release)、[ttt_imagenet_release](https://github.com/yueatsprograms/ttt_imagenet_release) | [notes](./ttt-self-supervised/notes.md) |
| 2 | Tent（2006.10726） | 2020（ICLR 2021） | foundational | 不用 source 数据，也不改训练，测试时最小化预测 entropy，只更新 BN 统计量和 $\gamma/\beta$；更新全部参数反而从未超过 source 模型 | [DequanWang/tent](https://github.com/DequanWang/tent)（已 archived） | [notes](./tent/notes.md) |
| 3 | Learning to (Learn at Test Time): RNNs with Expressive Hidden States（2407.04620） | 2024（ICML 2025） | foundational | 隐状态就是模型 $f$ 的权重 $W$，每个 token 在可学习的多视角重建损失上走一步 GD；这样得到 TTT-Linear 和 TTT-MLP，线性、batch GD、$W_0=0$ 时等价于 linear attention | [ttt-lm-jax](https://github.com/test-time-training/ttt-lm-jax)（训练）、[ttt-lm-pytorch](https://github.com/test-time-training/ttt-lm-pytorch)（教程） | [notes](./ttt-layers/notes.md) |
| 4 | Titans（2501.00663） | 2025 | milestone | 用带 momentum 和 forget gate 的 surprise 梯度，在测试时写入深层 MLP 长时记忆，再与 attention（短时记忆）和 persistent memory 组合成 MAC、MAG、MAL | 无官方代码；社区版 lucidrains/titans-pytorch | [notes](./titans/notes.md) |
| 5 | One-Minute Video Generation with TTT（2504.05298） | 2025（CVPR 2025） | milestone | 在 CogVideo-X 5B 中插入带门控、双向的 TTT-MLP 层：attention 只在 3 秒片段内做，TTT 层负责全局记忆；能生成 63 秒多场景视频 | [ttt-video-dit](https://github.com/test-time-training/ttt-video-dit)（TTT-MLP 训练只支持 H100） | [notes](./ttt-one-minute-video/notes.md) |
| 6 | LaCT: Test-Time Training Done Right（2505.23884） | 2025（ICLR 2026） | milestone | 把 fast weight 的更新单位改成 2K–1M token 的大 chunk，用来换取硬件利用率、大的非线性 state 和 Muon；chunk 内的局部结构交给 window attention | [a1600012888/LaCT](https://github.com/a1600012888/LaCT) | [notes](./lact/notes.md) |
| 7 | End-to-End Test-Time Training for Long Context（2512.23675） | 2025（arXiv） | frontier | 滑窗 Transformer 在测试时对上下文做 next-token 梯度下降，把上下文压进后 1/4 个 block 的 MLP；训练时用 meta-learning 学初始化 | [test-time-training/e2e](https://github.com/test-time-training/e2e) | [notes](./ttt-e2e/notes.md) |
| 8 | TTT with KV Binding Is Secretly Linear Attention（2602.21204） | 2026（ICML 2026） | frontier | 证明最后一层线性、无 bias 的 TTT-KVB 都能展开成 learned linear attention；据此解释几个"反记忆"现象，把模型化简到只更新最后一层，并改写成并行形式 | [nv-tlabs/tttla](https://github.com/nv-tlabs/tttla) | [notes](./ttt-kv-linear-attention/notes.md) |
| 9 | The Surprising Effectiveness of TTT for Few-Shot Learning（2411.07279） | 2024（ICML 2025） | milestone | 对每个任务用 leave-one-out 的 ICL 格式加可逆增广训练 LoRA，再配合"变换-逆变换"推理和层级投票；在 ARC 和 BBH 上都有明显提升 | [ekinakyurek/marc](https://github.com/ekinakyurek/marc)、[adamzweiger/Fewshot-TTT](https://github.com/adamzweiger/Fewshot-TTT) | [notes](./ttt-arc-fewshot/notes.md) |
| 10 | TTRL: Test-Time Reinforcement Learning（2504.16084） | 2025（NeurIPS 2025） | milestone | 在无标签测试题上，用 majority voting 当伪标签给 0/1 reward 跑 GRPO；伪标签错了也能训，靠的是 "Lucky Hit" | [PRIME-RL/TTRL](https://github.com/PRIME-RL/TTRL) | [notes](./ttrl/notes.md) |
| 11 | Learning to Discover at Test Time（2601.16175） | 2026（ICML 2026） | frontier | 针对单个问题在测试时用 RL 训练 LLM：entropic objective 让优化偏向最大值，PUCT 负责选起点；最后返回 reward 最高的解，而不是 policy | [test-time-training/discover](https://github.com/test-time-training/discover)（需要 Tinker API） | [notes](./ttt-discover/notes.md) |

## 5. 推荐学习路线

整体顺序是：先建坐标系，再读分布偏移 TTT 打好概念基础，然后读序列层分支（它和 world model 的长程记忆关系最近），再读 LLM 与 test-time RL（和 04 RL 线接上），最后回到具身交叉。阶段 2 和阶段 3 相互独立，可以并行。

### 阶段 0：建立坐标系

- **前置知识 checklist**
  - [ ] covariate shift 与 label shift 的定义
  - [ ] "训练—部署"两阶段的范式，知道 TTA、TTL、TTS 各指什么
  - [ ] （第 1 题用）推理时会自动演化的两类状态：RNN / RSSM 的 recurrent state（RSSM 见 01 线的 [Dreamer](../01_world_model/dreamer/notes.md)），以及 Transformer 的 KV cache
- **要读的笔记**：[综述](./survey-test-time-intelligence/notes.md)，只读"问题设定"和"核心方法"里 0–1 节（Update × Compute 平面、四个设计维度）
- **读完应能回答**
  1. 按综述 Sec.2.1 的判据，为什么 TTT-Layer 的 $W_t$ 更新算 TTL，而 RSSM 的 recurrent state 和 KV cache 不算？
  2. supervision horizon 和 persistence horizon 为什么是两根独立的轴？各举一个"单样本监督、跨流保留"和"batch 监督、episodic reset"的例子。
  3. 为什么 STaR / ReST^EM 不算 TTI，TTRL 却算？

### 阶段 1：分布偏移下的 TTT / TTA

- **前置知识 checklist**
  - [ ] multi-task learning 与共享 encoder（hard parameter sharing）
  - [ ] self-supervised pretext task（rotation prediction）
  - [ ] BatchNorm 与 GroupNorm：train/eval 统计量、仿射参数 $\gamma/\beta$，以及小 batch 为什么让 BN 失效
  - [ ] softmax、Shannon entropy，以及 entropy 对 logits 的梯度
  - [ ] 凸优化的 $\beta$-smooth descent lemma
  - [ ] CIFAR-10-C / ImageNet-C 协议：15 种 corruption × 5 个 severity
  - [ ] online、offline、episodic 三种适应协议
  - [ ] （读 PAD 用）视觉 RL 的基本配方与 [SAC](../04_rl/sac/notes.md)；inverse dynamics 与 forward dynamics 的区别，以及在特征空间做 forward 预测为什么有塌缩的平凡解
- **要读的笔记**：[TTT](./ttt-self-supervised/notes.md) → [Tent](./tent/notes.md) → 跨主题的 [PAD](../05_intersections/pad/notes.md)
- **读完应能回答**
  1. 在两层线性 toy model 里推出 $\mathrm{sign}(\eta^*)=\mathrm{sign}\langle\nabla l_m,\nabla l_s\rangle$。为什么 Theorem 1 的条件在测试时无法检验？
  2. 推导 $\partial H/\partial z_j=-p_j(\log p_j+H)$，用它解释 Tent 为什么会放大"自信的错误"（MNIST→SVHN 71.3% → 79.8%）。
  3. TTT 为什么必须改训练，Tent 为什么不用？两者各自付出了什么代价？
  4. batch size 降到 1、数据是时间相关的单帧流（机器人部署就是这样）时，Tent 的 normalization 和 transformation 两步分别会出什么问题？
  5. PAD 为什么选 inverse dynamics，而不选特征空间里的 forward dynamics 作为自监督任务？

### 阶段 2：TTT 作为序列建模层

- **前置知识 checklist**
  - [ ] self-attention、KV cache，以及 linear attention 的递归形式（$S_t=S_{t-1}+v_tk_t^\top$）
  - [ ] RNN/SSM、Mamba 2；DeltaNet 的 delta rule；Gated DeltaNet 的 forget gate
  - [ ] fast weights / fast weight programmers
  - [ ] meta-learning 与 bi-level optimization：inner loop / outer loop、gradients of gradients（MAML）
  - [ ] online、batch、mini-batch GD，以及 cumsum 形式的 GD
  - [ ] sliding-window attention；Roofline 模型（memory-bound 与 compute-bound）
  - [ ] Muon（Newton–Schulz 正交化）、带 momentum 的 SGD 展开
  - [ ] 评测：perplexity、RULER S-NIAH
  - [ ] （读 One-Minute Video 用）Diffusion Transformer（DiT）与 latent video diffusion 的基本结构，CogVideo-X 就是这一类
  - [ ] （第 6 题用）动作条件、逐帧自回归的 world model 是什么，见 01 线的 [Dreamer](../01_world_model/dreamer/notes.md)、[DIAMOND](../01_world_model/diamond/notes.md)
- **要读的笔记**：[TTT layers](./ttt-layers/notes.md) → [Titans](./titans/notes.md) → [One-Minute Video](./ttt-one-minute-video/notes.md) → [LaCT](./lact/notes.md) → [TTT-E2E](./ttt-e2e/notes.md) → [KV-binding ≈ linear attention](./ttt-kv-linear-attention/notes.md)。最后一篇要在读完前面几篇之后批判性地读，它用的术语 TTT-KVB 来自 TTT-E2E。
- **读完应能回答**
  1. 线性 $f$、batch GD、$W_0=0$、$\eta=1/2$ 时，为什么 TTT layer 等价于 linear attention？换成 mini-batch GD 后，是哪一项打破了等价，这一项为什么带来更强的表达力？
  2. 在 Titans 的 memory 更新中，$\eta_t$（momentum）和 $\alpha_t$（forget）分别在处理什么问题？为什么 $\eta_t=\alpha_t=0$ 时它就退化为 delta rule？
  3. 用 LaCT 的 Eq.3 说明 $b=16$ 为什么是 memory-bound。为什么 LaCT 在 LM 上要求 window ≥ chunk？
  4. TTT-E2E 和 TTT-naive（dynamic evaluation）的训练目标差在哪？$t<1$K、还没做任何 TTT 步时，TTT-E2E 的 loss 为什么已经低于 full attention？
  5. KV-binding 一文的梯度上升实验在 Frobenius 损失下为什么几乎是必然结果（见该笔记"笔者的批判性阅读"第 1 条）？要真正区分"记忆"和"linear attention"两种解释，应该做什么实验？
  6. One-Minute Video 的 non-causal mini-batch TTT 和双向扫描，搬到逐帧、带 action 条件的自回归 world model 上会出什么问题？

### 阶段 3：LLM 的 TTT 与 test-time RL

- **前置知识 checklist**
  - [ ] in-context learning；因果 LM 只在输出 token 上算 loss 的 masking
  - [ ] LoRA / QLoRA
  - [ ] self-consistency / majority voting；pass@k、maj@k、avg@k
  - [ ] policy gradient、PPO、GRPO 的组内 advantage 归一化；RLVR 与 rule-based reward（见 04 线的 [PPO](../04_rl/ppo/notes.md)、[DeepSeek-R1](../04_rl/deepseek-r1/notes.md)）
  - [ ] 指数倾斜 / log-sum-exp、$\mathrm{KL}(q\|u)=\log N-H(q)$
  - [ ] MCTS / PUCT（AlphaZero）
- **要读的笔记**：[ARC TTT](./ttt-arc-fewshot/notes.md) → [TTRL](./ttrl/notes.md) → [TTT-Discover](./ttt-discover/notes.md)
- **读完应能回答**
  1. Direct I/O 和 leave-one-out ICL 两种训练样本各长什么样？为什么在 ARC（18 对 29 题）和 BBH（51.5 对 57.8）上 Direct I/O 都更差？
  2. 标签错误时，TTRL 的奖励准确率为什么等于 $1-(m+g)$？它为什么会随训练下降？
  3. GRPO 的组内归一化下，二元 majority reward 的正负 advantage 怎样随 $m$ 变化？$m\to1$ 时会发生什么？
  4. 在 TTT-Discover 的 Table 8 里，没有 reuse 时 entropic 训练（5274.03 μs）和 Best-of-N（5352.36 μs）几乎一样。reuse 解决了哪个问题，entropic objective 又解决了哪个？
  5. TTT、TTRL、TTT-Discover 分别为每个测试实例构造了什么学习问题？最终产出的是参数、预测还是解？

### 阶段 4：回到具身，把本线接到 world model / real2sim2real / RL

- **前置知识 checklist**
  - [ ] RSSM / latent dynamics、MPC / MPPI（见 01 线的 [DreamerV3](../01_world_model/dreamerv3/notes.md)、[TD-MPC2](../01_world_model/tdmpc2/notes.md)）
  - [ ] RMA 的 adaptation module（隐式 system identification）
  - [ ] sim-to-real gap 的分类（见 [reality gap 综述](../03_real2sim2real/survey-reality-gap/notes.md)）
- **要读的笔记**：[PAD](../05_intersections/pad/notes.md)（如果阶段 1 已读，这里回看）→ [RMA](../03_real2sim2real/rma/notes.md)（作对照）→ [SimDist](../05_intersections/simdist/notes.md) → [World-Gymnast](../05_intersections/world-gymnast/notes.md) → 回读 [综述](./survey-test-time-intelligence/notes.md) 的 Sec.3.2.5、Sec.6.4、Sec.7
- **读完应能回答**
  1. PAD、SimDist、RMA、World-Gymnast 这四种"部署时适应"，按综述的 update target 和 persistence horizon 分别怎么归类？哪一个按综述定义不算 TTL？
  2. 如果用 world model 的转移预测误差作为测试时的自监督损失，TTT 的 Theorem 1 提示要先检验什么（梯度相关性）？SimDist 为什么只 finetune dynamics、不解冻 encoder？
  3. TTRL 的 majority voting 和 TTT-Discover 的精确 evaluator，在连续动作、learned world model 做 evaluator 的场景下分别会失效在哪里？
  4. 把 TTT-KVB 层放进 policy 来"适应 sim-to-real gap"，按 KV-binding 一文的视角，它和 RMA 的差别还剩多少？

## 6. 关键机制对比

本线的方法分成两大类，适合比较的维度不一样，所以拆成两张表。

### 6.1 部署时更新模型参数的方法（分支 A、C 和具身交叉）

| 方法 | 测试时构造的学习问题 | 更新对象 $\phi$ | 测试时信号（是否需要标签 / reward） | 训练端要改吗 | persistence | 额外代价 | 典型失败或适用边界 |
|---|---|---|---|---|---|---|---|
| [TTT](./ttt-self-supervised/notes.md)（standard） | 单个样本 | 共享 extractor $\theta_e$ | rotation 自监督损失；不需要标签 | 要，Y 形 joint training | 每个样本 reset；10 步 SGD | 约为普通推理的 2×batch_size×iterations 倍 | 自监督任务平凡或与主任务不相关（airplane 类）；需要 GN |
| TTT-Online | 一条测试流 | 同上 | 同上 | 要 | 沿流累积，每个样本 1 步 | 每个样本一次反向 | 假设整条流同分布；BN 下不可用 |
| [Tent](./tent/notes.md) | 一个 batch / 一条流 | BN 的 $\mu,\sigma$（由数据估计）和 $\gamma,\beta$（用梯度更新），不到 1% 的参数 | 预测 entropy；不需要标签 | 不改 | online 累积，也可 episodic | 每个测试点多一次反向 | 大 shift 下放大自信的错误；依赖大 batch 和 i.i.d.；输出必须是 categorical 分布 |
| [PAD](../05_intersections/pad/notes.md) | 一个部署 episode | policy encoder | inverse dynamics 自监督；不需要 reward | 要，RL 与 IDM 做 joint training | 每步 1 次更新，episode 内不 reset | 每步一次反向（32 个 crop 副本） | 自监督任务要手工挑；forward dynamics 在特征空间会塌缩 |
| [SimDist](../05_intersections/simdist/notes.md) | 真机上的少量交互 | 只更新 latent dynamics，其余冻结 | 真实转移的监督损失（supervised system identification） | 要，在仿真中大规模预训练 TD-MPC 式 world model | 跨 episode 累积（按轮 finetune） | 四足约 32–36 分钟真实数据 | 要求 gap 主要体现在动力学上；encoder 失效时没有退路 |
| [ARC TTT](./ttt-arc-fewshot/notes.md) | 单个任务的 K 个 demo | per-task LoRA | demo 上 leave-one-out 的 LM loss（有监督，但只用测试任务自带的示例） | ARC 上需要先在 ReARC 上 FT；BBH 不需要 | 每个任务训一个、用完即丢 | ARC 100 题约 12 h / 1×A100 | 增益依赖可逆增广先验；需要逐步计算的任务会变差（Boolean Expressions 85.7 → 80.1） |
| [TTRL](./ttrl/notes.md) | 整个无标签 benchmark（transductive） | LLM 参数（GRPO） | majority voting 伪标签给出 0/1 reward | 不改 | benchmark 级别，多个 episode | 每题 64 条 rollout × 10–80 个 episode；8×A100 | 多数但错误的答案会自我强化；先验不足时失败；需要能判等的离散答案 |
| [TTT-Discover](./ttt-discover/notes.md) | 单个问题（MDP：state 是候选解） | LoRA（gpt-oss-120b） | 连续、可验证的 reward，配合 entropic objective | 不改 | 每个问题 50 步 × 512 rollout；产出最好的解 | 约 \$500/run，另加 evaluator 成本 | 依赖精确的 evaluator，reward hacking 靠人工约束；没有 reuse 时与 Best-of-N 相当 |
| [RMA](../03_real2sim2real/rma/notes.md)（对照） | 最近 0.5 s 的 state-action 历史 | 无，参数不更新 | adaptation module 前向回归 extrinsics | 要，仿真中两阶段训练 | 靠历史窗口，不写入参数 | 只有一次前向 | 按综述定义属于"普通推理态"，不算 TTL；能适应的范围受仿真随机化覆盖面限制 |

### 6.2 TTT 作为序列建模层（分支 B）

这一类方法测试时都冻结 slow weights $\theta$，每条序列都从 $W_0$ 重新开始。

| 方法 | fast weight $f_W$ | inner loss | 更新粒度 | 状态管理 | outer loop 学什么 | 关键证据 | 代价或短板 |
|---|---|---|---|---|---|---|---|
| [TTT layers](./ttt-layers/notes.md) | TTT-Linear（方阵 $W$）/ TTT-MLP（两层） | 多视角重建 ‖f(θ_K x; W) − θ_V x‖²（KVB） | mini-batch $b=16$，dual form | LN + residual，learnable $W_0$ 与 $\eta(x)$，没有显式遗忘 | $\theta_K,\theta_V,\theta_Q,W_0$ 和 lr 头 | 125M 模型上 mini-batch（$b$ 从 T 改为 16）让 ppl 下降 1.70（Table 1 中贡献最大的一步）；32k 上下文下优于 Mamba | 只评测 perplexity；TTT-MLP 的 decode 明显更慢 |
| [Titans](./titans/notes.md) | 深层 MLP memory（$L_\mathcal M\ge1$） | MSE ‖M(k) − v‖²（KVB） | chunk 内在 chunk 起点处求梯度，momentum 用 parallel scan | momentum $\eta_t$、forget gate $\alpha_t$、写入强度 $\theta_t$ 都随 token 变化 | 同上，加上各个 gate，再与 attention 组合成 MAC、MAG、MAL | S-NIAH-W@16K：LMM 80.6、MAC 95.2，TTT 为 0.0；去掉 deep memory 后长上下文从 92.68 降到 85.34 | 没有官方代码；$b$ 等超参没有报告；deep memory 越深吞吐越低 |
| [One-Minute Video](./ttt-one-minute-video/notes.md) | TTT-MLP | 同 TTT layers | non-causal mini-batch $b=64$，双向扫描 | tanh 门控，$\alpha$ 初始化为 0.1 | 在 CogVideo-X 5B 上分 5 个阶段微调 | 63 秒视频的 Elo：1033 对 SWA 999；18 秒时 Gated DeltaNet 反而最好（1032 对 1004） | 推理是 local attention 的 2.5×，训练是 3.8×；non-causal，没有 action 条件 |
| [LaCT](./lact/notes.md) | SwiGLU，最多占参数的 40% | 负点积 $-f_W(k)^\top v$ | 大 chunk，2K–1M token | L2 weight norm + Muon（momentum 可选） | QKV 投影、lr 头、初始 fast weights | H100 上 515 TFLOPS，原版 TTT-MLP 为 31；state 越大越好 | 3B 模型 S-NIAH@32K 为 48.0，full attention 为 91.2；线性 LaCT 在 LM 上输给 GLA-SWA |
| [TTT-E2E](./ttt-e2e/notes.md) | 后 1/4 个 block 的 MLP，外加一个静态 MLP | 网络末端的 next-token CE（端到端） | $b=1$K，配 SWA $k=8$K（要求 $k\ge b$） | 静态 MLP 保存预训练知识，减轻遗忘 | 通过 gradients of gradients 学 $W_0$ 在内的全部参数 | 3B 模型在 8K–128K 上的 loss 与 full attention 同步；128K 时 prefill 快 2.7× | NIAH@128K 只有 0.03–0.06；8K 训练时慢 3.4× |
| [KV-binding 视角](./ttt-kv-linear-attention/notes.md) | 分析对象：最后一层线性、无 bias 的 TTT-KVB | Frobenius / 点积（实验只覆盖这一种） | — | — | — | 只更新最后一层（V1）在三个任务上都最好；并行形式使 TTT 层吞吐最多提高约 4×，端到端训练加速 1.19× | 不覆盖 MSE 损失和 TTT-E2E；没有 seed 方差 |

**笔者的一句话总结**：6.1 里的方法真正让模型在部署后变了，风险在于信号是否可靠、状态会不会漂移；6.2 里的方法测试时只改写序列内部的 fast weights，风险在于它到底比 linear attention 多学到了什么。

## 7. 开放问题与争议

1. **TTT 层到底是在"学习"，还是 learned linear attention？** [KV-binding 一文](./ttt-kv-linear-attention/notes.md)列出了几条证据：inner loop 改成梯度上升并重训后指标基本持平（有时略好）；推理时多走 inner steps 反而变差；只更新最后一层效果最好。它据此把 [LaCT](./lact/notes.md)、[Titans](./titans/notes.md)、[One-Minute Video](./ttt-one-minute-video/notes.md) 背后"更深的 fast weight = 更强记忆"的叙事重新解释为 linear attention。但笔记指出，在 Frobenius 损失下梯度上升与梯度下降几乎必然等价，这组实验区分不了两种解释；MSE 损失（Titans、TTT-MLP）下的情况还没有检验；[TTT-E2E](./ttt-e2e/notes.md) 用端到端 loss，也不在定理覆盖范围内。这个争议还带出一个定义问题：[综述](./survey-test-time-intelligence/notes.md)把 TTT-Layer 的 $W_t$ 算作 TTL，却把普通 hidden state 的演化排除在外；可如果 TTT-KVB 在代数上等价于 linear attention 的状态演化，这条边界就变得模糊了（笔者观察）。
2. **压缩与精确召回不可兼得。** TTT-E2E 在 loss 上随上下文长度追平了 full attention，NIAH@128K 却只有 0.03–0.06；LaCT 3B 的 S-NIAH@32K 为 48.0，full attention 为 91.2。Titans 的 S-NIAH 很高，但只测到 16K，而且没有官方代码可以核对。world model 的长程一致性需要哪一种记忆，目前没有实验回答。
3. **更新粒度与在线延迟。** [TTT layers](./ttt-layers/notes.md) 显示 $b$ 越小效果越好；LaCT 为了硬件效率改用大 chunk；TTT-E2E 用 $b=1$K 加 SWA 折中。机器人闭环控制时观测一步一步到达，$b$ 天然很小，LaCT 的效率论证就不成立了；One-Minute Video 还是 non-causal 的。在流式、因果的场景下怎样选粒度，还是空白。
4. **长时程部署下的稳定性与 collapse。** Tent 没有测连续适应；TTT-Online 所说的"forgetting 无害"前提是整条流同分布；PAD 把 episode 拉长到 10 倍，7 个任务中 6 个仍有提升，但只看了单个 episode 内部。[TTRL](./ttrl/notes.md) 的 Lucky Hit 会随训练自我削弱：奖励准确率 $1-(m+g)$ 随 majority ratio 上升而下降。综述转述了 entropy minimization 长期运行会"先涨后跌"，却没有讨论物理机器人上不可回滚的更新该怎么监控、怎么回滚。
5. **测试时信号怎么选、怎么对齐。** TTT 的 Theorem 1 要求梯度内积为正，但这个条件依赖未知的 $y$，测试时无法检验。Tent 的 entropy 在连续动作 policy 上没有良好定义，而且与 [SAC](../04_rl/sac/notes.md) 的 maximum entropy 目标、[PPO](../04_rl/ppo/notes.md) 的 entropy bonus 方向相反。PAD 放弃 forward dynamics 是为了避开 representation collapse。TTRL 的"多数"在连续动作上没有定义。ARC TTT 的增益有一半以上依赖可逆增广（去掉增广少解 16 题）。对 world model 来说，转移预测误差是最自然的信号，但它与控制目标是否对齐，要先测。
6. **算力口径不统一，缺少公平对照。** 综述承认 Compute 轴只是示意，没有统一度量。[ARC TTT](./ttt-arc-fewshot/notes.md) 没有和同等 FLOPs 下的多采样做比较。[TTT-Discover](./ttt-discover/notes.md) 的消融显示，没有 state reuse 时再好的训练目标也只相当于 Best-of-N，说明"学习"要在搜索结构之上才起作用。TTRL 每题 64 条 rollout、跑几十个 episode，成本远高于推理时投票。什么时候该更新参数、什么时候该只加推理算力，还没有公认的比较协议。
7. **评测口径与泄漏。** TTRL 在同一批测试题上反复训练，checkpoint 的选法也不透明；ARC TTT 的超参就在做消融的那 80 题上调；Tent 没有说明在无标签条件下怎么选超参；TTT-Discover 只跑了一次，prompt 里还带着前 SOTA 的方法描述；One-Minute Video 的 Elo 没有置信区间；KV-binding 一文不到 0.5 ppl 的差距没有报告 seed 方差。
8. **向具身迁移的具体障碍。** (a) 本线的视频 TTT 都不是可交互的：One-Minute Video 没有 action 条件，LaCT 的 AR 视频也没有测 action 条件。(b) 序列层 TTT 每条序列都 reset，部署期需要的是跨 episode 的持续适应，Titans 的 memory 能否跨序列保留，没有被测过。(c) TTT-Discover 的 entropic objective 会放大 evaluator 的漏洞；如果把 evaluator 换成 learned world model，就会直接变成 model exploitation，[World-Gymnast](../05_intersections/world-gymnast/notes.md) 的 VLM reward 已经面对这个问题。(d) 在 sim-to-real 上，显式的梯度式 TTL（PAD、SimDist）和隐式的 in-context 适应（RMA、TTT-KVB 层）究竟谁更好，综述没有给出对比。按综述"普通推理态演化不算 TTL"的判据，RMA 这类靠 hidden state 的适应被排除在 TTL 之外（笔记按判据做的归类），而 TTT-KVB 层的 $W_t$ 更新却被算作 TTL，这正是第 1 条说的定义张力。

## 8. 扩展阅读

下面 15 篇是从 scouting 阶段未收录的 notable 列表中挑出的，arXiv ID 已通过 arXiv API 核对标题。

| arXiv | 论文 | 为什么值得读 |
|---|---|---|
| 2209.07522 | Test-Time Training with Masked Autoencoders（TTT-MAE） | 用 MAE 重建代替 rotation 做单样本 TTT，并从 bias-variance 角度刻画收益。它是"自监督任务怎么选"的直接续篇，也可以作为用 world model 重建损失当 $l_s$ 的参照 |
| 2307.05014 | Test-Time Training on Video Streams | online TTT 从上一帧的模型出发，在当前帧和一个小窗口上训练，结果反而优于能看到整段视频的 offline 版，作者用 locality 来解释。这是机器人视频流上做在线适应最直接的前例 |
| 2203.13591 | Continual Test-Time Domain Adaptation（CoTTA） | 针对目标域持续变化：用权重平均和增广平均得到的伪标签减少 error accumulation，用 stochastic restore 缓解 forgetting。它补上了 Tent 没测的长时程场景 |
| 2302.12400 | Towards Stable Test-Time Adaptation in Dynamic Wild World（SAR） | 分析混合 shift、小 batch、在线 label shift 下 TTA 为什么不稳定：BN 是关键因素，大梯度的噪声样本会导致 collapse。这正是机器人时间相关观测流会遇到的情形 |
| 2310.13807 | Learning to (Learn at Test Time) | TTT layers 的前身，先提出了 inner loop / outer loop 的双层优化视角。读 TTT layers 时回看，能看清"自监督任务由 outer loop 学出"是怎么来的 |
| 2501.12352 | Test-time regression | 把 associative recall 形式化为 memorization（回归）加 retrieval。linear attention、SSM、fast weight programmer、softmax attention 都可以由回归权重、regressor 函数类、测试时优化算法三个选择得出。读 KV-binding 争议之前，先用它打理论底座 |
| 2504.13173 | It's All Connected（MIRAS） | Titans 作者提出的统一框架：序列模型都是用内部目标（attentional bias）学 key→value 映射的 associative memory，遗忘被看作 retention regularization。可以和 KV-binding 一文的"linear attention"解释对照 |
| 2604.06169 | In-Place Test-Time Training | 把现有 LLM 中 MLP 的最终投影矩阵当作 fast weights，目标改成与 next-token prediction 对齐，不需要从头训练就能给 LLM 加上 TTT。它对应 TTT-E2E 留下的"能否从预训练 Transformer 初始化"这个问题 |
| 2506.10943 | Self-Adapting Language Models（SEAL） | 模型自己生成微调数据和更新指令（self-edit），用 SFT 做持久的权重更新，再用 RL 训练生成 self-edit 的能力。综述把它列为 joint learning-and-scaling 的代表 |
| 2408.03314 | Scaling LLM Test-Time Compute Optimally | 不改参数、只加推理算力的 TTS 对照组：最优策略取决于题目难度，算力匹配时小模型有时能胜过大模型。判断"更新参数还是加算力"时，要以它为基线 |
| 2509.24510 | Specialization after Generalization | 对 TTT 为什么在 in-distribution 数据上也有效给出理论解释：基础模型整体仍然 underparameterized，TTT 把容量集中到与测试任务相关的概念上。作者在 linear representation hypothesis 下建模，并用 SAE 验证了关键假设 |
| 2512.14666 | EVOLVE-VLA | VLA 的 test-time training：用学出来的 progress estimator 提供 dense feedback，替代测试时拿不到的 oracle reward，并专门设计了抑制这个信号噪声的机制。它是 TTRL 伪奖励问题在 VLA 上的对应 |
| 2601.06748 | On-the-Fly VLA Adaptation via Test-Time RL（TT-VLA） | 部署时对 VLA 做 test-time RL，用逐步的任务进度作 dense reward，同时保留 SFT/RL 阶段的先验。综述在 external feedback 和 policy adaptation 两处都引用了它 |
| 2607.06988 | WAM-TTT | 通过自监督 video prediction，把无标签的人类视频写进冻结的 world action model 里的轻量 adaptive memory；meta-training 阶段用人机配对数据和 KV memory reconstruction 目标做对齐。它同时涉及 TTT-KVB 式记忆、world action model 和部署期适应，是离本项目交叉点最近的工作（很新，暂无代码） |
| 1803.11347 | Learning to Adapt in Dynamic, Real-World Environments through Meta-RL | 用 meta-learning 训练 dynamics model，使它部署时能用最近的少量转移快速在线适应。思路与 TTT-E2E 的"meta-learn 适合 TTT 的初始化"相同，只是对象换成了 world model 的 dynamics，可以作为 SimDist 的前身对照阅读 |

同样与本线相关、但篇幅所限没有列进表格的：MEMO（2110.09506）、TEMPO（2604.19295）、WorldAgen（2609.08162）、TTT3R（2509.26645）、tttLRM（2602.20160）、MesaNet（2506.05233）、Nested Learning（2512.24695）、经典 TTA 综述（2303.15361）。

## 9. 复现推荐

按 research-workflow "先复现再提假设"的原则，推荐先复现下面三个 baseline，每条分支各一个。复现报告放在项目根目录的 `复现/` 下。算力数字注明了出处：论文没有给的，标为笔记作者或笔者的估计。

### 9.1 TTT + Tent on CIFAR-10-C（分支 A 的基准）

- **理由**：这两篇代表了两种最基本的测试时信号（自监督辅助损失 vs 预测 entropy），也代表了两种更新对象（共享 extractor vs BN 仿射参数）。两者都有官方代码，都能单卡跑。复现完会留下一个可复用的 TTA 评测 harness，覆盖 batch size、非 i.i.d.、continual 流几个维度，而这些恰好是机器人部署时会被破坏的假设。
- **算力**：两篇论文都没有报告算力。笔记作者判断 CIFAR 部分单卡可行。standard TTT 每个样本要走 10 步，逐样本评测很慢，建议用 `torch.func` 按样本向量化。
- **最小实验**
  1. Tent：用官方 `cifar10c.py`，在 severity 5 的 15 种 corruption 上跑 `source`、`norm`、`tent` 三个配置，确认 Tent < BN < Source 的排序。仓库用的是 WRN，论文 Table 2 用的是 R-26，所以只比较相对排序。
  2. TTT：用 ResNet-26 + GN 做 Y 形 joint training（rotation head 接在第 2 个 group 后面）。在 level 5 Gaussian noise 上，目标是对上 Table A2 的 JT 49.4 / TTT 45.6 / TTT-Online 25.8，原始测试集约为 8.1 / 7.9 / 8.2。
  3. 补做论文没做的压力测试：batch size 从 128 降到 16 再到 1；按类别排序的非 i.i.d. 数据流；15 种 corruption 串成一条流、中间不 reset；逐样本计算 $\langle\nabla_{\theta_e}l_m,\nabla_{\theta_e}l_s\rangle$，看它与改进量的相关性。
- **复现后可以提的假设**：把 $l_s$ 换成 world model 的转移预测损失，检验 Theorem 1 的梯度相关性在控制任务上是否成立。这是接到 PAD 和 SimDist 之前的最小一步。

### 9.2 TTT-E2E toy 实验，附带 TTT layers 与 KV-binding 的等价性检查（分支 B 的机制基准）

- **理由**：这是序列层分支里机制最清楚、成本最低的一组可复现结果。一个 toy 就能同时看到两件事：TTT-naive（dynamic evaluation）和 TTT-E2E（meta-learning）之间的差距，以及 mini-batch 带来的锯齿。这两点直接关系到"TTT 层能不能充当 world model 的长程记忆"。配合几项 CPU 上的数值检查，可以把 TTT layers 的 Theorem 1 和 KV-binding 争议落到自己能验证的代码上。
- **算力**：CPU 检查在 1 小时以内（笔记估计）。TTT-E2E 的 toy 按附录 A 设置：2 个 block、dim 384、6 个 head、上下文 128、DCLM 数据、batch 为 125K tokens。笔记作者估计单卡可行。
- **最小实验**
  1. CPU/numpy：实现 TTT layers 的 primal 与 dual form，检查两者输出一致；验证 $W_0=0$、$\eta=1/2$、batch GD 时与 linear attention 一致（Theorem 1）；验证在 Frobenius 损失下，"对 $v$ 做梯度上升"等价于"对 $-v$ 做梯度下降"（KV-binding 笔记"笔者的批判性阅读"第 1 条）。
  2. 用官方 `test-time-training/e2e` 仓库复现 Figure 2 右图的五条曲线：有 attention、无 attention、TTT-naive、TTT-E2E $b=1$、TTT-E2E $b=16$。判据：TTT-naive 只比 bigram 基线好一点；E2E $b=1$ 接近 full attention；$b=16$ 出现锯齿。学习率在 [2e-3, 6e-3] 内按 0.5e-3 步长搜索。
- **复现后可以提的假设**：把 inner loss 换成 MSE，重做梯度上升实验，补上 KV-binding 一文缺的那项检验；再把 toy 换成 action-conditioned 的序列预测，看端到端 TTT 相对 KVB 的优势在控制序列上是否还在。

### 9.3 TTRL：Qwen2.5-Math-1.5B 在 MATH-500 上（分支 C 的基准）

- **理由**：这是 test-time RL 最简单的完整实例，算法就是 GRPO 只换了 reward，官方有基于 verl 的代码。它能直接测出伪奖励的质量（标签准确率 vs 奖励准确率）。把 test-time RL 搬到 world model 里做 RL（[WMPO](../05_intersections/wmpo/notes.md)、[World-Gymnast](../05_intersections/world-gymnast/notes.md) 用的都是学出来、会出错的 reward）之前，必须先掌握这套诊断方法。TTT-Discover 依赖付费的 Tinker API，所以排在后面。
- **算力**：论文只写了 8×A100 80GB，没有报告时长。最小配置是 1.5B 模型、10 个 episode。
- **最小实验**：GRPO，每题 64 条用于投票、32 条用于训练，温度 1.0，最大长度 3,072，峰值 lr $5\times10^{-7}$（cosine，AdamW）。目标是复现 Table 1 的 32.7 → 73.0，或 Fig.5 中约 30 步内从 ~0.34 升到 ~0.68 的曲线。训练过程中用真标签离线记录 majority ratio $m$、ground-truth ratio $g$、标签准确率和奖励准确率，检验"标签错时 RewardAcc $=1-(m+g)$"。再补做论文缺的投票数 $N\in\{8,16,64\}$ 消融，以及与答案级 entropy minimization 基线的对比。
- **复现后可以提的假设**：在连续动作或 learned evaluator 的场景下，用什么替代 majority voting（多条 rollout 的共识、成功检测器、world model 内的 reward model），并且奖励准确率要在训练全程监控。

**复现之后的第一个项目内实验**（来自[综述笔记](./survey-test-time-intelligence/notes.md)"复现可行性"一节的建议，不是论文内容）：在 DMControl 或 MuJoCo 上训练 latent world model + MPC agent，测试时改变质量或摩擦，比较四组：冻结、只加 compute（更多 MPC 采样或更长 horizon）、只加 update（用转移预测误差在线更新 dynamics）、两者同时用。episodic 和 continual 两种设置分别画 return 与额外算力的曲线，以及长流上的稳定性曲线。单卡可做。
