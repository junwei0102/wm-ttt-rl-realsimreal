# The Surprising Effectiveness of Test-Time Training for Few-Shot Learning

> arXiv [2411.07279](https://arxiv.org/abs/2411.07279) · ICML 2025（Poster；arXiv v1 2024-11；本笔记依据 v2，2025-03） · 主题：02_test_time_training · 层级：milestone
> 代码：https://github.com/ekinakyurek/marc（ARC），https://github.com/adamzweiger/Fewshot-TTT（BBH）（引言脚注 1） · 笔记依据：`source/TTT_arxiv.tex`（数字/公式）+ `paper.pdf`（图）

## 一句话

测试时拿到一个 few-shot 任务后，先把它自带的 K 个示例用 leave-one-out 改写成一批合成的 in-context 任务，再用可逆几何变换扩增，给这个任务单独训一个 LoRA（test-time training），最后做"变换→预测→逆变换"的 augmented inference 加层级投票。最关键的一点是：TTT 数据保持与推理时相同的 ICL 格式，梯度更新是在 in-context learning 之上叠加，并不取代它。

## 问题设定

- **输入/输出**：给定 demonstration $(x_1,y_1),\dots,(x_K,y_K)$ 和 $x_{\rm test}$，直接生成 $y_{\rm test}$。这是 direct transduction，不用 CoT，也不写程序（Sec 2.2）。
- **ARC**：输入是 2D 网格，最大 30×30，最多 10 种颜色；每个任务有 2–7 个 demo、1–3 个 test（Sec 4.1）。网格按 numpy 数组的打印格式转成文本（App A.1, Fig 10）。评测用 pass@2，要求 exact match（Sec 4.2）。
- **BBH**：共 27 个任务。每个任务随机取 10 条作为 demo，其余全部用来评测，一般每任务 240 条，Causal Judgment / Penguins / Snarks 分别为 177/136/168 条；结果取 5 个 seed 的平均（Sec 5.2）。
- **假设**：测试时只有该任务的少量带标签 demo，没有其他数据。ARC 上允许用领域先验（几何变换不改变规则），BBH 上不使用任何任务特定知识。
- **之前卡在哪**：
  1. 纯 ICL 在结构新颖的任务上很弱：80 个 ARC 验证任务上 zero-shot 为 0（Fig 1）。
  2. 常规的 test-time compute，也就是温度采样加 self-consistency，在 ARC 上用不起来：直接输出网格时，没法同时保证样本之间有多样性、样本内部又自洽（Sec 4.2 "Inference"）。
  3. 经典 TTT（[Sun et al.](../ttt-self-supervised/notes.md)）只对输入 $x$ 做无监督任务，few-shot 场景里现成的 $(x,y)$ 对没用上（Sec 2.2）。

## 核心方法

**0. 通用 TTT 形式**（`source/TTT_arxiv.tex` Sec 2.2，未编号公式）。从 $\boldsymbol\theta_0$ 出发，对每个测试输入（或一批输入）$d$ 生成临时训练集 $\mathcal D_{\rm TTT}$，再做

$$\arg\min_{\boldsymbol\theta}\sum_{d_{\rm TTT}\in\mathcal D_{\rm TTT}}\mathcal L\big(\mathrm{LM}(d_{\rm TTT};\boldsymbol\theta)\big)$$

得到临时更新的参数 $\boldsymbol\theta_d$，用它预测完就丢弃。脚注 2 特意说明：这里的 TTT 与 [TTT layers](../ttt-layers/notes.md) / [Titans](../titans/notes.md) 里"RNN 隐状态即参数"的 TTT 相关，但不是一回事。

**1. 构造 $\mathcal D_{\rm TTT}$**（Sec 3.1，未编号公式）

- Leave-one-out：$d_j^{\rm ICL}=\big(\{(x_k,y_k)\}_{k\ne j},\,x_j,\,y_j\big)$，并额外打乱 demo 顺序。K 个 demo 生成 K 个"(K−1)-shot + query"的任务。推理时模型看到的是"K 个 demo + $x_{\rm test}$"，两者格式一致。
- 对照组 Direct I/O：$d_j^{\rm I/O}=(x_j,y_j)$，不带 context。
- 增广：取一组可逆变换 $\mathcal T$，满足 $t^{-1}(t(x))=x$，作用到任务中所有输入和输出网格上，得到 $\mathcal D_{\rm TTT}=\bigcup_{t\in\mathcal T}\bigcup_j t(d_j)$。具体变换见 App C.1 Table 5：Rotate 90/180/270、Flip、Reflect、RandomTranslateXY（最大平移 4）、Transpose、IncreaseResolution/Height/Width(2)，以及它们的 Chain 组合。之后再做颜色置换和示例打乱。每个任务最多保留 250 条（Sec 4.2）。

**2. 损失**（Sec 3.2，未编号公式）

$$\mathcal L^{\rm label}_{\rm LM}=\mathcal L_{\rm LM}(y_{\rm test}\mid x_1,y_1,\dots,x_K,y_K,x_{\rm test};\boldsymbol\theta)$$

$$\mathcal L^{\rm outputs}_{\rm LM}=\mathcal L^{\rm label}_{\rm LM}+\sum_{k=1}^{K}\mathcal L_{\rm LM}(y_k\mid x_1,y_1,\dots,x_k;\boldsymbol\theta)$$

$$\mathcal L^{\rm all}_{\rm LM}=\mathcal L^{\rm outputs}_{\rm LM}+\sum_{k=1}^{K}\mathcal L_{\rm LM}(x_k\mid x_1,y_1,\dots,y_{k-1};\boldsymbol\theta)$$

在 LOO 样本里，这里的 "test" 指被留出的 $(x_j,y_j)$。默认用 $\mathcal L^{\rm outputs}$：同一条序列里，每个前缀都是一个 (k−1)-shot 任务，都能提供监督。ARC 上求和从 $k=2$ 开始，因为只看 0 个 demo 推不出规则（脚注 3）。$\mathcal L^{\rm all}$ 连输入也一起建模，相当于 Sun et al. 那种无监督 TTT 目标。注：Sec 3.2 末尾把"loss over demo and test outputs"称为 "the first method"，其实它对应列表里的第二项；Fig 5/8 的实验也表明默认用的是 $\mathcal L^{\rm outputs}$。

**3. 参数化**（Sec 3.3, App C.2）

- 默认每个任务一个 LoRA（base 权重不动）。ARC 上（App C.2）LoRA 作用于 self-attention 的 q/v 投影、MLP，以及输出投影（最后这项只有 Llama-3 8B 在 torchtune 里支持）。
- ARC 超参（Table 6）：$r=128,\ \alpha=16$；lr 5e-5（1B/3B）或 1e-4（8B）；2 epochs，batch 2，AdamW。
- BBH 超参（Table 8 加粗项）：$r=64,\ \alpha=64$，lr 1e-4，40 steps，batch 5，cosine schedule；由 demo 的 40 种随机打乱生成 LOO 任务（Sec 5.2）。
- 对照组 Shared：所有任务共用一个 LoRA，相当于测试时版本的 MetaICL。

**4. TTT 之前的微调（只做 ARC；Sec 4.2, App B）**

- 在合成的 ARC 类任务上对 Llama-3 系列做全参数 FT，数据格式和目标与 TTT 完全相同。
- 数据来源：
  - ReARC 生成器，$d=(x,y)\sim{\rm eval}(g_i)$（App B Eq.1）；
  - 用 GPT-4/4o few-shot 生成的 6426 个新生成器（Eq.2–5, App D）；
  - 以 30% 的概率施加几何变换，分为只变输入、只变输出、两者都变三种。
- 超参（Table 4）：lr 2.5e-5，2 epochs，batch 32，AdamW，cosine 加 2k warmup；步数上限 16000 写在 App B.2 正文。

**5. Augmented inference + 层级投票**（Sec 4.2, App E.1 Eq.6–7）

$$\tilde y\sim\mathrm{LM}\big([t(x_1),t(y_1),\dots,t(x_{\rm test})]\big),\qquad y_t=t^{-1}(\tilde y)$$

- 推理用的 $\mathcal T$ 是 90°/180° 旋转、水平/垂直翻转和 transpose（App C.2）。每个 $t$ 再配 $n=2$ 种 demo 顺序，每个组合都做贪心解码，共得到 $n|\mathcal T|$ 个候选。
- 第一阶段（组内投票）：按 $t$ 分组，每组取频次 top-3。唯一答案不足 3 个时，用逐行多数和逐列多数拼出的网格补齐。
- 第二阶段（全局投票）：对各组候选再投一次票，取 top-2；平票时优先 identity 变换。
- 直觉：变换提供多样性，贪心解码保证单个样本自洽；模型如果真学到了规则，各个变换视角逆变换回来后应当给出一致答案。
- BBH 不做这一步，只用贪心解码（Sec 5.2）。

**流程小结**：训练阶段只做一次（ARC 上是合成任务 FT 得到 $\boldsymbol\theta_0$）。每遇到一个测试任务：构造 $\mathcal D_{\rm TTT}$ → 训练 LoRA → 用 vLLM 挂上这个 LoRA 生成 $n|\mathcal T|$ 个候选 → 逆变换 → 投票 → 丢弃 LoRA。

## 关键实验与证据

**ARC 80 任务开发子集**：easy/medium/hard/expert 各 20 个（App A.2 Table 2），报告 tasks solved（满分 80），按任务严格计分。

- **设计消融（Fig 5，1B）**：

  | 设置 | tasks solved | 相对默认 |
  |---|---|---|
  | FT（不做 TTT） | 5 | — |
  | FT+TTT（默认） | 29 | 0 |
  | No Transformations | 13 | −16 |
  | Direct I/O | 18 | −11 |
  | Shared TTT | 22 | −7 |
  | No Demo Loss | 26 | −3 |

  增广和 ICL 格式贡献最大，per-task adapter 次之，demo loss 最小。换算口径：Sec 4.3 写的 "5%→29%"、"26%→29%" 实际是题数（5/80 = 6.25%）；括号里的 38%/55%/24% 是相对 29 题的降幅。
- **模型规模（Fig 6 / Fig 13 右）**：FT → FT+TTT 分别为 1B 5→29（+24）、3B 11→29（+18）、8B 14→36（+22）。Fig 1 把 8B 结果换算成百分比是 17.5% → 45.0%（+27.5 pts）。Sec 4.4 和 App B.3 正文说"FT 性能随规模提升，8B 最高达 36%"，但 36 其实是 8B 在 FT+TTT 后的题数（8B 只做 FT 是 14 题）。
- **FT 数据消融（Fig 13 左，1B）**：No FT 0→9；All 5→24；No-Geom 5→20；No-LM 5→29。说明 TTT 替代不了 FT，而且 LM 生成的数据反而有害。
- **推理与投票（Fig 7）**：单一变换 Rotate 23、Transpose 18、Flip 21；Vanilla 22；Flattened voting 27；Hierarchical 29；Oracle 32。
- **QLoRA**：29 → 26（App C.2）。

**ARC 完整公开验证集（400 题）**（Table 1，pass@2；App A.3 注明此表允许按 test 给部分分）：

- 自家 8B FT 18.3% → 加 TTT 47.1%。
- 把本文 TTT 用到 BARC 的神经模型上得到 53.0%，BARC 原版（no synthesizer）是 39.3%，相对提升 35%。
- BARC 程序合成 + 自家 FT + 本文 TTT：58.5%。
- BARC 程序合成 + BARC FT + 本文 TTT：正文、摘要和 Table 1 的 caption 都写 61.9%（61.875%），Table 1 同一行却印 62.8%，论文内部不一致，以摘要为准。
- 参照线：人类平均 60.2%，人类最佳 97.8%，BARC ensemble 54.4%，o1-preview 21.0%，DeepSeek r1 20.5%，o3 82.8%。
- 互补性：BARC 神经模型能解出的题占程序合成已解题的 42.2%；加上本文 TTT 后升到 73.5%（Sec 4.6）。
- 官方 semi-private 集：ensemble 只有 47.5%。

**BBH**（Llama-3.1-8B，10-shot，5 seeds，Fig 8；SEM 见 Table 7，平均约 0.4 pts）：

| 设置 | 准确率 |
|---|---|
| zero-shot | 40.9 |
| ICL | 50.5 |
| TTT | **57.8** |
| No Example Permutation | 55.7 |
| Direct I/O | 51.5 |
| Shared TTT | 59.8 |
| No Demo Loss | 54.4 |
| Loss on Inputs and Outputs | 55.9 |

- 分任务结果（Fig 9，格式为 ZS/ICL/TTT）：Dyck Languages 3.6/18.5/68.8；Ruin Names 53.6/60.5/81.9；Movie Rec. 53.2/69.5/86.7；Hyperbaton 58.0/63.8/78.7；Boolean Expressions 71.0/85.7/80.1（正文写 80.4）。
- TTT−ICL 差值的分布（Table 9）：≥5 的有 9 个任务，[2,5) 有 3 个，|x|<2 有 13 个，下降至少 2 pts 的有 2 个（Boolean Expressions 为 x≤−5，Penguins In A Table 在 (−5,−2]）。

## 局限与论文没说的

- **作者承认的**：
  - 超参在 80 题子集上调，而消融也在这 80 题上评测，存在 optimization bias；
  - ARC/BBH 都是公开数据，可能已被预训练见过（Limitations）；
  - semi-private 集掉到 47.5%，作者归因于分布偏移；
  - BBH 上哪些任务受益"remains an open question"。
- **评测口径**：
  - 80 题实验按任务严格计分，Table 1 却允许部分分，两者不可直接比；
  - 正文里"%"和"题数"混用（见上），Table 1 的 62.8% 与正文 61.9% 也对不上；
  - 80 题子集取自验证集，最终又在包含这 80 题的完整验证集上报告。
- **增益很大程度来自领域先验**：ARC 去掉变换掉 16 题（>50%），投票同样依赖同一组可逆变换。BBH 没有这类先验，TTT 只 +7.3 pts，而且 Shared（59.8）反而优于 per-task（57.8）。所以"per-task 更好"不是普适结论，它取决于不同任务的输入是否容易混淆（Sec 5.3）。
- **没做 compute-matched 对照**：ARC 每题都要单独训一个 LoRA，App C.2 给的开销是 100 题约 12 小时 / 1×A100，没有与同等 FLOPs 下多采样、更大模型 ICL 的比较。
- **只测了 direct transduction**：没测 CoT/推理模型加 TTT，只测了 Llama 系列、三个规模；1B 和 3B 在 TTT 后同为 29 题，scaling 结论很弱。
- **瓶颈在候选生成**：Oracle 32 对 Hierarchical 29，选答案已接近饱和。
- **失败模式**：需要逐步计算、可能在预训练中见过的任务会变差（Boolean 85.7→80.1）；K 很小时 LOO 只能生成极少任务（ARC 每题只有 2–7 个 demo）；每题的 adapter 用完即丢，任务之间不积累经验。

## 在脉络中的位置

- **前驱**：
  - local learning（Bottou & Vapnik 1992）/ transductive learning（Joachims 1999）；
  - [TTT with self-supervision](../ttt-self-supervised/notes.md)：用测试样本的自监督 loss 更新参数，本文把自监督信号换成 demo 的 $(x,y)$；
  - Hardt & Sun 2024 的检索邻居 TTT；
  - MetaICL（Shared 变体的来源）、self-consistency、LoRA；
  - ReARC 与 BARC（Li et al.，同期工作，也做 TTT 与程序合成）。
  - 与 [Tent](../tent/notes.md) 这类熵最小化 TTA 不同，本文的测试时信号是有监督的。
- **后继 / 影响**：
  - [TTRL](../ttrl/notes.md) 把测试时更新从 SFT loss 换成以 majority voting 为伪奖励的 RL，投票这一环与本文的层级投票相通；
  - [TTT-Discover](../ttt-discover/notes.md) 进一步在单个问题上做测试时 RL；
  - [TTT-E2E](../ttt-e2e/notes.md) 同样"把 context 当训练数据写进权重"，并在训练阶段 meta-learn 初始化，对应本文 Fig 13 里"FT 决定 TTT 上限"的观察；
  - [测试时智能综述](../survey-test-time-intelligence/notes.md) 把这些工作放进 feedback-driven 的测试时学习框架。
- **与本项目其他主题的联系**：
  - *real2sim2real*：部署时只拿到 K 条真实轨迹，就对应这里的 K 个 demo。把仿真预训练的 world model 或 policy 当作 $\boldsymbol\theta_0$，用这些真实数据做 LoRA 式的少步更新，就是梯度式的 system identification；[SimDist](../../05_intersections/simdist/notes.md) 只更新 dynamics 的思路与本文 per-task adapter 最接近。[RMA](../../03_real2sim2real/rma/notes.md) 则是不做梯度的前向式适应，相当于纯 ICL 这一端。本文 Direct I/O < ICL 格式的发现提示：适应数据的格式要与推理时的条件化方式一致。
  - *world model*：[Dreamer](../../01_world_model/dreamer/notes.md) 类模型的多步预测 loss 可以像 $\mathcal L^{\rm outputs}$ 那样在每个前缀上都监督。可逆变换加逆变换投票对应机器人里的对称性增广（如左右镜像），这是我的推断，论文没有涉及。
  - *RL*：本文的 TTT 信号是模仿式的 SFT，没有 reward。换成 reward 就是 TTRL / [DeepSeek-R1](../../04_rl/deepseek-r1/notes.md) 式的 GRPO；对 VLA，可以设想部署时用少量示范做 per-task LoRA，再接 [SimpleVLA-RL](../../04_rl/simplevla-rl/notes.md) 式的 RL。

## 前置知识

1. In-context learning 与 few-shot prompting；因果 LM 的逐 token 交叉熵，以及"只在输出 token 上算 loss"的 masking。
2. LoRA（rank $r$、缩放 $\alpha$、作用于哪些投影矩阵）和 QLoRA。
3. 经典 TTT/TTA（[Sun et al.](../ttt-self-supervised/notes.md)、[Tent](../tent/notes.md)）：测试时用什么 loss、更新哪些参数。
4. Self-consistency / majority voting；pass@k。
5. ARC 的任务格式与评测规则；transductive 与 inductive（程序合成）两类解法的区别。
6. 数据增广中的等变性：$t^{-1}\circ f\circ t=f$ 的前提是规则对 $t$ 等变。

## 复现可行性

- **代码**：两个仓库都已开源，已用 WebFetch 确认存在。marc 的 README 列出了 HF 上的 `ekinakyurek/marc-8B-finetuned-llama3`、对应的 LoRA adapters 和预测结果，依赖作者 fork 的 torchtune 和 vLLM（Llama-3 与 Llama-3.2 需要分开建环境），ARC 数据需要从 Kaggle 自行下载。Fewshot-TTT 用 vLLM + torchtune，带 SLURM 脚本。
- **算力（论文所写）**：
  - TTT 前的 FT：最多 16000 步，1B 用 2×A100，3B/8B 用 4×A100，没给总时长（App B.2）；
  - ARC 的 TTT + 推理：100 题约 12 小时 / 1×A100（App C.2）；
  - BBH 的完整 TTT + 推理：约 15 分钟 / 1×A100（App F.1），没说明这是单个 seed 还是全部 5 个。
- **最小复现**：
  1. **最便宜**：BBH，Llama-3.1-8B（论文只写 Llama 3.1 8B；Fewshot-TTT README 的示例命令用的是 Llama-3.1-8B-Instruct），不需要预 FT，只跑 1 个 seed。对比 ICL 和 TTT（按 Table 8 加粗超参），目标是复现 Fig 8 的 50.5 → 57.8，以及 Dyck Languages 上 18.5 → 68.8 的大幅跳变。再加跑 Direct I/O（51.5），验证"ICL 格式"这一核心结论。
  2. **ARC**：直接用官方 8B FT checkpoint，跳过 FT。在 Table 2 的 80 题上跑默认 TTT，目标是 Fig 6 的 14 → 36 题；再关掉变换增广，看降幅是否接近 Fig 5 的规律（Fig 5 是 1B 上的结果）。
- **预计难点**：
  - torchtune fork 与 vLLM 多 LoRA 服务的版本要锁定；
  - 400 个 per-task adapter 的管理；
  - 超参按模型规模不同（lr 5e-5 或 1e-4）；
  - 注意严格计分和部分分两种口径；
  - 想从 1B 复现 Fig 5 的话，需要自己用 ReARC 做 FT，而 FT 数据配方（All 还是 No-LM）会直接改变 TTT 后的结果（24 对 29）。

## 自测问题

1. Direct I/O 和 leave-one-out 的训练样本分别长什么样？为什么在 ARC（18 对 29）和 BBH（51.5 对 57.8）上 Direct I/O 都更差？从"训练和推理时的条件化格式是否一致"来解释。
2. 写出 $\mathcal L^{\rm outputs}$。一条 LOO 序列提供了多少个监督信号？为什么 ARC 上求和从 $k=2$ 开始？$\mathcal L^{\rm all}$ 多出来的那一项对应经典 TTT 里的哪类目标，为什么在 BBH 上没帮上忙（55.9 对 57.8）？
3. Shared adapter 为什么在 ARC 上有害（22 对 29），在 BBH 上却有益（59.8 对 57.8）？这对"per-task 还是 shared 适应"的选择说明了什么？
4. Augmented inference 为什么必须用可逆变换并做 $t^{-1}$？层级投票相比 flattened 投票解决了什么问题？Oracle 32 对 Hierarchical 29 说明系统瓶颈在哪一环？
5. 如果把这套方法搬到"用 K 条真实轨迹适应一个仿真预训练 world model"，$\mathcal D_{\rm TTT}$、loss、参数化和"推理增广"分别该怎么对应？哪一步最可能失效？
