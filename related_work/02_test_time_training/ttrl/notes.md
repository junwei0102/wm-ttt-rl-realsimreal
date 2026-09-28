# TTRL: Test-Time Reinforcement Learning

> arXiv [2504.16084](https://arxiv.org/abs/2504.16084) · NeurIPS 2025（Poster，已在 OpenReview 核实；本地 `source/` 是 arXiv v3，用 COLM 2025 模板的 preprint 版）· 主题：02_test_time_training · 层级：milestone
> 代码：https://github.com/PRIME-RL/TTRL（论文作者栏给出；基于 verl，MIT）· 笔记依据：`source/colm2025_conference.tex`（数字/公式）+ `paper.pdf`（图）

## 一句话

在**没有标签的测试题**上，对每道题采样 $N$ 个回答，用 majority voting 选出的答案当伪标签，再用"是否与伪标签一致"的 0/1 rule-based reward 跑 GRPO。这样就把 test-time scaling（投票）变成了 test-time training（RL 更新参数）的奖励来源。Qwen2.5-Math-7B 在 AIME 2024 上的 pass@1 从 12.9 升到 40.2（+211.6%，Table 1）。

## 问题设定

- **设定**（§2 定义框）：给定一个预训练 LLM $\pi_\theta$ 和某个 benchmark 的全部测试题 $\lbrace x\rbrace$（只有题，没有答案），在这些题上用 RL 更新 $\theta$，然后在**同一批题**上评测。每个 benchmark 单独训一次（§3.1 "apply TTRL to each benchmark individually"），本质上是 transductive、数据集级别的适应，不是逐样本的流式 TTA。
- **输入/输出**：输入是题目 prompt $x$，输出是 CoT 加最终答案。answer extractor 从中抽出离散答案 $\hat y$（数学题的数值、GPQA 的选项），这样才能投票和做精确比对。
- **隐含假设**：① 答案可离散化、可判等；② 模型对目标任务已有足够先验，投票结果与正确答案有一定相关性（§4.3 明确说先验不足会失败）；③ 采样分布既不能完全坍到一个答案（否则没有梯度），也不能太发散。
- **之前卡在哪**：RL for reasoning（[DeepSeek-R1](../../04_rl/deepseek-r1/notes.md)、GRPO）依赖 ground-truth 才能算 rule-based reward。经典 TTT/TTA（[TTT](../ttt-self-supervised/notes.md) 的自监督辅助任务、[Tent](../tent/notes.md) 的熵最小化）是为分类/视觉设计的。"用 majority voting 挑自生成 CoT 做 SFT"的 self-training（Huang et al. 2022）被认为以初始模型的 maj@n 为上界（§4.1）。self-rewarding 类工作多做 instruction following，或者用的是 DPO 而不是 online RL（§5.2）。

## 核心方法

**1. RL 目标**（`source/colm2025_conference.tex` §2.1，Eq.1–2）：把 prompt 当 state、回答当 action，共识答案 $y^{\ast}$ 当"最优动作的代理"：
$$\max_\theta \mathbb{E}_ {y\sim\pi_\theta(\cdot|x)}\big[r(y,y^{\ast})\big],\qquad \theta\leftarrow\theta+\eta\nabla_\theta\mathbb{E}_ {y\sim\pi_\theta(\cdot|x)}\big[r(y,y^{\ast})\big].$$

**2. 标签估计**（Appendix C Eq.4，属于 test-time inference 的通用形式）： $y^{\ast}=\arg\max_{y_i}s(y_i,x)$。TTRL 里的打分函数 $s$ 取 majority voting，即 $P=\lbrace\hat y_i\rbrace_{i=1}^N$ 中出现次数最多的答案。

**3. 奖励**（§2.2 Eq.3，Listing 1）：

$$R(\hat y_i,y)=\begin{cases}1,&\text{if }\hat y_i=y,\cr0,&\text{otherwise}.\end{cases}$$

原文 Eq.3 把投票得到的估计标签记作 $y$；它就是 §2.1 里的共识答案 $y^{\ast}$，下文统一写 $y^{\ast}$。

**4. 策略优化**：论文直接套 GRPO（§3.1），也验证了 PPO 和 PRIME（Fig.5）。论文没写 GRPO 公式。按 GRPO 原文，组内 advantage 为 $A_i=(r_i-\bar r)/\mathrm{std}(r)$。下面是**我自己的推导**：设某题的 majority ratio（多数答案在组内的占比）为 $m$（近似：投票用 64 条，训练组是下采样的 32 条，组内占比只是约等于 $m$），二元奖励下均值为 $m$、标准差（总体标准差）为 $\sqrt{m(1-m)}$，于是
$$A^{+}=\sqrt{\tfrac{1-m}{m}},\qquad A^{-}=-\sqrt{\tfrac{m}{1-m}}.$$
推论： $m\to1$（所有样本一致）时组内方差为 0，这道题不产生梯度。 $m$ 小时，多数派样本拿到很大的正 advantage。所以 TTRL 的梯度实际上是把概率质量推向众数答案，也就是在答案层面让分布"变尖"（sharpening）。这个效果和 Tent 的熵最小化同向，不过作用在整条序列的最终答案上，而且是靠 on-policy 采样加 policy gradient 实现的；Tent 则直接对模型输出分布的熵求梯度。两者都是 online 的，区别不在这里。

**5. vote-then-sample**（§3.1）：每题采 64 个回答用于投票，再从中下采样 32 个进训练 batch。投票用大样本让标签更准，训练用小样本省算力。

**6. 为什么错标签也能训：Lucky Hit**（§4.2，Fig.10）。rule-based verifier 只做"是否等于标签"的比较。一个错误回答只要与（同样错的）伪标签不同，就会拿到 0，而这恰好是正确的奖励。用 Appendix B 的定义可以写成闭式（**我的推导**）。记 $g$ 为 ground-truth 答案在组内的占比：

$$\text{RewardAcc}=\begin{cases}1,&y^{\ast}=y_{\text{gt}}\cr1-(m+g),&y^{\ast}\neq y_{\text{gt}}\end{cases}$$

标签错时，只有"等于错标签的样本"（本应得 0 却得 1）和"等于真答案的样本"（本应得 1 却得 0）的奖励是错的。拿 Fig.10 的玩具例子验证：预测为 1 1 2 2 2 4 5 6，伪标签为 2，真值为 3， $m=3/8$， $g=0$，得 $1-0.375=62.5$%，与图中一致。所以模型越弱、答案越发散（ $m$ 小），奖励反而越准。

**7. Online**（§4.2）：每一步都用当前策略重新投票，模型变强后 maj@n 也跟着变好，标签质量随训练提升（Fig.6）。offline self-training 的标签在训练前就固定了，做不到这一点。

**流程**：训练阶段，对测试集跑若干 episode：采 64 → 投票 → 算 0/1 奖励 → 下采样 32 → GRPO 更新。推理阶段，用更新后的模型在同一 benchmark 上评 pass@1。主实验每题采 16 个（32k context 时采 4 个），T=0.6，top-p=0.95；Qwen2.5-Math 的分析实验用 greedy（§3.1 Evaluation Setup）。

**超参**（§3.1）：AdamW，cosine lr，峰值 $5\times10^{-7}$。rollout 温度 0.6（Qwen2.5-Math 和 LRM 用 1.0）。最大生成长度 3,072（LRM 为 32,768）。episode 数：MATH-500 为 10，AMC 为 30，AIME 2024 为 80。硬件 8×A100 80GB。

## 关键实验与证据

- **主结果**（Table 1，pass@1，16 样本平均）：
  - Qwen2.5-Math-1.5B 平均 23.5→41.0，MATH-500 32.7→73.0。
  - Qwen2.5-Math-7B：AIME 12.9→40.2，平均 31.1→54.9，但 GPQA 29.1→27.7（**−1.4**）。
  - Qwen2.5-7B 平均 33.8→48.5；Qwen2.5-32B 平均 32.4→51.1；LLaMA3.1-8B 平均 26.8→35.0；Qwen3-8B（non-thinking，3k）平均 53.8→64.5。
  - §3.2 正文称"AIME 2024 上 6 个模型的提升都 ≥105%"，但 Table 1 自己的数字与此不符：Qwen3-8B 的 AIME 只 ↑73.6%（26.9→46.7），其余 5 个模型在 105.2%–211.6% 之间。
- **更多模型**（Table 2）：大多有提升，但 Mistral-Nemo-Instruct 的 AIME 0.8→0。DeepSeek-R1-LLaMA-8B 的 AIME 51.7→69.2。
- **LRM**（Fig.3，AIME 2024，32k）：Skywork-OR1-Math-7B 66.7→75.0，Qwen3-8B（thinking）72.5→82.5。
- **跨 benchmark 泛化**（Fig.4，Qwen2.5-Math-7B，greedy）：在 AIME 上训练，AMC 39.8→60.1、MATH-500 52.6→75.4；在 MATH-500 上训练，AIME 13.3→23.3。说明收益不只是记住测试题。
- **RL 算法无关**（Fig.5）：在 MATH-500 / Qwen2.5-Math-1.5B 上，GRPO、PPO、PRIME 的准确率和熵曲线几乎重合。
- **超过 maj@n 上界**（Fig.7，Qwen2.5-Math-7B）：TTRL 的 avg@64 高于 backbone 的 maj@64。AIME 32.3 vs 30.0，AMC 63.9 vs 56.6，MATH-500 84.2 vs 66.4。
- **接近 RL (leakage)**（Fig.8）：在 MATH-500 / Qwen2.5-Math-7B 上，用真标签在测试集上做 RL 与 TTRL 的曲线前 ~40 步基本重合。
- **Lucky Hit 实证**（Fig.9，AIME / Qwen2.5-Math-7B）：初始标签准确率只有 37%，奖励准确率却有 92%。base 模型的众数答案只占 16.6%。正文称训练中标签准确率很少超过 50%，而奖励准确率始终高于 75%（按图读，最低点约 0.75）。
- **难度消融**（Table 3，MATH-500 按 L1–L5 拆开，Qwen2.5-Math-1.5B）：提升从 L1 的 +45.4 递减到 L5 的 +16.8；回答长度缩短比例从 73.3% 降到 43.7%。作者据此认为难题上先验不够。
- **失败案例**（Fig.11，AIME）：T=0.6 和 train batch 256 两个设置的熵一直降不下来，训练失败。
- **greedy 口径**（Table 4）：Qwen2.5-Math-7B 的 AIME 16.7→43.3，与用 8.9K 标注数据训练的 Oat-Zero-7B（43.3）持平。1.5B 的 AIME 为 20.0→**20.0**，没有提升。作者自己也注明这个比较"setup 不同、看起来不公平"。

## 局限与论文没说的

**作者承认的**：先验知识和超参（温度、episode 数）对成败影响很大，后续版本才会系统分析（§7）。难题上先验不足会失败；方法没有 data filtering 或 curriculum（§4.3）。流式在线 TTA、理论收敛分析、agentic 任务都列为 future work。

**我读出来的**：
1. **transductive 且反复刷同一批题**：AIME 跑 80 个 episode，"test-time"实际是在固定测试集上做数据集级 RL。它不是部署中逐条到来、只看一次的设定，和 Tent/TTT 的在线设定不能直接比。
2. **checkpoint 选择不透明**：Appendix B 说训练指标会"inform the selection of the optimal checkpoint"。Fig.12（Qwen2.5-Math-7B，AIME 2024）里准确率波动很大：按图读，第 54–59 步在 0.43–0.47 之间，最后一点（第 60 步）跌到约 0.37。同一附录的 Table 4 报告的是 43.3，既不是最后一点，也不是峰值。论文没说报告的是哪个 checkpoint、怎么选的；如果按真标签挑，就有泄漏。
3. **文字与图不一致**：§4.1 说 Fig.6 中"final avg@16 exceeds initial maj@16 by more than 20 points"。按图读，avg@16 最终约 0.46，初始 maj@16 约 0.43，只差约 3 点。Fig.7 里 AIME 的差距也只有 2.3。
4. **Lucky Hit 会随训练自我削弱**：按上面的闭式，标签错时奖励准确率是 $1-(m+g)$。TTRL 会推高 $m$（Fig.9 中 majority ratio 从 ~0.16 升到 ~0.5），奖励准确率也就从 ~0.92 掉到 ~0.75–0.85。这正是"多数但错误的答案自我强化"（confirmation bias）的通道，后续 T³RL、DARE 等就是针对这一点（见[综述笔记](../survey-test-time-intelligence/notes.md)）。
5. **基线太弱、消融缺失**：正文只和 backbone 比（§3.1 Baselines）；附录 A 的 Table 4 只额外列了用标注数据训练的 R1-Zero 类模型作参考。没和熵最小化、self-consistency SFT 等无标签方法比。投票数 $N$、下采样数、奖励形式（比如按票数比例的软奖励）都没有消融。
6. **评测噪声**：Fig.4 里的 AIME 数字，以及 Table 4 中 backbone 和 TTRL 两行的 AIME 数字，都是 3.3 的整数倍（Table 4 的参考模型行除外，比如 2.5、32.5）。这说明题量很小（论文没写题数），几道题就能造成大幅相对提升。同一 backbone 在 Table 1 和 Table 4 的起点不同（12.9 vs 16.7），"+211%"这类百分比高度依赖评测口径。
7. **增益的来源存疑**：Table 3 中回答长度大幅缩短：L1–L3 从约 2,100–2,340 降到约 610–670，L4/L5 从约 1,750–1,775 降到 784/985（原文未标单位）。在 3,072 的长度上限下，一部分增益**可能**来自学会及时收尾、按格式给答案，而不是推理能力本身（论文没拆开这一点）。
8. **适用面窄**：需要可判等的离散答案。GPQA 上增益很小，Qwen2.5-Math-7B 甚至是负的。开放式生成、证明、连续输出（例如机器人动作）都没有定义好的"多数"。
9. **算力没报 GPU-hours**：只写了 8×A100 80GB，每题 64 条 rollout 乘以几十个 episode，成本明显高于推理时投票。

## 在脉络中的位置

- **前驱**：
  - TTT 范式：[TTT with self-supervision](../ttt-self-supervised/notes.md)、[Tent](../tent/notes.md)、[TTT for ARC few-shot](../ttt-arc-fewshot/notes.md)。
  - 奖励来源：self-consistency / majority voting（test-time scaling）。
  - 优化器：GRPO 与 rule-based reward（[DeepSeek-R1](../../04_rl/deepseek-r1/notes.md)）、[PPO](../../04_rl/ppo/notes.md)、PRIME。
  - self-training 上界：LMSI（Huang et al. 2022）。
- **后继 / 影响**：
  - [TTT-Discover](../ttt-discover/notes.md) 明确拿 TTRL 做对照：它只在单道测试题上训练，用连续可验证的奖励，目标是找到一个极好的解，而不是提高整个集合的平均分。
  - [测试时智能综述](../survey-test-time-intelligence/notes.md) 把 TTRL 列为 consensus-derived test-time RL 的代表，后续的 TTRV（VLM）、T³RL（工具验证）、DARE/SCOPE/DDRL（稳定化）都在修补伪奖励的偏差。
  - [SimpleVLA-RL](../../04_rl/simplevla-rl/notes.md) 在 RL for reasoning 的相关工作中引用了它。
- **与本项目其他主题的联系**：
  - **TTT**：学习信号从"自监督辅助 loss"换成"自估计的 reward"。它和 Tent 一样，是在用模型自身的置信度做监督。
  - **RL**：算法上就是 GRPO 换了奖励函数。它最可迁移的经验是：该监控的是**奖励准确率**，不是标签准确率。
  - **world model / 机器人**：[WMPO](../../05_intersections/wmpo/notes.md)、[World-Gymnast](../../05_intersections/world-gymnast/notes.md) 在 world model 里做 RL 时，同样面对"奖励来自学出来的、会出错的判别器"这个问题。在部署侧，[PAD](../../05_intersections/pad/notes.md)（自监督 inverse dynamics 辅助任务）和 [RMA](../../03_real2sim2real/rma/notes.md)（适应模块）是机器人上的 test-time adaptation；TTRL 提示的可能路线是用多条 rollout 的共识或成功检测器当伪奖励做在线 RL，但连续动作空间里"多数"如何定义是开放问题。

## 前置知识

1. Policy gradient、PPO / GRPO（组内 advantage 归一化）、KL 与熵的作用
2. RLVR / rule-based reward（DeepSeek-R1 的做法）
3. Test-time scaling：self-consistency / majority voting、best-of-N；pass@k、maj@k、avg@k 的区别
4. TTT / TTA 的基本设定（TTT、Tent），以及 transductive 设定与测试集泄漏
5. LLM 采样：temperature、top-p、输出熵

## 复现可行性

- **代码**：官方开源（https://github.com/PRIME-RL/TTRL，MIT）。据 README，代码基于 verl，Python 3.10；新版本在 verl v0.4.1 上用 `+ttrl.enable=True` 开启；仓库里也有训练日志和多模型脚本。
- **算力**：论文只写"All experiments were conducted on 8 * NVIDIA A100 80GB GPUs"（§3.1），没报时长或 GPU-hours。
- **数据 / 模型**：AIME 2024、AMC、MATH-500、GPQA-Diamond，全部公开；backbone 都是 HF 上的开源权重。
- **最小复现**：Qwen2.5-Math-1.5B 在 MATH-500 上，10 episode，64 投票/32 训练，温度 1.0，最大长度 3,072。
  - 目标 1：复现 Table 1 的 32.7→73.0，或 Fig.5 中约 30 步内准确率 ~0.34→~0.68 的曲线。
  - 目标 2：同时用真标签离线记录 majority ratio $m$、ground-truth ratio $g$、标签准确率和奖励准确率，检验 $\text{RewardAcc}=1-(m+g)$（标签错时）。
  - 扩展（补论文缺的消融）：投票数 $N\in\lbrace8,16,64\rbrace$，以及与"答案级熵最小化"基线的对比。
- **难点**：温度、batch 等超参很敏感（Fig.11）；answer extractor / verifier 的判等细节会直接改变投票结果；无标签时怎么选 checkpoint；AIME 题量小，结果方差大。

## 自测问题

1. 用 Appendix B 的定义推导：majority label 错误时，奖励准确率如何由 majority ratio $m$ 和 ground-truth ratio $g$ 决定？为什么 Fig.9 中奖励准确率会随训练下降？
2. 在 GRPO 的组内归一化下，二元 majority 奖励的正/负 advantage 如何随 $m$ 变化？ $m\to1$ 时会发生什么？哪些题真正贡献梯度？
3. 为什么 majority-vote self-training（SFT）以初始 maj@n 为上界，而 TTRL 的 avg@n 能超过它？"online"和"RL（而非 SFT）"各自起什么作用？
4. TTRL 与 Tent 都是"用模型自身的置信度当监督"。两者在更新的参数、信号粒度（token/类别 vs 最终答案）、坍塌风险上有什么异同？
5. 要把 TTRL 的思路搬到机器人 policy 的部署时适应（连续动作、没有离散答案），majority voting 需要换成什么？本文的哪些隐含假设会失效？
