# DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning

> arXiv [2501.12948](https://arxiv.org/abs/2501.12948) · Nature 645, 633–638 (2025)，Nature 版标题为 "DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning"（出版信息不在 PDF 里，已在 nature.com/PubMed 核对）（本地 `paper.pdf` 是 arXiv v2，2026-01-04，内容为 Nature 版正文加 Supplementary） · 主题：04_rl · 层级：milestone
> 代码：https://github.com/deepseek-ai/DeepSeek-R1（已 WebFetch 确认：仓库里只有权重链接、使用说明和报告 PDF，**没有 RL/SFT 训练代码，也没有训练数据**）；推理代码在 https://github.com/deepseek-ai/DeepSeek-V3；权重放在 HuggingFace `deepseek-ai/`。 · 笔记依据：`source/sn-article.tex`（正文）、`source/appendix.tex` 和 `source/tables/*.tex`（数字与公式）+ `paper.pdf`（图）。Eq./Table/Figure 编号一律按 PDF。

## 一句话

R1-Zero 在 DeepSeek-V3-Base 上**不做 SFT**，直接用 GRPO 加规则可验证奖励（答案对错 + 格式）做大规模 RL。模型在训练中自己学会拉长 CoT、反思和验证，AIME 2024 pass@1 从 15.6% 升到 77.9%。R1 再用 cold-start SFT → RL → rejection sampling SFT → 全场景 RL 的多阶段流程，补上可读性和通用能力。核心论点是：推理能力不必靠人写的推理轨迹，只要有难题、可靠的 verifier 和足够的 RL 算力（§6）。

## 问题设定

- **形式**：LLM 就是策略 $\pi_\theta(o\mid q)$。输入是问题 $q$（数学、代码、STEM 选择题、逻辑题），输出是一整条 token 序列 $o$ = `<think>…</think><answer>…</answer>`（Table 1 模板）。一次生成算一个 episode，奖励只在序列末尾给一次（outcome reward），中间 token 没有奖励。状态转移是确定性的，也就是把已有 token 接上新 token。
- **假设**：最终答案能用规则校验。数学把预测答案与参考答案匹配，一致取 1、否则取 0（`sympy` 判等是 B.3.2 构造 cold-start 数据时用来过滤的，不是 RL 奖励的描述）；代码跑测试用例；STEM 选择题比对选项。证明题判对错太难，被排除在外（B.3.1）。base model 也要足够强（G.1）。
- **之前卡在哪**：
  1. 在人类 CoT 上做 SFT：依赖标注，能力上限被示范封住，而且人写的示范常常缺少显式的反思和验证步骤（§1、A.2）。
  2. neural reward model（ORM/PRM）在大规模 RL 中会被 reward hacking，重训 RM 又贵（§2.2）。PRM 还有两个难点：步骤难切分，中间步骤难标注（G.2）。
  3. MCTS 在 token 空间里搜索空间指数爆炸，细粒度 value model 很难训（G.2）。
  4. PPO 需要一个 value model，通常和 policy 差不多大（"of similar size"）。只有 outcome reward 时，要从部分回复预测最终奖励，这本身就很难。长 CoT 里前文会被后文推翻，就更难了（A.3）。

## 核心方法

**① GRPO**（`sn-article.tex` §2.1 Eq.1–3；附录 A.3 Eq.11–13 重复）。对每个 $q$，用 $\pi_{\theta_{old}}$ 采样一组 $G$ 个输出：

$$\mathcal J_{GRPO}(\theta)=\mathbb E_{q\sim P(Q),\thinspace\lbrace o_i\rbrace_{i=1}^G\sim\pi_{\theta_{old}}(O|q)}\frac1G\sum_{i=1}^G\Big(\min\big(\rho_iA_i,\ \mathrm{clip}(\rho_i,1-\varepsilon,1+\varepsilon)A_i\big)-\beta\thinspace\mathbb D_{KL}(\pi_\theta\Vert\pi_{ref})\Big),\quad \rho_i=\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)}$$

$$A_i=\frac{r_i-\mathrm{mean}(\lbrace r_1..r_G\rbrace)}{\mathrm{std}(\lbrace r_1..r_G\rbrace)},\qquad \mathbb D_{KL}=\frac{\pi_{ref}(o_i|q)}{\pi_\theta(o_i|q)}-\log\frac{\pi_{ref}(o_i|q)}{\pi_\theta(o_i|q)}-1$$

- **不要 critic**：baseline 取同一问题下 $G$ 个样本奖励的均值，再按标准差归一化。 $o_i$ 的所有 token 共用同一个 $A_i$（Figure 3 对比了 PPO 和 GRPO 的数据流）。
- **KL 放进 loss，而不是像 PPO 那样当 per-token dense reward**。A.3 的理由是：PPO 最大化累计回报，每个 token 扣一次 KL，等于隐式惩罚长度，会阻止回复变长。这里用 Schulman 的 k3 估计量。白板推导（我补的）：令 $x=\pi_{ref}/\pi_\theta$，在 $o\sim\pi_\theta$ 下 $\mathbb E[x]=1$，所以 $\mathbb E[x-\log x-1]=\mathbb E[\log\frac{\pi_\theta}{\pi_{ref}}]=\mathrm{KL}(\pi_\theta\Vert\pi_{ref})$。又因为 $x-\log x-1\ge0$，每个样本的估计都非负。（严格说，实际样本来自 $\pi_{\theta_{old}}$，只有 $\pi_\theta=\pi_{\theta_{old}}$ 时这个期望才严格等于 KL。）另外每 400 步把 $\pi_{ref}$ 重置为当前策略（§2.1；理由见 A.3），在探索范围和训练稳定之间折中。
- **退化组**（我的推导，论文没讨论）：若一组 16 个回答全对或全错，std=0， $A_i$ 无定义（分母加一个小常数后为 0），这个问题贡献不了 advantage 项的策略梯度，只剩 KL 项。所以只有"有时对、有时错"的题目才提供信号。

**② 奖励**
- R1-Zero： $Reward_{rule}=Reward_{acc}+Reward_{format}$，两项等权（Eq.4）。format 奖励要求推理放在 `<think>` 标签里。推理任务**刻意不用** neural RM（§2.2）。
- R1 额外加三类：
  - language consistency 奖励 $Reward_{language}=\frac{Num(Words_{target})}{Num(Words)}$（Eq.7）；
  - helpful RM（pairwise，66,000 对，只看最终 summary）；
  - safety RM（point-wise，106,000 条 prompt，看整条回复）（§3.1，Eq.5–6）。
  - 第二阶段 RL 的总奖励为 $Reward=Reward_{reasoning}+Reward_{general}+Reward_{language}$，其中 $Reward_{reasoning}=Reward_{rule}$， $Reward_{general}=Reward_{\verb|reward_model|}+Reward_{format}$（Eq.8–10）。每条通用 query 归属 safety 或 helpfulness 数据集之一，取对应 RM 的奖励（§3.1）。

**③ R1-Zero 训练配置**（§2.1）
- 基本设置：lr 3e-6，KL 系数 $\beta$=0.001，rollout 温度 1，每题 $G$=16。
- 最大长度：8.2k 步之前是 32,768 token，之后是 65,536。性能和长度都在 8.2k 步处跳升。
- 规模：共 10,400 步（1.6 epoch），每步 32 道不同的题，batch 512。
- 更新方式：每次 rollout 生成 8,192 条输出，随机切成 16 个 mini-batch，只过 1 个 inner epoch。
  - 我的换算：一次 rollout 是 512 题 × 16，对应 16 次参数更新。所以从第 2 个 mini-batch 起 $\pi_\theta\neq\pi_{\theta_{old}}$，ratio 和 clip 开始起作用。
- R1-Zero 的 $\varepsilon$ 正文没写。

**④ R1 四阶段**（Figure 2，§3，B.3，B.4）

| 阶段 | 做法 | 产物 |
|---|---|---|
| Cold start SFT | 用 R1-Zero 以温度 1.0 采样，保留答案正确且格式可读的轨迹；再让 DeepSeek-V3 润色推理和 summary，并把思考过程译成与问题同语言。人工参与方面（B.3.2，Figure 2 里的 "DeepSeek V3+Human"）：先由人工把推理轨迹改成对话风格（思考过程用第一人称），以此为示例让 LLM 改写更多数据，LLM 输出再经第二轮人工复核。共得到"数千条"长 CoT。SFT 超参：2–3 epoch，lr 5e-5→5e-6 cosine，上下文 32,768，batch 128 | Dev1 |
| 推理 RL | rule 奖励 + language 奖励。超参同 R1-Zero，但 **clip ratio $\varepsilon$=10**，最大长度 32,768（§3.2.1） | Dev2 |
| Rejection sampling + SFT | 从第一阶段 RL 的 checkpoint 做拒绝采样，得到约 600k 推理样本（部分题由 DeepSeek-V3 对照 ground truth 判分，相当于 generative RM）；再加约 200k 非推理样本（部分复用 V3 的 SFT 数据，另加软件工程类数据），共 804,745 条（Table 5）。**重新微调 V3-Base**（Figure 2） | Dev3 |
| 全场景 RL | 规则奖励 + RM。温度降到 0.7，共 1,700 步，**只在最后 400 步**才加入通用数据和 preference 奖励，原因是继续训会 reward hacking（§3.2.2） | R1 |

- 关于 $\varepsilon$=10，§3.2.1 的说法是：取小了会截断大量 token 的梯度，取大了训练不稳；作者贡献说明里也提到这是"large PPO clipping strategy"。我的分析：比率恒为正，所以下界 $1-\varepsilon<0$ 永不生效。结果是 $A<0$ 时完全不 clip， $A>0$ 时只在 $\rho>11$ 才截断，和 PPO 原意（[../ppo/notes.md](../ppo/notes.md)）的"小 trust region"已经相差很远。
- **Distillation**（F，B.4.3）：用这 800k 数据对 Qwen2.5 和 Llama 做纯 SFT，2–3 epoch，batch 64，不做 RL。
- **推理与评测**（D.1）：温度 0.6，top-p 0.95。pass@1 取 $k$ 个样本的平均正确率：AIME 和 GPQA 用 $k$=64，MATH 和 CF 用 16，LCB 用 8。
- **Infra**（B.1，Figure 5）：分四个模块。
  - Rollout 模块：vLLM，加跨节点 expert parallelism，再用 MTP 做 self-speculative decoding；
  - Inference 模块：负责 RM 和 ref 的前向；
  - Rule-based reward 模块：异步执行，把延迟和其他模块重叠掉；
  - Training 模块：做 sequence packing，配合 DualPipe。
  - 每个阶段结束后都把模型 offload 出显存。

## 关键实验与证据

- **R1-Zero 的学习曲线**（Figure 1）：AIME 2024 pass@1 从 15.6% 升到 77.9%，self-consistency 后达到 86.7%（§2.3；Figure 1 图例为 cons@16）。平均回复长度在约 10k 步里持续增长。
- **行为涌现**（Figure 9，C.2）：10 个反思词（wait、mistake、verify、check……）的频率增长了 5–7 倍。"wait"在前期几乎不出现，4000–7000 步偶尔出现，8000 步后陡增。Table 2 就是那段"aha moment"。按难度看（Figure 8，C.1），MATH level 5 从约 0.55 升到约 0.90。
- **主结果**（Table 8）：R1 与 OpenAI-o1-1217 对比。
  - 持平或略胜：AIME 2024 79.8 vs 79.2，MATH-500 97.3 vs 96.4，LiveCodeBench 65.9 vs 63.4，SWE Verified 49.2 vs 48.9。
  - 略逊：Codeforces rating 2029 vs 2061，GPQA 71.5 vs 75.7，Aider 53.3 vs 61.7。
  - 同一 base 的 DeepSeek-V3 在 AIME 上只有 39.2。
- **各阶段逐步消融**（Table 3）：
  - AIME：R1-Zero 77.9 → Dev1 59.0 → Dev2 74.0 → Dev3 78.1 → R1 79.8。cold start 数据量小，先掉点；推理 RL 再把分数拉回来。
  - IF-Eval：46.6 → 83.3；AlpacaEval2.0：24.7 → 87.6；ArenaHard：53.6 → 92.3。通用能力有两次大跳：cold start 之后（Dev1：IF-Eval 71.7、AlpacaEval 50.1、ArenaHard 77.0），以及最后一轮 RL（AlpacaEval 62.1→87.6、ArenaHard 75.6→92.3）。IF-Eval 的提升主要来自 cold start。
  - 注意 GPQA（75.8 vs 71.5）和 CNMO（88.1 vs 78.8）上 **R1-Zero 高于 R1**，可读性和对齐是有代价的。
  - LiveCodeBench Hard 题从 17.09 升到 34.44（Table 14）。
- **GRPO vs PPO**（Figure 4，A.3）：在 DeepSeek-Coder-V2-Lite（16B MoE，2.4B 激活）上跑 MATH。PPO 用 GAE $\lambda$=0.95 时明显差于 GRPO，调到 $\lambda$=1.0 后接近 GRPO。
- **Reward hacking**（Figure 6，B.5）：用 helpful RM 训练时，RM 分数整体上升（约 3.5→4.4），CodeForces test pass@1 在后期却下降（读图：约 0.34 降到第 700 步的约 0.29）。这是只在最后 400 步使用 RM 的依据。
- **LC 奖励消融**（Figure 7，在 R1-Distill-Qwen-7B 上做）：不加 LC 奖励，语言一致性逐步变差；加了以后数学基本持平，代码略降。
- **Distill vs RL**（Table 16）：Qwen2.5-32B 做了 >10K 步 RL 得到 Qwen2.5-32B-Zero，AIME 47.0，与 QwQ-32B-Preview 的 50.0 相当；从 R1 蒸馏出的 R1-Distill-Qwen-32B 是 72.6。Table 15 里 1.5B 蒸馏模型 AIME 28.9，GPT-4o 只有 9.3。
- **防"base 见过推理数据"的对照**（Table 17）：Qwen2-Math-7B 发布于 o1 之前，RL 约 10,000 步后，AIME 2024/2025 为 22.3/18.1，对应 Instruct 版只有 7.9/4.6。
- **模型规模门槛**（G.1）：7B dense 和 16B MoE 做 RL 时 AIME 没有实质提升，回复一长就开始重复；32B dense、230B MoE、671B MoE 才有效。
- **Test-time scaling**（Figure 18，E.4）：在 2024 年 366 道竞赛题上，R1 平均用 8,793 个 thinking token，pass@1 61.8%。简单题用不到 7,000 token，最难的题超过 18,000。GPT-4o 平均 711 token，24.7%。AIME 上 GPT-4o 做 maj@64 只从 9.3 升到 13.4。R1 自己 pass@64 是 90.0。
- **新题泛化**（Table 13）：AIME 2025 得分 11.3/15（o1 为 12.0/15）；AMC 12 2024 得 143.7；USAMO index 256.7，超过 251.5 的资格线。

## 局限与论文没说的

**作者承认的**（§6，B.3.3，D.1）：
- 结构化输出和工具调用弱；
- overthinking，简单题也会想太久；
- 中英以外的语言会出现 language mixing；
- 对 prompt 敏感，few-shot 反而掉点；
- SWE 任务评测太慢，RL 做得少；
- 写作这类任务没有可靠 RM，所以只做几百步 RL；
- SFT 数据以单轮为主；
- n-gram 去污染挡不住改写题，2024 年以前的 benchmark 可能已被污染。

**我读出来的**：
1. **"纯 RL 涌现"有混杂因素**：A.1 承认 V3-Base 的预训练语料里有部分网页含大量 OpenAI 模型生成的回答，数学和代码内容中也有大量推理轨迹。所以 R1-Zero 更像是把已有能力**激发**出来，而不是从零学会。Table 17 只是部分回应了这一点。论文还引了 Oat-Zero，那篇的标题本身就在质疑 aha moment。B.3.2 也承认，R1 那种生动的"I"式思考是 DeepSeek 设计出来的启发式风格，不代表模型具有类人智能。
2. **在 671B 规模上没有做算法消融**： $G$、 $\varepsilon$、 $\beta$、ref 重置、std 归一化的作用都没有拆开测。GRPO vs PPO 只在一个 16B 模型、一个任务上比过，种子数也没报告。
3. **公式层级含糊**：Eq.1 写的是序列级 ratio，但 §3.2.1 讲 clip 时说"大量 token"，暗示实际按 token 级实现，论文没有说清楚。
4. **数据账对不上**（我的算术）：10,400 步 × 32 题 ÷ 1.6 epoch ≈ 208K 道题，但 Table 4 推理类 prompt 合计只有 26K+17K+22K+15K = 80K（B.3.1 正文另提 8k bug fixing 题，也只有 88K）。R1-Zero 到底用了多少题没交代。此外 §B.4.4 写"约 4 天，即约 80 小时"，这句本身就不自洽；Table 7 的 41K GPU·h ÷ 512 卡 ≈ 80 h。
5. **评测口径**：o1-1217 的分数直接抄官方报告，协议不同。AlpacaEval 和 ArenaHard 由 GPT-4-Turbo 当 judge，而且只给它看 summary。
6. **只看 outcome 的奖励不检查推理是否忠实**：过程错、答案碰巧对，照样拿奖励。
7. **退化组、std 归一化与长度之间的偏差论文都没分析**。后来 DAPO、Dr. GRPO（不在本项目内）专门讨论了这些问题。
8. **数据没有公开**：附录里写的是"release SFT and RL data … at xxx"，地址是个占位符。
9. **和机器人设定差别很大**：这里 episode 只有一次生成，转移是确定性的，verifier 几乎零成本，而且同一 $q$ 可以随意重采 16 次。真实机器人三条都不满足。

## 在脉络中的位置

- **前驱**：
  - [PPO](../ppo/notes.md)：提供 clipped surrogate 和 GAE；
  - DeepSeekMath：GRPO 的出处（arXiv 2402.03300，不在本项目内）；
  - InstructGPT 式 RLHF；
  - CoT、self-consistency、STaR；
  - PRM（Lightman et al.）、AlphaGo/AlphaZero 式 MCTS：两条都尝试过，都失败了（G.2）；
  - 通论可参考 [RL overview](../survey-rl-overview/notes.md)。
- **后继 / 影响**：
  - 论文自己引了 TinyZero、Oat-Zero、Open-R1 这些复现（H.3）。
  - 本项目里：
    - [TTRL](../../02_test_time_training/ttrl/notes.md)：用 majority vote 当伪奖励，在测试集上做 GRPO；
    - [TTT-Discover](../../02_test_time_training/ttt-discover/notes.md)：针对单个问题做 test-time RL；
    - [SimpleVLA-RL](../simplevla-rl/notes.md)：GRPO 加二值成功奖励，用在 VLA 上；
    - [π_RL](../pi-rl/notes.md)：把 PPO/GRPO 用到 flow VLA；
    - [WMPO](../../05_intersections/wmpo/notes.md)：在视频 world model 里做 GRPO；
    - [World-Gymnast](../../05_intersections/world-gymnast/notes.md)：用 VLM 当 reward，在 world model 里做 RL。
- **与本项目其他主题的联系**：
  - **TTT**：R1 是冻结权重、靠加长 CoT 来 scale test-time compute。H.2 把它和更新权重的 TTT 并列：[TTT](../../02_test_time_training/ttt-self-supervised/notes.md)、[TTT-ARC](../../02_test_time_training/ttt-arc-fewshot/notes.md)。TTRL 则把两者合在一起，综述见 [survey](../../02_test_time_training/survey-test-time-intelligence/notes.md)。
  - **World model**：GRPO 需要从**同一初始状态**做 $G$ 次 rollout，这要求环境能 reset 到同一状态。真实世界做不到，learned world model 可以（WMPO）。另一条路是 [DreamerV3](../../01_world_model/dreamerv3/notes.md) 那样在 imagination 里训练 critic。两者体现的是 critic-free 与 critic-based 的取舍。
  - **Real2sim2real**：仿真器里的成功判定，本质上就是 verifiable reward。Figure 6 那种 RM 被 hack 的现象，和策略钻仿真器漏洞是同一类问题（[reality gap 综述](../../03_real2sim2real/survey-reality-gap/notes.md)）。
  - **RL**：Figure 4 中 $\lambda$=1 更好（此时 GAE 等于 Monte Carlo 回报减 value baseline，不做 bootstrapping）。我的解读是：在长 horizon、只有末端稀疏奖励时，value bootstrapping 不可靠；论文只说 PPO 对 $\lambda$ 敏感，没给这个解释。机器人 locomotion 的设定不同，是 dense reward 加 PPO/GAE，参见 [legged-gym](../../03_real2sim2real/legged-gym/notes.md)。

## 前置知识

1. Policy gradient 与带 baseline 的 REINFORCE、advantage 的含义。
2. PPO 的 importance ratio 与 clip；GAE 中 $\lambda$ 对 bias-variance 的影响。
3. RLHF：Bradley-Terry 式 reward model、KL-to-reference 正则、KL 估计量 k1/k3、reward hacking / over-optimization。
4. 自回归 LLM 作为策略；SFT、rejection sampling（STaR / RFT）；distillation。
5. CoT、self-consistency（majority vote）、pass@k 与 cons@k。
6. （可选）MoE、vLLM、speculative decoding：用于看懂 infra。

## 复现可行性

- **开源情况**：只放了权重（R1、R1-Zero、6 个 distill 模型）和推理代码。训练代码和 RL/SFT 数据都没有公开，附录里的数据地址写的是"xxx"。
- **原规模的算力**（Table 7，B.4.4）：
  - R1-Zero：64×8 H800，约 198 h，101K GPU·h；
  - SFT 数据构造：5K GPU·h；
  - R1：41K GPU·h；
  - 合计 147K H800 GPU·h，按 \$2/GPU·h 折合约 \$294K；
  - 前期先在 A100 上用约 30B 的模型做探索。
  - 这个规模不可复现。
- **最小可复现实验**：做 R1-Zero 式实验，不做 R1 全流程。
  - 模型：base 选 Qwen2.5-Math-1.5B/7B 这一类。注意 G.1 警告 7B dense 可能不涨，Table 17 的 Qwen2-Math-7B 却有效，所以选数学特化的 base 更稳。
  - 数据与模板：带数值答案的数学题，套 Table 1 模板，奖励 = 答案匹配（0/1）+ 格式。
  - 超参：尽量照搬论文， $G$=16， $\beta$=0.001，温度 1，lr 在 2e-6 到 3e-6 之间（B.4.1），每 400 步重置 ref。
  - 监控：reward、平均回复长度、C.2 那 10 个反思词的频率、MATH-500 pass@1（温度 0.6， $k$=16）。
  - 消融：
    - (a) GRPO vs PPO（ $\lambda$=0.95 / 1.0），对照 Figure 4；
    - (b) KL 放进 loss vs 作为 per-token reward，看长度是否被抑制；
    - (c) 去掉 std 归一化；
    - (d) 不同 $\varepsilon$。
  - 工具：可以用 TinyZero / Open-R1（论文 H.3 引用），也可以用 veRL、TRL 的 GRPO 实现（这两个是论文外的工具）。论文没给小规模实验的算力。
- **预计难点**：
  - 长序列 rollout 的吞吐（32K–64K token），这是主要瓶颈；
  - 退化组太多时，有效 batch 会缩小，需要按难度筛题；
  - 代码题的测试用例要自己造，参考 B.3.2 的两阶段过滤；
  - base 能力不够时，回复会重复甚至崩掉；
  - 大 $\varepsilon$ 可能让训练不稳。

## 自测问题

1. 写出 GRPO 的 $A_i$。为什么它可以替代 value model？一组 16 个回答全对时梯度是多少？这对 RL 训练集的难度分布提出了什么要求？
2. 推导 k3 估计量的期望等于 $\mathrm{KL}(\pi_\theta\Vert\pi_{ref})$。为什么"KL 放进 loss"比"per-token KL 作为 reward"更有利于回复变长？
3. Figure 4 中 PPO 取 $\lambda$=1.0 为什么比 0.95 好？从 GAE 的 bias-variance 与 outcome-only reward 两个角度解释。
4. R1 的四个阶段各自修的是 R1-Zero 的什么问题？Table 3 中 Dev1 的 AIME 为什么低于 R1-Zero？第三阶段的 SFT 数据从哪来，微调的是哪个 checkpoint？
5. 为什么 preference RM 只在最后 400 步使用？结合 Figure 6，说明规则奖励为什么更难被 hack。如果把机器人 RL 的 reward 换成 VLM 打分，会出现哪种对应风险？
