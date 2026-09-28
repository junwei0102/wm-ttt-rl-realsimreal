# Learning to Discover at Test Time

> arXiv [2601.16175](https://arxiv.org/abs/2601.16175) · ICML 2026（Spotlight，已在 OpenReview 核实；PDF 为 arXiv v2，2026-02-05，正文没写会议名） · 主题：02_test_time_training · 层级：frontier
> 代码：https://github.com/test-time-training/discover（MIT，已打开确认存在；`pip install ttt-discover`；项目页 https://test-time-training.github.io/discover） · 笔记依据：`source/main.tex`（全文单文件，数字/公式）+ `source/prompts_and_code/*.tex`（prompt 与发现的代码）+ `paper.pdf`（图）

## 一句话

针对单个科学/工程问题，在测试时用 RL 继续训练 LLM（gpt-oss-120b + LoRA）。因为 discovery 只要求找到一个超过 SOTA 的解，不要求 policy 的平均表现好，所以作者把训练目标换成偏向最大值的 entropic objective（$\beta$ 由 KL 预算自适应确定），并把起点选择换成以"子代最大 reward"作 $Q$ 的 PUCT state reuse。这个方法叫 TTT-Discover。

## 问题设定

- **形式化**（`source/main.tex` §2.1）：问题以文本描述 $d$ 给出，$d$ 始终作为 policy 的 context。state $s$ 是一个候选解，比如一个 kernel 实现或一个阶梯函数。$d$ 诱导出连续 reward $R(s)\in\mathbb R$ 和 transition。记已知最好解为 $s_{\text{sota}}$、$r_{\text{sota}}=R(s_{\text{sota}})$。**Discovery** 定义为找到满足 $R(s)>r_{\text{sota}}$ 的 $s$，差值越大，发现越重要。
- **MDP**：action $a\sim\pi_\theta(\cdot\mid d,s)$ 由 thinking tokens 加一段代码组成。coding 类任务的 transition 是 $s'=\texttt{Parse}(a)$，数学类任务是 $s'=\texttt{Python}(\texttt{Parse}(a))$，即先执行代码（Table 1）。只考虑单步；多步由 state reuse 等效实现（§2.1）。无效解的 reward 为 0（Table 1 caption）。
- **假设**：reward 连续、可验证、可以反复调用（每个问题约 25600 次评估）；评测器就是真值，不存在 reward model 误差。输出不是 policy，而是 buffer 里 reward 最高的那个 state（Algorithm 1 最后一行）。
- **与标准 RL 的两点区别**（§1、§3.1）：(1) 只需解这一个问题，不要求泛化，这是 TTT 的老主题；(2) 只需要一个最好的解，policy 只是手段，这一点是 discovery 独有的。因此没有部署阶段，policy 不需要在初始状态分布上稳健。
- **之前卡在哪**：
  - AlphaEvolve、ShinkaEvolve、OpenEvolve 这类 evolutionary search（论文称之为 state-action reuse）对 frozen LLM 做 prompting。积累的经验只能改进下一个 prompt，不能改进 $\theta$，而且依赖手工设计的 mutation/crossover/fitness 启发式。
  - 直接套用 PPO/GRPO 做 naive test-time RL 有三个问题（§3.1）：
    - **Objective**：优化的是平均 reward，对 SOTA 不敏感。例如 2000 μs 到 1900 μs 可能是一次突破，但两者得到的 reward 几乎一样。
    - **Short effective horizon**：每次从头开始，单次尝试能走的步数有限。
    - **Exploration**：policy 会塌缩到安全、高均值的动作；reuse 如果只按 reward 排序，又会过度开发少数几个 state。

## 核心方法

**0. 搜索基线的统一写法**（§2.2）。设 buffer 为 $\mathcal H_i$。
- Best-of-$N$：$s=s_{\text{sota}}$ 或 $\texttt{<empty>}$，$a_i\sim\pi_\theta(\cdot\mid d,s)$，各 rollout 独立同分布。
- State reuse：$s_i\sim\texttt{reuse}(\mathcal H_i)$，$a_i\sim\pi_\theta(\cdot\mid d,s_i)$，$\mathcal H_{i+1}=\mathcal H_i\cup\{(s'_i,r_i)\}$。复用一个旧解，就等于给那条轨迹多加了一个 timestep。
- State-action reuse（即 evolutionary search）：reuse 还会把旧 action 里的信息（thinking、中间结果）整理成自然语言 context $c_i$。

**1. 总框架**（Algorithm 1）。对 $i=0,\dots,N-1$ 循环：
- $s_i,c_i\sim\texttt{reuse}(\mathcal H_i)$
- $a_i\sim\pi_{\theta_i}(\cdot\mid d,s_i,c_i)$
- $s'_i=T(a_i)$，$r_i=R(s'_i)$
- 把 $(s_i,a_i,s'_i,r_i)$ 加入 $\mathcal H$
- $\theta_{i+1}=\texttt{train}(\theta_i,\cdot)$

最后返回 $\arg\max_i r_i$ 对应的 state。naive RL 等价于 $\texttt{train}$ 取期望 reward 的 policy gradient、$\texttt{reuse}=\delta_{\texttt{<empty>}}$（§3.1）。TTT-Discover 取（§3.2）

$$\texttt{train}:\ \theta_{i+1}=\theta_i+\eta\nabla_\theta J_{\beta(s_i)}(\theta_i),\qquad \texttt{reuse}:\ s_i\sim\mathrm{PUCT}(\mathcal H_i).$$

**2. Entropic objective**（§3.2；App. A.1）

$$J_\beta(\theta)=\mathbb E_{s\sim\texttt{reuse}(\mathcal H)}\Big[\log\mathbb E_{a\sim\pi_\theta(\cdot\mid s)}\big[e^{\beta(s)R(s,a)}\big]\Big]$$

对内层求导（这里把 reuse 分布视为常数），$\nabla\log\mathbb E[e^{\beta R}]=\mathbb E[e^{\beta R}\nabla\log\pi]/\mathbb E[e^{\beta R}]$，于是

$$\nabla_\theta J_\beta=\mathbb E\big[w_{\beta(s)}(a)\,\nabla_\theta\log\pi_\theta(a\mid s)\big],\qquad w_{\beta(s)}(a)=\frac{e^{\beta(s)R(s,a)}}{\mathbb E_{\pi_\theta}[e^{\beta(s)R(s,a)}]}.$$

因为 $\mathbb E[w]=1$，用 $-1$ 作 baseline。再加上对初始模型的 KL shaping，advantage 为

$$A(a;s)=w_{\beta(s)}(a)-1-\lambda\log\frac{\pi_\theta(a\mid s)}{\pi_{\theta_0}(a\mid s)}.$$

$\lambda=0.1$，只有 AtCoder 用 $0.01$（App. A、Table 9）。

- **直觉**（笔者补的推导）：展开得 $\log\mathbb E[e^{\beta R}]\approx\beta\mathbb E[R]+\tfrac{\beta^2}{2}\mathrm{Var}[R]$，所以 $\beta\to0$ 时 $J_\beta/\beta$ 退化为期望 reward；论文说明 $\beta\to\infty$ 时趋于 $\max$。$w$ 实际上就是按 reward 做 softmax 加权，高 reward 的 rollout 拿走绝大部分梯度。
- 并发工作 jiang2025risk 用同一个 objective 在训练期优化 pass@k，推荐 $\beta=2$。本文发现常数 $\beta$ 不好设：训练早期 $\beta$ 太大会不稳定；训练后期改进越来越小，$\beta$ 小了 advantage 会消失。

**3. 自适应 $\beta$**（App. A.1）。定义 tilted distribution $q_\beta(\tau\mid s)=\pi_\theta(\tau\mid s)e^{\beta r}/\mathbb E_{\pi_\theta}[e^{\beta r}]$，于是 $w_\beta=q_\beta/\pi_\theta$，$\beta$ 就相当于这次重加权的"步长"。仿照 REPS，取 $\beta(s)$ 使 $\mathrm{KL}(q_{\beta(s)}\|\pi_\theta)=\gamma$，全文固定 $\gamma=\ln2$。

在 batch 上的做法：同一 $s$ 下有 $N$ 条 rollout，经验分布是均匀的 $u(n)=1/N$。令 $q_\beta(n)=e^{\beta r_n}/\sum_m e^{\beta r_m}$，在 $\beta\ge0$ 上二分求解

$$\mathrm{KL}(q_\beta\|u)=\sum_n q_\beta(n)\log(Nq_\beta(n))=\gamma .$$

然后算 leave-one-out entropic advantage：

$$\hat Z_{-n}=\frac{1}{N-1}\sum_{m\neq n}e^{\hat\beta(s)(r_m-r_{\max})},\qquad A_n=\frac{e^{\hat\beta(s)(r_n-r_{\max})}}{\hat Z_{-n}+\varepsilon}-1 .$$

几点理解：
- 笔者补的推导：$\mathrm{KL}(q\|u)=\log N-H(q)$，所以 $\gamma=\ln2$ 等价于 $e^{H(q)}=N/2$。也就是说，不管 reward 的尺度如何，权重的"有效样本数"总是固定在组内的一半，本文组大小为 64，即 32 条。
- 论文指出的性质：
  - 这个估计量对 reward 的仿射变换 $wr+b$（$w>0$）不变，所以 1/runtime、分数、1/MSE 这些尺度差别很大的任务可以共用同一个 $\gamma$。
  - 改进一直很小的 state（论文举例：高 value / 接近目标的 state）会被允许用更大的 $\beta$。
  - 偶尔出现大幅跳变的 state（论文说常见于训练早期，或还有很大提升空间的低 value state）会被压成更小的 $\beta$，避免更新被少数离群 rollout 主导。

**4. PUCT reuse**（§3.2；App. A.2）。把"从 archive 里挑一个起点"看作虚拟根节点上的一个 action，打分为

$$\text{score}(s)=Q(s)+c\cdot\text{scale}\cdot P(s)\frac{\sqrt{1+T}}{1+n(s)} .$$

各项含义：
- $Q(s)$：从 $s$ 出发生成的子代中的**最大** reward $m(s)$；$s$ 还没被展开时取 $R(s)$。
- $P(s)\propto|\mathcal H|-\text{rank}(s)$：按 reward 线性排名的先验。直觉是快的 kernel 更可能孕育更快的 kernel。
- $n(s)$：访问计数，会回传给**所有祖先**。
- $T$：已展开的总次数。
- $\text{scale}=R_{\max}-R_{\min}$。
- $c=1.0$。

维护规则：
- 每展开一个父节点，只保留 reward 最高的 2 个子代入库。
- 全局只保留 reward 前 1000 的 state，初始 seed 永远保留。
- 同一个 batch 内屏蔽整条 lineage（祖先和后代），以保证多样性。

与 AlphaZero 的 PUCT 有四点不同：用 max 而不是 mean；先验是 rank-based 而不是学出来的；计数回传整条 lineage；用 lineage blocking 代替 virtual loss。

**5. Context $c_i$ 实际是什么**（`source/prompts_and_code/*_prompt.tex`，各任务不完全一样）。prompt 里除了问题描述，还可能有：
- "Here is the last code we ran"（被复用解对应的代码；AC1/AC2、TriMul、AHC 的 prompt 有，附录给出的 Erdős prompt 没有，初始构造以 `initial_h_values` 预先导入）；
- 当前分数：AC1/AC2 给出运行前后的 bound 和截断后的 "Previous Program Output"（stdout）；kernel 与 AHC 给出当前 runtime/分数和与 target 的 gap；
- 一个目标值。例如 Erdős 的 prompt 写明当前记录 $C_5\le0.38092$、目标 $0.38080$；AC1 和 AC2 的 prompt 还附上了"A previous state of the art used the following approach"，AC1 那段是基于 LP 的迭代描述。

从这些 prompt 看，实现上复用的是 state 加上一部分 action 信息（代码和输出），没有出现 thinking tokens。

**6. 训练/推理怎么走**（§3.3、Table 9）。测试时的"训练"本身就是搜索，没有单独的推理阶段。
- 模型与平台：gpt-oss-120b，reasoning effort 设为 high，在 Tinker API 上训练 LoRA（rank 32），共 50 步。
- batch：每步 512 条 rollout，分成 8 组、每组 64 条。每组共享 PUCT 选出的同一个起点和 context，自适应 $\beta$ 和 LOO baseline 都在组内计算。
- 更新：每个 batch 只做 1 步梯度，没有 off-policy 步。用 importance sampling ratio 修正 sampler/learner mismatch。
- 优化器与采样：Adam，lr $4\times10^{-5}$，$\beta_1=0.9$、$\beta_2=0.95$、$\epsilon=10^{-8}$；temperature 1.0。
- 长度控制：context 32768。大多数领域把 prompt 加 thinking 限制在 26000 token，超出后用 token forcing 插入 "... okay, I am out of thinking tokens..." 强制模型给出最终答案。
- 例外（App. D、E）：AHC039 的该限制降到 22000，AHC058 降到 25000 且 lr 改为 $2\times10^{-5}$；denoising 的 max tokens 为 20000。
- 基线预算：Best-of-$N$ 取 $N=50\times512=25600$，与本方法的采样预算相同。

## 关键实验与证据

**评测口径**（§4）：每个任务都报告 best human 和 best AI。并用同一模型、同一采样预算跑 Best-of-25600 和 OpenEvolve（同样 25600 个样本）。作者提醒，OpenEvolve 的 prompt 很长，在 32k context 下大量 rollout 被截断，但为了忠于原实现没有修改。

**数学**（Table 2、Figure 2/3）：
- **Erdős minimum overlap**（↓）：TTT-Discover 为 **0.380876**。对比 AlphaEvolve / V2 的 0.380924、best human 的 0.380927、OpenEvolve 的 0.380965、Best-of-25600 的 0.380906。注意 Best-of-25600 自己也超过了 AlphaEvolve。本文相对 AlphaEvolve 的改进量是 AlphaEvolve 相对前 SOTA 改进量的 16 倍（§4.1.1）。得到的构造是一个 600 段的非对称阶梯函数；AlphaEvolve 的是 95 段，人类最好结果是 51 段（Figure 2）。
- **AC1**（↓）：**1.50287**（Table 2；正文和 Figure 3 写的是 1.50286），为 30000 段、从零开始搜出来的。此前 ThetaEvolve 在 AlphaEvolve V2 构造的基础上精修，得到 1.50314（1319 段）。
- **AC2**（↑）：0.9591，低于 AlphaEvolve V2 的 0.9610，没有做出 discovery。
- **Qwen3-8B 版本**：用于与 ThetaEvolve 对比。AC1 为 1.50525，ThetaEvolve 为 1.50681；AC2 为 0.9472，ThetaEvolve 为 0.9468。本文用 50×512 个样本，ThetaEvolve 用 65×512（32 组×16），且 ThetaEvolve 用的是 DeepSeek-R1-0528-Qwen3-8B，论文称原版 Qwen3-8B 据报道比它弱（§4.1.2）。
- **Circle packing**（Table 3，Qwen3-8B）：n=26 得 2.635983，n=32 得 2.939572，只是追平已知最好值。

**Kernel（GPUMode）**（Table 4/5/10）：
- **TriMul**：训练时只用 H100 的 runtime 作 reward，得到的 kernel 迁移到其他 GPU 同样有效。与第 1 名人类对比（μs，↓）：

  | GPU | TTT-Discover | 第 1 名人类 |
  |---|---|---|
  | A100 | **2198.2** | 4531.5 |
  | H100 | **1161.2** | 1371.1 |
  | B200 | **914.2** [907.3, 921.1] | 1038.9 |
  | MI300X | **1555.7** | 2515.8 |

  其中 A100 和 H100 的数字是正式 leaderboard 提交返回的结果。按表中数字算，A100 上约快 2.06×（摘要写 "up to 2×"，§4.2 写 "50% faster"）。Best-of-25600 在 A100 和 H100 上是 9219.7 / 5390.3，比第 5 名人类还差。
- **Kernel 挑选协议**（App. C.1）：每个任务先按训练 verifier 选出 top-20；目标硬件与训练硬件不同时，另外每 10 步加入 20 个随机的正确 kernel。然后在目标硬件上各测 3 次，取平均最快的那个。
- **MLA-Decode（MI300X）**：训练时用 H200。在 3 个实例上的结果为 1669.1 / 1706.1 / 1671.3，第 1 名人类为 1653.8 / 1688.6 / 1668.7。论文明确说没有一个实例在统计意义上优于人类（Table 5）。只保留 Triton kernel 时结果更慢，为 1740.6 / 1754.4 / 1707.1（Table 10）。

**AtCoder**（Table 6）：
- ahc039：和 ShinkaEvolve 一样从 ALE-Agent 的程序起步（该程序相当于第 5 名水平）。TTT-Discover 得 **567,062**，第 1 名人类 566,997，ShinkaEvolve 558,026，Best-of-25600 554,171。
- ahc058：从零开始。TTT-Discover 得 **848,414,228**，ALE-Agent 848,373,282，第 1 名人类 847,674,723，Best-of-25600 772,429,752。
- 训练时用本地 150 个 case（seed 0–149），只要有一个 case 失败 reward 就是 0。最终取 top-3 程序提交到官方隐藏测试集（App. D）。

**单细胞 denoising**（Table 7）：
- 训练用 Pancreas 数据集，报告在 held-out 的 PBMC 和 Tabula 上的 Score：TTT-Discover 为 **0.71 / 0.73**，MAGIC (A,R) 为 0.64 / 0.64，OpenEvolve 为 0.70 / 0.71。
- 验证时拒绝 normalized Poisson 不在 [0.97, 1] 内的解，因此实际只优化 MSE（App. E）。作者附了 disclaimer：benchmark 指标不等于生物学有效性。

**消融**（Table 8、Figure 4；TriMul H100，用作者自己的 evaluator，报告每个配置 run 中的 best kernel runtime，μs；论文未报告重复次数）：

| train | reuse | runtime |
|---|---|---|
| adaptive entropic | PUCT | **1203.10** |
| constant $\beta$（$\beta=2$，取自 jiang2025risk） | PUCT | 1483.83 |
| expected reward | PUCT | 1985.67 |
| No TTT | PUCT | 2060.70 |
| adaptive entropic | $\epsilon$-greedy（$\epsilon=0.1$） | 1328.89 |
| adaptive entropic | no reuse | 5274.03 |
| expected reward（naive test-time RL） | no reuse | 5328.73 |
| No TTT（Best-of-$N$） | no reuse | 5352.36 |

（best human 为 1371.1）

- 从表里能读出三点：
  - **reuse 是前提**：没有 reuse 时，即使用 entropic 训练也和 Best-of-N 差不多。这对应前面的 horizon 问题。
  - **在有 reuse 的前提下，学习本身很关键**：No TTT 为 2060.70，full 为 1203.10。
  - **objective 的形式也很关键**：runtime 从 expected reward 的 1985.67 到 constant $\beta$ 的 1483.83，再到 adaptive 的 1203.10，依次变快。
- PUCT 相对 $\epsilon$-greedy 只好一点，论文把 $\epsilon$-greedy 的表现部分归因于早期碰巧出现的好 kernel，并说其他任务的早期实验更缺探索，但没有给出数字。
- Figure 4 显示：constant $\beta$ 在后期停滞；No TTT 的 mean reward 基本不涨；max reward 在末段越过 "Previous SOTA" 虚线的是 full 方法和 Greedy（$\epsilon$-greedy）两条曲线，与 Table 8 里二者都快于 best human 的 1371.1 一致，其中 full 方法越过得更多。

## 局限与论文没说的

**作者承认的**：
- 只适用于连续 reward，sparse/binary reward 和不可验证领域留作未来工作（§6）。
- AC2 没做出 discovery，circle packing 只是追平，MLA-Decode 没有显著胜出。
- 消融配置没有单独调参，未测 PUCT 系数、lr、batch 的敏感性（§4.5）。
- GPUMode 评审指出，fp16 激活在完整工作负载里可能有数值稳定性问题。作者自己在 App. C.2 也承认，TTT kernel 对 block size 的 auto-tuning 比最好的人类 kernel 少。
- 生物学评审指出，benchmark 上的提升不一定能转化为新的生物学发现。

**笔者读出来的**：
1. **没有方差**：每个任务的主结果似乎都只有一次运行，论文没报告 run 数和 seed 间方差；消融同样没有报告重复运行，而且只在 TriMul 一个任务上做。"best-of-run" 本身就是极值统计，单次运行的可信度有限。
2. **数字前后不一致**：
   - H100 runtime：Table 4 为 1161.2，Table 8 为 1203.10。前者是 leaderboard 结果，后者是作者自己的 evaluator，但没说明是不是同一次 run。
   - AC1：1.50287 与 1.50286。
   - 数学任务的时限：§4.1 写 10 分钟，App. B 写最长 1100 s，AC1 的 prompt 写 1000 s。
   - Figure 4 的 caption 写 $N=50\times512=256000$，应为 25600。
3. **prompt 注入了人类先验**：prompt 里有目标值，AC1 和 AC2 还附了前 SOTA 的方法描述（AC1 是 LP 迭代），而 §4.1.2 说 policy 从 1.510 降到 1.504 "mostly used LP"。作者强调"never intervened"指的是运行过程中没有干预，但 prompt 设计的贡献没有消融。
4. **与其他 AI 方法的比较没有对齐算力**：AlphaEvolve、ShinkaEvolve、ALE-Agent 用的模型和预算都不同。真正对齐预算的只有 Best-of-25600 和 OpenEvolve，而 OpenEvolve 又受截断影响。
5. **算力口径不完整**："约 \$500/run" 是按平均 3000 prompt token、16000 采样 token 估算的 Tinker 费用，不含 evaluator 开销：TriMul 在 Modal 上的 H100 计时、数学任务每条 rollout 2 CPU 且最长 1100 s、AHC 每条跑 150 个 case。训练 GPU 的型号、数量和 wall-clock 都藏在 Tinker 后面，论文没报告。
6. **每个问题都从头学、学完就扔**：得到的 policy 是否对同类问题有迁移价值没有测试。唯一的"迁移"是解本身跨硬件：H100 训练出的 kernel 在 A100/B200/MI300X 上成立，而 H200 训练、目标 MI300X 的 MLA 没有显著超过第 1 名人类。这可能说明代理 evaluator 与目标 evaluator 之间的 gap 会影响结果能否迁移，但两者任务也不同，论文没有做对照。
7. **reward hacking 的防护靠人工写约束**：denoising 要加 Poisson 区间约束，kernel 依赖官方的正确性检查。entropic objective 会放大任何能被钻空子的地方，evaluator 一旦不精确，风险比优化期望 reward 更高。
8. **超参只有一组**：$\gamma=\ln2$、$c=1.0$、LoRA rank 32、50 步都是固定的，没有给出"更多步数是否持续有效"的 scaling 曲线。

## 在脉络中的位置

- **前驱**：
  - TTT 的定义"对每个测试实例构造一个新的学习问题"沿用 sun2023learning，与 [TTT with self-supervision](../ttt-self-supervised/notes.md) 同属 Yu Sun 这条线。
  - evolutionary search：AlphaEvolve / ShinkaEvolve / OpenEvolve。
  - LLM RL：[PPO](../../04_rl/ppo/notes.md)、GRPO（[DeepSeek-R1](../../04_rl/deepseek-r1/notes.md)）。
  - entropic objective：并发工作 jiang2025risk；自适应温度借鉴 REPS（Peters 2010）；PUCT 来自 AlphaGo/AlphaZero。
  - Bello 2016 的 neural combinatorial optimization 已经做过 per-instance 的 policy gradient。
  - 并发工作：EvoTune、MiGrATe、ThetaEvolve。
  - related work 里对照的 TTT 形态：[TTT for few-shot / ARC](../ttt-arc-fewshot/notes.md)（生成增强数据后做 SFT）、[TTRL](../ttrl/notes.md)（在整个测试集上用多数投票当伪标签）。
- **后继 / 影响**：本项目 related_work 中暂无直接建立在它之上的工作。[测试时智能综述](../survey-test-time-intelligence/notes.md)把它放在 Test-Time Reinforcement Learning 下的 "task-grounded rewards" 一类（问题自带 objective 或可执行 evaluator），并称它是 "search-derived task rewards" 的代表。related work 的叙述借用了同组的 [TTT-E2E](../ttt-e2e/notes.md)（tandon2025end）。
- **与本项目其他主题的联系**：
  - **RL**：权重 $w\propto e^{\beta R}$ 在形式上和 [IQL](../../04_rl/iql/notes.md) 做 policy extraction 时的 $\exp(\beta A)$ 加权（AWR）同源，只是目的相反：这里是偏向极值，那里是在数据支撑内提升策略。它和 [HIL-SERL](../../04_rl/hil-serl/notes.md)、[SimpleVLA-RL](../../04_rl/simplevla-rl/notes.md)、[π_RL](../../04_rl/pi-rl/notes.md) 这些面向单任务 / 机器人的 RL 形成鲜明对照：后者要部署 policy，关心的是平均成功率，所以 TTT-Discover 的"policy 只是手段"不能直接搬过去。
  - **real2sim2real**：system identification、仿真参数拟合（[SimOpt](../../03_real2sim2real/simopt/notes.md)、[ASAP](../../03_real2sim2real/asap/notes.md)、[PhysTwin](../../03_real2sim2real/phystwin/notes.md)）在结构上正好是 discovery 形态：单个实例、连续误差、只需要一个最好的参数或代码。这是笔者的推测，论文没测。kernel 从 H100 迁移到其他 GPU、MLA 从 H200 到 MI300X 失败，可以类比为"代理 evaluator 与目标环境之间的 gap"。[RIALTO](../../03_real2sim2real/rialto/notes.md) 在重建出的单个场景 digital twin 里做 RL，也是 per-instance 训练，只是产出的是 policy。
  - **world model**：TTT-Discover 默认 evaluator 精确。[WMPO](../../05_intersections/wmpo/notes.md)、[World-Gymnast](../../05_intersections/world-gymnast/notes.md)、[Ctrl-World](../../05_intersections/ctrl-world/notes.md) 则用学出来的 world model 充当 simulator 或 evaluator。如果把偏向 max 的 entropic objective 放到 learned simulator 上，它会优先去找模型误差，即 model exploitation。这是两条线交汇时首先要解决的问题。PUCT 也把 model-based planning 里的 MCTS 思想搬到了"选起点"这一步。

## 前置知识

1. Policy gradient / REINFORCE、baseline 与 advantage，GRPO 的组内 baseline，对 reference policy 的 KL penalty（[PPO](../../04_rl/ppo/notes.md)、[DeepSeek-R1](../../04_rl/deepseek-r1/notes.md)）。
2. 指数倾斜、log-sum-exp、softmax 温度，KL 与熵的关系（$\mathrm{KL}(q\|u)=\log N-H(q)$），REPS 的 KL 约束温度。
3. MCTS / PUCT / UCB 的 exploration bonus（AlphaZero）。
4. LLM 的 RL 基础设施：LoRA、on-policy 更新、sampler/learner mismatch 与 importance sampling 修正。
5. Test-time scaling 的搜索基线：Best-of-N、AlphaEvolve 式的 evolutionary search。
6. TTT 的基本概念（[TTT with self-supervision](../ttt-self-supervised/notes.md)）。
7. （可选）各应用领域的背景：阶梯函数的 autoconvolution、Triton kernel fusion、AHC 的计分规则。

## 复现可行性

- **代码**：官方仓库 https://github.com/test-time-training/discover，MIT 许可，`pip install ttt-discover`。需要 `TINKER_API_KEY`、`HF_TOKEN`、W&B。README 列出的环境有 Erdős、AC1/AC2、TriMul（A100/H100/B200/MI300X）、AtCoder（ahc039/ahc058）、denoising；`results/` 下放了所有 SOTA 构造、kernel 和程序，可以直接验证。
- **算力**：论文只给了 Tinker 费用，约 \$500 每 run（50 步 × 512 rollout，§3.3），没有报告训练 GPU 的型号、数量和时长。evaluator 的资源需求：
  - 数学：每条 rollout 1 GB 内存、2 CPU，最长 1100 s（App. B）。
  - TriMul：Modal 上的 H100；MLA：H200。
  - AHC：ALE-Bench 的 C++20 docker，时限 2 s、内存 1024 MB，每条 rollout 跑 150 个 case。
  - denoising：3 GB 内存，时限 400 s。

  一次完整 run 要做 25600 次评估，评测的吞吐和沙箱是隐性成本。
- **最小可复现实验**：
  1. 先离线做单元测试：实现二分求 $\beta$ 加 LOO advantage（App. A.1 的公式），检查 $\mathrm{KL}(q\|u)=\ln2$，以及对 reward 做 $wr+b$ 变换后 advantage 不变。
  2. 再选纯 CPU 的 AC1 或 Erdős，用 Qwen3-8B，保持 8×64、50 步。目标是复现论文里 Qwen3-8B 的 AC1 结果 1.50525（Table 2）；Erdős 的打分核心只有一行 `np.correlate`（prompt 里写的是 `C5 = max(np.correlate(h, 1-h, mode="full") * dx)`），外加 $h\in[0,1]$、$\int h=1$ 的约束检查。
  3. 在同一设置下跑三组对照：expected reward vs adaptive entropic、PUCT vs no reuse、Best-of-25600。以此检验 Table 8 的排序能否在数学任务上复现，论文只在 TriMul 上做了消融。
- **难点**：
  - 依赖付费的 Tinker API。换成本地 vLLM 加 LoRA 训练时，要自己处理 sampler/learner mismatch 的 IS 修正。
  - 执行任意生成代码需要安全的沙箱和超时机制。
  - kernel 任务要求目标 GPU 与官方 harness 一致。
  - 单次运行的方差未知，至少需要多个 seed 才能判断改进是否可靠。

## 自测问题

1. 在 Table 8 中，adaptive entropic 加 no reuse（5274.03）几乎和 Best-of-N（5352.36）一样。§3.1 的三个缺陷里，哪个是 reuse 解决的，哪个是 entropic objective 解决的？为什么缺了 reuse，训练目标再好也没用？
2. 从 $J_\beta=\log\mathbb E[e^{\beta R}]$ 推出 $\nabla J_\beta=\mathbb E[w\nabla\log\pi]$，并说明为什么 $w-1$ 是合法的 baseline 形式。分析 $\beta\to0$ 和 $\beta\to\infty$ 两个极限，以及常数 $\beta$ 在训练早期和后期分别会出什么问题。
3. 组大小为 64、$\gamma=\ln2$ 时，KL 约束对权重分布意味着什么（有效样本数是多少）？为什么这个估计量对 reward 的仿射变换不变？这一点对"一套超参跑遍数学、kernel、AtCoder、生物"为什么重要？
4. 本文的 PUCT 与 AlphaZero 的 PUCT 有哪四点不同？$Q$ 取 max 而不取 mean、访问计数回传整条 lineage，分别是在对付什么问题？
5. TTT-Discover、[TTRL](../ttrl/notes.md) 和原始 [TTT](../ttt-self-supervised/notes.md) 分别为每个测试实例构造了什么样的学习问题？最终产出的是参数、预测还是解？如果把精确的 evaluator 换成 learned world model，entropic objective 会放大哪种失败模式？
