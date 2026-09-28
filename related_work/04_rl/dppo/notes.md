# Diffusion Policy Policy Optimization

> arXiv [2409.00588](https://arxiv.org/abs/2409.00588) · ICLR 2025（Poster，已在 OpenReview 核实；本笔记依据 arXiv v3，2024-12-09，源码里留有 ICLR 投稿版的开关，正文没有写录用信息） · 主题：04_rl · 层级：milestone
> 代码：https://github.com/irom-princeton/dppo（论文只给了 diffusion-ppo.github.io。repo 已访问确认：MIT license，提供论文用的预训练 checkpoint，预训练数据自动下载） · 笔记依据：`source/body/*.tex`（数字、公式）+ `paper.pdf`（图）。公式、图、表编号沿用 PDF。

## 一句话

DPPO 把 Diffusion Policy 的每一步去噪看作内层 MDP 的一步动作，再把它嵌进环境 MDP，得到一个两层的 "Diffusion Policy MDP"。这样每一步的似然都是可解析的高斯分布，BC 预训练好的 diffusion policy 就能直接用 PPO 微调。论文的关键发现是：在 fine-tuning 设定下，这种 on-policy policy gradient 比基于 Q-learning 的 diffusion RL 更稳，而且会在示范数据流形附近做结构化探索。

## 问题设定

- **输入**：一个用 BC 预训练的 Diffusion Policy $\pi_\theta$，条件是状态 $s$（或像素加本体感知，论文脚注说实现上支持 POMDP），输出一个 **action chunk**：预测 $T_p$ 步，执行 $T_a\le T_p$ 步。另外有一个可并行的仿真器和奖励。Gym 用 dense reward；Franka-Kitchen、Robomimic、Furniture-Bench 用 sparse reward（完成任务或完成某个阶段给 1）。**微调阶段不再用示范数据。**
- **输出**：微调后的 diffusion policy，部署时照常跑完整条去噪链。
- **卡点**：
  1. DP 没有显式的边缘似然 $p_\theta(a^0\mid s)$，而标准 PG/PPO 需要 $\log\pi_\theta(a\mid s)$。
  2. 已有的 diffusion RL 基本都绕开似然，改用 state-action Q 函数更新 actor。DQL 让 critic 的梯度反传穿过整条去噪链；IDQL 用 critic 对采样动作做重选；DIPO 用 action gradient 改 buffer 里的动作；QSM 让 score 对齐 $\nabla_a Q$。在稀疏奖励、连续高维 action chunk 的设定下 Q 估不准，微调一开始性能就掉，之后可能掉到 0 再也回不来（Fig. 5，`source/body/extended_related.tex`）。
  3. QSM 的作者推测，穿过去噪链做 PG 会让有效 horizon 变长、方差变大，所以低效。DPPO 在 fine-tuning 设定下的结果与这个推测相反。

## 核心方法

**(1) 背景：DDPM 与 DP。** 反向过程见 Eq. (3.3)（`source/body/formulation.tex`）：
$$x^{k-1}\sim p_\theta(x^{k-1}\mid x^k)=\mathcal N\big(x^{k-1};\mu_k(x^k,\epsilon_\theta(x^k,k)),\sigma_k^2 I\big).$$
这里 $\mu_k$ 是与 $\theta$ 无关的固定函数，$\sigma_k$ 由 schedule 固定。预训练用标准的 $\epsilon$-prediction BC loss，Eq. (A1)：$\mathcal L_{BC}(\theta)=\mathbb E^{(s_t,a_t^0)\sim\mathcal D_{\text{off}}}\|\epsilon_t-\epsilon_\theta(a^0_t,s_t,k)\|^2$（原文记号如此；按文字说明，网络实际输入的是对 $a_t^0$ 加噪后的动作），采用 cosine schedule。

**(2) 两层 MDP**（`source/body/approach.tex` §4.1，PDF Fig. 3）。定义索引 $\bar t(t,k)=tK+(K-k-1)$，它随环境步 $t$ 递增，随去噪步 $k$ 递减：
$$\bar s_{\bar t(t,k)}=(s_t,a_t^{k+1}),\quad \bar a_{\bar t(t,k)}=a_t^k,\quad \bar R_{\bar t(t,k)}=\begin{cases}0,&k>0\\R(s_t,a_t^0),&k=0\end{cases}$$
- 转移：当 $k>0$ 时，转移是确定性的，把刚去噪得到的 $a_t^k$ 放进下一个状态 $(s_t,a_t^k)$。当 $k=0$ 时，环境执行 $a_t^0$，转到 $s_{t+1}\sim P$，同时重新采样 $a_{t+1}^K\sim\mathcal N(0,I)$。**纯噪声 $a^K$ 算作环境的一部分**，所以它不进入似然。初始分布为 $\bar P^0=P_0\otimes\mathcal N(0,I)$。
- 策略就是单步去噪，Eq. (4.1)：
$$\bar\pi_\theta(\bar a_{\bar t}\mid\bar s_{\bar t})=\mathcal N\big(a_t^k;\ \mu(a_t^{k+1},\epsilon_\theta(a_t^{k+1},k{+}1,s_t)),\ \sigma_{k+1}^2 I\big).$$
  展开后 $\log\bar\pi_\theta=-\|a_t^k-\mu_\theta\|^2/(2\sigma_{k+1}^2)+\text{const}$，可以直接求梯度，不需要边缘似然。
- 在 $\mathcal M_{DP}$ 上套标准 PG，Eq. (4.2)：$\nabla_\theta\bar{\mathcal J}=\mathbb E\big[\sum_{\bar t}\nabla_\theta\log\bar\pi_\theta(\bar a_{\bar t}\mid\bar s_{\bar t})\,\bar r(\bar s_{\bar t},\bar a_{\bar t})\big]$。
- 与 QSM 推导的区别（`source/body/extended_related.tex`）：QSM 先在去噪步上求和，再对环境步取期望；DPPO 把去噪步也当成 MDP 的时间步，对两者一起取期望。

**(3) PPO 实例化**（§4.2 + App. B）。
- 用 clipped surrogate，Eq. (A3)：$\mathcal L_\theta=\mathbb E\min\big(\hat A\,\rho,\ \hat A\,\mathrm{clip}(\rho,1-\epsilon,1+\epsilon)\big)$，其中 $\rho=\bar\pi_\theta/\bar\pi_{\theta_{old}}$，按单个去噪步计算。
- **两层折扣的 advantage**（§4.2，未编号式）：
$$\hat A(\bar s_{\bar t(t,k)},\bar a_{\bar t(t,k)})=\gamma_{\text{denoise}}^{\,k}\big(\bar r(\bar s_{\bar t},\bar a_{\bar t})-\tilde V(s_t)\big),\qquad \bar r=\textstyle\sum_{t'\ge t}\gamma_{\text{env}}^{\,t'-t}R(s_{t'},a^0_{t'}).$$
  原文里指数写的是 $\gamma_{\text{env}}^t$，按上下文应为 $t'-t$。因为 $k>0$ 时奖励为 0，只需要在 $k=0$（也就是环境步）上用 GAE 算 advantage，Eq. (A2)，$\lambda=0.95$。算出来的 advantage 乘上 $\gamma_{\text{denoise}}^k$ 就能用于每一个去噪步。$\gamma_{\text{denoise}}<1$ 会调低噪声较大的早期去噪步在梯度里的权重。
- **critic 只看环境状态**：$\hat V(\bar s)=\tilde V(s_t)$，不看 $a_t^{1}$。训练目标是 MC 回报的 MSE，Eq. (A4)。
- 训练流程见 Alg. 1：$N$ 个并行环境 rollout，把每个去噪步的 $(k,\bar s,\bar a,\bar R)$ 都存进 buffer → 在环境步上算 GAE → 做 $N_\theta$ 轮 minibatch 更新，actor 和 critic 同步训练，$N_\theta=N_\phi$。因为 batch 里同时包含环境步和去噪步，batch size 大约取成 Gaussian PPO 的 batch 乘以 $K'$（App. B）。

**(4) Best practices**（§4.3，App. B）。
- **只微调最后 $K'$ 步去噪**：把参数复制一份得到 $\theta_{FT}$，只用于并更新最后 $K'$ 步，前面的步用冻结的 $\theta$。GPU 显存随 $K'$ 线性增长（App. C.2）。
- **微调 DDIM**，Eq. (4.3)：方差项变成 $\eta\sigma_k^2$。训练时取 $\eta=1$（等价于 DDPM，提供探索噪声，也避免对 Dirac 分布求似然），评估时取 $\eta=0$。像素任务和 Furniture 任务用 $K=100$ 预训练、5 步 DDIM 微调。
- **两个 noise clip**：采样时把 $\sigma_k$ 下限截到 $\sigma^{\text{exp}}_{\min}$（论文建议 0.01–0.1，实际用 0.1/0.08/0.04），用来保证探索；计算似然时下限截到 $\sigma^{\text{prob}}_{\min}=0.1$，避免 $1/\sigma^2$ 把 log-prob 放得太大。评估时 $\sigma^{\text{exp}}_{\min}=0.001$。
- **PPO clip 的 $\epsilon$**：在 $\{0.1,0.01,0.001\}$ 中取满足 clipping fraction 为 10%–20% 的最大值。另外设 $\epsilon_{k=K-1}=0.1\,\epsilon_{k=0}$，中间各步按指数插值。原文说"早期去噪步用更大的 $\epsilon$"，但式子给出的是更小的值，**文字和公式互相矛盾**。
- **网络结构**：MLP 微调更稳。UNet 只对 $a^k$ 做卷积，所以可以预训练时 $T_p=16$、微调时 $T_a=8$。

## 关键实验与证据

**任务**（§5 + Table A6）：Gym {Hopper, Walker2D, HalfCheetah}，用 D4RL medium 预训练，$T_a=4$，dense reward；Franka-Kitchen 三个数据集；Robomimic {Lift, Can, Square, Transport}，state 用 300 条 MH 示范预训练，pixel 用 100 条；Furniture-Bench {One-leg, Lamp, Round-table} × {Low, Med}，用 50 条仿真示范、$T_a=8$；D3IL Avoid 用于机制分析。

- **对比其他 diffusion RL 方法**（Fig. 5；超参数见 Table A7：$K=20$，$K'=10$，$\gamma_{\text{denoise}}=0.99$，$\epsilon=0.01$）：
  - Gym 上 IDQL 和 DIPO 与 DPPO 接近，其余方法更差、训练也更不稳。
  - Robomimic 上 DPPO 最强。IDQL 和作者自己做的 DRWR 在 Lift/Can/Square 上是强基线，但在 Transport 上失败。
  - Q-learning 类基线在微调初期普遍掉点。
- **对比 demo-augmented RL**（Fig. 6）：
  - RLPD、Cal-QL、IBRL 在 HalfCheetah 上的样本效率明显高于 DPPO。
  - Kitchen 上 Cal-QL 有竞争力，DPPO 整体最好，在 Complete 上尤其明显；Partial/Mixed 的示范本身不完整，限制了 DPPO 的上限。
  - Robomimic Can/Square（PH 和 MH 都测了）上，RLPD 和 Cal-QL 学不起来，IBRL 停在较低水平。注意：基线在 Robomimic 上被限制在 1e6 样本。
- **对比 Gaussian/GMM + PPO**（Fig. 7、A9）：
  - state 输入：Transport 超过 90%。
  - pixel 输入（ViT-MLP）：Gaussian 在 Transport 上始终是预训练时的 0%。
  - 作者称 DPPO 是**第一个**在 Transport 上达到 >50% 成功率的 RL 算法，state 和 pixel 都算。
- **Furniture-Bench 与 sim-to-real**：
  - 仿真（Fig. 8 上）：DPPO 在 6 个设定里都有提升。Gaussian-MLP 在 3 个 Med 任务上塌到 0（Lamp 有 1 个 seed 例外），在 Round-table Low 上也塌到 0。DPPO 在 Lamp Low 上略逊于 Gaussian。
  - One-leg 零样本上真机（Fig. 8 下，每种方法 20 次，单盲评测，10 Hz）：

    | | 仿真 | 真机 |
    |---|---|---|
    | 预训练 Diffusion | 0.47 | 0.6 |
    | 预训练 Gaussian | 0.45 | 0.3 |
    | DPPO 微调后 | 0.87 | **0.8（16/20）** |
    | Gaussian 微调后 | 0.88 | **0.0** |
    | Gaussian + BC 正则（$\alpha_{BC}=0.1$） | 0.53 | 0.5 |

    仿真训练中做了 domain randomization：观测噪声按真机静止时测到的范围设定；动作上加 std=0.03 的噪声，模拟不理想的底层控制器（App. E.8）。
- **机制分析**（§6，D3IL Avoid，Fig. 10–13）：
  - DPPO 在微调第一轮的探索轨迹铺在示范流形附近，覆盖面宽。Gaussian 的探索噪声缺少结构，GMM 覆盖面窄。
  - 在训练中人为加入动作噪声后，Gaussian 和 GMM 都崩了；只要至少用 4 个去噪步，DPPO 仍然稳定。
  - $T_a$ 到 16 时 DPPO 仍能训练。
  - 微调过程中多步逐渐细化的结构保留了下来，没有塌到某一步。
  - 微调后的策略对动作扰动更鲁棒，并且能从更大范围的初始状态收敛到（近）最优路径（Fig. 13）。
- **消融**（App. C.2，Fig. A2–A5）：
  1. advantage 估计：在较难的任务上，$\tilde V(s_t)$ 优于 $V(s_t,a^1_t)$ 和 $Q(s_t,a^0_t)$；只有在简单的 Hopper 上，依赖动作的估计更好。
  2. $\gamma_{\text{denoise}}\in\{0.5,0.8,0.9,1\}$：Hopper/Cheetah 上 0.8 最好；操作任务对这个值不敏感。
  3. $\sigma^{\text{exp}}_{\min}\in\{0.001,0.01,0.1,0.2\}$：Can 上不敏感；Square 上要取到 0.1 才不崩（Fig. A4 中 0.1 和 0.2 都能训练，0.001/0.01 崩溃）。
  4. $K'$：取 3 太少，取 10 的整体效率最好。
- **其他补充实验**：
  - 只在最后一步加噪（去掉结构化探索）：在 Can/Square 上收敛更慢（Fig. A11）。
  - 用 probability-flow ODE 计算精确似然再做单层 PG：Hopper 上有提升但不如 DPPO，Can 上降到 0（Fig. A10）。
  - One-leg 只用 10 条示范，DPPO 能到 60%（Fig. A6）。
  - 从零训练（$T_a=1$）：Gaussian 学得更快；DPPO 最终性能也合理，但墙钟时间约为 6 倍（Fig. A7）。
- **墙钟时间**（App. D）：state 输入下，DPPO-MLP 比 Gaussian/GMM 慢 21%–24%，UNet 版再慢 49%；pixel 输入下慢 14%；Furniture 上慢 20%。

## 局限与论文没说的

**作者承认的局限**：
- 样本效率低于 off-policy 方法，结论一节把它列为主要局限。
- 效果依赖预训练的 mode coverage（Kitchen Partial/Mixed）。
- on-manifold 探索在需要激进探索时可能拖后腿（Lamp Low）。
- 从零训练没有优势，而且更慢。

**我读出来的问题**：
- **样本量极大，只能在仿真里做**。按图的横轴，Robomimic 从 3e6（Lift/Can，Fig. 5）到 4e7（Transport-state，Fig. 7）步，Furniture 到 7e8–1e9 步，Furniture 用了 1000 个 IsaacGym 并行环境。真实世界只做了零样本部署，没有做真机上的在线 RL。
- **sim-to-real 的证据很薄**：只有 1 个任务、每种方法 20 次，而且是 state 输入。真机上靠 4 个相机加 AprilTag，再做滤波、缓存等大量工程才撑起来。**两种方法的 DR 设置可能不对称**：App. E.8 写的是对 DPPO 采样的动作加 0.03 的噪声，并说给 Gaussian 加这种噪声会导致 0% 成功率；由此推断 Gaussian 基线没有加这项动作噪声（原文未明说 Gaussian 最终用的 DR 配置）。Gaussian 真机 0% 不能完全归因于参数化本身。
- **比较口径不统一**：
  - DPPO 的 actor lr 是 1e-4，其他 diffusion 基线是 1e-5（Table A7；表注说基线的 lr 在 1e-4 到 1e-5 之间调过）。
  - 两处实验里 DPPO 的 batch 差 10 倍：Lift/Can 上 Table A7 是 7.5k、Table A9 是 75k（Square/Transport 是 10k 对 100k）。App. B 的解释是 batch 一般取 Gaussian batch × 微调步数，但在 Robomimic 上更小的 batch（接近 Gaussian 的）也能用，而且样本效率明显更高。
  - Robomimic 上 demo-augmented 基线（RLPD/Cal-QL/IBRL）被限制在 1e6 样本；Fig. 6 里 DPPO 也只画到 1e6，但 DPPO 在其他实验里的横轴远超这个数。
  - 结果全是 3–5 个 seed 的曲线，没有给出最终数字的表格。
- **信用分配很粗**：同一环境步的 $K'$ 个去噪步共用一个 advantage，只按 $\gamma_{\text{denoise}}^k$ 缩放；形式上 $\mathcal M_{DP}$ 的长度是环境 MDP 的 $K$ 倍（$\bar t=tK+(K-k-1)$）。[RL 综述](../survey-rl-overview/notes.md) 也指出这会得到很长 horizon 的问题，并介绍了后续工作 NCDPO（`../survey-rl-overview/source/policy-rl.tex`）。
- **规模和观测设置**：模型最大约 10.6M 参数（Furniture 上的 Gaussian-MLP；DPPO-UNet 为 6.86M，Table A9–A11），没有在 VLA 或多任务大模型上验证。Kitchen、Robomimic、Furniture 的预训练都明确不用历史观测（App. E.5/E.6/E.8）。
- **缺少约束**：微调时没有 KL 或 BC 正则项（只有 PPO clip，外加 KL 达到 1 时提前结束本轮更新的保护，作者说实际从未触发，App. B），reward hacking 和"偏离示范后行为是否仍然自然"都没有系统讨论（作者只在 App. A.1 称 on-manifold 探索让微调后的行为保持自然）。

## 在脉络中的位置

- **前驱**：
  - [PPO](../ppo/notes.md) 加 GAE。
  - Diffusion Policy（Chi et al. 2023，不在本项目）、DDPM/DDIM。
  - DDPO（Black et al. 2023）把 text-to-image 的去噪过程建成 MDP。
  - QSM 的 PG 推导。
  - 被比较的 Q-learning 类方法：DQL，以及以 [IQL](../iql/notes.md) 式 critic 为基础的 IDQL；RLPD 基于 [SAC](../sac/notes.md)。
  - Gaussian 基线的 BC 正则取自 [RialTo](../../03_real2sim2real/rialto/notes.md)（`torne2024reconciling`）。
- **后继 / 影响**：
  - [π_RL](../pi-rl/notes.md) 明确"按 DPPO 构造两层 MDP"，把去噪过程耦合进环境交互，并把 ODE 改成 SDE 来获得似然（`../pi-rl/source/main.tex`）。
  - [π\*0.6](../pi-star-06/notes.md) 用 DPPO/FPO 的变体当 PPO 基线（`../pi-star-06/source/main.tex`）。
  - [RL 综述](../survey-rl-overview/notes.md) 把 DPPO 作为"diffusion policy 上的 PPO"的代表方法。
- **与本项目其他主题的联系**：
  - *World model*：DPPO 的瓶颈是需要海量 on-policy 样本，这正是"在 world model 里训 RL"的动机。作者在结论里点名 [Genie](../../01_world_model/genie/notes.md) 和 [UniSim](../../05_intersections/unisim/notes.md) 这类 generative simulation。相关工作还有 [Robotic World Model](../../05_intersections/robotic-world-model/notes.md)、[WMPO](../../05_intersections/wmpo/notes.md)、[World-Gymnast](../../05_intersections/world-gymnast/notes.md)、[Dreamer 4](../../01_world_model/dreamer4/notes.md)。[DreamZero](../../01_world_model/dreamzero/notes.md) 用 flow matching 联合生成视频和动作（ODE 采样），原则上可以像 π_RL 那样先转成 SDE 再接 DPPO 式的两层 MDP（推测，未见实验）。
  - *Real2sim2real*：Furniture 部分是标准的 [domain randomization](../../03_real2sim2real/domain-randomization/notes.md) 加零样本迁移。[X-Sim](../../03_real2sim2real/x-sim/notes.md) 的 student 是 diffusion policy，[ACDC](../../03_real2sim2real/acdc-digital-cousins/notes.md) 的 cousins 只做了 BC，两者都可以接 DPPO 做仿真内 RL 微调。
  - *TTT*：DPPO 是部署前的训练期微调，部署时不做适应。它和 [PAD](../../05_intersections/pad/notes.md)（部署时自监督适应）、[TTRL](../../02_test_time_training/ttrl/notes.md)（测试时 RL）的问题设定正好互补。
  - *RL*："BC 预训练 + PG 微调"与 LLM 侧的 [DeepSeek-R1](../deepseek-r1/notes.md)（GRPO）、VLA 侧的 [SimpleVLA-RL](../simplevla-rl/notes.md) 同构。与真机 off-policy 的 [HIL-SERL](../hil-serl/notes.md) 形成对照：前者靠仿真里的 on-policy 样本，后者追求真机上的样本效率。

## 前置知识

1. Policy gradient 与 likelihood-ratio 估计，[PPO](../ppo/notes.md) 的 clip 目标，GAE。
2. DDPM：前向加噪、$\epsilon$-prediction、反向高斯、noise schedule；DDIM 及其 $\eta$。
3. Diffusion Policy 与 action chunking（$T_p$ 与 $T_a$）。
4. BC 预训练 + RL 微调范式；Q-learning 类 diffusion RL（DQL/IDQL/QSM）作为对照。
5. Sim-to-real 与 domain randomization（理解 Furniture 实验需要）。

## 复现可行性

- **代码**：官方开源 https://github.com/irom-princeton/dppo，MIT license，Hydra 配置。示例命令：`python script/run.py --config-name=ft_ppo_diffusion_mlp --config-dir=cfg/robomimic/finetune/can`。提供预训练 checkpoint。以上来自 repo 页面。
- **依赖**：Linux + NVIDIA GPU；MuJoCo（Gym-v2/D4RL 依赖 mujoco-py，安装麻烦）、robomimic、D3IL；Furniture-Bench 还需要 IsaacGym。
- **算力**（论文只报告了每轮迭代的耗时，没有报告总 GPU 小时）：

  | 任务 | 硬件与并行 | 每轮耗时（DPPO） | 来源 |
  |---|---|---|---|
  | Gym | 40 环境 / 40 CPU 线程 + RTX 2080 | 16.6–18.3 s（每轮 2 万步） | Table A1 |
  | Robomimic state | 50 环境 + L40 | 35.2 / 42.0 / 65.6 / 350.3 s | Table A2 |
  | Robomimic pixel | 50 环境 + L40 | 194.9–871.3 s | Table A4 |
  | Furniture | 1000 个 IsaacGym 环境 + 单张 L40 | 148.4 / 258.2 / 188.6 s | Table A5 |

  **我按图的横轴粗估的总时长**：
  - Can-state（Fig. 5 横轴）：约 3e6 步 ÷ 每轮 6e4 步 ≈ 50 轮，约 35 分钟。
  - Transport-state（Fig. 5 横轴；Fig. 7 的横轴到 4e7，时长约翻倍）：约 2e7 ÷ 1.6e5 ≈ 125 轮，约 12 小时。
  - One-leg（Fig. 8 横轴）：约 7e8 ÷ 7e5 ≈ 1000 轮，约 41 小时。
- **最小复现实验**：Robomimic **Can（state，MH）**。
  1. 加载官方预训练 DP（$K=20$），用 DPPO 微调最后 $K'=10$ 步，超参数取 Table A7/A9。对照组是 Gaussian-MLP + PPO。跑 3 个 seed，目标是复现 Fig. 5/A9 中接近 100% 的趋势。
  2. 在同一套设置上做两个消融：critic 用 $\tilde V(s)$ 还是 $V(s,a^1)$（Fig. A2）；在 Square 上比较 $\sigma^{\text{exp}}_{\min}=0.01$ 和 0.1，预期 0.01 会崩（Fig. A4）。
  3. 可选：D3IL Avoid M2，可视化结构化探索（Fig. 10），很便宜。
- **预计难点**：
  - mujoco-py 和 D4RL 环境的安装。
  - $\epsilon$ 要按 clipping fraction 调。
  - batch 大，显存随 $K'$ 线性增长。
  - Furniture 需要 IsaacGym 和大量样本。
  - 真机部分依赖 4 相机 AprilTag 状态估计这套工程，基本不可复现。

## 自测问题

1. 写出 $\bar t(t,k)$、$\bar s$、$\bar a$、$\bar R$ 以及 $k>0$ 和 $k=0$ 两种情况的转移。为什么要把 $a_t^K\sim\mathcal N(0,I)$ 算作环境的一部分？如果把它算进策略，PPO 的 ratio 会有什么问题？
2. 为什么 DPPO 只需要在 $k=0$ 上做 GAE，再乘以 $\gamma_{\text{denoise}}^k$？critic 为什么不看 $a_t^{1}$？结合 Fig. A2 说明它在什么任务上反而吃亏。
3. $\sigma^{\text{exp}}_{\min}$ 和 $\sigma^{\text{prob}}_{\min}$ 分别作用在采样还是似然计算上？如果沿用标准 cosine schedule，让 $\sigma_k$ 在 $k=0$ 降到约 1e-4，$\nabla\log\bar\pi$ 和 PPO ratio 会怎样？
4. DPPO 与 QSM 的 PG 推导、与"用 probability-flow ODE 算精确似然的单层 PG"分别有什么不同？$K'$ 和 $\gamma_{\text{denoise}}$ 为什么可以看成同一个旋钮的硬版本和软版本？
5. "结构化、on-manifold 探索"的机制是什么？为什么从零训练时它不占优，在 Lamp Low 上甚至略差？One-leg 上 Gaussian 仿真 88%、真机 0% 的结果，有哪些因素不能归到参数化上？
