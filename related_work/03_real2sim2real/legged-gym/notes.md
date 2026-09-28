# Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning

> arXiv [2109.11978](https://arxiv.org/abs/2109.11978) · CoRL 2021（PDF 元数据：Proceedings of the 5th Conference on Robot Learning）· 主题：03_real2sim2real · 层级：foundational
> 代码：https://github.com/leggedrobotics/legged_gym（论文给出的项目页为 https://leggedrobotics.github.io/legged_gym/；PPO 实现在配套的 rsl_rl 库）· 笔记依据：`source/merged.tex`（数字/公式）+ `paper.pdf` / `source/figures/*.png`（图）

## 一句话

作者把仿真、observation/reward 计算、策略推理和 PPO 更新全部放到单张 GPU 上（Isaac Gym）。配合数千个并行机器人、很短的每机器人 rollout，以及一个"游戏式"地形课程，他们在一张 RTX A6000 上用不到 20 分钟训练出一个感知式 rough-terrain 行走策略，并部署到了真实的 ANYmal C。论文的关键认识是：大规模并行时，on-policy 的 batch 必须在"机器人数 × 每机器人步数"之间重新分配，而且两个方向上都有下限。

## 问题设定

- **任务**（§3）：四足机器人（主要是 ANYmal C）穿越 flat、slope、random rough、discrete obstacles、stairs 五类程序化地形，同时跟踪 base heading 与线速度指令。指令在每个 episode 开始时随机采样，之后保持不变。
- **观测**（§3.2）：base 线速度与角速度、重力向量投影、关节位置与速度、上一步 action，以及 base 周围网格上采样的 **108 个地形高度**（每个值是地面到 base 的高度差）。策略不带记忆（§3.3 提到试过加记忆但没有收益）。论文**没有报告网络结构**，要看代码。
- **动作**：期望关节位置，policy 以 50 Hz 输出（A.1.1）。仿真里由 actuator network 把关节目标转换成力矩，真机上由 PD 控制器执行。
- **奖励**：9 项加权和（Table 2），不含任何与步态相关的项。
- **先前方法卡在哪**（§1、§2）：
  - MuJoCo、Bullet、RaiSim 都跑在 CPU 上，并行度低。CPU 仿真加 GPU 推理还会被 PCIe 数据搬运拖慢（论文引文称搬运可比 GPU 计算本身慢 50 倍）。
  - 先前 locomotion 工作的训练时间：Lee 2020 为 12 h，DeepGait 与牛津的工作分别为 82 h 和 88 h，Miki 2021 为 120 h（§1）。
  - OpenAI、DeepMind 的分布式方案在数千个 CPU 上平均梯度，每个 worker 的样本数并不减少。它们得到了百万级的 batch，改善了学习动态，但没有优化总训练时间。
  - DRL 需要反复调参、串行重跑，所以 wall-clock 时间本身就是研究瓶颈。

## 核心方法

**1. 端到端 GPU 管线（§2.1、A.1）。**
- Isaac Gym 的 PhysX 在同一个 sim 里跑数千个机器人，observation 和 reward 全部向量化。
- 因为所有机器人共用一张地形 mesh、无法在每次 reset 时重建，作者把**所有地形类型 × 难度等级平铺成一整张 mesh**，换难度就是把机器人"搬"到另一块地形上。
- 提升吞吐的技巧：
  - 仿真步长取 0.005 s，每个 policy step 对应 4 个仿真步。目标是把步长尽量取大，0.005 s 是能用的最大步长（原文字面写作 "can not use a time step smaller than 0.005 s"，但按"要最大化步长"的上下文，实际是上限）。这个限制来自 actuator network（它近似离散 PD）会失稳，而不是仿真器本身。
  - 碰撞体只保留 feet、shanks、knees 和 base。
  - 用低分辨率 heightfield 转成三角网格，再修正竖直面，而不是提高 heightfield 分辨率。
  - PhysX 即使忽略机器人之间的接触也仍会做检测，所以机器人散得越开越快。训练初期和末期的仿真时间差约 2 倍（A.1.2）。

**2. 大规模并行下的 batch 分配（§2.2.1）。** on-policy 每次更新用的样本数为（`source/merged.tex` §2.2.1）

$$B = n_{robots}\thinspace n_{steps}$$

$n_{robots}$ 提高了几个数量级，所以 $n_{steps}$ 必须变小，但它有下限。PPO 用 GAE（以下是标准形式，本文没有写出公式，见 [PPO 笔记](../../04_rl/ppo/notes.md)）：

$$\delta_t=r_t+\gamma V_\phi(s_{t+1})-V_\phi(s_t),\qquad \hat A_t=\sum_{l=0}^{n_{steps}-t-1}(\gamma\lambda)^l\thinspace\delta_{t+l}$$

在 rollout 末尾用 $V_\phi$ bootstrap，其中 $\gamma=0.99$， $\lambda=0.95$（Table 3）。
- **下限**：作者发现 $n_{steps}<25$（0.5 s 仿真时间）时算法难以收敛到最优解，解释是 GAE 需要多步 reward 才有效。
- **与 episode 长度的区别**： $n_{steps}$ 不等于 episode 长度。time-out 定为 20 s，环境不在每次迭代后 reset，所以一个 episode 横跨很多次策略更新（按 50 Hz 算，一个 episode 是 1000 步，约 42 次迭代，这是我算的）。
- **mini-batch**：用"数万"量级，比常规做法大得多。论文说这样能稳定学习，而且不增加训练时间。

最终使用的 PPO 配置（Table 3）：

| 项 | 值 |
|---|---|
| batch | 98304（4096×24） |
| mini-batch | 24576（4096×6），即每个 epoch 4 个 mini-batch |
| epochs | 5 |
| clip | 0.2 |
| entropy coef | 0.01 |

PPO 的目标函数是标准的 $L^{CLIP}=\mathbb E_t[\min(\rho_t\hat A_t,\ \mathrm{clip}(\rho_t,1\pm0.2)\hat A_t)]$（本文没有写出公式，只给了 clip range 0.2）。学习率按 KL 自适应（Alg. 1， $kl^{\ast}=0.01$）：

$$\alpha\leftarrow\begin{cases}\max(10^{-5},\ \alpha/1.5) & kl>2kl^{\ast}\cr \min(10^{-2},\ 1.5\alpha) & kl<0.5kl^{\ast}\cr \alpha & \text{otherwise}\end{cases}$$

**3. time-out bootstrapping（§2.2.2、A.2）。** critic 预测的是无限 horizon 的折扣回报。
- **failure 终止**（base 触地算 crash）可以由状态预测，因此把回报截断是对的。
- **time-out 终止**无法预测，因为观测里没有 episode 时间。如果把它也当成终止处理，critic target 就是错的。
- 论文只用文字描述了解法。我把它形式化如下：

$$y_t=\begin{cases}r_t & \text{failure 终止}\cr r_t+\gamma V_\phi(s_{t+1}) & \text{time-out 或未终止}\end{cases}\quad\Leftrightarrow\quad \tilde r_t=r_t+\gamma V_\phi(s_{t+1})\ \text{（time-out 时把 bootstrap 值并入 reward，再按终止处理）}$$

（这里用一步 TD target 示意；实际的 critic target 是 GAE 算出的 return，time-out 的处理方式相同。）

- 标准 Gym 接口不区分这两种终止，Stable-Baselines 也忽略了这一点（Spinning-up 在算法内部记录 episode 长度来绕开）。
- 在本文设定里每批只有 24 步，time-out 几乎必然发生在 batch 中间，所以不能再假设"只在 batch 最后一步超时"。作者因此修改了 Gym 接口，让它报告 time-out。

**4. 游戏式自动课程（§3.1、Fig. 3）。**
- **地形**：8 m 见方的地块，机器人从中心出发。slope 与 stairs 做成金字塔形，保证各方向都可通行。
- **状态**：每个机器人有自己的（地形类型, level）。stairs 和 obstacles 的台阶高度从 5 cm 升到 20 cm，坡度从 0° 升到 25°。
- **规则**：
  - **升级**：走出自己那块地的边界，下次 reset 时换到更难的地块。
  - **降级**：episode 结束时位移 $d<\tfrac12\lVert v^{\ast}\rVert\thinspace T$，即不到指令速度应走距离的一半（公式是我对文字的形式化）。
  - **循环**：通过最高级的机器人被随机送回某个 level，用来保持多样性、避免 catastrophic forgetting。
- **与 Lee 2020 的 particle filter 课程相比**：不需要重采样和重新生成地形。
- **与 Florensa 等人的 automatic goal generation 相比**：不需要用 generator 网络去学策略的性能分布。数千个机器人在各 level 上的分布本身就是当前策略的性能分布，而且计算开销接近 0。
- **进度**（Fig. 3，4000 个机器人）：
  - 第 500 次迭代时，已经能过 slope、能下楼梯，上楼梯和过 obstacle 还不行。
  - 第 1000 次迭代时，所有地形都已到最高级。
  - 总共训练 1500 次迭代。

**5. 奖励（Table 2， $\phi(x)=\exp(-\lVert x\rVert^2/0.25)$，z 轴与重力对齐）。**

| 项 | 定义 | 权重 |
|---|---|---|
| 线速度跟踪 | $\phi(v^{\ast}_ {b,xy}-v_{b,xy})$ | $1\thinspace dt$ |
| 角速度跟踪 | $\phi(\omega^{\ast}_ {b,z}-\omega_{b,z})$ | $0.5\thinspace dt$ |
| z 向线速度惩罚 | $-v_{b,z}^2$ | $4\thinspace dt$ |
| xy 角速度惩罚 | $-\lVert\omega_{b,xy}\rVert^2$ | $0.05\thinspace dt$ |
| 关节运动 | $-\lVert\ddot q_j\rVert^2-\lVert\dot q_j\rVert^2$ | $0.001\thinspace dt$ |
| 关节力矩 | $-\lVert\tau_j\rVert^2$ | $0.00002\thinspace dt$ |
| action rate | $-\lVert\dot q^{\ast}_ j\rVert^2$ | $0.25\thinspace dt$ |
| 碰撞 | $-n_{collision}$ | $0.001\thinspace dt$ |
| feet air time | $\sum_f(t_{air,f}-0.5)$ | $2\thinspace dt$ |

knee 或 shank 发生接触、脚与竖直面接触都算 collision。base 发生接触算 crash，直接 reset。原表把求和写成 $\sum_{f=0}^{4}$，应为 4 只脚。

**6. Sim-to-real 措施（§3.3、Table 4）。**
- **domain randomization**：地面摩擦系数从 $U[0.5,1.25]$ 采样；每 10 s 推一次机器人，把 base 在 x、y 方向加速到最多 ±1 m/s。
- **观测噪声**：按真机测量的噪声水平加均匀噪声，例如关节速度 ±1.5 rad/s，高度测量 ±0.1 m。
- **actuator network**：ANYmal 使用 series elastic actuator，难以直接建模。作者沿用 Hwangbo 2019 的做法用网络从关节指令算力矩，但改成只输入当前测量的 **LSTM**。作者还试过给 policy 加记忆，最终性能没有提升。

**训练与推理流程。**
- **训练**：每次迭代先让 4096 个环境各跑 24 步（GPU 上）；然后计算 GAE（time-out 时 bootstrap）；再做 PPO 5 epochs × 4 mini-batch，并按 KL 调学习率；reset 时更新课程 level。共 1500 次迭代。
- **部署**（§4.3）：策略固定不变，在线不做任何适应。高度观测来自机器人用 LiDAR 建的 elevation map，action 直接作为关节目标下发，不加滤波，也不做约束检查。

## 关键实验与证据

**并行规模研究（§4.1、Fig. 4）。**
- **设置**：为了把 total reward 当作单一指标，这组实验去掉了课程，并降低了最大台阶高度。每个配置训练 1500 次更新，重复 5 个 run。
- **上界 baseline**：20000 个机器人 × 50 步，batch 为 1M。reward 最高，但耗时较长。从图上读，reward 约 19，时间在 1–2 h 之间。
- **对比**：batch 取 49152、98304、196608 三档，机器人数从 128 扫到 16384（16384 接近单卡 rough 地形的上限）。
- **两侧失效**：
  - 机器人过多时 reward 急剧下降，因为每个机器人的 horizon 太短。例如 batch 49152 配 16384 个机器人时每个只有 3 步，reward 约 0（图读数）。batch 越大，这个下降点越往右移。
  - 机器人过少时 reward 缓慢下降。作者的解释是步间隔小，同一机器人的样本高度相关，更偏离 i.i.d. 假设。
- **训练时间**：在 4000 个机器人以内接近线性加速。batch 98304 时，128 个机器人约 4 h，4096 个约 20 min 以内（Fig. 4b 读数）。
- **结论**（Fig. 4c）：对这个任务，2048–4096 个机器人配约 100k 或约 200k 的 batch 是最佳折中。

**主策略与仿真评测（§4.2）。**
- 4096 × 24 的配置训练 1500 次更新，用时不到 20 min，硬件为 i9-11900k 加 RTX A6000（脚注 2）。平地策略不到 4 min（Abstract）。
- 通过性测试（Fig. 5）：前向指令 0.75 m/s，侧向指令在 $[-0.1,0.1]$ m/s 内随机；成功定义为穿过地形且 base 没有任何接触。
  - 楼梯：0.2 m 以内成功率接近 100%。0.2 m 是训练的最高难度，也接近运动学极限。0.25 m 时上楼约 15%、下楼约 28%（图读数）。
  - 离散障碍：成功率随高度平稳下降，0.3 m 时约 60%（图读数）。注意相邻块可正可负，实际最大落差是标称值的两倍。
  - 坡：原文说超过 25° 就爬不上去，但能以中等成功率滑下来。从图上读，上坡在约 27° 时仍有约 86%，31° 约 37%，35° 约 5%，之后接近 0；下坡在 45° 约 60%（图读数）。
- 策略总是收敛到 trot，但常有拖腿、base 过高或过低等 artifacts，要调 reward 权重才能上真机（§4.2）。

**跨机器人泛化（§4.2、Fig. 6，仅在仿真中、仅定性）。**
- ANYmal C 加装机械臂（重量约增加 20%）和 ANYmal B：reward 与超参都不改，性能"非常相似"。
- Unitree A1：去掉 actuator model，降低 PD gain 和力矩惩罚，改默认关节构型。
- Cassie：必须额外加一个"单脚站立"奖励才能走起来。

**真机（§4.3、Fig. 7）。** 只有定性展示（图和视频），能上下楼梯、动态越障。elevation map 不完美，导致从仿真到真机鲁棒性下降，问题主要出现在高速时，所以真机把线速度指令上限压到 0.6 m/s。

**消融与开销。**
- **time-out bootstrapping**（A.2、Fig. 10）：在 flat 和 rough 两个任务上都明显降低 critic loss，total reward 提高约 10%–20%。这是论文中唯一有曲线支撑的算法组件消融。
- **时间分解**（Fig. 8）：仿真最耗时，observation/reward 计算其次，policy 与 actuator net 推理基本是常数时间，learning 步的耗时与机器人数无关。
- **显存**（Fig. 9）：rough 地形 4096 个机器人开渲染要 9 GB、headless 要 6 GB；平地分别为 7 GB 和 5 GB。

## 局限与论文没说的

**作者承认的：**
- on-policy 算法对 $n_{robots}$ 有上限，因此用不满 GPU（§2.2.1）。
- 真机上 elevation map 不完美，所以降速部署。Miki 2021 用 teacher-student 方案应对地图不完美和状态估计漂移，作者计划把两者合并（§4.3）。
- 行为有 artifacts，需要调 reward（§4.2）。
- 本文目标不是做出最鲁棒的策略（§5）。

**我读出来的：**
1. **真机没有定量评测**：没有成功率，也没有和 Lee 2020、Miki 2021 在同一地形上的对比。"快几个数量级"是拿不同任务、不同管线、不同硬件的训练时间在比（§1 列的 12–120 h），不是受控对照。
2. **并行研究的口径有偏**：
   - Fig. 4 在去掉课程、降低难度的简化任务上做，结论能否迁移到带课程的完整任务没有验证。
   - 最终选的 4096 个机器人配 98304 batch，在 Fig. 4a 里的 reward（约 12.8，图读数）明显低于 2048 个机器人（约 15）。这个选择实际偏向训练时间。
   - $n_{steps}=24$ 已经略低于作者自己说的"<25 步会吃力"的门槛，说明这个门槛是软的，也依赖任务。
3. **其它组件都没做消融**：课程、大 mini-batch、自适应学习率、LSTM actuator net、各项 DR 都没有消融。Fig. 5 也没报告每个点的试验次数和方差。
4. **domain randomization 很窄**：只随机化摩擦和推力，外加观测噪声；没有报告质量、质心、电机参数的随机化。策略没有记忆，也就没有隐式 system identification（对比 [RMA](../rma/notes.md)）。能泛化靠的是 actuator network，而它需要真实电机数据，而且是 ANYdrive 专用（换 A1 就得去掉）。
5. **观测假设偏强**：观测里有 base 线速度（依赖状态估计，噪声只设了 ±0.01 m/s）和精确高度图。所以失效模式很可能出在感知这一侧：反光、软质或可变形地面、地图漂移。地形都是刚体。
6. **样本量巨大**：1500 × 98304 ≈ $1.5\times10^8$ 个样本，约相当于 34 天的机器人时间（按 50 Hz 算，是我算的）。"分钟级"换来的是 wall-clock 时间，不是样本效率，这正是 world model 路线要攻的点。
7. **reward 工程成本被低估**：9 项权重需要手调，换成双足还得加项。
8. **只研究了 PPO**：在大规模并行场景里没有和 off-policy 方法（如 SAC）比较。

## 在脉络中的位置

- **前驱**：
  - PPO + GAE（[PPO](../../04_rl/ppo/notes.md)）。
  - Hwangbo 2019 的 actuator network。
  - Lee 2020 的 blind rough-terrain 策略和 particle filter 课程。
  - Miki 2021 的感知式 teacher-student。
  - Pardo 等人关于 time limit 的工作。
  - GPU 并行仿真与训练：Isaac Gym、Stooke & Abbeel 的 accelerated deep RL、GPU 上的 humanoid、Brax。
  - 动力学层面的 [domain randomization](../domain-randomization/notes.md)（本文随机化了摩擦和推力，但没有引用 DR 文献，这条是我归的类）。
- **后继 / 影响**：
  - 同年的 [RMA](../rma/notes.md) 提供了"特权信息加 adaptation module"这一半，与本文的"大规模并行 PPO 加地形课程"合起来，成为 legged sim2real 的标准配方。
  - [ASAP](../asap/notes.md) 在引言中把本文列为 DR 路线的代表（`asap/source/sections/intro.tex`）。
  - [Reality-gap 综述](../survey-reality-gap/notes.md) 把它当作并行仿真推动 locomotion 突破的例子。
  - [SimDist](../../05_intersections/simdist/notes.md) 的 expert 用的是同一套配方："IsaacLab、4096 个环境、每次迭代 24 步、地形课程"（`simdist/source/sections/appendix.tex`）。不过 SimDist 正文没有引用本文，这里是配置上的对应。
  - [RWM](../../05_intersections/robotic-world-model/notes.md) 在 learned world model 里开 4096 个 imagination environments 跑 PPO（RWM 同样没有引用本文，是配置上的对应）。
  - [DayDreamer](../../05_intersections/daydreamer/notes.md) 把本文归入"大量仿真经验加随机化"一类，作为自己"不依赖仿真器"的对照。
- **与本项目其他主题的联系**：
  - **World model**：RWM 这类工作本质上是用神经网络替换本文里的 PhysX。本文"GAE 需要足够长的 $n_{steps}$"与 world model rollout 越长、误差累积越重之间存在张力，imagination horizon 的选择可以对照本文 Fig. 4 来理解（参见 [DreamerV3](../../01_world_model/dreamerv3/notes.md)）。
  - **TTT / TTA**：本文部署时完全不做适应，全靠训练期的随机化。它是 [RMA](../rma/notes.md) 的在线 extrinsics 估计和 [PAD](../../05_intersections/pad/notes.md) 的部署期自监督更新的"零适应"基线。
  - **Real2sim**：§1 提到分钟级训练使"用特定环境的扫描做定制训练"成为可能，这正是 [RialTo](../rialto/notes.md)、[ACDC](../acdc-digital-cousins/notes.md) 这类 digital twin / digital cousin 路线的前提：重建一次就能在当天重训。

## 前置知识

1. PPO 的 clipped objective、actor-critic、GAE 中 $\gamma$ 与 $\lambda$ 的作用（[PPO](../../04_rl/ppo/notes.md)）。
2. MDP 中 termination 与 truncation（time limit）的区别，以及 value target 的 bootstrapping。
3. on-policy 训练中 batch、mini-batch、epoch 的含义，以及基于 KL 的自适应学习率。
4. 足式机器人基础：关节 PD 位置控制、series elastic actuator 与 actuator network、elevation map、状态估计。
5. [Domain randomization](../domain-randomization/notes.md) 与 curriculum learning。
6. GPU 向量化仿真（Isaac Gym tensor API）的基本概念。

## 复现可行性

- **代码**：
  - legged_gym 已开源。README 写明依赖 Isaac Gym Preview 3、rsl_rl v1.0.2、Python 3.6–3.8、PyTorch 1.10 + CUDA 11.3。
  - README 声明环境已迁移到 Isaac Lab，本仓库今后只做有限维护。
  - 仓库里有 `resources/actuator_nets/anydrive_v3_lstm.pt`，所以仿真端不需要真实电机数据。
  - 训练入口：`python legged_gym/scripts/train.py --task=anymal_c_flat`。
- **算力**：论文用一块 RTX A6000 配 i9-11900k。rough 地形 4096 个机器人 headless 只要 6 GB 显存（Fig. 9），因此消费级 NVIDIA 卡也够（这是我的推断）。Isaac Gym 需要 Linux 加 NVIDIA GPU，本机 macOS 跑不了，要用 Linux GPU 服务器。
- **数据**：不需要数据集，只需要 URDF 和程序化地形。
- **最小可复现实验建议**：
  1. `anymal_c_flat` headless 训练，核对"平地不到 4 min"和 reward 曲线的形状。
  2. 复现 Fig. 10：按图中的横轴，flat 训练约 270 次迭代、rough 训练约 1500 次迭代（图读数），一组开 time-out bootstrap、一组关掉，对比 total reward（预期差约 10%–20%）和 critic loss。
  3. 缩小版 Fig. 4：关掉课程、降低台阶高度，固定 $B=98304$，令 $n_{robots}\in\lbrace512,2048,4096,16384\rbrace$（对应 $n_{steps}=192,48,24,6$），各跑 3 个 seed，验证 16384 处的性能断崖。
- **难点**：
  - Isaac Gym Preview 的环境（旧 Python、旧 CUDA）安装麻烦，可以考虑直接用 Isaac Lab 的对应任务，但数字不一定能对上。
  - 论文没写网络结构，要以代码配置为准。
  - 真机部分（ANYmal C、LiDAR elevation mapping）对多数人不可复现。

## 自测问题

1. 固定 $B=n_{robots}n_{steps}$ 时， $n_{robots}$ 过大和过小都会掉性能。两侧分别是什么机制？为什么 batch 越大，右侧的断崖越往右移？
2. failure 终止和 time-out 终止下，critic target 分别怎么写？为什么在"每次迭代 24 步、episode 20 s"的设定下，忽略两者的区别比在标准 Gym 设定下伤害更大？
3. 游戏式课程的升级、降级、循环规则各是什么？为什么说"机器人在各 level 上的分布"本身就是策略性能分布的估计？相比 particle filter 课程，它省掉了哪些计算？
4. 这个策略部署时参数完全固定。它靠哪几个训练期手段对抗 sim-to-real gap？与 RMA 相比缺了哪种能力，预期会在什么真实扰动下失效？
5. 如果把 PhysX 换成 learned world model 来做同样的大规模并行 PPO，本文关于 $n_{steps}$ 下限和样本相关性的结论会怎样变化或冲突？
