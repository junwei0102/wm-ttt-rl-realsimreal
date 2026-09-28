# Test-Time Training Done Right

> arXiv [2505.23884](https://arxiv.org/abs/2505.23884) · ICLR 2026（Poster，已在 OpenReview 核实；arXiv 预印本 2025，源码用 NeurIPS 2025 preprint 模板，arXiv 页面只有 v1、无录用信息） · 主题：02_test_time_training · 层级：milestone
> 代码：https://github.com/a1600012888/LaCT（MIT；论文正文只给了项目页 tianyuanzhang.com/projects/ttt-done-right，仓库链接来自 index.json，已核实存在） · 笔记依据：`source/*.tex`（数字/公式）+ `paper.pdf`（图）

## 一句话

LaCT（Large Chunk Test-Time Training）把 TTT 层更新 fast weights 的单位，从 16–64 token 的小 mini-batch 改成 2K 到 1M token 的大 chunk。这样纯 PyTorch 实现也能把 GPU 吃满，fast weights 于是可以换成大的非线性 SwiGLU-MLP（最多占模型参数的 40%），并用 Muon 做 test-time optimizer；chunk 内部的局部结构交给 window attention 处理。

## 问题设定

- **语境**：这里的 TTT 指"TTT 作为序列建模层"（[TTT layers](../ttt-layers/notes.md) 一线），不是应对分布偏移的 TTT（[Sun et al. 2019](../ttt-self-supervised/notes.md)、[Tent](../tent/notes.md)）。每个 token 投影出 $q_i,k_i,v_i$。fast weight 网络 $f_W$ 有两种操作：**update** $W\leftarrow W-\eta\nabla_W\mathcal{L}(f_W(k),v)$ 和 **apply** $o=f_W(q)$（`source/sections/02_preliminary.tex` Eq.1–2）。inner loop 在推理时把 KV 关联写进 $W$；outer loop 端到端训练 slow weights，包括 QKV 投影、lr 预测头和可学习的初始 fast weights。
- **卡点 1：硬件利用率低**。以 fast weight $h\times h$ 乘 chunk 输入 $b\times h$ 为例，BF16 下的理想 compute-to-memory ratio 是（Eq.3）：
  $$r=\frac{2h^2b}{2h^2+4hb}=\frac{b}{1+2b/h}\le\min(h/2,\,b)$$
  $h=64$ 或 $b=16$ 时，$r$ 远低于 H100 约 290 FLOPs/byte 的峰值，运算是 memory-bound。在 Fig.1a 的柱状标注里，原版 TTT-MLP 在 H100 上只有 31 TFLOPS，LaCT 是 515；A100 上 TTT-Linear 是 7，LaCT 是 205。定制 kernel 的做法是把 fast weights 留在 SRAM 里跨步更新，但这要求 fast weights 基本在各 SM 内独立演化以减少通信，对大的非线性 state（SwiGLU fast weight）和 Muon 更新不成立；而且 kernel 开发周期长（Sec.2.2）。
- **卡点 2：state 太小**。以往方法的 state/参数比为 0.1%–5%，LaCT 不低于 40%（Sec.2.2）。
- **卡点 3：结构假设**。小 mini-batch 默认数据是 1D 有序序列，不适合集合（多视角图像）或 N 维网格（视频）。
- **假设**：监督/自监督的端到端训练，不涉及 RL；NVS 需要已知相机位姿；视频用教师强制（teacher forcing）训练。

## 核心方法

**1. 大 chunk 更新**（`source/sections/03_method.tex` Eq.4–5）：
$$g=\nabla_W\sum_{i=1}^{b}\eta_i\,\mathcal{L}\big(f_W(k_i),v_i\big),\qquad W\leftarrow\mathrm{weight\text{-}update}(W,g)$$
同一个 chunk 里的所有 $q_i$ 共享更新后的 $W$。$\eta_i=\mathrm{softplus}(\mathrm{Linear}(x_i)+b)$ 由每个 token 预测，三个矩阵各一个（Alg.1）。

**2. Fast weight 与 loss**：fast weight 网络是无 bias 的 SwiGLU-MLP，$f_W(x)=W_2[\mathrm{SiLU}(W_1x)\circ(W_3x)]$（Eq.6）；loss 是负点积 $\mathcal{L}=-f_W(k_i)^\top v_i$（Eq.7）。
*白板推导（笔者）*：把 $f_W$ 换成线性的 $f_W(x)=Wx$，就有 $\nabla_W\mathcal{L}=-v k^\top$，更新变成 $W\leftarrow W+\sum_i\eta_i v_ik_i^\top$，也就是无衰减、按 chunk 做的 linear attention。LaCT 与它的区别主要来自三处：非线性、权重归一化、Muon（此外还有可选的 momentum）。用点积 loss 时不需要算完整的 $f_W(k)$；按三个矩阵的实际 state（$3d^2r/\mathit{nh}$）计，每 token 约 $6\times$State Size FLOPs（`supp.tex` Eq.15 写作 $18n\frac{d^2}{\mathit{nh}}r=6\ast\text{State Size}$，右边漏写了 $n$；而且 Eq.14 把 State Size 写成 $\frac{d^2}{\mathit{nh}}r$，漏了因子 3，与 Table 3 的 $2304d$、附录"4 heads 时每 block $\frac34 d^2$"不一致）。

**3. 非线性 weight-update**（Eq.8–10）：
$$W\leftarrow\mathrm{L2Norm}(W-g)\quad\text{或}\quad W\leftarrow\mathrm{L2Norm}(W-\mathrm{Muon}(g)),\qquad \mathrm{Muon}(g)\simeq UV^\top\ (g=USV^\top)$$
- Muon 用 5 次 Newton–Schulz 迭代做正交化，系数 $(a,b,c)=(3.4445,-4.7750,2.0315)$（supp Eq.16）。Muon 把梯度的绝对尺度归一化掉了，所以 $\eta_i$ 只表示 chunk 内 token 之间的相对重要性。
- 在 Alg.1 里，L2Norm 的具体做法是 `(w-g)/‖w-g‖·‖w‖`，沿输入维保持更新前的范数。这一步替代了 weight decay。作者的类比：把序列维看成深度，TTT 更新就是"时间上的残差"，这个归一化相当于 post-LN。
- Momentum 沿用 Titans 的做法：$\beta_i=\sigma(\mathrm{Linear}(x_i))$ 在 chunk 内取平均，$M\leftarrow M\bar\beta+g$，$W\leftarrow\mathrm{weight\text{-}update}(W,M)$（supp Eq.19–20）。

**4. update 与 apply 的顺序决定等价 mask**（Fig.3）：
- 整个序列作为一个 chunk，先 update 再 apply，约等于双向 full attention（依 Fig.3a 图示与 Alg.1 注释；Sec.3.1 正文此处误写成"apply followed by update"）；
- update 和 apply 交替，得到 block-wise causal；
- 先 apply 再 update，得到 shifted block-causal，chunk 内不泄漏未来信息；
- 只在部分 chunk 上 update、对所有 chunk apply，得到 strided block-causal。

**5. Window attention**：TTT 把 chunk 当作无序集合，局部结构由 window attention 负责。LM 和视频采用层内 hybrid：与 SWA 共享 QKV（GAU 风格），两路输出相加，TTT 一路再乘一个 per-head 的缩放（Alg.2）。LM 要得到无空洞的逐 token causal mask，唯一条件是 window ≥ chunk（supp B.2/Fig.10）。

**6. Context parallelism**：chunk 的梯度等于各 shard 梯度之和（Eq.11），做一次 all-reduce 即可，逻辑上和 DDP 一样。实测开销 1%–3%（Sec.3.4）。

**7. 三个实例**（Table 1）：
- **NVS**：图像 patch 加 Plücker ray 作为 token。每张图内部做双向 window attention。用全部输入 token 做一次 update，再 apply 到输入和目标 token（strided mask）。渲染阶段整个模型就是一个静态 ViT：fast weight 存场景，slow FFN 存渲染规律（supp B.1）。状态 $6d^2$/block，共 0.3B 参数，最长 1M token。
- **LM**：chunk 取 2K/4K，shifted block-causal 加上等宽 SWA。状态 $0.75d^2$，模型 0.7B/3B。
- **AR 视频扩散**：序列为 $S=[X_1^{noise},X_1,X_2^{noise},X_2,\dots]$（Eq.12），只在干净 chunk 上 update、对所有 chunk apply，window 为两个 chunk。token 利用率约 50%；noisy chunk 重复两次可升到约 67%（supp Eq.22）。

**训练与推理的分工**：训练时 inner update 在 autograd 中展开，outer loop 端到端训练（NVS 用 AdamW；all-reduce 可微，因此 CP 下也能端到端）。推理时只跑 update/apply，不更新 slow weights。

## 关键实验与证据

**NVS**
- Table 2（A100，48 张 512² 输入，196K token）：prefill 时间 LaCT 1.4 s，full attention 16.1 s，Perceiver 16.8 s；渲染速度 38.7 / 2.3 / 34.4 FPS。
- GSO 256²（Table 4），48 views：PSNR LaCT 37.6，full attention 37.9，Perceiver 33.7。GSO 512²（Table 5），48 views：39.1 vs 39.3。在所有视角数下 full attention 都略高 0.1–0.4 dB。
- DL3DV 960×536（Table 6）：16/32/64/128 views 的 PSNR 为 24.7/26.9/28.3/28.9，128 views 对应 1,031,520 个输入 token。与 3DGS、LongLRM 的对比只画在 Fig.4c，没有表格数字：LongLRM 只到 32 views 且更差；稀疏视角下 LaCT 优于 3DGS，到 128 views 两者接近。

**LM**
- Table 3（3B 模型，32K，A100-40GB）的训练 TPS：Transformer 4.1K，SWA 6.4K，GLA-SWA 5.0K，DeltaNet-SWA 5.1K，LaCT 的 GD/Momentum/Muon 为 5.0K/4.9K/4.3K。state 大小分别是 384d、128d、2304d。
- Fig.5：在两个规模上，LaCT 在大 token index 处的 per-position loss 都低于 GLA 和 DeltaNet。
- RULER S-NIAH（Table 8）：
  - 760M，S-NIAH-1 @32K：Muon 92.4，DeltaNet 86.8，Momentum 84.8，Transformer 68.0，Ours 14.8（表中未注明优化器，按 Table 3 的命名应为 GD 版），GLA 8.6。
  - 3B，S-NIAH 平均 @32K：Transformer 91.2，Muon 48.0，Momentum 40.6，DeltaNet 28.4，GLA 13.3。
- Multi-needle（Table 9，3B 平均）：@4K 时 Muon 48.8，明显低于 Momentum 90.4；@32K 时 Muon 21.2，Momentum 14.2，Transformer 43.0。

**视频**（Fig.6，只报告了验证 denoising loss）：LaCT 与 full block-causal attention 接近，优于 SWA 和 Mamba2+SWA。把 window 从 4 帧加到 6 帧也有收益，但小于加入 LaCT 的收益。两者都在 10 s 视频上再微调 1,000 iter 后评测，LaCT 相对 SWA 的优势仍在（Fig.6c）。最大配置为 14B 模型、8.8 s、56,160 个视觉 token（交错序列共 107K）。

**消融**（Sec.5.4）
- **State size**（Fig.7a）：NVS 取 $0.75d^2$–$12d^2$（14 blocks、$d=768$、单头，SwiGLU 中间维 192–3072；其中 $1.5d^2$ 与 $3d^2$ 两条曲线几乎重合），LM 取 $0.375d^2$–$3d^2$（760M）；整体随 state 增大而变好，而且序列越长差距越大（$12d^2$ 时 fast weights 占参数的 40%）。Fig.1c 的 LM 验证 loss 也随 state 增大而下降。
- **Optimizer**（Fig.7b）：NVS 子图只比较了 GD 与 Muon（Muon 更好，没有 Momentum 曲线）；LM 子图上 Muon 好于 Momentum，Momentum 好于 GD。
- **非线性**（Fig.8a）：NVS 上 SwiGLU $6d^2$ 胜过 Linear $9d^2$；LM 上 SwiGLU $0.375d^2$ 胜过 Linear $0.5d^2$。
- **chunk vs per-token**（Fig.8b）：
  - NVS：同为 $d^2$ 的 state，线性 LaCT 明显优于双向 Mamba2。
  - LM：线性 LaCT（$0.25d^2$）**不如** GLA-SWA；要换成 SwiGLU $1.5d^2$ 加 Muon 才反超。
  - 结论：大 chunk 本身在 1D 文本上并不占优，它的真正作用是让大的非线性 state 和 Muon 在工程上变得可行。

## 局限与论文没说的

**作者承认的局限**（Sec.7 Limitation）：
- SwiGLU/线性 fast weight 不具备 softmax/linear attention 那样的 rotation invariance（RoPE 等相对位置编码正依赖这一性质），其实际影响尚未研究；
- 只测了三个任务，其中 NVS 需要位姿，unposed 重建没有做；
- LM 的推理能力和参数规模扩展受算力所限没有做；
- 视频缺少可靠指标，只能看验证 loss。

**笔者读出的问题**：
- **粒度与延迟的代价**：chunk 内的 token 不能通过 TTT 互相看见，完全依赖 window attention；memory 也要攒满一个 chunk 才更新一次。对逐步到达的小观测（机器人闭环控制、流式输入）来说，$b$ 天然很小，Eq.3 的效率论证就不成立了。论文没有讨论在线延迟。
- **检索仍落后 full attention**：3B 的 S-NIAH@32K 为 48.0 vs 91.2。Fig.5 图注说"Muon 一致优于 Momentum"，这只在所画的 760M S-NIAH-1、3B S-NIAH-2 和 loss 曲线上成立：Table 8 中 760M 的 S-NIAH 平均在 4K/8K/16K 上 Muon（77.9/60.5/40.0）低于 Momentum（89.9/64.3/40.9）；3B multi-needle 平均 @4K 上 Muon 也明显差于 Momentum（48.8 vs 90.4，Table 9）。正文没有提这些。
- **评测口径**：LM 只训练了 40B/60B token，没有短上下文下游基准，RULER 也只评到训练长度；视频数据是内部私有集，没有 FVD 或人评；3DGS/LongLRM 的数字只在图里。引言说"outperforms 3DGS"，Fig.4 的图注却只说在 128 views 时"competitive"。
- **文本不一致**：Sec.5.4 写成"671 tokens"，应为 671B；场景数据在正文写 1.8T token，附录是 1.5T 预训练加 4 个各"约 100B"的微调阶段（合计约 1.9T，略有出入）；Sec.3.2 Eq.9 的 Muon 更新作用在 $g$ 上、没有 momentum，Table 3 的 Muon 行却是 $\mathrm{Muon}(M)$（附录称 Muon 默认带 momentum）；Fig.3a 的正文描述与图示顺序相反；supp Eq.14/15 的 State Size 与 FLOPs 公式有笔误（见核心方法第 2 条）。
- **速度**：Muon 版的 TPS（4.3K）低于有定制 kernel 的 GLA/DeltaNet。LaCT 的优势是"同样吞吐下 state 大得多"，不是更快。另外官方仓库的 LM 部分提供了融合 TTT 层的 Triton kernel（仓库 README 说明用于降低训练显存；非论文内容），与"无需定制 kernel"的卖点有些矛盾。
- **"是不是在学习"**：线性加点积 loss 的情形就是 linear attention。[TTT-KV = Linear Attention](../ttt-kv-linear-attention/notes.md) 进一步论证，带 MLP 和 momentum 的 KV-binding TTT 也可以改写成 learned linear attention。LaCT 的增益有多少属于"test-time learning"，值得再追问。
- **算力门槛高**：NVS 场景模型在 64×A100 上训练约 10 天；3B LM 每个 run 需要 64×A100 跑 50–60 h。

## 在脉络中的位置

- **前驱**：
  - [TTT layers](../ttt-layers/notes.md)：TTT-Linear/MLP 和 mini-batch TTT，是 LaCT 直接反对的"小 batch"设定；
  - [Titans](../titans/notes.md)：momentum 做法与 LaCT 同源，LaCT 则去掉了 weight decay；
  - fast weight programmers、DeltaNet、GLA、Mamba2（线性 recurrence 的基线）；
  - Muon 优化器；
  - GAU 和 InfiniAttention（chunk attention 与 recurrence 的混合）；
  - LVSM、GS-LRM、Long-LRM（NVS 骨架与基线）；
  - Wan 2.1（视频底座）。
- **后继 / 影响**：
  - [TTT-E2E](../ttt-e2e/notes.md)：同样是 SWA 加 TTT 做长上下文 LLM，沿同一方向推进；
  - [TTT-KV = Linear Attention](../ttt-kv-linear-attention/notes.md)：对这类 KV-binding TTT 的机制提出质疑，建议对照阅读；
  - [One-Minute Video](../ttt-one-minute-video/notes.md)：几乎同期，在视频 DiT 中用小 mini-batch 的 TTT-MLP，LaCT 没有引用它，正好可以对比"小 chunk 加定制 kernel"与"大 chunk 加 PyTorch"两条路线。
- **与本项目其他主题的联系**：
  - **World model**：AR 视频扩散里"只在已生成的干净帧上 update 的 $O(1)$ 记忆"，正是长时程视频 world model 需要的组件（[Genie](../../01_world_model/genie/notes.md)、[Cosmos](../../01_world_model/cosmos-wfm/notes.md)、[Dreamer 4](../../01_world_model/dreamer4/notes.md)、[Ctrl-World](../../05_intersections/ctrl-world/notes.md)）。但论文里没有动作条件，也没有交互式 rollout。
  - **Real2sim2real**：NVS 实验相当于一次前向就把多视角图像压进 fast weights（prefill 1.4 s），可以看作逐场景优化 3DGS（[SplatSim](../../03_real2sim2real/splatsim/notes.md) 等数字孪生管线）的前馈替代。不过它只输出图像，没有几何和物理量，不能直接当物理仿真器用。
  - **RL**：fast weights 可以当作 policy 的 in-context 适应记忆，思路上接近 [RMA](../../03_real2sim2real/rma/notes.md) 的适应模块和 [PAD](../../05_intersections/pad/notes.md) 的部署期自监督更新。LaCT 提示的方向是按 episode 或 chunk 更新，而不是逐步更新。这一点属于笔者的推测，论文没有涉及。

## 前置知识

1. Linear attention / fast weight programmer（$S_t=S_{t-1}+v_tk_t^\top$），以及 DeltaNet 的"在线回归"视角；
2. TTT layers 的 inner loop / outer loop 和 mini-batch TTT；
3. Roofline 模型：算术强度（FLOPs/byte），memory-bound 与 compute-bound；
4. SwiGLU、weight normalization、RMSNorm、RoPE；
5. Muon 与 Newton–Schulz 正交化、momentum；
6. Attention mask（block-causal、SWA）；DDP/all-reduce、context parallel、tensor parallel（Ulysses）；
7. 任务背景：LVSM 式的 NVS 与 Plücker ray；视频扩散的 teacher forcing 和 diffusion forcing；RULER NIAH。

## 复现可行性

- **代码**：官方仓库 a1600012888/LaCT（MIT）。据仓库 README（论文内未列出），包含最小 LaCT 层实现、NVS 训练代码和权重（HF `airsplay/lact_nvs`）、带 Triton kernel 的 LM 代码，以及基于 Wan T2V 的视频微调代码。依赖是 PyTorch；多卡训练需要 FSDP/CP。
- **论文给出的算力**：
  - NVS 场景模型：64 张 A100，预训练 8 天，每个微调阶段约 12 h（supp C.1）；
  - LM 760M：32 张 A100-40GB，约 20 h；LM 3B：64 张 A100-40GB，约 50–60 h（supp C.2）；
  - 视频 1.3B：64 张 A100-80GB，约 20 s/iter；视频 14B：约 80 s/iter（supp C.3）。
- **数据**：
  - NVS：Objaverse（每个物体渲染 32 个视角）、GSO（评测）、DL3DV（11K 训练场景、140 个测试场景）；
  - LM：Long-Data-Collections（68.8B token），评测用 Book-3 和 RULER；
  - 视频：数据私有，无法严格复现。
- **最小复现实验**：
  1. 用约 50 行 PyTorch 实现 Alg.1。先做单元测试：线性 $f$、不做归一化时，更新结果应等于 chunk 版 linear attention；再用 $\partial o_i/\partial x_j$ 的依赖模式验证 Fig.3 的四种 mask。
  2. 单卡微基准：扫 $b\in\{16,64,256,2048,16K\}$、$h\in\{64,256,768\}$，测实际 TFLOPS，检验 Eq.3 的趋势。
  3. 用官方 NVS 权重在 GSO 256² 上评测 4–48 views，核对 Table 4（48 views 为 37.6 dB）。
  4. 在小模型上复现两条消融：optimizer 排序（Fig.7b）和"线性 LaCT 在 LM 上输给 GLA"（Fig.8b）。例如在 Objaverse 子集 128² 分辨率、6–8 个 block 上训练。
- **难点**：
  - fast-weight lr 初始化（通过 `const_lr_bias` 设定）：NVS 用 0.01（NVS 用了 Muon，论文称因此对 lr 尺度不敏感），LM 和视频用 0.001（视频不用 Muon，附录说 Muon 在验证 loss 上无明显差异）；
  - 按 supp Eq.18 后的分析（$r=1$ 时），只有 chunk 大于 $\tfrac{5}{3}hd$，Muon 的开销才小于计算 token 输出的开销；
  - 反传穿过 inner update 的激活显存很大：论文从 $512\times288$ 分辨率起的高分辨率 NVS 训练都用了 chunk 内 CP（最大 chunk 超过 50 万 token、序列超过 100 万 token）；
  - 基线（GLA/DeltaNet + SWA）需要调到公平。

## 自测问题

1. 由 Eq.3 推导：为什么 $b=16$ 时 TTT 层是 memory-bound？当 $h=768$ 时，$b$ 至少要多大才能超过 H100 约 290 FLOPs/byte 的峰值？
2. 取线性 $f_W(x)=Wx$ 和点积 loss，写出 chunk update，说明它和 linear attention 的关系。LaCT 在它之上加的三样东西各自解决什么问题（容量、幅值爆炸或衰减、尺度与数值稳定）？
3. LM 里怎样用"先 apply 后 update 的 TTT"加上 SWA 拼出严格的逐 token causal mask？为什么要求 window ≥ chunk？如果改成先 update 后 apply，会泄漏什么？
4. AR 视频里为什么只在干净 chunk 上 update？这对应 Fig.3 的哪种 mask？noisy chunk 重复两次后，token 利用率为什么从约 50% 升到约 67%？
5. Fig.8b 显示，在 LM 上线性 LaCT 输给同 state 的 GLA，而在 NVS 上线性 LaCT 赢了 Mamba2。这说明大 chunk 的收益取决于数据的什么性质？
