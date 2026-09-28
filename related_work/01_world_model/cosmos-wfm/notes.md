# Cosmos World Foundation Model Platform for Physical AI

> arXiv [2501.03575](https://arxiv.org/abs/2501.03575) · NVIDIA 技术报告（arXiv），2025 · 主题：01_world_model · 层级：milestone
> 代码：https://github.com/nvidia-cosmos/cosmos-predict1 · 笔记依据：`source/*.tex`（数字/公式）+ `paper.pdf`（图）

## 一句话

NVIDIA 先用约 20M 小时视频（切成约 $10^8$ 个 clip）和因果 video tokenizer，预训练两族"视频预测器" $\hat x_{t+1}=\mathcal W(x_{0:t},c_t)$：扩散式（7B/14B，EDM + DiT，连续 latent）和自回归式（4B–13B，Llama 式 GPT，FSQ 离散 token）；再用小数据集 fine-tune，在 text 之外加入（或替换为）camera pose、机器人指令/动作或驾驶轨迹等条件。核心想法是：先在**无动作**的海量视频上学通用视觉动力学先验，再 post-train 成针对具体 Physical AI 场景的 world model。

## 问题设定

- **动机**：Physical AI 的数据难以 scale。训练需要 observation-action 交错的序列，而探索性动作可能损坏系统和环境（`sections/1_introduction.tex`）。传统做法依赖物理建模加 system identification，得到的模型是低维、针对单一系统的，很难迁移（`sections/8_related.tex`）。
- **形式化**（`sections/2_platform.tex`，Fig. 3）： $\hat x_{t+1}=\mathcal W(x_{0:t},c_t)$。 $x_{0:t}$ 是 RGB 视频。 $c_t$ 是 "perturbation"，可以是动作、随机扰动或一段文字描述。
- **预训练假设**：数据只有视频加 VLM 生成的 caption，**没有动作、没有奖励、没有本体状态**。扩散路线的预训练是 Text2World → Video2World（过去帧 + text → 未来帧）；自回归路线先做不带文本的纯视频预测，再加文本做 Video2World（`sections/2_platform.tex`）。动作这种模态直到 post-training 才加入（`sections/6_2_robotics.tex`："a new modality not encountered during pre-training"）。
- **输出**：像素视频（由 tokenizer decoder 解码），不输出可供规划的紧凑状态，也不输出 reward 或 value。
- **之前卡在哪里**：related work 把已有 world model 分成两类：多数（含 Dreamer 系、TD-MPC2）用 RNN 在 autoencoder 学到的 latent 空间里建模；较新的一类在观测空间做条件视频生成（自回归或扩散），本文属于后者。驾驶视频生成模型则在数据规模、分辨率和相机视角数上受限（`sections/8_related.tex`）。
- **注意**：Sec 2.1 列出了 policy evaluation、policy initialization、policy training（WFM 配合 reward model 做 RL）、planning/MPC、synthetic data（Sim2Real）五种用途，但作者也明确写道 "this paper does not include empirical results in applying Cosmos WFMs to them"。

## 核心方法

**1. 数据管线**（`sections/3_curation.tex`）
- 流程依次为 split → filter → annotate → dedup → shard。
- **split**：用 TransNetV2 做 shot detection。丢弃短于 2s 的 clip，超过 60s 的切成多段。
- **filter**：
  - 用光流 + ViT 做运动过滤，去掉静止画面和手持镜头的突变运动。
  - 用 DOVER 质量分剔除最差的 15%；aesthetic 分的阈值定为 3.5。
  - 过滤文字叠加的视频，并按视频类型重新调整分布，例如去掉游戏和动画。
- **annotate**：用 VILA-13B，每个 clip 取 8 帧生成 caption，平均 97 词。
- **dedup**：InternVideo2 embedding 上做 k-means（ $k=10{,}000$），簇内按距离去重，约去掉 30% 的数据。
- **规模**：约 $10^8$ 个 clip 用于预训练，约 $10^7$ 个用于 fine-tune。类别占比：nature dynamics 20%，手部与物体操作 16%，空间导航 16%，驾驶 11%，其余略。

**2. Cosmos Tokenizer**（`sections/4_tokenizer.tex`，Fig. 9）
- 自编码形式为 $\hat x_{0:T}=\mathcal D(\mathcal E(x_{0:T}))$（Eq. 1）。输入 $(1+T)\times H\times W\times 3$，编码为 $(1+T')\times H'\times W'\times C$ 的 token。
- **结构**：
  - 先做 2 级 Haar wavelet，把 $x,y,t$ 各下采样 4 倍。
  - 然后是**时间因果**的分解 3D 卷积：先做空间 $1\times k\times k$ 卷积，再做时间 $k\times1\times1$ 卷积，左侧 pad $k-1$。卷积之后接因果时空 attention。
  - 第一帧单独成为一组 $g_0$，所以同一个网络既能编码图像也能编码视频（Fig. 7）。
- **潜空间**：
  - 连续版是 vanilla AE，latent 维度 16，不加 KL 项。
  - 离散版用 FSQ，levels 为 $(8,8,8,5,5,5)$，词表大小为 $8^3\cdot5^3=64{,}000$，不需要 commitment loss。
- **两阶段训练**：
  - 阶段 1 用 $\mathcal L_1=\lVert\hat x_{0:T}-x_{0:T}\rVert_1$（Eq. 2）加 VGG perceptual loss（Eq. 3）。
  - 阶段 2 加入光流一致性损失（不编号）和 Gram 矩阵损失（Eq. 4），fine-tune 时再加对抗损失。
- **WFM 实际使用的两个 tokenizer**：扩散 WFM 用 CV8×8×8-720p，自回归 WFM 用 DV8×16×16-720p。

**3. 扩散式 WFM**（`sections/5_1_diffusion_models.tex`，Fig. 11）

采用 EDM 形式。单个噪声级 $\sigma$ 的 denoising score matching 损失（Eq. 5）：
$$\mathcal L(D_\theta,\sigma)=\mathbb E_{x_0,n}\big[\lVert D_\theta(x_0+n;\sigma)-x_0\rVert_2^2\big],\quad n\sim\mathcal N(0,\sigma^2 I)$$
总损失在各噪声级上做**不确定性加权**（Eq. 6–8）：
$$\mathcal L(D_\theta)=\mathbb E_\sigma\Big[\tfrac{\lambda(\sigma)}{e^{u(\sigma)}}\mathcal L(D_\theta,\sigma)+u(\sigma)\Big],\quad \lambda(\sigma)=\tfrac{\sigma^2+\sigma_{\rm data}^2}{(\sigma\sigma_{\rm data})^2},\quad \ln\sigma\sim\mathcal N(P_{\rm mean},P_{\rm std}^2)$$
其中 $u(\sigma)$ 由一个小 MLP 学出来。下面是我自己的推导：对 $u$ 求导并令其为 0，得 $e^{u^{\ast}}=\lambda\mathcal L$，代回后每个噪声级的项变成 $1+\log(\lambda\mathcal L)$。也就是说，各噪声级的损失被自动归一化成对数尺度，训练后期不会有某个 $\sigma$ 主导梯度。论文只说用了 EDM 的 preconditioning，没有给出具体形式，以及 $P_{\rm mean}$、 $P_{\rm std}$ 的数值。

- **Denoiser 结构**：
  - 基于 DiT，patch 为 $(p_t,p_h,p_w)=(1,2,2)$。
  - 每个 block 依次是 self-attn（时空 token）→ cross-attn（T5-XXL 文本）→ MLP。
  - 用 AdaLN 注入时间步。AdaLN 改成 LoRA 分解后，7B 模型的参数量从 11B 降到 7B，各项指标"持平"，但论文没有给出对应数字。
  - 位置编码是 FPS-aware 的 3D 分解 RoPE，每个 block 还加一个可学习的绝对位置编码；attention 前做 QK-RMSNorm。
- **训练技巧**：
  - image 和 video 交替训练；两者的 latent 各用独立估计的统计量做标准化，video latent 还额外做逐帧、逐通道的标准化。标准化让 SNR 与尺度无关。
  - video batch 的噪声 $\sigma$ 相对 image 按 $\sqrt{\text{帧数}}$ 放大。论文的解释是：帧间冗余使 video batch 的梯度更小，denoising loss 收敛比 image 慢。
  - 渐进式训练（Table 12）：先训 512p、57 帧，context 为 10,240 token；再训 720p、121 帧，context 为 56,320 token；最后在高质量子集上 fine-tune $\mathcal O(10k)$ 步。
- **Text2World → Video2World**：
  - 条件帧的 latent 沿时间维拼在待生成帧前面，同时在通道维拼一个 binary mask 区分两者。
  - loss 不计算条件帧的位置。
  - 条件帧额外加噪声，噪声水平按 $P_{\rm mean}=-3,P_{\rm std}=2$ 采样，提高对输入扰动的鲁棒性。
  - 条件帧数量在训练中随机变化，所以推理时既能用 1 帧（图像）也能用多帧作条件。
  - 生成长视频时，每次以上一段的最后 9 帧为条件继续滚动生成（Fig. 13）。
- **CFG**：训练时不做文本 dropout，推理时靠 negative prompt 做 guidance。
- **Prompt upsampler**：基于 Mistral-NeMo-12B 微调，把用户的短 prompt 改写成训练时那种长 caption 风格。

**4. 自回归式 WFM**（`sections/5_2_autoregressive.tex`，Fig. 14）
- 训练目标是标准的 next-token NLL（Eq. 9）： $\mathcal L_{NLL}=\sum_i-\log P(v_i\mid v_{<i};\Theta)$，另加 z-loss $\lambda\sum_i z_i^2$，其中 $\lambda=3\times10^{-4}$。
- **结构**：Llama 式 decoder，3D RoPE 加 3D 正弦 APE，QK-Norm，从零开始训练，不具备语言能力。
- **训练阶段**：
  - Stage 1：以 1 帧为条件预测后续 16 帧（17 帧 context）。
  - Stage 1.1：context 扩到 34 帧，时间轴用 YaRN 外推 RoPE。
  - Stage 2：新增 cross-attn 接入 T5 文本，4B 模型由此变为 5B-Video2World，12B 变为 13B。
  - 之后在高质量数据上 cooldown 30,000 步。分辨率固定为 640×1024。
- **序列长度**：我按 DV8×16×16 推算，每个 latent 帧是 $40\times64=2560$ 个 token，33 帧对应 5 个 latent 帧，共 12,800 个 token，与 Table 14 的 "Number of Tokens" 一致，也解释了 Table 20 为什么说 AR 最长只支持 33 帧（正文 Stage 1.1/2 写的是 34 帧 context，与此差 1 帧）。
- **加速**：用 Medusa speculative decoding，9 个 head，解冻最后两层和 unembedding。
- **Diffusion decoder**（Fig. 15/16）：
  - 目的：DV8×16×16 压缩太狠，直接解码的视频偏糊。
  - 训练：以 7B Text2World 为底模 fine-tune。离散 token 先嵌入成 16 维向量并在空间上 2× 上采样，再在通道维与加噪的 CV8×8×8 latent 拼接。denoiser 学习从中还原出连续 latent。
  - 推理：离散 token → 连续 latent → CV decoder 解码出 RGB。

**5. Post-training**（`sections/6_*.tex`，所有模型都带 "-Sample" 后缀，作者强调它们只是示例，不是产品）
- **Camera control**：
  - 在 DL3DV-10K 上用 GLOMAP 估计相机位姿作为条件。
  - 每个 latent 像素的 Plücker 嵌入 $r=(d,\thinspace c\times d)\in\mathbb R^6$（Eq. 11）按通道与 latent 拼接。由于时间压缩是 8 倍，每 8 帧取第 4 帧的 Plücker 嵌入。
- **机器人**：
  - Instruction 条件：使用 Cosmos-1X 数据，约 200h 的 1X EVE 人形机器人第一视角视频，约 12,000 个 episode。
  - Action 条件：在 Bridge 上训练，动作是 7 维末端增量 $(\Delta x,\Delta y,\Delta z,\Delta\theta_r,\Delta\theta_p,\Delta\theta_y,\Delta\text{Gripper})$。7B 扩散模型把 action MLP 的输出**加到 timestep embedding 上**，5B AR 模型则通过 cross-attn 接入。推理时逐帧自回归地 rollout。
- **自动驾驶**：
  - 数据是内部 RDS，约 360 万个 20s 的六路环视 clip，约合 20,000h。
  - 六个视角作为一个整体一起去噪。每个视角使用独立的 RoPE 加一个 view embedding，cross-attn 也按视角分开，每个视角只看自己的文本描述。
  - 可选的轨迹条件是 64 个 3D 点，间隔 0.1s。

**训练与推理路径小结**：视频先经 tokenizer 得到 latent（扩散）或 token（AR）；扩散路线做去噪，AR 路线做 next-token 预测；输出再经 decoder（AR 路线可以换成 diffusion decoder）变回像素。下游适配都是离线 fine-tune，推理时不做任何适应。

## 关键实验与证据

- **Tokenizer**（Table 5、Table 9）：
  - Cosmos-Tokenize1-CV4×8×8-360p 在 DAVIS 上 PSNR 35.85，CogVideoX-Tokenizer 为 29.29；在 TokenBench 上 38.42 对 32.06。
  - 速度（Table 9，单张 A100 80GB，720×1280，按每帧平均的编码+解码时间）：Cosmos-0.1-CV4×8×8 为 34.8 ms，CogVideoX 为 414 ms。
- **3D 一致性**（Table 19，RealEstate10K 的 500 个静态场景）：
  - Sampson error：VideoLDM 0.841，7B-Text2World 0.355，真实视频 0.431。
  - 相机位姿估计成功率：VideoLDM 4.4%，Cosmos 7B-Text2World / 7B-Video2World 为 62.6% / 68.4%，真实视频 56.4%。
  - 3DGS 留出视角 PSNR：Cosmos 7B-Text2World 33.02，真实视频 35.38。
- **物理对齐**（Table 20）：
  - 用 Isaac Sim/PhysX 渲染 8 类刚体场景共 800 个视频，指标在 33 帧上计算。
  - 7B-Video2World 从 1 帧条件换成 9 帧条件后，PSNR 从 17.34 升到 21.06，平均 IoU 从 0.332 升到 0.592。
  - 14B 在 9 帧条件下 IoU 为 0.598，与 7B 几乎一样；AR 模型在 9 帧条件下的 IoU 约为 0.48。
  - 作者的结论：**模型更大并不带来更好的物理对齐**，所有模型都"equally struggle"。
- **Camera control**（Table 22）：与 CamCo 相比，位姿估计成功率 82.0% 对 43.0%，旋转误差 1.646° 对 8.277°，FVD 120.49 对 433.24。模型只在 DL3DV 上训练、在 RealEstate10K 上测试，说明有一定的跨数据集泛化。
- **机器人**：
  - Instruction 条件（Fig. 24）：10 名评估者、23 个 episode 做人工偏好评测。7B 的 overall 偏好率 78.3%，VideoLDM 为 13.0%；5B 为 56.5% 对 30.4%。
  - Action 条件（Table 23，Bridge 的 100 个测试 episode）：7B 的 PSNR/SSIM/Latent L2/FVD 为 21.14/0.82/0.32/190，IRASim 为 19.13/0.64/0.38/593。
- **驾驶**（Table 24/25）：
  - FVD：Cosmos 210.23，VideoLDM-MultiView 884.46。
  - 跨视角 Sampson error（CSE）：Cosmos 2.11，真实视频 1.71。
  - 轨迹跟随误差（TFE）：Cosmos 7B-Text2World-Sample-MultiView-TrajectoryCond 20.20 cm，真实视频 oracle 13.49 cm。
- **AR 推理**：
  - Table 16：在 1×H100 上，4B 模型以单帧为条件生成 32 帧 640×1024 视频，不用 DD 需 31.04 s，用 DD 需 61.49 s。
  - Table 15：Medusa 用 9 个 head 时，4B 的 token 吞吐提升 2.0×，5B 提升 3.2×。
  - Table 17：降到 320×512 并在目标域 fine-tune 后，在 8×H100 上达到 10.08 帧/s。
  - Table 18：单帧条件下的失败率，4B 为 15%，12B 为 2%；换成 9 帧条件后所有模型都 ≤2%。
- **消融能说明什么**：
  - 模型侧真正有数字支撑的消融只有这几项：条件帧数（Table 20）、模型规模（Table 18/20）、Medusa head 数（Table 15）、DD/Medusa 对推理时间的影响（Table 16）、tokenizer 压缩率（Table 5–8）。数据管线侧另有 shot detection 算法、转码配置、captioning 推理引擎的数字对比（Table 1–3）。
  - diffusion decoder 的效果只有定性对比（Fig. 18）；AR 模型用 prompt upsampler 无效，只是文字陈述。
  - **"预训练本身带来多少收益"没有被单独隔离出来**：baseline VideoLDM 或 IRASim 与 Cosmos 在架构、规模和数据上同时不同，没有"同架构、从零训练"的对照。

## 局限与论文没说的

**作者承认的**（`sections/9_conclusion.tex`、`5_3_2_physics_alignment.tex`、`5_2_autoregressive.tex`）：
- 缺少 object permanence，contact-rich 动力学不准，instruction following 不稳定；画面真实不等于遵守物理规律（重力、光照、流体）。
- 物理评测只覆盖刚体，模型规模对物理对齐没有帮助。
- AR 模型会出现"物体从画面下方冒出"的失败模式。
- 人工评测主观，而且不一定与下游任务的指标正相关。
- Sec 2.1 列出的五种用途都没有给出实验。

**我读出来的**：
1. **没有任何闭环验证**：没有把 WFM 接入 RL 或 MPC，也没有检验"在 WFM 里的 policy 排序是否与真实世界一致"。机器人部分的动作条件评测只是开环的 PSNR/FVD。后来 [V-JEPA 2](../vjepa2/notes.md) 把 Cosmos-7B 在 DROID 上做了动作条件 fine-tune，再用 CEM 规划：在 RTX 4090 上每个动作需要 4 分钟，reach 成功率 80%，但 grasp cup 和 pick-and-place 都是 0%（见 vjepa2 `source/main.tex` 的 `tb:cosmos` 表）。
2. **指标口径**：生成视频的 Sampson error 和位姿估计成功率都**优于真实视频**（0.355 对 0.431，62.6% 对 56.4%）。一种可能的解释是该指标会奖励相机运动更少或更简单的画面（论文没有做分析验证），所以不宜直接等同于"3D 一致性更高"。
3. **baseline 偏弱**：多数实验只和 VideoLDM（2023）比较；CamCo 只能生成 256×256、14 帧，比较并不对等。没有和同期开源视频模型做 WFM 层面的对比。
4. **规模偏小的评测**：人工评测只有 10 人 × 23 个 episode，AV 的物体追踪只看了 20 个视频。
5. **缺失的超参**：采样步数、CFG 强度、batch size、总训练步数或 token 数、各模型分别用了多少算力，论文都没有写。
6. **长时程误差累积**：Video2World 以前 9 帧为条件往后续写，但没有量化误差随 horizon 增长的曲线。这对拿它当 RL simulator 来说是关键指标。
7. **动作接入很"薄"**：预训练阶段完全没有动作，7B 模型只是把动作加到 timestep embedding 上。可控性几乎完全依赖 post-training 用的小规模机器人数据（论文所用 Bridge 为 5 FPS、320×256 的第三人称视角视频）。
8. **小的不一致**：AR 部分的正文说没有使用 GQA，但 Table 14 写的是 32 个 attention head、8 个 KV head。
9. **数据私有**：20M 小时原始视频由专有视频数据和公开互联网视频混合而成，整理后的训练集没有公开；RDS、Cosmos-1X 是内部数据。因此无法从头复现。

## 在脉络中的位置

- **前驱**：
  - 概念上，"world model"一词来自 [World Models](../world-models/notes.md)。Cosmos 在 related work 里把 [Dreamer](../dreamer/notes.md)、[DreamerV3](../dreamerv3/notes.md)、[TD-MPC2](../tdmpc2/notes.md) 归为"用 RNN 在 latent 空间建模"的一类，把 [Genie](../genie/notes.md)（自回归）和 [DIAMOND](../diamond/notes.md)（扩散）归为"在观测空间做生成"的一类，自己属于后者，并把规模推到 foundation model 级别。
  - 在"从真实视频学交互式 simulator"这条线上，前驱是 [UniSim](../../05_intersections/unisim/notes.md)。[DayDreamer](../../05_intersections/daydreamer/notes.md) 被引作真实机器人上的 world model。
  - 技术组件：EDM、DiT、W.A.L.T 的 AdaLN-LoRA、FSQ、MAGVIT 的因果 tokenizer、Medusa；机器人部分的对比对象是 IRASim。
- **后继 / 影响**：
  - [V-JEPA 2](../vjepa2/notes.md) 用 cosmos-predict1 官方的动作条件 fine-tune 代码构建 baseline，把它当作"像素生成式 world model 做规划"的代表来对比。
  - [DreamZero](../dreamzero/notes.md) 没有直接引用本文，但在与已有 World Action Model 划清区别时引用了 Cosmos 系的后续工作 Cosmos Policy（arXiv 2601.16163），训练目标部分还引用了 Cosmos-Predict2.5（arXiv 2511.00062）；两者都只是文字上的区分，没有实验对比。
  - [Ctrl-World](../../05_intersections/ctrl-world/notes.md) 和 [WMPO](../../05_intersections/wmpo/notes.md) 都引用 Cosmos 作为大规模视频 world model 的代表；WMPO 指出这类模型用在机器人上会遇到分布不匹配。
  - 综述：[WM for robot learning](../survey-wm-robot-learning/notes.md)、[Reality Gap](../../03_real2sim2real/survey-reality-gap/notes.md)。
  - 官方的兄弟仓库有 Cosmos-Transfer1 和 Cosmos-Reason1；README 已指向新一代 Cosmos 3（据仓库 README，非论文内容）。
- **与本项目其他主题的联系**：
  - **RL**：Sec 2.1 设想的"WFM + reward model 代替物理世界训练 policy"，在这篇里没有兑现。真正去做的是 [WMPO](../../05_intersections/wmpo/notes.md)、[World-Gymnast](../../05_intersections/world-gymnast/notes.md)、[Robotic World Model](../../05_intersections/robotic-world-model/notes.md)，可以对照看它们怎么补上 reward 和闭环评测。它和 Dreamer 式 latent imagination 的差距主要在 rollout 成本：Cosmos 生成 32 帧就要几十秒到两分多钟（Table 16），V-JEPA 2 用它做 CEM 规划时每个动作要 4 分钟；Dreamer 式模型在紧凑 latent 上 rollout，成本低得多（本文没有做对比）。可以对照 [Dreamer 4](../dreamer4/notes.md)，看"可扩展 world model 里训练 agent"时怎么处理推理成本。
  - **Real2sim2real**：Cosmos 是**隐式的神经 simulator**，与 [RialTo](../../03_real2sim2real/rialto/notes.md)、[SplatSim](../../03_real2sim2real/splatsim/notes.md)、[PhysTwin](../../03_real2sim2real/phystwin/notes.md) 那种显式 digital twin（3D reconstruction 加物理引擎）形成对照。有意思的是，它的物理评测反过来用 Isaac Sim 当 ground truth，3D 一致性评测则在生成视频上拟合 Gaussian splatting。"WFM 做 policy evaluation"的设想可以和 [SIMPLER](../../03_real2sim2real/simpler/notes.md) 用仿真做 policy evaluation 对比着看。
  - **TTT**：Cosmos 的全部适配都是离线 post-training，部署时模型不变。能不能让 WFM 在部署时自适应新的机器人或场景（TTA/TTT，参见 [PAD](../../05_intersections/pad/notes.md)），或者用 TTT layer 改善长视频一致性（参见 [One-Minute Video TTT](../../02_test_time_training/ttt-one-minute-video/notes.md)），都是空白。[SimDist](../../05_intersections/simdist/notes.md) 走的是"在 sim 里预训练 world model、再快速适应真实世界"的反向路线，可以对照。

## 前置知识

1. 扩散模型与 score matching：EDM 的参数化（ $\sigma$、preconditioning、log-normal 噪声采样）、classifier-free guidance、latent diffusion。
2. 视觉 tokenizer：AE/VAE、VQ-VAE 与 FSQ 的区别，因果 3D 卷积。
3. Transformer 组件：DiT 与 AdaLN、cross-attention、RoPE（3D 分解形式，NTK/YaRN 外推）、QK-Norm。
4. 自回归生成：next-token NLL、KV cache、speculative decoding（Medusa）。
5. 评测相关的几何与指标：对极几何（基础矩阵、Sampson error）、SfM、Plücker 坐标、PSNR/SSIM/LPIPS/FID/FVD。
6. World model 在决策中的用法：model-based RL、MPC/CEM，用来理解 Sec 2.1 设想的那些用途。
7. （选读）分布式训练：FSDP、context parallelism、TP/SP。

## 复现可行性

- **代码与权重**：`nvidia-cosmos/cosmos-predict1`（论文 abstract 给出的链接）。权重是 NVIDIA Open Model License（论文 Sec 1）；代码是 Apache 2.0（据仓库 README，论文未写）。仓库包含推理、post-training（其中有一个用 IRASim 版 Bridge 数据做动作条件 post-training 的示例）和各个 tokenizer。截至 2026-09，README 已声明不再积极开发，并指向 github.com/NVIDIA/Cosmos（以上仓库信息均来自 README/文档，非论文内容）。TokenBench 和 ShotBench 也已开源（论文给出了链接）。
- **算力**：
  - 预训练：所有 WFM 合计用了 10,000 张 H100、训练 3 个月（`sections/5_0_pretraining.tex`），加上数据私有，**完全无法复现**。
  - 显存：14B 模型的参数、梯度和优化器状态约需 280 GB，高分辨率下激活约需 310 GB，因此必须用 FSDP 加 CP（Sec 5.1.4）。
  - post-training：论文**没有报告**所需算力。仓库文档对 7B Video2World 动作条件 post-training 的推荐是 8 张 H100-80GB 或 A100-80GB。
- **最小复现实验**：
  1. 下载 Cosmos-Predict1-7B-Video2World，用仓库的 action-control 流程在 Bridge（IRASim 划分）上做 post-training。在 100 个测试 episode 上计算 PSNR/SSIM/Latent L2/FVD，对照 Table 23 的 21.14/0.82/0.32/190。
  2. 在此基础上补做一个论文没做、但本项目关心的实验：画出 rollout 误差随 horizon 变化的曲线，并用若干个 Bridge policy 检验"WFM 内的 policy 排序"是否与真实结果一致。
  3. 更便宜的替代方案：单卡跑 tokenizer 在 DAVIS/TokenBench 上的重建（Table 5）；或者自己在 Isaac Sim 里渲染几个刚体场景，复现"1 帧 vs 9 帧条件"的 IoU 差异（Table 20 的 800 个视频的 benchmark，论文没有说明是否公开）。
- **预计难点**：
  - 显存门槛高：7B/14B 必须多卡 80GB。
  - 推理慢：V-JEPA 2 报告规划时每个动作要 4 分钟。
  - 依赖多：T5-XXL 文本编码器；guardrail 的 pre-Guard 用到 Aegis（基于 Llama-Guard 微调，Sec 7），模型下载的许可要求论文没有写。
  - 论文缺少采样和 CFG 超参，只能以仓库默认值为准。

## 自测问题

1. 为什么 Cosmos tokenizer 要做成时间因果的？第一帧单独成组对 joint image-video 训练有什么作用？对下游 Physical AI 又意味着什么？
2. 在 Eq. 6 中对 $u(\sigma)$ 求最优，每个噪声级的有效损失变成什么？另外，video batch 为什么要把 $\sigma$ 按 $\sqrt{\text{帧数}}$ 放大？
3. Video2World 是怎么从 Text2World 改出来的？条件帧如何注入、loss 如何 mask、为什么要给条件帧加噪？把它当 simulator 连续 rollout 时，误差会从哪里累积？
4. 自回归 WFM 为什么还需要 diffusion decoder？它训练时的条件、目标和底模分别是什么？
5. Table 19 中生成视频的 Sampson error 比真实视频还低，这暴露了指标的什么问题？如果要证明 Cosmos 可以当 RL 的 simulator 或 policy evaluator，你会补哪几个实验？
