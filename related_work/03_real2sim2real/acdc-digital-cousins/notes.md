# Automated Creation of Digital Cousins for Robust Policy Learning (ACDC)

> arXiv [2410.07408](https://arxiv.org/abs/2410.07408) · CoRL 2024 · 主题：03_real2sim2real · 层级：milestone
> 代码：https://github.com/cremebrule/digital-cousins（项目页 https://digital-cousins.github.io/ ）· 笔记依据：`source/*.tex`（数字/公式）+ `paper.pdf`（图）
> 源码结构：`source/main.tex` → `secs/{0-Abstract,1-Introduction,3-Methodology,4-Experiments_Ablations,2-RelatedWork,5-Conclusion}.tex`，附录 `appendix.tex` → `supp/{Method,Experiment}.tex`。PDF 章节号：§2 方法、§3 实验、附录 A（方法细节）/ B（实验细节）。

## 一句话

从**一张 RGB 图**出发，用 GPT-4o + Grounded-SAM-v2 + Depth-Anything-v2 抽取物体，再用 DINOv2 特征在 BEHAVIOR-1K 资产库里检索出一组"几何/语义 affordance 相近但不完全相同"的资产（**digital cousins**），自动拼成可交互仿真场景；在这些 cousins 上用脚本示范做 behavior cloning，得到的策略比只在 digital twin 上训练的策略更鲁棒，并能 zero-shot 回到真实场景（真实开门成功率 90% vs twin 25%，PDF Figure 5）。

## 问题设定

- **要解决的问题**：sim-to-real 两个极端都不理想。(a) domain randomization / 程序化场景：分布宽但与目标场景脱节，交互数据质量不够；(b) digital twin：贴合目标场景，但构建贵（需人工，如 [RialTo](../rialto/notes.md) 的 GUI 流程），且策略只对单一实例优化，场景一变就掉点（`secs/1-Introduction.tex`）。
- **digital cousin 的定义**：不显式复刻某个真实物体，但保留相似几何与语义 affordance 的虚拟资产/场景（例如把手、抽屉布局相近的柜子）。一个真实场景只有一个 twin，但可以有多个 cousins。
- **输入**：单张 RGB 图 $\mathbf{X}$ + 相机内参 $\mathbf{K}$（附录 B.2 还演示了用 PerspectiveFields 估计 $\mathbf{K}$ 的无内参版本）。
- **输出**：若干完全可交互、物理上稳定的仿真场景 + 在其中训练、可 zero-shot 部署的操作策略。
- **假设**：有一个带类别标注、可交互（含铰接）资产的大库（BEHAVIOR-1K，超过 10,000 个资产，附录 A.1）；仿真里有 ground-truth（把手位置、关节类型）供脚本技能采示范；策略观测为点云 + 本体感知，因此视觉纹理差异被绕开；只建模"on top"的支撑关系；**不做**任何物理参数（质量、摩擦、关节阻尼）辨识。

## 核心方法

整条 pipeline 分成 ACDC（real-to-sim 场景生成）和策略学习两段（`secs/3-Methodology.tex` §2.1–2.2，Figure 2）。

**Step 1：Real-world extraction。** GPT-4 为图中所有物体生成 caption $\lbrace\mathbf{c}_ j\rbrace_{j=1}^M$ → Grounded-SAM-v2 据此给出 mask $\lbrace\mathbf{m}_ i\rbrace_{i=1}^N$ → 再问 GPT 为每个 mask 从 caption 列表里选定标签 $\mathbf{l}_ i$（把两个模型的标签对齐）。深度不用深度相机（对反光表面不稳），改用 Depth-Anything-v2 估计 $\mathbf{D}$，再反投影成点云：

$$\mathbf{P} = \mathbf{D}\cdot\mathbf{K}^{-1}\qquad(\texttt{source/secs/3-Methodology.tex}，\S2.1)$$

（逐像素即 $\mathbf{p}(u,v)=D(u,v)\thinspace\mathbf{K}^{-1}[u,v,1]^\top$。）每个物体得到 $\mathbf{o}_ i=(\mathbf{l}_ i,\mathbf{m}_ i,\mathbf{p}_ i,\mathbf{x}_ i)$，其中 $\mathbf{p}_ i$ 为物体点云（再用 DBSCAN 去噪，附录 A.7）， $\mathbf{x}_ i$ 为 masked RGB。

**Step 2：Digital cousin matching（层级检索）。** 离线把每个资产在固定相机位姿下旋转拍多张快照，得到 $\mathbf{a}_ i=(\mathbf{t}_ i,\mathbf{I}_ i,\lbrace\mathbf{i}_ {is}\rbrace_{s=1}^{N_{snap}})$（类别、代表快照、多朝向快照；附录 A.1）。选择"2D 快照 + 视觉编码器"而不用点云配准/位姿估计，是因为单视角遮挡下拿不到完整点云。在线三级过滤：

1. **类别**：CLIP 文本相似度 $\text{sim}(\mathbf{l}_ i,\mathbf{t})$ 取 $\text{top-}k_{cat}$ 类别；
2. **模型**：在这些类别内比较 $\mathbf{x}_ i$ 与代表快照 $\mathbf{I}_ j$ 的 DINOv2 距离，取 $\text{top-}k_{cand}$；
3. **cousin + 朝向**：对候选的所有快照 $\mathbf{i}_ {js}$ 再算 DINOv2 距离，取最近的 $k_{cous}$ 个，每个 cousin 为 $(\mathbf{A}_ c,\mathbf{q}_ c)$，朝向 $\mathbf{q}_ c$ 来自匹配到的快照。

DINOv2 距离的定义在附录 A.6（`source/supp/Method.tex`，原文为文字描述，以下为本笔记的形式化）：设输入的 patch 特征为 $\lbrace\mathbf{e}_ u\rbrace$，候选 $j$ 的 patch 特征为 $\lbrace\mathbf{f}_ {j,v}\rbrace$，

$$d(\mathbf{x},\mathbf{i}_ j)=\frac{1}{|\mathcal{U}'|}\sum_{u\in\mathcal{U}'}\min_v\Vert\mathbf{e}_ u-\mathbf{f}_ {j,v}\Vert_2,$$

$\mathcal{U}'$ 为去掉最大 10% 最近邻距离后的 patch 集合（去掉离群值，排序更"salient"）。top-1 用**投票**选：每个输入 patch 在所有候选的全部 patch 中找最近邻，得票最多的候选胜出； $\text{top-}k$ 即反复取 top-1 再移除（faiss GPU 加速）。

GPT 精修（附录 B.1）：**GPT Model Selection** 先用 DINO 取 $k_{model}=10$ 个候选再让 GPT 挑；**GPT Orientation Selection** 从 DINO 给的 $k_{ori}=4$ 个朝向里让 GPT 挑（<10 s/物体，对比 DINO 重渲染选朝向约 60 s/物体）。铰接物体只在铰接资产里检索，且可以限定门/抽屉数量差（real-to-sim 最近 cousin 阈值设为 2，附录 A.7）。

**Step 3：Simulated scene generation。** 把资产包围盒中心放到 $\mathbf{p}_ i$ 的质心，按 $\mathbf{p}_ i$ 的范围缩放；拟合地面/墙面平面；让 GPT 判断挂载类型（Wall Mounted / On Floor or On Another Object / Mixture，附录 A.2）。后处理（附录 A.3）：按高度排序后，把包围盒投影到 x-y 平面成多边形，用下式推断 "on top"：

$$\text{area}(poly_i\cap poly_j) > 0.7\cdot\min\big(\text{area}(poly_i),\text{area}(poly_j)\big)$$

再按挂载类型做去穿透（ $j$ 为下方物体）：(1) Wall Mounted 先旋转使局部 x/y 轴对齐墙法向、缩放/平移使背面与墙共面，再在 $z(\mathbf{p}^{TR}_ j)>z(\mathbf{p}^{BL}_ i)$ 时把 $z(\mathbf{p}^{cen}_ i)$ 增加 $z(\mathbf{p}^{TR}_ j)-z(\mathbf{p}^{BL}_ i)$，否则不动，最后固定到 GPT 选的墙上；(2) On Floor or On Another Object 直接放到 $j$ 上，即 $\mathbf{p}^{cen}_ i$ 调整 $|z(\mathbf{p}^{TR}_ j)-z(\mathbf{p}^{BL}_ i)|$；(3) Mixture 朝向/尺度按 (1) 处理，位置按 (2) 处理。最后在 x-y 平面消除碰撞。耗时约：抽取 7 s/物体、匹配 20 s/每个 cousin、场景生成 <30 s/场景（附录 A.7）。

**策略学习（§2.2 + 附录 A.4–A.5、B.5）。**
- **示范**：基于 cuRobo 运动规划 + GPG 抓取采样实现 Open / Close / Pick / Place 四个脚本技能，依赖仿真 ground-truth（射线找最突出几何作为把手、读父关节类型算解析开合轨迹）。rejection sampling 只保留成功 episode，因此可以放大随机化范围；每个 episode 随机化机器人位姿、物体位姿与尺度。
- **策略**：behavior cloning，RNN（horizon 10，hidden 512）+ GMM 输出头（即 BC-RNN-GMM）；观测 = 末端位姿 + 夹爪状态 + 统一到机器人坐标系、去背景后的点云（附加一维 $e\in\lbrace0,1\rbrace$ 标记夹爪/场景点，映射成 128 维 embedding），2 层 512 维 PointNet 编码，FPS 降采样；动作 = 6 维 delta 末端位姿（IK 执行）；AdamW。标准目标形式（论文未写公式）： $\max_\theta \mathbb{E}_ {\mathcal{D}_ {cous}}\log\pi_\theta(a_t\mid o_{t-9:t})$， $\mathcal{D}_ {cous}$ 为在 cousins 上均分采集的示范。
- **推理**：真实部署时用 XMem 跟踪/抠掉无关物体，使真实点云与仿真点云对齐，zero-shot 执行，不做任何 fine-tuning。

## 关键实验与证据

**Q1 场景重建（sim-to-sim，Table 1 = `source/table/sim2sim_reconstruction_best_only.tex`）**：输入为仿真场景的单张渲染图，场景物体本身就在资产库里，所以有 ground-truth 类别/模型与位姿、尺度。4 个场景（尺度 3.42–10.23 m）类别与模型都全对（6/6、8/8、10/10、15/15）；位置误差 $\mathcal{L}_ 2$ 为 4.15±2.04 / 7.65±5.62 / 4.77±3.38 / 15.67±8.86 cm，朝向误差 0.03–0.12 rad，Bbox IoU 0.59–0.74。附录 Table 2（`supp/table/sim2sim_textwidth.tex`）在 8 个场景上比较 (a)–(d) 四种 DINO/GPT 组合：按逐场景相加（本笔记自算），纯 DINO 模型选择正确 39/61 个物体，GPT 模型选择 57/61；朝向上 GPT 与 DINO 大体相当。

**Q2/Q3 sim-to-sim 策略（Figure 4；原始数字 Table 4 = `supp/table/dc_sim2sim_raw_numbers.tex`）**：以一个 BEHAVIOR-1K 柜子作 twin，Door/Drawer 共 10000 条示范、Putting Away Bowl 2000 条，均分到训练实例上；每个设置 6 个 run（2 组最佳超参 × 3 seed，Bowl 为 5 个 run），每个 run 100 次 rollout；测试 twin 和 3 个 DINO 距离递增且**不在训练集**的资产。6 个 run 的均值（本笔记由 Table 4 自算，与 Fig. 4 曲线一致）：

| 任务（测试资产 DINO 距离） | Twin | 8 Cousins | All Assets（最近 3 类全部资产） |
|---|---|---|---|
| Door（0 / 7.25 / 7.59 / 18.93） | 90.0 / 81.8 / 67.5 / 63.3 | 93.2 / 93.3 / 93.3 / 75.2 | 77.7 / 81.2 / 78.2 / 63.3 |
| Drawer（0 / 9.42 / 14.97 / 17.6） | 68.3 / 73.7 / 65.5 / 33.0 | 69.7 / 79.5 / 79.3 / 84.2 | 42.8 / 55.8 / 55.3 / 57.2 |
| Bowl（0 / 14.17 / 14.44 / 17.73） | 16.2 / 0 / 0.6 / 0 | 8.6 / 5.2 / 23.8 / 0 | — |

结论：cousins 在 twin 上与 twin 策略持平（Q2），在未见资产上明显更稳（Q3）；全资产的"naive DR"在分布内反而最差，作者据此称 cousins 是一种 *conditional* randomization。作者观察到 twin 策略的性能大体随 DINOv2 距离增大而下降（Door 单调；Drawer 在 9.42 处反而略升），据此称 digital cousins（按 DINOv2 距离排序）可作为 out-of-distribution 表现的代理。Drawer 的 4-cousin 策略异常差，附录 B.6 归因于可选抽屉柜太少、4 个 cousin 的 DINO 距离断层（7.78, 9.32, 14.10, 14.90；对比 Door 的 6.49, 7.51, 8.13, 9.66）。

**消融（附录 B.1、B.6）**：
- 编码器（Figure 7 / Table 3，Door）：DINO / DINO+GPT 选的 2/4/8 cousins 训练的策略在 twin 上约 88–92%（均值自算，论文写 approximately 90%）；CLIP 选的 2/4/8 cousins 策略在 twin 上只有 71.8 / 58.5 / 77.5（均值自算），说明 DINO 比 CLIP 更能抓几何相似性。
- 与加大 DR / 混合数据对比（Figure 14 / Table 5，Door，DINO+GPT 选 cousins，3 seed）：在第 11、12 个未见 cousin 上，Twin 为 2.7 / 44.0，Twin（尺度随机化加到 ±75%）为 32.7 / 73.7，8 Cousins 为 49.0 / 93.7；Twin+3/7 Cousins（各一半数据来自 twin）与纯 Cousins 相近（41.7 / 87.0、34.0 / 88.0），作者据此称"精确 twin 并非必要"；但 Twin+1 Cousin 只有 12.7 / 51.7（均值自算）。

**Q4 sim-to-real（PDF Figure 5，`source/table/sim2real.tex`）**：真实 IKEA 柜子 Door Opening，仿真 50 次 / 真实 20 次：

| 策略 | Sim | Real |
|---|---|---|
| Twin | 100% | 25% |
| Twin + ↑DR（位姿、尺度） | 70% | 55% |
| Twin + Cousin | 92% | 95% |
| Cousin | 94% | 90% |

完整 real-to-sim-to-real（野外厨房图 → cousins → 开柜门）只在 Figure 1 与项目网站上定性展示。

## 局限与论文没说的

**作者承认**（§5、附录 B.3）：受资产库覆盖限制（BEHAVIOR-1K 只有 1 个锅、1 个烤面包机、2 个咖啡机）；继承各基础模型的失败模式（高频深度如植物/栅栏、遮挡、类别命名不一致如 cup vs coffee_cup）；只建模 "on top" 关系；策略学习较简单，可以换 diffusion policy。

**我读出来的**：
- **真实世界证据很薄**：定量只有 1 个任务、1 个柜子、每个策略 20 次试验；完整 pipeline 的厨房实验没有成功率。真实实验用了几个 cousin 也没写。
- **sim-to-sim 评测有循环性**："未见资产"与训练 cousins 取自同一资产库，并按同一种 DINO 距离排名，所以"DINO 距离是 OOD 代理"一定程度上是用同一把尺子量。对所有方法都不在 top-12 的 "OOD" 资产，用 DINO+GPT 选出的 cousin 策略也只有约 48–55%，twin 约 48–50%（Table 3/5，自算；最高的是 Twin+3 Cousins(↑Rand) 71.0 和 Twin+All Assets(↑Rand) 72.0，而后者训练时已经见过测试资产），在真正远离的分布上收益有限。
- **sim-to-sim 场景重建偏容易**：输入物体本身就在检索库里，"Mod." 正确等于检索回原资产；真实世界没有 ground truth，只有定性对比。
- **只补了几何/语义 gap**：物理参数不做辨识；§2.2 说可以加 visual / physics 随机化，但附录 A.5 实际只写了随机化机器人位姿、物体位姿与尺度（对比 [SimOpt](../simopt/notes.md)、[ASAP](../asap/notes.md)、[PhysTwin](../phystwin/notes.md)）；策略只用点云，所以"纹理不匹配"没有被检验过，RGB 策略能否受益未知（对比 [SplatSim](../splatsim/notes.md)）。
- **长时程任务基本没解决**：Putting Away Bowl 的最高均值只有约 24%，twin 在 twin 上 16.2%，cousin 8.6%，正文没讨论。另外附录 Figure 13 的 Task 3 子图横轴（7.25/7.59/18.93）与数值看起来是 Task 1 的复制，与 Table 4 不符，应以 Table 4 为准。
- **评测口径偏乐观**：超参按 twin 上的成功率挑选（等于在分布内测试集上调参），而且每个 run 取最佳 checkpoint（附录 B.5、B.6）。Drawer 的原始数字里单个 run 会掉到个位数，方差很大。
- **示范依赖特权信息**：把手检测靠"最突出几何"的启发式，关节类型读自资产；技能只有四个，扩到可变形物体或非铰接的复杂交互并不直接。说"也适用于 RL"，但没有做 RL 实验。
- **算力未报告**：训练 GPU、示范生成耗时（cuRobo 规划 × 10k 条）都没写；依赖 GPT-4o 付费 API，结果也带不确定性。

## 在脉络中的位置

- **前驱**：[Domain Randomization](../domain-randomization/notes.md)（在参数空间随机化；ACDC 改成在"与目标相近的资产集合"上随机化）；[SimOpt](../simopt/notes.md)（用真实 rollout 调整随机化分布；ACDC 用一张图来约束分布）；[RMA](../rma/notes.md)（ACDC 在 intro 中列为物理参数随机化的代表）；BEHAVIOR-1K 资产与仿真、URDFormer（单图生成铰接场景，附录 B.4 做了定性对比）、[RialTo](../rialto/notes.md)（human-assisted digital twin，概念上是 ACDC 的直接对照，但论文没有和它做实验比较）；DINOv2 / CLIP / Grounded-SAM / Depth-Anything 这套基础模型栈；robomimic 风格的 BC-RNN-GMM。
- **后继 / 影响**：[X-Sim](../x-sim/notes.md) 在局限部分（`x-sim/source/paper/limitations.tex`）引用 ACDC，把"从大型资产库检索 mesh"当作扩展到互联网视频的方案；[Reality Gap 综述](../survey-reality-gap/notes.md) 可对照其 real-to-sim 与 DR 的分类来理解 ACDC 的位置。
- **与本项目其他主题的联系**：
  - **World model**：ACDC 的"模拟器"是资产检索 + 物理引擎，作者强调它比生成模型合成的数据更 physically plausible（§4）。另一条路线是把学到的生成式 world model 当 simulator，如 [UniSim](../../05_intersections/unisim/notes.md)、[Ctrl-World](../../05_intersections/ctrl-world/notes.md)、[World-Gymnast](../../05_intersections/world-gymnast/notes.md)、[Cosmos](../../01_world_model/cosmos-wfm/notes.md)。[SimDist](../../05_intersections/simdist/notes.md) 在仿真中预训练 world model，cousin 场景分布可以当它的预训练环境来源。
  - **TTT/TTA**：ACDC 的鲁棒性完全来自训练时扩大分布，部署时不适应。可以对照 [PAD](../../05_intersections/pad/notes.md)、[TTT](../../02_test_time_training/ttt-self-supervised/notes.md)、[Tent](../../02_test_time_training/tent/notes.md) 这类在部署时自监督更新的方法。"DINO 距离 ∝ 掉点"的观察也提示可以用它来判断何时需要 test-time adaptation。
  - **RL**：ACDC 只做 BC；RialTo 在 twin 中用 [PPO](../../04_rl/ppo/notes.md) fine-tune。在 cousins 上做 RL fine-tune（如 [DPPO](../../04_rl/dppo/notes.md)）能否比 BC 更好地利用分布宽度，论文没有回答。

## 前置知识

1. sim-to-real gap、domain randomization、digital twin / system identification 的基本取舍（先读 [Domain Randomization](../domain-randomization/notes.md)、[RialTo](../rialto/notes.md)）。
2. 模仿学习：behavior cloning、BC-RNN-GMM（robomimic）、脚本示范加 rejection sampling。
3. 视觉基础模型：DINOv2 的 dense patch 特征与最近邻匹配；CLIP 的文本-图像嵌入；Grounded-SAM（开放词汇检测 + 分割）；单目深度估计。
4. 针孔相机模型：内参 $\mathbf{K}$、深度反投影成点云、DBSCAN 去噪、FPS 降采样、PointNet。
5. 铰接物体（revolute / prismatic joint、URDF）、运动规划（cuRobo）与抓取采样（GPG）。

## 复现可行性

- **代码**：官方开源 https://github.com/cremebrule/digital-cousins（已用 WebFetch 确认存在）。据 README（非论文）：Linux + Conda，建议 NVIDIA RTX GPU（24 GB 以上显存）+ CUDA 12.1+，仿真器为 OmniGibson；依赖 DINOv2、SAM2、GroundingDINO、Depth-Anything-V2、CLIP、PerspectiveFields、robomimic、faiss-gpu，需要 GPT API key。仓库里有 ACDC 主脚本、场景加载脚本、twin 与 cousin 策略的示例 rollout 脚本（附预期成功率），以及单元测试。
- **算力**：论文**未报告**训练 GPU 型号、数量和时长；只给了 ACDC 推理耗时（见上）。
- **数据/环境**：BEHAVIOR-1K 资产（>10,000 个）+ 仿真器；示范全部由脚本在仿真中生成，不需要人类示范。真实部署需要标定相机、XMem 做目标跟踪，以及与仿真一致的点云坐标系。
- **最小复现实验**：(1) 先跑仓库提供的 twin 和 cousin 策略 rollout 脚本，确认环境和预期成功率；(2) 用 ACDC 处理一张单柜子照片，检查 $\text{top-}k$ cousins 与 DINO 距离；(3) 缩小版 Door Opening sim-to-sim：固定一个 BEHAVIOR-1K 柜子作为 twin，用 DINO(+GPT) 选 8 个 cousins 并**剔除**测试资产，每个条件先采 2k 条示范、训 1 个 seed 的 BC-RNN-GMM，在 twin 与 2–3 个 DINO 距离递增的 held-out 柜子上各跑 100 次，对比 Twin / 8 Cousins / All Assets，目标是复现 Fig. 4 Task 1 的**趋势**，而不是精确数字。
- **预计难点**：OmniGibson/Isaac Sim 与多个基础模型依赖难装；GPT 输出不确定，影响检索结果；10k 条 cuRobo 示范生成耗时不小；超参要按论文的"2 组超参 × 3 seed"流程走，否则方差（尤其 Drawer）会淹没差异；真实迁移依赖点云对齐细节。难度：仿真部分 medium，真实部分 hard。

## 自测问题

1. 为什么 ACDC 用"多朝向 2D 快照 + DINOv2 patch 最近邻"来匹配资产和朝向，而不用点云配准或 FoundationPose 一类的位姿估计？DINO 距离里去掉最大 10% 最近邻距离起什么作用？
2. 层级匹配的三级（CLIP 类别 → DINO 代表快照 → DINO 多朝向快照）各过滤掉什么错误？GPT Model Selection 修正的是 DINO 的哪类失误（参考 Table 2 中 39/61 vs 57/61）？
3. 给定两个物体的 3D 包围盒，按附录 A.3 的规则手推：怎样判定"i 在 j 上面"，去穿透时 $z$ 要平移多少？贴墙物体为什么要单独处理？
4. 为什么在"最近 3 类的全部资产"上训练，在 twin 上的效果反而不如只在 twin 上训练，而 8 个 cousins 就不会这样？Drawer 任务里 4-cousin 的异常说明 cousin 集合应该满足什么性质？
5. Figure 5 中 Twin 在仿真 100%、真实 25%，Twin+↑DR 为 55%，Cousin 为 90%：cousin 分布大概覆盖了哪些 sim-to-real 误差来源（资产建模、感知/点云），哪些没有覆盖（动力学参数）？怎样设计实验把这两类因素分开？
