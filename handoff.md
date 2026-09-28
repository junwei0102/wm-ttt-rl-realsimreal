# Handoff

> 每个窗口结束前更新这份。**下一个 Agent 只读 AGENTS.md + 这份就能接上。**
> 写给"完全不知道前情的人"看，不要用只有你懂的简称。
> 最后更新：2026-09-28 · 第 1 个窗口

---

## 一句话现状

**文献调研阶段做完了：54 篇（50 篇论文 + 4 篇综述）已抓取、逐篇精读并对照 LaTeX 源码核查，五个主题 README 和总路线已写好；还没开始复现，下一步是按学习路线阶段 0 学 RL 基础并复现 ① PPO。**

---

## 已经确定不用再动的

- ✅ **项目骨架**：按 [research-workflow](https://github.com/skJack/research-workflow) 建好，`AGENTS.md` 已填（目标 / deadline 一行待定）。
- ✅ **选文**：5 个 scout agent 各做一条线的文献普查（共 188 个候选），curator 去重并按 foundational / milestone / frontier 选出 50 篇 + 4 篇综述，critic 用 arXiv API 逐个核对了 ID 与标题，并补了 Titans、legged-gym 两篇经典。落选但值得读的都记在各主题 README 的"扩展阅读"里，没有静默丢弃。清单与元数据：[related_work/index.json](related_work/index.json)。
- ✅ **抓取**：54 篇的 `paper.pdf` + 清洗过的 `source/` 都在本地（`.gitignore` 排除，不进版本库；新机器上跑 `bash scripts/fetch-all.sh` 重新抓）。
- ✅ **精读笔记**：每篇一份 `related_work/<topic>/<slug>/notes.md`，统一 9 节模板（一句话 / 问题设定 / 核心方法 / 关键实验与证据 / 局限 / 脉络 / 前置知识 / 复现可行性 / 自测问题）。
  - 验证方式：每篇由另一个独立 agent 对照 `source/*.tex` 和 PDF 逐条核查数字、公式、benchmark 名、代码链接，共检查 6206 条、修正 651 处；查不到出处的删除或标"（原文未报告）"。
- ✅ **综述层**：`related_work/<topic>/README.md` ×5（脉络、分类、机制对比表、学习路线、开放问题、扩展阅读、复现推荐）+ [related_work/README.md](related_work/README.md)（统一视角、概念地图、6 阶段学习路线、总索引、复现队列、12 个交叉开放问题）。
  - 验证方式：最终 critic 检查了全部相对链接和锚点（60 个文件，无断链）、模板标题完整性、README 与 notes 的事实一致性（修了 15 处）；发表 venue 另由 agent 对照 OpenReview / 会议论文集 / DBLP 核实。

## 正在做

- 🔄 **学习路线阶段 0：RL 基础**（[related_work/README.md §3](related_work/README.md#3-学习路线)，约 2–3 周）
  - 读：[PPO](related_work/04_rl/ppo/notes.md)、[SAC](related_work/04_rl/sac/notes.md)、[DQN](related_work/04_rl/dqn/notes.md)、[Murphy 综述](related_work/04_rl/survey-rl-overview/notes.md) 的相关章节
  - **下一步**：复现 ① PPO → 建 `复现/ppo/`，按 [related_work/README.md §5](related_work/README.md#5-复现队列) ① 的最小实验跑 Hopper / HalfCheetah × {不加约束, clip ε=0.2, adaptive KL} × 3 seeds，写 `复现/ppo/复现报告.md`。纯 CPU，本机可做。

## 还没动

- ⬜ **复现 ②–⑥**（PAD、TD-MPC2、Robotic World Model、SimDist、SIMPLER）—— 除 PAD 外都需要 Linux + NVIDIA GPU，本机 macOS 跑不了；等有 GPU 机器再做。队列与最小实验见 [复现/README.md](复现/README.md)。
- ⬜ **`experiment/evaluation.md`、`experiment/results.md`、`dataset.md` 仍是模板** —— 现在还没有自己的实验，等复现完、确定第一个假设时再定测评口径，别提前写死。
- ⬜ **`AGENTS.md` 的"目标"一行**（会议 / deadline）待定。
- ⬜ **brainstorm**：12 个交叉开放问题已列在 [brainstorm.md](brainstorm.md) 的"待想清楚"，但都还没讨论；按原则要复现之后再展开。
- ⬜ **扩展阅读里的论文没抓**（各主题 README 的"扩展阅读"节，约 50 篇）。需要时用 `bash scripts/fetch-paper.sh <arxiv-id> <topic>/<slug>` 加。

---

## 踩过的坑，别再踩

1. **上游 `fetch-paper.sh` 在论文没有任何 `\newcommand` 时会报错退出** —— `grep -c` 全部为 0 时返回 1，`find -exec … +` 跟着失败，在 `set -euo pipefail` 下脚本中断（此时 PDF 和清洗都已完成，只是没打印 ✓）。项目里的 `scripts/fetch-paper.sh` 已在那一行末尾加 `|| true` 修掉。
2. **模板 `.gitignore` 的 `related_work/*/paper.pdf` 匹配不到按主题分的子目录** —— 本项目是 `related_work/<topic>/<slug>/`，要用 `related_work/**/paper.pdf` 和 `related_work/**/source/`，否则会把 1.2 GB PDF / 源码提交进去。
3. **论文内部自相矛盾的数字不少** —— 例如 TD-MPC2 的 57.1 / 54.2 / 56.54、SIMPLER Table I 与 Table IV 的 Drawer 列、SplatSim 97.5 vs 96.25、ASAP 150 vs 173。笔记里都注明了，引用时看笔记里标的是哪张表。
4. **venue 不能从项目主页或 GitHub README 直接抄** —— 有几篇 index.json 初稿里的 venue 是错的或只是 workshop。以笔记头部（已核实）为准。
5. **arXiv 抓取要串行** —— 并发抓会被限流。`fetch-all.sh` 故意串行，全量约 20 分钟。

---

## 给下一个窗口的第一句话

> 读 `AGENTS.md` 和 `handoff.md`，然后读 `related_work/README.md` 的 §3 阶段 0 和 §5 ①。在 `复现/ppo/` 下用 Gymnasium MuJoCo 复现 PPO 的三组对比（不加约束 / clip / adaptive KL，Hopper + HalfCheetah，各 3 seeds，超参照 `related_work/04_rl/ppo/notes.md` 里的 Table 3），写 `复现/ppo/复现报告.md`。动手前先说要怎么改、为什么。不要改 `related_work/` 下的笔记，除非发现与 `source/` 不符的错误。
