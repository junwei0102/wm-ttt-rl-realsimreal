# wm-ttt-real2sim-rl

系统学习四条线——**World Model、Test-Time Training、Real-to-Sim-to-Real、Reinforcement Learning**——的核心问题、代表方法与它们的交汇点。

**核心主张：先把四条线各自"在解决什么问题、靠什么机制解决、在哪里失效"弄清楚，再去找它们的交叉点（如：world model 当 simulator 训 RL policy、TTT 让 world model / policy 在部署时适应、real2sim2real 用重建的数字孪生缩小 sim-to-real gap）。每篇笔记都必须能回答"它的机制是什么、证据是什么、局限是什么"，不写没有出处的结论。**

目标：系统学习 / 为后续选题打基础（会议与 deadline 待定，确定后改这一行）

---

## 你（Agent）每次进来先读什么

1. **本文件** —— 知道我们在做什么、每个目录是干什么的
2. **[handoff.md](handoff.md)** —— 上一个窗口做到哪了、下一步是什么
3. 要读文献先读 [related_work/README.md](related_work/README.md)（总索引 + 学习路线）
4. 要动实验就再读 [experiment/results.md](experiment/results.md) 和 [experiment/evaluation.md](experiment/evaluation.md)

**不要读完整个仓库再开始。** 上下文很贵，按需读。

---

## 目录

| 路径 | 是什么 | 什么时候看 |
|---|---|---|
| `related_work/` | 相关文献，按主题分子目录；每篇 `paper.pdf` + `source/` + `notes.md` | 要引用、要对比方法时 |
| `related_work/<主题>/README.md` | 该主题的综述：脉络、分类、阅读顺序、开放问题 | 入门某条线时先读 |
| `复现/` | 重要 baseline 的复现，每个带一份复现报告 | 要确认 baseline 数字时 |
| `brainstorm.md` | 问题和假设是怎么聊出来的 | 想不通为什么这么设计时 |
| `experiment/` | 实验代码 + 每一版结果 | 跑实验 |
| `DataSet/` | 数据集，处理说明见 `dataset.md` | 要动数据时 |
| `scripts/fetch-paper.sh` | 抓单篇 arXiv 论文（PDF + 清洗过的 LaTeX） | 加新文献时 |
| `scripts/fetch-all.sh` | 按 `related_work/index.json` 全量抓取（PDF / source 不进 git） | 新 clone 之后、`paper.pdf` 缺失时 |
| `handoff.md` | 进度交接 | **每次都读** |

---

## 读论文的规矩

`related_work/` 里每篇都是 `paper.pdf` + `source/`（LaTeX 源码）+ `notes.md`（阅读笔记），按要问什么选：

- 泛读（讲了什么、什么方法）→ `paper.pdf`
- 细查（表里的数、公式、引用）→ `source/*.tex`
- 看图 → 只能 `paper.pdf`，图不在源码里
- 快速回顾 → `notes.md`（笔记里的数字都应能在 `source/` 或 PDF 里找到出处）

用 `scripts/fetch-paper.sh` 抓的源码已清洗过（旧稿在 `source/.stale/`，注释已剥、原文存 `.orig`）。
手动下的记得自己清一遍。宏多的论文先读 `commands.tex` / `macros.tex` 这类文件。

`paper.pdf` 和 `source/` 不在版本库里；新 clone 后找不到就先跑 `bash scripts/fetch-all.sh`。

加新论文：

```bash
bash scripts/fetch-paper.sh <arxiv-id> <主题目录>/<slug>
```

---

## 硬规矩

1. **不要自己发明术语。**解释你在做什么的时候用论文里已有的词
   （world model, latent dynamics, RSSM, model-based RL, imagination / latent rollout, MPC, planning;
   test-time training (TTT), test-time adaptation (TTA), self-supervised auxiliary task, fast weights, inner loop / outer loop;
   sim-to-real gap, domain randomization, system identification, real-to-sim, digital twin, Gaussian splatting, 3D reconstruction;
   MDP, policy gradient, actor-critic, PPO, SAC, offline RL, reward model, RLHF, GRPO），别造新词。
2. **改实验之前先说你要改什么、为什么。**不要直接动手跑。
3. **每版实验都要有名字。**命名规则见 `experiment/results.md`，不许出现 `test2` `final_v2_new`。
4. **跑完必须走标准化测评。**见 `experiment/evaluation.md`，不要临时挑指标。
5. **窗口快满了跟我说，我让你更新 handoff。**不要硬撑到被截断。
