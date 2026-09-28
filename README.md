# wm-ttt-real2sim-rl

系统学习 **World Model × Test-Time Training × Real-to-Sim-to-Real × Reinforcement Learning** 四条线及其交汇点的研究项目。项目结构按 [skJack/research-workflow](https://github.com/skJack/research-workflow) 组织。

## 现在有什么

- **54 篇精读笔记**（50 篇论文 + 4 篇综述），分五个主题。每篇笔记的数字和公式都已对照 LaTeX 源码逐条核查过。
- **五个主题综述**：每个包含脉络、技术分类、机制对比表、学习路线、开放问题、扩展阅读和复现推荐。
- **总入口 [related_work/README.md](related_work/README.md)**，内容包括：
  - 统一视角
  - 概念地图（mermaid）
  - 6 个阶段的学习路线（约 21–25 周，按每周 10 小时估算）
  - 总索引
  - 复现队列
  - 12 个交叉开放问题

| 主题 | 篇数 | 入口 |
|---|---|---|
| 01 World Model | 11 | [related_work/01_world_model/](related_work/01_world_model/README.md) |
| 02 Test-Time Training | 12 | [related_work/02_test_time_training/](related_work/02_test_time_training/README.md) |
| 03 Real-to-Sim-to-Real | 12 | [related_work/03_real2sim2real/](related_work/03_real2sim2real/README.md) |
| 04 Reinforcement Learning | 11 | [related_work/04_rl/](related_work/04_rl/README.md) |
| 05 交叉 | 8 | [related_work/05_intersections/](related_work/05_intersections/README.md) |

## 从哪开始

1. [AGENTS.md](AGENTS.md)：项目总纲和硬规矩。每个新的 Agent 窗口第一个读它。
2. [handoff.md](handoff.md)：现在做到哪、下一步做什么。
3. [related_work/README.md](related_work/README.md)：从 §1 读到 §3，然后按学习路线的阶段 0 开始。

## 目录

```
AGENTS.md              项目总纲：在做什么、硬规矩
handoff.md             进度交接
brainstorm.md          问题与假设的讨论记录（倒序）
related_work/          文献：<topic>/<slug>/{notes.md, paper.pdf, source/}
  README.md            总入口：路线 / 索引 / 复现队列 / 开放问题
  index.json           全部论文的元数据
复现/                   baseline 复现，每个方法一份复现报告
experiment/            results.md（每版一行）+ evaluation.md（标准化测评）
DataSet/ + dataset.md  数据与处理说明
scripts/               fetch-paper.sh（抓单篇）· fetch-all.sh（按 index.json 全量抓取）
```

## 安装与准备

论文的 `paper.pdf` 和 LaTeX `source/` **不进版本库**，原因有两个：一是体积约 1.2 GB，二是版权。clone 之后执行：

```bash
bash scripts/fetch-all.sh
```

这一步从 arXiv 串行抓取全部 PDF 和源码，并自动清洗源码（隔离旧稿、剥掉注释），大约需要 20 分钟。依赖 `curl`、`perl`、`python3`。另外建议安装 `pdftotext`（poppler），用来快速浏览论文。

加一篇新论文：

```bash
bash scripts/fetch-paper.sh <arxiv-id> <topic>/<slug>
```

然后按 [related_work/README.md §7](related_work/README.md#7-如何加新论文) 写笔记并登记到 `index.json`。

## 工作节奏

```
抓文献 → 复现关键 baseline → 提假设 → 跑实验 → （循环）
```

目前处于 **复现** 阶段的起点，复现队列见 [复现/README.md](复现/README.md)。
