# 复现

> 按 research-workflow 的原则：**先复现关键 baseline，再提假设。**
> 论文只报平均值和成功配置，卡在哪、哪个 setting 崩了，只有自己跑一遍才看得见。

## 复现队列

排序与理由、算力、最小实验见 [related_work/README.md §5](../related_work/README.md#5-复现队列)。摘要：

| 顺序 | baseline | 主题 | 目录 | 本机 macOS |
|---|---|---|---|---|
| ① | PPO | 04 RL | `复现/ppo/` | 可以（CPU） |
| ② | PAD | 05 交叉 | `复现/pad/` | 官方依赖 CUDA，CPU / MPS 未验证 |
| ③ | TD-MPC2 | 01 World Model | `复现/tdmpc2/` | 需要 NVIDIA GPU |
| ④ | Robotic World Model | 05 交叉 | `复现/robotic-world-model/` | 不行（Isaac Lab） |
| ⑤ | SimDist | 05 交叉 | `复现/simdist/` | 不行（IsaacLab） |
| ⑥ | SIMPLER | 03 Real2Sim2Real | `复现/simpler/` | 不行（Linux + RTX） |

## 每个 `复现/<方法>/` 放什么

- `复现报告.md`：环境与代码 commit、运行命令、**论文数字 / 自己的数字**对照表、偏差及可能原因、踩到的坑
- 配置文件
- 日志（大文件、checkpoint 不提交）

评测至少 3 个 seed。指标、固定失败样例的方式照 [experiment/evaluation.md](../experiment/evaluation.md)。
