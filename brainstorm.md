# Brainstorm

> 问题和假设是怎么聊出来的。**不是 Agent 直接给的，是来回聊出来的。**
> **倒序写，最新的在最上面。**被自己否掉的想法也留着——那是写 motivation 的素材。

---

## 2026-09-28 · 文献调研后列出的候选问题（还没讨论）

这些问题来自 54 篇笔记的"局限"一节和五个主题 README 的"开放问题"一节，由调研 agent 整理，**不是结论，也还没经过讨论**。每个问题的"哪篇论文的哪个局限 + 哪条线的哪个工具可能解决它 + 最小切入实验"见 [related_work/README.md §6](related_work/README.md#6-交叉处的开放问题)。

按项目原则，**先复现再展开**：下面每条在复现相关 baseline 之前只记不聊。

---

## 待想清楚

- ⬜ Q1. 部署时用 world model 自带的自监督损失做 TTT，能否修正 model error？（01 × 02 × 05；依赖复现 ② PAD、③ TD-MPC2）
- ⬜ Q2. 部署时该更新哪一部分？没有 reward 时，怎样发现适应在变差、何时回滚？（02 × 03 × 05）
- ⬜ Q3. learned simulator 的 reality gap 怎么量化：model exploitation 诊断（01 × 03 × 04 × 05；依赖复现 ④ RWM）
- ⬜ Q4. world model 当 policy evaluator：在 MMRV 口径下能否达到 SIMPLER？被拿来训练之后还成立吗？（03 × 04 × 05；依赖复现 ⑥ SIMPLER）
- ⬜ Q5. 调参数、学残差，还是学整个 dynamics？（01 × 03 × 05）
- ⬜ Q6. 显式的梯度式适应与隐式的 in-context 适应，谁在 sim-to-real 上更好？（02 × 03 × 05）
- ⬜ Q7. fast weights 能否作为 world model 的长程记忆？（01 × 02 × 05）
- ⬜ Q8. 生成式还是表征空间、像素还是 latent：固定 agent、只换 world model 的受控实验（01 × 04 × 05）
- ⬜ Q9. learned reward 与伪奖励在 world model 里有多可靠？（02 × 04 × 05）
- ⬜ Q10. GRPO 靠 world model 提供 reset：binary reward 和基座强弱怎样影响结果？（04 × 05）
- ⬜ Q11. world model 的价值来自表征，还是来自 simulator？（01 × 04 × 05）
- ⬜ Q12. world model 与 policy 要不要交替迭代？和 SimOpt 式的闭环校准是什么关系？（03 × 05）
