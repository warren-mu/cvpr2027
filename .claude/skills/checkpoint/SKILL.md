---
name: checkpoint
description: 防跑偏检查：对照研究合同审视最近的工作、证据强度和剩余时间，发现偏航时让人类选择。用户输入 /checkpoint，或每隔约 4 小时、每次关卡前使用。
---

# /checkpoint —— 防止在错误的路上越走越远

## 读取
- `docs/RESEARCH_CONTRACT.md`、`STATUS.md`、`CLAIMS.md`
- `git log --oneline -20`（如果是 git 仓库）
- `tail -n 40 logs/commands.jsonl`、`ls -t results/ | head`

## 回答 5 个问题，写入 `reviews/CHECKPOINT_<YYYYMMDD-HHMM>.md`
1. 过去几小时做的每件事分别服务于合同中的哪条主张？对应不上的列为"偏航"。
2. 每条主张当前的证据路径与强度（强 / 弱 / 无）。
3. 剩余时间 vs 剩余任务估时（按截止 2026-11-16 23:59 AoE 计）。是否需要砍范围？砍什么？
4. 是否出现跑偏信号：
   - 同一实验反复调参 3 次以上才"赢" baseline
   - baseline 数字明显低于原论文
   - 新主张没有对应实验
   - 正文叙述与 results 的数字方向不一致
   - 在测试集上做了选择
5. 最大的单点风险是什么，最便宜的对冲办法是什么。

## 决策
- 一切正常：一句话汇报即可。
- 偏航 / 需砍范围 / 出现跑偏信号：列出 2–3 个选项（含"降级为 Findings / workshop"或"换更窄的主张"），用 AskUserQuestion 让人类选，选完再改 STATUS.md。
