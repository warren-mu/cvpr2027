---
name: review-round
description: 发起一轮独立评审：skeptic-reviewer + auditor 并行（互不可见），可选 Codex 跨模型审稿，然后由 lead 分诊成任务。用户输入 /review-round 时使用。
disable-model-invocation: true
---

# /review-round —— 一轮独立、多模型评审

独立性是这套流程的全部意义：审稿人不知道作者"觉得哪里好"，也看不到彼此的意见。

## 步骤
1. 轮次 N = `reviews/` 下已有 `ROUND_<k>_*` 的最大 k + 1。
2. 预处理（结果只给 auditor 参考）：
   ```bash
   python3 scripts/export_numbers.py
   python3 scripts/check_paper.py --fast || true
   (cd paper && latexmk -pdf -interaction=nonstopmode main.tex) || true
   ```
3. **在同一条消息里**并行启动两个子 agent：
   - `skeptic-reviewer`，提示词只写："第 N 轮评审。阅读 paper/（或 paper/main.pdf）、results/*.json、CLAIMS.md、lit/related_work.md，按你的格式写入 reviews/ROUND_N_skeptic.md。不要读 reviews/ 下的其他文件。"
   - `auditor`，提示词只写："第 N 轮审计，写入 reviews/ROUND_N_audit.md。"
   - 不要在提示词里加入你对论文质量的任何判断。
4. 跨模型审稿（强烈建议，可降低同一模型的相关性错误）：
   - 若 `mcp__codex__codex` 工具可用：先写 `reviews/REVIEW_BRIEF.md`（只列文件路径 + CVPR 评审表格式，不含结论），把"请阅读 <绝对路径>/reviews/REVIEW_BRIEF.md 并按其要求评审"发给 Codex，原文保存为 `reviews/ROUND_N_codex.md`。
   - 若不可用：提示用户在终端运行 `bash scripts/codex_review.sh N`。
5. 全部到齐后写 `reviews/ROUND_N_triage.md`：
   - 每条意见：`ACCEPT / REJECT / DEFER` + 一句理由 + 对应任务 ID
   - 两个以上审稿来源同时指出的问题，优先级最高
   - REJECT 的理由如果是"时间不够"，在论文 Limitations 中诚实说明
6. 把 ACCEPT 项写进 `STATUS.md` 任务板；向用户汇报：各来源评分、Top-3 致命问题、建议下一步、是否到达关卡（G3/G4）。
