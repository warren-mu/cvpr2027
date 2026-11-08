---
name: kickoff
description: 开题：在 2 小时内产出一份经人类签字的研究合同（选题、主张、baseline、数据集、指标、kill 条件）。用户输入 /kickoff 时使用。
disable-model-invocation: true
---

# /kickoff —— 选题与研究合同

目标：把"想蹭热点"变成一份**可证伪、5 天内能做完**的研究合同。全程不写论文正文。

## 步骤
1. 用 AskUserQuestion 问用户（最多 4 题）：
   - 已有资产：可复用的代码库 / 数据 / 初步结果 / 之前没投出去的工作
   - 算力：GPU 型号 × 数量 × 可用小时
   - 想靠近的热点方向（参考 `docs/RESEARCH_NOTES.md` 第 2 节）
   - 作者中是否有人已在 CVPR/ICCV/ECCV/NeurIPS/ICML/ICLR/AAAI 等发表过主会论文（决定 reciprocal reviewer 配额：没有的话每人最多投 1 篇）
2. 读 `docs/RESEARCH_NOTES.md`，结合资产提出 **2–3 个候选题目**，每个给出：
   - 一句话主张（可被实验证伪）
   - 最小可行实验 MVE：哪个 baseline、哪个数据集、哪个指标、预计 GPU·小时
   - 5 天内的 kill 条件（例如"Day2 晚上主指标仍不超过 baseline 0.5 个点就降级/换角度"）
   - 主要风险（被同期工作撞车、复现失败、算力不够）
3. 同时派 `scout` 子 agent 对每个候选做近 12 个月文献扫描，写 `lit/novelty_risk.md`。
4. 用户选定后，填写 `docs/RESEARCH_CONTRACT.md`（修改会弹窗请用户确认）。
5. 在 `STATUS.md` 建 Day0–Day2 的任务（带 [EXP]/[LIT]/[WRITE] 标签和完成标准）。
6. **提醒用户人工完成**（agent 不能代办）：
   - 2026-11-10 23:59 AoE 前在 OpenReview 注册标题、摘要、全部作者
   - 所有作者 OpenReview 档案完整；用非机构邮箱新建的档案审核可能要两周
   - 注册后不能再增删作者
