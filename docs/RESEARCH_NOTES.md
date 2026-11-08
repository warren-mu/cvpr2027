# 调研笔记（2026-11-07 整理，供 lead / scout 开题时参考）

> 本文件是对公开资料的概括，不是原文摘录。具体数字以链接中的原始来源为准。

## 1. CVPR 2027 规则要点（官方 Call for Papers / Author Guidelines）
- 论文注册（OpenReview 上建投稿、填标题摘要、加齐作者）：2026-11-10 AoE；正文：11-16 AoE；补充材料：11-23 AoE。日期不延期。
- 注册截止后不能增删作者；所有作者都要在注册截止前有完整 OpenReview 档案。用非机构邮箱新建的档案需要人工审核，最长可能两周。
- 正文 8 页；参考文献和 LLM Usage Statement 不计页数。
- **必须有 "LLM Usage Statement" 小节**（放在正文后、参考文献前）：写明用了哪些工具、影响了论文/研究的哪些部分、使用程度；缺失直接 desk reject。
- 引用不存在的文献、明显编造的事实、提示注入（隐藏给审稿人/AI 的指令）都列入 desk reject 理由。
- 新的 reciprocal reviewer 配额：如果一篇论文的所有作者都没有在指定主会/期刊（CVPR/ICCV/ECCV/WACV/ACM MM/TPAMI/IJCV、NeurIPS/ICML/ICLR/AAAI/IJCAI/JMLR、ICRA/IROS/RSS/CoRL、SIGGRAPH 等）发表过，则每位作者最多参与 1 篇这样的投稿。资格只看 OpenReview 档案里的论文。
- 2026-09-15 之后公开的工作算同期工作：不会因此被拒，但要引用讨论。
- 有 Findings 轨道（CVPR 2026 试行）：技术扎实但增量较小的论文由 AC 推荐。
- 审稿过程会使用会议批准的 LLM 工具，作者不能选择退出。

## 2. 热点地图（依据 CVPR 2026 接收论文统计 + CVPR 2027 CFP 新增主题）

CVPR 2026：16,092 篇投稿，约 4,090 篇主会接收（约 25%），另有 941 篇 Findings。

**升温方向**
| 方向 | 信号 | 5 天内相对可行的切入方式（建议，非来源结论） |
|---|---|---|
| 多模态大模型推理 + RL 后训练 | RL 相关论文占比从约 1% 跃升到 6.6%；VLM/MLLM 份额翻倍；CFP 新增 "Reasoning & Reinforcement Learning" | 在 2–7B 开源 VLM 上做针对某类感知/空间推理的奖励设计或 test-time 方法；需要提前确认算力 |
| 视频生成 / 世界模型 | 高关注样本中份额约 2.3 倍增长；CFP 新增 "Physical AI: World Models, VLA, Planning, Simulation"；ECCV 2026 多个 workshop 关注世界模型的应用导向评测 | 开源视频扩散模型的免训练加速 / 一致性改进；世界模型的诊断性评测协议 |
| 具身智能 / VLA / 自动驾驶 | 高关注样本中份额从 2.9% 到 6.2%；小模型、低延迟推理受关注 | 高效 VLA 推理、小模型蒸馏；注意真机实验 5 天内基本不可能 |
| 3D/4D 重建与生成 | 3DGS 出现在 5.1% 论文中，NeRF 降到约 1%；前馈式重建（VGGT → D4RT 获 2026 最佳论文）、线性复杂度重建、原生 3D 结构化隐变量生成 | 把前馈重建模型适配到新设定（新传感器/新场景）并给出系统分析 |
| 统一理解 + 生成模型 | ICML 2026、ECCV 2026 workshop 的持续主题 | 轻量的分析/改进型工作，避免和大厂拼预训练 |
| 扩散模型基础 | 直接预测干净图像（JiT）、MeanFlow 系列、扩散模型的奖励后训练 | 小规模 ImageNet-256 可做的理论/训练技巧型工作 |
| Agentic AI、生成式 AI 安全与真实性 | 2027 CFP 新增主题；deepfake 取证在 Findings 中占比高 | 检测/取证类工作要注意新颖性，否则容易落到 Findings |

**降温方向**：经典检测/分割/跟踪、NeRF、CLIP、点云、零样本/小样本框架叙事。

**重要提醒**
- 没有证据表明"热门方向"本身提高录用率：热门方向审稿人多、关注度高，但竞争者也最多，撞车风险最大。
- 5 天最现实的策略：热点 × 你已有的代码/数据/初步结果。从零开始的热门题目风险极高。
- 2026 年的趋势是工业界数据与算力门槛升高（如 20 亿图像预训练、64 卡 sim-to-real），学术小组应避开拼规模的题目。

## 3. 工具生态（安装前务必阅读 SKILL.md 和脚本，它们会在本机执行命令）
- **ARIS（wanshuiyin/Auto-claude-code-research-in-sleep）**：纯 Markdown 技能包，约 80 个技能，覆盖文献检索、找 idea、查新、实验、写作、rebuttal；核心卖点是 Claude Code 执行 + Codex 跨模型审稿。Codex MCP：`claude mcp add codex -s user -- codex mcp-server`
- **academic-research-skills（Imbad0202）**：研究→写作→审稿→修改→定稿的流水线，含多 agent 写作系统、审稿模拟、可选跨模型"魔鬼代言人"。安装：`/plugin marketplace add Imbad0202/academic-research-skills` 然后 `/plugin install academic-research-skills`
- **Orchestra-Research/AI-Research-SKILLs → ml-paper-writing**：顶会写作指南、LaTeX 模板、引用校验脚本（面向 NeurIPS/ICML/ICLR 等，用于 CVPR 需调整格式）。安装：`/plugin marketplace add orchestra-research/AI-research-SKILLs` 然后 `/plugin install ml-paper-writing@ai-research-skills`
- **引用校验 MCP**：refcheck-mcp（Crossref / Semantic Scholar / arXiv，`pip install refcheck-mcp`）、harcx-mcp（Semantic Scholar / DBLP）。本仓库自带 `scripts/verify_bib.py`，无需额外依赖。
- **Claude Code 原生能力**：subagents（`.claude/agents/*.md`）、agent teams（实验功能，`CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1`）、hooks（PreToolUse / Stop / TaskCompleted 等）、`claude agents`（agent view，集中监控多个会话）。

## 4. 前车之鉴
- ICLR 2026 对所有确认含幻觉引用的投稿做了 desk reject；GPTZero 在 NeurIPS 2025 已发表论文里也找到了上百条幻觉引用；ACL 2026 在 camera-ready 检查中标记了 100 多篇引用不存在文献的已录用论文。

## 来源
- https://cvpr.thecvf.com/Conferences/2027/CallForPapers
- https://cvpr.thecvf.com/Conferences/2027/AuthorGuidelines
- https://genawass.github.io/posts/cvpr2026.html
- https://www.bohrium.com/en/blog/research-notes/cvpr-2026-accepted-papers-highlights/
- https://code.claude.com/docs/en/agent-teams
- https://code.claude.com/docs/en/sub-agents
- https://code.claude.com/docs/en/hooks
- https://www.sourcepulse.org/projects/26134822 （ARIS）
- https://tosea.ai/blog/academic-research-skills-claude-code-suite-guide-2026
- https://claudemarketplaces.com/plugins/orchestra-research-ai-research-skills/ml-paper-writing
- https://pypi.org/project/refcheck-mcp
- https://blog.iclr.cc/2026/03/31/a-retrospective-on-the
- https://gptzero.me/news/neurips/
