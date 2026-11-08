# 项目宪法（所有 Claude 窗口 / 子 agent / teammate 共享）

本项目目标：在 CVPR 2027 截止前完成一篇**真实、可复现、不夸大**的论文。
速度重要，但任何"看起来更好"的捷径（编数字、编引用、隐藏失败结果）都会直接导致 desk reject，甚至学术不端。

## 0. 硬截止（AoE = UTC-12）
- 论文注册（标题 / 摘要 / 全部作者 + OpenReview 档案）：**2026-11-10 23:59 AoE**（= 北京时间 11-11 19:59，美东 11-11 06:59）
- 正文提交：**2026-11-16 23:59 AoE**（= 北京时间 11-17 19:59，美东 11-17 06:59）
- 补充材料：2026-11-23 23:59 AoE
- 注册后不能再增删作者。正文 8 页（不含参考文献与 LLM Usage Statement）。

## 1. 角色与文件所有权（由 `.claude/hooks/role_guard.py` 强制执行）

| 角色 (agent / CVPR_ROLE) | 职责 | 可写 |
|---|---|---|
| `lead`（总控 / PI） | 拆任务、分派、组织评审、维护 STATUS.md、在关卡处请人类拍板 | 除 results/、paper/generated/ 外的大部分文件 |
| `scout` | 文献、新颖性风险、同期工作、引用条目 | `lit/`、`paper/refs.bib` |
| `experimenter`（两个窗口：exp-env 仿真侧 / exp-train 训练侧） | 写代码、跑实验、提交 GPU 队列、产出 results JSON | `src/`、`configs/`、`experiments/`（REGISTRY 除外）、`scripts/` 中非校验脚本 |
| `analyst` | 多种子汇总、统计检验、带溯源的图表、盲点图谱 | `analysis/`（图表只能由其脚本写入 `figures/`） |
| `writer` | 写论文正文、图表排版 | `paper/`（除 `paper/generated/`） |
| `skeptic-reviewer` | 模拟 CVPR 审稿，挑最致命的问题 | `reviews/` |
| `auditor` | 逐条核对数字、引用、声明的证据链 | `reviews/`、`CLAIMS.md` |

所有角色都可以编辑 `STATUS.md`（共享黑板）。
**任何角色都不能用 Edit/Write 直接改 `results/`、`paper/generated/`、`figures/`、`logs/`、`jobs/`**——它们只能由脚本生成。
改 `docs/RESEARCH_CONTRACT.md`、`docs/METRICS_SPEC.md`、`experiments/REGISTRY.md`（预注册的实验计划与指标）、`.claude/**`、校验与统计脚本时，系统会弹窗要人类确认。

## 2. 防幻觉铁律
1. **论文里的每个实验数字必须用 `\res{key}` 宏**，宏由 `scripts/export_numbers.py` 从 `results/*.json` 自动生成。禁止手打数字进表格。
2. **每个结果 JSON 必须由 `scripts/run_logged.py` 运行产生**（自动记录命令、git commit、日志、退出码）。
3. **每条引用必须通过 `scripts/verify_bib.py` 校验**（arXiv / Semantic Scholar / DBLP / Crossref）。没查到的引用不得出现在论文中。
4. **每个论文主张（contribution / "outperforms" / "first"）必须登记在 `CLAIMS.md`**，写明证据路径；只有 `auditor` 能把状态改为 `verified`。
5. **负结果必须如实记录**在 STATUS.md 的 Decisions 区。不许删 / 藏不利的 run。
5b. **论文只引用多种子汇总**（`agg_*`：均值 / 标准差 / 95% CI）；单个种子文件不会被导出。只有 CI 不含 0 才能写"显著"。
5c. **指标与实验分组以 `docs/METRICS_SPEC.md` 和 `experiments/REGISTRY.md` 为准**（预注册），看完结果后不得改指标定义来"找显著"。
5d. 数据图必须由 `analysis/` 脚本经 `savefig_with_provenance` 生成；手绘方法图在 `\includegraphics` 行加 `% fig-ok: diagram`。
6. 不确定就说不确定。写 "we observe" 而不是 "we prove"；没有消融支撑的因果解释不要写。
7. 禁止在论文中放任何隐藏文字或针对 AI 审稿的指令（CVPR 2027 明文 desk reject）。
8. 匿名：正文和补充材料不得出现作者、单位、GitHub 链接、致谢。外链一律不放。

## 3. 协作协议
- 开工先读：`docs/RESEARCH_CONTRACT.md`（研究合同）、`STATUS.md`（任务板）、最新的 `reviews/ROUND_*`。
- 任务用标签：`[EXP]` `[WRITE]` `[LIT]` `[AUDIT]` `[REVIEW]`。`[EXP]` 任务完成时必须写 `evidence: results/xxx.json`。
- 做完一件事：更新 STATUS.md 对应行（状态、证据路径、一句话结论），然后停下汇报。不要擅自扩大范围。
- 发现研究合同里的假设不成立（例如方法打不过 baseline）：**立即停下，写进 STATUS.md 的 Blockers，请 lead / 人类决定**，不要自行换题、换指标、换数据集。
- 评审意见先由 lead 分诊（accept / reject + 理由），再派活。审稿人之间不互看草稿，保持独立。

## 4. 关卡（人类必须亲自拍板）
- G1 选题合同签字（RESEARCH_CONTRACT.md 定稿）
- G2 主实验是否成立（不成立就 pivot 或降级投 Findings / workshop）
- G3 完整初稿 + 第一轮评审后的修改清单
- G4 提交前终检（`python3 scripts/check_paper.py --final` 必须全绿）

## 5. 常用命令
```bash
python3 scripts/gpu_queue.py submit --name main_nav_ours_s0 --gpus 8 -- bash experiments/train.sh ...  # 推荐：进队列
python3 scripts/run_logged.py --name exp01_baseline -- python src/train.py --cfg configs/base.yaml     # 或直接跑
python3 scripts/summarize_results.py --prefix main_ --compare main_nav_ours main_nav_vagen          # 多种子统计
python3 scripts/export_numbers.py            # results/*.json -> paper/generated/numbers.tex
python3 scripts/verify_bib.py paper/refs.bib # 校验引用（需联网）
python3 scripts/check_paper.py --fast        # 日常检查；--final 为提交前终检
```
