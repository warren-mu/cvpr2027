# CVPR Agent Kit —— 多个 Claude Code 窗口互相监督写论文

> **先读 `docs/PLAYBOOK.md`**：8 个窗口的分工、开场 prompt、按天作战表、算力编排、质量闸与故障预案都在里面。

一套放进论文仓库就能用的 Claude Code 配置：7 个角色（实验员分两个窗口）、文件所有权隔离、"数字只能来自日志"、引用自动校验、独立评审轮次、人类关卡。目的不是让 AI 替你写论文，而是让多个 AI 窗口**互相卡住对方的幻觉和跑偏**，把你的注意力留给真正需要判断的地方。

```
                         ┌──────────────── 你（人类）────────────────┐
                         │  G1 合同签字  G2 主实验成立  G3 初稿  G4 终检 │
                         └───────────────────┬──────────────────────┘
                                             │ 关卡处必须你拍板
                                   ┌─────────▼─────────┐
                                   │  lead（PI/总控）    │  派任务、分诊评审、不碰数字
                                   └──┬──────┬──────┬───┘
                     STATUS.md 任务板 │      │      │  CLAIMS.md 主张账本
           ┌──────────────────────────┘      │      └───────────────────────────┐
   ┌───────▼───────┐   results/*.json ┌──────▼──────┐  \res{key}      ┌──────────▼─────────┐
   │ experimenter  │ ───────────────▶ │ export_     │ ──────────────▶ │ writer             │
   │ run_logged.py │  (带命令/commit/  │ numbers.py  │ numbers.tex     │ 只能用 \res{} 写数字 │
   └───────────────┘   日志/exit code) └─────────────┘                 └──────────┬─────────┘
   ┌───────────────┐   verify_bib.py  ┌─────────────┐                            │ paper/
   │ scout         │ ───────────────▶ │ refs.bib    │ ───────────────────────────┤
   └───────────────┘   (arXiv/DBLP/S2/Crossref)     │                            ▼
                                                     │     ┌─────────────────────────────────┐
                                                     └────▶│ 独立评审（互不可见，每轮新上下文） │
                                                           │ skeptic-reviewer │ auditor │ Codex │
                                                           └─────────────────────────────────┘
   hooks：role_guard（谁能改哪些文件）· stop_gate（写完自动跑检查）· task_gate（没证据不准"完成"）
```

## 七道闸：各自防什么

| 机制 | 防什么 | 实现 |
|---|---|---|
| 文件所有权 | writer 改实验、experimenter 改论文、谁都能"顺手"改结果 | `.claude/hooks/role_guard.py`（PreToolUse） |
| 数字溯源 | 编造 / 记错 / 抄错数字 | `run_logged.py` → `results/*.json` → `export_numbers.py` → `\res{key}`；表格里手写小数直接报错 |
| 引用校验 | 幻觉引用（CVPR 2027 明文 desk reject） | `verify_bib.py` 查 arXiv/DBLP/Semantic Scholar/Crossref；未校验的引用通不过检查 |
| 主张账本 | 夸大其词、intro 承诺了实验没做的东西 | `CLAIMS.md`，只有 auditor 能标 verified（hook 强制） |
| 自动门禁 | 写完就走、把问题留给明天 | Stop hook 跑 `check_paper.py --fast`；TaskCompleted hook 要求证据 |
| 独立 + 跨模型评审 | 同一个模型自己审自己的相关性盲点 | `/review-round`：新上下文的 skeptic + auditor + Codex，互不可见 |
| 人类关卡 + 防跑偏 | 在错误方向上越走越远 | `/checkpoint`、G1–G4、研究合同修改必须人类点确认 |

## 快速开始

**前置**：Claude Code（agent teams 需 v2.1.32+）、python3、tmux、LaTeX（latexmk）、pdftotext（可选，用于查页数）。

1. 把本目录内容复制到你的论文/代码仓库根目录（已有 `.claude/` 就合并），建议 `git init` 并提交一次。
2. 放入 CVPR 2027 官方 LaTeX 模板，见 `paper/README.md`。
3. （强烈推荐，你本来就在用 Codex）配置 Codex 作为跨模型审稿人：
   ```bash
   npm i -g @openai/codex          # 已安装可跳过
   claude mcp add codex -s user -- codex mcp-server
   ```
4. 第一次在该目录运行 `claude` 时接受"信任此文件夹"，否则项目级 hooks 不会生效。用 `/hooks` 确认 5 个 hook 已加载。

### 模式 A：多窗口（推荐，最稳）
```bash
bash launch_tmux.sh                              # 核心 5 窗：lead exp-env exp-train writer analyst
GPUS=0,1,2,3,4,5,6,7 bash launch_tmux.sh full     # 全 8 窗 + 本机 GPU 队列 worker
```
- 每个窗口通过 `scripts/claude_as.sh <角色>` 启动：保留 Claude Code 默认能力，追加角色说明，并设置 `CVPR_ROLE` 供 hook 识别。
- 在 **lead** 窗口输入 `/kickoff` 开题。之后在各窗口说"执行 STATUS.md 里分给你的任务"。
- 每轮评审前，在 **reviewer** 窗口先 `/clear`，保证审稿人是"新眼睛"；或直接在 lead 用 `/review-round`（会起新上下文的子 agent）。
- 想集中查看所有会话状态：另开终端运行 `claude agents`。

### 模式 B：原生 agent teams（实验功能，更自动）
```bash
bash launch_team.sh          # in-process，任何终端
bash launch_team.sh panes    # 每个 teammate 一个 tmux 分屏
```
在 lead 会话里说：
> 创建一个 agent team 推进这篇论文。按 .claude/agents 的定义 spawn 4 个 teammates：exp 用 experimenter、writer 用 writer、reviewer 用 skeptic-reviewer、auditor 用 auditor。把 STATUS.md 的任务导入共享任务列表，保留 [EXP]/[WRITE] 标签，[EXP] 任务描述里写 `evidence: <路径>`。reviewer 与 auditor 互不查看对方意见。等 teammates 完成后再汇总，不要替他们做任务。

## 技能（slash commands）
| 命令 | 何时用 |
|---|---|
| `/kickoff` | D0 开题：候选题目 + 新颖性扫描 → 研究合同 |
| `/checkpoint` | 每 4 小时 / 每个关卡前：对照合同查偏航、证据强度、剩余时间 |
| `/review-round` | 初稿后：独立三路评审 + 分诊成任务 |
| `/gate` | 提交前终检，逐项红绿清单 |

## 常用脚本
```bash
python3 scripts/gpu_queue.py submit --name main_nav_ours_s0 --gpus 8 -- <命令>  # 进 GPU 队列（自动留痕）
python3 scripts/gpu_queue.py worker --gpus all   # 每台机器一个 worker；status / cancel / retry
python3 scripts/run_logged.py --name exp01_main -- python src/train.py ...   # 直接跑（自动留痕）
python3 scripts/summarize_results.py --prefix main_ --compare A B          # 多种子均值/CI/Welch 比较
python3 scripts/export_numbers.py              # 结果 → \res 宏
python3 scripts/verify_bib.py --only-cited     # 校验被引用的文献（需联网）
python3 scripts/check_paper.py --fast|--final  # 门禁检查
bash scripts/codex_review.sh 1                 # 没配 Codex MCP 时的跨模型审稿备用方案
```

## 从 Codex 迁过来的对照
| Codex | Claude Code |
|---|---|
| `AGENTS.md` | `CLAUDE.md`（本仓库已写好；Claude Code 也会读取 AGENTS.md） |
| `codex exec` | `claude -p` |
| `~/.codex/config.toml` 的 MCP | `claude mcp add ...` 或 `.mcp.json`（样例见 `.mcp.json.example`） |
| skills | 同样的 SKILL.md 格式；ARIS、academic-research-skills 等两边都能用 |

## 已知限制（请务必读）
- role_guard 拦截 Edit/Write 和常见的 shell 写法（重定向、tee、sed -i），但拦不住所有 Bash 写文件方式（例如 `python -c` 直接写）。它是护栏不是保险箱；第二道防线是 run_logged 的日志、`metrics_sha256` 和 auditor 的逐条溯源。
- 模式 B 的 tmux 分屏 teammate 不一定把 `agent_type` 传给 hook（官方文档只明确了 in-process teammate），识别不到时按 lead 处理（宁可放宽也不误拦）。要最严格的隔离，用模式 A 或 in-process。
- `verify_bib.py` 在没有外网的环境下用模拟数据测过解析和判定逻辑；真实 API 调用请先在几条已知正确的 bib 上试跑。断网时它只会报 `error`（稍后重跑），不会把真论文误判为不存在。
- `codex_review.sh` 里的 `codex exec` 参数可能随版本变化，以 `codex exec --help` 为准。
- agent teams 是实验功能：`/resume` 不会恢复 in-process teammates，一个会话只能有一个 team，token 消耗随人数线性增长。
- 这套工具能显著减少幻觉和跑偏，但**不能替代你对每个数字和每句主张的最终判断**。CVPR 2027 规定作者对 LLM 产生的内容负全部责任。

## 文件清单
```
CLAUDE.md                     项目宪法（所有窗口共享）
STATUS.md / CLAIMS.md         共享任务板 / 主张账本
docs/RESEARCH_CONTRACT.md     研究合同模板（G1 后修改需人类确认）
docs/RESEARCH_NOTES.md        CVPR 2027 规则 + 热点地图 + 工具生态（含来源）
docs/5DAY_PLAN.md             通用 5 天计划（本项目以 PLAYBOOK 第 4 节为准）
docs/PLAYBOOK.md              作战手册（本项目专用）
docs/METRICS_SPEC.md          预注册指标定义（修改需人类确认）
experiments/REGISTRY.md       预注册实验登记表（修改需人类确认）
analysis/plotlib.py           出图并写溯源 sidecar
.claude/agents/               pi-lead, scout, experimenter, analyst, writer, skeptic-reviewer, auditor
.claude/hooks/                role_guard, stop_gate, task_gate, session_context, log_bash
.claude/skills/               kickoff, checkpoint, review-round, gate
.claude/settings.json         hooks 与常用命令免确认
scripts/                      run_logged, gpu_queue, summarize_results, export_numbers, verify_bib, check_paper, claude_as, codex_review
paper/sections/llm_usage.tex  LLM Usage Statement 模板（按实际情况修改）
launch_tmux.sh / launch_team.sh
```
