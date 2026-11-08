# 作战手册：用多个 Claude 窗口完成 CVPR 2027 投稿

> 项目：**Don't Reward Lucky Guesses / Flip, Stay, or Look**（见 `docs/RESEARCH_CONTRACT.md` v3.1）
> 原则：**并行干活，串行把关**。每个窗口只管一块，所有交接都经过磁盘上的文件和证据；人类只在关卡处做决定。
> 范围纪律：一个仿真器（AI2-THOR）、一个任务族（关系指称导航，以 select 结束）、两层标签（直接 Flip/Stay；Look→Flip/Stay）。核心约束：相同输入不得有不同即时标签。自博弈与学习型编辑器在 G2 前一律不做。

---

## 1. 编制：8 个 Claude 窗口 + Codex + 你

| # | 窗口（启动名） | 角色定义 | 主要产出 | 拥有的目录 | 何时在线 |
|---|---|---|---|---|---|
| 1 | `lead` | pi-lead | 任务板、分诊、关卡汇报 | STATUS.md | 全程 |
| 2 | `scout` | scout | 查新报告、校验过的引用 | lit/、paper/refs.bib | D0–D1 主力，之后按需 |
| 3 | `exp-env` | experimenter（仿真侧） | 任务族改造、世界对构造与有效性检查、统一观察接口、冻结评测集、诊断 | src/envs/ | D0–D3 |
| 4 | `exp-train` | experimenter（训练侧） | 信息集信用、证据后一致性、T0–T4 与 C2PO 式基线、GPU 队列 | src/train/、configs/ | D0–D5 |
| 5 | `analyst` | analyst | 多种子统计、带溯源的图、失败案例图谱 | analysis/ → figures/ | D1 起 |
| 6 | `writer` | writer | 论文正文 | paper/ | D0 起 |
| 7 | `reviewer` | skeptic-reviewer | 独立审稿意见 | reviews/ | D4、D5（每轮先 /clear） |
| 8 | `auditor` | auditor | 证据链审计、CLAIMS 核验 | reviews/、CLAIMS.md | 每晚 + 每轮评审 |
| — | Codex（你本机） | 跨模型审稿 + 独立实现 | 外部审稿、按 METRICS_SPEC 独立重算指标 | crosscheck/ | D2 起 |
| — | **你** | PI 的 PI | 关卡决定、OpenReview、最终通读与提交 | — | 每天 3 次各 15 分钟 |

启动：`GPUS=0,1,2,3,4,5,6,7 bash launch_tmux.sh full`（多节点：每台机器各跑一个 `python3 scripts/gpu_queue.py worker --gpus all`，共享同一个项目目录）。
监控：`gate`（论文门禁）、`queue`（GPU 队列）；另开终端 `claude agents` 集中查看所有会话。

---

## 2. 交接协议（所有窗口都遵守）
```
lead 在 STATUS.md 开任务（标签 + owner + 完成标准）
   → owner 窗口认领（doing）→ 干活；实验一律 gpu_queue / run_logged；图一律 plotlib
   → 写 evidence: <路径> + 一句话结论（done）
   → analyst 汇总进 agg_*，export_numbers 生成 \res 宏
   → writer 引用 \res，在 CLAIMS.md 登记主张（pending）
   → auditor 溯源核验（verified / weaken / failed）
   → lead 在关卡把证据摘要给你拍板
```
- 一个文件只有一个 owner 窗口在写（hook 强制 + 目录分工）。
- 与研究合同冲突的发现写进 Blockers 并停下，由 lead 升级给你。
- 完成一件事就停下汇报，不自行扩大范围。

---

## 3. 各窗口开场 prompt（复制粘贴）

**lead**
> 读 CLAUDE.md、docs/RESEARCH_CONTRACT.md、docs/PLAYBOOK.md、docs/METRICS_SPEC.md、experiments/REGISTRY.md、STATUS.md。今天是 D0（11-07）。按 PLAYBOOK 第 4 节把今天的任务拆到 STATUS.md（每个 ≤2 小时、有可验证完成标准），然后告诉我：今天需要我亲自做什么、下一个去留判断的标准是什么。

**scout**
> 认领 T1。按 lit/related_work_seed.md 的核实标记，把 ⚠ 和 ☑ 的条目逐一打开核实（尤其 LIBERO-Plus 第 6.2 节、AI2-THOR Rearrangement 出处）。补搜："具身/导航 + 反事实场景编辑 + 训练"、"VLM 智能体 + 主动感知/何时该看 + 场景变化"、2026-09-15 之后引用 VAGEN / Q-DEG / Grounding on Request 的工作。输出 lit/novelty_risk.md（风险等级 + 最接近 3 篇的两句话差异）和校验过的 paper/refs.bib。

**exp-env**
> 认领 T2、T3、T4。安装 VAGEN 并冒烟训练 Navigation 一次（记录每步耗时、显存）。然后在 src/envs/：①把 Navigation 改造成关系指称目标，以显式 `select(候选)` 动作结束；②**统一观察接口**：观察类动作执行后本轮结束、返回新帧再决策；③世界对构造：直接 Flip、直接 Stay、Look→Flip、Look→Stay 四种情况；④实现 METRICS_SPEC §1 全部检查——`SetObjectPoses` 必须传完整物体列表；零编辑恢复两次逐像素比对；"信息一致"用智能体输入逐像素比较，"证据可获取"用实例掩码区域的像素差，**不要用 `visible` 标志**（默认 1.5 米距离限制）；⑤每种情况先做 ≥20 对并输出截图供人工核验。

**exp-train**
> 读 VAGEN 的 rollout 与优势计算代码，认领 T5、T6。在 src/train/ 实现：①T1 独立训练（配对数据，各分支独立 GAE）；②T2 同世界"先看/直接承诺"分叉对比（C2PO 式改编，rollout 数与 T3 对齐）；③T3 信息集信用 A_look(h) = ½Σ_w[R(w,看)−R(w,不看)] − λc；④T4 = T3 + 证据后一致性（Flip 对动作级 twin margin，Stay 对目标分布对称 KL，只约束目标不约束路线）。每项先写单元测试，特别测试"t 时刻输入相同的两分支绝不会收到不同的即时动作监督"。冒烟通过后 main_T0 与 ctrl_vagen 先进队列。

**analyst**
> 读 METRICS_SPEC 与 REGISTRY。在 analysis/ 写好：成功率–观察成本曲线、相同预算下的过早承诺率、证据后更新准确率（Look→Flip 与 Look→Stay 分开）、三档变化频率下的校准图、免训练观察策略对照表。全部用 plotlib.savefig_with_provenance 出图，先用冒烟和 pilot 的真实输出跑通（不要造数据）。

**writer**
> 认领 T8。放入 CVPR 2027 模板，写骨架、Intro（核心论点：同世界分叉会奖励"蒙对"；正确的信用应在与智能体当前信息一致的世界集合上评估；世界干预让我们能构造这样的集合，并监督证据后的目标更新）、Related Work（按 lit/related_work_seed.md 的 A 组逐一正面区分，SeekVLN 放第一位）、Method 公式。不写"首次""新范式"；对"信息集信用在期望上与同世界分叉一致、优势在于方差与平衡"如实说明。数字一律 \res{...}。

**reviewer**（D4 起，每轮先 /clear）
> 第 N 轮评审。只读 paper/、results/、CLAIMS.md、lit/related_work.md，按你的格式写 reviews/ROUND_N_skeptic.md。特别检查：与 SeekVLN、Grounding on Request、Q-DEG 的区分是否站得住；T1 消融是否证明了跨回合关系的独立价值；收益是否只是因为看得更多（对照成本曲线）。

**auditor**（每晚 + 每轮）
> 运行审计流程，写 reviews/AUDIT_<日期>.md。用 docs/METRICS_SPEC.md 给 Codex 写独立实现任务（只给规格和 episode 日志路径，不给我们的代码），在 crosscheck/ 下重算 SR、Premature-commit、证据后更新准确率，差异超过 0.5 个百分点就开 [EXP] 修复任务。另外抽查 20 个世界对：t 时刻输入是否真的逐像素一致、观察后掩码差异是否真实、两个候选是否可区分。

---

## 4. 时间表（美东时间；截止均为 AoE）

| 日期 | 主要工作 | 去留判断 | GPU 队列 |
|---|---|---|---|
| **D0 六 11-07** | 启动窗口；scout 查新；exp-env 装 VAGEN + 冒烟 + 任务族改造；exp-train 写成对 RL；writer 骨架 | **G1**：你签合同 v3，冻结 METRICS_SPEC / REGISTRY | 冒烟（P3） |
| **D1 日 11-08** | 四种情况世界对各 ≥20 个并人工核验；评测集与测试世界对**冻结**；免训练观察策略对照（从不看/总是看/固定频率/缺失触发）；main_T0、ctrl_vagen 进队列 | **去留 #1**：无"相同输入不同标签"；任务确实需要选择性观察？否 → 重做任务设计或停下讨论 | 门槛实验（P3）+ 基线（P2） |
| **D2 一 11-09** | main_T1–T4 全部进队列（3 种子，rollout 数对齐） | 冒烟检查：信用与一致性项数值正常，单元测试全绿 | 主实验（P2） |
| **D3 二 11-10** | 消融进队列；analyst 第一版主表；writer 写 Method + Experiments；**你：OpenReview 注册（截止 = 美东 11-11 周三 06:59）** | **G2**：相同观察预算下 T3/T4 是否优于 T1、T2？T4 是否优于 T3？否 → 收缩为评测协议 + 负结果分析 | 消融（P1） |
| **D4 三 11-11** | 三档变化频率统一评测（eval_*）；完整初稿；`/review-round` | **G3**：分诊清单定稿 | 泛化评测、补种子 |
| **D5 四 11-12** | 按分诊修改；补最致命的实验；第二轮 `/review-round` | 能否用一条清楚的主张解释全部结果？ | 收尾；若 G2 有信号可试 opt_* |
| 11-13 ~ 11-16 | 打磨写作与图；补充材料；可选实验 | **G4**：`/gate` 全绿 + 你通读并亲自提交（11-16 AoE = 美东 11-17 周二 06:59） | 可选 |
| 11-23 AoE | 补充材料截止 | | |

你的固定检查时间：09:00 / 14:00 / 21:00。

---

## 5. 算力编排
- 所有训练 / 评测 / 搜索都进 `scripts/gpu_queue.py`；搜索预算（K×M 次 solver rollout）与训练交互预算分开登记、分开报告。
- 优先级：P3 冒烟、pilot、诊断 > P2 控制组与 2×2 > P1 消融 > P0 可选。控制组在 D1 最先进队列。
- 预算规则：D0 冒烟测出单次训练 GPU·小时后，lead 计算 REGISTRY 总需求 vs D4 中午前可用 GPU·小时；不够就按第 8 节砍，记录到 Decisions。
- 同一任务失败 2 次必须开 [EXP] 排查，不允许"换种子直到成功"。

---

## 6. 五道质量闸
| 闸 | 拦什么 | 由谁 |
|---|---|---|
| 1. run_logged / gpu_queue | 无来源的数字 | 自动 |
| 2. 预注册（METRICS_SPEC + REGISTRY + 冻结测试编辑） | 事后挑指标、挑分组、编辑器适应自己的测试集 | hook（修改需你确认） |
| 3. Codex 独立重算指标 + 有效性抽查 | 评测代码 bug、无效编辑混入 | auditor + Codex |
| 4. 独立评审（新上下文 + 跨模型） | 叙事漏洞、与近邻区分不清 | reviewer + Codex |
| 5. 关卡 + /checkpoint | 方向跑偏、范围蔓延 | lead + 你 |

---

## 7. 结果不完美时：合规地写出最强的论文
**可以做：**
- 以**行为级诊断**为卖点："成功率高的智能体，在该改变时不改变、在该看时直接猜"——如果诊断扎实，本身就有价值。
- 只在成立的设定上主张收益（例如 Look 类显著改善、Flip 类持平），其余写成"何时有效、何时无效"的分析。
- 用 2×2 诚实拆解收益来源：如果收益主要来自"定向难例"而不是"成对监督"，就如实这样写。
- 真实失败案例做定性图；主动写 Limitations（单仿真器、单任务族、3B 模型）。
- 若主会证据不足：接受 AC 推荐的 Findings 轨道，或改投更合适的会议。

**不可以做：** 改数字、删种子、只报最好的种子、在测试集上挑检查点、事后换指标、让编辑器接触冻结测试编辑、从不同设定的论文抄基线数字而不注明、编造实验或引用。

---

## 8. 时间不够时的砍法（按顺序）
1. 砍 opt_*（GLANCE 复现、7B）与 abl_cmaes
2. 消融降到 1 个种子（表注说明）
3. 变化频率评测只保留 0% 与 50%
4. **绝不砍**：T1 关键消融、T2 同世界分叉基线、免训练观察对照、3 种子主表、冻结测试世界对、信息一致性检查、Codex 指标交叉验证、引用校验、Limitations

---

## 9. Claude 用量控制
- D1 后 scout 下线；reviewer 平时关闭，用 lead 的 `/review-round` 代替常驻窗口。
- 长会话 `/compact`；每个窗口每天至少 `/clear` 一次（SessionStart hook 会重新注入合同、任务板与倒计时）。
- 额度紧张时：analyst、auditor 的 `model` 可改为 `sonnet`；lead、exp-train、writer 保持最强模型。

---

## 10. 故障预案
| 情况 | 处理 |
|---|---|
| VAGEN 装不上 / 跑不动 | exp-env 报 Blocker；备选 RLinf；lead 与你决定是否切换 |
| AI2-THOR 的 API 无法保证确定性恢复或像素级一致 | 先修环境；做不到就不能声称"信息一致"，停下与你讨论 |
| 有效性检查丢弃率过高（>50%） | 收紧世界对采样分布，而不是放松检查 |
| 免训练"总是先看"已接近最优 | 任务不需要选择性观察：提高观察成本或换场景布局，重做门槛实验 |
| 去留 #1 不通过 | 停下与你讨论换问题；不默认改成"基准论文" |
| G2 不通过 | 收缩为诊断 + 负结果分析，评估是否投主会 |
| scout 发现更近的撞车工作 | 立即停，lead 与你讨论差异化 |
| 指标与 Codex 重算不一致 | 冻结写作，先修评测代码，受影响结果全部重跑 |
| 截止前 24 小时仍有 ERROR | 只修 desk-reject 类问题，其余写进 Limitations |
