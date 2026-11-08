# STATUS（共享黑板 —— 所有窗口都读写这里）

当前阶段：T00b + T01（并行执行；T00 不重做）｜ 下一关卡：D1 选观察后端 → 周一（11/9）晚去留检查点 ｜ 更新：2026-11-08

## 执行方式（2026-11-07 起）
- Codex 在 Hopper 上执行，规则见 `AGENTS.md`；任务说明和进度见 `tasks/README.md`；Claude 的审查意见在 `reviews/`。
- 下面 v3.1 时期的旧任务板只作历史参考，以 `tasks/README.md` 为准。

## 当前执行与 premium 核对（2026-11-08）

- 人类明确确认：premium 指 Express/autox。提交统一 `#PBS -q autox`、项目 `CFP04-SF-125`、`cvpr27_` 作业名；不存在另一个名为 premium 的队列。
- 已用 `hpc autox`、`qstat -Qf`、`qstat -u e1583364`、`pbsnodes -a -F json`、`pbsnodes -aSj` 留存查询。可审计证据：`results/t00b_premium_capacity.json` 及 `logs/runs/t00b_premium_capacity/`。
- `autox` 路由至 `smallx`（1–2 卡，每用户运行上限 3）、`mediumx`（3–7 卡，上限 2）、`largex`（8–16 卡，上限 1）。三档相加为 6，与人类说明一致；这是运行上限，不是保证立即有空闲卡。`largex` 最短申请 walltime 为 `02:00:01`。
- 三个执行队列的 `resources_default.queue_charge_rate` 都为 `1`。预算按 PBS 实际分配卡数 × 已结束作业的 `resources_used.walltime` × 作业 `queue_charge_rate` 计算；排队/运行但未有最终会计记录的作业保留整个申请上限 × 费率。T00b+T01 合计不得超过 60 个计费 GPU·小时。队列配置和 PBS 字段不是独立项目余额账单；不宣称已与项目账单对账。
- 最近正式观察 `results/t00b_phase_watch_05.json`：`706408`（1 卡 D/E/F/G）与 `706410`（旧六场景 T01）均已结束。`706354`（2 卡训练）、`706448`（2 卡厨房 A）在 smallx 排队；`706436`（4 卡训练）、`706450`（4 卡客厅 B）在 mediumx 排队；`706439`（单节点 8 卡训练）在 largex 排队。全部已属 Express，不需要为 premium 别名迁移。
- 节点查询中的 NUS HPC 三种资源池共 28 个节点，每节点 8 张 GPU。JSON 对无分配节点省略 `resources_assigned.ngpus`，故聚合器保留未知；同次 `pbsnodes -aSj` 明确显示在线 `hopper-04` 为 `8/8` 空闲，其余 NUS HPC 在线节点为 `0/8`；离线 `hopper-03` 不算可用。资源池/预约/队列约束仍可能阻止调度，不能把这个快照当作可立即获得 8 卡的承诺。
- 人类本次指令后的首次时钟为 2026-11-08 07:40:58 UTC，一小时节点限制期限为 08:40:58 UTC。`706408`、`706410` 已在期限前启动，因此该条件未触发；未重提、未取消任何运行作业。新增 4/8 卡测速及 T01 A/B 已不限定主机；单个 UUID 的成功不代表整台节点正常。
- 后续名额：2 卡训练、D/E/F/G、2 卡厨房 A；4 卡原始拓扑训练、4 卡客厅 B（之后打包冒烟）；单节点 8 卡训练。每卡都有实际工作，T01 每卡 12 CPU、最多 8 个 THOR 实例；新作业每卡启动前出图检查，纯黑退出 86，保留 UUID/节点和失败记录再预算检查重提。
- 多卡典型排队时长尚未测得；当前 2 卡测速仍排队。每个已结束作业的实际 qtime→stime 已由 `src/tools/pbs_usage.py` 记录；不能用单卡等待推算多卡等待。
- 人类已确认 T01 判据：编辑物体落定后相对指定位姿的位置偏差 ≤ 2 cm，其他物体相对编辑前位移 ≤ 1 cm，目的地必须是合法 receptacle 放置点；2 cm 不是编辑移动距离上限。相关物体 mask 并集内任一 RGB 通道差 > 10 的像素在 255×255 下至少 250 个（其他分辨率按面积缩放），直接类在 t 时刻同样检查。逐像素相同的共同前缀标准不变。旧 collect-only 记录不补写未测得的合法放置或物理沉降测量，不能据此宣称新判据通过。
- 最新预算证据 `results/t00b_premium_budget_04.json`：已完成 PBS 计费 0.8044444444444443 GPU·小时，未完成作业按申请上限保留 47.33333333333333，总计 48.13777777777778 / 60。包含失败与排队作业；预留不是已消费。
- D/E/F/G 隔离诊断各完成 20 次，逐像素相同分别为 11/20、19/20、13/20、1/20，四种均未全部通过 exact-zero；没有放宽标准。D/F 最高通道差为 46/47，部分案例实际 cameraHorizon 末位变化，不能称全部状态完全相同；G 的最高通道差为 3，已报告状态相同的案例仍有广泛低幅像素差异。阴影归因未确定。E 的固定 0.5 m 编辑仅用于隔离诊断，不等于 T01 合法物理放置验证。证据：`results/t00b_retry2_isolation_summary.json` 及四份原始结果。`706408` 实际运行 138 秒、等待 4532 秒、计费 0.03833333333333333 GPU·小时（`results/t00b_phase_watch_02_accounting.json`）；E/F/G 的 P5 WARN 来自运行期间 watch 及其测试代码变动，diff 已保留，诊断代码没有在这些 diff 中变化。
- T01 新判据实现、合法放置与落定检查已提交。7 项策略测试、11 项严格合并测试通过；CPU fixture 不能替代实际试点。`t01_approved_dispatch_preflight_retry1` 确认 A/B 尚在 Q 后，提交 `9241b08` 将两者共享启动入口切换为批准判据版本。运行中的旧 `706410` 保持旧路径，其 collect-only 记录不被新合并器自动接纳。
- 旧 `706410` 最终 exit 1（程序 35 分钟时限），31 次尝试中仅收集 1 条直接 Stay、2 条 Look→Stay 候选，批准标准通过数为 0。两个已比较 Look 前缀严格相同，但未测得的新物理判据不能补写；不将这个小样本比例外推为新试点通过率。实际 PBS 运行 35:04、排队 4591 秒、计费 0.5844444444444444 GPU·小时，证据为 `results/pilot_pairs_collect2.json` 与 `results/t00b_phase_watch_05_accounting.json`。失败、原始 journals 与审查图均已推送。释放的小档名额按优先级用于批准判据下的旧六场景重新采集；2 卡 C 的代码与提交前预算检查准备中，尚未提交，不与 A/B 场景重叠。
- T02 后端无关接口 24 项 CPU 测试通过（`results/t02_interface_cpu_checks.json`）；T05 配对信用、S1/S2 记录接口及 torch 梯度检查共 21 项 CPU 测试通过（`results/t05_credit_backward_cpu_checks.json`）。这不是训练集成或 GPU 验收完成；S2 以完整 PPO rollout-batch 更新计数，每 20 次在固定训练场景留出集评估。
- R01 本轮执行已交审并停止，见 `tasks/R01.report.md`：官方 CVPR 2027 kit 安装完成，最终编译 5 页 PDF、记录 errors/warnings 均为空，全部页只读外观检查未见明显裁切/重叠。独立 `paper/latexmkrc` 补入官方颜色定义，`r01_manuscript_preservation` 证明 main/sections 相对本轮开始未变。27 个 cite key 的保护校验为 25 通过、2 not_found，另保留 GAE 正式会议元数据未取得、三项正文描述建议和数字导出冲突；不由 PDF 成功宣称这些问题已通过。`r01_final_audit` 为 156 files / 0 FAIL，R01 使用 0 GPU·小时。
- 等待 GPU 时并行推进 R01 文献/模板、T05 配对信用及 S1/S2 原始记录接口、T02 后端无关框架与 CPU 单测。T02 实时/预渲染选择等 D1；T04 批量训练池等 Claude 审完 T01 试点，当前不生成。T00b、T01、R01 分别报告、推送后停等审查。

## 旧任务板（v3.1，已由 tasks/ 取代）

| ID | 标签 | owner | 任务 | 完成标准 | 证据 | 状态 |
|---|---|---|---|---|---|---|
| T1 | [LIT] | scout | 核实 related_work_seed 中 ⚠/☑ 条目（含 SeekVLN 全文、C²Nav、LIBERO-Plus §6.2）；按 PLAYBOOK 补搜 | lit/novelty_risk.md：风险等级 + 最接近 3 篇差异 | lit/novelty_risk.md | todo |
| T2 | [EXP] | exp-env | 安装 VAGEN，冒烟训练 Navigation 一次 | STATUS 记录单次完整训练预计 GPU·小时 | logs/runs/setup_*.log | todo |
| T3 | [EXP] | exp-env | 关系指称任务 + `select` 结束 + 统一观察接口 + 世界对构造与 METRICS_SPEC §1 全部检查 | 四种情况各 ≥20 对，截图人工核验；无"相同输入不同标签" | results/pilot_pairs*.json | todo |
| T4 | [EXP] | exp-env | 冻结评测集与测试世界对（三档变化频率） | experiments/eval_sets/*.json 提交 git | | todo |
| T5 | [EXP] | exp-train | T1–T4 训练目标 + 单元测试（重点：相同输入绝不收到不同即时监督） | 单元测试通过；冒烟训练数值正常 | | todo |
| T6 | [EXP] | exp-train | 免训练观察策略：从不看/总是看/固定频率/缺失触发，在 base 与 T0 检查点上评测 | results/obs_*.json；成功率–观察成本表 | | todo |
| T7 | [EXP] | exp-env | 诊断 base / ppo / vagen 的主要指标 | results/diag_*.json | | todo |
| T8 | [WRITE] | writer | 模板骨架 + Intro（"不奖励蒙对"论点）+ Related Work（SeekVLN 首位）+ Method 公式 | check_paper.py --fast 通过 | paper/main.tex | todo |
| T9 | [WRITE] | lead | OpenReview 注册（docs/OPENREVIEW_DRAFT.md），**人类 11-10 AoE 前提交** | 注册截图 | | todo |

状态取值：todo / doing / blocked / done（done 必须有证据）

## Decisions（含负结果，只追加不删除）
- 2026-11-07：选定方向"多轮 VLM 智能体 × 世界模型推理 × RL 后训练"，以 VAGEN 为底座。
- 2026-11-07：v2"世界编辑自博弈"。
- 2026-11-07：经外部评审与二次查新，v2 的若干组件已有近邻（Q-DEG、DEER-3D、DUEL、RedVLA、Grounding on Request），撤回"阶段 1 即保底论文""首次自博弈""失败即转基准论文"三项判断；改为 v3"Flip / Stay / Look"，核心问题收紧为"世界变化时智能体是否应改变行为"，单仿真器、单任务族，自博弈降为可选扩展。

- 2026-11-07：第二轮外部评审后修订为 v3.1：标签改为两层定义；加入"相同输入不得有不同即时标签"的硬约束；方法核心改为"信息集信用 + 证据后一致性"；加入 SeekVLN 为首要近邻，C2PO 式同世界分叉作为必比基线；新增免训练观察策略门槛实验；CMA-ES 降为可选并更正命名。

- 2026-11-07：算力确定为只用 NUS Hopper H200（项目 CFP04-SF-125，Express 队列，余额约 8,919 GPU·小时，项目成员共用）。不租卡，不用 Vanda 和实验室服务器。
- 2026-11-07：执行方式改为 Codex 执行、Claude 规划与审查；新增 `AGENTS.md`、`tasks/`、`scripts/audit_provenance.py`；多种子汇总文件记录来源哈希。
- 2026-11-07：研究合同升为 v3.2：选题与方法不变；动作空间改为离散视点；观察后端（L 实时 / P 预渲染 / M 换仿真器）由 D1 按 T00 实测决定。原因：H200 图形硬件很少，实时渲染速度未知。

- 2026-11-08：人类确认（经 Claude 建议）：① T01 两个参数——被编辑物体偏差 ≤ 2 cm、其他物体位移 ≤ 1 cm；证据像素在 255×255 下 ≥ 250 个（通道差 > 10）；② Calibration 改为按训练变化频率 ρ ∈ {0.25, 0.5, 0.75} 检验观察频率随 ρ 上升，测试三档变化频率改报 SR 与过早承诺率（原因见 METRICS_SPEC）；③ 预注册次要终点 S1（证据前信用方差）与 S2（学习效率），均在任何训练结果出现之前登记。已写入 docs/METRICS_SPEC.md。
- 2026-11-08：premium 即 Express/autox，计费倍率 1；每人同时最多 6 个作业（三档 3/2/1 可叠加）。T00b+T01 预算 60 GPU·小时。

## Blockers（需要 lead / 人类拍板）
- VAGEN 是否发布训练好的 checkpoint 未确认；没有则 ctrl_vagen 需自训。
- AI2-THOR 能否做到确定性恢复与像素级输入比较；`SetObjectPoses` 需传完整物体列表（未列出的可移动物体会被删除）。由 T00 验证。
- H200 上 AI2-THOR 能否渲染、速度多少（T00 第 2 步）；计算节点是否有 Vulkan、能否联网。

## Gate log
- G1 研究合同：v3.2 方向已于 2026-11-07 获人类同意；载体待 D1 决定后补签
- D1 观察后端：
- 去留检查点（11/9 晚）：
- G2（11/13 晚）：
- G3 初稿 + 第一轮评审：
- G4 终检：
