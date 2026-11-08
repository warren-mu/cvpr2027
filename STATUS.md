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
- 已保存的阶段观察 `results/t00b_pending_wait_02_p2.json`：8 卡训练 `706439` 已在 hopper-07 启动并结束（exit 1）；全部 8 张 GPU 的出图 gate 通过，训练在 Ray 初始化阶段因 AF_UNIX socket 路径超过 107 字节失败，未进入 PPO 更新，性能变体未运行。作业实际计费 0.21333333333333335 GPU·小时，排队 9025 秒。`t00b_ray_deploy_preflight` 再次确认 `706354`（2 卡训练）与 `706436`（4 卡训练）仍为 Q 后，仅修复共同临时路径及 provenance，不改参数/拓扑/计时；T01 A/B/C 最近观察也均为 Q。没有取消作业。
- 节点查询中的 NUS HPC 三种资源池共 28 个节点，每节点 8 张 GPU。JSON 对无分配节点省略 `resources_assigned.ngpus`，故聚合器保留未知；同次 `pbsnodes -aSj` 明确显示在线 `hopper-04` 为 `8/8` 空闲，其余 NUS HPC 在线节点为 `0/8`；离线 `hopper-03` 不算可用。资源池/预约/队列约束仍可能阻止调度，不能把这个快照当作可立即获得 8 卡的承诺。
- 人类本次指令后的首次时钟为 2026-11-08 07:40:58 UTC，一小时节点限制期限为 08:40:58 UTC。`706408`、`706410` 已在期限前启动，因此该条件未触发；未重提、未取消任何运行作业。新增 4/8 卡测速及 T01 A/B 已不限定主机；单个 UUID 的成功不代表整台节点正常。
- 后续名额：2 卡训练、D/E/F/G、2 卡厨房 A；4 卡原始拓扑训练、4 卡客厅 B（之后打包冒烟）；单节点 8 卡训练。每卡都有实际工作，T01 每卡 12 CPU、最多 8 个 THOR 实例；新作业每卡启动前出图检查，纯黑退出 86，保留 UUID/节点和失败记录再预算检查重提。
- 多卡典型排队时长尚未测得；当前 2 卡测速仍排队。每个已结束作业的实际 qtime→stime 已由 `src/tools/pbs_usage.py` 记录；不能用单卡等待推算多卡等待。
- 人类已确认 T01 判据：编辑物体落定后相对指定位姿的位置偏差 ≤ 2 cm，其他物体相对编辑前位移 ≤ 1 cm，目的地必须是合法 receptacle 放置点；2 cm 不是编辑移动距离上限。相关物体 mask 并集内任一 RGB 通道差 > 10 的像素在 255×255 下至少 250 个（其他分辨率按面积缩放），直接类在 t 时刻同样检查。逐像素相同的共同前缀标准不变。旧 collect-only 记录不补写未测得的合法放置或物理沉降测量，不能据此宣称新判据通过。
- 最新预算证据 `results/t00b_premium_budget_09.json`：已完成 PBS 计费 1.2755555555555556 GPU·小时，未完成作业按申请上限保留 49.33333333333333，总计 50.608888888888885 / 60。包含失败与排队作业；预留不是已消费。每次重提仍执行正式提交预算检查。
- D/E/F/G 隔离诊断各完成 20 次，逐像素相同分别为 11/20、19/20、13/20、1/20，四种均未全部通过 exact-zero；没有放宽标准。D/F 最高通道差为 46/47，部分案例实际 cameraHorizon 末位变化，不能称全部状态完全相同；G 的最高通道差为 3，已报告状态相同的案例仍有广泛低幅像素差异。阴影归因未确定。E 的固定 0.5 m 编辑仅用于隔离诊断，不等于 T01 合法物理放置验证。证据：`results/t00b_retry2_isolation_summary.json` 及四份原始结果。`706408` 实际运行 138 秒、等待 4532 秒、计费 0.03833333333333333 GPU·小时（`results/t00b_phase_watch_02_accounting.json`）；E/F/G 的 P5 WARN 来自运行期间 watch 及其测试代码变动，diff 已保留，诊断代码没有在这些 diff 中变化。
- T01 新判据实现、合法放置与落定检查已提交。7 项策略测试、11 项严格合并测试通过；CPU fixture 不能替代实际试点。`t01_approved_dispatch_preflight_retry1` 确认 A/B 尚在 Q 后，提交 `9241b08` 将两者共享启动入口切换为批准判据版本。当时正在运行的旧 `706410` 保持旧路径，其 collect-only 记录不被新合并器自动接纳。
- 旧 `706410` 最终 exit 1（程序 35 分钟时限），31 次尝试中仅收集 1 条直接 Stay、2 条 Look→Stay 候选，批准标准通过数为 0。两个已比较 Look 前缀严格相同，但未测得的新物理判据不能补写；不将这个小样本比例外推为新试点通过率。实际 PBS 运行 35:04、排队 4591 秒、计费 0.5844444444444444 GPU·小时，证据为 `results/pilot_pairs_collect2.json` 与 `results/t00b_phase_watch_05_accounting.json`。失败、原始 journals 与审查图均已推送。释放的小档名额按优先级用于批准判据下的旧六场景重新采集；2 卡 C 已在完整预算检查及 6 项分区/11 项合并 CPU 测试通过后提交并确认入队，作业 `706582.hopper-m-02`、walltime 1 小时、无 host 限制、每卡 12 CPU/1 Unity、逐卡出图 gate。明确接续关系为 `706410 → 706582`：旧作业已自然结束，没有取消/删除作业；新 v3 仅接管固定旧六场景，不与 v2 A/B 场景重叠。回执见 `results/t01_c_approved_submit.json` 与 `logs/pbs/pilot_pairs_c_approved1-20261108T085050Z-4495bab2/`。
- T02 后端无关接口 24 项 CPU 测试通过（`results/t02_interface_cpu_checks.json`）；T05 配对信用、S1/S2 记录接口及 torch 梯度检查共 21 项 CPU 测试通过（`results/t05_credit_backward_cpu_checks.json`）。这不是训练集成或 GPU 验收完成；S2 以完整 PPO rollout-batch 更新计数，每 20 次在固定训练场景留出集评估。
- R01 本轮执行已交审并停止，见 `tasks/R01.report.md`：官方 CVPR 2027 kit 安装完成，最终编译 5 页 PDF、记录 errors/warnings 均为空，全部页只读外观检查未见明显裁切/重叠。独立 `paper/latexmkrc` 补入官方颜色定义，`r01_manuscript_preservation` 证明 main/sections 相对本轮开始未变。27 个 cite key 的保护校验为 25 通过、2 not_found，另保留 GAE 正式会议元数据未取得、三项正文描述建议和数字导出冲突；不由 PDF 成功宣称这些问题已通过。`r01_final_audit` 为 156 files / 0 FAIL，R01 使用 0 GPU·小时。
- GPU 等待期间已完成 R01 本轮执行与 T02/T05 的 CPU 子步骤；R01 停等审查，T02/T05 不宣称完整任务验收。T02 实时/预渲染选择等 D1；T04 批量训练池等 Claude 审完 T01 试点，当前不生成。T00b、T01、R01 分别报告、推送后停等审查。

- Ray 路径修复：为每个 baseline/性能变体在 scratch/r 下原子保留独占短目录；按 Ray 实际 session/socket 后缀与 10 位 PID 检查 UTF-8 总字节不超过 107。保存 ray_temp.json 与 config 来源；保留 scratch 子日志，不宣称这些子日志均已进入仓库。5 项纯 CPU 路径 fixtures 已在提交 b7e4a59 后通过（results/t00b_ray_scratch_cpu_checks.json）；未在登录节点启动 Ray 或 GPU。

- 8 卡训练修复重提已确认：`706439 → 706682.hopper-m-02`，新 run 为 `t00b_scale8_ray_retry`，Express/autox→largex、8 GPU、02:10:00、无 host 限制。旧作业自然结束，没有 qdel。正式提交记录为 `results/t00b_scale8_ray_retry_submit.json`，回执 `logs/pbs/t00b_scale8_ray_retry-20261108T101459Z-6cbb85c7/submission.json`。`results/t00b_ray_retry_watch.json` 确认 706354/706448/706582 在 smallx、706436/706450 在 mediumx、706682 在 largex，六者均 Q（ngpus 资源不足）；每卡仍配 12 CPU，启动重新逐卡出图。监控器遇到旧 706439/706410 的 existing-retry 拒绝原因是已保留明确替代 payload，root 已核对 706682/706582 的成功回执，不再重复提交。

- 后续观察 `results/t00b_pending_wait_03_p3.json`：`706436`（4 卡训练）和 `706450`（T01 B）在 hopper-11 自然结束，均 exit 1；其余 706354/706448/706582/706682 仍为 Q。两者分别排队 11330/10676 秒，实际计费 0.21222222222222223/0.04555555555555556 GPU·小时（同前缀 accounting.json）。4 卡训练所有卡出图通过，短 Ray 目录成功；模型已加载后，Triton 在加载共享缓存 __triton_launcher.so 时返回 ENOENT，0 更新，未运行性能变体。随后只读查看该 .so 已存在，不能断言已证明并发竞争或 NFS 根因；拟隔离每个 run/variant 的 scratch 编译缓存后验证，不改训练参数。T01 B 的逻辑 GPU0 在 Initialize 期间 UnityPlayer.so native SIGSEGV（signo11, addr0x4），其余三卡出图通过；30 秒超时是崩溃后的表象，不归为黑帧，不改退出码 86 规则，也不因本次故障放宽 timeout/判据。失败 Player.log、UUID、截图和运行记录已提交 `4fa9c27`。B 拟按原参数显式重提，先做预算检查；没有取消任何作业。

- 编译缓存隔离缓解已部署：`t00b_compile_cache_preflight` 正式确认 706354/706682 都为 Q 后，2/4/8 卡的训练子进程各用本 run/variant 的 scratch/work/compile_cache 下 TRITON_CACHE_DIR、TORCHINDUCTOR_CACHE_DIR；旧缓存完整保留，同一 run 内各 rank 仍共享。仅更换两项环境变量，不改模型、TP、eager/编译模式或计时逻辑；路径和继承旧值记入 config/resources.jsonl。新缓存造成的初始化/首步成本照实计入，不能宣称已证实原 ENOENT 根因或缓解已有效。代码提交 1d88fcd 后 4 项纯 CPU fixture 全部通过（results/t00b_compile_cache_cpu_checks.json）；未在登录节点运行编译或 Ray/GPU 服务。T01 B 原样人工重提回执为 `706450 → 706728.hopper-m-02`（run pilot_pairs_parallel_b_02），4 卡 02:00:00、无 host；旧作业自然结束，无 qdel。回执位于 `logs/pbs/pilot_pairs_parallel_b_02-20261108T104109Z-99c43c96/submission.json`。

- 当前六名额正式观察（2026-11-08T12:10:17.338228+00:00，`results/t00b_pending_wait_05_p7.json`）：706354（2 卡训练）、706448（2 卡 T01 A）、706582（2 卡 T01 C）在 smallx；706732（4 卡训练缓存隔离重试）、706728（4 卡 T01 B 原样重试）在 mediumx；706682（8 卡训练重试）在 largex。六者均 Q，ngpus 资源不足，均无 host 限制。新 4 卡映射 `706436 → 706732.hopper-m-02`，run `t00b_scale4_cache_retry`、03:00:00；回执 `logs/pbs/t00b_scale4_cache_retry-20261108T104642Z-68de14f9/submission.json`。旧作业自然结束，未 qdel；无额外黑帧重提动作。源码只读核对表明固定 VAGEN/VERL 和 Ray runtime_env 逐键合并继承环境，未发现覆盖两项编译缓存路径的代码；这不是新 GPU 运行逐 actor 环境快照，隔离是否有效仍待实测。

- T00b 最终测速汇总 producer 已补齐：`src/tools/t00b_training_summary.py` 与测试均为新增文件，既有 parser/校验统计脚本未改。提交 6748818 后 15 个纯 CPU synthetic fixtures 通过（`results/t00b_training_summary_cpu_checks.json`），覆盖来源/缺失/失败、同作业变体关联、速度比公式、整作业会计去重及分配卡数交叉核对。尚未进行真实训练汇总；程序成功与六槽训练资格及最终任务验收是不同字段，不将夹具当作GPU实测。

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
