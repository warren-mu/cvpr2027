# STATUS（共享黑板 —— 所有窗口都读写这里）

当前阶段：T01v2 试点（最高优先级）+ T02/T05 CPU 代码 ｜ D1 = P 已定（11/08 15:03）｜ 下一关卡：去留检查点 11/10（周二）美东中午 ｜ 更新：2026-11-08

## 当前作业与交审状态

最近正式观察：2026-11-08T16:29:54.562277+00:00，来源 results/t00b_active_watch_04.json。Q=排队，R=运行，F=已结束。

| PBS 作业 | GPU | 工作 | 队列 | 状态 / 节点 |
|---|---:|---|---|---|
| 706354 | 2 | 训练测速 | smallx | Q / 未分配 |
| 706448 | 2 | T01 厨房 A | smallx | Q / 未分配 |
| 706582 | 2 | T01 原六场景 C | smallx | Q / 未分配 |
| 707054 | 4 | 训练测速：worker 缓存验证 | mediumx | R / hopper-13/0*48 |
| 707057 | 4 | 诊断停止；独立pack通过 | mediumx | F / hopper-11/1*48 |
| 706682 | 8 | 训练测速 | largex | Q / 未分配 |

- 已完成计费2.207777777777778 GPUh；未完成保留41.33333333333333，合计43.541111111111114 /60（results/t00b_premium_budget_12.json）。运行中的申请仍保留全额。
- 707054仍运行：四卡gate通过，已有首个完整PPO更新中期快照 results/t00b_scale4_update1_snapshot.json，耗时345.462秒、gen243.159秒；不是20步验收或完整测速。后续已完成更新原始行见中期快照 results/t00b_scale4_midrun_snapshot_02.json；基线仍运行。清理EBUSY异常保留在训练进度报告。
- 707057自然结束，外层exit1；独立pack退出0、passed=true，报告tasks/T00b.packing.report.md。PBS实际185秒、等待1804秒、计费0.20555555555555555 GPUh；联合账单只计一次，不拆分两个子步骤。
- 碰撞探针三个场景因原位Teleport拒绝后旋转metadata微变而严格停止，另一个原始落定失败；未放宽标准、未接纳或重标旧世界对。原始证据归档与诊断子报告已完成，中档一个名额已释放；物理判定边界交Claude/人类审查，现有训练及A/C监控继续。
- R01本轮报告已推送并停等审查；T02/T05 CPU子步骤报告已推送，均不等于完整任务验收。T00b最终报告等待2/4/8卡完整测速；T01等待A/C与诊断结论。T04批量池继续等T01审查，T02后端等D1。

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
- 最新预算证据 `results/t00b_premium_budget_10.json`：已完成 PBS 计费 2.002222222222222 GPU·小时，未完成作业按申请上限保留 29.333333333333332，总计 31.33555555555556 / 60。包含失败与排队作业；预留不是已消费。每次重提仍执行正式提交预算检查。
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

- 当前六名额正式观察（2026-11-08T15:22:01.589676+00:00，`results/t00b_active_watch_02.json`）：706728（T01 B）与706732（4卡训练）均在 hopper-13 自然结束、exit 1；706354（2卡训练）、706448（T01 A）、706582（T01 C）、706682（8卡训练）仍为 Q，ngpus 资源不足。两个中档名额待训练缓存修复与独立碰撞诊断/打包冒烟准备完成后依优先级补充；没有取消或修改运行作业。新 4 卡映射 `706436 → 706732.hopper-m-02`，run `t00b_scale4_cache_retry`、03:00:00；回执 `logs/pbs/t00b_scale4_cache_retry-20261108T104642Z-68de14f9/submission.json`。旧作业自然结束，未 qdel；无额外黑帧重提动作。源码只读核对表明固定 VAGEN/VERL 和 Ray runtime_env 逐键合并继承环境，未发现覆盖两项编译缓存路径的代码；这不是新 GPU 运行逐 actor 环境快照，隔离是否有效仍待实测。

- T00b 最终测速汇总 producer 已补齐：`src/tools/t00b_training_summary.py` 与测试均为新增文件，既有 parser/校验统计脚本未改。提交 6748818 后 15 个纯 CPU synthetic fixtures 通过（`results/t00b_training_summary_cpu_checks.json`），覆盖来源/缺失/失败、同作业变体关联、速度比公式、整作业会计去重及分配卡数交叉核对。尚未进行真实训练汇总；程序成功与六槽训练资格及最终任务验收是不同字段，不将夹具当作GPU实测。

- 等待期间资源复核（2026-11-08T13:14:24.665158+00:00，`results/t00b_capacity_wait_13utc.json`）仍确认 Express 三档运行上限 3/2/1、倍率均为 1；`own_jobs.txt` 中六个本阶段作业及无关的 706017 均为 Q，没有本账号运行作业。`node_summary.txt` 中 hopper-04 显示 8/8 空闲，但节点 JSON 将其 gpu_model 标为 H100；不能把这个快照当作可立即使用的 H200。现有 thor_allocation/thor_common 卡型校验要求实际 H200，没有为追逐空闲节点改硬件、节点限制或队列；其他项目/资源池未操作。当前等待原因仍以逐作业 PBS comment 的 ngpus 不足为准，未声称已得到可预测的开始时间。

- T01 B 子步骤报告 `tasks/T01.b02.report.md`：706728 的四卡 gate 通过，全部既定场景已结束，但 FloorPlan222/226 因固定间距的可达视点不足在尝试前失败。正式 B 汇总 `results/pilot_pairs_b02_checkpoint_merge.json` 记录 1000 次尝试、四类各250拒绝，批准与pending均为0；全部尝试在编辑前的baseline物理代理谓词处被拒绝。所查203首末例和204/205/206首例落定通过而collision谓词不通过；上游全collider查询不排除source trigger且不提供穿透深度，不能据非空列表断言非法穿模。旧数据不重标，准备独立诊断，不改生产验收判据。该作业等待16072秒、实际分配用量0.5177777777777778 GPUh（`results/t00b_pending_wait_10_p1_accounting.json`）。

- 4卡缓存隔离重试706732的四卡gate通过，但仍在首次编译时发生.cubin FileNotFoundError，未完成训练更新，vllm04未启动；run级私有目录没有解决问题。失败日志与源码只读检查显示编译流水线put_group后立即读取产物失败，后续查看产物又已存在；不能证明NFS或并发谁是根因。保持官方compile/TP/参数/计时，下一候选为Ray worker启动时隔离缓存，需正式CPU验证和Q状态预检后再部署。该作业等待16205秒、实际分配用量0.2088888888888889 GPUh（`results/t00b_active_watch_02_accounting.json`）；完整失败记录保留。

- Ray worker 缓存候选已完成 CPU 子步骤：部署前正式记录 t00b_worker_cache_preflight 确认原2/8卡训练均为Q；代码提交515b21d后，t00b_worker_cache_cpu_checks通过68项CPU测试（含9项新worker缓存fixture）。每个Ray worker启动时使用独立的scratch缓存，普通编译子进程继承所属worker路径，driver保留run级路径；只修改两项编译缓存变量。冻结hook/config哈希和worker原始manifest进入归档；新增归档发生在完成计时取值之后。CPU成功不证明GPU编译故障已修复，下一步仅按原参数4卡重试706732，保留所有失败证据。

- 4卡worker缓存验证已提交并确认Q：706732 → 707054.hopper-m-02，run t00b_scale4_worker_cache_retry、03:00:00、无host限制，正式提交记录results/t00b_scale4_worker_cache_retry_submit.json；t00b_pending_wait_11_p6于15:51:12 UTC确认mediumx排队，原四项仍Q。旧作业已自然结束，无取消动作。提交前已完成及预留31.33555555555556，加本次12 GPUh申请上限仍低于60。

- 独立碰撞诊断与打包入口代码2c8b202后，四组CPU测试分别13/8/13/9项通过（results/t01_diag_pack_*_cpu.json）。固定203/204/205/206逐卡比对原查询和暂停物理后的原位数组TeleportObject，保留完整元数据/位图哈希；仅诊断、不输出物理合格标签或接纳世界对。严格状态变化不被忽略，普通诊断失败在已证实清理后允许独立pack继续且整体仍失败；黑帧86或清理不确定立即停。新增重提守卫验证实际probe/pack子链、UUID与原始黑帧PNG，原T01 shard真实Python -u -m argv也有fixture；未放宽黑帧标准。拟在另一中档名额以4卡00:20:00运行诊断后pack，每卡12CPU，无节点限制。

- 六名额已经补齐并正式确认：results/t00b_active_watch_03.json（2026-11-08T15:54:52.370585+00:00）显示706354/706448/706582为smallx，707054/707057为mediumx，706682为largex，全部Q、无host限制。707057.hopper-m-02为新独立碰撞诊断后打包冒烟，run t01_collision_pack_01、4卡00:20:00；回执results/t01_collision_pack_01_submit.json。没有取消作业。新预算results/t00b_premium_budget_11.json：已完成计费2.002222222222222 GPUh，未完成按申请上限预留42.666666666666664，合计44.66888888888889 /60。联合诊断/pack的整作业会计只计一次，不编造两子步骤的独立PBS扣费。

- 碰撞诊断证据归档已完成：代码c2eab0d后12项CPU测试通过，results/t01_collision_summary_01.json汇总验真成功，416个原始文件共82566298字节已由脚本按hash复制到logs/runs/t01_collision_summary_01/evidence。原probe退出1、diagnostic_completed=false、physical_validity/task_acceptance=null保持不变。正式已测对象结果为33个旧列表非空→原位Teleport拒绝、1个旧列表空→拒绝、1个旧列表空→成功；这是有选择停止的未完成诊断，不外推通过率或物理合法性。三个场景的末项有旋转metadata微变，205在落定阶段停止。诊断子步骤报告tasks/T01.collision.report.md已完成交审；明确未取得合格碰撞代理替代，不从查询成功推定物理有效性，未提交采用替代判据的新采集。

- 4卡训练707054的第三份正式中期快照results/t00b_scale4_midrun_snapshot_03.json保存了第13/14次更新原始计时，分别237.855/248.816秒，之后进入第15步；这不是完整基线或性能变体验收。原2卡/8卡训练、T01 A/C仍排队，已有运行未修改或取消。当前最终已完成会计为2.207777777777778 GPUh，未完成申请上限预留41.33333333333333，合计43.541111111111114/60（results/t00b_premium_budget_12.json）；该会计数不包含尚在运行的707054最终用量，后者仍按整笔12 GPUh预留。独立碰撞报告已推送等待判定方案审查，未改生产物理标准。

- 已按人类最新授权停止 T01 v1：正式 preflight 记录 A706448/C706582 均Q、无stime/usage，qdel退出0，postflight均F且仍无stime/usage；B706728此前已自然结束，未再次操作。证据 results/t01v1_cancel_preflight.json、t01v1_cancel_ac.json、t01v1_cancel_postflight.json；v1全部记录保留。T01v2为当前最高优先级，试点子预算8 GPUh且仍纳入阶段总60；物理判定改按已批准落定反应，R01按PASS只补两条refs人工官方核验注释，不重跑。2/4/8卡训练继续，8卡706682已包含短Ray路径修复。

- 4卡707054的baseline已完成20次实际更新且run_logged退出0，逐步新RGB计数完整；中位step=259.0075s、gen=176.6365s。原始GPU采样错误1次及Python临时目录EBUSY警告均保留，详见results/t00b_scale4_worker_cache_retry_baseline.json；性能变体和PBS最终会计继续等待。t00b_scale4_baseline_complete_audit为390份、0FAIL。取消前未运行的v1 A/C因PBS没有run_count，旧会计观察失败1已保存，正在补显式Q→qdel成功→F证据链支持，不把缺失用量直接推成0。

- T01v2 首轮实现提交d37b922及后续预检绑定补丁：51项v2 CPU检查通过，取消/监控/提交回归分别5/9/19项通过，受影响的采集mock再测8项通过；这些不是GPU产出。t01v2_prequeue_audit_01为397份0FAIL。v1 A/C的完整Q→qdel成功→F链正式记账为0分配GPUh（results/t01v1_cancel_accounting_01.json），run_count/stime缺失值原样保留，排队等待时间未知。首轮厨房1..12与客厅201..212固定分区并行提交，全部255×255、每卡12CPU、最多1构造+1顺序回放实例、无host；共同受8GPUh子预算/60总预算约束。按人类新增要求，各分片四类任一不足5对，下一轮启用完整列表复制候选，首轮不暗中替换已测规则。

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

- 2026-11-08 下午（Claude 提议，人类转发总 prompt 即视为批准）：① D1 = 实时渲染 + 前缀共用（同进程、agent 不动，构造两个世界；前缀最大通道差 ≤ 3 且编辑物体与锚点前缀掩码为 0 才算信息一致，之后两个世界共用同一组前缀帧），已写入 METRICS_SPEC §1；② T01 改为 v2 按构造生成（tasks/T01v2.md），物理有效用落定反应判定，不再用碰撞查询；③ 去留检查点从周一晚顺延到周二（11/10）美东中午；④ 数字导出只包含 agg_/obs_/diag_/data_ 前缀的结果。

- 2026-11-08 12:45（人类确认，回答 Codex 的提问）：`restore_deterministic` 的逐像素零差改为诊断记录；硬性门槛改为状态级确定性（从快照恢复后物体位置偏差 ≤ 1 cm、agent 位姿一致、见证在两个世界回放成功）。前缀一致按 D1 的前缀规则判定。已写入 METRICS_SPEC §1。

- 2026-11-08 下午（Claude，生成设计，不改任何阈值）：厨房首轮 0 对的原因已对上（4 场景旧预检排除、6 场景缺天然候选、2 场景锚点苹果落定后滚动 2.3–4.5 cm 致 124/124 次物理检查被拒，之后存储上限报错）。复查第二轮代码发现锚点与复制源按字母序选（Apple 居首）、每场景只复制一次、复制后丢弃天然候选组。T01v2 补充 3：锚点和复制源排除会滚动的类别；复制最多 6 次；天然与复制候选组并用；组合间轮换，连续 10 次物理拒绝即跳过；存储上限不再中断搜索。第二轮 GPU 作业须在接入后提交。

- 2026-11-08 下午（Claude 提议，METRICS_SPEC 一行待人类确认）：发现补充 2 放宽 Look 证据条件后，可能产生靠观察无法做对的 Look→Stay 对（一个世界看得见 X，另一个世界 X 也在 A 旁但这一步看不见，智能体会推断成 B）。新增「可解性」：每个世界至少有一个一步观察动作能看到 X（≥ 250 像素）。写入 T01v2 补充 4 与 METRICS_SPEC §1（待确认）。

- 2026-11-08 下午（独立审查 agent 发现，Claude 核实代码属实）：构造顺序使 r=0.5 时约 75% 回合的正确目标是按名字排第一的候选，训练中可被记忆。T01v2 补充 5：本轮只报告偏差，建池时用随机交换 Flip 标签和平衡 Stay 两侧消除；下一轮生成代码随机分配两侧。不改任何指标。

- **2026-11-08 15:03（人类确认，均在任何训练或评测结果出现之前）**：
  ① D1 = P：只预渲染起点附近 D 步内的视角，训练和评测查表（tasks/T02.md 重写）。原因：实时双仿真器在训练循环里会遇到黑帧、Unity 崩溃和渲染噪声，两个世界同步要改 VAGEN 底层；预渲染把仿真器限制在离线阶段。代价：具身导航的味道变弱（只在起点附近转头、走几步），论文中如实说明。
  ② 对照解读改为 2×2（研究合同 v3.3 §4 第 3 条）：独立审查指出，同步配对下证据前同一步的两份梯度相加，T1 与 T3 一阶上只差基线（价值网络 vs Q_commit），原「T1 不输则配对机制不成立」撤回。T3 对 T2 检验配对，T3 对 T1、T2 对 T0 检验 Q_commit 基线，T4 对 T3 检验一致性。必做改为 T0、T2、T3、T4，T1 改为应做。这是 Claude 写合同时的错误。
  ③ METRICS_SPEC：可解性（含预算 D 内可解）、无锚点泄露、后端一致（加掩码 IoU）、前缀复用合法；§1b 两个时刻 τ / τ_seen；过早承诺与证据后更新准确率改按 τ_seen；S1 改为按独立决策单元折算；T4 一致性只在本世界看到锚点后施加；没有可选标号的证据前步用价值网络基线。
  ④ 待办（writer）：paper/sections/method.tex 按 ② 改写方法描述——信息集信用 = 同步配对采样 + 立即选择基线，并说明两者分别由哪组对比检验。

- 2026-11-08 15:55（人类要求「调整显卡使用、减少排队、尽早开跑」，Claude 制定）：
  ① 排队策略写进 AGENTS.md：默认 1 卡作业（今天 8/8 在 1–3 分钟内启动）；不申请 2 卡和 8 卡以上（2 卡 5 个都没启动过）；需要更多并行时，用 3–7 卡作业打包。
  ② 新增 tasks/E2E.md：第二轮一出世界对，就用最早的几对把整条流程各跑几步，提前暴露 VAGEN 接入和信用分配的问题，拿到真实训练速度。
  ③ 新增 tasks/G01.md：扩大生成与第二轮验收并行。场景划分在任何评测结果出现之前定下：训练 x01–x20、测试 x21–x30（四类房间）。
  ④ 两者都不改任何指标或阈值。

- 2026-11-08 16:10（人类决定）：时间有限，以「最小代价做出一篇完整论文」为准。
  - 不加测其他模型（7B/32B），不做外部基准。
  - 训练只做必做的 T0、T2、T3、T4，每臂 3 个种子；T1、ctrl_vagen 只在有空余时做，不做就在论文里写明。
  - 次要终点 S1、S2 直接从训练日志算，不需要额外运行。
  - 校准（多个 ρ）需要额外训练，等 E2E 测出速度后再定，不做也要在论文里如实说明。
  - 相关工作里引用 ESI-Bench 等工作，作为「过早下结论」现象的外部证据（只是写作，不花算力）。

- 2026-11-08 晚（Claude 审查）：首批 5 张审查图（logs/review/t01v2_checkpoint_single_02，均为 FloorPlan203 的直接类）没有阻塞问题，可以启动 E2E。只有直接类时，T3 的证据前逻辑测不到，Look 对出来后补跑 E2E 第 5 步。直接 Stay 两对的 X 位移很小，建池时再定最小位移。第二轮 Look 为 0：原因是放置只取最近的 6 个点、`PlaceObjectAtPoint` 只试一个点；另有 3 个场景恢复基线出错。T01v2 补充 6 只改搜索方式，不改阈值。

- 2026-11-08 18:25（人类批准）：T01v2 补充 7。候选放宽为 v0 画面中任意两个可拾取物体（优先同类），指令改为「Select the marked object closest to the <anchor>」，默认不再复制候选；下一轮扩到 40 个训练场景（厨房 1–20、客厅 201–220）。有效性标准、方法和指标都不变。产出世界对是第一优先，暂停新增附属工具。

- 2026-11-08 19:30：①（人类确认）「无锚点泄露」只用于 Look 类，直接类里的这类状态归为分叉、只计数，E2E 可按此重跑第 1 步；②（人类授权 Claude 决定）Stay 对位移生成时不限制，建池时丢弃 < 0.15 m 的，数量不够就保留并报告分布。新一批 5 张审查图（FloorPlan204）没有阻塞问题。

- 2026-11-08 19:45（Claude）：E2E 首对的诊断显示，两个世界各自落定导致椅子亚毫米级抖动（0.12 mm），画面差就超过 3。这很可能也是 Look 为 0 的主因之一。T01v2 补充 9：w1 只替换锚点位姿，其他物体沿用 w0；渲染时暂停物理。阈值不变。

- 2026-11-08 20:00（Claude 决定，已告知人类；人类此前授权「以做完论文为先，技术细节你定」）：当前视场角 100°、转头 45°，让「起点藏住、转头可见」的空间很小，物体也偏小。T01v2 补充 10：下一批采集用 100° 和 60° 两组对照，选择规则事先定好，阈值不变。

- 2026-11-08 21:00（人类提议、Claude 同意）：先集中解决 Look。T01v2 补充 12：暂停直接类；在 3 个场景 × 2 种视场角上做 20 分钟的 Look 专用小试，每轮给出漏斗表，针对卡住的那一关修改；Look ≥ 3 对且审查通过后，再恢复完整采集和 G01。

## Blockers（需要 lead / 人类拍板）
- VAGEN 是否发布训练好的 checkpoint 未确认；没有则 ctrl_vagen 需自训。
- AI2-THOR 能否做到确定性恢复与像素级输入比较；`SetObjectPoses` 需传完整物体列表（未列出的可移动物体会被删除）。由 T00 验证。
- H200 上 AI2-THOR 能否渲染、速度多少（T00 第 2 步）；计算节点是否有 Vulkan、能否联网。

## Gate log
- G1 研究合同：v3.2 方向已于 2026-11-07 获人类同意；载体待 D1 决定后补签
- D1 观察后端：P（起点附近视角预渲染），人类 2026-11-08 15:03 确认；研究合同升为 v3.3
- 去留检查点（11/9 晚）：
- G2（11/13 晚）：
- G3 初稿 + 第一轮评审：
- G4 终检：

### T01v2 首轮双区域并行（2026-11-08 18:20 UTC）
厨房 707189（4 GPU，20 分钟，FloorPlan1–12）与客厅 707191（2 GPU，40 分钟，FloorPlan201–212）已独立提交到 Express/autox，无节点限制，逐卡出图门控、每卡12 CPU。客厅没有等待厨房结果。18:12 UTC 的正式观察两者均 Q；2卡706354和8卡706682仍Q，4卡707054仍R，未触碰训练作业。首轮尚无试点结果，不能报告通过数。
预算记录 results/t01v2_round1_budget.json：已结束作业实际调度扣费 2.207777778 GPU·小时；未结束作业按完整walltime预留 38.000000000；合计 40.207777778/60。其中T01v2预留 2.666666667/8。已结束实际数不包含仍运行的4卡测速，其成本仍完整预留。queue_charge_rate=1，系PBS费率记账而非独立项目账单核销。
按人类最新规则，任何首轮分片任一情况<5对，下一轮启用同一完整SetObjectPoses列表重复source的候选复制，复制只发生在共同基线构造阶段，之后w0→w1仍只编辑X；合法放置、落定和独立回放标准不变。第二轮用独立入口准备，首轮源文件保持冻结。独立合并器新增身份/门控证据绑定与失败来源ledger处理，17个CPU fixtures通过（results/t01v2_merge_cpu_02.json）；这是程序验证，不是GPU科学结果。原始动作证据仍在scratch，当前没有声称数据可脱离scratch完全复核。

复制候选后端子步骤：独立模块已提交1312a8d；7项后端控制流+10项CPU政策fixtures均通过（results/t01v2_clone_backend_cpu_01.json、results/t01v2_clone_policy_cpu_01.json）。只验证程序，不含Unity实测/新增GPU用量；第二轮入口仍在接线，尚未提交第二轮作业。

第二轮提交与首轮汇总子步骤：56549a5新增独立clone入口、同一8/60预算与黑帧门控支持、显式已结束分片汇总器；95项v2 CPU集成测试通过（results/t01v2_clone_infra_cpu_01.json）。首轮源文件未改；数量缺失保持unknown，只有完整四类计数的<5规则触发clone。尚未提交第二轮GPU作业。

4卡测速707054变体已结束（results/t00b_scale4_worker_cache_retry_vllm04.json）：gpu_memory_utilization=0.4记录5个完成更新，但后续validation阶段达到2100.0秒watchdog，exit124，timed_out=True，acceptance_passed=False，cleanup_remaining非空。外层作业记录也exit124。失败原始训练/server/资源/逐帧证据已完整分片归档，必须保留；完成更新的计时只作失败运行诊断，不当成功测速或主实验成本依据。baseline此前20更新成功不改写。PBS最终占用时间、排队时间及扣费待下一次终态记账；预算仍按该作业原12 GPU·小时上限预留。

复制候选完整接线已提交f0e32dc：第二轮专用生成器先验证原场景、再创建共同候选基线，实际新name/lineage、合法pivot、落定、两次新进程重建均由独立合并器重算。一次clone准备失败即停止该scene并保留证据；无可用singleton结构才允许显式natural fallback。异常回放的真实清理错误不会丢失。每个分片归档仅JSON证明闭包（图像/掩码另存），含索引总计硬上限128MiB，单文件32MiB，达到上限即失败不导出合格对。117项v2 CPU集成fixtures全部通过（results/t01v2_clone_integration_cpu_01.json）；没有新增GPU运行或把fixtures视作试点数据。

4卡测速最终PBS记账：707054.hopper-m-02 F/exit124，分配walltime02:22:08，排队与eligible等待均943秒；实际分配用量9.475555556 GPU·小时，queue_charge_rate=1。results/t00b_monitor16_p4_accounting.json和原始PBS字段保留；Stageout_status=1表示stage-out成功，不能误称失败原因。新预算results/t01v2_after_scale4_budget.json：已结束实际11.683333333，未结束完整预留26.000000000，合计37.683333333/60；T01v2仍预留2.666666667/8。18:34 UTC正式观察厨房707189、客厅707191、2卡706354、8卡706682均Q，原样保留，监控进入下一有界轮次。

T01v2厨房首轮707189完成：12个分片四类均0；FloorPlan1/5/8/9原场景落定预检失败排除，2/3/4/6/11/12缺天然双候选。7/10各160次尝试，分别34/90次w0物理拒绝后出现126/70次证据存储guard失败，最终worker_failed。首轮逐卡gate通过、cleanup无错误、无黑帧；原始scratch及所有失败journal保留。正式汇总results/t01v2_kitchen_r1_summary.json记录全部分片；707189实际0.166666667 GPU·小时、等待1547秒（results/t00b_monitor17_p1_accounting.json）。已触发人类的<5复制规则，下一轮固定用原预检通过的FloorPlan2/3/4/6/7/10/11/12，每卡两scene；客厅707191继续原队列。第二轮每simulator原始scratch证据上限升至2GiB（273e364、14项针对CPU测试通过），portable证明仍128MiB；此为存储保护，不改变2cm/1cm、证据像素或D1验收。首轮真物理拒绝原因正在读原始证据诊断，未改写成通过。


2026-11-08 19:00 UTC 前的新预检调整：已拉取人类补充3d91bc1，旧规格原始摆放基准不重标。厨房707189首轮原始诊断已正式归档为results/diag_t01v2_kitchen_r1_physics.json；所查FloorPlan7/10首个w0物理拒绝的编辑偏差分别0.0453748958/0.0230046913m，均真实超过2cm，合法点和parent检查均通过，不能用新基准掩盖这类拒绝。旧客厅707191与刚提交但仍Q的旧预检复制作业707216已按最新人类要求取消，三段证据results/t01v2_stable_cancel_{preflight,qdel,postflight}.json确认两者从未启动；results/t01v2_stable_cancel_accounting.json正式计0 GPUh，等待时间未取得而保留null。原始提交回执与所有失败证据保留。后续厨房全部12scene采用落定快照+50步稳定窗口并合并clone，客厅201..212同样新入口；正在运行的作业未修改。新旧替代映射待新回执后补充。
诊断与旧复制提交使用的原producer c79e6ea因并发上游task更新曾被rebase；已用保留树的合并将原commit纳入main祖先，保证结果_meta.git_commit可从GitHub追溯，没有修改结果或日志。


新stable入口子步骤：代码6d078e4及兼容修复9538237实现落定后基准（N50，nativeRest或10次每物体单步<1mm，初始→当前位移上限0.5m）和完整M50窗口（相对落定基准≤1cm）。两world非编辑物体统一对settled/validated-clone共同基准检查，防止逐次累计漂移；X继续用PlaceObjectAtPoint返回位姿作2cm漂移基准。拒绝尝试只存简要JSON与小缩略图，合格pair/每种完整拒绝原因前三例保留raw；黑帧原图和UUID证据完整落盘后才允许86。上游补充2（1f1359e）已同步：Look首次观察只在X掩码并集内计算证据，不再要求两world各可见X；Direct两world的X均≥250像素；v0双方candidate均≤1.5m，原D1/状态恢复/见证门槛保留。METRICS_SPEC §1b仍标为提案，未用于本轮采集。
正式CPU首次161项中3项因Python3.10的encoding=locale文本view兼容失败，失败记录完整保留；修复后162项v2集成tests全部通过（results/t01v2_stable_integration_cpu_02.json），独立phase_watch回归9项通过。以上为程序验证，没有GPU科学结果。每场景预检、clone准备和物理拒绝分类的source-reported汇总入口已准备，缺失值保留unknown。先审计推代码，再提交厨房全部12scene与客厅201..212。


补充3执行检查点（635afb6）：已先取消新stable采集707229（厨房4卡20min，替代707216）与707230（客厅2卡40min，替代707191）；二者均Q/no allocation后qdel成功，终态F仍无运行字段。results/t01v2_supp3_cancel_accounting_02.json验证完整三段原始证据，实际均0GPUh；运行次数与排队等待时间缺失仍为null，未补为0。两份原始提交回执和所有取消/失败核账记录均保留。第一份核账失败是记录器使用Python -m入口而旧证据helper只识别文件入口；a6a319b仅新增严格的同一模块argv身份支持，6项CPU边界测试通过，未更改PBS原字段、结果或研究阈值。2卡706354、8卡706682训练任务未动。T01v2实际用量仍为首轮707189的0.1666666667GPUh。
新增采集暂停于代码整改阶段：锚点/clone源排除规定滚动类别及切片/碎裂变体，天然不移动候选不限；最多6个固定顺序源各fresh Controller，保留天然∪clone组合，组合轮换及连续10物理拒绝退组；raw上限降为存储停止而非worker停止，证据不足的科学候选不会被伪报成可验收数据；增加锚点w0通过/尝试及逐scene clone ledger。将先推代码和CPU测试结果再提交，预算仍8/60。


补充3代码与CPU验证（2026-11-08 19:42 UTC）：b40ac6d新增版本化search入口；规定滚动/切片/碎裂类别仅从锚点与clone源排除，天然不移动候选保留；固定首次预检的最多6源，各fresh Controller，首成功即停止，保留天然∪clone组及必要fresh natural fallback。各组合轮换，连续10次实际物理拒绝退组。所有Controller共用scene绝对deadline；每个组合首次查询原生放置点前先恢复共同基线，仍只使用已批准的状态恢复门槛，不要求旧像素零差。raw满后停止保存raw但继续测量与compact记录，证据不全不能入选；原始文件不删除，memory-only引用显式materialized=false；真实黑帧的完整小证明和清理仍须满足严格门槛才可86重提。
results/t01v2_search_integration_cpu_01.json：222项v2 CPU测试全部通过，覆盖补充3第1–5点及旧入口回归；results/t01v2_search_watch_cpu_01.json：9项phase_watch回归全部通过。results/t01v2_search_code_audit.json审计477文件、0 FAIL。均为CPU程序验证，无新增GPU科学结果；T01v2已结束实际仍0.1666666667GPUh。先推送本代码/测试检查点，再经最新8/60预算guard提交无节点限制的厨房与客厅独立分片；新旧作业映射在实际回执后记录。


补充3新作业已入队（2026-11-08 19:49 UTC）：旧707229→新707255.hopper-m-02（pilot_pairs_v2_search_kitchen_r4，4卡30min，FloorPlan1–12），旧707230→新707257.hopper-m-02（pilot_pairs_v2_search_living_r3，2卡60min，FloorPlan201–212）。旧两项已验证启动前取消0GPUh；新两项的代码8504b05在qsub前已推送。两项均Express/autox、rate=1、无host限制，逐卡255px分割出图gate、12CPU/GPU；每scene单一500秒绝对预算覆盖所有fresh Controller，物理/图像/见证阈值未改。实际回执及run_logged提交记录已保留。
19:47:58 UTC的results/t00b_monitor19_p5.json正式观察：707255/707257及保留的706354/706682均Q，无分配节点；没有取消或修改训练作业。results/t01v2_search_submitted_budget.json：已结束实际11.85，未结束完整预留27.3333333333，合计39.1833333333/60 GPUh；T01v2实际+预留4.1666666667/8，其中已结束实际仍0.1666666667。此处为PBS queue_charge_rate调度扣费账本，不伪称独立项目账单核销。
运行中只读快照工具b8a5729及6项CPU测试通过（results/t01v2_live_progress_cpu_01.json）；按确切run/scene/PBS/UUID/gate/submit绑定、保存完整journal字节前缀和原始结束时间，缺失unknown，不修改运行中来源也不把source-reported通过数当最终验收。首份results/t01v2_search_progress_01.json尚无场景观察，不能报告0对。后续每10分钟运行中的拒绝计数写本文件并推送；每个job最先完成的两个scene若合格数0则立即通知。报告将给出按锚点类别w0测量通过/尝试和每scene clone尝试结果。


人类确认（2026-11-08）：补充4的Look可解性要求已获批准，同时明确“不取消已经排队的作业”。707255/707257保持原PBS作业号，不qdel、不重提；已开始运行也保留。新增独立版本枚举全部现有一步observation_actions，每world至少一个X掩码≥250的动作；公共首观察证据阈值及物理/D1/见证门槛均不变。受保护METRICS_SPEC仍保留上游文字，本条记录人类批准。新代码/测试推送后仅切换后续启动的外层入口；已加载的旧producer保持不变。若任何作业先以旧版启动，其真实结果/用量保留，但缺少补充4证明的旧Look不会计入新的最终合格试点。并发文档已merge，原生产提交历史未重写；results/t01v2_supp4_merge_audit.json审计489文件0FAIL。


补充4调度子步骤：32f7bb5准备独立solvable outer及后续明确提交入口；原search Python producer和当前排队shell dispatch保持不变，尚未激活新版采集。外层config与hashed metrics同时记录solvability_version和scene_producer_module，phase_watch只有两者一致并且分片精确module/version匹配才允许新版黑帧重提；8/60 charged预算仍为同一账本。已加载旧outer不变，未来切换仅发生在新outer启动边界。results/t01v2_solvable_infra_cpu_01.json共32项新旧入口/预算/黑帧CPU测试全部通过；本子步骤没有GPU运行。新版全部观察动作证据、独立复核和Look额外审查视角尚在接线，完成且代码/测试推送后才切换现有队列入口，作业707255/707257不取消。


已合并4793625与50108ac的上游决定（补充5及D1=P/合同v3.3）。tasks/README.md与T02.md明确：T02的GPU视角表导出依赖T01v2试点通过审查，因此本轮8GPUh范围仍是构造阶段+补充4的一步可见性；完整P后端的D预算BFS、graph_mismatch、non_anchor_divergent和≥200缓存重渲染检查均留到T02，当前不能声称完整METRICS_SPEC已全部通过。补充5本轮如实报告场景/类别的candidate-a目标比例和Stay两侧数量；新版本生成会使用场景+尝试编号固定种子随机Flip方向、交替Stay两侧，不以合成平衡替代实际通过集统计。不会打断/取消现有排队作业。不同规范hash的历史来源将显式区分版本与适用阶段，原日志与结果不改写。上游合并后的审计497文件0FAIL（results/t01v2_backend_decisions_merge_audit.json）。


补充4+5核心代码1594e41完成：新solvable入口分别从两world恢复状态，调用既有observation_actions得到各自原生合法域，测量确定性并集中的全部本world合法动作；两个world的可见集合可不同，缺测unknown，不把不可达动作填成0像素。公共首步仍按原X-mask并集通道差>10且≥250验收，新进程见证/物理/D1门槛不变；所有新Look导出包含全动作原生证据与两world可见X审查视角，独立合并器从真实掩码/元数据/动作复算。补充5使用candidate_roles_sha256_view_round_robin_v1：固定seed0+scene+实际attempt编号SHA256决定Flip方向，Stay在展开视角后的实际候选尝试间交替两侧；不伪造缺失侧，也不把proposal顺序平衡当作合格数据已平衡。
results/t01v2_solvable_integration_cpu_01.json：275项v2 CPU集成测试全部通过（包括新旧生成/证据/角色/导出/历史规范/调度回归）。历史规范只允许明确已知的构造兼容版本并核验来源commit内真实文档hash；旧结果不改写。最终试点的验收范围显式为T01构造+补充4(a)，T02的P后端完整检查仍not_run，full_METRICS_compliance=null。旧Look的正向事后证书将由单独程序从真实已存首步见证帧验证，未测全动作不声称complete_catalog；一侧未看到保持unknown。
results/t01v2_search_watch_02.json已恢复正式调度观察：2026-11-08 20:26:30 UTC，707255/707257/706354/706682均Q、无分配节点，errors=[]，无自动重提。旧有界monitor20在SSH连接中断后只有已完成观察记录可用，未补造其最终结果。新代码先审计推送，再切换后续外层入口；现有作业不取消。


补充4+5入口已在2026-11-08 20:34:21.969904 UTC原子切换（107e677）：精确代码先推至GitHub，再替换现有search shell的后续outer入口；PBS作业没有取消、重提、hold或修改。已加载旧Python producer仍保持原样。程序生成logs/pbs/dispatch_activation_107e677b34a8.json记录发布先于替换及文件hash。20:35:35 UTC的results/t01v2_search_progress_02.json中24个scene均not_yet_observed，计数为null，不能写成0对。当前没有新增已观察到的GPU实测结果，T01v2已结束实际仍0.1666666667 GPUh。
已加入只读事后可解性工具：对旧Look来源先独立复核原构造证据，再从真实已存新进程首步见证中验证两world看到X的正向存在证明；无完整动作目录、不可见或域不明均保留unknown。新版全动作来源不重复降级为旧式未知。此工具不启动仿真器、不改旧结果；CPU验证随后运行。

事后检查CPU子步骤：results/t01v2_posthoc_cpu_01.json中7项测试全部通过，包括250像素边界、真实world/action绑定、缺掩码与旧move域不明确的unknown处理、归档hash和权限边界。仅验证程序，无新增GPU运行；正向旧证书尚未接入最终merger，若有旧来源合格候选再接入，不把诊断报告当最终试点验收。


T02 P后端CPU代码子步骤：新增D4视角表/五轮查表环境、同进程逐状态w0→w1导出适配器、共享/分叉/非锚点泄漏分类、基于success_eligible的标号、τ与本worldτ_seen、D预算共同前缀BFS和按目标位置回放见证。§5原生world/camera/domain/掩码证据绑定、≥200 distinct(pair,state)双world fresh缓存检查及全体阻断，§7固定pair_id Flip标签交换/逐scene Stay下采样/实际A与左半边比例已实现。配置max_steps=max_model_calls=D+1=5；合成表不会解锁真实训练。GPU导出、真实缓存验收、吞吐20倍与实测成本仍未运行，等待T01v2试点审查。提交代码后再运行CPU测试。
20:45:45 UTC的results/t01v2_search_progress_04.json仍为24scene not_yet_observed/errors=[]；上一03命令误用文件入口导致src导入失败，原失败log/meta完整保留，没有结果文件也不补写。已修正为模块入口。最新PBS现有观察两采集仍Q，原因为ngpus不足，无节点限制，作业未修改。进度工具将来源run_logged无时区结束时间明确标成source_time，不再误称UTC；原记录不改。

T02 P代码CPU验证完成：results/t02_P_table_cpu_01.json的44项查表/导出检查/缓存/平衡测试全部通过；results/t02_P_paired_env_regression_01.json的24项旧环境接口回归通过；监控时间标签回归6项通过。测试使用合成表，仅验证程序，不等同真实GPU导出、cache_matches_live或T02正式验收。当前D=4、max_steps/max_model_calls=5，真实训练开关保持关闭，T01v2两采集不取消。T02本子步骤GPU用量0。


T05新规格CPU代码子步骤：Q_commit在完整当前公开select标号集合上归一化；证据前无标号或无法解析输出按GAE回退，T3/T4以世界先验加权后双方使用相同优势。配对提示每轮比较完整图像字节与text token，允许精确重合后重新共享，真实生成token/模型调用单独记账。T4仅在自己t>=τ_seen的select生效，Stay要求两边满足；S1按独立episode/pair_instance保留逐步原始信用及加权c_u（paired只除2一次），包括实际0证据前步单元；S2按实际完成更新每20步记录。配置maximum_episode_rounds_from_T04=5、renormalize_over_select_candidates=true；必做T0/T2/T3/T4、T1应做。生产prompt/scorer、T04冻结配置、GPU冒烟仍未完成，training_execution_enabled=false。先提交后正式CPU测试。

T05新规格CPU验证：results/t05_P_credit_cpu_01.json的32项测试全部通过（包括select归一化/no-mark GAE、共享优势、逐轮实际prompt去重、自己的τ_seen边界、S1/S2/实际回合与token会计及torch梯度）。这是信用与记录层的合成CPU验证，不是T04公共配置冻结、真实模型接线或T05 GPU冒烟验收；GPU用量0。随后补窄接口测试验证真实T02合成查表环境输出能被配对回合层消费。


按人类最新7166628排队指令执行：707257（旧2卡客厅）、706354（2卡导航测速）、706682（8卡导航测速）已取消；logs/results中preflight→qdel→postflight三段证明均为Q无分配→qdel exit0→F无分配。results/t01v2_single_cancel_accounting.json正式验证三者实际均0 GPUh，run_count与提交到启动等待时间缺失保留null，不补0。707255厨房4卡保留；其他作业未动。并发monitor21曾先于专用取消核账触发通用核账，因无run_count/用量不能自行认定0而失败；该失败及外层monitor21完整保留，不影响随后通过的专用证据核账。monitor20不完整父log也保留，未补造最终结果。
客厅拟拆为201–204、205–208、209–212三个独立1卡单元，各walltime=4×500×1.30=2600秒（00:43:20），scene500/outer2570/gate180；无host，Express/autox，完整逐卡出图和黑帧证据规则不变。现有代码支持1卡，不改生成器阈值。先推本记录，再由现有8/60预算守卫逐项检查并提交。全部客厅及其黑帧后继结束后，如果fresh查询厨房707255仍从未启动且Q，再按同样方式取消并拆1–4/5–8/9–12。训练速度改由E2E自己的环境与设置测，不重提取消的导航测速。
E2E仅在至少2类各1合格且人类转达Claude图审无问题后触发（全1卡，≤4GPUh）；G01仅在每类≥5且同样图审条件后触发，先测每卡2/4/6场景进程吞吐再估算，超过20GPUh报告等批准。图像一产出即推送告知人类，尚未有合格图，不把CPU测试当触发条件。


单卡客厅拆分提交完成：旧707257→707307.hopper-m-02（201–204）、707309.hopper-m-02（205–208）、707310.hopper-m-02（209–212），各1GPU/12CPU/225GB、walltime00:43:20、autox无host。现有生成器带补充3/4/5及逐卡出图，源代码e23afc5已在首个qsub之前推送。每次提交均通过8/60双预算守卫并保留原始费率/预留快照、payload和回执；三项提交命令均exit0。取消旧训练与客厅的0用量证明已推，707255厨房保留。21:05:36 UTC的首个单卡进度快照尚无场景结果（当时只包含厨房12+已知第一组4），不能报0对；随后已扩展监控到24场景。未声称单卡已启动，待新PBS观察。排队表终态按真实qtime→stime给出，未启动取消项记未启动/null。


单卡首批门控与自动重提：707307/707309/707310均在hopper-22、同一GPU UUID GPU-5ff2e10b-a949-4c82-fa91-0e22f6075d56上发生真实黑帧，Initialize/Pass成功但RGB与分割全0；FIFO实际收到新图像包，非缓存旧帧，cleanup_errors=[]。Unity Player.log也确认选择Vulkan设备0的NVIDIA H200；静态核查未发现单卡映射错误，但不能由此断言节点/GPU/驱动根因。原始PNG、Player.log、UUID、失败outer/gate与用量均保留；没有场景搜索，所以不是两个场景0对。正式核账results/t01v2_single_watch_01_accounting.json：三项GPUh为0.0052777778/0.0055555556/0.0052777778，合计0.0161111111。

| 作业 | GPU数 | 提交到启动（分钟；原始秒） | 结局/后继 |
|---|---:|---:|---|
| 707307 | 1 | 3.93；236秒 | 黑帧86 → 707329 |
| 707309 | 1 | 2.38；143秒 | 黑帧86 → 707330 |
| 707310 | 1 | 1.22；73秒 | 黑帧86 → 707331 |

自动重提映射分别是pilot_pairs_v2_living_201_204_a00_r32d46927、pilot_pairs_v2_living_205_208_a00_r292586e2、pilot_pairs_v2_living_209_212_a00_r1ba34adc；三项均保持1GPU/2600秒/原场景/无host，通过同一8/60预算守卫。21:19:55 UTC的results/t01v2_single_watch_02.json观察厨房707255及707329/330/331均Q，errors=[]。results/t01v2_single_after_retry_budget.json记录已结束实际11.8661111111、未结束完整预留4.1666666667、合计16.0327777778/60；T01v2实际+预留4.3494444444/8，费率1，仍为PBS调度扣费而非独立账单核销。
早期图审checkpoint和T02→T05接线子步骤完成：10项checkpoint CPU tests及3项实际TableEnvironment→T02EpisodePort→collect_pair接口tests全部通过（results/t01v2_review_checkpoint_cpu_01.json、results/t02_t05_table_integration_cpu_01.json）。独立checkpoint复用完整来源/whole-journal/native验真、补4/5政策与去重；少量合格可以发布真实图和清单，不占用最终80/40固定输出路径，真正证据矛盾禁止发布。人类图审和正式T02验收仍unknown，数据数量门槛不自动等于批准。接口tests验证标号距离过滤、公开输入共享、分叉再返回共享时τ保留/独立τ_seen及五轮限制。GPU生成器与最终merger未改。


T01v2周期进度（21:29 UTC观察）：results/t00b_monitor22_p2.json中707255、707329、707330、707331均Q，无分配节点，errors=[]。results/t01v2_single_progress_02.json（21:25:56 UTC）覆盖当前厨房12与三个客厅后继12，共24scene均not_yet_observed，attempt/合格/拒绝计数保留null，不报0。仍无两个已结束场景可触发零合格告警，也无图审产物；后台有界观察继续，旧失败记录不删除。排队空档继续补齐原T02 CPU范围的离线导出producer与原生snapshot/clone来源适配，GPU导出/训练仍未触发。


T02来源适配CPU子步骤（dc59f73）：从已审计的最终pilot或独立checkpoint重查原scene、完整journal、preflight/hash及原有check_pair；保留完整world/clone快照和原始canonical agent命令（不拿readback浮点horizon重新当输入）；见证move按原生TeleportFull目标和前置GetShortestPathToPoint证明换算，不复用旧move编号。Look原完整catalog/固定reachable保持不变，Direct缺catalog明确None，必须由owned导出器发真实native query。缺runtime原图/元数据/掩码失败关闭；函数不推断人类图审或T02通过。results/t02_P_source_cpu_01.json中11项CPU测试全部通过；本子步骤GPU用量0，正式producer接线仍在静态复核。
周期计数：results/t01v2_single_progress_03.json（21:39:38 UTC）当前24scene仍not_yet_observed，错误为空，合格/拒绝/尝试均unknown。results/t00b_monitor22_p4.json（21:37:07 UTC）707255及707329/330/331均Q、无分配。尚未有两已结束scene可作零合格判定，审查图/人工审阅/E2E与G01数据触发均未发生。

### 2026-11-08 21:50 UTC — 单卡重试与 T02 导出入口

监控 t00b_monitor22_p5 记录 707329/707330/707331 在 hopper-25 黑帧 exit86，已通过原始图像、UUID、PBS 与清理证据校验后自动重提为 707341/707342/707343（仍为1卡、2600秒、无host限制）。旧记录保留。707341 的 gate 已在 hopper-27 通过。t01v2_single_progress_04 在 21:50:48 UTC 也已观察到厨房 707255 的四卡 gate 通过、场景子进程开始；该作业已启动，继续保留，不执行排队拆分条件。逐场景计数见该脚本结果，未完成场景不能解释为最终零对。

T02 CPU 准备新增 src/tools/t02_export.py：绑定真实单卡 PBS gate，独占 Controller 恢复原始/复制快照，导出后从保存字节重新加载并核验原生证据，区分科学拒绝、黑帧和清理/运行错误。独立静态审查已通过；现在先提交代码，再通过 run_logged 跑 CPU 入口测试。正式缓存≥200状态仍未执行，task_acceptance 为 null，训练未解锁；没有提交 T02 GPU 作业。

### 2026-11-08 21:55 UTC — 首两个客厅场景零合格告警

已立即通知人类：707341 的 FloorPlan201、202 均完成并报告四类0对；201为160次尝试，202为0次（无天然/复制候选并集）。201的原生 PlaceObjectAtPoint 放置阻挡占主要拒绝原因，另有 target_uniqueness_or_ratio=10、w0_physical=9；完整各原因计数保存在已结束场景结果与 t01v2_single_progress_05。该快照无错误：厨房1/7/10以及客厅203仍在运行，报告的部分尝试数分别13/11/7/9，均尚无通过；这不是最终计数。厨房3/4失败原因为 Common baseline state restoration failed before native proposal queries，保留失败记录；其他预检/候选排除原因逐场景保留。所有运行中作业未动。

独立静态复核发现后续同一组合yield后的尝试没有再次恢复共同基线，而 native PlaceObjectAtPoint 发生在完整 SetObjectPoses 之前，可能读取前次尝试遗留的世界。这是状态隔离缺口，但不能据此认定本次所有放置阻挡的原因。正在准备独立新版本修复及 CPU 回归；旧生产者及正在运行的源码保持不变，阈值不变。

T02导出入口11项CPU测试通过（t02_P_export_cpu_01）；审计 t01v2_single_audit_07 为583文件、0 FAIL。最新 PBS 记账预算 t01v2_single_after_retry2_budget：已结束实际扣费11.883333333333333 GPU小时，未结束按完整walltime预留4.166666666666666，合计16.05/60；仍运行作业尚未做最终结算。新增第二批三次黑帧实际0.017222222222222222 GPU小时，等候秒数1505/1456/1344，见 t00b_monitor22_p5_accounting。E2E/G01 尚未满足数据与审查双重触发条件。

### 2026-11-08 21:59 UTC — 首批非零候选与十分钟计数

t01v2_single_progress_06（21:59:49 UTC）无错误：FloorPlan203实时记录 direct_flip=3、direct_stay=1，其余两类0；厨房7为112次尝试、厨房10为49次，均暂无通过。随后203已因新见证预留时间不足退出，最终原始计数为3 Flip、2 Stay，均为直接类；场景失败/未完状态保留，独立逐对检查 t01v2_checkpoint_single_02 决定其已完成证据能否使用。先推原始审查图，未把生产者自报pass当作最终验收。

另查明厨房3的共同基线恢复失败：物体位置恢复≤1cm为true、RGB诊断maxdiff=0，agent JSON仅有+0.0/-0.0差异但字节比较不一致；厨房4除正负零外y也从0.9009998440742493变为0.9009997844696045，RGB诊断maxdiff=2。保持原始检查结果与硬门槛不变，不补写通过。源证据为各场景 proposal_setup 原始文件。

### 2026-11-08 22:10 UTC — 已复核5对，等待图审转达

t01v2_checkpoint_single_02 独立复核通过 FloorPlan203 的5对（direct_flip=3，direct_stay=2，Look两类均0）；五张审查图已于0d7768e推送并通知人类。E2E数据条件为true，G01为false；尚未收到人类转达Claude看图无问题，未启动E2E/G01。该场景原运行因见证时间预留不足exit1，仍保留其95次尝试和失败结尾，已完成的五对逐对证据独立有效。

t01v2_single_progress_07（22:10:06 UTC）记录厨房12场景均结束且四类0；其中1/7/10因场景期限退出，尝试数24/135/143；3/4为恢复失败；其余为预检或候选排除。客厅201/202/203/204尝试数160/0/95/160，只有203产出上述5对；209仍运行，57次尝试暂无通过。已结束场景与部分观察分别记录，不把后续未观察场景记成0。

707255最终0.8088888888888889 GPU小时、等待7451秒（t00b_monitor23_p3_accounting），已结束，不再触发排队厨房拆分。707342在hopper-17 gate黑帧86，watch04已按完整证据、清理与预算检查自动重提707350，旧新对应及UUID保留；707343在hopper-22的另一分配上gate通过，继续收集209–212。因此目前不能按节点名推定所有卡都黑或都正常。

### T02 CPU 子步骤 — 正式缓存检查入口

新增正式 >=200 distinct(pair,state) 缓存检查 CLI：完整显式导出集合、固定seed0采样、每对新Controller、同相机逐状态双世界渲染，重读PNG/NPZ和原生动作证据后认证；任何失败/缺失/清理未知都阻断整个流程，不删除失败状态或更换抽样。仅完成CPU代码和独立静态审查，真实GPU检查尚未运行，未宣称T02验收。另修复每对Unity日志路径独立，避免后一个Controller覆盖前一个Player.log；不影响已存实验数据。先提交，再run_logged测试。

### 2026-11-08 22:16 UTC — T02 正式缓存 CPU 检查通过

代码44b9ba1先推送后测试：t02_P_cache_cpu_01 的11项、t02_P_export_cpu_02 的12项、t02_P_table_regression_02 的55项全部通过。实际缓存GPU检查未运行；测试夹具不能作为真实 cache_matches_live 证明，本子步骤GPU用量0。

### 2026-11-08 22:20 UTC — 逐次共同基线隔离修复

新增独立 isolated 生产入口：每次尝试先恢复并验证同一个完整共同基线，然后由旧构造代码设置v0并执行原生放置。w0→w1之间不插入任何恢复/reset/agent动作；物理、像素、可解性、类别、seed、角色顺序和原始恢复判定均不变。额外保存真实恢复及第一条w0放置前连续原生动作引用，独立校验器复核；缺证据不得通过。普通单次恢复失败记实际拒绝，不伪造w0物理测量。原有solvable入口和正在运行作业不变，新入口只有明确提交时才启用。

新入口同样使用Express/autox、原8/60预算锁、逐卡gate和严格黑帧证明；旧作业不会被标记成新版本。代码、证据校验、调度兼容均已独立静态审查通过，先提交推送再跑CPU回归。本修复不能声称已改善通过率，GPU尚未测。

### 2026-11-08 22:21 UTC — isolated CPU 验证通过，准备小分片实测

代码4d2b736已先推送；t01v2_isolated_cpu_01（206项）、t01v2_isolated_infra_cpu_01（82项）、t01v2_isolated_merge_cpu_01（41项）、t01v2_isolated_watch_cpu_01（9项）全部通过。尚未实测GPU效果。下一步只提交固定厨房1–4的1卡分片，2600秒walltime（4×500秒加30%），原T01v2 8/整体60预算锁；其余在跑分片继续。没有取消或替换任何运行作业，也不将旧结果标为新入口产生。

最新逐场景计数见 t01v2_single_progress_08；物理拒绝、逐场景预检和每场景复制尝试的中期汇总为 t01v2_single_stable_summary_01，明确区分未知与已测0。已复核审查图仍以 t01v2_checkpoint_single_02 为准，E2E 等人类转达图审。

### 2026-11-08 22:26 UTC — isolated 单卡作业707366已入队

代码和CPU审计均先推送至e2f7e23，再由 t01v2_submit_isolated_kitchen_001_004_a00 提交707366（1GPU、厨房1–4、43:20、无host、Express/autox、CFP04-SF-125）；提交器已在同一锁内检查T01v2 8 GPU小时与整体60上限，排队按完整walltime预留。新旧生产者/作业证据独立保留，尚无修复效果结论。

补充5偏差：首批独立检查点 direct_flip 的3对，在rho=0.5时正确目标A概率0.5；direct_stay的2对均在A侧，B侧0，正确目标A概率1.0。Look两类无样本，比例为null；未做T02池平衡，不能宣称平衡通过。707350最终实际用量0.10888888888888888 GPU小时、排队161秒，见 t00b_monitor25_p1_accounting。

### 2026-11-08 22:37 UTC — single-GPU closure and bounded isolated check

All original kitchen/living allocations have ended. The latest full-input early checkpoint (`t01v2_checkpoint_single_03`) independently verifies the same 3 direct_flip and 2 direct_stay pairs, with zero Look pairs, but its whole-input status is invalid_evidence: old kitchen FloorPlan7 and FloorPlan10 have terminal journal rows beyond the final reported attempt counters. Their unchanged original files and failed checkpoint are retained; this is not a reason to waive source checks. The previously published five-image checkpoint `t01v2_checkpoint_single_02` remains independently valid and awaits the human relay of Claude's image review. E2E and G01 are not authorized by a data threshold alone.

`t01v2_single_stable_summary_02` records every old-round scene's preflight, clone attempts, and source-reported rejection distributions (missing measurements remain unknown, not zero); it does not independently accept pairs. The 22:37 progress snapshot `t01v2_single_progress_10` observes 20 completed attempts in the new FloorPlan1 run, all rejected, with 20 measured w0 physical passes across the recorded anchors. This partial scene is not a completed zero-pair result. The other three new scenes were not yet observed.

New isolated job 707366 is running on hopper-21; its per-card real-render gate passed. It retains the original restoration and acceptance thresholds. The logged budget `t01v2_single_budget_02` records 13.437777777777777 scheduler-charged GPU hours completed plus 0.7222222222222222 reserved, total 14.16 / 60; the T01v2 subbudget is 2.4766666666666666 / 8 including the reservation. These use PBS's recorded charge rate and allocation time, not an independently reconciled project credit bill. No additional GPU job has been submitted.

### 2026-11-08 22:47 UTC — bounded isolated check ended; queue table

707366 ended F/exit1 with clean recorded cleanup and no black frame. Actual allocation/recorded-rate charge is 0.22694444444444445 GPUh and qtime-to-start is 281 seconds (`t00b_monitor27_p2_accounting`). FloorPlan1 reports 23 attempts, zero passes and parent deadline; FloorPlan2 reports no eligible natural/clone union, zero attempts; FloorPlan3 and FloorPlan4 report common-baseline restoration failure before proposal queries, zero attempts. The first-two-scenes-zero condition was immediately relayed to the human. `t01v2_isolated_summary_01` retains each scene's preflight/clone history and unknown missing physical measurements; `t01v2_single_progress_11` is the end-of-run counter snapshot. No running T01v2 job remains in this round.

`t01v2_pbs_wait_cpu_01` passes 11 CPU fixtures. `t01v2_pbs_wait_table_02` and its generated Markdown at `logs/reports/t01v2_pbs_wait_table_02.md` report 15 explicitly supplied final/cancelled jobs, preserving every black retry and unknown cancellation waits. It is a completed-job table, not the whole project budget. `t01v2_single_budget_03` records completed scheduler charges 13.664722222222222 / 60 GPUh, no pending reservation, with T01v2 1.9813888888888889 / 8. The early-checkpoint invocation `t01v2_isolated_checkpoint_01` failed because its name did not have the tool-required prefix; its log/meta remain preserved, and the corrected invocation is `t01v2_checkpoint_isolated_01`.

The next code-only corrections address two independently identified implementation issues: a bounded terminal receipt prevents stale progress counters from being paired with a newer final journal; a pure deterministic permutation interleaves existing native placement/view combinations instead of exhausting all views of the earliest geometry. Neither correction changes acceptance thresholds, eligible points/views, scene split, role alternation, or the seed-0 actual-attempt assignment formula. Search reordering can change which proposal receives an attempt index; preservation means the canonical pre-orientation proposal/view domain, not an identical time-limited sample. No improved GPU success rate is claimed. New GPU submission waits for the corrected code, CPU checks, audit and push.

### 2026-11-08 22:59 UTC — next collector code and CPU checks

Producer changes are pushed through 003182a. `t01v2_coverage_terminal_cpu_01` passes 234 generator/terminal/isolation tests; `t01v2_coverage_merge_cpu_01` passes 46 merger tests; `t01v2_coverage_watch_cpu_01` passes 9 retry tests; `t01v2_coverage_infra_cpu_02` passes 82 dispatch/provenance tests. The first infrastructure run failed one historical-fixture check because it built an old producer from new coverage-tagged configuration; the fixture now constructs the actual frozen solvable policy. That failed run remains committed with the successful rerun. The standalone 8 coverage-domain tests also pass. All are CPU program tests, not GPU scientific results.

The isolated entry now shortens its worker's soft deadline by 30 seconds inside the existing scene allowance so it can seal the complete journal and finish cleanup before the unchanged parent hard deadline. A verified terminal receipt binds full returned measurements and journal bytes; the parent checks them again after owned cleanup. Missing/inconsistent receipt retains explicitly nonfinal progress and exits 1. Historical counters and journals are untouched. The optional new coverage policy is bound in both configuration and hashed metrics; historical sources without those tags remain valid under their original checks.

Next submission plan: three 1-GPU living shards with the original disjoint 201–204, 205–208, 209–212 ranges, each 2600 seconds walltime, 500 seconds per scene, 12 CPU cores/225GB, Express/autox with per-card render gate and no host restriction. Every submission must pass the existing 8/60 charged-budget guard; actual job IDs and updated reservations will be logged after the receipt. This continues T01v2 only. Human review of the existing five images remains pending for E2E; G01's four-class threshold is unmet.

### 2026-11-08T23:18:24.539140+00:00 — coverage batch counters

New receipts: 707377 (201–204), 707378 (205–208), 707379 (209–212), each 1 GPU and 2600 seconds. All three real-render gates pass; observed nodes are hopper-15, hopper-21, hopper-29. The logged budget t01v2_coverage_budget_01 records 13.664722222222222 completed plus 2.1666666666666665 reserved = 15.831388888888888 / 60; T01v2 including reserves is 4.148055555555556 / 8. This snapshot predates final replacement of reservations. Later t00b_monitor28_p2_accounting records 707378 ended exit0, actual0.16361111111111112GPUh and67seconds queuewait; the other two remain running. First-two-scenes-zero alerts have been relayed for201/202 and205/206 in this coverage round.

The following copies recorded source/live counters from t01v2_coverage_progress_02; source-reported passes are not independent pair acceptance. Missing scenes/measurements remain unknown.

{"scene": "FloorPlan201", "status": "completed_search", "attempts": 160, "passed_by_class": {"direct_flip": 0, "direct_stay": 0, "look_flip": 0, "look_stay": 0}, "rejection_counts": {"RuntimeError: PlaceObjectAtPoint: Spawn area not clear (CoffeeTable|-02.33|+00.00|+04.92) is in the way), can't place object at that point": 103, "target_uniqueness_or_ratio": 16, "RuntimeError: PlaceObjectAtPoint: Spawn area not clear (Bowl|-02.64|+00.60|+04.61) is in the way), can't place object at that point": 27, "w0_physical": 8, "RuntimeError: PlaceObjectAtPoint: Spawn area not clear (Bowl|-02.45|+00.60|+04.79) is in the way), can't place object at that point": 5, "w1_physical": 1}}

{"scene": "FloorPlan202", "status": "completed_search", "attempts": 0, "passed_by_class": {"direct_flip": 0, "direct_stay": 0, "look_flip": 0, "look_stay": 0}, "rejection_counts": {}}

{"scene": "FloorPlan203", "status": "observed_partial", "attempts": 79, "passed_by_class": {"direct_flip": 0, "direct_stay": 0, "look_flip": 0, "look_stay": 0}, "rejection_counts": {"RuntimeError: PlaceObjectAtPoint: Spawn area not clear (Book|-03.72|+00.76|-00.64) is in the way), can't place object at that point": 5, "v0_candidates_visibility": 43, "w0_physical": 12, "RuntimeError: PlaceObjectAtPoint: Spawn area not clear (Chair|-03.42|+00.01|-00.32) is in the way), can't place object at that point": 10, "w1_physical": 5, "RuntimeError: PlaceObjectAtPoint: Spawn area not clear (DiningTable|-04.30|+00.01|-00.38) is in the way), can't place object at that point": 1, "RuntimeError: PlaceObjectAtPoint: Spawn area not clear (Chair|-03.90|+00.01|-00.83) is in the way), can't place object at that point": 3}}

{"scene": "FloorPlan204", "status": "not_yet_observed", "attempts": null, "passed_by_class": null, "rejection_counts": null}

{"scene": "FloorPlan205", "status": "completed_search", "attempts": 0, "passed_by_class": {"direct_flip": 0, "direct_stay": 0, "look_flip": 0, "look_stay": 0}, "rejection_counts": {}}

{"scene": "FloorPlan206", "status": "completed_search", "attempts": 0, "passed_by_class": {"direct_flip": 0, "direct_stay": 0, "look_flip": 0, "look_stay": 0}, "rejection_counts": {}}

{"scene": "FloorPlan207", "status": "completed_search", "attempts": 0, "passed_by_class": {"direct_flip": 0, "direct_stay": 0, "look_flip": 0, "look_stay": 0}, "rejection_counts": {}}

{"scene": "FloorPlan208", "status": "completed_search", "attempts": 160, "passed_by_class": {"direct_flip": 0, "direct_stay": 0, "look_flip": 0, "look_stay": 0}, "rejection_counts": {"RuntimeError: PlaceObjectAtPoint: Spawn area not clear (Chair|-03.13|+00.00|-04.42) is in the way), can't place object at that point": 8, "neither_hidden_shared_prefix_nor_direct_evidence": 35, "w0_physical": 21, "RuntimeError: PlaceObjectAtPoint: Spawn area not clear (DiningTable|-03.10|+00.00|-03.61) is in the way), can't place object at that point": 26, "RuntimeError: PlaceObjectAtPoint: Spawn area not clear (Statue|-03.04|+00.78|-03.75) is in the way), can't place object at that point": 23, "RuntimeError: PlaceObjectAtPoint: Spawn area not clear (Chair|-03.18|+00.00|-02.81) is in the way), can't place object at that point": 27, "v0_candidates_visibility": 8, "RuntimeError: PlaceObjectAtPoint: Spawn area not clear (Chair|-02.69|+00.00|-03.30) is in the way), can't place object at that point": 7, "w1_physical": 5}}

{"scene": "FloorPlan209", "status": "incomplete", "attempts": 116, "passed_by_class": {"direct_flip": 0, "direct_stay": 0, "look_flip": 0, "look_stay": 0}, "rejection_counts": {"w0_physical": 42, "RuntimeError: PlaceObjectAtPoint: Spawn area not clear (Box|-02.36|+00.25|-01.65) is in the way), can't place object at that point": 42, "target_uniqueness_or_ratio": 18, "RuntimeError: PlaceObjectAtPoint: Spawn area not clear (Watch|-02.79|+00.58|-02.07) is in the way), can't place object at that point": 2, "RuntimeError: PlaceObjectAtPoint: Spawn area not clear (FPSController) is in the way), can't place object at that point": 4, "RuntimeError: PlaceObjectAtPoint: Spawn area not clear (CoffeeTable|-02.47|+00.00|-02.49) is in the way), can't place object at that point": 4, "w1_physical": 2, "v0_candidates_visibility": 1, "deadline_or_interrupt": 1}}

{"scene": "FloorPlan210", "status": "observed_partial", "attempts": 40, "passed_by_class": {"direct_flip": 0, "direct_stay": 0, "look_flip": 0, "look_stay": 0}, "rejection_counts": {"RuntimeError: PlaceObjectAtPoint: Spawn area not clear (ArmChair|-04.38|+00.01|+01.40) is in the way), can't place object at that point": 7, "w0_physical": 13, "RuntimeError: PlaceObjectAtPoint: Spawn area not clear (FPSController) is in the way), can't place object at that point": 6, "RuntimeError: PlaceObjectAtPoint: Spawn area not clear (Statue|-04.37|+00.41|+00.35) is in the way), can't place object at that point": 7, "RuntimeError: PlaceObjectAtPoint: Spawn area not clear (SideTable|-04.29|+00.01|+00.41) is in the way), can't place object at that point": 1, "v0_candidates_visibility": 6}}

{"scene": "FloorPlan211", "status": "not_yet_observed", "attempts": null, "passed_by_class": null, "rejection_counts": null}

{"scene": "FloorPlan212", "status": "not_yet_observed", "attempts": null, "passed_by_class": null, "rejection_counts": null}


Read-only Look diagnostic CPU01 exposed a bound default-root bug in the temporary-root fixtures; explicit evidence-root passing fixes it and CPU02 passes8tests. Both runs are retained. This was a tooling-test failure, not a change to the dataset or acceptance criteria.


## 2026-11-08 23:53 UTC — Supplement 6/7 preparation and completed coverage
Merged human-approved 0e783c3. Next collection will use both supplements 6 and 7, natural marked A/B, 40 training scenes, and the approved T01v2 16 charged GPU-hour subbudget. Existing budget guard also enforces the separate E2E 4-hour cap (conservatively counted within the shared 60-hour ledger); no new accounting/queue tools are being added.
707377–707379 finished without cancellation. Independently checked coverage pairs: {"direct_flip": 1, "direct_stay": 4, "look_flip": 0, "look_stay": 0}. Review images: logs/review/t01v2_checkpoint_coverage_01. Look remains zero. Actual allocated GPU-hours {"707377.hopper-m-02": 0.36194444444444446, "707378.hopper-m-02": 0.16361111111111112, "707379.hopper-m-02": 0.4075}; scheduler rate 1. Queue waits (seconds) {"707377.hopper-m-02": 159, "707378.hopper-m-02": 67, "707379.hopper-m-02": 34}. All failed/time-limited attempts remain in the source records.
Required saved-state diagnosis results/t01v2_restore_diag_supp6_01.json has no unknowns/errors: FP3 and FP212 fail only exact encoded agent equality for signed zero in rotation.x/horizon; FP4 also has agent y change -5.960464477539063e-08 m. All three retain identical stable-name inventory and maximum movable/pickupable position displacement 0 m. Some location-derived object IDs changed, but this is not the failing predicate. This cause was reported to the human before repair. New baseline establishment will replay a recorded canonical native agent setup; no acceptance tolerance or historical evidence is changed.
Logged restore fixtures, budget boundary fixtures and parallel infrastructure checks pass. Preparation audit: 722 files, 0 FAIL. E2E approved five Direct pairs remain the fixed first smoke input; T3 evidence-before logic is not covered until Look data exist.

E2E code 3e840da pushed; logged smoke entry/scope fixtures and 55 table contract CPU tests passed. Step 1 uses the already approved five FP203 Direct pairs, D=4 and exactly 20 fresh paired states. Initial one-GPU walltime cap is 1h (protective cap, not measured export runtime); total E2E cap remains 4 charged GPU-hours. Formal T02 200-state minimum is unchanged.


## 2026-11-09 00:05 UTC — Supplement 6+7 collection launch
Code 0bc2617 and 1935629 implements both supplements: natural marked candidates (same class first), no default cloning, exact original thresholds, unique nonrolling anchor, same-Controller safe native point retries capped at 40, actual successful tokens, Look/Direct projection pools with global alternation, actual Stay XZ displacement, and canonical initial native camera setup before unchanged N50/M50 preflight. Core/merger/source/PBS CPU suites pass; latest audit 734 files, 0 FAIL.
First fixed shards: kitchen 1–2 and living 201–202, each one GPU without host pin. Per scene 700s; 1650s requested walltime per two-scene job, outer deadline 1620s and gate cap180s. Planning basis only: prior coverage slowest scene mean was 3.3345813737088896s/attempt (FP204); reserve about190s of previously observed setup overhead; (100*3.3346+190)*1.3 is below700s. New safe-point retries change cost, so these first four scenes also establish actual next-round timing; <100 measured trials will be reported and addressed, never filled in. The full fixed forty scenes are 1–20 and201–220. At these request caps twenty two-scene allocations reserve9.1667GPUh, within the remaining16-hour pilot budget; this is a cap forecast, not a result.
E2E707395 is submitted separately with1GPU/1hcap, fixed approved old5Direct pairs; no new T01v2 collection was submitted before both supplements were committed and CPU-checked.


## 2026-11-09 00:16 UTC — marked collection queued; E2E stops at export
New supplement6+7 one-GPU receipts: 707403 kitchen1–2, 707475 living201–202; each1650s walltime, no host restriction. Both recorded as queued at latest read. Existing8-hour-era outputs are preserved; next collection uses the human-approved16-hour subbudget. No new ancillary monitoring/accounting tools added.
E2E707395 finished exit1 on hopper-24 after84s allocation:0.023333333333333334 charged GPUh at recorded rate1; waited863s. Gate passed. First reviewed FP203 pair floorplan203_00038 hits non_anchor_divergent in D4 export: both anchor masks absent, raw RGB max channel difference>3. This predicate alone does not identify physics versus shadows. No complete table or fresh20-state check; steps2–5 unstarted. E2E stopped for review per task; see tasks/E2E.report.md. T01v2 continues. Audit742files0FAIL; full failed run preserved.


## 2026-11-09T00:28:32.493906+00:00 — first marked-collection live counts
Read human-approved4e010a3: Direct hidden-anchor RGB differences>3 are forked states (count only), Look restriction unchanged. Stay displacement remains diagnostic during generation; pool-only0.15m filtering/fallback is deferred to pool construction. Human confirms coverage five images reviewed without blocking issues.
Existing helper invocation t01v2_marked_live_01 failed on a relative archive path; its records remain. Corrected absolute-path invocation t01v2_marked_live_02 passed. Following are snapshots of complete producer journal rows, not independent final acceptance:
{"discard_reason_counts": {"neither_hidden_shared_prefix_nor_direct_evidence": 4, "target_uniqueness_or_ratio": 1, "w1_physical": 2}, "name": "pilot_pairs_v2_marked_kitchen_1_2_01_j707403_hopper_m_02_g0_floorplan1", "observed_complete_attempts": 7, "observed_reported_passed_by_class": {"direct_flip": 0, "direct_stay": 0, "look_flip": 0, "look_stay": 0}, "scope": "producer_reported_live_not_independent_acceptance", "source": {"file": "logs/runs/t01v2_marked_live_02/snapshots/pilot_pairs_v2_marked_kitchen_1_2_01_j707403_hopper_m_02_g0_floorplan1.jsonl", "sha256": "8d6266e95f4e3fceb62e49272a38d87fe507ced647bbecc33e2159f45e001016"}, "w0_by_anchor": {"Book": {"measured_attempts": 1, "measurement_unavailable": 0, "passed": 1, "rejected": 0}, "Bowl": {"measured_attempts": 1, "measurement_unavailable": 0, "passed": 1, "rejected": 0}, "Bread": {"measured_attempts": 2, "measurement_unavailable": 0, "passed": 2, "rejected": 0}, "ButterKnife": {"measured_attempts": 1, "measurement_unavailable": 0, "passed": 1, "rejected": 0}, "Cup": {"measured_attempts": 1, "measurement_unavailable": 0, "passed": 1, "rejected": 0}, "Fork": {"measured_attempts": 1, "measurement_unavailable": 0, "passed": 1, "rejected": 0}}}
{"name": "pilot_pairs_v2_marked_kitchen_1_2_01_j707403_hopper_m_02_g0_floorplan2", "status": "not_yet_observed"}
{"discard_reason_counts": {"anchor_visibility_below_250_at_t0": 1, "neither_hidden_shared_prefix_nor_direct_evidence": 8, "w0_physical": 4}, "name": "pilot_pairs_v2_marked_living_201_202_01_j707475_hopper_m_02_g0_floorplan201", "observed_complete_attempts": 14, "observed_reported_passed_by_class": {"direct_flip": 0, "direct_stay": 1, "look_flip": 0, "look_stay": 0}, "scope": "producer_reported_live_not_independent_acceptance", "source": {"file": "logs/runs/t01v2_marked_live_02/snapshots/pilot_pairs_v2_marked_living_201_202_01_j707475_hopper_m_02_g0_floorplan201.jsonl", "sha256": "7090580933f0b1d946d7352bf373f4ea680ad72e554a7720d344bf593e31b7a9"}, "w0_by_anchor": {"Bowl": {"measured_attempts": 3, "measurement_unavailable": 0, "passed": 3, "rejected": 0}, "CreditCard": {"measured_attempts": 4, "measurement_unavailable": 0, "passed": 4, "rejected": 0}, "KeyChain": {"measured_attempts": 3, "measurement_unavailable": 0, "passed": 0, "rejected": 3}, "Newspaper": {"measured_attempts": 1, "measurement_unavailable": 0, "passed": 1, "rejected": 0}, "Plate": {"measured_attempts": 1, "measurement_unavailable": 0, "passed": 1, "rejected": 0}, "RemoteControl": {"measured_attempts": 1, "measurement_unavailable": 0, "passed": 1, "rejected": 0}, "Watch": {"measured_attempts": 1, "measurement_unavailable": 0, "passed": 0, "rejected": 1}}}
{"name": "pilot_pairs_v2_marked_living_201_202_01_j707475_hopper_m_02_g0_floorplan202", "status": "not_yet_observed"}
707403 and707475 continue untouched. T01v2 GPU budget remains16, shared60; pending/running reservations stay at full requested walltime until final accounting. Redundant in-memory pool copying and large repeated query arrays are being removed for later jobs; no acceptance threshold change.


## 2026-11-09T00:38:35.722504+00:00 — supplement9 merge and live counters
Merged da61b8f preserving both histories. No new GPU submission;707403/707475 remain running. Updated counters from logged t01v2_marked_live_03 (source-reported live, not final acceptance):
{"discard_reason_counts": {"neither_hidden_shared_prefix_nor_direct_evidence": 5, "target_uniqueness_or_ratio": 1, "w1_physical": 2}, "name": "pilot_pairs_v2_marked_kitchen_1_2_01_j707403_hopper_m_02_g0_floorplan1", "observed_complete_attempts": 8, "observed_reported_passed_by_class": {"direct_flip": 0, "direct_stay": 0, "look_flip": 0, "look_stay": 0}, "source": {"file": "logs/runs/t01v2_marked_live_03/snapshots/pilot_pairs_v2_marked_kitchen_1_2_01_j707403_hopper_m_02_g0_floorplan1.jsonl", "sha256": "557a8b98a6e9d694ca08a9cc0923541bd6891e63ebf3605969b92456d6b4fd57"}}
{"discard_reason_counts": {"w0_physical": 5, "w1_physical": 3}, "name": "pilot_pairs_v2_marked_kitchen_1_2_01_j707403_hopper_m_02_g0_floorplan2", "observed_complete_attempts": 8, "observed_reported_passed_by_class": {"direct_flip": 0, "direct_stay": 0, "look_flip": 0, "look_stay": 0}, "source": {"file": "logs/runs/t01v2_marked_live_03/snapshots/pilot_pairs_v2_marked_kitchen_1_2_01_j707403_hopper_m_02_g0_floorplan2.jsonl", "sha256": "920372ebce41f3836c1ea46a9e9c2f1602ea82852b84ab9d28ea0edc33a8f4a1"}}
{"discard_reason_counts": {"anchor_visibility_below_250_at_t0": 1, "neither_hidden_shared_prefix_nor_direct_evidence": 10, "w0_physical": 5}, "name": "pilot_pairs_v2_marked_living_201_202_01_j707475_hopper_m_02_g0_floorplan201", "observed_complete_attempts": 17, "observed_reported_passed_by_class": {"direct_flip": 0, "direct_stay": 1, "look_flip": 0, "look_stay": 0}, "source": {"file": "logs/runs/t01v2_marked_live_03/snapshots/pilot_pairs_v2_marked_living_201_202_01_j707475_hopper_m_02_g0_floorplan201.jsonl", "sha256": "13f5e7a2f1bb4ee72f452251580b976fa15d4001818f4fd1f01078401f942192"}}
{"discard_reason_counts": {"neither_hidden_shared_prefix_nor_direct_evidence": 6, "one_action_evidence_unavailable": 1, "target_uniqueness_or_ratio": 11, "w0_physical": 12, "w1_physical": 3}, "name": "pilot_pairs_v2_marked_living_201_202_01_j707475_hopper_m_02_g0_floorplan202", "observed_complete_attempts": 33, "observed_reported_passed_by_class": {"direct_flip": 0, "direct_stay": 0, "look_flip": 0, "look_stay": 0}, "source": {"file": "logs/runs/t01v2_marked_live_03/snapshots/pilot_pairs_v2_marked_living_201_202_01_j707475_hopper_m_02_g0_floorplan202.jsonl", "sha256": "628676d3456d5344b0d234dc50ec0bc224a706d9a55d288fc323705a01e017d5"}}
Independent checkpoint t01v2_checkpoint_marked_initial_01 verifies1DirectStay fromFP201, noLook; review image in logs/review/t01v2_checkpoint_marked_initial_01. Supplement8 table58 andscope12CPUtests passed; the first T02fixture run failed because fixture lackedkind, preserved andfixturecorrection committed. Recheck still required after merge. Audit762files0FAIL.
Existing raw scene results for FP1/FP201 contain large repeated native coordinate/pose ledgers (over50MiB). Original files remain unchanged on Hopper; byte-identical gzip copies are committed as results/<original>.json.gz after gzip|cmp validation in t01v2_marked_compress_01. Fresh checkout reviewers must decompress each .json.gz to its corresponding .json before source audit/review (gzip -cd FILE.json.gz > FILE.json). This is lossless transport, not a rewritten or dropped result. Producer compact-record fix is being integrated with supplement9; no new auxiliary tool.


## 2026-11-09 00:44 UTC — first marked batch ended; supplement9 in progress
707403/707475 ended naturally exit1 without cancellation or black frame. FP1:8 measured attempts,0pairs; FP2:10,0; FP201:17,1DirectStay; FP202:44,0. All ended on scene deadline and none reached100; this is unmet, not filled in. First-two-kitchen-scenes-zero alert was relayed immediately after final observation.
Actual charge in t01v2_marked_final_accounting_01:707403=0.3863888888888889GPUh(wait513s),707475=0.38722222222222225GPUh(wait145s),rate1. t01v2_supp9_budget_01: T01v2 3.6880555555555556/16, E2E0.023333333333333334/4, shared15.39472222222222/60; zero unfinished reserve at capture.
Supplement9 exact requested frozen pose lists onlyreplaceX; originalphysicalsettling evidence retained. Actualrestoration equality/maxposition/rotation deviations will be recorded, not fabricatedzero or a new acceptance cutoff. No new GPU job before CPUchecks/push. E2Estep1 rerun waits for this integration.
Before-change t01v2_supp9_before_01 passed forFP1/201:15 neither rejections comprise4visible,2hiddenRGB>3,9unknownmissingcompactprefixmeasurements. Existingtool full-four invocation t01v2_supp9_before_02 retained its failure:FP2/202 spec provenance is not recognized by old construction-version allowlist. Exact sourcecommit/document hash will be checked before recognizing the newly human-approved version; mismatch rejection stays.


### 2026-11-09 00:58 UTC — supplements 9/10 integration checkpoint
Supplement 9 and search serialization fixes are committed as `1836322`. Original physical w0/w1 evidence remains intact; rendered w1 requests reuse w0 nonanchor poses, native rendering stays paused, and requested/actual position and rotation deviations are reported without a new tolerance. Frozen targets and the existing ratio are rechecked. `t01v2_supp9_cpu_01` passed 440 CPU tests (22.224 s unittest runtime); no native GPU success is claimed yet. The minimally extended source compatibility list recognizes approved `4e010a3` only when the declared document hash exactly matches its source commit. `t01v2_supp9_before_03` now reads all four ended marked scenes with no source/raw errors: the old neither-prefix category has 24 attempts, of which 5 have X visible, 4 have X hidden with RGB maximum >3, and 15 lack retained prefix measurements and remain unknown. FloorPlan2 has no attempts in that category. Earlier failed diagnosis 02 is preserved. Human approved the new FloorPlan201 Direct-Stay image and explicitly authorized the supplement10 odd-scene 100-degree / even-scene 60-degree contrast; winner selection follows the preregistered Look-total, then Direct-total, then 100-degree tie rule without another approval. E2E remains a separate 100-degree five-pair process/speed smoke. GPU totals remain the last finalized values: T01v2 3.6880555555555556/16, E2E 0.023333333333333334/4, shared 15.39472222222222/60 charged GPU-hours; no new submission at this checkpoint.

E2E continuation: `smoke_e2e_views_02` was submitted as PBS `707488.hopper-m-02` (1 GPU, unpinned autox/smallx, 00:30:00 requested upper bound = 0.5 charged GPUh reserved). It was observed running on hopper-27 with PBS stime 2026-11-09 09:02:16 Asia/Singapore. It uses the original five reviewed 100-degree Direct pairs and supplement8/9 semantics, independent of the upcoming FOV contrast. The 30-minute ceiling is a bounded protective allocation for the still-incomplete full five-pair export/fresh-check procedure, not a measured complete-export duration. Submission audit `smoke_e2e_views_submit_audit_02`: 780 files, zero FAIL. Running result not yet known.


### 2026-11-09 01:11 UTC — FOV contrast ready
`d1b91e6` implements the fixed supplement10 allocation in `configs/t01v2_fov_contrast.json`: twenty odd-numbered training scenes use 100 degrees, twenty even-numbered scenes use 60; both arms use identical supplement6/7/9 search/render/acceptance code. `t01v2_fov_cpu_01` passed 446 tests in 22.873 s. Selection wording clarification: compare Look totals first; compare Direct totals only if both Look totals are zero; keep the current 100-degree setting for a remaining tie and report it. No winner has been selected from partial data. First calibration jobs will cover FloorPlan201/A100 and FloorPlan202/B60. Their previous measured full-scene durations were 687.1 s / 17 eligible attempts and 673.5 s / 43 eligible attempts, respectively. Planning uses these observed end-to-end rates times 100 attempts times 1.3, giving rounded scene ceilings 5260 s and 2040 s, plus gate/cleanup margins; requested PBS ceilings are 01:32:00 and 00:38:00. These are prospective reservations, not claimed post-fix performance, and later scenes will use observed new throughput. E2E707488 failed the native render gate on hopper-27 (GPU-6e4785f0-652c-3903-a3d9-b1f131c1266a), exit86, complete cleanup. Final accounting records 29 s = 0.008055555555555555 GPUh, queue wait84 s; automatic retry remains authorized. No successful native frozen export has yet been obtained.


### 2026-11-09 01:21 UTC — contrast allocations active
A100 FloorPlan201 is PBS `707493.hopper-m-02`, observed running on hopper-24 (PBS stime09:15:45 Asia/Singapore); B60 FloorPlan202 is `707495.hopper-m-02`, observed running on hopper-21 (stime09:16:01). Both passed their per-GPU native render gates. The running A source journal contains a generator-passed Direct-Stay pair `floorplan201_00002`; ended-source independent acceptance remains pending. Black-gate E2E707488 was automatically retried by the existing validated watcher as `707497.hopper-m-02` / `smoke_e2e_views_02_r8335e243`, preserving old records and its retry mapping. The retry remains unpinned, one GPU, same reviewed five100-degree Direct pairs and30-minute ceiling. Original-model step3 evaluator code `c647d33` passed its seven CPU tests (`smoke_e2e_eval_cpu_01`); GPU step3 still waits for actual step1/2 success. T0/T3 reward-format ambiguity was explicitly relayed to the human: task rewards versus upstream +0.5 valid-action bonus; no training run chooses a reward before that answer. No paper metric or pre-evidence coverage is claimed.


### 2026-11-09 01:28 UTC — first running FOV snapshot and native frozen diagnosis
`t01v2_fov_progress_01` archives complete journal prefixes observed at 01:26:15.760645 UTC (running source counts, not final independent acceptance). FloorPlan201/A100:17 attempts, Direct-Stay1, all other passes0; rejections: w0_physical5, anchor_visible_at_v0_but_direct_evidence_failed6, one_action_evidence_unavailable4, anchor_visibility_below_250_at_t0 1. FloorPlan202/B60:32 attempts, all pass classes0; rejections:w0_physical14,w1_physical3,target_uniqueness_or_ratio10,visible-anchor prefix2,one-action evidence2,t0 anchor visibility1. Neither arm records a hidden-X RGB>3 rejection in this prefix; this is a partial snapshot of different FOV arms, not a causal rate comparison with the older runs.
Read-only inspection of the already-retained attempts FP201_00003/_00007 and FP202_00012/_00023 found no native exception or failed restore: all action/world records observed, restores and continuations pass, all four start prefixes have X masks0/0 and maximum RGB difference0. Their actual nonanchor cross-world positions/rotations are exact (stored maximum position/rotation differences0). Requested-to-actual position differences are0; maximum rotation-component differences in both worlds are3.0517578125e-05 degrees for FP201_00003/_00007 and FP202_00023, and3.814697265625e-06 degrees for FP202_00012. Exact requested readback remains false; these are reported representation differences, not overwritten measurements. FP201 CreditCard00003 remains invisible;00007 right-turn masks41/19 and evidence30. FP202 Book00012 remains invisible. FP202 Book00023 move3 has masks451/509 and both-world solvability, but evidence219<250, so it correctly remains rejected. Raw attempts remain in the source scratch trial directories (201 trials000011/000020;202 trials000015/000029), referenced by the archived journals. Thresholds and running jobs unchanged.
`a098867` adds the direct-only real VAGEN PPO E2E4/5 adapter; six CPU contract tests passed in `smoke_e2e_training_cpu_01`. Native training remains unrun. It records actual tokenwise GAE, real Q_commit scoring, phase timings, empty pre-evidence sums with unavailable S1 variance, and no fabricated S2 point before update20. The human reward-format decision is still pending. E2E707497 has created the first pair's checked table and started the second export; the complete five-pair/fresh20-state result is not yet available.

Human confirmed E2E T0/T3 reward policy: correct select+1, wrong select0, observation−0.05, invalid output0, and no extra valid-format bonus. The published smoke config now records `explicit_task_zero`; both training invocations will pass that exact CLI policy and share the same common smoke configuration hash. No research contract or metrics document changed.


### 2026-11-09 01:44 UTC — second running snapshot
`t01v2_fov_progress_02` archived complete source journal prefixes at01:42:50.059585 UTC. A100/FP201:51 measured attempts, source-reported Direct-Flip1 and Direct-Stay1, Look0; rejection counts visible-X prefix17, hidden-X RGB>3 0, one-action evidence7, t0 anchor visibility3, target ratio1, candidate visibility1, w0 physics18,w1 physics2. B60/FP202:90 measured attempts, all pass classes0; visible-X prefix6, hidden-X RGB>3 0, one-action evidence12,t0 anchor visibility5,target ratio24,w0 physics38,w1 physics5. These are running generator reports, not independently accepted final totals. Snapshot times are recorded exactly; the previous snapshot was01:26:15 UTC.
Concurrent Claude commit4e8a305 only changed task documents: T04 agrees with the human-confirmed zero-format-bonus policy; supplement11 asks for far-to-near Stay w1 search, without changing evidence250 or other thresholds. The merge with our reward confirmation was pushed asb3b7867. Supplement11 is being prepared as additive opt-in producer modules, to keep current collection and E2E code files unchanged while workers run. Current allocations continue.

E2E707497 / smoke_e2e_views_02_r8335e243 completed step1, exit0: five checked D4 tables and fresh20-state check passed. Worker export durations and state counts are in tasks/E2E.report.md; PBS final cost0.4675 GPUh, queue67 s. This is only the E2E20-state exception, not formal T02 acceptance. Step2 pending; all prior failures retained.


Supplement12 be43fc4 read: new mixed collection paused; current707493/707495 allowed to finish. Next six Look-only units compare FP203/204 and the historical-most-attempted kitchen at both100/60degrees. Each unit reserves20minutes; E2E keeps one1GPU slot. Supplement11 far-first Stay and supplement9 frozen rendering remain required. E2E step2 CPU witness replay passed; step3 is next.


### 2026-11-09 02:04 UTC — final B60 and live A100; E2E evaluation submitted
The archived source journal snapshot t01v2_fov_progress_03 at02:04:26UTC has A100/FP20166attempts: DirectFlip1,DirectStay3,Look0; rejected w0physics23,w1physics2,visible-X-prefix21,one-action evidence9,t0visibility4,candidatevisibility2,ratio1. B60/FP202 ended with117attempts and0passed pairs, source status incomplete/deadline rather than accepted pilot; rejected w0physics51,w1physics6,ratio32,visible-X-prefix8,t0visibility6,one-action evidence14. Neither reports hidden-X RGB>3. All are actual producer counts; A independent acceptance pending. PBS707495 actual0.5658333333333333GPUh,wait27s (t01v2_fov60_accounting_01).
E2E step2 records and report were audited808files/0FAIL and pushed d8a6afd. Step3 original-model evaluation is707539,1GPU,unpinned Express/autox,1200s ceiling. Budget before submission: T01v2 measured+reserved5.787222222222223/16, E2E0.49888888888888894/4, shared17.969444444444445/60; this allocation reserves another1/3GPUh in E2E/shared. No new mixed collection submitted.


### Supplement12 sequential-only execution plan (latest human steering)
No3GPU packing will be submitted. Two1GPU jobs each run FloorPlan203 -> FloorPlan204 -> FloorPlan10, one FOV per job:100 first,60 only after707493 finishes. Each scene receives1200s, native deadline1190s; parent3870s and PBS3900s provide gate/launch/cleanup headroom. Each ended child is separately reportable while the parent continues. All thresholds and seed0 unchanged; Stay uses supplement11 far-first points; only Look pool is proposed and no Direct branch is collected. Expansion remains blocked until at least3Look pairs plus reviewed images.
Kitchen selection02 uses authenticated ended journals: FP10 has167 actual boolean w0physical checks (68passed),234non-storage rows; FP7 has95,FP1 has54. Unknown checks remain unknown. Failed scan01 is preserved; it exposed cached parent-summary counts143/135 versus actual final hashed journals144/136 for oldFP10/7, explained by last-progress versus final-journal timing. Original records unchanged.
Snapshot04 at02:15:05UTC: A10081attempts,DirectFlip1/Stay3,Look0; B60final117attempts,0pairs. A adds3evidence-retention-cap rejections; search continues. E2E707539 failed gateexit86 onhopper-27/GPU-6e4785f0-652c-3903-a3d9-b1f131c1266a with clean shutdown; no model evaluation started. Small evaluator fix allows legitimate empty action domains on an unconstrained final turn, preserving forced-select validation; it is deployed only after707539ended.

Supplement11/12 sequential implementation9e2474a passed595 CPU tests in23.444s (t01v2_look12_cpu_01), including Look-only scheduling, same-scene FOVs, far-first Stay, source guards, truthful partial funnels and deferred empty-action eval fix. E2E707539 verified clean black-gate retry eligibility with existing validator; actual19s allocation=0.005277777777777778GPUh,queue244s. Full failed source retained. Next submit100degree sequential1GPU first;60degree remains withheld until707493ends.


### 2026-11-09 02:25 UTC — sequential Look100 and E2E retry receipts
100degree Look-only1GPU sequential job is707546 (pilot_pairs_v2_look100_seq_01), orderedFP203/204/10, PBS65min. No3GPU job and no60degree job submitted. 60 remains conditional on707493ending. E2E retry707548 (smoke_e2e_eval_02),1GPU20min, records retry_of707539 in immutable payload; black proof and failed run are preserved. Both use unpinned Express/autox. Submission budgets retain full requested walltimes until final accounting; live charge rate verified by submitter.
Snapshot t01v2_fov_progress_05 at02:24:58.737927UTC: A100/FP20192attempts,DirectFlip1/Stay3,Look0; rejected w0physics32,w1physics3,visible-X-prefix27,one-action evidence13,t0visibility5,candidatevisibility4,ratio1,storage-evidence-incomplete3. B60 remains final117attempts/0pairs. These remain producer counts, not independent acceptance.


Running source snapshot t01v2_fov_progress_06 at2026-11-09T02:35:15.730070+00:00: A100/FP201 103attempts; source-reported passes={"direct_flip": 1, "direct_stay": 3, "look_flip": 0, "look_stay": 0}; rejected={"anchor_visibility_below_250_at_t0": 5, "anchor_visible_at_v0_but_direct_evidence_failed": 33, "evidence_incomplete_storage_cap": 3, "one_action_evidence_unavailable": 14, "target_uniqueness_or_ratio": 1, "v0_candidates_visibility": 4, "w0_physical": 36, "w1_physical": 3}. Producer counts only; independent final acceptance pending. Sequential60degree job remains withheld until707493ends.

Look100707546 passed its per-card render gate onhopper-21 and started the first sequential unitFP203. E2E707548 ended with clean black-gateexit86 onhopper-18/GPU-9317c80e-33a9-8f2c-b611-9a459bb1818e before model loading. Final accounting={"allocation_gpu_hours_by_job": {"707548.hopper-m-02": 0.005833333333333334}, "complete_allocation_accounting": true, "eligible_wait_seconds_by_job": {"707548.hopper-m-02": 525}, "queue_wait_seconds_by_job": {"707548.hopper-m-02": 525}, "total_allocation_gpu_hours": 0.005833333333333334, "verified_never_started_cancelled_jobs": []}. Existing reviewed black gate rule verifies retry eligibility; original failed sources preserved.

First Look producer pass observed during707546: FOV100/FP203/look_stay/floorplan203_00035. Exact existing review PNG and authenticated journal prefix published to logs/review/t01v2_look_first_01. Provisional only: source unit remains running; ended-source independent checkpoint pending. No expansion or FOV selection yet.

First60degree Look producer pass during707558: FP203/look_stay/floorplan203_00001. Exact existing review PNG and journal prefix published to logs/review/t01v2_look_fov60_first_01. Source still running; independent end-state checkpoint pending. BothFOVs now have provisionalLookStay evidence; noFOVselection or expansion from partial units.


### First ended Look-only unit:100degrees/FloorPlan203
Child ended at10:52:40Asia/Singapore,exit1/status incomplete with verified terminal cleanup and sealed journal; this is preserved. Its58recorded attempts are allStay; Flip has0attempts. The requested sequential funnel has cumulative observed passes10 ->10 ->2 ->1 ->1 ->1;38attempts failed the earlier physical check (36w0,2w1), so their later image gates were not measured. Three further records retain unknown downstream gates. No missing value filled in. The oneLookStay independently passed all construction-stage checkpoint checks; formalT02/table/cache acceptance is still pending. See logs/review/diag_look_funnel100_fp203_01.md and logs/review/t01v2_checkpoint_look100_fp203_01. Checkpoint invocation with an invalid run-name prefix failed before writing a result; its log/meta remain, and the corrected t01v2_checkpoint_* invocation passed.
Look60continues onhopper30; additional provisional original images are published as they appear. E2E707555step3passed actualoriginal-model evaluation; report/accounting are being finalized beforeT0.

E2E step3passed:30actualepisodes,10perstrategy; originalQwen2.5-VL-3B,complete episode logs. PBS707555 actual0.10388888888888889GPUh,queue106s; initial accounting03 failed becauseqstat was absent fromPATH, and the logged corrected03_retry1 succeeded; both preserved. Reportupdated. NextT0 is exactly5updates×64episodes,5turns,max256tokens,explicit_task_zero reward,noformatbonus,critic_offload=false first. Request1GPU/1hour ceiling within4GPUh E2E subbudget; this is a protective cap for the first measured training run, not a claimed runtime. Existing Direct/tau0 pool means证据前逻辑未覆盖.


### Second ended Look-only unit:60degrees/FloorPlan203
The source ended at11:07:03Asia/Singapore,exit1, with28recorded attempts (allStay,Flip0). Source-recorded cumulative funnel14 ->14 ->10 ->6 ->6 ->6; physical gates21pass/6fail/1unknown. Six generator-passed LookStay pairs remain provisional: independent checkpoint rejected allsix with Portable pair preflight differs from the logged source preflight; independently verified count0. Both successful funnel and failedcheckpoint records are preserved. Investigating the actual controller/preflight association using existing evidence; no threshold changes, source rewriting, or automatic acceptance. See logs/review/diag_look_funnel60_fp203_01.md.
T0smoke is707594,1GPU,5updates,explicit_task_zero reward,1hourcap. It started and passed its native gate; no training-update result yet.

### Look single-GPU units: FOV100 FloorPlan204 ended
The second100-degree unit ended at2026-11-09T11:12:10 source time. Authenticated checkpoint records70 attempts (Flip43,Stay27), zero qualified pairs. Physics rejected35;29 were rejected by target ratio/realized-mode checks and5 by visible anchor;one terminal interruption is retained. The per-mode gate table and available rejected examples are in logs/review/diag_look_funnel100_fp204_01.md; unobserved later gates are explicitly unknown. Independent checkpoint found no evidence errors and no accepted pairs. The parent continues FloorPlan10. FOV60 FloorPlan204 provisional Look images are published in logs/review/t01v2_look_fov60_fp204_01; independent per-controller validation is pending.

E2E T0 job707594 failed before its first PPO update: pinned verl's colocated-worker factory requires a common immediate base class for actor/ref and critic wrappers. No step timing or training pass is claimed; failure records retained. A shared hook base compatibility fix is being prepared before retry.

### E2E colocated-worker compatibility fix
FailedT0 job707594 consumed0.08361111111111111 allocatedGPU-hours at confirmedrate1 and waited12 seconds (results/smoke_e2e_T0_accounting_01.json). It completed no update. Commit1f4f36d gives actor/ref and critic the shared observation/seed-hook immediate base required by the pinned verl factory, preserving upstream constructors and update implementations. The real installed Ray/verl CPU wiring regression and training-contract suite passed9 tests in39.596s; no test skipped. Retry retains the same training/reward configuration and a one-GPU,one-hour maximum reservation.

### Look FOV60 FloorPlan204 unit and per-Controller acceptance
The second60-degree unit ended at 2026-11-09T11:26:33 source time:38 recorded attempts, Flip7 and Stay31. Producer passes:0 LookFlip,5 LookStay. The Stay cumulative funnel is10 ->10 ->6 ->5 ->5 ->5; Flip has0 reaching the first hidden-prefix gate, with6 physical failures and1 later target-mode rejection. See logs/review/diag_look_funnel60_fp204_01.md for actual gate counts, unknown measurements and recorded rejected examples.
The source-final preflight mismatch was an association bug: accepted FP203 pairs belonged to marked_controller_000, while the final source reference describes marked_controller_001. Additive adapter ef3f48b authenticates the identical pair/raw Controller ID, unique audited visit, original index/preflight/baseline/setup/physical snapshots and process identities, then calls the unchanged full validators. Original sources and failed checkpoint remain unchanged. All13 CPU behavior tests passed (t01v2_controller_association_cpu_01). Fresh checkpoints t01v2_checkpoint_look60_fp203_02 and t01v2_checkpoint_look60_fp204_02 independently accepted6 and5 LookStay respectively, errors[]; these are construction checks, not formalT02 acceptance. Both image directories are published. The prior100degree FP203 independently accepted1 LookStay.

Human relayed Claude's no-blocker review of t01v2_checkpoint_look100_fp203_01 and t01v2_look_fov60_fp204_01. NEW expansion gate: at least1 independently accepted LookFlip AND at least3 accepted Look total AND no-blocker image review. No expansion yet: LookFlip remains0. Supplement11 far-first Stay is already active. Latest fetched main contains supplement12 only; requested missing detailed Flip fix text before implementing an assumed design. Stable object-label behavior is being checked.
E2E T0 retry is707612,smoke_e2e_T0_02,1GPU/unpinned Express/autox/one-hour cap, retry_of707594. Before submission E2E actual0.6975/4GPUh and T01v2 measured+reserved7.877222222222223/16; full newT0 walltime reserved. Active collection parents continue their kitchen unit.

### 100-degree kitchen unit completed; overnight authorization
FloorPlan10 at100degrees completed at2026-11-09T11:31:44 Asia/Singapore (Nov8 22:31:44 America/New_York),72 recorded Stay attempts and0 Flip attempts,0 accepted pairs. Rejections:60 anchor_visible_at_v0,5 w1_physical,7 target_uniqueness_or_ratio. Unmeasured gates remain unknown in logs/review/diag_look_funnel100_fp10_01.md. Independent checkpoint t01v2_checkpoint_look100_fp10_01 found no evidence errors and no qualified pairs. Parent707546 final allocation0.9791666666666666 GPUh at scheduler rate1; queue613 seconds (t01v2_look100_accounting_01). All three100degree units are now complete;60degree FloorPlan10 still runs.
Human explicitly authorized task supplement14/47ea528 on Nov8 evening: autonomously iterate20-minute single-GPU Look pilots within search/order/proposal changes, retain all failures and push every funnel/image round; expand only after at least1 independently accepted LookFlip and3 Look total, without waiting for overnight image-review relay. G01 expansion cap20 GPUh; thresholds and task structure unchanged. Write overnight STATUS summary before Nov9 08:30 America/New_York.
Supplement13 label diagnosis: legacy T01 mark_observation labels by each view's spatial bounding-box order, so a visible candidate can change label across views. T02 label_assignment already binds pair_id and stable object identity. Existing construction checkpoints are historical-policy passes, not proof of new supplement13 fixed-label compliance. New versioned generation/review/witness mapping is being prepared without modifying old evidence or active workers. Supplement11 far-first Stay remains active.

### Flip bottleneck diagnosis from retained records
Logged diagnostic diag_look_flip_existing_01 and its concise report logs/review/diag_look_flip_existing_01/report.md authenticate four ended living-room sources. FloorPlan203/100 has0 dual-sided Look domains among72 recorded catalogs; its attempts are58Stay/0Flip. FloorPlan203/60 has1 dual-sided catalog among76 and28Stay/0Flip; that single catalog's exact scheduling/deadline history remains under investigation. FloorPlan204/100 has43Flip attempts rejected by w0physical19,target ratio14,realized target mode9,andterminal1; FloorPlan204/60 has7Flip rejected by w0physical6,target ratio1. No Flip in either reached a measured hidden-prefix gate. Existing raw detailed examples confirm closest_ratio_below_1_5; omitted compact target details are not inferred. The old proposal side calculation is horizontal, while acceptance measures3D distance; only12near Look points per side are retained. Next version will use supplement13's full legal Flip proposal domain, edge-view priority, a scene-wide actual Flip-attempt share of at least onehalf, and fixed per-object labels. It will keep the unchanged physical/ratio/prefix/evidence/solvability/witness thresholds.

### Completed six-unit FOV choice and fixed-label CPU checks
The last60-degree FloorPlan10 unit ended Nov8 22:46:06 America/New_York:39 recorded attempts, all rejected by w0 physical validity,0 independently accepted pairs. Its final funnel is logs/review/diag_look_funnel60_fp10_01.md. Parent707558 used0.9775 GPUh at rate1 and waited50s; combined two-parent cost1.9566666666666666 GPUh.
The predeclared same-scene comparison selects60degrees:100degrees has1 historical construction-verified LookStay,60degrees has11; both LookFlip counts0. See tasks/T01v2.report.md, results/t01v2_look_fov_selection_01.json and configs/t01v2_selected_fov.json. Rule clarification02 preserves the initial diagnostic/config and precisely limits the Direct fallback to both Look totals beingzero; no tied branch was reached, no source count or actual60-degree choice changed. New fixed-label compliance remains pending GPU evidence, so these old pairs do not unlock expansion under the new version.
Commit6e476a3 adds opt-in fixed per-object labels sharing T02's mapping, plus required E2E parse/timing reporting. All13 CPU behavioral tests passed (10labels,3report); old spatial-label evidence remains unchanged and explicitly historical. The next pilot integrates this policy with full legal Flip proposals and at least onehalf scene-wide actual Flip attempts, at60degrees only.

### E2E T0 five-update smoke passed
Job707612 / smoke_e2e_T0_02 completed5 actual updates on1H200 with critic_offload=false, saved scratch checkpoints, and exited0 with errors[]/cleanup_errors[]. PBS actual0.3675GPUh,queue206s. Update total seconds are177.29877089895308,119.1544680770021,123.43704485008493,120.91650402010418,116.92604623490479; exact source values and split timings are in tasks/E2E.report.md and results/smoke_e2e_T0_report_02.json. The logged mean131.5465668162098s yields a planning-only200-update projection7.308142600900545h; it excludes setup/checkpoint and assumes these Direct/tau0 short trajectories, so it is not a measured main-training runtime.
Original-model step3: each policy0/10 syntax parse failures;40forced obs_always observations excluded. Each T0 update0/64 physical-output parse failures, and0 syntax-valid-but-unavailable actions. Actual allocator peak statistics are retained with their allocator-only scope; reserved bytes are not physical device occupancy. Evidence-before-seeing logic remains uncovered.
Next T3 uses the same approved100-degree Direct smoke pool/common config and5updates; request1GPU with35-minute PBS ceiling and2040-second worker cap, allowing headroom over the measured22:03T0 allocation for added teacher-forced Q_commit scoring. This cap is a planning reservation, not a predicted or observed T3 duration.

### Next Flip pilot ready after source and CPU checks
Supplement13 implementation dafc07f passed518 CPU tests in11.710s (t01v2_flip13_cpu_01), covering the original generation/evidence checks plus complete native3D Flip proposals, opposite-edge priorities, actual-attempt quota across Controller restarts, fixed per-object labels and strict new-source checks. Next two single-GPU20-minute units are FloorPlan204 andFloorPlan203 at60degrees; each PBS ceiling23minutes includes its gate/finalization. No thresholds or task structure changed; the iteration aims to reach an independently accepted fixed-label LookFlip.
The authenticated follow-up diag_look_flip_followup_01 resolves the earlier FP203/60 ambiguity: its bilateral Book/Bowl/Vase pool000990 had before-capA/B20/952 and retained12/12. Terminal attempt00027 used older pool000324, requestedFlip but fell back toStay because that stream hadFlip0/StayA66, then ended on the recorded deadline. The statement that this bilateral stream was not reached before timeout is an inference from those records plus scheduler order, not a newly measured event. The new quota refuses this Stay fallback when Flip is owed.
T3 smoke707642 has been submitted on1GPU with35-minute ceiling; budget guard ran before submission. It uses the exact successful T0 common configuration and reviewed100-degree Direct pool. Final T0 records/report are pushed c768890.


### Supplement14 first fixed-label Flip iteration and G01 guard
Two 60-degree, one-GPU twenty-minute pilots were submitted after code and CPU tests were pushed: FloorPlan204 job707653 and FloorPlan203 job707656. FP203 passed its per-card gate and started. FP204 ended at its native render gate with exit86 on hopper-22 / GPU-5ff2e10b-a949-4c82-fa91-0e22f6075d56, complete cleanup and no scene started. Its canonical PBS accounting records0.006388888888888889 GPUh at rate1 and queue423s; the original failed source is preserved before authorized automatic retry.
The existing submit_phase guard now includes the human-approved20 chargedGPUh G01 training-expansion subbudget inside the shared60-hour ceiling, retaining full reservations until authenticated final accounting. This is budget preparation only: no G01 GPU work is allowed until new-version independent LookFlip>=1 and Look total>=3. CPU boundary checks follow this code commit.

The G01 budget addition passed7 focused boundary tests and65 existing scheduler/budget regression tests; no GPU work was used for these checks. Expanded collection remains gated by independent new-version Look counts.


### 2026-11-09 04:25 UTC — fixed-label Flip first progress
Archived journal prefix t01v2_flip_progress_01 at04:25:24.002019UTC reports6 actual FP203 attempts,0 producer-passed pairs. Rejections: w0_physical3, v0_candidates_visibility1, anchor_visible_at_v0 1, one_action_evidence_unavailable1. w0 passes by anchor: Boots0/3, Box1/1, CellPhone1/1, CreditCard1/1. These are observed running-source counts, not final independent acceptance. New Flip proposals have reached real physics; later gates remain controlling.
The strictly verified black-gate job707653 was automatically retried as707668 / pilot_pairs_v2_flip60_fp204_r01_retry1, same one-GPU23-minute allocation and20-minute scene, no host pin; retry_of is recorded in immutable submission payload. The original failed source and exact new-version black proof diag_flip_black_retry_707653 are retained.
T3 job707642 completed five actual updates, saved checkpoints, exit0 with errors[]/cleanup_errors[]. PBS records0.2961111111111111 GPUh at rate1 and queue161s; exact split timing/parse report is smoke_e2e_training_report_01. All examples areDirect/tau0: evidence-before-seeing logic and prefix sampling deduplication remain uncovered.


### First fixed-label FP203 unit ended; next scheduling-only iteration
Job707656 ended at12:36:48 Asia/Singapore with worker_failed/parent deadline and no verified terminal receipt. The last observed progress reported15 attempts and0 producer passes, but final attempt/class counters are unavailable; no missing seal or counter is repaired. Parent cleanup errors are empty. Canonical PBS accounting records0.33416666666666667 GPUh at rate1, queue179s. The04:35:30UTC archived prefix independently contains14 complete observed rows; its rejected reasons are w0physical7, visible-X4, candidatevisibility1, one-action evidence1, ratio1. Ended-prefix diagnostics will retain this nonfinal scope.
New add-only search a4a0d6d changes only discovery cadence: discover one new domain, then give each already-active domain one visit before another expensive discovery. Native domains, recorded queries, seed0 assignments, physical10-skip and every scientific threshold remain unchanged; no query caching or inferred coordinates were added. The first CPU run failed6 new fixture cases because attempt_journal was missing. Commitb08350d adds that required fixture field without changing assertions or production logic; t01v2_flip_rounds_cpu_02 then passed532 tests in13.022s. Both runs are retained.
FP204 retry707668 passed its gate and started at12:36:50 Asia/Singapore; it remains untouched. The next FP203 twenty-minute iteration will use the newly tested scheduling version after its code and diagnostics are pushed. G01 remains blocked on an independently accepted fixed-label LookFlip and at least3 new-version Look pairs.

Ended-prefix diagnostic diag_flip_partial_fp203_r01 authenticated 15 recorded attempts; actual observed quota={"counted_flip": 8, "counted_stay": 7, "next_required_mode": "stay", "scope": "across the entire scene, including Controller restarts: counted Flip attempts >= counted Stay attempts; failed native placement does not earn quota"}. The failed source is unsealed, so these remain observed-prefix diagnostics only. Per-gate Flip/Stay counts and the first available rejected examples are in logs/review/diag_flip_partial_fp203_r01.md; strict checkpoint t01v2_checkpoint_flip60_fp203_r01 independently admits0 pairs and leaves final attempt counts unknown. Recorded timing scope={"discovery_seconds": null, "logged_total_duration_sec": 1192.4, "observed_counted_generation_seconds_sum": 80.80041176197119, "outside_observed_counted_spans_seconds": 1111.599588238029, "scope": "Residual includes uncounted placements, startup/preflight/discovery/serialization/finalization/cleanup; not attributed to discovery"}.

FP204 retry running prefix t01v2_flip_progress_03 at04:46:08.944155UTC:3 observed attempts,0 producer passes; rejected visible-X2,w0physical1. No independent acceptance is inferred.

### Overnight live prefix 2026-11-09T05:01:33.720365+00:00

707691 / FloorPlan203 cadence-only r02 continues. Complete observed journal rows: 17; producer-reported classes: {"direct_flip": 0, "direct_stay": 0, "look_flip": 0, "look_stay": 0}; rejection counts: {"anchor_visible_at_v0": 6, "one_action_evidence_unavailable": 1, "target_uniqueness_or_ratio": 1, "v0_candidates_visibility": 1, "w0_physical": 8}. This is an archived live prefix, not a final source or independent acceptance. Source: results/t01v2_flip_progress_04.json.


### FP204 Flip unit 707668: sealed failure funnel

PBS 707668.hopper-m-02 的 FloorPlan204/60° 单元已结束，原运行状态为 `incomplete`、exit 1，失败原因 `TimeoutError: Worker deadline expired`。这是已封存的终止记录，并非完成采集：原记录共 10 次计入尝试，Flip/Stay 为 5/5；原生放置失败而未计入的提议为 12。所有失败及终止尝试保留；独立验收另看 checkpoint，下面仅是生成器原记录漏斗，不能当作独立合格数。

表中每个 gate 均为「通过/失败/未知」，分母是该行实际尝试；尚未执行的检查保持未知，没有补成零。各 gate 是独立测量列，并非假定依次全部执行；原执行顺序仍先物理检查。

| Group | Mode | Actual attempts | Source-reported Look passes | X hidden | Prefix ≤3 | First evidence ≥250 | Solvable both | Physical both | Witness both |
|---|---|---:|---:|---|---|---|---|---|---|
| All anchors | flip | 5 | 0 | 2/3/0 | 2/3/0 | 1/1/3 | 0/2/3 | 5/0/0 | 0/0/5 |
| All anchors | stay | 5 | 0 | 0/1/4 | 0/1/4 | 0/0/5 | 0/0/5 | 1/3/1 | 0/0/5 |
| CellPhone | flip | 3 | 0 | 1/2/0 | 1/2/0 | 1/0/2 | 0/1/2 | 3/0/0 | 0/0/3 |
| CreditCard | flip | 2 | 0 | 1/1/0 | 1/1/0 | 0/1/1 | 0/1/1 | 2/0/0 | 0/0/2 |
| CreditCard | stay | 1 | 0 | 0/1/0 | 0/1/0 | 0/0/1 | 0/0/1 | 1/0/0 | 0/0/1 |
| KeyChain | stay | 2 | 0 | 0/0/2 | 0/0/2 | 0/0/2 | 0/0/2 | 0/2/0 | 0/0/2 |
| Pillow | stay | 2 | 0 | 0/0/2 | 0/0/2 | 0/0/2 | 0/0/2 | 0/1/1 | 0/0/2 |

拒绝原因原计数：anchor_visible_at_v0=4; one_action_evidence_unavailable=2; terminal_interruption=1; w0_physical=3。来源：`results/diag_flip_sealed_fp204_retry1.json` 及其原日志中的 `SEALED_FLIP_ANCHOR_FUNNEL_JSON`，绑定 `results/pilot_pairs_v2_flip60_fp204_r01_retry1_j707668_hopper_m_02_g0_floorplan204.json` 与完整 journal；阈值和原始记录均未改。

### Next overnight Flip iteration: ranked native proposals

Commit137bc6a adds an opt-in producer with the existing discovery cadence, exactly equal complete native tokens, and stable Look-only ordering by predicted whole-anchor visibility. It retains every old candidate point, all thresholds, physical checks, frozen renders, seeded orientation, far-first Stay, fixed labels, and independent witness replay. Reason: sealed FP204 and observed FP203 funnels still reject visible anchors at v0; large native-token work also limits the number of trials. Predicted boxes are proposal priorities only; actual rendered masks remain authoritative. CPU regression result: results/t01v2_flip_ranked_cpu_01.json (557 tests, successful). Next units are60°/FloorPlan203 and204, each full1200seconds,1GPU, premium/autox, budget-guarded before submission.

### Overnight live prefix 2026-11-09T05:10:32.638193+00:00

707691 / FloorPlan203 cadence-only r02 continues. Complete observed journal rows: 56; producer-reported classes: {"direct_flip": 0, "direct_stay": 0, "look_flip": 0, "look_stay": 0}; rejection counts: {"anchor_visible_at_v0": 18, "one_action_evidence_unavailable": 5, "target_uniqueness_or_ratio": 7, "v0_candidates_visibility": 2, "w0_physical": 20, "w1_physical": 4}. This is an archived live prefix, not a final source or independent acceptance. Source: results/t01v2_flip_progress_05.json.


### FP203 cadence Flip unit 707691: sealed failure funnel

PBS 707691.hopper-m-02 的 FloorPlan203/60° 单元已结束，原运行状态为 `incomplete`、exit 1，失败原因 `TimeoutError: Error encountered when running action {'action': 'PlaceObjectAtPoint', 'objectId': 'Box\|-03.71\|+01.07\|-00.63', 'position': {'x': -6.2486042976379395, 'y': 0.3065462112426758, 'z': -0.0050275325775146484}, 'rotation': {'x': 359.0010070800781, 'y': 327.9556884765625, 'z': 359.19390869140625}, 'forceKinematic': True, 'sequenceId': 2423} in scene FloorPlan203_physics.`。这是已封存的终止记录，并非完成采集：原记录共 57 次计入尝试，Flip/Stay 为 29/28；原生放置失败而未计入的提议为 9。所有失败及终止尝试保留；独立验收另看 checkpoint，下面仅是生成器原记录漏斗，不能当作独立合格数。

表中每个 gate 均为「通过/失败/未知」，分母是该行实际尝试；尚未执行的检查保持未知，没有补成零。各 gate 是独立测量列，并非假定依次全部执行；原执行顺序仍先物理检查。

| Group | Mode | Actual attempts | Source-reported Look passes | X hidden | Prefix ≤3 | First evidence ≥250 | Solvable both | Physical both | Witness both |
|---|---|---:|---:|---|---|---|---|---|---|
| All anchors | flip | 29 | 0 | 2/13/14 | 2/13/14 | 0/2/27 | 0/2/27 | 15/13/1 | 0/0/29 |
| All anchors | stay | 28 | 0 | 3/7/18 | 3/7/18 | 0/3/25 | 0/3/25 | 17/11/0 | 0/0/28 |
| Boots | flip | 4 | 0 | 0/0/4 | 0/0/4 | 0/0/4 | 0/0/4 | 0/4/0 | 0/0/4 |
| Boots | stay | 6 | 0 | 0/0/6 | 0/0/6 | 0/0/6 | 0/0/6 | 0/6/0 | 0/0/6 |
| Box | flip | 3 | 0 | 0/2/1 | 0/2/1 | 0/0/3 | 0/0/3 | 2/0/1 | 0/0/3 |
| Box | stay | 2 | 0 | 0/2/0 | 0/2/0 | 0/0/2 | 0/0/2 | 2/0/0 | 0/0/2 |
| CellPhone | flip | 6 | 0 | 0/6/0 | 0/6/0 | 0/0/6 | 0/0/6 | 6/0/0 | 0/0/6 |
| CellPhone | stay | 3 | 0 | 0/2/1 | 0/2/1 | 0/0/3 | 0/0/3 | 3/0/0 | 0/0/3 |
| CreditCard | flip | 3 | 0 | 2/1/0 | 2/1/0 | 0/2/1 | 0/2/1 | 3/0/0 | 0/0/3 |
| CreditCard | stay | 5 | 0 | 3/0/2 | 3/0/2 | 0/3/2 | 0/3/2 | 5/0/0 | 0/0/5 |
| KeyChain | flip | 4 | 0 | 0/0/4 | 0/0/4 | 0/0/4 | 0/0/4 | 0/4/0 | 0/0/4 |
| KeyChain | stay | 3 | 0 | 0/0/3 | 0/0/3 | 0/0/3 | 0/0/3 | 0/3/0 | 0/0/3 |
| Laptop | flip | 2 | 0 | 0/0/2 | 0/0/2 | 0/0/2 | 0/0/2 | 0/2/0 | 0/0/2 |
| Laptop | stay | 2 | 0 | 0/0/2 | 0/0/2 | 0/0/2 | 0/0/2 | 2/0/0 | 0/0/2 |
| Newspaper | flip | 2 | 0 | 0/2/0 | 0/2/0 | 0/0/2 | 0/0/2 | 2/0/0 | 0/0/2 |
| Newspaper | stay | 3 | 0 | 0/2/1 | 0/2/1 | 0/0/3 | 0/0/3 | 3/0/0 | 0/0/3 |
| Pillow | flip | 3 | 0 | 0/0/3 | 0/0/3 | 0/0/3 | 0/0/3 | 0/3/0 | 0/0/3 |
| Plate | stay | 3 | 0 | 0/0/3 | 0/0/3 | 0/0/3 | 0/0/3 | 1/2/0 | 0/0/3 |
| RemoteControl | flip | 2 | 0 | 0/2/0 | 0/2/0 | 0/0/2 | 0/0/2 | 2/0/0 | 0/0/2 |
| TissueBox | stay | 1 | 0 | 0/1/0 | 0/1/0 | 0/0/1 | 0/0/1 | 1/0/0 | 0/0/1 |

拒绝原因原计数：anchor_visible_at_v0=18; one_action_evidence_unavailable=5; target_uniqueness_or_ratio=7; terminal_interruption=1; v0_candidates_visibility=2; w0_physical=20; w1_physical=4。来源：`results/diag_flip_sealed_fp203_r02.json` 及其原日志中的 `SEALED_FLIP_ANCHOR_FUNNEL_JSON`，绑定 `results/pilot_pairs_v2_flip60_fp203_r02_j707691_hopper_m_02_g0_floorplan203.json` 与完整 journal；阈值和原始记录均未改。

### Supplement15 geometry prefilter: CPU component complete

Commit772fa9e implements full native-point input, predicted centre3D role ratios, complete start box outside10px, independent one-step box visibility, lazy full tuples, and fallback point validation.10 focused tests passed in results/t01v2_flip_geometry_cpu_01.json, including same-screen-side candidates and surface/centre disagreement. The actual producer/independent checker wiring is still being integrated; no supplement15 GPU job has been submitted. All final rendered and physical thresholds remain unchanged.

| Job | GPUs | Queue wait (minutes) | Measured allocation GPU-hours | Recorded charge rate |
|---|---:|---:|---:|---:|
| 707668.hopper-m-02 | 1 | 11.566666666666666 | 0.32666666666666666 | 1 |
| 707691.hopper-m-02 | 1 | 1.0833333333333333 | 0.32666666666666666 | 1 |

### Supplement15 actual producer ready for the next bounded pilots

Commit601b3f2 connects the approved complete-tuple geometry before physical construction, after the original seed exchange, and at every actual native retry validator. The original successful w0 fallback becomes w1's fixed counterpart. Same-screen-side roles are allowed; original Stay, physical/pixel/solvability/witness checks and fixed labels remain. Focused actual-native/spawn wiring:11 tests passed (results/t01v2_flip_s15_producer_cpu_01.json); complete generation regression:571 tests passed (results/t01v2_flip_s15_regressions_01.json). The strict read-only v2 independent-checkpoint adapter is being completed in parallel; no new pair is independently counted or used to unlock G01 before it passes. Next: two separate1GPU20-minute60° pilots in FloorPlan203/204, premium/autox, no host pin, original per-card image gate and submission budget guard.

### Supplement15 bounded GPU pilots submitted

Both receipts confirm premium/autox submission,1GPU/12CPU, no host restriction,23-minute allocation ceiling and a full1200-second scene unit with the original per-card image gate. The generator is committed/tested/pushed before submission. The source-specific independent read-only checker continues in parallel; generated candidates remain pending independent acceptance.

| Pilot | PBS job | Reserved charged GPU-hours |
|---|---|---:|
| pilot_pairs_v2_flip60_fp203_s15_r01 | 707737.hopper-m-02 | 0.38333333333333336 |
| pilot_pairs_v2_flip60_fp204_s15_r01 | 707740.hopper-m-02 | 0.38333333333333336 |

After the second requested reservation, the submission ledger bound is T01v2=9.427777777777777/16 and shared T00b+T01=22.47222222222222/60 charged GPU-hours. These are measured-or-reserved upper bounds, not claims that queued allocations have run. Receipt directories: logs/pbs/pilot_pairs_v2_flip60_fp203_s15_r01-20261109T053733Z-cae9d210, logs/pbs/pilot_pairs_v2_flip60_fp204_s15_r01-20261109T053936Z-a2bd404f.

### Supplement15 first provisional LookFlip images

Published 3 generator-reported Look→Flip candidates from the running FP203/FP204 pilots in logs/review/t01v2_s15_provisional_lookflip_02. Independent acceptance remains pending terminal source records and the strict source adapter; none is counted toward expansion yet. Existing images were copied only after binding their hashes to authenticated complete journal prefixes and immutable gate/submission identity. See diag_t01v2_s15_provisional_lookflip_02.


### Supplement 15 live journal snapshot t01v2_flip_s15_progress_01

2026-11-09T05:53:12.741094+00:00：仅统计已归档的完整 journal 行；正在运行的源尚未终态验收，reported pass 不等于独立合格，未观察的后续尝试不补为零。原字节快照和 SHA 见 `results/t01v2_flip_s15_progress_01.json` 及对应日志。

| Scene | Complete observed attempts | Reported LookFlip | Reported LookStay | Stored rejection counts |
|---|---:|---:|---:|---|
| FloorPlan203 | 60 | 9 | 0 | anchor_visible_at_v0=10; one_action_evidence_unavailable=5; target_uniqueness_or_ratio=14; w0_physical=20; w1_physical=2 |
| FloorPlan204 | 55 | 1 | 0 | anchor_hidden_but_prefix_rgb_diff_gt3=2; anchor_visible_at_v0=12; one_action_evidence_unavailable=4; target_uniqueness_or_ratio=1; w0_physical=28; w1_physical=7 |

### T02 supplement15 source loading CPU check

The opt-in T02 loader now binds ranked v2 checkpoints to the exact authenticated scene and per-pair Controller baseline, retains all scientific checks, and verifies fixed candidate labels. The original source path is unchanged. All 69 pair-table CPU tests passed (t02_ranked_source_cpu_01). No new GPU was used; actual Look table export awaits independent pair acceptance.

### Supplement15 sealed pilot funnels

Both sources ended at their bounded deadlines with clean cleanup. Complete source journals and the first recorded failed examples are preserved. Independent pair acceptance is separate; the table below is source-reported, and each gate is pass/fail/unknown.

| Scene | Mode | Counted | Reported Look | Hidden | Prefix | Evidence | Solvable | Physical | Witness |
|---|---|---:|---:|---|---|---|---|---|---|
| FloorPlan203 | flip | 55 | 16 | 28/0/27 | 28/0/27 | 16/12/27 | 16/12/27 | 40/15/0 | 16/0/39 |
| FloorPlan203 | stay | 55 | 0 | 4/20/31 | 2/22/31 | 0/2/53 | 0/2/53 | 35/19/1 | 0/0/55 |
| FloorPlan204 | flip | 52 | 1 | 4/0/48 | 2/2/48 | 1/1/50 | 1/1/50 | 7/44/1 | 1/0/51 |
| FloorPlan204 | stay | 51 | 0 | 16/15/20 | 15/16/20 | 0/15/36 | 0/15/36 | 32/19/0 | 0/0/51 |

Detailed real rejected examples and w0 per-anchor passed/measured/unknown counts: logs/review/diag_flip_sealed_fp203_s15_r01.md.
Detailed real rejected examples and w0 per-anchor passed/measured/unknown counts: logs/review/diag_flip_sealed_fp204_s15_r01.md.

### Supplement15 allocation accounting

| Job | GPUs | Queue wait (minutes) | Allocation GPU-hours | Recorded billing multiplier | Charged GPU-hours |
|---|---:|---:|---:|---:|---:|
| 707737.hopper-m-02 | 1 | 3.183333333333333 | 0.3263888888888889 | 1.0 | 0.3263888888888889 |
| 707740.hopper-m-02 | 1 | 2.3333333333333335 | 0.32666666666666666 | 1.0 | 0.32666666666666666 |

This round consumed 0.6530555555555555 scheduler-charged GPU-hours. Current accounted-or-reserved ledger: {"E2E": 1.3611111111111112, "G01": 0.0, "T01v2": 9.427777777777777}, shared T00b+T01 22.47222222222222/60. Source: results/t01v2_flip_s15_accounting_01.json. Billing uses the scheduler multiplier, not an independent project-credit reconciliation.

2026-11-09 补充15独立检查：`results/t01v2_checkpoint_flip_geometry_01.json` 已 exit 0、errors=[]；213 次原始尝试中的已完成候选经原始 Controller/基线/native query、实际初选与成功点、全部重试、物理/像素/见证/固定标签完整重验，去重后发布 14 对 LookFlip（FloorPlan203 13 对、FloorPlan204 1 对）及 `logs/review/t01v2_checkpoint_flip_geometry_01/` 下 14 张图。源报告的 17 条候选中，`floorplan203_00060`、`floorplan203_00082`、`floorplan203_00106` 被原 `proposal_identity` 规则列入 duplicate_pairs：身份依据是场景、稳定候选/锚点名、v0 视角和有序 w0/w1 的原生 receptacle/selected_position，非图像相似度或结果筛选；重复原记录全部保留。两源超时 exit 1 的失败记录仍保留，接受仅限已封存、清理正常且完整证据通过的对。按人类批准的补充14夜间推进条件，独立 LookFlip 门槛已达到，G01 准备启动；这是 construction + 补充4(a) checkpoint，不是正式 T01 或 T02 全任务验收，P 表/BFS/缓存检查和相应审查仍另行完成。本次独立检查仅用 CPU，新增 GPU 用量 0；采集作业 707737/707740 的用量按另行 PBS 实账记录。审计 973 files、0 FAIL。

### G01 initial concurrency benchmark ready

The supplement14 expansion gate is met by 14 unique independently checked LookFlip pairs. G01 now uses the successful supplement15 LookFlip proposal geometry and retains the original Direct/Stay scientific gates, fixed labels, far-first Stay, and actual per-class cap8. Focused G01/source/resource checks:58 passed (g01_s15_cpu_01); complete generation/checkpoint regression:600 passed (g01_s15_regressions_01). The first job will repeat the same six training scenes (1,7,203,204,301,401) on one GPU at2,4,6 scene concurrency, each scene receiving a full1200-second cap. All lanes share the allocated12CPU; persistent witness admission limits total AI2-THOR instances to8. The allocation execution ceiling is7840seconds plus30seconds PBS wrapper margin; this is a scheduling bound, not a measured throughput. No scaling claim or further bulk submission is made before the real benchmark. G01 total cap remains20chargedGPUh and shared phase cap60.


### Blocker limited to Look E2E credit backfill; G01 continues

The completed Direct E2E T0/T3 smoke remains valid for its declared scope (tasks/E2E.report.md), with pre-evidence logic explicitly uncovered. configs/smoke_e2e_training.json is direct_only and explicitly not the T04 public-configuration freeze. The actual resolved masked_gae branch uses gamma=1 and lam=1; high_level_gamma=.95 is not used by that branch. Its executed ordering is token environment reward minus fixed .001 KL, token GAE/critic returns, then masked whitening.

tasks/T05.md defines decision-level G-Q and GAE fallback, while configs/t05_credit_draft.json still leaves the token-to-decision adapter, realized-return definition and normalization order unfilled. Before executing Look T3, Claude/human must specify: (1) which token GAE or aggregation defines fallback decision GAE, and how a scalar decision advantage is applied over the full reply token span; (2) whether G includes token KL shaping and how the shared KL term is retained after prefix replacement; (3) replacement versus whitening order/statistical domain, preserving identical paired pre-evidence advantages and unchanged post-evidence GAE. Critic targets remain the existing targets. These choices are not inferred by deleting the Direct guard. A question has been sent asynchronously to the human; G01 proceeds independently.

The actual E2E common hash includes the evaluation/source pool. A new Look pool therefore requires a matched new T0 baseline before reporting a same-config T3/T0 timing ratio; the old Direct T0 baseline will not be silently reused. CPU-only source/export preparation can proceed. No thresholds, research-contract text or training method were modified.


### Stay evidence diagnosis after supplement 15

只读既有记录诊断见 `results/diag_stay_existing_s15_compare_02.json`；首条 `_01` 参数错误失败记录保留。新 FP203/204 的 Stay 尝试为 55/51；可查 w1 排序证明 40/32 份，远到近尾序列验证 40/32，实际原生尝试与记录顺序相符 40/32；未到 w1 的尝试不补判，因此没有发现补充11未接入。原规则在 w0 未 fallback 时保留计划 w1 首点，其余才按实际 w0 距离排序，并非每次重复全池最远点。新 FP203 证据阶段仅两例 CreditCard：00017 水平位移 0.21078948194113034 m，move3 证据 146、锚点掩码 69/77；旧独立通过 Book 00001 移动 0.07780965510747653 m，move1 证据 1370。新 FP204 CellPhone 00043 移动 0.0598363876361773 m，六个已观察动作两边锚点掩码均0；旧独立通过同类00004移动 0.2826513530216032 m，move1掩码 397/369、证据 766。两轮视角、候选组合不同，旧对照仅含独立通过样例，不能作受控因果归因；证据支持先改善 Stay 组合的观察可见性，不能只继续加大位移。G01 基准后的最小顺序建议：仅对已有 Stay 组合按两世界各自最好一步预测投影面积的较小值降序优先；组合内远点规则、全部原生点、Flip顺序及实际尝试配额保持。此建议未实现或运行，真实掩码250、证据通道差10、前缀差3等阈值均不改。

### G01 first benchmark submitted

PBS707807.hopper-m-02 is the first G01 concurrency benchmark: oneGPU/12CPU, premium/autox, no host pin, walltime02:11:10. Its requested ceiling reserves 2.186111111111111 chargedGPUh; actual runtime will replace this only after final PBS accounting. It repeats the same six training scenes under2/4/6 concurrency; each GPU will first pass the original render gate. No additional bulk jobs are submitted before actual throughput/stability results.

Current accounted-or-reserved subbudgets are {"E2E": 1.3611111111111112, "G01": 2.186111111111111, "T01v2": 9.314166666666667}; shared phase 24.54472222222222/60. The earlier S15 accounting measurements remain preserved, but the budget's strict producer check only recognizes its canonical file-path invocation; t01v2_flip_s15_accounting_canonical now supplies that original-entry final record without changing any accounting tool.

Receipt: logs/pbs/gen_train_concurrency_s15_01-20261109T061808Z-d55c575e. Code and CPU evidence were pushed before submission (19d8c4a). Queue wait remains pending until PBS records a start/final history.

### G01 first start failed before render gate; namespace fixed

PBS707807.hopper-m-02 stopped before launching its render gate or any scene. The shared historical child launcher only accepted pilot_pairs_ names, while G01 requires gen_train_. The original orchestration_failed result, stack trace and all unstarted assignments are retained. PBS measured allocation: 0.0008333333333333334GPUh, charge multiplier1.0, charged0.0008333333333333334GPUh; queue wait5.2minutes (source:results/g01_concurrency_failed_start_accounting_01.json).

G01 now has its own non-overwriting gen_train_ launcher with the same run_logged/seed0/environment/private-session/cleanup contract; shared historical tools are unchanged. Regression fixtures exercise the actual gate and scene names and every existing evidence collision path.60 focused G01/resource/source tests passed (g01_launcher_fix_cpu_01). After audit/push, retry the same benchmark with a new name; no active job was canceled.

### G01 first provisional LookStay images

The running c2 FloorPlan7 source reports LookStay candidates floorplan7_00001 and00008. Existing images were hash-bound to archived live journal prefixes and immutable gate/admission/submission identity, then published in logs/review/g01_provisional_lookstay_01. These remain pending ended-source independent acceptance; they are not counted as qualified pairs yet. Logged evidence:diag_g01_provisional_lookstay_01.


G01 2026-11-09T06:47:13.561304+00:00，作业 707841.hopper-m-02（`diag_g01_live_02`）：gen_train_concurrency_s15_02_c2_floorplan1：完整行尝试 34，source-reported {"direct_flip": 1, "direct_stay": 6, "look_flip": 0, "look_stay": 0}；started_complete_journal_prefix；gen_train_concurrency_s15_02_c2_floorplan7：完整行尝试 25，source-reported {"direct_flip": 1, "direct_stay": 4, "look_flip": 0, "look_stay": 3}；started_complete_journal_prefix。未观察到启动的其余计划源全部为 unknown；以上只是源报告的完整 journal 前缀计数，不是独立验收。原始前缀和 immutable shard/plan 已由本次记录归档，未修改运行源。

### G01 retry running; next bounded substeps ready

The original start-failure707807 was retained; retry707841.hopper-m-02 (gen_train_concurrency_s15_02) passed the real image gate and began the c2 stage. It uses the same1GPU/12CPU allocation and2/4/6 benchmark, without a host constraint. The new receipt reserves2.186111111111111chargedGPUh; before submission the G01 ledger was0.0008333333333333334, T01v29.314166666666667 and the shared phase22.359444444444442. Queue history will be recorded by the original final PBS accounting.

The new opt-in Stay pilot only prioritizes existing domains by the smaller of the two planned worlds best one-action projected box areas. It keeps each domain's original far-point/fallback order, stable unknown/tie ordering, unchanged Flip round order and actual Flip>=Stay quota. No point or threshold is added or removed. CPU suites passed:[["t01v2_stay_priority_cpu_01", 13], ["t01v2_stay_priority_regressions_01", 613], ["smoke_e2e_look_plan_cpu_01", 39]]. The same five fixed LookFlip E2E inputs also passed the real read-only source preflight; that does not claim any new render/cache/credit result.

Next GPU work after this push: a bounded1GPU Stay-priority pilot in FloorPlan203, and the five-pair Look E2E export/cache check in its separate1GPU slot. E2E training still awaits the explicitly listed credit definitions. G01 benchmark sources remain on their already-running unchanged producer.
