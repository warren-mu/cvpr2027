# STATUS（共享黑板 —— 所有窗口都读写这里）

当前阶段：T00b + T01（并行执行；T00 不重做）｜ 下一关卡：D1 选观察后端 → 周一（11/9）晚去留检查点 ｜ 更新：2026-11-08

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
