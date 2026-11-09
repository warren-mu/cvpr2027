# E2E 报告：历史 Direct 流程完成，Look 补跑仍受完整视角规则阻塞

## 当前状态更新（2026-11-09 夜间）

历史 100° Direct 流程已完成步骤 1–5。人类已批准在固定 60° Look 池上补跑 T0/T3，且 T05 §1b 已冻结；当前阻塞是原定五对中的首对在完整视角导出时触发原 Look non_anchor_divergent 规则，尚未得到该固定池的完整表，后续步骤未在新池上运行。不会自行替换样本或改变阈值。

旧 Direct T0/T3 使用奖励端 KL；它们不是新 §1b 的同配置基线。新共同配置为 actor KL loss (low_var_kl, 0.001)，use_kl_in_reward=false，G 只含任务奖励；原 GAE → 证据前替换 → 整批 masked whiten 一次，critic 仍用原回报。t05_frozen_training_cpu_01 已完成 CPU 回归，但不代表新配置 GPU 训练或证据前逻辑已验证。须先在同一合格 Look 池补跑 T0 五步，再跑 T3 五步。原历史费用段落仅代表其当时状态；累计费用以原 canonical PBS 记录和 STATUS 为准。

首对 floorplan203_00002 的失败发生在后续可达位置 v_5c39249b3450164be27f:h0，不是起始共享前缀：原起点 x=-4.5，失败视角 x=-4.25。原帧 000135/000138 中，锚点 Box_8e5b2c6b 在两世界的掩码均为 0；239 个像素的最大通道差 >3，全图最大差 77，集中在右下角 x=241–254、y=235–254 的一块连续地板内部区域，未落在候选物体或分割边界上。45 个非锚点物体的位置和旋转严格相同。差异块在二维投影上邻接/重叠 w1 锚点包围盒的画面边缘，w0 锚点投影则在画面左外；该区域在 w1 整体变暗。现有证据支持局部光照变化的解释，未显示非锚点位姿漂移，但仅凭 RGB、物体掩码和投影不能确定是锚点阴影、反光或具体光路。来源为 results/diag_e2e_look_non_anchor_01.json、results/diag_e2e_look_anchor_projection_01.json，原图在 logs/review/diag_e2e_look_non_anchor_01/original_frames_and_difference.png。这与旧 Direct 对的椅子微位移诊断是不同的失败。

原固定 17 对的完整视角诊断正在收尾；后续 G01 新增独立 Look 另批记录，不将冻结 17 对称为最新全部库存。以下早期阶段记录、失败和耗时保留。

# E2E 报告（第 1 步失败，后续步骤停止）
截至 2026-11-09 00:16 UTC。固定输入为人类确认 Claude 已审过的 FP203 五对 Direct 样本；未挑选子集继续。

第 1 步未通过：707395 在 hopper-24 的单卡真实出图门禁通过，首对 floorplan203_00038 导出 D=4 视角表时由 src/envs/pair_table.py::classify_views 抛出 non_anchor_divergent。此判据是两个世界的锚点掩码都为 0，但原始 RGB 最大通道差 >3。它不是非锚点物体位姿差异的直接测量；当前不能把原因归为物理漂移或阴影。尚未形成完整表，也未开始 20 状态的缓存/实时一致性抽查。失败子进程实际耗时 61.170814926037565 秒，保留 412 个动作和 413 个帧记录，清理无异常、非黑帧。原始证据在 scratch 的 smoke_e2e/smoke_e2e_views_01/707395.hopper-m-02/export_0000；结果/日志已保留，阈值未变。

第 2–5 步未运行，因此没有模型评测、T0/T3 每步耗时或 200 步训练时长依据。已通过的 55 项 CPU 表接口测试只是软件检查，不能替代真实视角表验收。现有输入均为 τ=0：证据前逻辑未覆盖。按任务的“每步失败就停下”要求，E2E 在这里停止，等审查后决定修复路径；T01v2 采集继续。

运行入口：scripts/run_logged.py 包装 python3 -u -m src.tools.smoke_e2e_views，参数见 results/smoke_e2e_views_01.json 的 _meta.command。提交记录为 results/smoke_e2e_submit_views_01.json；真实运行、门禁、CPU 契约测试分别为 results/smoke_e2e_views_01.json、results/smoke_e2e_views_01_gate.json、results/smoke_e2e_contract_cpu_01.json。计费来源 results/t00b_monitor31_accounting.json；PBS 分配 GPU 小时乘已核实 queue_charge_rate=1，非独立项目账单核对。

| PBS 作业 | GPU | 排队时间 | 实际占用 | 计费 GPU·小时 |
|---|---:|---|---|---:|
| 707395.hopper-m-02 | 1 | 14 分 23 秒（863 秒） | 1 分 24 秒 | 0.023333333333333334 |

E2E 已用 0.023333333333333334 / 4 GPU·小时。下一次 GPU 运行的用量尚不能确定：需先定位真实导出失败，当前没有后续提交。审计 t01v2_e2e_launch_audit_01：742 文件、0 FAIL。

## 人类批准补充 8 后续（2026-11-09）
原第1步失败完整保留。人类现已批准 Direct 的隐藏锚点差异状态作为分叉、不丢弃；修订代码后将重新导出同一固定五对，然后按步骤继续，不删除失败输入。

首对失败发生在已有帧 000409/000412。锚点 TissueBox 和两本候选 Book 的掩码均为 0；158 个像素的通道差超过 3，最大差 177，集中在画面右下方两把椅子及餐桌边缘。两把椅子的已记录位置分别相差约 0.122 mm、0.052 mm，分割边界也有变化。因此主要差异并不落在被锚点压住的候选本体上，更符合非候选物体的细微位姿及边缘渲染差异；仅凭现有两帧不能确认锚点阴影的因果贡献。来源：results/smoke_e2e_existing_difference_01.json 及其 run_logged 日志；未新增工具、未另行渲染。


## Supplement8/9 continuation: E2E step1 PASS

PBS707497 on hopper-26 completed the same five reviewed 100-degree Direct pairs, D=4, native checks1–4 and the approved20-state fresh-process cache smoke. Exit0, no errors, no cleanup errors. Formal T02 200-state acceptance remains unclaimed.

| Pair | Exported states | Worker seconds | Hidden-anchor divergent Direct states |
|---|---:|---:|---:|
| floorplan203_00038 | 73 | 268.8341847109841 | 4 |
| floorplan203_00039 | 73 | 268.2378776760306 | 10 |
| floorplan203_00041 | 73 | 268.1510090410011 | 7 |
| floorplan203_00080 | 73 | 268.00961209391244 | 2 |
| floorplan203_00081 | 73 | 270.50699310505297 | 1 |

These Direct divergent states retain their own world images under supplement8; they are counted, not filtered. Rendering uses supplement9 paused/frozen construction and retains readback deviations.
PBS accounting:0.4675 GPUh and67 seconds queue wait for707497. Failed707395 and black-gate707488 remain preserved;707488→707497 is recorded in the existing retry evidence.
Sources: results/smoke_e2e_views_02_r8335e243.json, results/smoke_e2e_views_accounting_02.json, results/smoke_e2e_contract_cpu_02.json (64 tests passed).
Step2 is next; steps3–5 remain unrun. T0/T3 reward confirmed by human: select correct1/wrong0, observe−0.05, invalid0, no format bonus. The five pairs are direct/tau0: **证据前逻辑未覆盖**; Look requires a later step5 run.


## Step2 PASS; original-model evaluation next

The CPU witness replay passed for both worlds of all five approved Direct pairs (results/smoke_e2e_replay_01.json). The canonical table interface suite passed64 tests (results/smoke_e2e_contract_cpu_02.json). This used no GPU. All selected pairs remain tau0: 证据前逻辑未覆盖. Next is step3 original Qwen2.5-VL-3B with obs_never/obs_always/diag_base, one episode per pair/world.

## Step3 PASS; T0/T3 training smoke next

PBS 707555 on hopper-27 completed the original Qwen/Qwen2.5-VL-3B-Instruct evaluation with exit 0. The fixed five approved 100-degree Direct pairs were evaluated in both worlds: obs_never, obs_always and diag_base each completed 10 episodes, 30 episodes in total. All episodes contain the selected object and success outcome; the retained decision traces include public observations, generated actions/token IDs, rewards, state transitions and table temporal proofs. No execution or cleanup errors were recorded. Model loading took 58.24130738002714 seconds; the run_logged interval was 372.1 seconds.

The source is results/smoke_e2e_eval_03.json, with detailed records in logs/runs/smoke_e2e_eval_03/episodes.jsonl and the matching run log/meta. This is an end-to-end smoke check, not a formal T03 acceptance or paper result. All selected pairs have tau=0: **证据前逻辑未覆盖**. Pre-evidence credit and paired-generation deduplication remain unmeasured; no T3 correctness claim is made. Steps4–5 had not started when this section was written.

Two prior attempts stopped at the black-frame gate before model evaluation; their failure records are retained. The successful attempt and both failures have complete final PBS allocation accounting:

| PBS job | Outcome | Queue wait (seconds) | Allocated GPUs | PBS walltime | PBS allocation GPU-hours |
|---|---|---:|---:|---|---:|
| 707539.hopper-m-02 | black-frame gate, exit86 | 244 | 1 | 00:00:19 | 0.005277777777777778 |
| 707548.hopper-m-02 | black-frame gate, exit86 | 525 | 1 | 00:00:21 | 0.005833333333333334 |
| 707555.hopper-m-02 | step3 PASS, exit0 | 106 | 1 | 00:06:14 | 0.10388888888888889 |

Accounting sources: results/smoke_e2e_eval_black_accounting_01.json, results/smoke_e2e_eval_black_accounting_02.json and results/smoke_e2e_eval_accounting_03_retry1.json. These report allocated GPUs multiplied by measured PBS walltime, not an independently reconciled project credit bill. The first accounting command, smoke_e2e_eval_accounting_03, failed on the login node because its non-login PATH did not contain qstat; its failed log/meta are retained, and the retry loaded the established PBS PATH. This accounting failure used no GPU.

Next: step4 T0, then step5 T3, with the human-confirmed explicit_task_zero reward policy. The current Direct-only step5 cannot cover pre-evidence logic; a later Look-pair run remains required.

## Step4 T0 PASS after colocated-worker compatibility repair

The first T0 allocation707594 failed before any update because pinned verl required the same immediate base class for colocated actor/ref and critic wrappers. The shared seed/observation-hook base fixed that interface; upstream actor/critic update implementations and training settings were preserved. The failed run remains recorded (0.08361111111111111GPUh). The real pinned-worker CPU regression passed before retry.

Retry707612 onhopper-21 passed all5updates,64episodes per update, critic_offload=false, explicit_task_zero rewards, no format bonus. Final checkpoint is on scratch; producer/run_logged exit0,errors[],cleanup_errors[]. Actual PBS allocation0.3675GPUh,queue206s. Model initialization115.94174984202255seconds is separate from the update timings below.

| Update | Generation s | Scoring s | Critic update s | Actor update s | Total s | Syntax parse failures / physical outputs |
|---:|---:|---:|---:|---:|---:|---:|
|1|20.95697337493766|61.86384365800768|33.788221759023145|60.68973210698459|177.29877089895308|0/64|
|2|10.03317175898701|33.341926231049|36.998113421024755|38.78125666594133|119.1544680770021|0/64|
|3|8.911296078003943|34.50496250006836|41.01187270495575|39.00891356705688|123.43704485008493|0/64|
|4|8.629455079091713|31.69141386600677|40.93558763200417|39.66004744300153|120.91650402010418|0/64|
|5|8.747530417982489|33.540992914000526|36.28870978101622|38.34881312190555|116.92604623490479|0/64|

The original model step3 separately recorded0/10 syntax parse failures under each of obs_never,obs_always anddiag_base;40 forced observations inobs_always are excluded from the model-output denominator. Every T0 update recorded0/64 syntax parse failures and0 syntactically valid-but-unavailable actions. Rates use the actual raw response with parse_action; they are not inferred from reward or success. Sources: results/smoke_e2e_T0_report_02.json and its authenticated original journals.

Largest recorded CUDA allocator peak allocated142252924416bytes; peak reserved163231825920bytes. These are the colocated PPO worker allocator statistics, not sampled whole-device nvidia-smi occupancy. The reserved statistic exceeds physical capacity and must not be interpreted as physical GPU consumption. This smoke succeeded with the critic on GPU; it does not prove all main-training settings fit.

Planning only: the five measured updates average131.5465668162098seconds; multiplying that mean by200 gives7.308142600900545hours, excluding all setup and checkpoint overhead. This extrapolation assumes the same Direct/tau0 short trajectories and is not a measured200-step run or a reliable bound for Look training. Each training uses one GPU; independent repeats can be packed only after their actual memory/concurrency and budget checks.

All current pairs areDirect/tau0: **证据前逻辑未覆盖**. T3 next uses the same reviewed100-degree smoke pool and common config, adding paired sampling and Q_commit scoring. It cannot certify pre-evidence credit until a separately authorized Look smoke is implemented and run.


## Step5 PASS; Direct-only E2E steps1–5 completed

PBS 707642.hopper-m-02 on hopper-30 completed all 5 real T3 PPO updates, using the same common configuration and fixed five reviewed 100-degree Direct pairs as T0. Each update contains 64 episodes / 32 sampled pairs. The producer and run_logged exited0; errors and cleanup_errors are empty. Both actor and critic checkpoints were saved on scratch. Critic offload remained false. This establishes the bounded Direct smoke pipeline; it is not formal T03/T04/T05 acceptance or a paper result.

| Update | Generation seconds | Scoring seconds | Actor+critic update seconds | Total seconds | Raw-model syntax failures / outputs |
|---|---:|---:|---:|---:|---:|
| 1 | 19.4622292999411 | 66.92308282596059 | 95.81631059700157 | 182.20162272290327 | 0/64 |
| 2 | 11.18495745700784 | 61.85113084106706 | 64.1086006090045 | 137.1446889070794 | 0/64 |
| 3 | 7.965900354087353 | 48.72496871300973 | 64.91962409997359 | 121.61049316707067 | 0/64 |
| 4 | 8.076449931017123 | 48.77525428298395 | 64.08328905899543 | 120.93499327299651 | 0/64 |
| 5 | 8.110607679001987 | 48.52600291301496 | 64.13653016707394 | 120.7731407590909 | 0/64 |

The syntax parser is the existing parse_action applied to recorded raw responses. These are per-update training-policy outputs, kept separate from original-model greedy Step3 strategies. Canonical but unavailable actions are counted separately in results/smoke_e2e_training_report_01.json; they are not syntax failures. Shared physical sample IDs are counted once in the physical-output denominator while both logical world decisions remain recorded.

| Update | T0 candidate-scoring RPCs | T3 candidate-scoring RPCs | T3 candidate action scores | T0 scoring seconds | T3 scoring seconds |
|---|---:|---:|---:|---:|---:|
| 1 | 0 | 4 | 128 | 61.86384365800768 | 66.92308282596059 |
| 2 | 0 | 4 | 128 | 33.341926231049 | 61.85113084106706 |
| 3 | 0 | 4 | 128 | 34.50496250006836 | 48.72496871300973 |
| 4 | 0 | 4 | 128 | 31.69141386600677 | 48.77525428298395 |
| 5 | 0 | 4 | 128 | 33.540992914000526 | 48.52600291301496 |

Q_commit used actual teacher-forced probabilities for every complete public select action plus EOS, normalized over the complete public candidate set. Scoring_seconds includes candidate scoring, actor/reference log-probability recomputation, critic values, advantages and related logging. Its T0/T3 difference is not a separately measured isolated Q_commit time. The source-recorded total-time ratios T3/T0 for updates1–5 are [1.0276530502670231, 1.1509823435110411, 0.9852025647144047, 1.0001529092578565, 1.0329019465556655]. Both jobs used an H200, on different nodes, so this smoke comparison is not a causal estimate of algorithm overhead.

Model/Ray initialization was 65.45255886495579 seconds, separately from the update timings. The run_logged interval was 1064.5 seconds and includes setup/finalization. Update1 reported CUDA allocator peak allocated 142252924416 bytes and peak reserved 163229728768 bytes. These are counters inside the colocated PPO process, not sampled whole-device physical occupancy; reserved bytes must not be interpreted as physical GPU use.

**证据前逻辑未覆盖。** All reviewed inputs are tau=0; no pre-evidence T3 advantage replacement or shared-prefix generation path was exercised. S1 variance remains unavailable, and the five-update run did not reach the S2 evaluation interval20. A Look-pair Step5 run remains required before claiming that coverage.

| PBS job | GPUs | Queue wait seconds | PBS walltime | PBS allocation GPU-hours |
|---|---:|---:|---|---:|
| 707642.hopper-m-02 | 1 | 161 | 00:17:46 | 0.2961111111111111 |

These are PBS allocation hours, not an independent project credit-bill reconciliation. Sources: results/smoke_e2e_T3_01.json, results/smoke_e2e_training_report_01.json, results/smoke_e2e_T3_accounting_01.json, and the matching run logs plus ppo/update_*.jsonl journals. The retained run diff contains only STATUS.md progress notes; training source/configuration were unchanged. The source audit has no FAIL and retains its dirty-tree warning. The warning is not removed or treated as a clean checkout.
