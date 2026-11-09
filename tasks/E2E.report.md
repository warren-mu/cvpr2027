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
