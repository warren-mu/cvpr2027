只统计完整原始日志的真实提议类型；生成器通过不等于独立验收。物理检查先执行；六关之外的目标/选择/恢复/存储拒绝保留原原因。表中“首个失败”是按物理优先的展示顺序观测到的失败，不推断未记录关卡或证据与可解性的因果先后。跨域成功单列，未知不补零。

| 批次 | 实际提议 | 关卡 | 已读尝试 | 扫描完整 | 通过 n(%) | 失败 n(%) | 未知 n(%) | 原结果 / 首个可观察失败 |
|---|---|---|---:|---|---|---|---|---|
| A | look_stay | hidden_both | 371 | True | 41 (11.05%) | 31 (8.36%) | 299 (80.59%) | {"passed_cross_over_to_direct_stay": 3, "passed_expected_class": 7, "rejected": 361}<br>{"first_observed_failure_first_evidence_ge250": 17, "first_observed_failure_hidden_both": 28, "first_observed_failure_physical_both": 208, "first_observed_failure_prefix_le3": 5, "first_observed_failure_solvable_both": 9, "no_observed_failure_in_reported_gates": 94}<br>占已读尝试比例={"first_observed_failure_first_evidence_ge250": 0.04582210242587601, "first_observed_failure_hidden_both": 0.07547169811320754, "first_observed_failure_physical_both": 0.5606469002695418, "first_observed_failure_prefix_le3": 0.013477088948787063, "first_observed_failure_solvable_both": 0.02425876010781671, "no_observed_failure_in_reported_gates": 0.25336927223719674} |
| A | look_stay | prefix_le3 | 371 | True | 36 (9.70%) | 36 (9.70%) | 299 (80.59%) |  |
| A | look_stay | first_evidence_ge250 | 371 | True | 18 (4.85%) | 17 (4.58%) | 336 (90.57%) |  |
| A | look_stay | solvable_both | 371 | True | 9 (2.43%) | 26 (7.01%) | 336 (90.57%) |  |
| A | look_stay | physical_both | 371 | True | 163 (43.94%) | 208 (56.06%) | 0 (0.00%) |  |
| A | look_stay | witness_both | 371 | True | 14 (3.77%) | 0 (0.00%) | 357 (96.23%) |  |
| A | direct_flip | physical_both | 509 | True | 132 (25.93%) | 375 (73.67%) | 2 (0.39%) | {"passed_expected_class": 19, "rejected": 490}<br>{"first_observed_failure_physical_both": 375, "no_observed_failure_in_reported_gates": 115}<br>占已读尝试比例={"first_observed_failure_physical_both": 0.7367387033398821, "no_observed_failure_in_reported_gates": 0.22593320235756384} |
| B | look_stay | hidden_both | 130 | True | 14 (10.77%) | 24 (18.46%) | 92 (70.77%) | {"rejected": 130}<br>{"first_observed_failure_first_evidence_ge250": 14, "first_observed_failure_hidden_both": 24, "first_observed_failure_physical_both": 73, "no_observed_failure_in_reported_gates": 19}<br>占已读尝试比例={"first_observed_failure_first_evidence_ge250": 0.1076923076923077, "first_observed_failure_hidden_both": 0.18461538461538463, "first_observed_failure_physical_both": 0.5615384615384615, "no_observed_failure_in_reported_gates": 0.14615384615384616} |
| B | look_stay | prefix_le3 | 130 | True | 14 (10.77%) | 24 (18.46%) | 92 (70.77%) |  |
| B | look_stay | first_evidence_ge250 | 130 | True | 0 (0.00%) | 14 (10.77%) | 116 (89.23%) |  |
| B | look_stay | solvable_both | 130 | True | 0 (0.00%) | 14 (10.77%) | 116 (89.23%) |  |
| B | look_stay | physical_both | 130 | True | 56 (43.08%) | 73 (56.15%) | 1 (0.77%) |  |
| B | look_stay | witness_both | 130 | True | 0 (0.00%) | 0 (0.00%) | 130 (100.00%) |  |
| B | direct_flip | physical_both | 425 | True | 101 (23.76%) | 323 (76.00%) | 1 (0.24%) | {"passed_expected_class": 3, "rejected": 422}<br>{"first_observed_failure_physical_both": 323, "no_observed_failure_in_reported_gates": 99}<br>占已读尝试比例={"first_observed_failure_physical_both": 0.76, "no_observed_failure_in_reported_gates": 0.23294117647058823} |
