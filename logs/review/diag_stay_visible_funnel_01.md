# Stay-priority completed-unit funnel

Original source status: incomplete; exit 1. Sealed terminal, not completed collection.

仅汇总已结束、已审计来源的记录，不代表独立验收。物理检查实际先执行；未测的图像或后续关卡保留未知。连续通过要求本关及前关均实测通过；样例是每关最先出现的至多三个实际失败，可能有未知前关。X 为两世界 v0 锚点像素，Δ 为前缀最大差，e 为同一共同首步的原图/标号图证据像素（展示原图证据最多的动作）；? 为未测。

| FOV | 模式 | 关卡 | 连续通过 | 累计未知 | 本关通过/失败/未知 | 拒绝样例关键值 |
|---|---|---|---:|---:|---|---|
| 60 | flip | X 两边不可见 | 34 | 32 | 34/0/32 | 无已观测失败样例 |
| 60 | flip | 前缀差 ≤3 | 34 | 32 | 34/0/32 | 无已观测失败样例 |
| 60 | flip | 同一首步证据 ≥250 | 5 | 33 | 5/28/33 | floorplan203_00010: X=0/0, Δ=0, e=turn_left:167/167<br>floorplan203_00014: X=0/0, Δ=0, e=turn_left:137/137<br>floorplan203_00020: X=0/0, Δ=0, e=turn_left:130/130 |
| 60 | flip | 两边可解 | 5 | 33 | 5/28/33 | floorplan203_00010: X=0/0, Δ=0, e=turn_left:167/167, S=False/False<br>floorplan203_00014: X=0/0, Δ=0, e=turn_left:137/137, S=False/False<br>floorplan203_00020: X=0/0, Δ=0, e=turn_left:130/130, S=False/False |
| 60 | flip | 物理检查 | 5 | 15 | 48/18/0 | floorplan203_00000: X=?, Δ=?, e=?, P=False/?<br>floorplan203_00002: X=?, Δ=?, e=?, P=False/?<br>floorplan203_00008: X=?, Δ=?, e=?, P=False/? |
| 60 | flip | 见证回放 | 5 | 15 | 5/0/61 | 无已观测失败样例 |
| 60 | stay | X 两边不可见 | 5 | 33 | 5/27/33 | floorplan203_00003: X=1686/1405, Δ=114, e=?<br>floorplan203_00009: X=1686/973, Δ=114, e=?<br>floorplan203_00017: X=1686/318, Δ=114, e=? |
| 60 | stay | 前缀差 ≤3 | 1 | 33 | 1/31/33 | floorplan203_00003: X=1686/1405, Δ=114, e=?<br>floorplan203_00009: X=1686/973, Δ=114, e=?<br>floorplan203_00017: X=1686/318, Δ=114, e=? |
| 60 | stay | 同一首步证据 ≥250 | 0 | 33 | 0/1/64 | floorplan203_00007: X=0/0, Δ=0, e=move 3:146/146 |
| 60 | stay | 两边可解 | 0 | 33 | 0/1/64 | floorplan203_00007: X=0/0, Δ=0, e=move 3:146/146, S=False/False |
| 60 | stay | 物理检查 | 0 | 15 | 47/18/0 | floorplan203_00001: X=?, Δ=?, e=?, P=False/?<br>floorplan203_00005: X=?, Δ=?, e=?, P=False/?<br>floorplan203_00011: X=?, Δ=?, e=?, P=False/? |
| 60 | stay | 见证回放 | 0 | 15 | 0/0/65 | 无已观测失败样例 |
