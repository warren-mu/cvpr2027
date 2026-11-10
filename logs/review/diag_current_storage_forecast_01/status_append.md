存储只读测量与规划：`diag_current_owned_storage_01`（98.3秒，最多2路du；原结果未修改）。完整路径/SHA见 `logs/review/diag_current_owned_storage_01/report.json`；本次规划修正见 `logs/review/diag_current_storage_forecast_01/report.json`。单位为十进制GB。

| 原任务独有 scratch（父与全部子目录） | 当前留存 GB |
|---|---:|
| gen_test_s16_round2_s01 | 12.806 |
| gen_test_s16_round2_s02 | 10.947 |
| gen_test_s16_round2_s03 | 11.502 |
| t02_export_train_six_16 | 5.606 |
| t02_export_train_six_17 | 7.625 |
| t02_export_train_six_18 | 5.023 |
| t02_export_train_six_19 | 5.826 |
| t02_export_cache_six09_14_01 | 0.709 |
| t02_export_cache_six09_14_01_retry1 | 0.128 |
| t02_export_cache_six05_08_01_retry2 | 1.001 |

这些数字是本次新建独有目录目前留存，不是历史写入峰值，也不含已有source表、home产物及共享缓存。两个31缓存均失败，其留存不用于完整31规划。

原161候选manifest中，锁内队列快照尚未领取20–26批共41对；这41对与已导出、等待正式cache的固定31对交集为0。原52正式cache通过单独保留；161或已导出均不等于训练池合格。

后续规划：41次导出按近期最大每attempt×1.3预留67.738 GB；S04修复重跑及S05–S07共22场景按最大每scene×1.3预留61.042 GB。唯一成功23对/200states缓存留存1.001 GB，按31/23×1.3假设预留1.754 GB；200states不变，因此该线性式仅为工程假设，不是实测上界。

四次短测/T0/T4/T2各最后actor按7GB共28GB；以后main12×7GB为84GB。当前pipeline+pilot规划新增158.533 GB，连以后main为242.533 GB；未包含无限重试和未来测试集完整T02导出，不能当硬上界。以root最新879.8GB/1TB观察为基线，容量不足。只报告，不删除文件，不改变记录规则；任何仅针对未来拒绝记录的精简方案等待人类/Claude决定。
