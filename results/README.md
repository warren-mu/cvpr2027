# results/

只能由 `scripts/run_logged.py` 运行的实验写入；Edit/Write 和 shell 重定向都会被 hook 拦截。

每个文件格式：
```json
{
  "metrics": {"ours_miou_ade20k": 47.31, "ours_miou_ade20k_std": 0.21, "baseline_miou_ade20k": 45.02},
  "_format": {"ours_miou_ade20k": ".1f", "ours_miou_ade20k_std": ".2f"},
  "notes": "mean of 3 seeds (0,1,2); val split for HP selection",
  "_meta": "由 run_logged.py 自动写入：命令、git commit、日志路径、exit_code、metrics_sha256"
}
```
key 只能包含字母、数字、`_ . : -`。论文中用 `\res{ours_miou_ade20k}` 引用。
旧结果不会被覆盖，而是归档到 `results/_archive/`（负结果也要保留）。
