# PBS queue wait and recorded charges

Times are PBS server Asia/Singapore (+08:00). Exact values and raw timestamp strings are retained in the JSON.

| Job | GPUs | Node | qtime (raw) | stime (raw) | Wait seconds | Wait minutes | Actual walltime | Allocation GPUh | Recorded rate | Charged GPUh |
|---|---:|---|---|---|---:|---:|---|---:|---:|---:|
| 706354.hopper-m-02 | 2 | unknown | Sun Nov  8 13:53:40 2026 | unknown | unknown | unknown | unknown | 0 | 1 | 0 |
| 706682.hopper-m-02 | 8 | unknown | Sun Nov  8 18:15:59 2026 | unknown | unknown | unknown | unknown | 0 | 1 | 0 |
| 707255.hopper-m-02 | 4 | hopper-14 | Mon Nov  9 03:45:25 2026 | Mon Nov  9 05:49:36 2026 | 7451 | 124.183333333 | 00:12:08 | 0.808888888889 | 1 | 0.808888888889 |
| 707257.hopper-m-02 | 2 | unknown | Mon Nov  9 03:47:39 2026 | unknown | unknown | unknown | unknown | 0 | 1 | 0 |
| 707307.hopper-m-02 | 1 | hopper-22 | Mon Nov  9 05:04:10 2026 | Mon Nov  9 05:08:06 2026 | 236 | 3.93333333333 | 00:00:19 | 0.00527777777778 | 1 | 0.00527777777778 |
| 707309.hopper-m-02 | 1 | hopper-22 | Mon Nov  9 05:06:29 2026 | Mon Nov  9 05:08:52 2026 | 143 | 2.38333333333 | 00:00:20 | 0.00555555555556 | 1 | 0.00555555555556 |
| 707310.hopper-m-02 | 1 | hopper-22 | Mon Nov  9 05:08:05 2026 | Mon Nov  9 05:09:18 2026 | 73 | 1.21666666667 | 00:00:19 | 0.00527777777778 | 1 | 0.00527777777778 |
| 707329.hopper-m-02 | 1 | hopper-25 | Mon Nov  9 05:13:48 2026 | Mon Nov  9 05:38:53 2026 | 1505 | 25.0833333333 | 00:00:23 | 0.00638888888889 | 1 | 0.00638888888889 |
| 707330.hopper-m-02 | 1 | hopper-25 | Mon Nov  9 05:15:05 2026 | Mon Nov  9 05:39:21 2026 | 1456 | 24.2666666667 | 00:00:19 | 0.00527777777778 | 1 | 0.00527777777778 |
| 707331.hopper-m-02 | 1 | hopper-25 | Mon Nov  9 05:17:21 2026 | Mon Nov  9 05:39:45 2026 | 1344 | 22.4 | 00:00:20 | 0.00555555555556 | 1 | 0.00555555555556 |
| 707341.hopper-m-02 | 1 | hopper-27 | Mon Nov  9 05:44:34 2026 | Mon Nov  9 05:47:35 2026 | 181 | 3.01666666667 | 00:17:54 | 0.298333333333 | 1 | 0.298333333333 |
| 707342.hopper-m-02 | 1 | hopper-17 | Mon Nov  9 05:46:41 2026 | Mon Nov  9 06:04:30 2026 | 1069 | 17.8166666667 | 00:00:19 | 0.00527777777778 | 1 | 0.00527777777778 |
| 707343.hopper-m-02 | 1 | hopper-22 | Mon Nov  9 05:48:22 2026 | Mon Nov  9 06:04:47 2026 | 985 | 16.4166666667 | 00:19:59 | 0.333055555556 | 1 | 0.333055555556 |
| 707350.hopper-m-02 | 1 | hopper-18 | Mon Nov  9 06:11:14 2026 | Mon Nov  9 06:13:55 2026 | 161 | 2.68333333333 | 00:06:32 | 0.108888888889 | 1 | 0.108888888889 |

Unique-job allocation GPUh: 1.58777777778.
Recorded-rate charged GPUh: 1.58777777778; known subtotal: 1.58777777778.

Cancelled never-started jobs retain a null wait; their GPU count is requested, not allocated. Missing rates stay unknown. Repeated final records are counted once only when they agree.
These are supplied completed-job costs, not a remaining-budget calculation. Pending reservations and actual project-credit reconciliation are outside this table.

## Accounting sources

- 706354.hopper-m-02: [t01v2_single_cancel_accounting](../../results/t01v2_single_cancel_accounting.json); wait unavailable: `run_count_not_one_total_wait_unknown`
- 706682.hopper-m-02: [t01v2_single_cancel_accounting](../../results/t01v2_single_cancel_accounting.json); wait unavailable: `run_count_not_one_total_wait_unknown`
- 707255.hopper-m-02: [t00b_monitor23_p3_accounting](../../results/t00b_monitor23_p3_accounting.json)
- 707257.hopper-m-02: [t01v2_single_cancel_accounting](../../results/t01v2_single_cancel_accounting.json); wait unavailable: `run_count_not_one_total_wait_unknown`
- 707307.hopper-m-02: [t01v2_single_watch_01_accounting](../../results/t01v2_single_watch_01_accounting.json)
- 707309.hopper-m-02: [t01v2_single_watch_01_accounting](../../results/t01v2_single_watch_01_accounting.json)
- 707310.hopper-m-02: [t01v2_single_watch_01_accounting](../../results/t01v2_single_watch_01_accounting.json)
- 707329.hopper-m-02: [t00b_monitor22_p5_accounting](../../results/t00b_monitor22_p5_accounting.json)
- 707330.hopper-m-02: [t00b_monitor22_p5_accounting](../../results/t00b_monitor22_p5_accounting.json)
- 707331.hopper-m-02: [t00b_monitor22_p5_accounting](../../results/t00b_monitor22_p5_accounting.json)
- 707341.hopper-m-02: [t01v2_single_watch_04_accounting](../../results/t01v2_single_watch_04_accounting.json)
- 707342.hopper-m-02: [t01v2_single_watch_04_accounting](../../results/t01v2_single_watch_04_accounting.json)
- 707343.hopper-m-02: [t01v2_single_watch_05_accounting](../../results/t01v2_single_watch_05_accounting.json)
- 707350.hopper-m-02: [t00b_monitor25_p1_accounting](../../results/t00b_monitor25_p1_accounting.json)
