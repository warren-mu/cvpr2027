# Frozen59 diagnostic PBS usage

Exact explicitly frozen PBS export allocations for all59 Look diagnostics, including all documented black-gate retries; excludes G01 generation and does not double-charge reused E2E707904.

New diagnostic allocations: 3.3422222222222224 charged GPU-hours. The original707904 E2E failure is reused evidence and is listed separately in report.json, not charged again here.

PBS allocation GPU-hours multiplied by scheduler queue_charge_rate; not an independent credit-ledger reconciliation

| PBS job | GPUs | Queue wait (minutes) | Allocated walltime | Charge rate | Charged GPUh | Retry of |
|---|---:|---:|---|---:|---:|---|
| 708051.hopper-m-02 | 1 | 4.167 | 00:09:28 | 1 | 0.157777778 |  |
| 708054.hopper-m-02 | 1 | 32.783 | 00:01:39 | 1 | 0.027500000 |  |
| 708071.hopper-m-02 | 1 | 30.000 | 00:10:38 | 1 | 0.177222222 |  |
| 708121.hopper-m-02 | 1 | 4.650 | 00:01:24 | 1 | 0.023333333 | 708054.hopper-m-02 |
| 708146.hopper-m-02 | 1 | 0.700 | 00:19:52 | 1 | 0.331111111 | 708121.hopper-m-02 |
| 708196.hopper-m-02 | 1 | 0.967 | 00:00:40 | 1 | 0.011111111 |  |
| 708217.hopper-m-02 | 1 | 5.400 | 00:00:39 | 1 | 0.010833333 | 708196.hopper-m-02 |
| 708235.hopper-m-02 | 1 | 0.333 | 00:00:41 | 1 | 0.011388889 | 708217.hopper-m-02 |
| 708241.hopper-m-02 | 1 | 1.567 | 00:02:01 | 1 | 0.033611111 |  |
| 708242.hopper-m-02 | 1 | 1.700 | 00:02:00 | 1 | 0.033333333 |  |
| 708245.hopper-m-02 | 1 | 3.117 | 00:03:54 | 1 | 0.065000000 |  |
| 708258.hopper-m-02 | 1 | 0.667 | 00:00:31 | 1 | 0.008611111 | 708235.hopper-m-02 |
| 708290.hopper-m-02 | 1 | 7.833 | 00:06:56 | 1 | 0.115555556 | 708258.hopper-m-02 |
| 708293.hopper-m-02 | 1 | 10.600 | 00:31:07 | 1 | 0.518611111 | 708241.hopper-m-02 |
| 708299.hopper-m-02 | 1 | 10.150 | 00:24:31 | 1 | 0.408611111 | 708242.hopper-m-02 |
| 708358.hopper-m-02 | 1 | 13.350 | 00:00:22 | 1 | 0.006111111 |  |
| 708386.hopper-m-02 | 1 | 1.517 | 00:17:18 | 1 | 0.288333333 | 708358.hopper-m-02 |
| 708455.hopper-m-02 | 1 | 5.500 | 00:00:23 | 1 | 0.006388889 |  |
| 708475.hopper-m-02 | 1 | 2.817 | 00:00:19 | 1 | 0.005277778 |  |
| 708547.hopper-m-02 | 1 | 9.633 | 00:17:32 | 1 | 0.292222222 | 708455.hopper-m-02 |
| 708549.hopper-m-02 | 1 | 9.983 | 00:00:23 | 1 | 0.006388889 | 708475.hopper-m-02 |
| 708566.hopper-m-02 | 1 | 0.617 | 00:19:29 | 1 | 0.324722222 | 708549.hopper-m-02 |
| 708720.hopper-m-02 | 1 | 0.500 | 00:28:45 | 1 | 0.479166667 |  |

Exact queue seconds, terminal PBS fields, original canonical accounting references and failed-job status are retained in report.json.

Actual-or-reserved phase totals from a fresh budget_state (unfinished allocations retain full requested ceilings): {"E2E": 1.4341666666666666, "G01": 4.252777777777777, "T01v2": 12.983055555555556}; shared 30.35333333333333/60.

Billing does not promote unknown/failed diagnostic rows or prove a frozen training pool.
