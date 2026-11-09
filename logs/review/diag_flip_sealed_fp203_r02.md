# FP20360° /707691 sealed failure funnel

The source ended incomplete after57 counted attempts, with no independently accepted pairs (checkpoint: results/t01v2_checkpoint_flip60_fp203_r02.json). Flip/Stay29/28. This table copies the authenticated original funnel; each gate column is measured separately, and cumulative pass follows the requested gate sequence. Unknown means unreached/unrecorded. Failed examples are the first up to3 actually observed failures; no examples are invented for unmeasured gates.

| Mode | Gate | Pass | Fail | Unknown | Cumulative pass | Recorded rejected examples |
|---|---|---:|---:|---:|---:|---|
| flip | hidden_both | 2 | 13 | 14 | 2 | floorplan203_00002: X=[{"w0": 973, "w1": 1550}]; RGBmax=174; observed evidence=[]; solvable=[None, None]<br>floorplan203_00004: X=[{"w0": 5, "w1": 59}]; RGBmax=197; observed evidence=[]; solvable=[None, None]<br>floorplan203_00006: X=[{"w0": 5, "w1": 0}]; RGBmax=196; observed evidence=[]; solvable=[None, None] |
| flip | prefix_le3 | 2 | 13 | 14 | 2 | floorplan203_00002: X=[{"w0": 973, "w1": 1550}]; RGBmax=174; observed evidence=[]; solvable=[None, None]<br>floorplan203_00004: X=[{"w0": 5, "w1": 59}]; RGBmax=197; observed evidence=[]; solvable=[None, None]<br>floorplan203_00006: X=[{"w0": 5, "w1": 0}]; RGBmax=196; observed evidence=[]; solvable=[None, None] |
| flip | first_evidence_ge250 | 0 | 2 | 27 | 0 | floorplan203_00032: X=[{"w0": 0, "w1": 0}]; RGBmax=0; observed evidence=[23, 43, 44, 0, 69]; solvable=[False, False]<br>floorplan203_00040: X=[{"w0": 0, "w1": 0}]; RGBmax=0; observed evidence=[39, 43, 79, 0, 69]; solvable=[False, False] |
| flip | solvable_both | 0 | 2 | 27 | 0 | floorplan203_00032: X=[{"w0": 0, "w1": 0}]; RGBmax=0; observed evidence=[23, 43, 44, 0, 69]; solvable=[False, False]<br>floorplan203_00040: X=[{"w0": 0, "w1": 0}]; RGBmax=0; observed evidence=[39, 43, 79, 0, 69]; solvable=[False, False] |
| flip | physical_both | 15 | 13 | 1 | 0 | floorplan203_00000: X=null; RGBmax=None; observed evidence=[]; solvable=[None, None]<br>floorplan203_00008: X=null; RGBmax=None; observed evidence=[]; solvable=[None, None]<br>floorplan203_00012: X=null; RGBmax=None; observed evidence=[]; solvable=[None, None] |
| flip | witness_both | 0 | 0 | 29 | 0 | No observed failing example |
| stay | hidden_both | 3 | 7 | 18 | 3 | floorplan203_00009: X=[{"w0": 5, "w1": 3}]; RGBmax=214; observed evidence=[]; solvable=[None, None]<br>floorplan203_00013: X=[{"w0": 1686, "w1": 1405}]; RGBmax=114; observed evidence=[]; solvable=[None, None]<br>floorplan203_00027: X=[{"w0": 281, "w1": 246}]; RGBmax=216; observed evidence=[]; solvable=[None, None] |
| stay | prefix_le3 | 3 | 7 | 18 | 3 | floorplan203_00009: X=[{"w0": 5, "w1": 3}]; RGBmax=214; observed evidence=[]; solvable=[None, None]<br>floorplan203_00013: X=[{"w0": 1686, "w1": 1405}]; RGBmax=114; observed evidence=[]; solvable=[None, None]<br>floorplan203_00027: X=[{"w0": 281, "w1": 246}]; RGBmax=216; observed evidence=[]; solvable=[None, None] |
| stay | first_evidence_ge250 | 0 | 3 | 25 | 0 | floorplan203_00007: X=[{"w0": 0, "w1": 0}]; RGBmax=0; observed evidence=[0, 70, 0, 0, 146]; solvable=[False, False]<br>floorplan203_00019: X=[{"w0": 0, "w1": 0}]; RGBmax=0; observed evidence=[0, 52, 0, 0, 88]; solvable=[False, False]<br>floorplan203_00049: X=[{"w0": 0, "w1": 0}]; RGBmax=0; observed evidence=[0, 53, 0, 0, 92]; solvable=[False, False] |
| stay | solvable_both | 0 | 3 | 25 | 0 | floorplan203_00007: X=[{"w0": 0, "w1": 0}]; RGBmax=0; observed evidence=[0, 70, 0, 0, 146]; solvable=[False, False]<br>floorplan203_00019: X=[{"w0": 0, "w1": 0}]; RGBmax=0; observed evidence=[0, 52, 0, 0, 88]; solvable=[False, False]<br>floorplan203_00049: X=[{"w0": 0, "w1": 0}]; RGBmax=0; observed evidence=[0, 53, 0, 0, 92]; solvable=[False, False] |
| stay | physical_both | 17 | 11 | 0 | 0 | floorplan203_00001: X=null; RGBmax=None; observed evidence=[]; solvable=[None, None]<br>floorplan203_00003: X=null; RGBmax=None; observed evidence=[]; solvable=[None, None]<br>floorplan203_00005: X=null; RGBmax=None; observed evidence=[]; solvable=[None, None] |
| stay | witness_both | 0 | 0 | 28 | 0 | No observed failing example |

Source: results/diag_flip_sealed_fp203_r02.json; unchanged original metrics and failed run retained.

Recorded PBS allocation from t01v2_flip60_fp203_r02_accounting_01.json (allocation duration and queue charge rate; no independent credit reconciliation):

| Job | GPUs | Queue wait (minutes) | Measured allocation GPU-hours | Recorded charge rate |
|---|---:|---:|---:|---:|
| 707691.hopper-m-02 | 1 | 1.0833333333333333 | 0.32666666666666666 | 1 |
