# FP20460° /707668 sealed failure funnel

The source ended incomplete after10 counted attempts, with no independently accepted pairs (checkpoint: results/t01v2_checkpoint_flip60_fp204_retry1.json). Flip/Stay5/5. This table copies the authenticated original funnel; each gate column is measured separately, and cumulative pass follows the requested gate sequence. Unknown means unreached/unrecorded. Failed examples are the first up to3 actually observed failures; no examples are invented for unmeasured gates.

| Mode | Gate | Pass | Fail | Unknown | Cumulative pass | Recorded rejected examples |
|---|---|---:|---:|---:|---:|---|
| flip | hidden_both | 2 | 3 | 0 | 2 | floorplan204_00000: X=[{"w0": 89, "w1": 0}]; RGBmax=124; observed evidence=[]; solvable=[None, None]<br>floorplan204_00002: X=[{"w0": 0, "w1": 8}]; RGBmax=75; observed evidence=[]; solvable=[None, None]<br>floorplan204_00004: X=[{"w0": 51, "w1": 0}]; RGBmax=123; observed evidence=[]; solvable=[None, None] |
| flip | prefix_le3 | 2 | 3 | 0 | 2 | floorplan204_00000: X=[{"w0": 89, "w1": 0}]; RGBmax=124; observed evidence=[]; solvable=[None, None]<br>floorplan204_00002: X=[{"w0": 0, "w1": 8}]; RGBmax=75; observed evidence=[]; solvable=[None, None]<br>floorplan204_00004: X=[{"w0": 51, "w1": 0}]; RGBmax=123; observed evidence=[]; solvable=[None, None] |
| flip | first_evidence_ge250 | 1 | 1 | 3 | 1 | floorplan204_00008: X=[{"w0": 0, "w1": 0}]; RGBmax=0; observed evidence=[125, 0, 0, 205, 0, 75]; solvable=[False, False] |
| flip | solvable_both | 0 | 2 | 3 | 0 | floorplan204_00006: X=[{"w0": 0, "w1": 0}]; RGBmax=0; observed evidence=[269, 0, 0, 625, 0, 216]; solvable=[True, False]<br>floorplan204_00008: X=[{"w0": 0, "w1": 0}]; RGBmax=0; observed evidence=[125, 0, 0, 205, 0, 75]; solvable=[False, False] |
| flip | physical_both | 5 | 0 | 0 | 0 | No observed failing example |
| flip | witness_both | 0 | 0 | 5 | 0 | No observed failing example |
| stay | hidden_both | 0 | 1 | 4 | 0 | floorplan204_00007: X=[{"w0": 8, "w1": 0}]; RGBmax=75; observed evidence=[]; solvable=[None, None] |
| stay | prefix_le3 | 0 | 1 | 4 | 0 | floorplan204_00007: X=[{"w0": 8, "w1": 0}]; RGBmax=75; observed evidence=[]; solvable=[None, None] |
| stay | first_evidence_ge250 | 0 | 0 | 5 | 0 | No observed failing example |
| stay | solvable_both | 0 | 0 | 5 | 0 | No observed failing example |
| stay | physical_both | 1 | 3 | 1 | 0 | floorplan204_00001: X=null; RGBmax=None; observed evidence=[]; solvable=[None, None]<br>floorplan204_00003: X=null; RGBmax=None; observed evidence=[]; solvable=[None, None]<br>floorplan204_00005: X=null; RGBmax=None; observed evidence=[]; solvable=[None, None] |
| stay | witness_both | 0 | 0 | 5 | 0 | No observed failing example |

Source: results/diag_flip_sealed_fp204_retry1.json; unchanged original metrics and failed run retained.

Recorded PBS allocation from t01v2_flip60_fp204_retry1_accounting_01.json (allocation duration and queue charge rate; no independent credit reconciliation):

| Job | GPUs | Queue wait (minutes) | Measured allocation GPU-hours | Recorded charge rate |
|---|---:|---:|---:|---:|
| 707668.hopper-m-02 | 1 | 11.566666666666666 | 0.32666666666666666 | 1 |
