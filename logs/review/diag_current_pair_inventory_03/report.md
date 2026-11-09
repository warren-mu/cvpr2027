# Frozen morning construction inventory

Unique independently constructed pairs: 76. Counts: {"direct_flip": 12, "direct_stay": 28, "look_flip": 28, "look_stay": 8}.
Look pairs: 36; the existing31-pair full-table diagnostic cohort remains unchanged; 5 new Look pairs are separately awaiting full-table diagnostics.

Original proposal deduplication is unchanged. Historical100-degree Direct and nonfixed-label LookStay are not upgraded. Full T02/cache/balancing acceptance is not established here; no training pool is frozen. Later sources are outside this morning cutoff.

| Scene | DirectFlip | DirectStay | LookFlip | LookStay |
|---|---:|---:|---:|---:|
| FloorPlan1 | 2 | 6 | 0 | 0 |
| FloorPlan7 | 1 | 4 | 0 | 3 |
| FloorPlan8 | 0 | 0 | 0 | 0 |
| FloorPlan9 | 0 | 0 | 0 | 0 |
| FloorPlan10 | 1 | 0 | 0 | 0 |
| FloorPlan203 | 0 | 3 | 19 | 0 |
| FloorPlan204 | 2 | 3 | 1 | 0 |
| FloorPlan205 | 0 | 0 | 0 | 0 |
| FloorPlan206 | 0 | 2 | 0 | 0 |
| FloorPlan207 | 0 | 3 | 0 | 4 |
| FloorPlan301 | 0 | 1 | 7 | 1 |
| FloorPlan302 | 6 | 5 | 1 | 0 |
| FloorPlan303 | 0 | 1 | 0 | 0 |
| FloorPlan304 | 0 | 0 | 0 | 0 |
| FloorPlan401 | 0 | 0 | 0 | 0 |
| FloorPlan402 | 0 | 0 | 0 | 0 |
| FloorPlan403 | 0 | 0 | 0 | 0 |
| FloorPlan404 | 0 | 0 | 0 | 0 |

Entries above8 per scene/class remain visible and require the original pool-freeze balancing step. Zero/excluded/partial source coverage and earlier duplicate records remain in the JSON.

## New Look pairs after the frozen31 census

| Scene | Pair | Class | Checkpoint |
|---|---|---|---|
| FloorPlan207 | floorplan207_00009 | look_stay | results/t01v2_checkpoint_g01_next12_01.json |
| FloorPlan207 | floorplan207_00017 | look_stay | results/t01v2_checkpoint_g01_next12_01.json |
| FloorPlan207 | floorplan207_00063 | look_stay | results/t01v2_checkpoint_g01_next12_01.json |
| FloorPlan207 | floorplan207_00137 | look_stay | results/t01v2_checkpoint_g01_next12_01.json |
| FloorPlan302 | floorplan302_00088 | look_flip | results/t01v2_checkpoint_g01_next12_01.json |
