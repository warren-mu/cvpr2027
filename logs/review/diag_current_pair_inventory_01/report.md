# Current construction inventory and next bounded batch

Unique independently constructed pairs: 53. Counts: {"direct_flip": 5, "direct_stay": 17, "look_flip": 27, "look_stay": 4}.

Original proposal deduplication is unchanged. Historical100-degree Direct and nonfixed-label LookStay are not upgraded. Full T02/cache/balancing acceptance is not established here; the original17 census remains separate. No pool is frozen.

| Scene | DirectFlip | DirectStay | LookFlip | LookStay |
|---|---:|---:|---:|---:|
| FloorPlan1 | 2 | 6 | 0 | 0 |
| FloorPlan7 | 1 | 4 | 0 | 3 |
| FloorPlan203 | 0 | 3 | 19 | 0 |
| FloorPlan204 | 2 | 3 | 1 | 0 |
| FloorPlan301 | 0 | 1 | 7 | 1 |

Inventory entries above8 per scene/class remain visible and require the original pool-freeze balancing step.

Next batch: FloorPlan8, FloorPlan9, FloorPlan10, FloorPlan205, FloorPlan206, FloorPlan207, FloorPlan302, FloorPlan303, FloorPlan304, FloorPlan402, FloorPlan403, FloorPlan404.
One GPU, concurrency6, each scene a full1200s grant; runner2740s, parent2860s, PBS at least00:48:10. Reserved charge upper bound at the original rate: 0.8027777777777778 GPU-hours.
These are launcher execution caps, not predicted successful-pair throughput. Root must apply the existing live budget guard before submission. This expansion round remains capped at20 GPU-hours; no promise to finish1700 pairs.
