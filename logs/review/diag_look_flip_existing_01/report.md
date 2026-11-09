# Existing Look Flip diagnosis

The100-degree FloorPlan203 source has no recorded catalog with nonempty Look pools on both sides; all its72 catalogs therefore lack a Flip domain. At60 degrees only1 of76 recorded catalogs has both sides; no Flip was counted, and the exact scheduling/deadline cause for that one catalog still needs tracing. FloorPlan204 Flip attempts fail before any measured hidden-prefix gate: physical validity, target ratio or realized target mode. Retained detailed target measurements explicitly include closest_ratio_below_1_5; compact rejected records without those details remain unknown. The existing proposal code uses horizontal side distance and a12-point near-pool cap, while acceptance checks3D center distance. Next pilot changes only proposal order/domain and attempt allocation; final thresholds remain unchanged.

| FOV / scene | Counted Flip / Stay | Catalogs with both Look sides / all | Uncounted native placement failures Flip / Stay | Flip rejection counts |
|---|---:|---:|---:|---|
| 100 / 203 | 0 / 58 | 0 / 72 | 0 / 12 | {} |
| 100 / 204 | 43 / 27 | 75 / 89 | 10 / 5 | {"realized_target_mode": 9, "target_uniqueness_or_ratio": 14, "terminal_interruption": 1, "w0_physical": 19} |
| 60 / 203 | 0 / 28 | 1 / 76 | 0 / 15 | {} |
| 60 / 204 | 7 / 31 | 19 / 85 | 14 / 12 | {"target_uniqueness_or_ratio": 1, "w0_physical": 6} |

The existing per-unit funnel reports contain gate-level counts and up to three recorded rejection examples. Full per-view A/B distributions and source references are retained in logs/runs/diag_look_flip_existing_01.log; this table does not infer missing scenes or outcomes.
