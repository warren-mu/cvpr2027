# S06 independent construction check

Original independent construction checkpoint only; not T02/cache/training acceptance. No rule or source changed.

| Class | Original physics true | Original physics false |
|---|---:|---:|
| direct_flip | 0 | 0 |
| direct_stay | 2 | 1 |
| look_flip | 1 | 0 |
| look_stay | 1 | 0 |

Reasons are counted only for journal rows already classified look_stay. Earlier unclassified failures remain unknown for actual class. Original Look-pool aggregates include Flip and Stay; proposal-mode counts are preserved separately and are not used to invent class-specific reasons.

Classified LookStay rejection reasons: {"one_action_evidence_unavailable": 133, "terminal_interruption": 1}
Unclassified Look-pool attempts: 494

Original review files:
- logs/review/t01v2_checkpoint_test_s16_s06_01/00000_look_stay.png (floorplan229_00092)
- logs/review/t01v2_checkpoint_test_s16_s06_01/00001_look_flip.png (floorplan229_00163)
- logs/review/t01v2_checkpoint_test_s16_s06_01/00002_direct_stay.png (floorplan328_00001)
- logs/review/t01v2_checkpoint_test_s16_s06_01/00003_direct_stay.png (floorplan329_00115)
- logs/review/t01v2_checkpoint_test_s16_s06_01/00004_direct_stay.png (floorplan429_00089)
