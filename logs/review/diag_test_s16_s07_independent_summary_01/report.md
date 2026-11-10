# S07 independent construction check

Original independent construction checkpoint only; not T02/cache/training acceptance. No rule or source changed.

| Class | Original physics true | Original physics false |
|---|---:|---:|
| direct_flip | 1 | 0 |
| direct_stay | 0 | 0 |
| look_flip | 1 | 0 |
| look_stay | 1 | 0 |

Reasons are counted only for journal rows already classified look_stay. Earlier unclassified failures remain unknown for actual class. Original Look-pool aggregates include Flip and Stay; proposal-mode counts are preserved separately and are not used to invent class-specific reasons.

Classified LookStay rejection reasons: {"one_action_evidence_unavailable": 39}
Unclassified Look-pool attempts: 189

Original review files:
- logs/review/t01v2_checkpoint_test_s16_s07_01/00000_look_stay.png (floorplan330_00003)
- logs/review/t01v2_checkpoint_test_s16_s07_01/00001_look_flip.png (floorplan330_00050)
- logs/review/t01v2_checkpoint_test_s16_s07_01/00002_direct_flip.png (floorplan330_00052)
