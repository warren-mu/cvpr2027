# Current owned storage

Only current retained unique scratch allocation; no historical peak, source-table reuse, shared-cache increment or scientific PASS inference

| Original run | Category | Retained GB (10^9) | Measurement status |
|---|---|---:|---|
| gen_test_s16_round2_s01 | test_generation | 12.806 | 0 |
| gen_test_s16_round2_s02 | test_generation | 10.947 | 0 |
| gen_test_s16_round2_s03 | test_generation | 11.502 | 0 |
| t02_export_train_six_16 | table_export | 5.606 | 0 |
| t02_export_train_six_17 | table_export | 7.625 | 0 |
| t02_export_train_six_18 | table_export | 5.023 | 0 |
| t02_export_train_six_19 | table_export | 5.826 | 0 |
| t02_export_cache_six09_14_01 | fresh_cache | 0.709 | 0 |
| t02_export_cache_six09_14_01_retry1 | fresh_cache | 0.128 | 0 |
| t02_export_cache_six05_08_01_retry2 | fresh_cache | 1.001 | 0 |

Planning uses the largest measured current footprint per scene/attempt plus 30%; this is not a guaranteed upper bound. S04 is included as a new rerun without deleting its prior bytes. Repeated failures and future test-table exports have no bounded count here.

{
  "scope": "planning only; observed-max extrapolation plus 30%, not a hard bound or measured future use",
  "remaining_export_attempts": 41,
  "future_test_slices": [
    {
      "slice": "s04",
      "reference": {
        "file": "configs/gen_test_s16_round2_s04_quota.json",
        "sha256": "d7232a1856fdd65e74f2ec34cd7cad69b1a7803a1204131ef97452d5e2ffbfc6",
        "bytes": 47061
      },
      "scene_count": 6,
      "scope": "new run only; S04 previous failed bytes remain retained"
    },
    {
      "slice": "s05",
      "reference": {
        "file": "configs/gen_test_s16_round2_s05_quota.json",
        "sha256": "0e07394b1cc1eae1f4dd5532b9c85dd081e17d3819310b5a6ce19129e1969056",
        "bytes": 47060
      },
      "scene_count": 6,
      "scope": "new run only; S04 previous failed bytes remain retained"
    },
    {
      "slice": "s06",
      "reference": {
        "file": "configs/gen_test_s16_round2_s06_quota.json",
        "sha256": "89cf84a0b5e6b1e1cc4c34889f4222a603e78674e8fdc84662ba37934f0168a8",
        "bytes": 47061
      },
      "scene_count": 6,
      "scope": "new run only; S04 previous failed bytes remain retained"
    },
    {
      "slice": "s07",
      "reference": {
        "file": "configs/gen_test_s16_round2_s07_quota.json",
        "sha256": "8ee8bafe1adab80f67c4482f12f73af7d1eb17188f03553c975fb482b14bbae8",
        "bytes": 46411
      },
      "scene_count": 4,
      "scope": "new run only; S04 previous failed bytes remain retained"
    }
  ],
  "retry_count_unbounded": true,
  "new_test_full_table_exports_not_included": true,
  "hard_total_upper_bound_bytes": null,
  "pilot_actor_assumption_gb_each": 7,
  "pilot_actor_count": 4,
  "pilot_actor_projection_gb": 28,
  "main_actor_assumption_gb_each": 7,
  "main_actor_count": 12,
  "main_actor_projection_gb": 84,
  "test_observed_max_bytes_per_scene": 2134319786.6666667,
  "export_observed_max_bytes_per_attempt": 1270878890.6666667,
  "future_test_reserve_bytes": 61041545898.66667,
  "future_export_reserve_bytes": 67737844872.53334,
  "one_additional_same31_cache_reserve_bytes": 922116915.2,
  "current_pipeline_plus_pilot_reserve_bytes": 157701507686.40002,
  "including_later_main_actor_reserve_bytes": 241701507686.40002
}
