# S04 original unsealed evidence archive

Lossless preservation of originally unsealed failed child evidence only; no source repair, independent pass, cache or training authorization.

Original PBS job_state F and obittime authenticate allocation end. They do not repair the missing child identity/cleanup proofs or independently confirm every native PID disappeared.

Original parent cleanup errors:
- Unverified child identity/cleanup: gen_test_s16_round2_s04_c6_floorplan426
- Unverified child identity/cleanup: gen_test_s16_round2_s04_c6_floorplan425
- Unverified child identity/cleanup: gen_test_s16_round2_s04_c6_floorplan226
- Unverified child identity/cleanup: gen_test_s16_round2_s04_c6_floorplan326

| Scene | Original error | Original audit |
|---|---|---|
| FloorPlan426 | OSError: [Errno 122] Disk quota exceeded | FAIL retained; _meta absent |
| FloorPlan425 | KeyboardInterrupt: Interrupted by signal 15 | FAIL retained; _meta absent |
| FloorPlan226 | OSError: [Errno 122] Disk quota exceeded | FAIL retained; _meta absent |
| FloorPlan326 | OSError: [Errno 122] Disk quota exceeded | FAIL retained; _meta absent |

No original result, log, metadata or failure status was modified.
