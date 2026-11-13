# Matched model and policy trajectories

Each case matches two actual episodes by dataset key, logical world, original world and initial RGB. The 255 × 255 environment frames come from the original cached table and match the recorded marked image hashes. The original observations have only the two training candidate labels when those labels satisfy their visibility and distance criteria.

**Q01–Q02.** Both arms sampled actions during update 50 of the E2E pilot. The cases include their original 255 × 255 observations and the verified 512 × 512 image history used when sampling each response. A model can choose correctly without seeing the anchor.

**Q03–Q08.** Original Qwen2.5-VL-Instruct models made the final choices in DIAG02. Observation turns were scripted. The previous view images were included in order in the final model request. Identical public requests may reuse the recorded original generation within a single model run.

`catalog.json` indexes the case folders. Each has an editable self-contained SVG, a print PNG, individual lossless input images, and a manifest with linked source lines, actions, targets, world mapping and hashes. Held-out training scenes FloorPlan15, FloorPlan204 and FloorPlan301 were excluded. All examples are drawn from training scenes, and were selected to illustrate particular behaviors.
