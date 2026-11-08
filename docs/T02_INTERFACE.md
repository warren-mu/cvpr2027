# T02 interface framework (D1 pending)

This is the user-authorized CPU preparation for T02. It does not implement or
select L/P, certify real identical prefixes, replay real witnesses, register a
VAGEN service, measure throughput, or perform the P backend's 200-sample check.

`src/envs/paired_env.py` defines `PairEnvironment.reset(pair_id, world)` and
`step(action) -> (observation, reward, done, info)`. A backend supplies a real
marked RGB plus a separate raw RGB for privileged auditing. The immutable
`PublicObservation` contains only marked RGB bytes/shape/dtype, public text and
public marks (label/anchor/bounding box). `policy_dict()` is its exhaustive
policy representation. Stable candidate names, target, kind, pair/world/episode
identities, instance-mask counts, distance and evidence proof are only in
`info`/backend records. Text formatting never includes these identities.

Each move/45-degree turn/open executes one backend action and returns a new
capture before another decision. `select` is terminal: reward 1 requires the
correct stable target, a current nonempty instance mask, legal category and
measured distance <=1.5m. Public select availability depends only on the explicit
public candidate mark role and current mask, never hidden distance/category.
A visible public candidate selected too far away or ineligible for success
receives 0, just as a valid wrong select does. Observation actions
cost 0.05 in the draft config. No AI2-THOR `visible` flag is accepted as evidence.

`EpisodeBudget` makes step/model-call/token limits explicit. All methods must
share the same config. Model calls are registered with actual sample IDs; a
shared sample can count once in each logical world, while the paired rollout
controller separately records the number of physical samples. The VAGEN
scaffold enforces response limits using an injected actual tokenizer. Direct
environment stepping is also usable for witness replay without model calls.
Draft values 8/8/256 follow T04 and do not freeze `rl_common`. The invalid-action
reward must be explicitly supplied; the draft leaves it null until T04's
"VAGEN format penalty" is resolved against the pinned valid-format bonus.

Evidence comparisons require opposite worlds, independent capture/source IDs,
exact pixel hashes and a measured relevant instance-mask union. They apply the
METRICS_SPEC >10 per-channel / area-scaled 250-pixel threshold. Evidence must
also remain visible after mark painting in the actual policy RGB. A missing
comparison yields `evidence_status=unknown`, not verified pre-evidence.
`evidence_entered` is monotonic once verified input entered the history. Real
backends must implement verification against persisted source renders, matched
viewpoint/orientation/container state, and actual mask provenance. IDs alone
are insufficient. Mock proofs are explicitly synthetic and cannot assert
`capture_provenance_verified` or `evidence_provenance_verified`.

`compare_prefix` compares the observations each backend actually returned and
does not repair differences or supply one world's pixels to the other. A
separate strict field also requires exact unmarked raw RGB for METRICS_SPEC
auditing. A production `prefix_verified_identical` additionally requires independently
verified source renders. Complete history equality/permanent divergence belongs
to the paired rollout controller (T05), including image, text, marks and actions.
Noise that causes different input is not automatically discriminative evidence.

`paired_mock.py` accepts explicit per-world fixture frames/transitions only.
`paired_vagen.py` supplies the pinned legacy BaseEnv return shape through duck
typing with injected episode selection, image conversion, response parsing and
token counting. It imports no VAGEN/torch/simulator modules. Registry, service,
serializer and real training integration remain unimplemented.

Synthetic tests are in `tests/test_paired_env.py`; they cover exact public input,
private-field separation, mask/distance/category select conditions, one-action
turns, budgets, evidence/unknown/noise distinctions, counterfeit capture rejection
and adapter contracts. Passing them would validate only this interface, never
T02's real-backend acceptance criteria. Root must commit and execute through
`run_logged`; no CPU fixture or GPU run was executed while preparing these files.
