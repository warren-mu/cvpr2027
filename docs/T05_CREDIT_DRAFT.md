# T05 CPU implementation contract — not a training-ready configuration

This is early code preparation authorized before T04 freezes its common config.
No L/P backend, real model scorer, VAGEN trainer adapter, training run, GPU smoke,
or research statistic is supplied or claimed here. `configs/t05_credit_draft.json`
explicitly disables training execution and leaves unresolved integration choices
null. It does not replace or freeze `configs/rl_common.yaml`.

## Pure credit and loss components

`src/rl/credit.py` accepts decision-level `CreditStep` records and the actual
upstream masked-GAE scalar for each decision. T0/T1 return these exact scalars;
post-evidence steps of every arm do the same. The standalone scalar GAE function
is a CPU reference only, without whitening. Production must retain the upstream
GAE and critic-return path rather than replace it with that reference.

T2 pre-evidence observation credit is supplied realized `G - Q_commit`; select
credit is realized immediate `R - Q_commit`. G already includes observation
cost, so no second lambda deduction occurs. T3/T4 compute both world components
and weight them by `(1-rho,rho)`, then assign the same computed scalar to both.
They require an aligned pair instance/decision, verified equal complete input,
one shared action/sample, equal prior and the same policy probabilities for
public action texts. Noise divergence or missing counterpart/evidence proof is
`UnsupportedCredit`, not fabricated evidence or independent credit disguised
as the paired mechanism. Invalid pre-evidence actions are also explicitly
unsupported until their advantage treatment is approved in the common config.

`pair_id` is the frozen pool identity; `pair_instance_id` identifies a particular
draw, so drawing the same pool pair twice cannot join unrelated trajectories.
All arms use 64 episodes: T0/T2 receive 64 explicit pool draws and sample one world
from rho; T1/T3/T4 receive 32 draws and execute both worlds. Pool selection is a
caller responsibility. The paired rollout code is identical across T1/T3/T4.

`ActionScores` retain stable names, current public select text, complete-action
token log probabilities, action log probability, input digest, policy/scoring
IDs, scoring contract and actual scoring-call metadata. A score must be the sum
of token log probabilities for a complete action event, including its termination
boundary. A length-normalized score or probability conditional on selecting is
rejected. `Q_commit` uses unconditioned probability mass; a missing currently
scoreable correct target contributes zero, without inventing an invisible mark.
The 1e-6 arithmetic validation tolerance checks floating-point sums only; it does
not clip, smooth or rescale probabilities.

T4 eligibility requires both branch decisions to be selects after verified
evidence and an actual fork. Stable target names align Flip and Stay terms.
Flip returns unreduced per-world log-probability hinge terms with margin 1 and
beta 0.1. Both targets must have actual scoreable public actions in each world;
there is no invented label or probability for an invisible candidate.
Stay accepts a caller-supplied normalized distribution, explicit distribution
provenance and an explicit `sum` or `half_sum` symmetric-KL convention. It does
not automatically condition raw action probabilities, intersect supports or
add epsilon. Raw scalar audit components and gradient-preserving tensor helpers
are separate. Their future trainer integration must preserve gradients through
current-policy auxiliary logits, and approve the per-world/batch reduction.
The CPU fixtures do not yet test a VAGEN loss/backward integration.

## Pair execution and public/private boundary

`src/rl/paired_rollout.py` exposes `EpisodePort`, `collect_pair` and an explicit
`T02EpisodePort` adapter to `PairEnvironment`. Only immutable `PolicyInput` is
passed to the sampling callback: actual marked RGB bytes/shape/dtype, complete
public text, public mark geometry, full previous responses and executed actions,
and the common prompt-contract identifier. Target names, pair/world IDs, Look
labels and evidence metadata never enter that callback.

Both ports independently return captures. Shared sampling requires full public
history equality, matching recorded raw-RGB hashes, and distinct verified origin
captures. It invokes the policy once and applies the same sample/action to both
worlds. Any input difference or terminal asymmetry permanently forks; the code
never substitutes one world's frame for the other's. Logical calls are counted
per episode and physical sampling calls separately. The scorer must separately
record teacher-forced Q scoring calls; these are not extra environment rollouts.
Token counts come from the actual tokenizer supplied by the policy adapter;
over-budget output becomes the common invalid action, never a partially executed
multi-action response. Decoding, system prompt and response parsing remain T04
adapter responsibilities. Any uncaught backend/policy error must be recorded by
the enclosing logged trainer; it is not returned as a successful rollout.

Unknown evidence stays unknown. T02's monotonic `evidence_entered` is distinct
from the fork flag; the latter never certifies task evidence. Real captures need
provenance, while synthetic CPU fixtures require explicit opt-in and cannot
claim real shared-prefix certification. The core raises for an unknown or
unverified pre/post-evidence classification rather than silently omit it from S1.

## S1/S2 hook order from the first training batch

`src/rl/credit_records.py` writes a new append-only, bounded, SHA-indexed journal
directory. It refuses reuse. A production integration should place these raw
files on scratch and include the small index/hash references in its logged result.

1. Before any rollout, fix a training-scene holdout manifest and frozen common
   config hash. `TrainingHoldout.load` checks scene/test separation and exclusion
   from optimization pairs and frozen test pairs, then records manifest and
   partition-set hashes. The caller must read these sets from the real frozen
   manifests, not make them up. The raw heldout episode list includes pair,
   scene, world and seed. No frozen test set is used for S2.
2. After each full 64-episode batch, obtain actual upstream GAE, realized returns
   and complete credit/evidence records. Call `begin_update` before optimization,
   supplying all decisions, per-episode budget/completion summaries and exact
   applied advantages. It checks complete episode/decision coverage. Every
   pre-evidence decision is written, including both paired worlds and unaveraged
   world credit components. No per-world normalization may break equal T3/T4
   advantages. Zero-pre-evidence batches still get explicit begin/completion
   records. Synthetic data requires a separate explicit fixture-session flag.
3. Call `finish_update` after the actual completed rollout-batch PPO actor/critic
   cycle, or pass an error without incrementing completion. This count is NOT
   the configured total, a rollout announcement, or each inner AdamW minibatch
   call. Existing configured 21 / completed 20 semantics therefore trigger S2
   at the real twentieth completed PPO cycle. Raw S1 records join to the append-only
   `update_end` by update-attempt ID, retaining failed optimizer attempts too.
4. Every 20 completed cycles, `finish_update` returns a fixed-holdout evaluation
   request. Record per-episode booleans and original identities with `record_s2`;
   incomplete/error evaluations stay explicit and cannot become a success-rate
   point. Resolve the due evaluation or record its failure before another update.
   The manifest hash is checked again. No synthetic step-zero or final endpoint
   is appended, and no AUC/statistical convention is chosen.
5. Close the recorder in the trainer's finally path. Its index exposes unfinished
   updates/evaluations and hashes all chunks. Complete recording is not successful
   optimization or experiment acceptance. The enclosing command still writes its
   ordinary `results/<name>.json` and `FINAL_METRICS_JSON` under `run_logged`.

S1 records contain arm, seed, rho, draw/episode/world/decision identity, evidence
proof, exact input/sample IDs, G/R/Q, upstream GAE, both unaveraged world terms,
prior weights, raw and applied advantages, scoring provenance and transformation
definition. No empirical-variance ddof, weighting/decomposition convention,
success-rate aggregate or AUC is computed. Those remain independently recomputable.

## Validation scope

`tests/test_credit.py` contains only hand-constructed CPU fixtures: GAE reference,
lucky-guess and unnecessary-observation cases, rho weighting, unchanged baseline
GAE, full action probability mass, stable-name hinge/KL, strict shared sampling,
permanent noise fork, 64-episode budgets, first-update S1 coverage, actual-update
S2 scheduling and heldout leakage rejection. Run after commit through the root's
`run_logged` CPU harness. Only AST/static inspection was performed while writing
these files; no test pass, real rollout or training claim is made by this draft.
