# Framework integrations and CI

QuotientGuard separates **source-bound reconstruction** from **explicit
production coefficient capture**. A method name, variable name, successful
replay, or GitHub PR does not prove which estimator actually ran. See
[REFERENCE.md](REFERENCE.md) for certificate meanings and
[OBJECTIVE_AUDIT.md](OBJECTIVE_AUDIT.md) for the optional PPO oracle and
runtime-confirmation gates.

## Framework-neutral capture

For moving implementations (verl, OpenRLHF, custom trainers, newer TRL),
record the coefficient at the actual loss/gradient contribution point.
Do **not** guess upstream internals.

```python
from quotientguard import VerlRuntimeCapture

capture = VerlRuntimeCapture(
    policy_snapshot_id="step-1200",
    framework_version="your-verl-version",
)
callback = capture.callback()

# Inside the trainer at the state-consuming point:
callback(
    batch_id="batch-17",
    sample_index=0,
    class_id="answer:4",
    trajectory_id="trajectory-sha256",
    realized_coefficient=H_i,
    state_consuming_point="loss_contribution",
    score=score_sketch,
)
certificate = capture.certificate()
```

Also available: `OpenRLHFRuntimeCapture` and
`CallbackRuntimeCapture` for arbitrary frameworks. The caller supplies
the realized scalar; these adapters do **not** automatically infer or
certify third-party code defects.

For PPO objective-aware comparisons, use `ObjectiveRuntimeCapture` and
`ObjectiveTrace` with the actual loss graph. A synthetic
`CallbackRuntimeCapture` coefficient cannot substitute for production
backward/reconstruction validation.

## Frozen TRL reconstruction (historical research path)

Only this exact TRL source family is supported by the reconstructed
sequence-level GRPO/BNPO/Dr.GRPO adapter:

- commit: `a98fa6a4428f9aae58dfb26d729d7437f662f27a`;
- `trl/trainer/grpo_trainer.py` SHA-256:
  `e9f5ad165c620c8da1eceade84b00e405c76691d9af363681e8f7cbbfae8a709`;
- sequence importance sampling, `beta=0`, no extra entropy/off-policy/vLLM
  multiplier in the reconstructed coefficient, standard two-sided clipping.

Both commit **and** source hash must match. Other TRL versions fail closed.

```bash
quotientguard verify-source /path/to/trl/trainer/grpo_trainer.py
quotientguard check fixtures/paper/a1_equal_length_clipping.json
quotientguard explain fixtures/paper/a1_equal_length_clipping.json
```

If the exact source is installed:

```python
from quotientguard import TRLRuntimeCapture

capture = TRLRuntimeCapture.from_installed(
    policy_snapshot_id="checkpoint-1200",
    config={
        "loss_type": "grpo",
        "importance_sampling_level": "sequence",
        "beta": 0.0,
        "epsilon_low": 0.2,
        "epsilon_high": 0.2,
        "batch_size": 8,
        "common_divisor": 1.0,
    },
)
```

Record at `loss_contribution` or `gradient_contribution` using
`TRLRuntimeCapture.record_sequence_sample()`. The bridge checks the
captured coefficient against the *frozen* reconstruction; a mismatch
fails closed.

For the frozen implementation, the sequence importance ratio is
`exp(mean(valid_token_log_ratios))`, not `exp(sum(valid_token_log_ratios))`.
The active policy-ascent coefficient `H_i` is normalized differently:

| Loss | Active coefficient |
| --- | --- |
| GRPO | `H_i = ratio_i * advantage_i / (B * valid_length_i)` |
| BNPO | `H_i = ratio_i * advantage_i / N_active` |
| Dr.GRPO | `H_i = ratio_i * advantage_i / (B * max_completion_length)` |

A clipped branch gives zero derivative away from its boundary; the
boundary itself is ambiguous and rejected. In the A1 fixture with
positive advantage, ratio 1.3488 is clipped (coefficient 0); ratio
0.8181 is active (`H=0.8181/8=0.1022625`). Equal length and reward do
not make the effective coefficients identical. This is a compatibility
**observation**, not a proof of a third-party bug.

## CI and regressions

```bash
python -m unittest discover -s tests -v
python -m quotientguard certify
quotientguard check fixture.json --out candidate.json
quotientguard compare baseline.json candidate.json --fail-on-new-gap
quotientguard objective-audit examples/ppo-objective.json --oracle --out objective.json
python tools/check_release_docs.py
```

Install `.[objective]` for the optional PyTorch oracle. Compatibility
policy failures occur only for thresholds the caller explicitly sets;
they do not imply degraded reward or downstream quality.

```python
from quotientguard import (
    assert_no_measured_gap,
    assert_no_new_measured_gap,
    assert_policy_passed,
)

assert_policy_passed(certificate)
assert_no_measured_gap(certificate)
assert_no_new_measured_gap(comparison)
```

GitHub Actions example, pinned to the latest **released** tag:

```yaml
- uses: ot4562-glitch/QuotientGuard@v1.1.0
  with:
    input: fixtures/estimator.json
    output: quotientguard-certificate.json
```

The composite action checks a supplied fixture; it does not instrument
or repair an arbitrary trainer automatically. The repository's
`.github/workflows/ci.yml` tests the Python/platform matrix and
objective replay, and its version metadata gate prevents a future
release version from drifting away from the README or changelog.

`quotientguard sentinel` operates on separate snapshot certificates:

```bash
quotientguard sentinel certs/step-*.json --max-gap-ratio 0.05 --require-same-estimator
```

The measured v1.0 local microbenchmark lives in
[benchmarks/README.md](../benchmarks/README.md). Release-specific
verification for v1.1.0 is in [VALIDATION_V1_1.md](VALIDATION_V1_1.md).
No source-equivalent CPU replay alone authorizes an "upstream defect
confirmed" claim.

## Source survey and controlled-callable APIs (v1.2)

The source survey never installs, imports or executes inspected repositories.
Keep investigation stores and exports outside a public Git checkout.

```bash
quotientguard survey-discover --query "RLVR language:Python" --start-date 2008-01-01 --end-date 2026-10-27 --out /local/audit/discovery.json
quotientguard survey-inventory /local/audit/discovery.json --max-repositories 50 --out /local/audit/inventory.json
quotientguard survey-screen /local/audit/inventory.json --store /local/audit/store --run-id run-1 --out /local/audit/report.json
quotientguard survey-export --store /local/audit/store --run-id run-1 --out /local/audit/export.json
quotientguard survey-compare run-1 run-2 --store /local/audit/store
```

Acquisition is bounded by request/file/byte limits and stops on API exhaustion.
The public GitHub HTTP client supports date-partitioned search and reports caps,
incomplete responses and missing scope. Metadata collection, immutable commit
pinning and Python AST inspection are separate stages. Keyword matches do not
establish RLVR relevance or independent implementation. AST rules track bounded
within-function assignments, aliases and conservative branch joins. General
interprocedural aliasing, tensor types and input reachability remain unknown. Their
output is `STATIC_RISK_CANDIDATE`, never a confirmed defect. Identical source
families do not establish independent implementations.

`SurveyStore` persists commit/config-bound checkpoints and source objects with
hash verification. Completed entries are reused; changed commits/configuration
use separate evidence. `survey-screen --retry-failed` explicitly retries
transient acquisition failures; normal resume does not repeatedly retry failures.
Unprocessed/omitted files remain visible. Config changes require another run ID.
The client makes no automatic network retries and does not log authentication.

### File-level incremental inspection

Legacy run snapshots stay immutable. To extend a partial inspection, use a new
store and explicit verified import, then opt into file-level continuation:

```bash
quotientguard survey-screen /local/audit/inventory.json --store /local/audit/round2 --run-id next --incremental --import-store /local/audit/store --import-run-id run-1 --max-files 12 --max-store-bytes 100000000 --out /local/audit/next.json
quotientguard survey-screen /local/audit/inventory.json --store /local/audit/round2 --run-id next --incremental --resume-partial --max-files 12 --out /local/audit/continued.json
```

With `--incremental`, file/total-byte limits grant additional acquisition per
step. Verified cached files are reused and do not consume that acquisition
budget. The per-file parsing bound still applies. `--remaining-path` prioritizes
an explicit omitted path; inspected coverage still reports all known omissions.
`--resume-partial` resolves the continuation chain and returns a new immutable
child `run_id` when more work remains. The original run's export is unchanged.
Retain the returned run ID for comparisons. Explicit failed-run retries also
preserve their prior snapshot. There is no automatic retry loop.

Source reuse binds repository ID, full commit, path and content hash; analysis
reuse additionally binds detector policy, implementation identity and engine
version, plus the Python parser runtime (implementation, cache tag, full version
and optimization level). Reports retain this runtime beside the analysis
version. A parser change requires a new run ID and reanalysis of verified bytes;
it does not require another source download. Old snapshots keep their original
evidence and remain readable. Analysis v3 adds this binding; older untagged
analysis is not silently reused. Increasing the acquisition budget does not
invalidate file analysis.
Each file is committed before the next source acquisition. Cumulative store
bytes/object limits include source objects and SQLite; quota exhaustion is an
explicit failure. Inventories, source bytes and AST work are bounded, and
coverage reports successful files, omitted paths and tree truncation.

The store's byte quota includes transient rollback-journal capacity. It reserves
the terminal snapshot across source publication, including both copies used by
legacy repository caching. Admission requires DELETE journaling and disabled
cache spilling; temporary SQL data stays in memory. A quota too small for schema
or recovery capacity is rejected before writing, and an existing store is never
deleted or shrunk to fit. This is a single-writer logical data-file budget;
filesystem block slack/metadata and OS process memory require separate limits.

`--max-result-bytes` bounds each repository's serialized file/candidate buffer;
`--max-inventory-bytes` bounds its path inventory (both default 4,000,000).
`--max-report-bytes` controls conservative whole-report materialization, import
row and local input admission (default 64,000,000). Result/inventory exhaustion
is explicit partial scope; report/import admission failures identify the limit
before parsing large JSON. These budgets bound data structures, not OS RSS.
Per-file AST and HTTP response budgets apply separately.

An incremental result-buffer overflow retains verified source/index/analysis
before stopping. Increase the relevant budget with `--resume-partial`, or use
explicit `--remaining-path` scope, to continue without downloading those bytes
again. An unchanged exhausted memory budget is rejected before adding another
child snapshot. Increase `--max-store-bytes` when the journal/snapshot reservation
cannot fit; the lower steady-state file total alone is insufficient capacity.

The static analyzer distinguishes raw/sanitized observations, multiplicative
masking, subtraction/exp ordering, guarded active-count denominators, tensor
means, selection and stop-gradient. These are syntax/provenance observations.
A recognized selection is not a whole-graph safety proof; unknown calls,
different masks, later arithmetic and backward hazards require actual tests.

`quotientguard.survey_classify.classify_python` and `classify_repository` report
source-bound loss components, external trainer wrappers and evaluation/data
candidates. An inspected expression graph is not a supported runtime objective
contract. Bounded component evidence does not establish an exclusive repository
role or independent implementation. Parent/source comparisons are required
before interpreting a fork's objective changes; identical hashes alone do not
establish independence. Analysis limits return explicit unsupported scope.

For reviewed, trusted integrations, `quotientguard.native_validation.validate_callable`
accepts a callable and an explicit `invoke(callable, inputs)` argument adapter.
Inputs include `current_log_probs` and a same-shaped binary `mask`; excluded-token
mutations hold active inputs fixed. Reports retain loss, backward gradient,
exact excluded zeros, dtype/device, optional adapter-supplied raw/sanitized
observations, source SHA, loaded-code/source comparison and actual-call observation.
These checks do not attest globals, transitive dependencies, input reachability
or the correctness of an adapter. They are not a sandbox. Review dependencies
first and use a separate isolated process for external code.

Source-bound native evidence is `NATIVE_CONTROLLED_REPRODUCED` with
`execution_scope="LOSS_CALLABLE"`, `trainer_executed=False` and
`semantic_confirmation_eligible=False`. AST extraction, source-equivalent and
independent Torch controls retain distinct lower grades. A callable replay is
not a full Trainer run, GPU execution or naturally observed training failure.
`not_executed_report` explicitly inventories missing branches;
`write_native_report` validates strict JSON content hashes before export.

Callers may name additional floating tensor inputs with
`gradient_inputs=("importance_weights", "auxiliary_values")`. These inputs are
cloned into independent differentiable leaves alongside current logprobs. Reports
retain their actual gradients, connectivity, dtype/device and finite status in
`input_gradients`; `finite_monitored_gradients` is unknown when a requested
gradient is missing. Nonfinite additional gradients contribute to numerical
failure stages even when forward loss and policy gradients are finite. Original
input tensors and parameter `.grad` buffers are untouched. The observations use
each input's own units and do not certify a KL/entropy/CISPO objective, AMP
unscaling, optimizer execution or natural input reachability. Only explicitly
named inputs are monitored; this is not an automatic audit of every model
parameter or transitive graph branch.
## Full-objective and controlled optimizer APIs (v1.2)

The opt-in `validate_callable(..., objective_contract=FullObjectiveSpec(...))`
retains the original callable identity, executed source hash, decorator chain,
return ancestry and strict JSON gates. It additionally compares the actual
loss and all explicit input-coordinate gradients with `audit_full_objective`.
It probes old/reference/advantage/weight/entropy connections on cloned leaves;
adapters must declare any frozen caller boundary explicitly in `invoke`.
This is a controlled probe, not an assertion that an original Trainer made
every input trainable. Optional returned `loss_terms` compare component values.
Actual floating input dtypes must match the homogeneous declared precision.
The source evidence grade and full-objective comparison remain separate: a
correct source identity does not establish correct math, and a math match does
not establish a Trainer run, optimizer step or natural-input reachability.

`validate_training_step` in `quotientguard.native_validation` executes an
explicit accumulation window on caller-supplied **disposable** parameters and
plain SGD optimizer. It clears their existing `.grad`, applies optional
autocast around the supplied loss builder, runs actual scaled backward,
`GradScaler.unscale_`, optional parameter gradient clipping, and an actual
optimizer step (including scaler skip). It records unscaled and clipped
gradients, before/after parameters, unused inputs and normal no-update cases.
Closed-form SGD updates and an independent scalar L2 clipping coefficient
check those transformations. Per-microbatch autograd partials independently
check accumulation/unscale, but share the caller's loss graph and do not prove
the loss formula; combine this with the separate full-objective native audit.
Unsupported optimizers return `UNSUPPORTED` before invoking the loss builder.
Momentum/weight decay, DDP/DeepSpeed and complete Trainer lifecycles remain
outside this update contract. No external source is imported automatically.

`validate_parameter_graph(spec, trace, actual_loss, named_parameters)` compares
the independent analytic loss-coordinate partials through a measured model VJP
with actual parameter gradients. Its evidence is `SUPPLIED_PARAMETER_GRAPH`;
the model Jacobian uses actual autograd and is not independently certified.
The input coordinates, including frozen/unused trainable inputs, must be
distinct siblings rather than ancestors/descendants or aliases. Retained named
nonleaf gradients are rejected before probes; parameter `.grad` is not
accumulated. Arbitrary autograd hooks and unlisted retained intermediates can
still execute, so callers should use a disposable graph. This API makes no
source-identity, Trainer or optimizer-step claim. A hidden parameter-only
backward branch can fail this comparison despite identical forward loss.

The gated old/rollout correction producer is also available to native
`objective_contract` comparisons. It independently recomputes the coefficient
from declared old/rollout inputs, rather than trusting an adapter-supplied
weight. `ImportanceCorrectionSpec.level="sequence"` means a sequence rejection
decision with per-token survivor weights. Keep source-pinned callable checks,
model-generated component runs, full vLLM/Trainer runs and naturally reached
failures as separate evidence levels.

For source GSPO comparisons, bind the rollout reference through
`correction="rollout_ratio", correction_placement="inside_clip"`, declare the
sequence mean ratio, its pre-surrogate mask multiplication and
`nonempty_sequence_mean` reduction explicitly. Frozen rollout references belong
at the adapter's caller boundary; live-reference probes verify the source
partial without claiming the actual rollout producer was trainable. Raw
old-policy diagnostics returned by a source callable are separate from the
primary loss contract when old logprobs are unused by that loss.

For accumulation adapters, `audit_objective_window` independently reconstructs
explicit denominator units from supplied masks before comparing effective
term-wise loss and coordinate gradients against a full-batch reference. Capture
the mask at the actual count producer as `count_masks` when later filtering
changes the loss mask. For example, a window token count produced before
overlong filtering must not be replaced by the surviving-token count. Capture
raw returned-loss gradients separately from local Trainer-scaled backward
gradients, and declare the latter's `backward_scale`. The API simulates process
reduction after local backward rounding; it does not execute collectives or
attest that an adapter supplied every rank. Preserve source identity, producer
observations, real Trainer/optimizer execution and reference replay as separate
evidence levels. See [counted window contracts](OBJECTIVE_AUDIT.md).
