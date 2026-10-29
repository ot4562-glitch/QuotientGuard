# PPO objective-aware audit

The v1 compatibility screen and v1.1 objective diagnostic answer different
questions. A coefficient spread inside a declared class remains a research
observation even when the implementation exactly follows PPO.
`COEFFICIENT_MISMATCH_OBSERVED`, `ESTIMATED_CLASS_CONSTANCY` and
`INSUFFICIENT_REPETITION` retain their v1 meanings.

## Objective and coefficient

For current log probability x, detached old log probability o, detached
advantage A, binary completion/action mask m, detached nonnegative correction w
and constant denominator D:

    r = exp(x - o)
    L = -sum(m * w * min(r*A, clamp(r, lower, upper)*A)) / D

Away from boundaries, dL/dx is zero when A > 0 and r > upper, or A < 0 and
r < lower; otherwise it is -m*w*A*r/D. Zero advantage gives zero. Ratios
different from one and unequal per-sample gradients do not imply a defect.
This uses gradient descent on the negative surrogate. v1 adapter coefficients
can use different signs/units and must be converted before comparison.

`ratio_level="sequence_product"` defines r_i = exp(sum_t m_it*(x_it-o_it)).
dL/dx_ij is m_ij times the sum of active per-token derivatives in sequence i.
This is a sequence probability product ratio, not a geometric-mean ratio or
general GRPO support. The frozen TRL adapter retains its separate semantics.

`importance_level="sequence"` takes [sequence, 1] weights; token weights use
[sequence, token]. None means unit weights. Correction is outside clipping.
Differentiable or inside-clip corrections are unsupported. Reduction is `sum`,
`active_mean` (sum of binary mask), or `explicit` with a positive finite
denominator. All-masked sum is zero; all-masked mean is undefined. Requested
replay executes that mean and records its nonfinite loss. Without execution or
observed output the undefined denominator is insufficient evidence.

`gradient_scale` is a measured constant in [0,1], applied uniformly after
backward to both expected and observed coefficients. Forward loss is unscaled.
It represents a known clipping scale in logprob derivative space, not inferred
parameter-space clipping or AMP unscaling. Callers must use the declared units.

The analytic engine uses Python float64 independently of torch. `dtype` declares
homogeneous calculation precision: float16, bfloat16, float32, float64. Exact
or near clip boundaries within four dtype epsilons of the sign-relevant bound
are insufficient because branch rounding/subgradient conventions vary. Replay
still records the executed gradient. Default absolute/relative tolerances are
.005, .04, 2e-5, 1e-10 respectively; finite nonnegative overrides are supported.
Nonfinite values and nonzero excluded-token gradients cannot be excused by these
tolerances. Mixed-precision/autocast graphs are outside this version's contract.

## API and actual backward

`ObjectiveSpec` describes math/precision. `ObjectiveTrace` contains rectangular
input tensors or matrices, optional observed coefficients/loss/parameter
gradients, and policy/model/code provenance. `audit_objective(spec, trace,
oracle=True)` lazily executes a CPU torch loss and autograd. This is a reference
replay, not execution of an external trainer. Without the oracle, there is no
torch dependency and dtype intermediates are not independently executed.

```python
import torch
from quotientguard import ObjectiveRuntimeCapture, ObjectiveSpec, ObjectiveTrace

x = torch.tensor([[0., 0.]], requires_grad=True)
ratio = x.exp()
loss = -torch.minimum(ratio, ratio.clamp(.8, 1.2)).mean()
capture = ObjectiveRuntimeCapture(
    policy_snapshot_id="step-1", model_version="model-hash",
    code_version="trainer-commit", framework="custom",
)
result = capture.record_backward(
    ObjectiveSpec(),
    ObjectiveTrace(x, torch.zeros_like(x), torch.ones_like(x), torch.ones_like(x)),
    loss, reproduction_id="batch-1-replay-1",
)
loss.backward()
```

Capture uses `autograd.grad(loss, x, retain_graph=True)` and leaves parameter
`.grad` untouched. `parameter_gradients` can carry gradients supplied after
trainer backward. Finite observed gradients do not prove unobserved parameters
are finite. No optimizer step is executed by capture.
Runtime capture checks actual graph connectivity for declared detached old
policy, advantages, importance weights and masks. Connected dependencies or
a completely detached loss remain insufficient even if current-policy
coefficients happen to match. Nonfinite actual observations keep diagnostic
priority over unsupported graph connectivity.

## Verdicts and evidence

- `EXPECTED_OBJECTIVE_EFFECT`: supplied/captured coefficients match supported
  analytic math; requested replay agrees and there is no numerical event.
  Manual input interprets only those supplied values.
- `NUMERICAL_ANOMALY_OBSERVED`: nonfinite observed loss/coefficient/parameter
  gradient, or nonfinite/underflow reference arithmetic. Events distinguish
  `SUPPLIED_INPUT`, `SUPPLIED_OBSERVATION`, `REFERENCE_REPLAY`. Nonfinite inputs
  alone do not prove a trainer's masked loss is invalid and yield insufficient
  evidence without executed anomalies. Extreme finite inputs also require
  execution when analytic range checks indicate possible overflow/underflow.
- `SEMANTIC_MISMATCH_CONFIRMED`: runtime-only after every gate below. JSON/manual
  provenance and synthetic controls cannot authorize it.
- `INSUFFICIENT_EVIDENCE`: missing observations, unsupported math, boundary
  ambiguity, unverified finite differences or contract/reconstruction uncertainty.

Runtime confirmation requires finite actual backward mismatch, analytic/torch
agreement, connected production graph, stable policy/model/code/framework IDs,
`execution_kind="FRAMEWORK_INTEGRATION"`, `synthetic_inputs=False` and a production
loss callable whose executed source file matches an independently supplied
`source_sha256`. A supplied tensor alone fails this source gate.

Also required: an `equivalence_contract` with id, version and coefficient claim;
an executed `contract_verifier(trace)` returning True; a separate
`reconstruction_loss()` whose forward and logprob derivative match production;
and at least two distinct reproduction IDs for identical spec, input, observed
coefficient, provenance and contract. Duplicate IDs or changed inputs do not
accumulate. The verifier must establish domain equivalence and applicability
of PPO math, not merely equality of arbitrary labels. Reconstruction must be
independent and preserve the production graph.

The verifier and nonsynthetic provenance are **trusted integrator assertions**.
The tool cannot derive semantic truth from arbitrary callbacks, source hashes
or IDs. Hashes cover one executed file, not transitive dependencies. Independent
execution is not remotely attested. No new integrated external framework defect
is claimed in v1.1.

## Numerical output and CLI

Torch records log ratios, exp ratios, unclipped/clipped products, importance
products, masked products, forward loss and backward gradients. It checks exp
overflow/underflow, finite-product overflow, `inf*0`, `-inf-(-inf)` and excluded
gradients. It executes literal arithmetic and never uses `nan_to_num`.
A production graph avoiding invalid masked arithmetic can be safer than the
literal replay; replay alone cannot prove an integrated framework defect.

`result.to_dict()` is strict JSON. Nonfinite floats become strings `NaN`,
`Infinity`, `-Infinity`; stage/indices evidence is retained. These strings can
also be used in input matrices. `audit_sha256` hashes the payload excluding its
own hash. `objective_report(result)` renders Markdown.

`quotientguard objective-audit INPUT [--oracle] [--json] [--out PATH]
[--markdown PATH]` exits 1 for numerical/confirmed semantic diagnostics.
Insufficient evidence exits 0 and must be explicitly checked by strict CI users.

## Certificate compatibility

Schema remains `quotientguard.certificate.v2`. Existing APIs preserve their
structure/decisions with tool version 1.1.0. Optionally calling
`FrameworkRuntimeCapture.attach_objective_audit(result)` adds the separately
versioned `objective_interpretation`. The whole certificate hash includes the
sidecar; observation/estimator/contract hashes retain their meaning. Old
certificates remain readable. Consumers requiring exact key sets should use
standalone results. Frozen `TRLRuntimeCapture` is unchanged; use standalone
results alongside it. There is no automatic v1 migration or claim promotion.

## Explicit ratio and reduction contracts (v1.2)

The published v1.1 behavior above remains the default. New modes must be
declared explicitly; an algorithm name does not select a contract.

- `ratio_level="sequence_masked_geometric_mean"` uses
  `r_i = exp(sum(m_it*(x_it-o_it))/max(1,sum(m_it)))`. Its derivative couples
  the active surrogate terms in each row and divides by the active length.
- `ratio_level="sequence_token"` has the same forward ratio, but the log ratio
  is `selected_x - stop_gradient(selected_x) + stop_gradient(sequence_mean)`.
  Its derivative is local to each token. It is distinct from the differentiable
  sequence mean. `sequence_product` continues to use the sum without division.
- `reduction="per_sequence_active_mean"` means each row's active-token mean,
  then batch mean. `fixed_length_mean` uses `B * completion_length` with a
  declared positive integer length. `sum`, `active_mean` (global active count)
  and `explicit` retain their old meanings. These primitives express declared
  GRPO/BNPO/Dr.GRPO reductions without certifying arbitrary algorithms with those
  names. The frozen TRL adapter is unchanged.
- `empty_mask_policy="undefined"` preserves undefined empty means. With
  `"zero"`, active counts are clamped to one; empty rows contribute zero and
  remain included in the batch mean. A sequence ratio with no active tokens is
  one for finite/selected inputs. Literal invalid padding can still poison it.
- `masking="literal"` preserves the v1.1 arithmetic. `"select_inputs"` selects
  excluded current/old log probabilities to zero **before subtraction**, and
  excluded advantages/correction weights before surrogate arithmetic. This
  declares a different graph; it does not repair or attest an external graph.

The independent Python derivative and Torch oracle cover these contracts and
four homogeneous CPU dtypes. Per-sequence means preserve the two successive
reductions rather than casting `B * active_count` into a low-precision divisor.
The existing sign-relevant clip-boundary ambiguity and exact excluded-zero gates
remain. The legacy `audit_objective` API does not infer differentiable corrections,
auxiliary KL/entropy losses, CISPO, mixed precision, AMP unscaling or parameter
transforms. The explicit full-objective and controlled training APIs below cover
their declared subset separately. A numerical event on an executed graph retains
priority even when its objective is unsupported.

## Explicit full-objective API (v1.2)

`FullObjectiveSpec` / `FullObjectiveTrace` and `audit_full_objective` are an
opt-in family. They do not change `audit_objective`, runtime confirmation gates,
or the published v1.1 contract. Install the `objective` extra for execution.

```bash
quotientguard full-objective-audit examples/full-objective.json --json --out full.json
```

```python
from quotientguard import (ObjectiveSpec, ObjectiveTrace, FullObjectiveSpec,
    FullObjectiveTrace, KLSpec, audit_full_objective)

spec = FullObjectiveSpec(
    policy=ObjectiveSpec(dtype="float64", masking="select_inputs"),
    kl=KLSpec(coefficient=.04, score_weight=True,
              score_detach_on_delta_clamp=True,
              delta_bounds=(-20, 20), value_bounds=(-10, 10)),
    entropy_coefficient=.01,
)
trace = FullObjectiveTrace(
    ObjectiveTrace([[-.9]], [[-1.]], [[1.]], [[1]]),
    reference_log_probs=[[-1.2]], entropy=[[.7]],
)
result = audit_full_objective(spec, trace).to_dict()
assert result["status"] == "SUPPORTED"
```

Let `x,o,A,f,H,w` denote current/old logprob, advantage, reference logprob,
supplied entropy and correction weight. `sg` removes a derivative path.
The policy ratio uses the four explicit `ObjectiveSpec.ratio_level` definitions
above. `ratio_delta_bounds` optionally clamps its log ratio before exp.
`ratio_masking="select_delta"` selects excluded differences to zero **after**
subtraction. This differs from `policy.masking="select_inputs"`, which selects
each input before arithmetic, and from `output_masking="select"`, which selects
already-computed token terms before reduction.

The supported token policy formulas are:

- `ppo_clipped`: `P = -min(r*A, clamp(r, lower, upper)*A)`.
- `cispo`: `P = -sg(clamp(r, coefficient_lower, coefficient_upper))*A*x`.
  Coefficient bounds are absolute. NeMo's two-sided `.8,1.2` and ms-swift's
  upper-only epsilon cap require different explicit bounds. Clipping a detached
  coefficient does not saturate the direct derivative with respect to `x`.
- `sapo_smooth`: `P = -A*4*sigmoid(tau*(r-1))/tau`, with `tau_positive` for
  `A>0` and `tau_negative` otherwise. Away from sign boundaries, its log-ratio
  derivative is `-4*A*s*(1-s)*r`.
- `sapo_detached`: `P = -A*sg(4*sigmoid(tau*(r-1))/tau)*x`. This is the distinct
  Trinity score surrogate; its direct current-logprob derivative is `-A*sg(gate)`.

`policy_filter` is a supplied binary, detached policy-only selection, such as
an already-computed entropy quantile filter. It does not change the ratio's
sequence-aggregation mask or the normalization denominator. This API does not
infer a quantile filter from entropy or observed gradients. A saved execution
without the actual selection mask cannot certify that path.

For KL, `direction` defines **the algebraic delta**, not an inferred sampling
distribution: `d=f-x` (`reference_minus_current`) or `d=x-f`
(`current_minus_reference`). Optional `delta_bounds` clamp this delta. Estimators
are `k1=d`, `k2=d*d/2`, and `k3=exp(d)-d-1`. Thus NeMo k1 uses
`current_minus_reference`, while its k3 uses `reference_minus_current`.
`score_weight=True` multiplies by `exp(x-sg(x))` **before** optional value
clamping. Its forward value is one; its derivative adds the estimator value to
the current-logprob gradient. `score_detach_on_delta_clamp=True` suppresses that
score path when clamping changes the delta, matching NeMo's explicit condition.
`detach_reference` is independent of the policy's old-logprob boundary.

NeMo's distinct on-policy approximation can instead declare
`KLSpec(importance_weight="current_rollout", weight_nonfinite="zero", ...)`.
It uses `exp(kl_current_logprob-rollout_logprob)` before output clamping, with
the same optional input-clamp score detachment. `rollout_log_probs` is an
explicit trace coordinate; `detach_rollout` declares its boundary. The partial
derivatives include the live weight's contribution to KL-current logprob and,
when requested, rollout logprob. `weight_nonfinite="zero"` executes the source's
explicit `nan_to_num` on that weight; it does not sanitize subsequent gradients
or convert a failed backward into a pass. Optional `kl_current_log_probs` keeps
unfiltered KL policy logprobs separate from the policy surrogate's current
logprobs, preserving both input gradients. These coordinates still require an
adapter to establish the filtering/temperature provenance of their producers.

`correction="none"` disables weights. `"weights"` multiplies `P` by an explicit
token `[B,T]` or sequence `[B,1]` nonnegative input outside the policy surrogate,
before the policy filter and reduction. `detach_importance=False` returns the
partial derivative with respect to that weight; sequence weight derivatives
sum the token contributions. This does not define the weight's upstream
rollout/current/old-logprob, temperature, truncation or sampling-processor graph.
Such a graph must be observed in a native adapter. Inside-clipping correction
and automatic algorithm-name inference remain `UNSUPPORTED`.

The additional `correction="gated_old_rollout"` producer takes old training
log density `o` and rollout sampling log density `b` directly. Set
`importance_correction=ImportanceCorrectionSpec(...)`; JSON input uses a nested
`importance_correction` object. The pinned OpenRLHF PPO contract defines
`w=exp(sg(o-b))`. Token ratio gating tests `w`; sequence ratio gating tests
`exp(sum(mask*(o-b))/sum(mask))`. Sequence gates keep distinct per-token `w`
values rather than replacing them with the geometric mean. With `mode="mask"`,
`coefficient=where(lower<=stat<=upper,w,0)`. With `mode="clip"`, the coefficient
is `clamp(w,lower,upper)`; this source supports clipping only for token ratio
gates. Coefficients multiply the already-clipped PPO surrogate, outside its
minimum, and remain detached even when the old-policy PPO path is live.

For `gating="binary_kl"`, the statistic is
`p*log(p/q)+(1-p)*log((1-p)/(1-q))`, where
`p=clamp(exp(b),1e-6,1-1e-6)` and `q=clamp(exp(o),1e-6,1-1e-6)`.
For `"tv"` it is `abs(p-q)`. These are sampled-token Bernoulli statistics, not
categorical KL/TV certificates. Sequence divergence gates average their token
statistics with the action mask. Divergence modes require lower bound zero;
their surviving coefficients still use token `w`, not the divergence statistic.

This producer is restricted to token PPO and detached token coefficients;
sequence refers to the gate level. Its literal masked mean does not invent an
empty-row denominator. Empty-row gate NaN can be contained by rejection, while
the policy reduction's own declared empty-mask rule still applies. Excluded
NaN can poison a sequence gate before rejection; such a controlled mutation is
not evidence of naturally generated bad rollout inputs. Boundary checks apply
to active policy positions. Normal identical-distribution zero divergence does
not fail a zero lower bound; a zero upper bound remains a threshold boundary.

Run `quotientguard full-objective-audit examples/ppo-rollout-correction.json`.
Returned scalar/Torch producer evidence includes token ratios, statistics,
keep decisions and coefficients, plus ordered nonfinite stages. Old training
and rollout sampling distributions may differ in temperature or processing:
for old probabilities `(.8,.2)` and temperature-2 behavior `(2/3,1/3)`, the
processed-density weights are `(1.2,.6)`, whereas supplying raw old logprobs
would define different weights `(1,1)`. These input meanings require producer
evidence; the API does not attest a sampler or assume a raw-logprob mismatch
in a caller that explicitly requests processed logprobs.

For the distinct GSPO reference replacement, declare
`correction="rollout_ratio", correction_placement="inside_clip"` with
`policy.ratio_level="sequence_masked_geometric_mean"`. This uses
`r_i=exp(sum(mask*(current-rollout))/max(1,sum(mask)))` **inside** the PPO
minimum/clipping expression. Old logprobs are `UNUSED` by the loss in this
mode; changing an unused old input does not establish a loss anomaly even if
an external source's separate old-policy diagnostic becomes nonfinite.
`detach_rollout_policy` controls this policy reference path independently of
`KLSpec.detach_rollout`. A shared rollout coordinate stays `CONNECTED` if either
term has a live path. Correction off retains the current-minus-old ratio.
The outside-PPO `gated_old_rollout` coefficients are not reused in this mode.

Set `ratio_output_masking="multiply"` to match the source GSPO operation
`ratio=exp(sequence_mean)*mask` **before** the surrogate. This differs from
masking token losses later: an overflowing sequence ratio times an excluded
zero can become NaN before the final mask. Literal events stay observations
of the declared replay, not evidence that a live sampler produced such inputs.

`ReductionSpec(reduction="nonempty_sequence_mean", denominator_epsilon=1e-8)`
adds a separate sequence reduction while preserving the original five:
`n_i=sum(mask_i)`, `v_i=sum(mask_i*loss_i)/(n_i+epsilon)`,
`s_i=float32(n_i>0)`, `L=sum(s_i*v_i)/sum(s_i)`. Empty rows are omitted from
the outer count, unlike `per_sequence_active_mean`, whose outer mean includes
every row. Selection here is literal multiplication after each row mean;
an invalid row mean can still poison the result.

An explicit positive `denominator` replaces only the outer nonempty count;
`process_count*scale` still applies last. This represents the source's global
sequence count scaling without inferring what an integrator actually counted
or executing a collective. `empty_mask_policy="zero"` explicitly clamps inner
counts and an inferred local outer count to at least one. It never clamps a
supplied explicit denominator. With the default `"undefined"`, all-empty local
means remain undefined; an empty microbatch with positive global count can
still contribute zero when the row epsilon is representable.

The float32 sequence selector is an explicit source calculation boundary.
Half/bfloat16 row arithmetic can yield a float32 loss; `loss_dtype` and
`term_dtypes` record that fact. In float16 an epsilon of `1e-8` rounds to zero,
so an empty inner mean can be nonfinite despite the subsequent sequence mask.
The source's supported fp32 Actor logprob path and a direct fp16 stress case
are separate evidence scopes. These contracts do not imply arbitrary mixed
precision graph support.

Run `quotientguard full-objective-audit examples/gspo-rollout-correction.json`.
The example explicitly replaces the ratio reference and excludes an empty row
from the outer count; algorithm labels alone do not select these operations.

The entropy term is `E=-entropy_coefficient*H`; this is an auxiliary **input**
contract, not a claim to reconstruct categorical entropy or Gaussian policy
parameters from one sampled logprob. `kl_mask` and `entropy_mask` can independently
override the completion mask for those terms. The traditional Gaussian
PPO/value/entropy objective remains a separate native-callable family.

`ReductionSpec` preserves the five reductions and empty-mask policies above,
with explicit denominator epsilon, final `process_count*scale`, and denominator
scope (`microbatch`, `batch`, `accumulation_window`, `global_batch`). Nonlocal
counts require `reduction="explicit"`. These fields declare what an integrator
counted; the tool does not infer a distributed count or execute an all-reduce.
For example, a global count divided by process count can be expressed as a
global explicit denominator with final process scaling. `scale` can declare
accumulation compensation, and is distinct from legacy `gradient_scale`.

`audit_objective_window(ObjectiveWindowSpec(reference, counts), batches)` adds
count reconstruction and effective microbatch/full-batch equivalence. Each
`ObjectiveWindowBatch` declares its local contract, trace, rank, microbatch index
and `backward_scale`. The tool checks every term's loss and all coordinate
gradients against a merged full-batch analytic/autograd reference. Unequal token
counts require appropriate weights: averaging microbatch token means equally
does not generally equal a full-window token mean. Sum reductions followed by
one window division can be equivalent even though their local units differ.

Explicit denominators require a `DenominatorCountSpec` binding with units
`active_tokens`, `sequences`, `nonempty_sequences`, `fixed_length_slots` (a
positive completion length is required), or `none` (identity divisor 1).
Counts use the loss mask unless `mask_stage="pre_filter"` is declared; that mode
requires each batch's separate `count_masks[term]` and verifies loss tokens are
a subset. An optional `declared_total` is compared exactly, even for zero loss.
No numeric tolerance can excuse a wrong count. Fixed slots must accommodate
each row's active length. An optional `minimum` declares a count clamp.

For explicit local reductions, scope `microbatch` selects that rank's snapshot,
`batch` selects all ranks with the same microbatch index, and
`accumulation_window`/`global_batch` select the entire supplied window.
`divide_by_process_count=True` divides these microbatch denominators by the
declared process count; the full reference uses the undivided count. Local loss
scales and process factors remain separate: local backward runs first, with its
dtype rounding and nonfinite-stage checks, then simulated process averaging
scales the rounded gradients. Every declared rank must be supplied. The tool
pads variable widths with excluded zero tokens when constructing the reference.

`observed_backward_gradients` uses local post-`backward_scale`, pre-process-average
units. Ordinary trace observations retain the raw returned-loss units.
Count or equivalence failures return `COUNT_MISMATCH_OBSERVED` or
`CONTRACT_MISMATCH_OBSERVED`; missing explicit count units/masks/ranks yield
`INSUFFICIENT_EVIDENCE`. The evidence grade is `REFERENCE_WINDOW_REPLAY`:
supplied masks do not attest a framework count producer, Trainer, collective,
model Jacobian or optimizer execution. Separate source-bound/runtime evidence
is required for those claims. Run
`quotientguard objective-window-audit examples/objective-window.json --json`.

The default composite is `R_policy(w*P*policy_filter) + beta*R_kl(K) + R_entropy(E)`.
Each term can have its own reduction, mask and denominator. With
`composition="before_reduction"`, token terms are added before a shared
reduction; the API requires identical masks and reductions. Operation order in
the tensor replay is preserved, including low-precision intermediate overflow.

The independent float64 engine uses closed-form scalar derivatives and row
Jacobian accumulation, separately from the literal Torch graph. Results include
each term, total loss, term-wise and total gradients for current/old/reference
logprob, advantage, correction weight and entropy. `CONNECTED` includes a real
zero derivative. `DETACHED` is a declared removed path; `UNUSED` has no path for
that term, and neither is fabricated as a zero matrix. Live advantage sign/tie
boundaries and dtype-near clamp boundaries yield `INSUFFICIENT_EVIDENCE`.

Full-trace `observed_loss`, `observed_terms` and `observed_gradients` refer to
the composite. Nested legacy `trace.policy.observed_loss` and
`observed_coefficients` are interpreted as policy-term observations, preserving
their original units. Nonzero observed excluded-token gradients cannot use a
numeric tolerance to pass, even if their magnitude is very small.

Homogeneous CPU/CUDA replay accepts float16, bfloat16, float32 and float64.
Actual native comparisons reject declared/input dtype mismatches. Results retain
ordered numerical stages and the first nonfinite stage, even when a detached
gate contains intermediate overflow and final loss/gradients remain finite.
`NUMERICAL_ANOMALY_OBSERVED` requires a nonfinite output, monitored gradient,
supplied observation, or nonfinite correction arithmetic that changes the
coefficient relative to the independent contract. Contained saturation whose
coefficient and final gradients agree is not treated as a defect.
`CONTRACT_MISMATCH_OBSERVED` is a local finite disagreement. Neither status
confirms natural reachability or an upstream bug. All serialization is strict
JSON with a content-addressed audit hash.

`SUPPORTED` describes the explicit local contract checked by this execution,
not support for every mode of the named framework. CLI exit codes are 0 for
supported, 1 for numerical/mismatch observations, and 2 for unsupported or
insufficient evidence. This new command always executes reference replay;
the original `objective-audit` exit behavior is unchanged.

The benchmark now keeps the original control/holdout partitions separate from
new development partitions with distinct fixed seeds. All are local constructed
controls, not prospective external accuracy measurements.

## Sources

The objective follows [Schulman et al., PPO (2017)](https://arxiv.org/abs/1707.06347).
Backward masking hazards are explained in [PyTorch autograd documentation](https://docs.pytorch.org/docs/stable/notes/autograd.html#division-by-zero-in-autograd).
