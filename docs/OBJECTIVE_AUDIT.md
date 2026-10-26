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

## Sources

The objective follows [Schulman et al., PPO (2017)](https://arxiv.org/abs/1707.06347).
Backward masking hazards are explained in [PyTorch autograd documentation](https://docs.pytorch.org/docs/stable/notes/autograd.html#division-by-zero-in-autograd).
