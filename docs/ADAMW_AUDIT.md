# AdamW state and step audit

The audit compares the authorized state to the state consumed by AdamW. It
then computes the expected parameter and optimizer state from the actual
consumed state and gradient using the [AdamW recurrence](https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html).
The API reads a supplied trace and never invokes an optimizer.

## Run the example

```bash
python -m pip install .
quotientguard adamw-audit examples/adamw-audit.json --json --out adamw-report.json
```

The [example trace](../examples/adamw-audit.json) records a CPU Float64 step
after a real PyTorch `state_dict` restore. Both boundaries return `MATCH`.
The command returns exit 0 for a match, 1 for a measured mismatch, and 2
for unsupported, inconclusive or invalid evidence. Output files contain
the supplied tensor values and should be handled with the same care as traces.

```python
import json

from quotientguard import audit_adamw_step, summarize_training_audits

with open("examples/adamw-audit.json", encoding="utf-8") as handle:
    optimizer = audit_adamw_step(json.load(handle))
print(optimizer["lineage"]["status"], optimizer["transition"]["status"])
summary = summarize_training_audits(optimizer=optimizer)
for boundary in summary["boundaries"]:
    print(boundary["boundary"], boundary["status"])
```

## Capture evidence

The top-level `binding` identifies the audited tensor and step through
`run_id`, `step_id`, `parameter_id` and `shape`. Every record repeats this
binding. Tensor values are flat, finite numbers in row-major order. Use a
stable tensor name or ID to confirm the optimizer-to-model mapping. Identical
shapes and positional checkpoint IDs alone are insufficient. Confirm that
the authorized state comes from the intended checkpoint or initialization,
including the effective optimizer options. Preserve immutable values so a
later step cannot mutate an earlier snapshot. [PyTorch state mapping](https://docs.pytorch.org/docs/2.14/generated/torch.optim.Optimizer.state_dict.html)

| Record | Required stage | Contents |
| --- | --- | --- |
| `authorized` | `authorized` | Approved parameters, state and effective hyperparameters |
| `consumed` | `before_step` | Actual parameters, state and effective hyperparameters immediately before the audited step |
| `gradient` | `consumed_gradient` | The gradient actually supplied to AdamW after upstream transformations. Use `kind: dense` and `values`, or `kind: none` for an observed absent gradient. |
| `after` | `after_step` | Actual parameters and optimizer state immediately after the step |

Each record provides `source` and `verified: true` when the trace provider
has independently checked the record's provenance and tensor/step binding.
The flag is a caller assertion. The tool checks consistency within the trace,
with no external attestation. Unknown or unchecked sources can be marked
`verified: false` or omitted. In particular, a gradient recomputed from the
loss is not evidence that AdamW consumed it.

State has either `{"kind": "uninitialized"}` or `kind: initialized` with
the integer `step`, `exp_avg` and `exp_avg_sq`. Hyperparameters are
`lr`, `beta1`, `beta2`, `eps` and `weight_decay`. The `execution` object
identifies the supported path. Use the example to see the required fields.
Capture optimizer settings after any runtime change and gradients after
aggregation, unscaling or clipping if applicable. The audited optimizer mode
itself must still satisfy the scope below.

## Interpret the report

- **Lineage** compares the approved and consumed parameters, moments, step
  counters and hyperparameters. This comparison is exact for recorded values.
- **Transition** checks the output parameters, moments and step counter from
  the consumed state, effective options and actual gradient. The report
  includes the reference bias correction and discrepant coordinates. Float
  comparisons use fixed relative tolerance `1e-12` and absolute tolerance
  `1e-15`. Step counters and state initialization are compared exactly.
- A lineage mismatch with a transition match means that AdamW computed
  correctly using a state or setting different from the authorized input.
  Missing authorized evidence leaves lineage inconclusive even if the
  transition matches.
- An explicit absent gradient leaves the parameter and state unchanged for
  this tensor. A dense zero gradient can initialize or advance its state and
  still permits decoupled weight decay. An absent record gives an inconclusive
  verdict.
- Reports distinguish `MATCH`, `MISMATCH`, `INCONCLUSIVE`,
  `INVALID_EVIDENCE` and `UNSUPPORTED`. A report with a confirmed mismatch
  and a separate undecidable boundary retains both findings.

For example, the shipped trace matches both contracts. Replaying a step
after clearing its first moment produces a lineage mismatch if the authorized
moment is still recorded. The transition still matches when the captured
output follows the cleared state.

The supported scope is one nonempty dense CPU Float64 tensor in a single
executed `torch.optim.AdamW` step. The optimizer uses the non-fused,
non-foreach minimizing path with scalar hyperparameters, no AMSGrad,
and no capturable or differentiable mode. Float32 and Float64 step counters
are supported where the active increment is exact. GPU, mixed precision,
sharded states, sparse gradients, skipped steps and other optimizer
implementations are outside this contract. Avoid undeclared hooks or
mutations within the captured step. Numerical results are tolerance-based
comparisons in this scope.

## Migrate from v1.2

Existing objective, parameter-gradient, window and training validation APIs
keep their arguments and behavior. Continue using the v1.2 training validator
for controlled plain SGD. Supply a separate AdamW trace for the new audit.
Pass existing report dictionaries (using `.to_dict()` where available) as
`objective`, `gradient` and `window` to `summarize_training_audits` along
with the AdamW `optimizer` report. The summary displays the boundaries in
training order and labels omitted reports. It preserves independent verdicts.
The previous report schemas do not establish shared run/step provenance with
an optimizer report. The summary explicitly leaves pipeline-level verification
unestablished.

The oracle processes the tensor in linear time and reports the compared
coordinates. A trace with authorized, consumed and post-step tensors plus
gradient takes up to ten scalar values per coordinate for these arrays.
JSON values and Python objects add overhead. The full report also holds
the independent expected and observed values. No GPU latency or storage
claim follows from CPU runs.

On one Windows Python 3.12.10 host, repeating the two-coordinate example
values to match a larger tensor took the following time, including the
report hash. Times are medians of five audits of in-memory traces.
Sizes are compact JSON and peak is additional Python allocation measured
separately with `tracemalloc`. This excludes tensor capture and file I/O.

| Coordinates | Audit | Input | Report | Additional peak |
| ---: | ---: | ---: | ---: | ---: |
| 10,000 | 0.14 s | 1.92 MB | 2.54 MB | 8.01 MB |
| 100,000 | 1.53 s | 19.20 MB | 25.35 MB | 78.73 MB |

A separate CPU copy-to-list measurement for ten Float64 arrays took a
median of 1.8 ms at 10,000 coordinates and 25 ms at 100,000 coordinates,
over 12 repeats. Those arrays contain 0.8 MB and 8 MB of raw tensor data,
respectively. Measurements exclude checkpoint loading and distributed
synchronization.
