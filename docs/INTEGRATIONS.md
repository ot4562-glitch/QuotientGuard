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
