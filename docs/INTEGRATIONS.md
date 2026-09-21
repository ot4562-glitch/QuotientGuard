# Framework integrations

## Integration principle

There are two integration modes.

### 1. Frozen reconstructed adapter

Use this only where QuotientGuard has frozen and verified source semantics.

Currently:
- Hugging Face TRL sequence-level GRPO/BNPO/Dr.GRPO scope from the research revision.

The source commit and source-file SHA-256 must both match.

### 2. Explicit runtime capture

Use this for:
- verl;
- OpenRLHF;
- newer/different TRL revisions;
- private trainers;
- any custom post-training loop.

The caller records the scalar coefficient actually consumed at the loss/gradient contribution point. QuotientGuard does **not** reconstruct or guess the upstream loss formula.

## verl

```python
from quotientguard import VerlRuntimeCapture

capture = VerlRuntimeCapture(
    policy_snapshot_id="step-1200",
    framework_version="your-verl-version",
)

callback = capture.callback()

# Call at the actual estimator contribution point:
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

## OpenRLHF

```python
from quotientguard import OpenRLHFRuntimeCapture

capture = OpenRLHFRuntimeCapture(
    policy_snapshot_id="step-1200",
    framework_version="your-openrlhf-version",
)
```

The record/callback interface is identical to the verl example.

## Custom trainer

```python
from quotientguard import CallbackRuntimeCapture

capture = CallbackRuntimeCapture(
    framework="my-trainer",
    policy_snapshot_id="checkpoint-42",
    source_commit="abc123",
)
```

## Frozen TRL source verification

```bash
quotientguard verify-source /path/to/trl/trainer/grpo_trainer.py
```

For an installed exact frozen TRL source:

```python
from quotientguard import TRLRuntimeCapture

capture = TRLRuntimeCapture.from_installed(
    policy_snapshot_id="checkpoint-42",
    config=trl_config,
)
```

If the installed source hash does not match the frozen manifest, QuotientGuard fails closed.

## Why no automatic monkeypatching

The scientific object is the estimator value actually consumed by the loss/gradient path.

Function names and nearby variables are not enough to establish semantics. Explicit capture keeps the integration point auditable.
