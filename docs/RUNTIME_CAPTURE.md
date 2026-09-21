# TRL Runtime Capture

QuotientGuard provides an explicit runtime capture bridge rather than silently monkeypatching a Trainer.

## Why explicit capture

The research result is about the estimator state that is actually consumed by the loss/gradient computation. Function names or nearby variables are not sufficient evidence of semantics.

`TRLRuntimeCapture.record_sequence_sample()` must therefore be called at a declared state-consuming point:

- `loss_contribution`; or
- `gradient_contribution`.

The bridge reconstructs the coefficient using the frozen supported TRL semantics and compares it against the coefficient reported by production code. A mismatch fails closed.

## Frozen source identity

The reconstructed TRL path is accepted only when both match:

- commit: `a98fa6a4428f9aae58dfb26d729d7437f662f27a`;
- `trl/trainer/grpo_trainer.py` SHA-256:
  `e9f5ad165c620c8da1eceade84b00e405c76691d9af363681e8f7cbbfae8a709`.

Verify a source file directly:

~~~bash
quotientguard verify-source /path/to/trl/trainer/grpo_trainer.py
~~~

## Installed exact frozen TRL

If the exact frozen source is installed:

~~~python
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
~~~

`from_installed()` locates the installed `grpo_trainer.py`, hashes its bytes, and rejects it unless the frozen manifest matches.

## Explicit constructor

~~~python
from quotientguard import TRLRuntimeCapture

capture = TRLRuntimeCapture(
    policy_snapshot_id="checkpoint-1200",
    source_provenance={
        "framework": "trl",
        "source_commit": "a98fa6a4428f9aae58dfb26d729d7437f662f27a",
        "source_hash": "e9f5ad165c620c8da1eceade84b00e405c76691d9af363681e8f7cbbfae8a709",
    },
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
~~~

## Other framework revisions

Do not use the frozen reconstruction path for a different TRL source.

For newer/different TRL, verl, OpenRLHF, or private trainers, use the explicit runtime capture surface documented in `INTEGRATIONS.md`. That path records the scalar coefficient actually consumed by the production estimator and does not guess upstream formulas.
