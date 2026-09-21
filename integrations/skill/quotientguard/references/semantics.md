# QuotientGuard reference

## Result hierarchy

Exact evidence:
- `CERTIFIED_BY_EXACT_CLASS_CONSTANCY`: exact class-constant coefficient sufficient condition.
- `MEASURED_COMPATIBLE`: exact full-score gap within numerical tolerance.
- `MEASURED_GAP`: exact non-zero full-score gap.
- `MEASURED_PROJECTED_GAP`: exact non-zero projected component.
- `PROJECTED_ZERO_NOT_CERTIFIED`: selected projection is zero; full space remains uncertified.
- `NOT_CERTIFIED`: exact coefficient-only evidence is insufficient.

Sampled evidence:
- `COEFFICIENT_MISMATCH_OBSERVED`: realized/estimated coefficients differ inside one equivalence class.
- `ESTIMATED_CLASS_CONSTANCY`: sampled means agree within tolerance; not an exact certificate.
- `INSUFFICIENT_REPETITION`: coefficient-only data lack enough repetition.
- `MEASURED_SAMPLE_GAP`: sampled non-zero full-score gap at one fixed snapshot.
- `MEASURED_PROJECTED_SAMPLE_GAP`: sampled non-zero projected gap.
- sampled zero variants remain uncertified at population/full-space level.

## Framework strategy

Frozen reconstructed TRL path:
- commit `a98fa6a4428f9aae58dfb26d729d7437f662f27a`
- `grpo_trainer.py` SHA-256 `e9f5ad165c620c8da1eceade84b00e405c76691d9af363681e8f7cbbfae8a709`
- unknown/missing source identity must fail closed.

Explicit runtime path:
- `VerlRuntimeCapture`
- `OpenRLHFRuntimeCapture`
- `CallbackRuntimeCapture`

Record the scalar coefficient actually consumed at a loss/gradient contribution point. Provenance metadata is required.

## Policy semantics

A QuotientGuard `FAIL` means an explicit user-supplied estimator-compatibility policy failed. It does not by itself imply downstream task harm.
