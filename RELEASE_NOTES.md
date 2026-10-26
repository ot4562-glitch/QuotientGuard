# QuotientGuard v1.1 — Objective-Aware Numerical Auditing

QuotientGuard v1.1 adds a conservative mathematical interpretation layer to the
existing estimator-compatibility audit. A coefficient difference can be expected
under PPO; this release helps explain that difference while preserving the v1
research evidence.

## PPO Objective-Aware Audit

- Typed objective specs, traces and independently hashed audit results.
- Sign-dependent PPO clipping with binary masks, detached token/sequence importance
  corrections, sum/active-token/explicit normalization and sequence-product ratios.
- Explicitly measured uniform gradient scaling; ambiguous clip boundaries fail closed.

## Expected Clipping vs Numerical Anomaly Classification

Four separate diagnostic verdicts distinguish expected objective effects, observed
numerical anomalies, confirmed semantic mismatch and insufficient evidence. A ratio
other than one or a nonzero coefficient spread is not automatically a defect.
The original compatibility status remains a separate analytical layer.

## Gradient and Coefficient Verification

An optional independent PyTorch oracle executes the declared loss and backward in
Float16, BFloat16, Float32 and Float64. Runtime capture differentiates the provided
production loss graph without consuming the trainer's subsequent backward.
Semantic confirmation requires finite repeated source-bound integration evidence,
independent reconstruction and an executable verified contract. Manual, synthetic
and source-equivalent observations cannot confirm an external framework defect.

## Non-Finite Value Detection

Evidence records exponential overflow/underflow, finite-product/reduction overflow,
masked invalid arithmetic, nonfinite losses and gradients, and excluded-token zero
gradients. Reference-replay events are explicitly separated from production
observations. No NaN/Inf sanitization is used to conceal evidence.

## Reproducible Audit Evidence

Use `quotientguard objective-audit` for strict JSON and Markdown reports and
`quotientguard objective-benchmark` for independent local controls and seeded
holdouts. The local release gate passes 103 tests without skips, the existing
5-case public certification suite, and 232 benchmark cases. In that local benchmark,
176 normal cases had zero defect false positives and 40 numerical cases had zero
false negatives; 16 cases correctly remained insufficient. These figures describe
specified local controls, not external framework or population error rates.
Wheel and sdist builds, clean installation and installed CLI/oracle smokes are
included in the release gate. SHA256SUMS identifies the attached validated binaries.

## Backward Compatibility with v1

Existing APIs, CLI commands, certificate v2 compatibility statuses, frozen TRL
manifests and certification fixtures retain their meanings. Objective results use
a separate versioned schema. Optional runtime certificate sidecars are included
in the certificate hash. The base package remains dependency-free; install the
`objective` extra for torch execution.

## Known Limitations

- General GRPO/ICEPOP, mixed-precision/autocast graphs and arbitrary parameter-space
  gradient transformations are unsupported.
- Clip-boundary subgradient conventions remain insufficient evidence.
- Contract verifiers and nonsynthetic provenance are trusted local integration
  assertions; source hashes alone do not prove semantic truth.
- Replay anomalies do not prove an unexecuted trainer has the same error.
- The historical ten-repository audit was not rerun. Its seven numerical candidates
  were not used as ground-truth labels; integrated confirmed defects remain zero.
- Holdout uses a distinct fixed seed from the same generator, not an independent
  external distribution. No universal RL error detection or mathematical completeness
  is claimed.
