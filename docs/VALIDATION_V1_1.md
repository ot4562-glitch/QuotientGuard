# QuotientGuard v1.1 validation

Date: 2026-10-26. Local environment: Windows x64, Python 3.12.10,
PyTorch 2.14.1+cpu. This document describes local gates; GitHub CI is a separate
release gate and must pass before tagging/publishing.

## Reproduce

```bash
python -m pip install ".[objective]" build
python -m unittest discover -s tests -q
python -m quotientguard certify
python -m quotientguard objective-benchmark --out objective-benchmark.json
python tools/validate_release.py
```

The validation script requires torch and rejects skipped tests in the full suite.
It writes detailed logs in ignored `.venv/release-validation/` and public summaries
in `benchmarks/release-validation-v1.1.0.json`, `benchmarks/certification-v1.1.0.json`
and `benchmarks/objective-v1.1.0.json`. It never publishes. The script builds both
packages and installs each into a separate environment. It runs installed demo,
certify, version/identity and objective JSON smokes outside the checkout; it also
installs the wheel in the torch environment and checks its oracle outside source.

## Local results

- Full suite: 103 tests PASS, zero failures, zero skips. Original 69 tests retained.
- Public certification: 5/5 PASS.
- Float16, BFloat16, Float32, Float64: executed CPU forward/backward controls PASS.
- Benchmark controls: 68/68 PASS (40 expected, 20 numerical, 8 insufficient).
- Seeded holdout: 164/164 PASS (136 expected, 20 numerical, 8 insufficient).
- Normal cases: 0/176 false positives; numerical cases: 0/40 false negatives.
- 142 normal coefficient-spread cases were mathematically explained. This is not
  a measured error-rate reduction against a previous defect classifier: v1 recorded
  coefficient mismatch and did not label framework defects.
- Base-package dependency-free compatibility, certificate hashes and optional
  sidecar mutation isolation have regression coverage.
- Wheel/sdist build and separate clean installs PASS; CLI smokes and installed
  wheel torch oracle PASS. Checksums cover the validated binaries.

## Coverage and limits

Normal controls include both advantage signs and clipping directions, active
ratios different from one, masked zero derivatives, sum/explicit/active means,
token/sequence correction weights, sequence-product derivative coupling and
known global norm clipping scale. Numerical controls execute exp overflow,
underflow, masked `inf*0`, padding `-inf-(-inf)`, finite product overflow and
empty-mask means in all four dtypes. Separate regressions cover finite-forward
nonfinite-backward, nonfinite parameter gradients, dtype input-cast overflow,
low-precision-only underflow, reduction overflow and ambiguous clip boundaries.
The frozen GRPO boundary adapter remains fail-closed. Unsupported general GRPO
objectives return insufficient; this is not a new GRPO objective implementation.

Benchmark labels come from construction of independent negative-surrogate torch
controls, not the audit engine's output or historical candidates. The oracle uses
a positive surrogate/minimum implementation and the analytic engine has a separate
Python derivative. Controls and holdout use fixed distinct seeds. The holdout is
not a prospective external dataset and draws from the same specified generator.
Zero local errors do not imply zero population errors.

The owner's historical summary identified ten repositories and seven numerical
candidates with zero integrated confirmed defects. No historical audit artifact
was available in this public checkout. Protected private planning was not accessed.
Open-Instruct, Prime RL, Verifiers, Axolotl, ms-swift, trlx, DeepSpeedExamples,
Open-Reasoner-Zero, torchtune and CleanRL are all `NOT_EXECUTED` in this release's
historical review. None is relabeled as a confirmed bug or counted in detector
accuracy. External source integration is the next validation priority.

## Review fixes

A separate read-only review identified ambiguous clamp boundaries, a sequence
unit-weight shape error, missing target-dtype range checks and mutable returned
certificate sidecars. Each was reproduced and added as a regression before its
fix. Confirmation from synthetic/manual observations remains prohibited. Failed
intermediate test runs were not used as release evidence; the final gate reruns
the complete suite and package validation.
