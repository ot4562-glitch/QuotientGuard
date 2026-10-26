# Changelog

## 1.1.0 - 2026-10-26

### Added

- Typed PPO objective specifications, traces and independently hashed diagnostic
  results with expected/observed derivatives, clip branches, masks and provenance.
- Optional PyTorch forward/backward replay and actual loss-graph derivative capture.
- Four verdicts separating expected effects, numerical anomalies, verified semantic
  mismatches and insufficient evidence, with conservative runtime confirmation gates.
- Token and sequence-product ratios, detached token/sequence importance corrections,
  explicit normalization and measured uniform gradient scaling.
- `objective-audit` JSON/Markdown CLI and independent seeded `objective-benchmark`.
- Float16, BFloat16, Float32 and Float64 CPU numerical regression controls.

### Changed

- Added a mathematical interpretation layer above the unchanged v1 compatibility
  screen. Intentional PPO clipping can explain a coefficient spread without
  removing its research compatibility observation.
- Added optional `objective` dependencies; base audit/certify remains dependency-free.

### Fixed

- Diagnose exp overflow/underflow, masked nonfinite propagation, finite-product
  overflow and nonfinite losses/gradients without sanitizing evidence.
- Preserve exact excluded-token zero checks and reject ambiguous clip boundaries.

### Compatibility

- Existing v1 APIs, CLI commands and certificate v2 evidence meanings remain.
- Objective diagnostics use their own v1 schema. Optional attached sidecars are
  included in the certificate hash; existing certificate fields are not repurposed.
- Frozen reconstructed TRL/GRPO source manifests and semantics remain unchanged.

### Known Limitations

- General GRPO/ICEPOP, mixed-precision graphs and arbitrary gradient transformations
  are unsupported. Clip-boundary conventions remain insufficient evidence.
- Semantic verification trusts executable integration contracts and input provenance;
  a source hash alone is not semantic proof. Synthetic/manual results cannot confirm
  a framework defect. No new external framework defect is claimed.
- Historical ten-repository candidate results were not rerun; their seven candidates
  are not benchmark labels. Holdout is a separate seed from the same local generator.

## 1.0.0 - 2026-09-21

First feature-frozen public release.

- Added deterministic exact and sampled estimator-compatibility audits for RL/RLVR.
- Added strict fixed-policy-snapshot, batch, trajectory, estimator, and equivalence-contract provenance.
- Separated exact/analytic evidence from finite-sample observations to prevent false certification.
- Added direct sampled compatibility-gap estimation with leave-one-batch-out cluster jackknife uncertainty.
- Added conservative full-score and deterministic hash-sign sketch semantics.
- Added verifier/decoded-text equivalence contracts.
- Added one-command A1 research demo, automatic `check`, mechanical `explain`, regression `compare`, Sentinel, and benchmark commands.
- Added deterministic JSON certificates, Markdown/text reporting, SHA-256 identities, user-defined policy exits, and dependency-free CI assertions.
- Added frozen reconstructed TRL GRPO/BNPO/Dr.GRPO support gated by both research commit and `grpo_trainer.py` SHA-256.
- Added installed-source verification and runtime coefficient reconstruction validation for the exact frozen TRL source.
- Added stable explicit-runtime adapters for verl, OpenRLHF, and arbitrary custom trainers without guessing upstream loss semantics.
- Added deterministic Trainer Certification Suite with paper-derived, structural, and synthetic controls.
- Added reusable GitHub Action and Python 3.10-3.14 CI workflow.
- Added final documentation, security policy, contribution policy, release metadata, and permanent non-goals.
- Frozen public v1.0 feature scope; future changes are limited to correctness, security, documentation, reproducibility, and portability fixes.

## 0.3.0.dev0 - 2026-09-21

- Added the public `check` front door with automatic receipt, JSONL observation, and supported TRL fixture detection.
- Added a built-in research-grounded A1 `demo`.
- Added decoded-text and verifier-output equivalence contracts, including integer-answer normalization.
- Added deterministic baseline/candidate certificate comparison with user-defined regression policies.
- Added concise terminal reports and dependency-free CI assertion helpers.
- Hardened the TRL adapter to fail closed on missing or unsupported declared source provenance.
- Added a shared coefficient trace used by receipt generation and mechanical `explain` output.
- Added explicit `TRLRuntimeCapture` for state-consuming loss/gradient points with production-vs-reconstruction validation.
- Added Sentinel v3 snapshot metadata, maximum-gap-ratio policy, and estimator-fingerprint stability policy.
- Added deterministic hash-sign score sketches with conservative projected-zero semantics.
- Added a local audit microbenchmark command; it explicitly does not claim end-to-end Trainer overhead.
- Added local Python 3.10-3.14 GitHub Actions CI configuration.
- Added batch-cluster-aware leave-one-batch-out jackknife uncertainty using sufficient-statistic optimization.
- Preserved runtime state-consuming-point/coefficient-trace metadata in deterministic observation provenance.
- Expanded the dependency-free unittest suite to 60 tests.

## 0.2.0.dev0 - 2026-09-21

- Added receipt/certificate schema v2 with mandatory fixed policy snapshot identity.
- Added batch, sample, and concrete trajectory identity for sampled observations.
- Separated exact coefficient evidence from sampled/realized evidence.
- Exact class constancy now uses `CERTIFIED_BY_EXACT_CLASS_CONSTANCY`; sampled equality is never promoted to that certificate.
- Added direct fixed-snapshot sampled gap estimation via `audit_sampled_observations()`.
- Prevented the invalid sampled aggregation shortcut `E[H]E[s]` for `E[Hs]`; aggregate receipts are coefficient-only.
- Migrated the TRL sequence adapter to `trl-sequence-v2` and added frozen TRL provenance.
- Hardened Sentinel around distinct policy snapshots and duplicate-snapshot rejection.
- Added deterministic estimator and equivalence-contract fingerprints.
- Expanded correctness tests to 37 dependency-free unittest cases.

## 0.1.0 - 2026-09-20

- Initial QuotientGuard MVP.
- Added coefficient-only class-constancy screen.
- Added full quotient/fiber compatibility-gap audit.
- Added deterministic JSON certificates.
- Added optional policy thresholds and CI exit status.
- Added Markdown reporting.
- Added dependency-free examples and tests.
- Added sampled-observation aggregation for trainer logs.
- Added conservative full-vs-sketch score semantics.
- Added a narrow direct TRL sequence-level adapter for GRPO, BNPO, and Dr.GRPO.
- Added the paper's equal-length clipping witness as a coefficient-screen fixture.
