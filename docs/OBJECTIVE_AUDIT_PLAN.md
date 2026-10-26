# Objective-aware audit design and implementation ledger

The v1.1 work implements the owner's 2026-10-26 specification. Existing v2
compatibility certificates and their research meaning remain unchanged. The
separate `quotientguard-objective-audit-v1` diagnostic compares derivatives of
the **negative** PPO clipped surrogate loss with respect to current log
probabilities. A new typed API, lazy optional PyTorch oracle, runtime attachment,
JSON CLI, and Markdown report expose this layer.

## Decisions

- Add an interpretation layer rather than redefining mismatch statuses or
  rebuilding framework adapters. Reuse the current checkout on a feature branch;
  preserve the pre-existing untracked local AGENTS.md.
- Support rectangular sequence/token traces, token ratios and sequence-product
  ratios, detached token/sequence importance weights, binary masks, sum,
  active-token mean, and explicit constant denominator reductions. Make old
  policy/advantage/importance detach assumptions explicit. Unsupported graphs
  return insufficient evidence.
- Compute analytic derivatives in Python float64 independently of a literal
  PyTorch clipped loss and backward executed in the declared calculation dtype.
  Record non-finite intermediates without sanitizing them. Tensor hooks capture
  production loss and gradients separately from this reference execution.
- Never promote scalar/manual/synthetic/source-equivalent observations into a
  confirmed framework defect. Confirmation requires repeated actual runtime
  backward captures, a verified executable equivalence contract, objective
  reconstruction validation, and stable policy/model/code identities. Evidence
  is local verification, not a cryptographic attestation of external framework code.
- Diagnose standard post-backward global norm clipping only with an explicit
  observed scale; arbitrary parameter-gradient transformations are unsupported.
- Prior ten-repository results are historical candidate counts supplied by the
  owner, not labels. No prior audit artifacts exist in the public checkout.
  Do not inspect protected private planning. Build independent deterministic
  numerical controls and seeded holdouts; mark external repositories NOT_EXECUTED.

## Tasks and verification gates

1. Write failing analytic, autograd, numerical and evidence-gating tests; implement
   objective dataclasses, analytical branches, strict JSON evidence and oracle.
2. Add actual-backward runtime capture, certificate sidecar and CLI/report tests;
   implement only optional additions, preserving default v1 outputs.
3. Add independent benchmark/holdout and existing GRPO boundary regression tests;
   measure false positives/negatives without treating historical candidates as truth.
4. Update package metadata, docs, changelog, CLI examples and mandatory torch CI.
5. Run full legacy/new tests, certify, all four CPU dtypes, wheel/sdist build and
   isolated install/README smoke checks. Review diff and release contents.
6. Review the complete change with a fresh reviewer, fix material findings, rerun
   affected gates. Commit only release files. Fast-forward main only if remote
   remains compatible; wait for required CI, then tag and publish stable release
   with validated packages/checksums and verify remote identities/assets.

## Execution ledger

- Baseline: 1.0.0; origin/main d065b11d111345b58984abd33626b7a3f0755980;
  remote default main, unprotected; existing release/tag v1.0.0.
- Initial torch probes: absent in Python 3.12 and 3.14. A project-local ignored
  Python 3.12 virtual environment will host CPU torch and build tooling.
- Implementation complete: optional typed PPO engine, actual backward/stop-gradient
  checks, sidecar with full hash coverage, JSON/Markdown CLI, independent benchmark.
- Final source tests: 103 passed with torch, no skips. Original 69 retained.
  Dependency-free Python 3.14 run remains green (optional torch cases skipped).
- Independent review findings reproduced and fixed: clipping boundaries, sequence
  weight shape, dtype cast/underflow/reduction range and sidecar aliasing.
  Reviewer verified both final blockers resolved; focused 13 tests passed.
- Benchmark: controls 68/68, holdout 164/164; 0/176 normal false positives,
  0/40 numerical false negatives, 16 correctly insufficient. Historical repositories
  remain NOT_EXECUTED; no framework defect claimed.
- Final release gate rebuilds and validates both distributions and installed CLIs.
  GitHub publication remains gated on remote CI success and unchanged ancestry.
