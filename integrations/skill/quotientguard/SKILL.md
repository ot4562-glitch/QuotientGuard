---
name: quotientguard
description: Use QuotientGuard to audit or regression-test RL/RLVR estimator compatibility when a reward, verifier, or environment defines equivalent trajectories. Trigger for QuotientGuard receipts, JSONL observations, certificates, trainer changes, TRL/verl/OpenRLHF/custom runtime capture, compatibility-gap investigation, Sentinel history, certification suites, or source-manifest verification. Always rely on QuotientGuard's deterministic CLI/library outputs rather than replacing the certificate math with model reasoning.
---

# QuotientGuard

Use the installed `quotientguard` package as the source of truth for estimator-compatibility calculations.

## Workflow

1. Identify the user's evidence surface.
   - Exact receipt or sampled JSONL: run `quotientguard check INPUT`.
   - Frozen supported TRL fixture: run `quotientguard check INPUT`, then `quotientguard explain INPUT` when mechanism details are useful.
   - Baseline and candidate certificates: run `quotientguard compare BASELINE CANDIDATE` with only the policy flags the user requested.
   - Snapshot certificate history: run `quotientguard sentinel ...`.
   - Repository self-test: run `quotientguard certify`.
   - Frozen TRL source file: run `quotientguard verify-source SOURCE_FILE`.

2. Preserve the evidence level exactly.
   - Do not promote sampled equality to an exact theorem certificate.
   - Do not turn coefficient mismatch alone into a measured compatibility gap.
   - Do not turn projected zero into a full-space certificate.
   - Do not convert `OBSERVED` into `PASS` or `FAIL` unless an explicit user policy exists.

3. Preserve the scientific object.
   - Keep one audit tied to one fixed `policy_snapshot_id`.
   - Never pool raw observations across optimizer steps.
   - Keep concrete `trajectory_id` distinct from quotient/equivalence `class_id`.
   - Treat `batch_id` as the uncertainty cluster when batch coupling is present.

4. Treat equivalence as user/environment supplied.
   - Never infer broad semantic equivalence with an LLM for the deterministic audit.
   - Use verifier/environment outputs, explicit contracts, or precomputed class IDs.

5. Integrate frameworks conservatively.
   - Use reconstructed TRL semantics only when its frozen commit and source-file SHA-256 manifest verify.
   - For verl, OpenRLHF, newer TRL, or custom trainers, use explicit runtime capture at the actual `loss_contribution` or `gradient_contribution` point.
   - Do not guess upstream loss semantics from function names.

## Interpretation Rules

Report QuotientGuard results as estimator-compatibility evidence, not model-quality verdicts.

Never claim from a certificate alone that:
- downstream reward or quality was harmed;
- removing a gap will improve training;
- a measured gap is universally unsafe;
- a zero sketch proves the full score-space gap is zero.

When a result is surprising, report the certificate status, evidence level, policy status, source/contract fingerprint, and relevant class metrics before offering interpretation.

## Common Commands

```bash
quotientguard demo
quotientguard check INPUT
quotientguard compare BASELINE.json CANDIDATE.json
quotientguard explain INPUT
quotientguard sentinel certs/*.json --out sentinel.json
quotientguard certify
quotientguard verify-source SOURCE_FILE
quotientguard benchmark --samples 256 --dimension 64 --repeats 7
```

For detailed status semantics and framework integration choices, consult `references/semantics.md`.
