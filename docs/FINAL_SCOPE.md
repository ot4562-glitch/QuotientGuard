# QuotientGuard v1.0 Product Freeze

Date: 2026-09-21
Status: FINAL PUBLIC SCOPE / NO FEATURE ROADMAP

## Product identity

- Repository name: `QuotientGuard`
- Python package: `quotientguard`
- Release: `v1.0.0`
- License: MIT
- Paper DOI: `10.5281/zenodo.22859044`
- Repository description: **Find gradient updates your reward did not ask for — deterministic estimator-compatibility audits and regression tests for RL/RLVR.**

## Final architecture

### Layer A — Audit
Deterministic, offline estimator-compatibility audit.

Includes:
- exact coefficient sufficient-condition checks;
- direct fixed-snapshot sampled gap estimation;
- batch-cluster jackknife uncertainty;
- full-score and deterministic-sketch evidence;
- user-defined policy evaluation;
- deterministic JSON certificates and Markdown/text reports.

### Layer B — Sentinel
Certificate-level persistence/regression monitoring.

Includes:
- repeated measured-gap counts;
- consecutive gap streaks;
- max gap ratio;
- first/last policy snapshot;
- estimator fingerprint changes;
- optional timestamp metadata;
- user-defined alert policies.

Sentinel never pools raw observations across policy snapshots and makes no monotonic-drift assumption.

### Layer C — Integrations

1. **Frozen reconstructed TRL adapter**
   - GRPO / BNPO / Dr.GRPO sequence-level supported scope;
   - commit + source-file SHA-256 manifest verification;
   - runtime coefficient reconstruction validation;
   - automatic installed-source discovery when the exact frozen source is installed.

2. **Explicit runtime adapters**
   - verl;
   - OpenRLHF;
   - arbitrary custom trainer callback.
   - These adapters do not infer upstream loss semantics. The caller provides the scalar coefficient actually consumed at the loss/gradient contribution point.

3. **CI / GitHub**
   - CLI exit codes;
   - Python assertion helpers;
   - certification fixture suite;
   - reusable GitHub Action/workflow.

4. **Agent-neutral wrapper**
   - optional Skill-compatible workflow layer;
   - never replaces deterministic certificate computation.

## Final public CLI

```text
quotientguard demo
quotientguard check INPUT
quotientguard compare BASELINE CANDIDATE
quotientguard explain INPUT
quotientguard sentinel CERTS...
quotientguard certify MANIFEST
quotientguard verify-source SOURCE_FILE
quotientguard benchmark
```

Advanced compatibility commands remain:
- `audit`;
- `aggregate`;
- `trl-receipt`.

## Final product functions

1. Equivalence contracts
2. Exact/sampled coefficient screen
3. Full/projected compatibility-gap audit
4. Trainer regression comparison
5. Mechanical mechanism explanation
6. Sentinel persistence monitoring
7. Trainer Certification Suite
8. Source-manifest verification
9. Framework-neutral runtime callback capture
10. Deterministic score sketches

## Permanent non-goals

The following are intentionally **out of scope**, not unfinished roadmap items:

- GUI/dashboard;
- hosted web service;
- database or telemetry backend;
- LLM-based semantic-equivalence inference in the mathematical core;
- automatic gradient correction;
- automatic optimizer replacement;
- claims that every measured gap harms downstream reward/quality;
- claims that a zero low-dimensional sketch certifies the full score space;
- silent support for unknown framework source semantics.

## Maintenance philosophy

v1.0 is designed to remain useful without framework-chasing updates:

- the mathematical core is framework independent;
- reconstructed semantics are allowed only for cryptographically frozen source manifests;
- newer or different frameworks use explicit-runtime capture rather than guessed formulas;
- unsupported reconstructed versions fail closed.

There is no post-v1.0 feature roadmap in the public repository.
