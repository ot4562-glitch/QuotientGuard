# QuotientGuard

> **Find gradient updates your reward did not ask for.**

Current stable release: **v1.2.0** ·
[GitHub Release](https://github.com/ot4562-glitch/QuotientGuard/releases/tag/v1.2.0) ·
[Changelog](CHANGELOG.md) ·
[Paper](https://doi.org/10.5281/zenodo.22859044)

**v1.2.0** adds explicit full-objective, input/parameter-gradient,
controlled training-step and accumulation-window audits, plus resumable source-only
repository inspection. See [Objective audit](docs/OBJECTIVE_AUDIT.md),
[Integrations](docs/INTEGRATIONS.md) and [v1.2 scope and validation](docs/VALIDATION_V1_2.md).

The v1.2 API provides explicit composite policy/KL/entropy loss
and input-gradient contracts (`FullObjectiveSpec`, `audit_full_objective`),
source-bound callable comparisons, and controlled AMP/accumulation/plain-SGD
step validation. Run `quotientguard full-objective-audit examples/full-objective.json`
with the `objective` extra installed. Framework modes and unsupported paths
remain explicit; a local contract match does not certify an entire Trainer.
`quotientguard objective-window-audit examples/objective-window.json` additionally
checks explicit denominator counts and weighted microbatch/full-batch loss and
gradient equivalence from supplied masks and process snapshots.

QuotientGuard is a deterministic, offline **estimator-compatibility and
objective-aware numerical audit** tool for RL/RLVR. It checks whether
reward/verifier-equivalent trajectories receive different update treatment,
and (from v1.1) whether a supplied PPO loss graph agrees with the
*declared* clipping/masking objective. It is **not** a replacement optimizer,
a downstream-quality predictor, or an autonomous universal bug finder.

## Quick start

```bash
pip install .
quotientguard demo
quotientguard certify
quotientguard check fixtures/controls/exact_compatible.json
```

The demo uses the paper's equal-length A1 case: two verifier-equivalent
answers share reward, length and advantage but differ in effective
coefficient because one PPO branch clips and the other remains active.
A coefficient mismatch is **not automatically a software defect**.

### PPO objective-aware audit (v1.1)

```bash
pip install ".[objective]"
quotientguard objective-audit examples/ppo-objective.json --oracle --out objective.json --markdown objective.md
quotientguard objective-benchmark --out objective-benchmark.json
```

```python
from quotientguard import ObjectiveSpec, ObjectiveTrace, audit_objective

result = audit_objective(
    ObjectiveSpec(reduction="sum"),
    ObjectiveTrace(
        current_log_probs=[[0.4054651081081644, 0.0]],
        old_log_probs=[[0.0, 0.0]],
        advantages=[[1.0, 1.0]],
        mask=[[1, 1]],
        observed_coefficients=[[0.0, -1.0]],
    ),
    oracle=True,
)
assert result.verdict == "EXPECTED_OBJECTIVE_EFFECT"
```

Interpretation verdicts:

| Verdict | Meaning |
| --- | --- |
| `EXPECTED_OBJECTIVE_EFFECT` | Observations match the supported declared PPO math |
| `NUMERICAL_ANOMALY_OBSERVED` | Nonfinite/underflow evidence in a supplied execution or reference replay; provenance matters |
| `SEMANTIC_MISMATCH_CONFIRMED` | Strictly gated repeated, source-bound **actual** integration with executable contract/reconstruction verification |
| `INSUFFICIENT_EVIDENCE` | Unsupported objective, unverified inputs, clip-boundary ambiguity or missing observations |

A replay event is labeled `REFERENCE_REPLAY`, **not** an executed upstream
defect. Manually entered/synthetic coefficients never confirm an
external framework bug. General GRPO/ICEPOP objectives, mixed-precision
graphs and arbitrary parameter-space transformations are not supported
by the v1.1 PPO oracle; see [objective audit specification](docs/OBJECTIVE_AUDIT.md).

## What the product contains

| Capability | Purpose |
| --- | --- |
| Coefficient screen | Show estimator treatment differences inside declared equivalence classes |
| Full/projected compatibility gap | Audit estimator movement within an equivalence class, with conservative sketch semantics |
| `compare` and Sentinel | Compare trainer revisions and track gaps across *separate* fixed-policy-snapshot certificates |
| Objective-aware PPO (v1.1) | Explain expected clip effects versus numerical anomalies using optional PyTorch forward/backward |
| Explicit runtime adapters | Record a trainer's actual coefficients without guessing upstream internals |
| Frozen TRL adapter | Source-hash-gated reconstruction of a specific historical GRPO/BNPO/Dr.GRPO path |
| Deterministic evidence | Machine-readable, SHA-256-addressed certificates and reports, user-defined CI policies |

```bash
quotientguard compare baseline.json candidate.json --fail-on-new-gap
quotientguard explain fixtures/paper/a1_equal_length_clipping.json
quotientguard sentinel certs/step-*.json --max-gap-ratio 0.05 --require-same-estimator
quotientguard verify-source /path/to/trl/trainer/grpo_trainer.py
```

Core CLI/certification requires no runtime dependencies. The optional
`objective` extra installs PyTorch for reference replay. Exact vs
sampled evidence, realized coefficients vs expected class means, and
full vs projected score spaces are **never silently conflated**.

### Code map for contributors and coding agents

| Area | Canonical implementation |
| --- | --- |
| Compatibility math and sampled audit | `quotientguard/core.py`, `observations.py`, `sampled.py` |
| Evidence identities, reports, certificates | `schema.py`, `provenance.py`, `report.py`, `certification.py` |
| PPO semantics and PyTorch oracle | `objective.py`, `objective_torch.py`, `objective_runtime.py` |
| Framework support | `quotientguard/adapters/`, `source_manifests.py` |
| CLI, comparisons and Sentinel | `cli.py`, `compare.py`, `sentinel.py` |
| Tests and release gates | `tests/`, `tools/validate_release.py`, `tools/check_release_docs.py` |

### Source-only inspection evidence (2026-10-29)

A pinned, immutable-commit cohort of **952 repositories** was inspected with the
v1.2 source-only survey. All **92,418 known Python paths** were accounted for;
**92,232** parsed successfully with Python AST. The remaining **186** paths had
explicit syntax, decoding, link or unsafe-path statuses. Repository coverage was
**914 complete / 38 partial** under the declared source-inspection policy.

The survey produced **8,881 static risk candidates** (6,186
source/pattern/line groups), but **zero independently confirmed software
defects**. These observations are not demonstrated training failures or
ready-to-submit upstream fixes. The third-party sources were neither installed,
imported nor executed. Aggregate coverage and evidence limitations are in
[v1.2 validation](docs/VALIDATION_V1_2.md); detailed candidate triage and raw
inspection evidence remain outside the public distribution.

## Verified upstream fixes

**No merged upstream fixes are eligible for this table as of 2026-10-29.**

A submitted PR is not a merged fix, and a QuotientGuard-assisted
source-level audit is not autonomous discovery. This section exists
to record **verified, upstream-merged** fixes as they arrive.

| Repository | Merged upstream PR | Demonstrated fix | QuotientGuard's verified role | Evidence |
| --- | --- | --- | --- | --- |
| *No qualifying merged PRs yet* | — | — | — | — |

To add an entry, first verify the upstream merge commit/date, the
reproduction and relevant test evidence, then label its provenance
accurately as **QG-assisted triage**, **native QG runtime audit**, or
**autonomous discovery** only when independently demonstrated. Record open,
withdrawn and unverified work in an **internal, local-only** engineering ledger
outside this public repository. Such work is not a shipped feature, upstream
adoption or confirmed autonomous QuotientGuard discovery. A merged external
code fix is evidence of accepted contribution, **not** proof that the
QuotientGuard tool autonomously found or certified the bug.

## Versions and release discipline

| Version | Released capabilities | Evidence |
| --- | --- | --- |
| [v1.2.0](https://github.com/ot4562-glitch/QuotientGuard/releases/tag/v1.2.0) | Explicit PPO/GSPO/CISPO/SAPO + KL/entropy contracts, controlled native/parameter/AMP/SGD/window checks and source-only repository inspection | [v1.2 validation scope](docs/VALIDATION_V1_2.md) |
| [v1.1.0](https://github.com/ot4562-glitch/QuotientGuard/releases/tag/v1.1.0) | Optional PPO objective-aware interpretation, backward replay, numerical diagnostics; original compatibility semantics retained | [v1.1 validation](docs/VALIDATION_V1_1.md) |
| [v1.0.0](https://github.com/ot4562-glitch/QuotientGuard/releases/tag/v1.0.0) | Exact/sampled compatibility, source-bound TRL adapter, regression/CI, Sentinel | [frozen benchmark](benchmarks/README.md) |

Future versions must update the **current stable release line**, this
version table, `pyproject.toml`, `CHANGELOG.md`, and release evidence
together. The [release-doc metadata check](tools/check_release_docs.py)
is a CI gate. **Do not change historical tags** to rewrite prior
evidence. Roadmap and reviewer feedback are tracked separately and
are not claims of shipped functionality.

## Documentation (minimal canonical set)

- [Reference](docs/REFERENCE.md): estimator math, evidence levels,
  certificates, sketches, Sentinel, limitations and research scope
- [Objective audit](docs/OBJECTIVE_AUDIT.md): v1.1 PPO contracts,
  oracle, runtime capture and diagnostic limitations
- [Integrations & CI](docs/INTEGRATIONS.md): framework adapters,
  frozen TRL identity and verifiable test hooks
- [v1.2 validation](docs/VALIDATION_V1_2.md): accepted scope, local
  reproducibility gates and limits of the 952-repository source-only survey
- [v1.1 validation](docs/VALIDATION_V1_1.md): historical release gate
  results; not a live third-party bug benchmark

Source of record for released functionality: [CHANGELOG.md](CHANGELOG.md).
Research: **Minseong Sim (2026)**, *Diagnosing Estimator Homomorphism
Compatibility in Reinforcement Learning with Verifiable Rewards*,
DOI [10.5281/zenodo.22859044](https://doi.org/10.5281/zenodo.22859044).
The [included preprint](paper/Sim_2026_Estimator_Homomorphism_Compatibility_RLVR_Preprint.pdf)
is under CC BY 4.0; the software is [MIT licensed](LICENSE).
