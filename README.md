<div align="center">
  <h1>QuotientGuard</h1>
  <p><strong>Numerical audits for reinforcement learning</strong></p>
  <p>Check a recorded loss, gradient, accumulation window, or optimizer step against its mathematical contract.</p>
  <p>
    <a href="https://github.com/ot4562-glitch/QuotientGuard/releases/latest"><img alt="Latest GitHub release" src="https://img.shields.io/github/v/release/ot4562-glitch/QuotientGuard?color=3D856D&amp;labelColor=14252F"></a>
    <a href="https://github.com/ot4562-glitch/QuotientGuard/actions"><img alt="Main branch CI" src="https://github.com/ot4562-glitch/QuotientGuard/actions/workflows/ci.yml/badge.svg?branch=main"></a>
    <a href="LICENSE"><img alt="MIT license" src="https://img.shields.io/badge/license-MIT-467C96?labelColor=14252F"></a>
  </p>
  <p><a href="#quick-start">Quick start</a> · <a href="#what-the-product-contains">Audit capabilities</a> · <a href="#verified-upstream-fixes">Upstream evidence</a> · <a href="#documentation">Documentation</a></p>
</div>

Current stable release: **v2.0.0** ·
[GitHub Release](https://github.com/ot4562-glitch/QuotientGuard/releases/tag/v2.0.0) ·
[Changelog](CHANGELOG.md) ·
[Paper](https://doi.org/10.5281/zenodo.22859044)

QuotientGuard v2.0 adds a read-only audit for one CPU Float64 AdamW step.
It separately checks whether the optimizer received the authorized state and
whether it updated the state and parameter correctly using the gradient it
actually consumed. Existing objective, gradient and accumulation reports remain
available alongside the optimizer report.

```bash
python -m pip install .
quotientguard adamw-audit examples/adamw-audit.json --json
```

The AdamW audit reads JSON without importing PyTorch. The example records
a CPU step after restoring optimizer state. See the [AdamW trace guide](docs/ADAMW_AUDIT.md)
for evidence requirements, API usage, supported paths and migration from v1.2.

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
| AdamW state handoff and step replay | Check which state the optimizer consumed and independently replay one supported CPU Float64 step |
| Full-objective and window audits | Compare recorded loss and gradients against an explicit clipping, masking and normalization contract |
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

**Confirmed merged fixes: 0 as of 2026-11-22.** The table records public
submissions and the evidence each provides. Most historical results come from
independent CPU reproductions after QuotientGuard-assisted triage. The last
column records GitHub merge status for each submission.

| Repository | Opened issue or PR | Change | Evidence | QuotientGuard feature | Merge status |
| --- | --- | --- | --- | --- | --- |
| AReaL | [Issue #1986](https://github.com/areal-project/AReaL/issues/1986) | Report a NaN gradient from a version-masked PPO token with a finite loss | [Pinned CPU actor replay with a real policy-version mask and independent gradient](https://github.com/areal-project/AReaL/issues/1986) | Full-objective loss and gradient audit flagged a numerical anomaly | Open issue |
| OpenRLHF | [PR #1425](https://github.com/OpenRLHF/OpenRLHF/pull/1425) | Add a regression test for clipped, masked PPO token-mean gradients | [Native loss and backward replay, plus passing pre-commit](https://github.com/OpenRLHF/OpenRLHF/pull/1425/checks) | Objective and accumulation-window audit matched the contract | Open |
| OpenRLHF | [PR #1382](https://github.com/OpenRLHF/OpenRLHF/pull/1382) | Propose rejecting sequences whose weighted PPO loss overflows | [CPU arithmetic reproduction and regression tests](https://github.com/OpenRLHF/OpenRLHF/pull/1382) | Runtime coefficient capture rejected non-finite inputs | Closed, unmerged |
| NeMo RL | [PR #4883](https://github.com/NVIDIA-NeMo/RL/pull/4883) | Propose rejecting seq-mask TIS sequences with non-finite active weights | [Source-equivalent CPU before/after and four-dtype regression](https://github.com/NVIDIA-NeMo/RL/pull/4883) | Coefficient comparison on synthetic traces | Closed, unmerged |
| trlx | [PR #614](https://github.com/CarperAI/trlx/pull/614) | Propose masking excluded PPO log-ratios before exponentiation | [Source-equivalent CPU loss and gradient reproduction](https://github.com/CarperAI/trlx/pull/614) | Coefficient comparison on synthetic traces | Closed, unmerged |
| torchtune | [PR #3006](https://github.com/meta-pytorch/torchtune/pull/3006) | Propose zeroing PPO padding log-ratios before exponentiation | [CPU before/after and padding regression tests](https://github.com/meta-pytorch/torchtune/pull/3006) | Coefficient comparison on synthetic traces | Closed, unmerged |
| Open-Reasoner-Zero | [PR #77](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero/pull/77) | Propose isolating masked PPO tokens and defining the empty-mask result | [Source-equivalent CPU before/after and regression tests](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero/pull/77) | Coefficient comparison on synthetic traces | Closed, unmerged |
| DeepSpeedExamples | [PR #1022](https://github.com/deepspeedai/DeepSpeedExamples/pull/1022) | Propose masking excluded DeepSpeed-Chat PPO log-ratios before exponentiation | [Source-equivalent CPU before/after and regression test](https://github.com/deepspeedai/DeepSpeedExamples/pull/1022) | Coefficient comparison on synthetic traces | Closed, unmerged |
| Axolotl | [PR #4209](https://github.com/axolotl-ai-cloud/axolotl/pull/4209) | Propose masking excluded async GRPO tokens before computing importance ratios | [Source-equivalent CPU replay and token/sequence regression tests](https://github.com/axolotl-ai-cloud/axolotl/pull/4209) | Coefficient comparison on synthetic traces | Closed, unmerged |
| LLM-Dojo | [PR #40](https://github.com/mst272/LLM-Dojo/pull/40) | Propose excluding rejected ICEPOP weights before multiplication | [Isolated policy-loss CPU replay across four dtypes](https://github.com/mst272/LLM-Dojo/pull/40) | Coefficient capture rejected a NaN; finite coefficient spread was expected | Closed, unmerged |
| MusicRLVR | [PR #1](https://github.com/feliperodriguezborquez/MusicRLVR/pull/1) | Propose masking excluded GRPO policy and KL log-probabilities before exponentiation | [Pinned CPU loss replay and before/after regression tests](https://github.com/feliperodriguezborquez/MusicRLVR/pull/1) | Source-only numerical-risk screen flagged the site for review | Closed, unmerged |
| sanketsans/rlvr | [PR #5](https://github.com/sanketsans/rlvr/pull/5) | Propose zero-centered singleton MOPD advantages | [Native loss/backward reproducer and 41 passing patched tests](https://github.com/sanketsans/rlvr/pull/5) | Source-only screen guided triage; manual review found the defect | Closed, unmerged |
| ms-swift | [Issue #10431](https://github.com/modelscope/ms-swift/issues/10431) | Report a possible masked GRPO overflow | [Source-equivalent CPU example; trainer reachability unverified](https://github.com/modelscope/ms-swift/issues/10431) | Coefficient comparison on synthetic traces | Closed issue |

The v2.0.0 evaluation exercised pinned CPU loss code from TRL, verl,
OpenRLHF, NeMo RL and AReaL. All 102 finite-input comparisons matched their
independent reference. OpenRLHF also passed 30 accumulation-window checks.
The separate AReaL boundary case above triggered a direct numerical verdict.
The evaluation covered loss and gradient boundaries. It did not run full
training jobs or measure the boundary case's frequency in real rollouts.

A bounded source-only screen covered 679 Python files from 20 pinned
repositories. Two had complete coverage under that screen and 18 had partial
coverage. Its 390 source candidates require runtime evidence before filing.

## Versions and release discipline

| Version | Released capabilities | Evidence |
| --- | --- | --- |
| [v2.0.0](https://github.com/ot4562-glitch/QuotientGuard/releases/tag/v2.0.0) | Read-only AdamW state lineage and step audits with a boundary report summary | [AdamW trace guide](docs/ADAMW_AUDIT.md) |
| [v1.2.0](https://github.com/ot4562-glitch/QuotientGuard/releases/tag/v1.2.0) | Explicit PPO/GSPO/CISPO/SAPO + KL/entropy contracts, controlled native/parameter/AMP/SGD/window checks and source-only repository inspection | [v1.2 validation scope](docs/VALIDATION_V1_2.md) |
| [v1.1.0](https://github.com/ot4562-glitch/QuotientGuard/releases/tag/v1.1.0) | Optional PPO objective-aware interpretation, backward replay, numerical diagnostics; original compatibility semantics retained | [v1.1 validation](docs/VALIDATION_V1_1.md) |
| [v1.0.0](https://github.com/ot4562-glitch/QuotientGuard/releases/tag/v1.0.0) | Exact/sampled compatibility, source-bound TRL adapter, regression/CI, Sentinel | [frozen benchmark](benchmarks/README.md) |

Future versions must update the **current stable release line**, this
version table, `pyproject.toml`, `CHANGELOG.md`, and release evidence
together. The [release-doc metadata check](tools/check_release_docs.py)
is a CI gate. **Do not change historical tags** to rewrite prior
evidence. Roadmap and reviewer feedback are tracked separately and
are not claims of shipped functionality.

## Documentation

- [Reference](docs/REFERENCE.md): estimator math, evidence levels,
  certificates, sketches, Sentinel, limitations and research scope
- [Objective audit](docs/OBJECTIVE_AUDIT.md): v1.1 PPO contracts,
  oracle, runtime capture and diagnostic limitations
- [AdamW audit](docs/ADAMW_AUDIT.md) explains the optimizer trace,
  verdicts, measurement costs and migration.
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
