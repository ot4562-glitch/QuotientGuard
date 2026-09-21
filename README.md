# QuotientGuard

> **Find gradient updates your reward did not ask for.**

QuotientGuard is a deterministic estimator-compatibility audit and regression-test tool for RL/RLVR training.

A reward, verifier, or environment can say two concrete trajectories are equivalent while the production estimator gives them different update treatment. QuotientGuard detects and certifies that mismatch without requiring an LLM, API key, server, or dashboard.

Research basis: **Minseong Sim (2026), _Diagnosing Estimator Homomorphism Compatibility in Reinforcement Learning with Verifiable Rewards_** — DOI **10.5281/zenodo.22859044**. The published PDF is also included in this repository at [`paper/Sim_2026_Estimator_Homomorphism_Compatibility_RLVR_Preprint.pdf`](paper/Sim_2026_Estimator_Homomorphism_Compatibility_RLVR_Preprint.pdf).

## 10-second demo

```bash
pip install .
quotientguard demo
```

Expected shape:

```text
QuotientGuard
snapshot: paper-a1-step2
policy: OBSERVED
classes: 1 | measured gaps: 0 | certified: 0 | not certified: 1
- accepted-answer-4: COEFFICIENT_MISMATCH_OBSERVED | coeff_spread=0.102263
Interpretation: compatibility diagnostic only; no downstream reward/quality claim.
```

The demo reproduces the paper's equal-length A1 clipping screen: two one-token verifier-equivalent answers have the same length, reward, and advantage, yet one path is clipped while the other remains active.

## What it does

QuotientGuard gives you four evidence layers:

1. **Coefficient screen** — detect representative-dependent estimator treatment.
2. **Full/projected gap audit** — measure update movement inside a reward/verifier-equivalent class.
3. **Regression comparison** — detect new gaps or evidence downgrades between trainer versions/configurations.
4. **Sentinel** — track persistence across separate fixed-policy-snapshot certificates.

Every result is machine-readable and hash-addressed.

## Quick start

### Check an input

```bash
quotientguard check fixtures/controls/exact_compatible.json
quotientguard check fixtures/controls/sampled_projected_gap.jsonl
quotientguard check fixtures/paper/a1_equal_length_clipping.json
```

`check` auto-detects:
- exact QuotientGuard receipts;
- sampled observation JSONL;
- the frozen supported TRL sequence fixture format.

### Explain a TRL mechanism

```bash
quotientguard explain fixtures/paper/a1_equal_length_clipping.json
```

This reports mechanical facts such as clipping branch, sequence ratio, completion length, normalization denominator, and effective coefficient. It does not invent causal percentages.

### Compare a trainer change

```bash
quotientguard compare baseline.json candidate.json --fail-on-new-gap
```

Without a user policy, differences are observations. With an explicit policy, CI can fail.

### Run the public certification suite

```bash
quotientguard certify
```

The v1 suite contains:
- exact compatible positive control;
- exact measured-gap negative control;
- paper A1 clipping fixture;
- verifier-equivalence structural fixture using `"4"` and `"+4"`;
- sampled projected-gap control.

Synthetic controls are labeled as synthetic and are never presented as paper measurements.

## Framework integrations

### Frozen reconstructed TRL path

The reconstructed TRL adapter is deliberately narrow and fail-closed.

Frozen source:
- commit: `a98fa6a4428f9aae58dfb26d729d7437f662f27a`;
- `trl/trainer/grpo_trainer.py` SHA-256:
  `e9f5ad165c620c8da1eceade84b00e405c76691d9af363681e8f7cbbfae8a709`.

Verify a source file:

```bash
quotientguard verify-source /path/to/trl/trainer/grpo_trainer.py
```

A missing or different hash is rejected.

### verl / OpenRLHF / custom trainers

For framework versions whose loss semantics are not frozen by QuotientGuard, use explicit runtime capture: pass the scalar coefficient actually consumed at the loss/gradient contribution point.

```python
from quotientguard import VerlRuntimeCapture

capture = VerlRuntimeCapture(
    policy_snapshot_id="step-1200",
    framework_version="your-version",
)

capture.record(
    batch_id="batch-17",
    sample_index=0,
    class_id="answer:4",
    trajectory_id="trajectory-sha256",
    realized_coefficient=H_i,
    state_consuming_point="loss_contribution",
    score=score_sketch,
)

certificate = capture.certificate()
```

The same stable explicit-capture surface is available as:
- `OpenRLHFRuntimeCapture`;
- `CallbackRuntimeCapture` for arbitrary trainers.

This design avoids guessing changing upstream internals.

## Equivalence contracts

The mathematical core does not use an LLM to decide semantic equivalence.

```python
from quotientguard import IntegerAnswerContract

contract = IntegerAnswerContract()

assert contract.classify("4") == contract.classify("+4")
```

You can also use:
- `VerifierContract`;
- `DecodedTextContract`;
- precomputed `class_id` values from your environment.

## Evidence semantics

QuotientGuard intentionally separates evidence strength.

Important statuses include:

- `CERTIFIED_BY_EXACT_CLASS_CONSTANCY` — exact class-constant coefficient sufficient condition.
- `COEFFICIENT_MISMATCH_OBSERVED` — sampled/realized coefficient mismatch inside one class.
- `MEASURED_GAP` — non-zero exact full-score compatibility gap.
- `MEASURED_PROJECTED_GAP` — non-zero exact projected component.
- `MEASURED_SAMPLE_GAP` / `MEASURED_PROJECTED_SAMPLE_GAP` — non-zero fixed-snapshot sample gap.
- `PROJECTED_ZERO_NOT_CERTIFIED` and sampled zero variants — a zero projection/sample is not promoted into a full-space/population certificate.

See [docs/CERTIFICATES.md](docs/CERTIFICATES.md).

## Batch-aware uncertainty

Sampled direct-gap audits use leave-one-batch-out cluster jackknife standard errors. Rows are not treated as iid when `batch_id` coupling is present.

## Sentinel

Sentinel consumes a sequence of already-issued snapshot-level certificates.

```bash
quotientguard sentinel certs/step-*.json \
  --out sentinel.json \
  --max-gap-ratio 0.05 \
  --max-consecutive-gap-events 1 \
  --require-same-estimator
```

It never pools raw observations across optimizer steps and makes no assumption that compatibility gaps grow monotonically.

## CI

QuotientGuard has no runtime dependencies for the core path.

Use the CLI directly:

```bash
quotientguard check fixture.json --out certificate.json
```

Or use the repository's composite GitHub Action:

```yaml
- uses: ot4562-glitch/QuotientGuard@v1.0.0
  with:
    input: fixtures/estimator.json
    output: quotientguard-certificate.json
```

Python assertion helpers are also available:

```python
from quotientguard import assert_no_measured_gap

assert_no_measured_gap(certificate)
```

See [docs/CI.md](docs/CI.md).

## CLI

Public surface:

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

Advanced compatibility commands:
- `audit`;
- `aggregate`;
- `trl-receipt`.

## Deterministic score sketches

`HashProjection` provides a dependency-free deterministic linear hash-sign projection.

A non-zero projected gap proves a non-zero component exists in the observed source space. A zero projected gap does **not** certify the unobserved full space.

## Performance

The repository includes a repeatable audit microbenchmark:

```bash
quotientguard benchmark --samples 256 --dimension 64 --repeats 7
```

Benchmark numbers measure QuotientGuard audit runtime only, not end-to-end trainer overhead.

The frozen v1.0 local run (256 observations, 64-dimensional scores, 7 repeats) measured a 9.173 ms median on Python 3.14.4 under WSL2. See [benchmarks/README.md](benchmarks/README.md) for the environment and limitations.

## What QuotientGuard does not claim

QuotientGuard does **not** claim that:
- every measured gap harms downstream reward or quality;
- every non-constant coefficient causes a harmful update;
- removing a measured gap improves task performance;
- generic clipping or response-length bias is newly discovered;
- a zero sketch certifies the full model;
- it is a replacement optimizer;
- it is a legal patentability determination.

## Final v1.0 scope

The public v1.0 scope is frozen. There is no feature roadmap.

Intentionally out of scope:
- GUI/dashboard;
- hosted service;
- LLM semantic clustering in the mathematical core;
- automatic correction;
- automatic optimizer replacement.

See [docs/FINAL_SCOPE.md](docs/FINAL_SCOPE.md).

## Documentation

- [Concepts](docs/CONCEPTS.md)
- [Certificates and statuses](docs/CERTIFICATES.md)
- [Framework integrations](docs/INTEGRATIONS.md)
- [TRL reconstructed adapter](docs/TRL_SEQUENCE_ADAPTER.md)
- [Runtime capture](docs/RUNTIME_CAPTURE.md)
- [Score sketches and Sentinel](docs/SKETCH_SENTINEL.md)
- [CI and regression testing](docs/CI.md)
- [Research relationship](docs/PAPER.md)
- [What QuotientGuard is and is not](docs/COMPARISON.md)
- [Platform validation](docs/PLATFORM.md)
- [Frozen v1.0 benchmark](benchmarks/README.md)

## Citation

```bibtex
@misc{sim2026quotientguard,
  author       = {Minseong Sim},
  title        = {QuotientGuard: Deterministic Estimator-Compatibility Audits for RL/RLVR},
  year         = {2026},
  note         = {Software derived from Diagnosing Estimator Homomorphism Compatibility in Reinforcement Learning with Verifiable Rewards},
  doi          = {10.5281/zenodo.22859044}
}
```

## License

QuotientGuard software is licensed under the [MIT License](LICENSE).

The included research paper is the Zenodo-published preprint and is distributed under **CC BY 4.0**, matching the Zenodo record for DOI `10.5281/zenodo.22859044`.
