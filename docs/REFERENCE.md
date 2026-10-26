# QuotientGuard reference

This is the compact reference for the deterministic v1 compatibility core. For the
optional PPO diagnostic, use [OBJECTIVE_AUDIT.md](OBJECTIVE_AUDIT.md); for exact
framework capture and CI, use [INTEGRATIONS.md](INTEGRATIONS.md).

## Core idea and evidence

QuotientGuard asks whether an estimator assigns different update treatment to
concrete trajectories that an **explicit task/verifier equivalence contract**
places in one abstract outcome class. Equivalence is supplied by a verifier,
environment, or user; the deterministic core does not infer it with an LLM.
Examples include `"4"` and `"+4"` under an integer-answer verifier.

For trajectory `i`, the effective update coefficient is `H_i`; the score
direction is `s_i`. One realized `H_i` is **not** an estimate of the population
conditional mean `h(y)=E[H_i | Y=y]` without appropriate repetition.

Within a fixed-policy-snapshot equivalence class, a sampled compatibility-gap
estimate uses the *joint* score/coefficient observations:

```text
s_bar = mean(score_i)
fiber_i = score_i - s_bar
Gamma_hat = mean(H_i * fiber_i)
```

Do not substitute `E[H] E[s]` for `E[H s]`. A nonzero measured gap means
within-class estimator movement, **not** necessarily lower reward, degraded
model quality, or an implementation defect. Exact sufficient conditions and
sampled observations are deliberately different evidence levels.

Each certificate binds policy snapshot, estimator and equivalence-contract
fingerprints, evidence level, input hash, and deterministic SHA-256. Never pool
raw observations from distinct optimizer snapshots into one theorem audit.

### Compatibility status dictionary

| Status | Correct interpretation |
| --- | --- |
| `CERTIFIED_BY_EXACT_CLASS_CONSTANCY` | Exact/analytic class-constant coefficients satisfy a sufficient condition |
| `MEASURED_COMPATIBLE` | Supplied exact full-score model has a gap within tolerance |
| `MEASURED_GAP` | Nonzero gap in supplied exact full-score model |
| `MEASURED_PROJECTED_GAP` | Nonzero gap component in an exact projection |
| `PROJECTED_ZERO_NOT_CERTIFIED` | Projection saw zero; unseen full-space directions remain |
| `NOT_CERTIFIED` | Exact coefficient-only evidence is insufficient |
| `COEFFICIENT_MISMATCH_OBSERVED` | Sampled/realized coefficients differ within one class |
| `ESTIMATED_CLASS_CONSTANCY` | Repeated sampled estimates agree; no exact certification |
| `INSUFFICIENT_REPETITION` | Too few independent observations for the estimate |
| `MEASURED_SAMPLE_GAP` | Nonzero fixed-snapshot sampled full-score gap |
| `MEASURED_PROJECTED_SAMPLE_GAP` | Nonzero sampled projected gap |
| `SAMPLED_ZERO_NOT_CERTIFIED` | Sampled full-score estimate was near zero; no population certificate |
| `SAMPLED_PROJECTED_ZERO_NOT_CERTIFIED` | Sampled sketch was near zero; no full-space/population certificate |

A policy becomes `PASS`/`FAIL` **only** against a user-declared threshold.
Without a policy, the status is `OBSERVED`. A compatibility certificate and
v1.1 objective interpretation are independent: intentional PPO clipping may
explain coefficient variation while the compatibility observation remains true.

## Features and boundaries

- **Exact and sampled audits:** declared equivalence classes, per-snapshot
  evidence, batch-cluster jackknife for sampled direct gaps.
- **HashProjection:** deterministic linear hash-sign score sketch. Nonzero
  projected gap proves an observed source-space component; **zero projection
  never proves full-space zero**.
- **Regression compare:** user-selected `--fail-on-new-gap` or numerical
  threshold; no universal safe threshold is inferred.
- **Sentinel:** compares *issued snapshot certificates*, tracking gap event
  counts/streaks, gap ratios, fingerprints and first/last snapshot. It never
  pools raw trajectories across steps or presumes monotonic degradation.
- **Independent certification:** `quotientguard certify` checks the frozen
  exact, paper-derived, verifier and sampled controls.
- **Core dependency policy:** no required runtime dependencies for the v1
  math/CLI/certification; PyTorch is optional for v1.1 reference replay.
- **Non-goals:** hosted telemetry/dashboard, LLM-derived equivalence in the
  mathematical core, automatic gradient repair, automatic optimizer replacement,
  silent reconstruction of unfamiliar trainers, or downstream-harm claims.

## Reproduction and research

```bash
quotientguard demo
quotientguard check fixtures/paper/a1_equal_length_clipping.json
quotientguard compare baseline.json candidate.json --fail-on-new-gap
quotientguard sentinel certs/step-*.json --max-gap-ratio 0.05 --require-same-estimator
quotientguard benchmark --samples 256 --dimension 64 --repeats 7
```

The paper's equal-length A1 demonstration uses same reward, advantage and
completion length for two verifier-equivalent answers; different PPO clip
branches yield different effective coefficients. It is **not** a new general
claim about clipping or a measured downstream-quality improvement.

Research: Minseong Sim (2026), *Diagnosing Estimator Homomorphism
Compatibility in Reinforcement Learning with Verifiable Rewards*, DOI
[10.5281/zenodo.22859044](https://doi.org/10.5281/zenodo.22859044).
The [preprint PDF](../paper/Sim_2026_Estimator_Homomorphism_Compatibility_RLVR_Preprint.pdf)
is distributed separately under CC BY 4.0; software is MIT.

The v1.0 public feature scope is historically frozen at
[Git tag v1.0.0](https://github.com/ot4562-glitch/QuotientGuard/tree/v1.0.0);
v1.1 adds optional interpretation without changing those compatibility
certificates. Old plan/validation documents remain recoverable from their
original Git tags, rather than duplicated in today's documentation.

For a historical local performance measurement, see
[benchmarks/README.md](../benchmarks/README.md). Do not equate its isolated
audit microbenchmark with end-to-end training overhead.
