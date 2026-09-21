# Certificates and status semantics

Every QuotientGuard result is tied to:
- a fixed policy snapshot;
- estimator provenance/fingerprint;
- an equivalence-contract fingerprint;
- an evidence level;
- source input hash;
- deterministic certificate SHA-256.

## Exact statuses

### `CERTIFIED_BY_EXACT_CLASS_CONSTANCY`
Exact/analytic representative-conditioned coefficients are class-constant, satisfying the paper's sufficient condition.

### `MEASURED_COMPATIBLE`
An exact supplied full score model has a compatibility gap within numerical tolerance.

### `MEASURED_GAP`
An exact supplied full score model has a non-zero compatibility gap.

### `MEASURED_PROJECTED_GAP`
An exact supplied projection/sketch contains a non-zero gap component.

### `PROJECTED_ZERO_NOT_CERTIFIED`
The supplied exact projection saw zero, but unobserved score-space directions remain.

### `NOT_CERTIFIED`
Available exact coefficient-only evidence is insufficient to certify compatibility.

## Sampled statuses

### `COEFFICIENT_MISMATCH_OBSERVED`
Realized or estimated coefficients differ inside one equivalence class.

### `ESTIMATED_CLASS_CONSTANCY`
Repeated sampled coefficient estimates agree within tolerance. This is not an exact theorem certificate.

### `INSUFFICIENT_REPETITION`
Coefficient-only data do not contain enough repetition to estimate representative-conditioned means.

### `MEASURED_SAMPLE_GAP`
A non-zero sampled full-score gap was observed at one fixed policy snapshot.

### `MEASURED_PROJECTED_SAMPLE_GAP`
A non-zero sampled projected/sketched gap was observed.

### `SAMPLED_ZERO_NOT_CERTIFIED`
No sampled full-score gap was observed above tolerance; population compatibility is not certified.

### `SAMPLED_PROJECTED_ZERO_NOT_CERTIFIED`
No sampled projected gap was observed above tolerance; the full score space is not certified.

## Policy status

`PASS` and `FAIL` are only emitted against explicit user policy thresholds.

Without a policy threshold, QuotientGuard reports `OBSERVED`.

A policy failure means the declared estimator-compatibility policy failed. It does not mean model quality necessarily degraded.
