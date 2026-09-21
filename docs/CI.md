# QuotientGuard CI / Regression Testing

QuotientGuard is designed to act as a semantic regression test for RL/RLVR estimators. CI policy is user supplied; the tool does not invent a universal safe threshold.

## Fast CLI path

Run a certificate:

~~~bash
quotientguard check fixture.json --out candidate.json
~~~

Compare a candidate against a frozen baseline:

~~~bash
quotientguard compare baseline.json candidate.json --fail-on-new-gap
~~~

Optionally reject a large increase in measured gap ratio:

~~~bash
quotientguard compare baseline.json candidate.json --max-gap-ratio-increase 0.02
~~~

The compare command exits non-zero only when an explicitly requested comparison policy fails.

## Public certification suite

~~~bash
quotientguard certify
~~~

This verifies the shipped positive, negative, paper-derived, verifier-equivalence, and sampled-gap regression controls.

## Python assertions

~~~python
from quotientguard import (
    assert_no_measured_gap,
    assert_no_new_measured_gap,
    assert_policy_passed,
)

assert_policy_passed(certificate)
assert_no_measured_gap(certificate)
assert_no_new_measured_gap(comparison)
~~~

These helpers have no pytest dependency and work inside unittest, pytest, or custom CI scripts.

## GitHub Action

~~~yaml
- uses: ot4562-glitch/QuotientGuard@v1.0.0
  with:
    input: fixtures/estimator.json
    output: quotientguard-certificate.json
~~~

The composite action installs the checked-out QuotientGuard release locally and runs `quotientguard check`.

## Sentinel policy

Sentinel consumes separate fixed-policy-snapshot certificates. It does not pool raw observations across optimizer steps.

~~~bash
quotientguard sentinel certs/step-*.json \
  --out sentinel.json \
  --max-measured-gap-events 2 \
  --max-consecutive-gap-events 1 \
  --max-gap-ratio 0.05 \
  --require-same-estimator
~~~

The order supplied on the command line is the history order. Sentinel reports first/last snapshot ids and estimator fingerprint changes; it does not assume monotonic drift.

## Scope boundary

A CI failure means the user-declared estimator-compatibility policy failed. It does not mean model quality or downstream reward necessarily degraded.
