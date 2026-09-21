# What QuotientGuard is — and is not

## Versus ordinary gradient logging

Gradient logging records what update occurred.

QuotientGuard adds a task-defined equivalence contract and asks whether the estimator update contains movement *within* a reward/verifier-equivalent class.

## Versus metamorphic testing

Metamorphic tests check whether behavior is invariant under chosen transformations.

QuotientGuard can use a similar equivalence idea, but the audited object is the stochastic estimator/update geometry, with explicit coefficient and score evidence.

## Versus tokenizer analysis

Tokenizer tools explain how text maps to token sequences.

QuotientGuard is not a tokenizer checker. Distinct decoded strings may share one verifier-defined class, and arbitrary environment equivalence can be supplied by the user.

## Versus training observability dashboards

Observability systems track metrics over time.

QuotientGuard emits deterministic per-snapshot compatibility certificates; Sentinel summarizes those certificates without pooling raw observations across optimizer steps.

## Versus a replacement optimizer

QuotientGuard does not alter training by default.

It is an assurance and semantic-regression-testing tool. Automatic correction/optimizer replacement is intentionally outside v1.0 scope.
