# Concepts

QuotientGuard asks one narrow question:

> If a reward, verifier, or environment says several concrete trajectories mean the same outcome, does the estimator update treat them equivalently?

## Equivalence class

A quotient/equivalence class groups concrete trajectories that the task treats as the same outcome.

Examples:
- different tokenizations of the same decoded answer;
- `"4"` and `"+4"` when a verifier parses both as integer 4;
- different action sequences that reach the same environment-defined terminal state.

The deterministic core never invents semantic equivalence with an LLM. The user or environment defines it.

## Effective coefficient

For a sampled trajectory, the production estimator contributes a score direction multiplied by a scalar. QuotientGuard calls the realized scalar `H_i`.

A representative-conditioned conditional mean is conceptually `h(y)=E[H_i|Y_i=y]`.

One observed `H_i` is not automatically `h(y)`.

## Compatibility gap

Inside an equivalence class, the policy score can be decomposed into:
- movement of the abstract outcome itself; and
- movement among concrete representatives while staying inside the same outcome class.

QuotientGuard measures the estimator update in the second direction.

For a sampled fixed-policy snapshot:

```text
s_bar = mean(score_i)
fiber_i = score_i - s_bar
Gamma_hat = mean(H_i * fiber_i)
```

A non-zero measured gap is an estimator-compatibility observation. It is not automatically a model-quality or downstream-reward verdict.

## Exact versus sampled evidence

Exact/analytic class-constant coefficients can satisfy the paper's sufficient condition.

Sampled coefficients are weaker evidence:
- unequal realized/estimated coefficients can trigger escalation;
- equal finite-sample estimates are not promoted to an exact theorem certificate.

## Full score versus sketch

A deterministic linear sketch can cheaply expose a non-zero component.

- non-zero projected gap => a non-zero source-space component exists;
- projected zero => the selected projection saw zero, but unobserved directions may remain.

## Fixed policy snapshots

One audit corresponds to one fixed `policy_snapshot_id`.

Raw observations from different optimizer steps must not be pooled into one theorem audit. Sentinel combines already-issued per-snapshot certificates instead.
