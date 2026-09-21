# TRL sequence-level reconstructed adapter

QuotientGuard includes a deliberately narrow reconstructed adapter for the Hugging Face TRL `GRPOTrainer` source used by the research paper.

## Validated scope

The adapter supports the frozen source/configuration family for:

- sequence-level importance sampling;
- beta = 0;
- no extra entropy/off-policy/vLLM policy multiplier in the reconstructed coefficient;
- loss types: GRPO, BNPO, and Dr.GRPO;
- standard two-sided PPO clipping;
- optional full score vectors or deterministic score sketches.

It does not silently extrapolate beyond this scope.

## Frozen source identity

- commit: `a98fa6a4428f9aae58dfb26d729d7437f662f27a`;
- `trl/trainer/grpo_trainer.py` SHA-256:
  `e9f5ad165c620c8da1eceade84b00e405c76691d9af363681e8f7cbbfae8a709`.

Every reconstructed TRL input must declare both the exact commit and exact source hash. Missing or different provenance is rejected.

The hash is independently verifiable:

~~~bash
quotientguard verify-source /path/to/trl/trainer/grpo_trainer.py
~~~

## Source relationship

For the frozen path:

- sequence log-ratio is the mean masked token log-ratio;
- PPO ratio is `exp(sequence log-ratio)`;
- GRPO averages per-sequence loss;
- BNPO divides total active-token loss by the active-count denominator;
- Dr.GRPO divides by batch size times max completion length.

The paper's actual-Trainer direct-gradient gate used this frozen source and reported 5/5 valid fresh seeds under its preregistered protocol.

## Effective coefficient convention

QuotientGuard reports the policy-ascent coefficient `H_i`. Gradient descent on the negative surrogate therefore corresponds to a positive `H_i` times the sequence score.

For an unclipped active surrogate `a_i = ratio_i * advantage_i`:

- GRPO: `H_i = a_i / (B * L_i)`
- BNPO: `H_i = a_i / N_active`
- Dr.GRPO: `H_i = a_i / (B * L_max)`

If the PPO branch is clipped, the derivative through the clamped ratio is zero away from the boundary, so `H_i = 0` for that surrogate branch.

Exact clip boundaries are rejected rather than assigned an arbitrary derivative convention.

## Paper A1 fixture

The canonical fixture is:

`fixtures/paper/a1_equal_length_clipping.json`

It encodes:

- both monitored representatives have completion length 1;
- positive advantage;
- ratio A = 1.3488;
- ratio B = 0.8181;
- epsilon = 0.2;
- batch size = 8.

The adapter emits:

- representative A: clipped-high, `H = 0`;
- representative B: active, `H = 0.8181 / 8 = 0.1022625`.

Because no score vector is supplied in this lightweight fixture, the coefficient screen reports `COEFFICIENT_MISMATCH_OBSERVED`. It does not promote that observation to a measured non-zero compatibility gap or downstream-quality claim.

## CLI

~~~bash
quotientguard demo
quotientguard check fixtures/paper/a1_equal_length_clipping.json
quotientguard explain fixtures/paper/a1_equal_length_clipping.json
~~~

## Different TRL versions

A newer/different TRL source is not reconstructed by guesswork. Integrate it through the framework-neutral explicit runtime capture API in `docs/INTEGRATIONS.md`.
