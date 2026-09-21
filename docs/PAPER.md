# Research relationship

QuotientGuard is the productization of the estimator-compatibility framework developed in:

**Minseong Sim (2026), _Diagnosing Estimator Homomorphism Compatibility in Reinforcement Learning with Verifiable Rewards_.**

DOI: **10.5281/zenodo.22859044**

Repository copy: [`../paper/Sim_2026_Estimator_Homomorphism_Compatibility_RLVR_Preprint.pdf`](../paper/Sim_2026_Estimator_Homomorphism_Compatibility_RLVR_Preprint.pdf). The paper is distributed under CC BY 4.0 according to the Zenodo record; QuotientGuard software is separately MIT-licensed.

## What the product carries forward

- quotient/reward-equivalence contracts;
- representative-conditioned estimator coefficients;
- quotient/fiber score decomposition;
- compatibility-gap measurement;
- exact class-constancy sufficient-condition screening;
- source-sensitive TRL evidence;
- fixed-policy-snapshot discipline;
- conservative projected-gap semantics.

## Research-backed TRL source

The reconstructed TRL adapter is frozen to:

- commit: `a98fa6a4428f9aae58dfb26d729d7437f662f27a`;
- `trl/trainer/grpo_trainer.py` SHA-256:
  `e9f5ad165c620c8da1eceade84b00e405c76691d9af363681e8f7cbbfae8a709`.

The research program directly exercised the actual frozen `GRPOTrainer` path. The research protocol's actual-Trainer direct-gradient v0.3 gate reported 5/5 valid fresh seeds under its frozen protocol.

The product does not extrapolate reconstructed semantics to unknown TRL versions.

## Public fixtures

The repository separates:
- paper-derived fixtures;
- paper-relation structural controls;
- synthetic positive/negative regression controls.

Synthetic controls are never presented as research measurements.

## Limitations preserved

QuotientGuard does not claim:
- every compatibility gap harms downstream reward;
- removing a gap improves task quality;
- generic clipping or length bias is novel;
- a projected zero certifies the full model;
- the software is a legal patentability determination.
