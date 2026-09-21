# QuotientGuard v1.0.0

QuotientGuard is a deterministic estimator-compatibility audit and semantic regression-test tool for RL/RLVR training.

## Main surfaces

- deterministic exact and sampled compatibility audits;
- batch-cluster jackknife uncertainty;
- full-score and deterministic-sketch gap evidence;
- baseline/candidate regression comparison;
- Sentinel persistence monitoring;
- frozen source-hash-verified TRL reconstruction path;
- explicit runtime capture for verl, OpenRLHF, and custom trainers;
- verifier/decoded equivalence contracts;
- Trainer Certification Suite;
- reusable GitHub Action and CI helpers;
- optional agent-neutral workflow wrapper.

## Research basis

Derived from:

Minseong Sim (2026), *Diagnosing Estimator Homomorphism Compatibility in Reinforcement Learning with Verifiable Rewards*.

DOI: 10.5281/zenodo.22859044

The Zenodo-published PDF is included in the repository and attached to the GitHub Release. The paper is CC BY 4.0; the QuotientGuard software is MIT-licensed.

## Frozen TRL provenance

- commit: `a98fa6a4428f9aae58dfb26d729d7437f662f27a`
- `grpo_trainer.py` SHA-256:
  `e9f5ad165c620c8da1eceade84b00e405c76691d9af363681e8f7cbbfae8a709`

Unknown reconstructed source semantics fail closed.

## Scope

v1.0 is feature-frozen. GUI, hosted services, LLM semantic clustering in the mathematical core, and automatic optimizer/correction behavior are intentionally outside scope.
