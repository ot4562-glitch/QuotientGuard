# Contributing

QuotientGuard v1.0 is feature-frozen.

Contributions are welcome for:
- correctness fixes;
- documentation corrections;
- test hardening;
- security fixes;
- reproducibility fixes;
- portability fixes that do not change scientific semantics.

Feature expansion is intentionally not part of the public v1.0 plan.

## Before submitting a change

Run:

```bash
python -m compileall -q quotientguard tests
python -m unittest discover -s tests -v
python -m quotientguard certify
```

A change must preserve:
- fixed-policy-snapshot discipline;
- exact versus sampled evidence separation;
- projected-zero conservative semantics;
- user-defined policy thresholds;
- fail-closed reconstructed source support;
- deterministic certificate hashing.

Do not add:
- silent support for unknown trainer semantics;
- LLM semantic equivalence inside the deterministic core;
- downstream-harm labels derived solely from a compatibility gap;
- automatic training correction.

See `docs/FINAL_SCOPE.md`.
