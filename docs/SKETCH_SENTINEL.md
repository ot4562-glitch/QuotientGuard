# Score Sketches and Sentinel

## Deterministic score sketches

`HashProjection` implements a dependency-free deterministic linear hash-sign projection.

~~~python
from quotientguard import HashProjection

projection = HashProjection(dimension=64, seed="my-audit-v1")
sketch = projection.project(full_score)
~~~

The projection is reproducible for a fixed seed.

Interpretation is deliberately one-sided:

- non-zero projected gap establishes a non-zero observed source component;
- zero projected gap does **not** certify that the full source-space gap is zero.

## Sentinel

Sentinel compares a sequence of already-issued certificates. It never merges raw observations from different policy snapshots.

Tracked fields include:

- first and last policy snapshot id;
- repeated measured gap events;
- longest consecutive gap-event streak;
- maximum measured gap ratio;
- coefficient spread;
- estimator fingerprint changes;
- optional snapshot timestamps supplied by the caller.

No monotonicity assumption is made.

## Local microbenchmark

`quotientguard benchmark` measures the audit routine itself:

~~~bash
quotientguard benchmark --samples 256 --dimension 64 --repeats 7
~~~

It is not a measurement of end-to-end Trainer overhead. End-to-end overhead must be measured inside a real supported training run.
