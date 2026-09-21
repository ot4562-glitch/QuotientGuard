# QuotientGuard v1.0 benchmark

This benchmark is a reproducible local measurement of the QuotientGuard sampled fixed-snapshot audit routine.

It is **not** an end-to-end trainer-overhead benchmark.

## Frozen run

Command:

```bash
quotientguard benchmark --samples 256 --dimension 64 --repeats 7
```

Environment:
- QuotientGuard: 1.0.0 installed wheel
- Python: 3.14.4
- OS: Linux 6.18.33.2-microsoft-standard-WSL2 x86_64
- observations: 256
- score dimension: 64
- repeats: 7

Result:
- median: 9.172859 ms
- min: 9.009766 ms
- max: 9.473605 ms
- median throughput: about 27,908 observations/s

Artifact:
- `v1.0.0-local.json`
- SHA-256: `744732c68f5e0559f5b91bf6cfc6ea214c32bde9795a54ad850a59a5ba099fc0`

Runtime varies by hardware and environment. The value above is evidence that the deterministic audit itself is lightweight on this local fixture; it is not a universal performance guarantee.
