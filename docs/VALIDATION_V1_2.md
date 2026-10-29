# v1.2.0 release: scope and validation

Version 1.2.0 is the stable public release. The local acceptance receipt below
records a pre-publication validation on 2026-10-29; it must not be mistaken for
proof of a remote CI matrix result or independent third-party bug verification.

## Accepted product scope

- Existing compatibility audits, certificates, CLI exits and v1.1 PPO contracts.
- Explicit token/product/geometric-mean/sequence-token ratios; five reductions;
  empty-mask policies; masking order and stop-gradient boundaries.
- Composite PPO/CISPO/SAPO policy, k1/k2/k3 KL, supplied entropy, gated old/rollout
  correction and inside-clipping GSPO rollout-reference replacement, with
  independently derived input gradients and literal PyTorch replay.
- Source-bound controlled callable comparisons; measured parameter VJP comparison;
  disposable plain-SGD/autocast/scaler/accumulation/clipping validation; supplied
  process-count/mask-bound microbatch/full-batch loss and gradient checks.
- Bounded, immutable-commit source-only Python inspection, explicit candidates,
  component classification, verified caching, quotas, exports and continuation.

Detailed contracts and unsupported combinations are specified in
[Objective audit](OBJECTIVE_AUDIT.md) and [Integrations](INTEGRATIONS.md).

## Reproducible acceptance gate

Install the `objective` extra and `build`, then run:

```bash
python tools/check_release_docs.py
python tools/validate_release.py
```

The gate requires the complete suite with zero skipped tests, legacy certification,
independent seeded objective controls/holdout, compile and CLI examples, wheel and
sdist builds, clean dependency-free installs, installed-package identities, and
installed full/correction/GSPO/window/parameter/training/source-inspection checks.
Logs and machine-readable receipts bind the exact artifacts and environment.
The base package also has a separate dependency-free test gate; optional Torch
tests are skipped only there. Missing optional CLI dependencies produce an
actionable message and exit 2.

CPU mathematical checks cover float16/bfloat16/float32/float64. Prior source-bound
CUDA/native and representative Trainer observations are separate environment-
specific evidence. A CPU package gate never claims fresh CUDA or remote CI
execution. The repository's CI defines broader Python/OS checks; local success
does not claim that unexecuted matrix has passed.

## Fixed 952-repository source inspection

A pinned 952-repository, immutable-commit cohort was processed using the
installed v1.2.0 package and bounded Python-source inspection. The final local
receipt recorded 914 `COMPLETED`, 38 `PARTIAL`, 0 failed or pending repositories;
92,418 known Python paths were processed. Python AST succeeded for 92,232 files;
the other 186 statuses were 77 syntax errors, one decode error, three links
skipped and 105 unsafe paths. These are not silently counted as successful
source analysis.

The scanner reported 8,881 static risk candidates and 6,186
source/pattern/line groups. **No third-party defect was confirmed through an
executed reproduction.** The surveyed repositories were not installed,
imported or executed. Results are tied to pinned historical commits rather
than each project's current default branch. The internal local evidence store
and candidate-review ledger are intentionally not included in this release.

## Evidence boundaries

Static candidates are syntax/provenance observations. Controlled native input
mutations, independent reference replay, full Trainer observations and naturally
occurring failures retain separate grades. Source family identity does not prove
implementation independence. Inspection completion does not certify correctness
of every repository or establish that a candidate is a defect.

Arbitrary optimizer mathematics (including Adam, momentum and weight decay),
distributed collectives, arbitrary mixed-precision graphs, entropy/advantage/
sampling-producer inference, automatic objective reconstruction and undeclared
framework branches are outside this version's supported contracts. Unsupported
paths remain explicit; no natural failure or universal framework certification
is required or implied by a local contract match.
