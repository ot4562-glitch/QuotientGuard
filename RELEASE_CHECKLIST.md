# QuotientGuard v1.0 Release Checklist

## Identity
- [x] Repository name frozen: `QuotientGuard`.
- [x] Python package name frozen: `quotientguard`.
- [x] Repository description frozen.
- [x] License: MIT.
- [x] Paper DOI linked.
- [x] GitHub repository-name search found no existing `QuotientGuard` repository on 2026-09-21.
- [x] PyPI index check on 2026-09-21 returned `No matching distribution found for quotientguard`.
- [x] Re-checked immediately before public create/upload on 2026-09-21: GitHub repository absent; PyPI returned no matching distribution.

## Product
- [x] Audit core.
- [x] Exact/sample evidence separation.
- [x] Batch-cluster uncertainty.
- [x] Full/projected gap semantics.
- [x] Equivalence-contract helpers.
- [x] Compare/regression.
- [x] Mechanism explanation.
- [x] Sentinel.
- [x] Deterministic score sketches.
- [x] TRL commit + source SHA-256 verification.
- [x] TRL runtime reconstruction validation.
- [x] verl explicit runtime adapter.
- [x] OpenRLHF explicit runtime adapter.
- [x] custom callback runtime adapter.
- [x] Trainer Certification Suite.
- [x] Certification suite bundled inside the wheel; `quotientguard certify` is self-contained.
- [x] GitHub Action/CI surface.
- [x] Agent-neutral Skill validated and packaged with skill-creator tooling.

## Fixtures
- [x] Exact compatible control.
- [x] Exact measured-gap control.
- [x] Paper A1 clipping fixture.
- [x] Verifier-equivalence structural fixture.
- [x] Sampled projected-gap control.
- [x] Research versus synthetic provenance is explicit.
- [x] Public certification suite passes 5/5.
- [x] v1.0 certification result SHA: `86c5bc36c25b561ab7ab8c01c999ca49cb4b9953b9952d307248655d6bcc99fe`.

## Tests and packaging
- [x] Source compile passes.
- [x] Source unit suite passes: 69/69.
- [x] Frozen TRL vendor source hash verifies.
- [x] Clean wheel and sdist build without packaging warnings.
- [x] sdist contains docs, fixtures, Skill source, action, workflows, and release docs.
- [x] Wheel contains the built-in certification fixtures.
- [x] Clean installed-wheel CLI smoke passes.
- [x] Clean installed-wheel unit suite passes: 69/69 outside the source tree.
- [x] Clean installed-wheel built-in certification passes: 5/5 without repository fixtures.
- [x] Linux/WSL clean install passes on Python 3.14.4.
- [x] Wheel is platform-independent `py3-none-any`.
- [ ] Public CI matrix green on Ubuntu + Windows + macOS, Python 3.10-3.14.
- [x] Local native-Windows limitation documented: the development PC exposes no PowerShell `python`/`py` command; Windows is therefore a mandatory public-CI gate.

## Benchmark
- [x] Frozen v1.0 audit benchmark stored at `benchmarks/v1.0.0-local.json`.
- [x] Benchmark documentation records environment and limitations.
- [x] Frozen local median: 9.172859 ms for 256 observations x 64 dimensions x 7 repeats.
- [x] Benchmark is explicitly not presented as end-to-end trainer overhead.

## Repository hygiene
- [x] Remove local `.venv`.
- [x] Remove `__pycache__`.
- [x] Remove generated `*.egg-info`.
- [x] Remove generated local `build/` and root certificate/report/temp outputs.
- [x] `.gitignore` excludes local/build artifacts.
- [x] Secret/private-path scan passes.
- [x] No private planning, patent, or unpublished research notes inside the public repo.
- [x] No accidental absolute local paths/usernames in public files.

## Documentation
- [x] README final first screen/quickstart.
- [x] Concepts.
- [x] Certificate/status semantics.
- [x] Integrations.
- [x] TRL scope.
- [x] Runtime capture.
- [x] Sketch/Sentinel.
- [x] CI.
- [x] Paper relationship.
- [x] Comparison/non-goals.
- [x] Platform validation.
- [x] Frozen benchmark.
- [x] Final feature freeze.
- [x] SECURITY.md.
- [x] CONTRIBUTING.md.
- [x] RELEASE_NOTES.md.
- [x] No public post-v1 feature roadmap.

## GitHub publication
- [ ] Create repository `ot4562-glitch/QuotientGuard`.
- [ ] Set description exactly:
  `Find gradient updates your reward did not ask for — deterministic estimator-compatibility audits and regression tests for RL/RLVR.`
- [ ] Set topics: `reinforcement-learning`, `rlhf`, `rlvr`, `grpo`, `ppo`, `post-training`, `audit`, `ci`, `trl`.
- [ ] Default branch `main`.
- [ ] Push clean source.
- [ ] Confirm all 15 OS/Python CI jobs green.
- [ ] Tag `v1.0.0` only after CI is green.
- [ ] Create GitHub Release titled `QuotientGuard v1.0.0`.
- [ ] Attach wheel, sdist, SHA256SUMS, certification result, benchmark result, `skill.zip`, the published paper PDF, and MIT `LICENSE`.
- [ ] Verify release notes and DOI links.
- [ ] Final public-page check from logged-out view.

## Release invariant

Do not tag or publish v1.0.0 until every unchecked item above the GitHub-publication section is either completed or explicitly resolved by the public CI gate. v1.0 has no feature roadmap; later changes are limited to correctness, security, documentation, reproducibility, and portability fixes.
