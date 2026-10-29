# Contributing to QuotientGuard

QuotientGuard has a stable v1 compatibility contract plus the optional v1.1
PPO objective-aware audit. Contributions should be correctness-driven,
reproducible and conservative about evidence; future objective support must
not silently reinterpret existing certificate statuses.

## Verify a change

```bash
python -m compileall -q quotientguard tests tools
python -m unittest discover -s tests -v
python -m quotientguard certify
python tools/check_release_docs.py
```

For changes involving PPO/PyTorch, also install `.[objective]`, run
`quotientguard objective-benchmark` and validate all four supported
CPU dtypes. For packaging or a release, run `python tools/validate_release.py`
with the objective environment installed; a skipped test is not a pass.

## Scientific and safety contracts

- Preserve fixed-policy-snapshot provenance, exact versus sampled evidence,
  conservative projected-zero semantics, batch-aware sampling, deterministic
  certificate hashing and explicit policy thresholds.
- Keep the old compatibility decision separate from objective-aware verdicts.
  Normal clipping and nonunit importance ratios do not imply a bug.
- Do not claim production framework confirmation from synthetic/manual
  coefficient input, source-equivalent replay or open PRs.
- Do not infer verifier equivalence with an LLM in the mathematical core.
  Unsupported objective modes must fail closed as insufficient evidence.
- Do not commit private plans, unverified external defect assertions,
  credential material or local test artifacts.

## Every new version: mandatory README and release synchronization

Before tagging a new stable version:

1. Implement and test actual functionality; document unsupported modes.
2. Update the version in `pyproject.toml` and
   `quotientguard/version.py` (if maintained separately).
3. Update **both** the README's `Current stable release` line and
   its version-history table, plus the corresponding dated
   `CHANGELOG.md` section. Only describe *shipped and validated* features.
4. Review the **local-only internal** upstream evidence ledger (not part of
   the public source tree). Promote an entry into the README's `Verified
   upstream fixes` table **only** after the upstream PR actually merges and
   is supported by reproducible evidence; identify QuotientGuard's role
   accurately. Never commit internal notes as a public documentation file.
5. Run `python tools/check_release_docs.py` and
   `python tools/validate_release.py`, verify independent wheel/sdist
   install, all required tests/certification, and supported CI environments.
6. Verify GitHub default-branch CI is green and tag matches the code;
   do not force-push or overwrite a published release/tag. A new tag
   must correspond to a fully validated, current README.
7. The tagged-release workflow derives assets, tag and release notes
   from versioned metadata and `CHANGELOG.md`, not an old hardcoded
   wheel version or stale release-note file.

A post-release documentation correction on `main` is not a new
software release and does **not** rewrite the published tag. Fixes
should keep historical release evidence and SHA-based receipts intact.

## Canonical documentation

Start with [README.md](README.md), then
[docs/REFERENCE.md](docs/REFERENCE.md),
[docs/OBJECTIVE_AUDIT.md](docs/OBJECTIVE_AUDIT.md) and
[docs/INTEGRATIONS.md](docs/INTEGRATIONS.md).
Internal feedback and open-PR tracking stay **outside this public checkout**;
they are not shipping or bug-confirmation claims. Historical public research
scope and prior release documents remain retrievable from versioned Git tags.
