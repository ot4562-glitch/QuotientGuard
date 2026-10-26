# QuotientGuard v1.1 release checklist

Release identity: 1.1.0 / v1.1.0 / stable. The owner's 2026-10-26 instruction
authorizes implementation, validation, GitHub main update, tag and release.
No PyPI publication is included.

## Required gates

- Full local tests with torch imported and no skips.
- Public `python -m quotientguard certify` (5 cases).
- Independent PPO control and holdout benchmarks, all four supported CPU dtypes.
- Existing certificate/API regression and GRPO boundary controls.
- Wheel/sdist build, independent clean installs, README CLI smokes, installed oracle.
- Final review and diff checks for secrets, private/local paths and large artifacts.
- Commit only release files; preserve pre-existing untracked local AGENTS.md.
- Recheck remote main ancestry; never force-push or overwrite a tag/release.
- After pushing, all CI checks must succeed before tagging/publishing.
- Attach validated wheel, sdist, SHA256SUMS and JSON verification evidence.
- Recheck release stable/non-draft state, tag commit, default branch and assets.

See `docs/VALIDATION_V1_1.md` and `benchmarks/release-validation-v1.1.0.json` for
local verification. Remote release identity/CI status is reported after deployment;
this checklist does not claim publication before it occurs.
