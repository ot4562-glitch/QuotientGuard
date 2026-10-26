# Engineering feedback and upstream evidence ledger

**Updated:** 2026-10-27. This document is an engineering backlog and an
evidence-status ledger, **not** a list of QuotientGuard-autonomously discovered
or confirmed production defects. Source-level numerical findings and synthetic
coefficient captures are explicitly weaker than native framework integration.

## Axolotl PR #4209: reviewer feedback to turn into product tests

Review: [axolotl-ai-cloud/axolotl#4209](https://github.com/axolotl-ai-cloud/axolotl/pull/4209).
The reviewer asked whether masking before the exponential preserves the
sequence-level denominator and whether flattened/non-flattened log-prob
forward paths share the same guard. Our reply and added test commit:
[3507a09](https://github.com/axolotl-ai-cloud/axolotl/commit/3507a09182f9bb4c27544cabab41e759471a2c7f).
The PR remains open as of the date above. Full Axolotl runtime/GPU validation
is **not** established.

### 1. Denominator semantics are part of the objective

The reviewed `AsyncGRPOTrainer._compute_loss` sequence-level correction uses:

```text
valid_count_i = sum_t mask[i,t]
mean_log_ratio_i = sum_t(mask[i,t] * (new_logp[i,t]-old_logp[i,t])) / max(valid_count_i, 1)
sequence_ratio_i = exp(mean_log_ratio_i)
grpo_loss = mean_i(sum_t(mask[i,t] * token_loss[i,t]) / max(valid_count_i, 1))
```

The mask is applied to the log ratio **before exponentiation**. Setting a
masked log-ratio to zero makes its token ratio `exp(0)=1`, which is harmless
only because the objective also excludes that token from both active-token
sum and the applicable denominator. The reduction denominator is not the
padded sequence length. DR-GRPO deliberately uses another normalizer
(`batch_size * max_completion_length`) and must not be relabeled as broken
for that reason.

**Gap in v1.1:** `ObjectiveSpec(ratio_level="sequence_product")` uses
`exp(sum_t(mask * log_ratio))`, whereas Axolotl's sequence mean ratio is
`exp(sum_t(mask * log_ratio)/max(sum_t mask, 1))`. They are *different
objectives*. For two active log ratios of 0.1, the first yields `exp(0.2)`
(~1.2214) and the second `exp(0.1)` (~1.1052). With a 1.2 upper clip,
a positive-advantage gradient can be clipped in one and active in the other.
Never map these modes to one another by name.

### 2. Execution-path coverage is part of the evidence

`_get_per_token_logps_and_entropies` and
`_get_per_token_logps_and_entropies_flattened` are two scoring paths
feeding the **same** shared importance-weight/objective block in
`AsyncGRPOTrainer._compute_loss`. Normal
`FastAsyncGRPOTrainer._compute_loss` also delegates to that method.
An audit on only one mocked scorer cannot certify an alternate scorer's
real tensor shapes, padding construction or numerical behavior. Branch
coverage and real forward/backward integration are separate levels.

The revised upstream tests exercise token/sequence importance modes,
overflow/`-inf - -inf` masked cases, both scoring branches and exact
gradient assertions. A nonzero log-ratio test with sequences containing
2 versus 3 active tokens catches full-length denominator dilution.

### 3. Operator ordering and forward/backward evidence

- `exp(log_ratio)` followed by `* 0` can create `inf * 0 = NaN`.
- `(-inf) - (-inf)` can create NaN on excluded positions even before exp.
- Masking *before exp* via a branch selection, not a post-hoc
  `nan_to_num`, is a defensible numerical fix when the contract excludes
  those positions.
- Verifying a finite forward loss is insufficient: require actual
  `torch.autograd.grad`/backward evidence, exact zero gradients for excluded
  positions, and unchanged valid-token gradients and clipping.
- `torch.where` applied **only after** unsafe arithmetic is not a universal
  backward fix. Track the full operator graph and stop-gradient boundaries.

## Proposed next-version additions (NOT shipped in v1.1.0)

| Priority | Capability | Acceptance test |
| --- | --- | --- |
| P0 | Explicit `sequence_masked_geometric_mean` ratio level | Two vs three active tokens yield the same `exp(0.1)` and autograd matches |
| P0 | Reduction semantics: per-sequence valid mean vs global active mean vs fixed-length Dr.GRPO | Expected coefficients and normalizers match active tokens and all-masked contract |
| P0 | Mask-order and inactive-path numerical audit | Distinguish safe pre-exp masking from post-exp `inf*0`, including backward NaNs |
| P1 | Path-coverage/provenance manifest for flattened, normal, FastAsync, optional KL and off-policy routes | Unexecuted paths remain `INSUFFICIENT_EVIDENCE`, never `CONFIRMED` |
| P1 | Adversarial matrix of clip signs, masks, ratios, dtypes and rejection weights | Normal PPO clipping is never mislabeled a defect; faults get specific stage evidence |
| P1 | Before/after upstream integration harness | Real imported loss and backward, supported config, pinned commit, provenance and actual tests required |

Do not weaken v1.1 confirmation gates, silently change the v1 compatibility
certificate, or present local synthetic benchmarks as external sensitivity or
false-positive measurements.

## Upstream contribution tracker

Snapshot verified through GitHub on **2026-10-27**. These contributions
**mention/use QuotientGuard in an audit or provenance role**; the original
findings were based primarily on source review and isolated PyTorch
reproductions. In particular, the historical ten-repo screening produced
seven numerical candidates, **zero native framework confirmations**, and
**zero autonomous QuotientGuard bug discoveries**. None of the PRs listed
below had merged when checked.

| Repository | PR | Status at snapshot | Evidence/use limitation |
| --- | --- | --- | --- |
| OpenRLHF | [#1382](https://github.com/OpenRLHF/OpenRLHF/pull/1382) | OPEN, not merged | Manual source/PyTorch investigation; QG-mentioned |
| NVIDIA NeMo RL | [#4883](https://github.com/NVIDIA-NeMo/RL/pull/4883) | OPEN, not merged | Manual source/PyTorch investigation; QG-mentioned |
| LLM-Dojo | [#40](https://github.com/mst272/LLM-Dojo/pull/40) | OPEN, not merged | Manual source/PyTorch investigation; QG-mentioned |
| Open-Reasoner-Zero | [#77](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero/pull/77) | OPEN, not merged | Source-equivalent CPU & synthetic QG coefficient controls |
| Axolotl | [#4209](https://github.com/axolotl-ai-cloud/axolotl/pull/4209) | OPEN, not merged | Source-level fix and expanded mocked-scorer regression; native full tests outstanding |
| trlx | [#614](https://github.com/CarperAI/trlx/pull/614) | OPEN, not merged | Source-equivalent CPU & synthetic QG coefficient controls |
| DeepSpeedExamples | [#1022](https://github.com/deepspeedai/DeepSpeedExamples/pull/1022) | OPEN, not merged | Source-equivalent CPU; DCO pass does not establish bug confirmation |
| torchtune | [#3006](https://github.com/meta-pytorch/torchtune/pull/3006) | CLOSED, not merged | Withdrawn; never claim upstream adoption |

Also pending: [ms-swift issue #10431](https://github.com/modelscope/ms-swift/issues/10431)
is a potential masked GRPO numerical edge case, **not** a submitted PR or
confirmed training bug.

### Promotion policy for README "Verified upstream fixes"

1. Verify the **upstream PR really merged** (not merely approved, green CI, or
   an open fork PR); record upstream repo, merged URL, merge SHA and date.
2. Confirm exactly what the fix and regression establish. Distinguish
   `QG-assisted triage`, `QG-native audit`, and independently autonomous
   discovery. A PR containing "QuotientGuard" is **not** autonomous discovery.
3. Link reproduction/test evidence and disclose material limits
   (synthetic/source-equivalent vs real framework integration).
4. Add the merged entry to the README table and annotate the applicable
   release in `CHANGELOG.md`. If a PR does not merge, do **not** count it
   as a feature, integration, adoption, or confirmed product detection.
5. Recheck status before each version's public README/release; the snapshot
   above becomes stale and is not an ongoing monitor.

## Documentation and release maintenance

The README is the public quick start, product scope and current **stable
version**; the changelog holds version-by-version release facts.
`docs/REFERENCE.md` holds v1 evidence math,
`docs/OBJECTIVE_AUDIT.md` the v1.1 supported semantics,
`docs/INTEGRATIONS.md` actual capture/CI paths and
`docs/VALIDATION_V1_1.md` historical validation.
Keep this feedback ledger as the actionable backlog and PR tracker.

Every future version bump must update `pyproject.toml`, `README.md`,
`CHANGELOG.md` and release evidence together; the CI release-docs check
must fail if stable-version metadata disagree. Past release tags and
validation artifacts are immutable; documentation cleanup does not rewrite
the released v1.1.0 tag.
