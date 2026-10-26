"""Independent controls and seeded holdout, never trained on historical candidates."""
from __future__ import annotations

import math
import random

from .objective import ObjectiveSpec, ObjectiveTrace, audit_objective


def _execute(spec, current, old, advantages, mask, weights=None):
    """Control implementation: max of *negative* surrogates, separate from oracle."""
    import torch
    dtype = getattr(torch, spec.dtype)
    x = torch.tensor(current, dtype=dtype, requires_grad=True)
    old = torch.tensor(old, dtype=dtype)
    a = torch.tensor(advantages, dtype=dtype)
    m = torch.tensor(mask, dtype=dtype)
    delta = x - old
    if spec.ratio_level == "sequence_product":
        delta = (delta * m).sum(dim=1, keepdim=True)
    ratio = delta.exp()
    raw = -(ratio * a)
    capped = -(ratio.clamp(spec.clip_lower, spec.clip_upper) * a)
    loss_tokens = torch.maximum(raw, capped)
    if weights is not None:
        loss_tokens = loss_tokens * torch.tensor(weights, dtype=dtype)
    loss_tokens = loss_tokens * m
    denominator = (1 if spec.reduction == "sum" else spec.denominator
                   if spec.reduction == "explicit" else m.sum())
    loss = loss_tokens.sum() / denominator
    loss.backward()
    trace = ObjectiveTrace(x, old, a, m, importance_weights=weights,
        observed_coefficients=x.grad * spec.gradient_scale, observed_loss=loss,
        provenance={"execution_kind": "SYNTHETIC_TORCH_CONTROL", "synthetic_inputs": True})
    return audit_objective(spec, trace, oracle=True)


def _partition(seed, normal_count):
    rng = random.Random(seed)
    cases = []
    def add(name, expected, result, coefficient_spread=False):
        cases.append({"case": name, "expected": expected, "actual": result.verdict,
            "pass": result.verdict == expected, "audit_sha256": result.to_dict()["audit_sha256"],
            "coefficient_spread": coefficient_spread})

    for dtype in ("float16", "bfloat16", "float32", "float64"):
        for i in range(normal_count):
            # Ratios deliberately separated from clip boundaries, including bf16 ULPs.
            ratios = [rng.choice((.35, .5, 1., 1.5, 1.8)) for _ in range(6)]
            advantages = [[rng.choice((-3., -1., 0., 1., 2.)) for _ in ratios]]
            mask = [[rng.choice((0, 1)) for _ in ratios]]
            mask[0][0] = 1
            spec = ObjectiveSpec(dtype=dtype, reduction="sum" if i % 3 == 0 else
                "explicit" if i % 3 == 1 else "active_mean", denominator=7.,
                importance_level="sequence" if i % 2 else "token",
                gradient_scale=.25 if i % 5 == 0 else 1.)
            weights = [[2.]] if i % 2 else [[rng.choice((.5, 1., 2.)) for _ in ratios]]
            current = [[math.log(r) for r in ratios]]
            result = _execute(spec, current, [[0.]*6], advantages, mask, weights)
            observed = result.to_dict()["observed_coefficients"][0]
            add(f"{dtype}-normal-{i}", "EXPECTED_OBJECTIVE_EFFECT", result,
                len(set(observed)) > 1)
        # Explicit sequence-product semantics: derivative is coupled across active tokens.
        add(f"{dtype}-sequence-product", "EXPECTED_OBJECTIVE_EFFECT",
            _execute(ObjectiveSpec(dtype=dtype, ratio_level="sequence_product"),
                [[.05, -.05, 1.], [.3, .1, 1.]], [[0.]*3]*2,
                [[2.]*3, [-1.]*3], [[1, 1, 0], [1, 1, 0]]))
        add(f"{dtype}-all-masked-sum", "EXPECTED_OBJECTIVE_EFFECT",
            _execute(ObjectiveSpec(dtype=dtype, reduction="sum"), [[0., 0.]], [[0., 0.]],
                     [[1., -1.]], [[0, 0]]))
        numerical_cases = [
            ("masked-exp-overflow", [[1000., 0.]], [[0., 0.]], [[1., 1.]], [[0, 1]]),
            ("padding-inf-subtract", [[-math.inf, 0.]], [[-math.inf, 0.]], [[1., 1.]], [[0, 1]]),
            ("finite-product-overflow", [[math.log(1024.), 0.]], [[0., 0.]],
             [[-1e308 if dtype == 'float64' else -3e38 if dtype != 'float16' else -60000., 1.]], [[1, 1]]),
            ("exp-underflow", [[-1000., 0.]], [[0., 0.]], [[1., 1.]], [[1, 1]]),
            ("empty-mean", [[0., 0.]], [[0., 0.]], [[1., 1.]], [[0, 0]])]
        for name, x, o, a, m in numerical_cases:
            add(f"{dtype}-{name}", "NUMERICAL_ANOMALY_OBSERVED",
                _execute(ObjectiveSpec(dtype=dtype), x, o, a, m))
        add(f"{dtype}-unverified-manual-difference", "INSUFFICIENT_EVIDENCE",
            audit_objective(ObjectiveSpec(dtype=dtype), ObjectiveTrace([[0]], [[0]], [[1]], [[1]],
                observed_coefficients=[[99.]])))
        add(f"{dtype}-unsupported-grpo", "INSUFFICIENT_EVIDENCE",
            audit_objective(ObjectiveSpec(dtype=dtype, objective="grpo"),
                ObjectiveTrace([[0]], [[0]], [[1]], [[1]], observed_coefficients=[[-1.]])))
    normal = [c for c in cases if c["expected"] == "EXPECTED_OBJECTIVE_EFFECT"]
    numerical = [c for c in cases if c["expected"] == "NUMERICAL_ANOMALY_OBSERVED"]
    insufficient = [c for c in cases if c["expected"] == "INSUFFICIENT_EVIDENCE"]
    positive_verdicts = {"NUMERICAL_ANOMALY_OBSERVED", "SEMANTIC_MISMATCH_CONFIRMED"}
    false_positive = sum(c["actual"] in positive_verdicts for c in normal)
    false_negative = sum(c["actual"] not in positive_verdicts for c in numerical)
    return {"seed": seed, "cases": cases, "metrics": {
        "case_count": len(cases), "passed": sum(c["pass"] for c in cases),
        "normal_cases": len(normal), "numerical_cases": len(numerical),
        "insufficient_cases": len(insufficient), "false_positives": false_positive,
        "false_negatives": false_negative,
        "false_positive_rate": false_positive / len(normal),
        "false_negative_rate": false_negative / len(numerical),
        "normal_coefficient_spread_cases_explained": sum(c["coefficient_spread"] and c["pass"] for c in normal)}}


def run_objective_benchmark():
    import torch
    controls = _partition(104729, 8)
    holdout = _partition(130363, 32)
    names = ["Open-Instruct", "Prime RL", "Verifiers", "Axolotl", "ms-swift", "trlx",
             "DeepSpeedExamples", "Open-Reasoner-Zero", "torchtune", "CleanRL"]
    return {"schema_version": "quotientguard-objective-benchmark-v1",
        "outcome": "PASS" if all(c["pass"] for p in (controls, holdout) for c in p["cases"]) else "FAIL",
        "environment": {"torch": torch.__version__, "device": "cpu",
                        "dtypes": ["float16", "bfloat16", "float32", "float64"]},
        "controls": controls, "holdout": holdout,
        "historical_review": {"source": "owner-supplied historical summary; artifacts unavailable in public checkout",
            "historical_numerical_candidates": 7, "historical_integrated_confirmed_defects": 0,
            "used_as_ground_truth": False,
            "repositories": [{"repository": name, "execution": "NOT_EXECUTED",
                              "verdict": "INSUFFICIENT_EVIDENCE"} for name in names]},
        "limitations": ["Seeded local controls are not a prospective external framework benchmark.",
            "Holdout uses separate seeds and more draws from the same specified generator, not an independent distribution.",
            "Historical candidates were not relabeled, reconstructed, or executed.",
            "Zero observed error rates do not establish zero population error."]}
