"""Core estimator-homomorphism compatibility calculations.

The full-gap path implements the quotient/fiber decomposition described in
Sim (2026), DOI 10.5281/zenodo.22859044.

A coefficient-only class can be certified by the sufficient condition when
its representative-conditioned effective coefficient is class-constant.
Non-constancy alone is not reported as a measured non-zero gap.
"""

from __future__ import annotations

import copy
from typing import Any

from .mathops import add, ensure_finite_scalar, ensure_vector, max_abs, norm, scale, sub, weighted_mean
from .provenance import sha256_json
from .version import __version__

DEFAULT_NUMERICAL_TOLERANCE = 1e-12
PROBABILITY_TOLERANCE = 1e-9

def _normalize_probabilities(representatives: list[dict[str, Any]]) -> list[float]:
    probabilities = [
        ensure_finite_scalar(rep["conditional_probability"], "conditional_probability")
        for rep in representatives
    ]
    if any(p < 0.0 for p in probabilities):
        raise ValueError("conditional_probability must be non-negative")
    total = sum(probabilities)
    if abs(total - 1.0) > PROBABILITY_TOLERANCE:
        raise ValueError(
            f"conditional probabilities must sum to 1 (observed {total:.17g})"
        )
    return probabilities


def evaluate_policy(metrics: dict[str, Any], policy: dict[str, Any] | None) -> dict[str, Any]:
    if not policy:
        return {"status": "OBSERVED", "violations": []}

    checks: list[tuple[str, float, float]] = []
    if policy.get("max_gap_norm") is not None and metrics.get("gap_norm") is not None:
        checks.append(("max_gap_norm", metrics["gap_norm"], float(policy["max_gap_norm"])))
    if policy.get("max_gap_ratio") is not None and metrics.get("gap_ratio") is not None:
        checks.append(("max_gap_ratio", metrics["gap_ratio"], float(policy["max_gap_ratio"])))
    if policy.get("max_coefficient_spread") is not None:
        checks.append(
            (
                "max_coefficient_spread",
                metrics["coefficient_spread"],
                float(policy["max_coefficient_spread"]),
            )
        )

    if not checks:
        return {"status": "OBSERVED", "violations": []}

    violations = [
        {"rule": name, "observed": observed, "limit": limit}
        for name, observed, limit in checks
        if observed > limit
    ]
    return {"status": "FAIL" if violations else "PASS", "violations": violations}


def audit_class(
    class_record: dict[str, Any],
    *,
    numerical_tolerance: float = DEFAULT_NUMERICAL_TOLERANCE,
    policy: dict[str, Any] | None = None,
    score_kind: str = "full",
) -> dict[str, Any]:
    class_id = str(class_record.get("class_id", ""))
    if not class_id:
        raise ValueError("each class requires class_id")

    representatives = class_record.get("representatives")
    if not isinstance(representatives, list) or len(representatives) < 2:
        raise ValueError(f"class {class_id!r} requires at least two representatives")

    probabilities = _normalize_probabilities(representatives)
    coefficients = [
        ensure_finite_scalar(rep["effective_coefficient"], "effective_coefficient")
        for rep in representatives
    ]
    coefficient_spread = max(coefficients) - min(coefficients)
    class_constant = coefficient_spread <= numerical_tolerance
    coefficient_evidence = str(class_record.get("coefficient_evidence", "exact"))
    if coefficient_evidence not in {"exact", "sampled", "realized"}:
        raise ValueError(
            f"class {class_id!r} coefficient_evidence must be exact, sampled, or realized"
        )

    score_presence = ["score" in rep and rep["score"] is not None for rep in representatives]
    if any(score_presence) and not all(score_presence):
        raise ValueError(
            f"class {class_id!r} mixes representatives with and without score vectors"
        )
    if coefficient_evidence != "exact" and any(score_presence):
        raise ValueError(
            "sampled score observations must use audit_sampled_observations; "
            "separately averaged coefficients and scores do not preserve E[H*s]"
        )

    base = {
        "class_id": class_id,
        "representative_count": len(representatives),
        "coefficient_mean": sum(p * h for p, h in zip(probabilities, coefficients)),
        "coefficient_spread": coefficient_spread,
        "class_constant_within_tolerance": class_constant,
        "coefficient_evidence": coefficient_evidence,
        "numerical_tolerance": numerical_tolerance,
    }

    if not any(score_presence):
        if coefficient_evidence == "exact":
            status = (
                "CERTIFIED_BY_EXACT_CLASS_CONSTANCY"
                if class_constant
                else "NOT_CERTIFIED"
            )
        else:
            status = (
                "ESTIMATED_CLASS_CONSTANCY"
                if class_constant
                else "COEFFICIENT_MISMATCH_OBSERVED"
            )
        result = {
            **base,
            "evidence_level": f"coefficient_only_{coefficient_evidence}",
            "compatibility_status": status,
            "gap": None,
            "gap_norm": None,
            "gap_ratio": None,
            "decomposition_residual_max_abs": None,
        }
        result["policy"] = evaluate_policy(result, policy)
        return result

    scores = [ensure_vector(rep["score"], "score") for rep in representatives]
    dim = len(scores[0])
    if any(len(score) != dim for score in scores):
        raise ValueError(f"class {class_id!r} has inconsistent score dimensions")

    quotient_score = weighted_mean(scores, probabilities)
    fiber_scores = [sub(score, quotient_score) for score in scores]
    h_bar = base["coefficient_mean"]

    gap = [0.0] * dim
    expected_update = [0.0] * dim
    for p, h, score, fiber in zip(probabilities, coefficients, scores, fiber_scores):
        gap = add(gap, scale(fiber, p * h))
        expected_update = add(expected_update, scale(score, p * h))

    quotient_update = scale(quotient_score, h_bar)
    reconstructed = add(quotient_update, gap)
    residual = sub(expected_update, reconstructed)

    gap_norm = norm(gap)
    expected_update_norm = norm(expected_update)
    quotient_update_norm = norm(quotient_update)
    denominator = max(expected_update_norm, numerical_tolerance)
    gap_ratio = gap_norm / denominator

    if class_constant:
        compatibility_status = "CERTIFIED_BY_EXACT_CLASS_CONSTANCY"
    elif gap_norm > numerical_tolerance and score_kind == "sketch":
        compatibility_status = "MEASURED_PROJECTED_GAP"
    elif gap_norm <= numerical_tolerance and score_kind == "sketch":
        compatibility_status = "PROJECTED_ZERO_NOT_CERTIFIED"
    elif gap_norm > numerical_tolerance:
        compatibility_status = "MEASURED_GAP"
    else:
        compatibility_status = "MEASURED_COMPATIBLE"

    result = {
        **base,
        "evidence_level": "exact_gap",
        "score_kind": score_kind,
        "compatibility_status": compatibility_status,
        "score_dimension": dim,
        "quotient_score": quotient_score,
        "gap": gap,
        "gap_norm": gap_norm,
        "gap_ratio": gap_ratio,
        "expected_update_norm": expected_update_norm,
        "quotient_update_norm": quotient_update_norm,
        "decomposition_residual_max_abs": max_abs(residual),
    }
    result["policy"] = evaluate_policy(result, policy)
    return result


def audit_receipt(receipt: dict[str, Any]) -> dict[str, Any]:
    if receipt.get("schema_version") != "quotientguard.receipt.v2":
        raise ValueError("schema_version must be 'quotientguard.receipt.v2'")

    policy_snapshot_id = str(receipt.get("policy_snapshot_id", "")).strip()
    if not policy_snapshot_id:
        raise ValueError("receipt requires policy_snapshot_id")

    classes = receipt.get("classes")
    if not isinstance(classes, list) or not classes:
        raise ValueError("receipt requires a non-empty classes list")

    numerical_tolerance = float(
        receipt.get("numerical_tolerance", DEFAULT_NUMERICAL_TOLERANCE)
    )
    if numerical_tolerance < 0:
        raise ValueError("numerical_tolerance must be non-negative")

    policy = receipt.get("policy")
    if "score_space" in receipt:
        score_space = copy.deepcopy(receipt["score_space"])
    else:
        score_present = [
            rep.get("score") is not None
            for class_record in classes
            for rep in class_record.get("representatives", [])
        ]
        score_space = {"kind": "full" if any(score_present) else "none"}
    score_kind = score_space.get("kind", "none")
    if score_kind not in {"none", "full", "sketch"}:
        raise ValueError("score_space.kind must be 'none', 'full', or 'sketch'")

    score_presence = [
        rep.get("score") is not None
        for class_record in classes
        for rep in class_record.get("representatives", [])
    ]
    if score_kind == "none" and any(score_presence):
        raise ValueError("score_space.kind='none' cannot contain score vectors")
    if score_kind in {"full", "sketch"} and score_presence and not all(score_presence):
        raise ValueError("receipt mixes scored and unscored representatives")
    if score_kind in {"full", "sketch"} and score_presence and not any(score_presence):
        raise ValueError("declared score space requires score vectors")

    class_results = [
        audit_class(
            class_record,
            numerical_tolerance=numerical_tolerance,
            policy=policy,
            score_kind=score_kind,
        )
        for class_record in classes
    ]

    policy_statuses = [row["policy"]["status"] for row in class_results]
    if "FAIL" in policy_statuses:
        overall_policy_status = "FAIL"
    elif policy_statuses and all(status == "PASS" for status in policy_statuses):
        overall_policy_status = "PASS"
    else:
        overall_policy_status = "OBSERVED"

    measured_gap_statuses = {
        "MEASURED_GAP",
        "MEASURED_PROJECTED_GAP",
        "MEASURED_SAMPLE_GAP",
        "MEASURED_PROJECTED_SAMPLE_GAP",
    }
    measured_gaps = sum(
        row["compatibility_status"] in measured_gap_statuses
        for row in class_results
    )
    uncertified_statuses = {
        "NOT_CERTIFIED",
        "PROJECTED_ZERO_NOT_CERTIFIED",
        "SAMPLED_PROJECTED_ZERO_NOT_CERTIFIED",
        "SAMPLED_ZERO_NOT_CERTIFIED",
        "ESTIMATED_CLASS_CONSTANCY",
        "COEFFICIENT_MISMATCH_OBSERVED",
        "INSUFFICIENT_REPETITION",
    }
    uncertified = sum(
        row["compatibility_status"] in uncertified_statuses
        for row in class_results
    )
    sufficient = sum(
        row["compatibility_status"] == "CERTIFIED_BY_EXACT_CLASS_CONSTANCY"
        for row in class_results
    )
    measured_compatible = sum(
        row["compatibility_status"] == "MEASURED_COMPATIBLE"
        for row in class_results
    )

    input_digest = sha256_json(receipt)
    estimator = copy.deepcopy(receipt.get("estimator", {}))
    equivalence_contract = copy.deepcopy(receipt.get("equivalence_contract", {}))
    certificate_core = {
        "schema_version": "quotientguard.certificate.v2",
        "tool": {"name": "QuotientGuard", "version": __version__},
        "source_receipt_sha256": input_digest,
        "policy_snapshot_id": policy_snapshot_id,
        "snapshot_metadata": copy.deepcopy(receipt.get("snapshot_metadata", {})),
        "run_id": receipt.get("run_id"),
        "estimator": estimator,
        "estimator_fingerprint_sha256": sha256_json(estimator),
        "equivalence_contract": equivalence_contract,
        "equivalence_contract_sha256": sha256_json(equivalence_contract),
        "score_space": score_space,
        "summary": {
            "class_count": len(class_results),
            "measured_gap_count": measured_gaps,
            "measured_compatible_count": measured_compatible,
            "sufficiently_certified_count": sufficient,
            "not_certified_count": uncertified,
            "policy_status": overall_policy_status,
        },
        "classes": class_results,
    }
    certificate_core["certificate_sha256"] = sha256_json(certificate_core)
    return certificate_core
