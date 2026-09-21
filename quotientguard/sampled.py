"""Direct fixed-snapshot audit for sampled trainer observations."""

from __future__ import annotations

import copy
import math
from collections import defaultdict
from typing import Any

from .core import DEFAULT_NUMERICAL_TOLERANCE, evaluate_policy
from .mathops import add, max_abs, norm, scale, sub, zeros
from .provenance import sha256_json
from .schema import Observation, normalize_observations
from .version import __version__


def _mean_vector(vectors: list[tuple[float, ...] | list[float]]) -> list[float]:
    if not vectors:
        raise ValueError("cannot average an empty vector list")
    dim = len(vectors[0])
    out = zeros(dim)
    for vector in vectors:
        if len(vector) != dim:
            raise ValueError("all score vectors must have the same dimension")
        out = add(out, vector)
    return scale(out, 1.0 / len(vectors))


def _jackknife_scalar_standard_error(delete_estimates: list[float]) -> float | None:
    cluster_count = len(delete_estimates)
    if cluster_count < 2:
        return None
    center = sum(delete_estimates) / cluster_count
    variance = (
        (cluster_count - 1)
        / cluster_count
        * sum((value - center) ** 2 for value in delete_estimates)
    )
    return math.sqrt(max(variance, 0.0))


def _jackknife_vector_standard_error(
    delete_estimates: list[list[float]],
) -> list[float] | None:
    cluster_count = len(delete_estimates)
    if cluster_count < 2:
        return None
    dim = len(delete_estimates[0])
    if any(len(vector) != dim for vector in delete_estimates):
        raise ValueError("jackknife vectors must share one dimension")
    centers = [
        sum(vector[j] for vector in delete_estimates) / cluster_count
        for j in range(dim)
    ]
    scale_factor = (cluster_count - 1) / cluster_count
    return [
        math.sqrt(
            max(
                scale_factor
                * sum(
                    (vector[j] - centers[j]) ** 2
                    for vector in delete_estimates
                ),
                0.0,
            )
        )
        for j in range(dim)
    ]


def _trajectory_coefficient_estimates(
    by_trajectory: dict[str, list[Observation]],
) -> list[dict[str, Any]]:
    estimates: list[dict[str, Any]] = []
    for trajectory_id in sorted(by_trajectory):
        rows = by_trajectory[trajectory_id]
        mean_value = sum(row.realized_coefficient for row in rows) / len(rows)
        batch_ids = sorted({row.batch_id for row in rows})
        delete_estimates: list[float] = []
        if len(batch_ids) >= 2:
            for batch_id in batch_ids:
                remaining = [row for row in rows if row.batch_id != batch_id]
                if not remaining:
                    delete_estimates = []
                    break
                delete_estimates.append(
                    sum(row.realized_coefficient for row in remaining)
                    / len(remaining)
                )

        standard_error = _jackknife_scalar_standard_error(delete_estimates)
        estimates.append(
            {
                "trajectory_id": trajectory_id,
                "mean": mean_value,
                "observation_count": len(rows),
                "batch_count": len(batch_ids),
                "standard_error": standard_error,
                "uncertainty_method": (
                    "leave_one_batch_out_jackknife"
                    if standard_error is not None
                    else "unavailable_insufficient_batches"
                ),
            }
        )
    return estimates


def _gap_components(rows: list[Observation]) -> dict[str, Any]:
    scores = [row.score for row in rows]
    if not scores or any(score is None for score in scores):
        raise ValueError("gap components require score vectors for every row")

    quotient_score = _mean_vector(
        [score for score in scores if score is not None]
    )
    gap = zeros(len(quotient_score))
    expected_update = zeros(len(quotient_score))

    for row in rows:
        assert row.score is not None
        fiber = sub(row.score, quotient_score)
        gap = add(gap, scale(fiber, row.realized_coefficient))
        expected_update = add(
            expected_update,
            scale(row.score, row.realized_coefficient),
        )

    gap = scale(gap, 1.0 / len(rows))
    expected_update = scale(expected_update, 1.0 / len(rows))
    quotient_update = scale(
        quotient_score,
        sum(row.realized_coefficient for row in rows) / len(rows),
    )
    residual = sub(expected_update, add(quotient_update, gap))
    return {
        "quotient_score": quotient_score,
        "gap": gap,
        "expected_update": expected_update,
        "quotient_update": quotient_update,
        "residual": residual,
    }


def _cluster_jackknife_gap_uncertainty(
    rows: list[Observation],
) -> dict[str, Any]:
    batch_ids = sorted({row.batch_id for row in rows})
    base = {
        "cluster_unit": "batch_id",
        "cluster_count": len(batch_ids),
    }
    if len(batch_ids) < 2:
        return {
            **base,
            "method": "unavailable_insufficient_batches",
            "standard_error_vector": None,
            "gap_norm_standard_error": None,
        }

    first_score = rows[0].score
    if first_score is None:
        raise ValueError("cluster gap uncertainty requires score vectors")
    dim = len(first_score)

    total_count = len(rows)
    total_h = 0.0
    total_s = zeros(dim)
    total_hs = zeros(dim)
    trajectory_counts: dict[str, int] = defaultdict(int)
    batch_stats: dict[str, dict[str, Any]] = {
        batch_id: {
            "count": 0,
            "sum_h": 0.0,
            "sum_s": zeros(dim),
            "sum_hs": zeros(dim),
            "trajectory_counts": defaultdict(int),
        }
        for batch_id in batch_ids
    }

    for row in rows:
        assert row.score is not None
        score = list(row.score)
        total_h += row.realized_coefficient
        total_s = add(total_s, score)
        total_hs = add(
            total_hs,
            scale(score, row.realized_coefficient),
        )
        trajectory_counts[row.trajectory_id] += 1

        stats = batch_stats[row.batch_id]
        stats["count"] += 1
        stats["sum_h"] += row.realized_coefficient
        stats["sum_s"] = add(stats["sum_s"], score)
        stats["sum_hs"] = add(
            stats["sum_hs"],
            scale(score, row.realized_coefficient),
        )
        stats["trajectory_counts"][row.trajectory_id] += 1

    delete_gaps: list[list[float]] = []
    delete_gap_norms: list[float] = []
    for batch_id in batch_ids:
        stats = batch_stats[batch_id]
        remaining_count = total_count - stats["count"]
        if remaining_count <= 0:
            return {
                **base,
                "method": "unavailable_insufficient_batches",
                "standard_error_vector": None,
                "gap_norm_standard_error": None,
            }

        remaining_support = sum(
            (
                trajectory_counts[trajectory_id]
                - stats["trajectory_counts"].get(trajectory_id, 0)
            )
            > 0
            for trajectory_id in trajectory_counts
        )
        if remaining_support < 2:
            return {
                **base,
                "method": "unavailable_insufficient_cluster_support",
                "standard_error_vector": None,
                "gap_norm_standard_error": None,
            }

        remaining_h = total_h - stats["sum_h"]
        remaining_s = sub(total_s, stats["sum_s"])
        remaining_hs = sub(total_hs, stats["sum_hs"])

        mean_h = remaining_h / remaining_count
        mean_s = scale(remaining_s, 1.0 / remaining_count)
        mean_hs = scale(remaining_hs, 1.0 / remaining_count)
        gap = sub(mean_hs, scale(mean_s, mean_h))

        delete_gaps.append(gap)
        delete_gap_norms.append(norm(gap))

    return {
        **base,
        "method": "leave_one_batch_out_jackknife",
        "standard_error_vector": _jackknife_vector_standard_error(delete_gaps),
        "gap_norm_standard_error": _jackknife_scalar_standard_error(
            delete_gap_norms
        ),
        "note": (
            "Cluster jackknife uses batch sufficient statistics to recompute "
            "the quotient cross-moment after deleting one batch; rows are not "
            "treated as iid."
        ),
    }


def _class_audit(
    rows: list[Observation],
    *,
    numerical_tolerance: float,
    policy: dict[str, Any] | None,
) -> dict[str, Any]:
    class_id = rows[0].class_id
    by_trajectory: dict[str, list[Observation]] = defaultdict(list)
    for row in rows:
        by_trajectory[row.trajectory_id].append(row)

    if len(by_trajectory) < 2:
        raise ValueError(
            f"class {class_id!r} has only one observed trajectory; at least two are required"
        )

    coefficient_estimates = _trajectory_coefficient_estimates(by_trajectory)
    trajectory_means = {
        row["trajectory_id"]: row["mean"] for row in coefficient_estimates
    }
    coefficient_spread = (
        max(trajectory_means.values()) - min(trajectory_means.values())
    )
    class_constant = coefficient_spread <= numerical_tolerance
    min_trajectory_count = min(len(group) for group in by_trajectory.values())
    batch_count = len({row.batch_id for row in rows})

    base: dict[str, Any] = {
        "class_id": class_id,
        "observation_count": len(rows),
        "trajectory_count": len(by_trajectory),
        "batch_count": batch_count,
        "coefficient_evidence": "sampled",
        "coefficient_mean": (
            sum(row.realized_coefficient for row in rows) / len(rows)
        ),
        "coefficient_spread": coefficient_spread,
        "class_constant_within_tolerance": class_constant,
        "minimum_observations_per_trajectory": min_trajectory_count,
        "trajectory_coefficient_estimates": coefficient_estimates,
        "numerical_tolerance": numerical_tolerance,
    }

    if rows[0].score is None:
        if coefficient_spread > numerical_tolerance:
            status = "COEFFICIENT_MISMATCH_OBSERVED"
        elif min_trajectory_count >= 2:
            status = "ESTIMATED_CLASS_CONSTANCY"
        else:
            status = "INSUFFICIENT_REPETITION"

        result = {
            **base,
            "evidence_level": "sampled_coefficient_only",
            "compatibility_status": status,
            "gap": None,
            "gap_norm": None,
            "gap_ratio": None,
            "decomposition_residual_max_abs": None,
            "uncertainty": {
                "method": "trajectory_leave_one_batch_out_jackknife",
                "cluster_unit": "batch_id",
                "cluster_count": batch_count,
                "trajectory_estimates": coefficient_estimates,
                "note": (
                    "No population compatibility certificate is inferred from "
                    "sampled coefficient equality."
                ),
            },
        }
        result["policy"] = evaluate_policy(result, policy)
        return result

    components = _gap_components(rows)
    quotient_score = components["quotient_score"]
    gap = components["gap"]
    expected_update = components["expected_update"]
    quotient_update = components["quotient_update"]
    residual = components["residual"]

    gap_norm = norm(gap)
    expected_update_norm = norm(expected_update)
    gap_ratio = gap_norm / max(expected_update_norm, numerical_tolerance)
    score_kind = rows[0].score_kind

    if score_kind == "sketch":
        status = (
            "MEASURED_PROJECTED_SAMPLE_GAP"
            if gap_norm > numerical_tolerance
            else "SAMPLED_PROJECTED_ZERO_NOT_CERTIFIED"
        )
    else:
        status = (
            "MEASURED_SAMPLE_GAP"
            if gap_norm > numerical_tolerance
            else "SAMPLED_ZERO_NOT_CERTIFIED"
        )

    result = {
        **base,
        "evidence_level": "sampled_direct_gap",
        "score_kind": score_kind,
        "compatibility_status": status,
        "score_dimension": len(quotient_score),
        "quotient_score": quotient_score,
        "gap": gap,
        "gap_norm": gap_norm,
        "gap_ratio": gap_ratio,
        "expected_update_norm": expected_update_norm,
        "quotient_update_norm": norm(quotient_update),
        "decomposition_residual_max_abs": max_abs(residual),
        "uncertainty": _cluster_jackknife_gap_uncertainty(rows),
    }
    result["policy"] = evaluate_policy(result, policy)
    return result


def audit_sampled_observations(
    observations: list[dict[str, Any]],
    *,
    run_id: str | None = None,
    estimator: dict[str, Any] | None = None,
    equivalence_contract: dict[str, Any] | None = None,
    score_kind: str = "sketch",
    policy: dict[str, Any] | None = None,
    snapshot_metadata: dict[str, Any] | None = None,
    numerical_tolerance: float = DEFAULT_NUMERICAL_TOLERANCE,
) -> dict[str, Any]:
    if score_kind not in {"full", "sketch"}:
        raise ValueError("score_kind must be 'full' or 'sketch'")
    if numerical_tolerance < 0:
        raise ValueError("numerical_tolerance must be non-negative")

    normalized = normalize_observations(
        observations,
        default_score_kind=score_kind,
    )
    grouped: dict[str, list[Observation]] = defaultdict(list)
    for row in normalized:
        grouped[row.class_id].append(row)

    class_results = [
        _class_audit(
            grouped[class_id],
            numerical_tolerance=numerical_tolerance,
            policy=policy,
        )
        for class_id in sorted(grouped)
    ]

    policy_statuses = [row["policy"]["status"] for row in class_results]
    if "FAIL" in policy_statuses:
        policy_status = "FAIL"
    elif policy_statuses and all(status == "PASS" for status in policy_statuses):
        policy_status = "PASS"
    else:
        policy_status = "OBSERVED"

    measured_gap_statuses = {
        "MEASURED_SAMPLE_GAP",
        "MEASURED_PROJECTED_SAMPLE_GAP",
    }
    uncertified_statuses = {
        "COEFFICIENT_MISMATCH_OBSERVED",
        "ESTIMATED_CLASS_CONSTANCY",
        "INSUFFICIENT_REPETITION",
        "SAMPLED_ZERO_NOT_CERTIFIED",
        "SAMPLED_PROJECTED_ZERO_NOT_CERTIFIED",
    }

    estimator_payload = copy.deepcopy(estimator or {})
    contract_payload = copy.deepcopy(equivalence_contract or {})
    source_rows = [row.to_mapping() for row in normalized]
    metadata_rows = [row.metadata for row in normalized if row.metadata is not None]
    state_consuming_points = sorted(
        {
            str(metadata["state_consuming_point"])
            for metadata in metadata_rows
            if "state_consuming_point" in metadata
        }
    )

    certificate: dict[str, Any] = {
        "schema_version": "quotientguard.certificate.v2",
        "audit_mode": "sampled_fixed_snapshot",
        "tool": {"name": "QuotientGuard", "version": __version__},
        "source_observations_sha256": sha256_json(source_rows),
        "policy_snapshot_id": normalized[0].policy_snapshot_id,
        "snapshot_metadata": copy.deepcopy(snapshot_metadata or {}),
        "observation_metadata_summary": {
            "metadata_observation_count": len(metadata_rows),
            "state_consuming_points": state_consuming_points,
        },
        "run_id": run_id,
        "estimator": estimator_payload,
        "estimator_fingerprint_sha256": sha256_json(estimator_payload),
        "equivalence_contract": contract_payload,
        "equivalence_contract_sha256": sha256_json(contract_payload),
        "score_space": {
            "kind": normalized[0].score_kind
            if normalized[0].score is not None
            else "none"
        },
        "summary": {
            "class_count": len(class_results),
            "measured_gap_count": sum(
                row["compatibility_status"] in measured_gap_statuses
                for row in class_results
            ),
            "measured_compatible_count": 0,
            "sufficiently_certified_count": 0,
            "not_certified_count": sum(
                row["compatibility_status"] in uncertified_statuses
                for row in class_results
            ),
            "policy_status": policy_status,
        },
        "classes": class_results,
    }
    certificate["certificate_sha256"] = sha256_json(certificate)
    return certificate
