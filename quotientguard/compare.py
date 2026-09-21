"""Deterministic comparison of QuotientGuard certificates."""

from __future__ import annotations

from typing import Any

from .provenance import sha256_json
from .version import __version__


MEASURED_GAP_STATUSES = {
    "MEASURED_GAP",
    "MEASURED_PROJECTED_GAP",
    "MEASURED_SAMPLE_GAP",
    "MEASURED_PROJECTED_SAMPLE_GAP",
}

EVIDENCE_RANK = {
    "coefficient_only_realized": 0,
    "sampled_coefficient_only": 1,
    "coefficient_only_sampled": 1,
    "sampled_direct_gap": 2,
    "coefficient_only_exact": 3,
    "exact_gap": 4,
}


def _classes(cert: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["class_id"]): row for row in cert.get("classes", [])}


def compare_certificates(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    fail_on_new_gap: bool = False,
    max_gap_ratio_increase: float | None = None,
    fail_on_evidence_downgrade: bool = False,
) -> dict[str, Any]:
    for label, cert in (("baseline", baseline), ("candidate", candidate)):
        if cert.get("schema_version") != "quotientguard.certificate.v2":
            raise ValueError(f"{label} is not a QuotientGuard v2 certificate")

    if (
        baseline.get("equivalence_contract_sha256")
        != candidate.get("equivalence_contract_sha256")
    ):
        raise ValueError("certificates use different equivalence contracts")

    base_score = baseline.get("score_space", {}).get("kind")
    cand_score = candidate.get("score_space", {}).get("kind")
    if base_score != cand_score:
        raise ValueError("certificates use different score spaces")

    base_rows = _classes(baseline)
    cand_rows = _classes(candidate)
    class_ids = sorted(set(base_rows) | set(cand_rows))
    rows = []
    new_gap_classes = []
    removed_gap_classes = []
    evidence_downgrades = []

    for class_id in class_ids:
        before = base_rows.get(class_id)
        after = cand_rows.get(class_id)
        before_status = before.get("compatibility_status") if before else None
        after_status = after.get("compatibility_status") if after else None

        before_gap = before_status in MEASURED_GAP_STATUSES
        after_gap = after_status in MEASURED_GAP_STATUSES
        if after_gap and not before_gap:
            new_gap_classes.append(class_id)
        if before_gap and not after_gap:
            removed_gap_classes.append(class_id)

        before_evidence = before.get("evidence_level") if before else None
        after_evidence = after.get("evidence_level") if after else None
        downgrade = False
        if before_evidence is not None and after_evidence is not None:
            downgrade = EVIDENCE_RANK.get(after_evidence, -1) < EVIDENCE_RANK.get(
                before_evidence,
                -1,
            )
        if downgrade:
            evidence_downgrades.append(class_id)

        before_ratio = before.get("gap_ratio") if before else None
        after_ratio = after.get("gap_ratio") if after else None
        ratio_delta = (
            after_ratio - before_ratio
            if before_ratio is not None and after_ratio is not None
            else None
        )

        before_spread = before.get("coefficient_spread") if before else None
        after_spread = after.get("coefficient_spread") if after else None
        spread_delta = (
            after_spread - before_spread
            if before_spread is not None and after_spread is not None
            else None
        )

        rows.append(
            {
                "class_id": class_id,
                "baseline_status": before_status,
                "candidate_status": after_status,
                "new_measured_gap": after_gap and not before_gap,
                "removed_measured_gap": before_gap and not after_gap,
                "baseline_evidence": before_evidence,
                "candidate_evidence": after_evidence,
                "evidence_downgrade": downgrade,
                "gap_ratio_delta": ratio_delta,
                "coefficient_spread_delta": spread_delta,
            }
        )

    violations = []
    if fail_on_new_gap and new_gap_classes:
        violations.append(
            {
                "rule": "fail_on_new_gap",
                "classes": new_gap_classes,
            }
        )

    if max_gap_ratio_increase is not None:
        exceeded = [
            row["class_id"]
            for row in rows
            if row["gap_ratio_delta"] is not None
            and row["gap_ratio_delta"] > max_gap_ratio_increase
        ]
        if exceeded:
            violations.append(
                {
                    "rule": "max_gap_ratio_increase",
                    "limit": max_gap_ratio_increase,
                    "classes": exceeded,
                }
            )

    if fail_on_evidence_downgrade and evidence_downgrades:
        violations.append(
            {
                "rule": "fail_on_evidence_downgrade",
                "classes": evidence_downgrades,
            }
        )

    result: dict[str, Any] = {
        "schema_version": "quotientguard.comparison.v1",
        "tool": {"name": "QuotientGuard", "version": __version__},
        "baseline_certificate_sha256": baseline.get("certificate_sha256"),
        "candidate_certificate_sha256": candidate.get("certificate_sha256"),
        "equivalence_contract_sha256": candidate.get("equivalence_contract_sha256"),
        "score_space": candidate.get("score_space", {}),
        "estimator_changed": (
            baseline.get("estimator_fingerprint_sha256")
            != candidate.get("estimator_fingerprint_sha256")
        ),
        "summary": {
            "class_count": len(rows),
            "new_measured_gap_count": len(new_gap_classes),
            "removed_measured_gap_count": len(removed_gap_classes),
            "evidence_downgrade_count": len(evidence_downgrades),
            "policy_status": "FAIL" if violations else (
                "PASS"
                if any(
                    (
                        fail_on_new_gap,
                        max_gap_ratio_increase is not None,
                        fail_on_evidence_downgrade,
                    )
                )
                else "OBSERVED"
            ),
        },
        "policy": {
            "fail_on_new_gap": fail_on_new_gap,
            "max_gap_ratio_increase": max_gap_ratio_increase,
            "fail_on_evidence_downgrade": fail_on_evidence_downgrade,
            "violations": violations,
        },
        "classes": rows,
    }
    result["comparison_sha256"] = sha256_json(result)
    return result
