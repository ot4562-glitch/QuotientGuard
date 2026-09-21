"""Aggregate multiple QuotientGuard certificates into a persistence summary."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from .provenance import sha256_json
from .version import __version__

MEASURED_GAP_STATUSES = {
    "MEASURED_GAP",
    "MEASURED_PROJECTED_GAP",
    "MEASURED_SAMPLE_GAP",
    "MEASURED_PROJECTED_SAMPLE_GAP",
}
UNCERTIFIED_STATUSES = {
    "NOT_CERTIFIED",
    "PROJECTED_ZERO_NOT_CERTIFIED",
    "SAMPLED_PROJECTED_ZERO_NOT_CERTIFIED",
    "SAMPLED_ZERO_NOT_CERTIFIED",
    "ESTIMATED_CLASS_CONSTANCY",
    "COEFFICIENT_MISMATCH_OBSERVED",
    "INSUFFICIENT_REPETITION",
}
CERTIFIED_STATUSES = {
    "CERTIFIED_BY_EXACT_CLASS_CONSTANCY",
    "MEASURED_COMPATIBLE",
}


def _max_run(statuses: list[str], target: set[str]) -> int:
    best = 0
    current = 0
    for status in statuses:
        if status in target:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def summarize_certificates(
    certificates: list[dict[str, Any]],
    *,
    max_measured_gap_events: int | None = None,
    max_uncertified_events: int | None = None,
    max_consecutive_gap_events: int | None = None,
    max_gap_ratio: float | None = None,
    require_same_estimator: bool = False,
) -> dict[str, Any]:
    if not certificates:
        raise ValueError("at least one certificate is required")
    if max_gap_ratio is not None and max_gap_ratio < 0:
        raise ValueError("max_gap_ratio must be non-negative")

    history: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_hashes = []
    snapshot_ids = []
    estimator_fingerprints = []
    snapshot_metadata = []

    for index, cert in enumerate(certificates):
        if cert.get("schema_version") != "quotientguard.certificate.v2":
            raise ValueError(f"input {index} is not a QuotientGuard v2 certificate")

        snapshot_id = str(cert.get("policy_snapshot_id", "")).strip()
        if not snapshot_id:
            raise ValueError(f"input {index} is missing policy_snapshot_id")
        if snapshot_id in snapshot_ids:
            raise ValueError(f"duplicate policy_snapshot_id: {snapshot_id}")
        snapshot_ids.append(snapshot_id)

        estimator_fingerprint = cert.get("estimator_fingerprint_sha256")
        estimator_fingerprints.append(estimator_fingerprint)
        metadata = dict(cert.get("snapshot_metadata") or {})
        snapshot_metadata.append(
            {
                "index": index,
                "policy_snapshot_id": snapshot_id,
                "run_id": cert.get("run_id"),
                "captured_at": metadata.get("captured_at"),
                "estimator_fingerprint_sha256": estimator_fingerprint,
                "certificate_sha256": cert.get("certificate_sha256"),
            }
        )

        source_hashes.append(cert.get("certificate_sha256"))
        for row in cert.get("classes", []):
            history[str(row["class_id"])].append(
                {
                    "index": index,
                    "policy_snapshot_id": snapshot_id,
                    "run_id": cert.get("run_id"),
                    "captured_at": metadata.get("captured_at"),
                    "estimator_fingerprint_sha256": estimator_fingerprint,
                    "status": row["compatibility_status"],
                    "gap_norm": row.get("gap_norm"),
                    "gap_ratio": row.get("gap_ratio"),
                    "coefficient_spread": row.get("coefficient_spread"),
                }
            )

    distinct_estimator_fingerprints = {
        value for value in estimator_fingerprints if value is not None
    }
    global_violations = []
    if require_same_estimator and len(distinct_estimator_fingerprints) > 1:
        global_violations.append(
            {
                "rule": "require_same_estimator",
                "observed_fingerprint_count": len(distinct_estimator_fingerprints),
            }
        )

    class_rows = []
    any_class_alert = False
    policy_present = any(
        value is not None
        for value in (
            max_measured_gap_events,
            max_uncertified_events,
            max_consecutive_gap_events,
            max_gap_ratio,
        )
    ) or require_same_estimator

    for class_id in sorted(history):
        events = history[class_id]
        statuses = [event["status"] for event in events]
        measured_gap_events = sum(
            status in MEASURED_GAP_STATUSES for status in statuses
        )
        uncertified_events = sum(
            status in UNCERTIFIED_STATUSES for status in statuses
        )
        certified_events = sum(status in CERTIFIED_STATUSES for status in statuses)
        max_consecutive_gap = _max_run(statuses, MEASURED_GAP_STATUSES)

        gap_norms = [
            event["gap_norm"] for event in events if event["gap_norm"] is not None
        ]
        gap_ratios = [
            event["gap_ratio"] for event in events if event["gap_ratio"] is not None
        ]
        spreads = [
            event["coefficient_spread"]
            for event in events
            if event["coefficient_spread"] is not None
        ]

        violations = []
        if (
            max_measured_gap_events is not None
            and measured_gap_events > max_measured_gap_events
        ):
            violations.append(
                {
                    "rule": "max_measured_gap_events",
                    "observed": measured_gap_events,
                    "limit": max_measured_gap_events,
                }
            )
        if max_uncertified_events is not None and uncertified_events > max_uncertified_events:
            violations.append(
                {
                    "rule": "max_uncertified_events",
                    "observed": uncertified_events,
                    "limit": max_uncertified_events,
                }
            )
        if (
            max_consecutive_gap_events is not None
            and max_consecutive_gap > max_consecutive_gap_events
        ):
            violations.append(
                {
                    "rule": "max_consecutive_gap_events",
                    "observed": max_consecutive_gap,
                    "limit": max_consecutive_gap_events,
                }
            )
        if max_gap_ratio is not None and gap_ratios and max(gap_ratios) > max_gap_ratio:
            violations.append(
                {
                    "rule": "max_gap_ratio",
                    "observed": max(gap_ratios),
                    "limit": max_gap_ratio,
                }
            )

        if violations:
            outcome = "ALERT"
            any_class_alert = True
        elif policy_present:
            outcome = "PASS"
        else:
            outcome = "OBSERVED"

        class_rows.append(
            {
                "class_id": class_id,
                "event_count": len(events),
                "first_policy_snapshot_id": events[0]["policy_snapshot_id"],
                "last_policy_snapshot_id": events[-1]["policy_snapshot_id"],
                "measured_gap_events": measured_gap_events,
                "uncertified_events": uncertified_events,
                "certified_events": certified_events,
                "max_consecutive_gap_events": max_consecutive_gap,
                "max_gap_norm": max(gap_norms) if gap_norms else None,
                "max_gap_ratio": max(gap_ratios) if gap_ratios else None,
                "max_coefficient_spread": max(spreads) if spreads else None,
                "outcome": outcome,
                "violations": violations,
                "events": events,
            }
        )

    any_alert = any_class_alert or bool(global_violations)
    overall = "ALERT" if any_alert else ("PASS" if policy_present else "OBSERVED")
    result = {
        "schema_version": "quotientguard.sentinel.v3",
        "tool": {"name": "QuotientGuard", "version": __version__},
        "source_certificate_hashes": source_hashes,
        "policy_snapshot_ids": snapshot_ids,
        "snapshot_metadata": snapshot_metadata,
        "estimator_fingerprints": estimator_fingerprints,
        "policy": {
            "max_measured_gap_events": max_measured_gap_events,
            "max_uncertified_events": max_uncertified_events,
            "max_consecutive_gap_events": max_consecutive_gap_events,
            "max_gap_ratio": max_gap_ratio,
            "require_same_estimator": require_same_estimator,
            "global_violations": global_violations,
        },
        "summary": {
            "certificate_count": len(certificates),
            "class_count": len(class_rows),
            "first_policy_snapshot_id": snapshot_ids[0],
            "last_policy_snapshot_id": snapshot_ids[-1],
            "estimator_fingerprint_count": len(distinct_estimator_fingerprints),
            "outcome": overall,
            "alert_class_count": sum(row["outcome"] == "ALERT" for row in class_rows),
            "global_violation_count": len(global_violations),
        },
        "classes": class_rows,
    }
    result["sentinel_sha256"] = sha256_json(result)
    return result


def sentinel_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# QuotientGuard Sentinel Report",
        "",
        f"- Sentinel: {result['sentinel_sha256']}",
        f"- Certificates: {result['summary']['certificate_count']}",
        f"- Classes: {result['summary']['class_count']}",
        f"- First snapshot: {result['summary']['first_policy_snapshot_id']}",
        f"- Last snapshot: {result['summary']['last_policy_snapshot_id']}",
        f"- Estimator fingerprints: {result['summary']['estimator_fingerprint_count']}",
        f"- Outcome: **{result['summary']['outcome']}**",
        "",
        "| class | events | measured gaps | uncertified | certified | max consecutive gap | max gap ratio | outcome |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in result["classes"]:
        ratio = "n/a" if row["max_gap_ratio"] is None else f"{row['max_gap_ratio']:.6g}"
        lines.append(
            "| {class_id} | {events} | {gaps} | {uncert} | {cert} | {run} | {ratio} | {outcome} |".format(
                class_id=row["class_id"],
                events=row["event_count"],
                gaps=row["measured_gap_events"],
                uncert=row["uncertified_events"],
                cert=row["certified_events"],
                run=row["max_consecutive_gap_events"],
                ratio=ratio,
                outcome=row["outcome"],
            )
        )
    if result["policy"]["global_violations"]:
        lines += [
            "",
            "## Global policy violations",
            "",
        ]
        for violation in result["policy"]["global_violations"]:
            lines.append(f"- {violation['rule']}: {violation}")
    lines += [
        "",
        "Sentinel summarizes separate fixed-policy-snapshot certificates. "
        "It never pools raw observations across optimizer steps and does not "
        "assume monotonic gap growth.",
        "",
    ]
    return "\n".join(lines)
