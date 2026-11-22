"""Present independent training-boundary reports without asserting linkage."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .native_validation import _seal


def summarize_training_audits(
    *, objective: Mapping[str, Any] | None = None,
    gradient: Mapping[str, Any] | None = None,
    window: Mapping[str, Any] | None = None,
    optimizer: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize serialized reports in training order, preserving local verdicts.

    Earlier report formats do not bind their observations to the same step as
    an optimizer trace. A local match does not establish evidence continuity.
    """
    entries = (
        ("objective", "quotientguard-full-objective-v1", objective),
        ("gradient", "quotientguard-parameter-validation-v1", gradient),
        ("accumulation_reduction", "quotientguard-objective-window-v1", window),
        ("optimizer", "quotientguard-adamw-audit-v1", optimizer),
    )
    reports = []
    for boundary, schema, report in entries:
        if report is None:
            reports.append({"boundary": boundary, "status": "NOT_PROVIDED"})
            continue
        if not isinstance(report, Mapping) or report.get("schema_version") != schema:
            raise ValueError(f"{boundary} requires a {schema} report")
        if not isinstance(report.get("status"), str) or not report["status"]:
            raise ValueError(f"{boundary} report requires a status")
        entry = {"boundary": boundary, "status": report["status"], "schema_version": schema}
        for field in ("binding", "reason", "reasons", "mismatches", "report_sha256"):
            if field in report:
                entry[field] = report[field]
        if boundary == "optimizer":
            for field, source in (("state_lineage", "lineage"), ("state_transition", "transition")):
                verdict = report.get(source)
                if not isinstance(verdict, Mapping) or not isinstance(verdict.get("status"), str):
                    raise ValueError(f"optimizer report requires a {source} verdict")
                entry[field] = verdict["status"]
        reports.append(entry)
    return _seal({
        "schema_version": "quotientguard-training-summary-v1",
        "status": "BOUNDARY_REPORTS_ONLY",
        "pipeline_verified": False,
        "reason": "cross_boundary_evidence_binding_not_established",
        "boundaries": reports,
    })
