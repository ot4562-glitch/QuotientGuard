"""Small dependency-free assertion helpers for CI and pytest/unittest."""

from __future__ import annotations

from typing import Any

from .compare import MEASURED_GAP_STATUSES


def assert_policy_passed(certificate: dict[str, Any]) -> None:
    status = certificate.get("summary", {}).get("policy_status")
    if status != "PASS":
        raise AssertionError(f"QuotientGuard policy status is {status!r}, expected 'PASS'")


def assert_no_measured_gap(certificate: dict[str, Any]) -> None:
    offenders = [
        row["class_id"]
        for row in certificate.get("classes", [])
        if row.get("compatibility_status") in MEASURED_GAP_STATUSES
    ]
    if offenders:
        raise AssertionError(
            "QuotientGuard measured compatibility gaps in classes: "
            + ", ".join(map(str, offenders))
        )


def assert_no_new_measured_gap(comparison: dict[str, Any]) -> None:
    offenders = [
        row["class_id"]
        for row in comparison.get("classes", [])
        if row.get("new_measured_gap")
    ]
    if offenders:
        raise AssertionError(
            "QuotientGuard regression introduced measured gaps in classes: "
            + ", ".join(map(str, offenders))
        )
