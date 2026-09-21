"""Deterministic trainer/estimator certification-suite runner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .check import check_path
from .provenance import sha256_json
from .version import __version__


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def builtin_certification_manifest() -> Path:
    return (
        Path(__file__).resolve().parent
        / "data"
        / "fixtures"
        / "certification.json"
    )


def run_builtin_certification_suite() -> dict[str, Any]:
    return run_certification_suite(builtin_certification_manifest())


def run_certification_suite(manifest_path: str | Path) -> dict[str, Any]:
    path = Path(manifest_path)
    manifest = _load_json(path)
    if manifest.get("schema_version") != "quotientguard.certification-suite.v1":
        raise ValueError("unsupported certification suite schema")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("certification suite requires non-empty cases")

    results: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(f"case {index} must be an object")
        case_id = str(case.get("id", "")).strip()
        relative_input = str(case.get("input", "")).strip()
        if not case_id or not relative_input:
            raise ValueError(f"case {index} requires id and input")
        input_path = (path.parent / relative_input).resolve()
        certificate = check_path(input_path)

        expected = case.get("expected", {})
        if not isinstance(expected, dict):
            raise ValueError(f"case {case_id} expected must be an object")
        violations: list[dict[str, Any]] = []

        expected_policy = expected.get("policy_status")
        if expected_policy is not None:
            observed_policy = certificate["summary"]["policy_status"]
            if observed_policy != expected_policy:
                violations.append(
                    {
                        "field": "summary.policy_status",
                        "expected": expected_policy,
                        "observed": observed_policy,
                    }
                )

        expected_statuses = expected.get("class_statuses", {})
        if not isinstance(expected_statuses, dict):
            raise ValueError(f"case {case_id} class_statuses must be an object")
        observed_by_class = {
            str(row["class_id"]): row["compatibility_status"]
            for row in certificate.get("classes", [])
        }
        for class_id, expected_status in expected_statuses.items():
            observed_status = observed_by_class.get(str(class_id))
            if observed_status != expected_status:
                violations.append(
                    {
                        "field": f"classes[{class_id}].compatibility_status",
                        "expected": expected_status,
                        "observed": observed_status,
                    }
                )

        expected_gap_count = expected.get("measured_gap_count")
        if expected_gap_count is not None:
            observed_gap_count = certificate["summary"]["measured_gap_count"]
            if observed_gap_count != expected_gap_count:
                violations.append(
                    {
                        "field": "summary.measured_gap_count",
                        "expected": expected_gap_count,
                        "observed": observed_gap_count,
                    }
                )

        results.append(
            {
                "id": case_id,
                "input": relative_input,
                "certificate_sha256": certificate["certificate_sha256"],
                "outcome": "PASS" if not violations else "FAIL",
                "violations": violations,
            }
        )

    outcome = "PASS" if all(row["outcome"] == "PASS" for row in results) else "FAIL"
    result: dict[str, Any] = {
        "schema_version": "quotientguard.certification-result.v1",
        "tool": {"name": "QuotientGuard", "version": __version__},
        "suite_name": manifest.get("name"),
        "suite_version": manifest.get("version"),
        "manifest_sha256": sha256_json(manifest),
        "summary": {
            "case_count": len(results),
            "pass_count": sum(row["outcome"] == "PASS" for row in results),
            "fail_count": sum(row["outcome"] == "FAIL" for row in results),
            "outcome": outcome,
        },
        "cases": results,
    }
    result["certification_sha256"] = sha256_json(result)
    return result


def certification_text(result: dict[str, Any]) -> str:
    summary = result["summary"]
    lines = [
        f"QuotientGuard certification: {result.get('suite_name') or 'unnamed'}",
        (
            f"outcome: {summary['outcome']} | "
            f"cases: {summary['pass_count']}/{summary['case_count']} passed"
        ),
    ]
    for row in result["cases"]:
        lines.append(f"- {row['id']}: {row['outcome']}")
        for violation in row["violations"]:
            lines.append(
                "  {field}: expected={expected!r} observed={observed!r}".format(
                    **violation
                )
            )
    lines.append(f"suite certificate: {result['certification_sha256']}")
    return "\n".join(lines)
