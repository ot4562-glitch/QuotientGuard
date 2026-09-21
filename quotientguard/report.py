"""Human-readable Markdown report generation."""

from __future__ import annotations

from typing import Any


def markdown_report(certificate: dict[str, Any]) -> str:
    summary = certificate["summary"]
    estimator = certificate.get("estimator", {})
    lines = [
        "# QuotientGuard Audit Report",
        "",
        f"- Certificate: {certificate['certificate_sha256']}",
        f"- Source: {certificate.get('source_receipt_sha256') or certificate.get('source_observations_sha256')}",
        f"- Tool version: {certificate['tool']['version']}",
        f"- Policy snapshot: {certificate.get('policy_snapshot_id') or 'n/a'}",
        f"- Run ID: {certificate.get('run_id') or 'n/a'}",
        f"- Estimator: {estimator.get('name', 'unspecified')}",
        f"- Policy status: **{summary['policy_status']}**",
        "",
        "## Summary",
        "",
        f"- Classes audited: {summary['class_count']}",
        f"- Measured non-zero gaps: {summary['measured_gap_count']}",
        f"- Measured compatible classes: {summary['measured_compatible_count']}",
        f"- Certified by class-constant coefficient: {summary['sufficiently_certified_count']}",
        f"- Coefficient-only classes not certified: {summary['not_certified_count']}",
        "",
        "## Classes",
        "",
        "| class | evidence | compatibility | coeff spread | gap norm | gap ratio | policy |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for row in certificate["classes"]:
        gap_norm = "n/a" if row["gap_norm"] is None else f"{row['gap_norm']:.6g}"
        gap_ratio = "n/a" if row["gap_ratio"] is None else f"{row['gap_ratio']:.6g}"
        lines.append(
            "| {class_id} | {evidence} | {compat} | {spread:.6g} | {gap_norm} | {gap_ratio} | {policy} |".format(
                class_id=row["class_id"],
                evidence=row["evidence_level"],
                compat=row["compatibility_status"],
                spread=row["coefficient_spread"],
                gap_norm=gap_norm,
                gap_ratio=gap_ratio,
                policy=row["policy"]["status"],
            )
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "- CERTIFIED_BY_EXACT_CLASS_CONSTANCY: exact/analytic representative-conditioned effective coefficients are class-constant, a sufficient condition for zero compatibility gap.",
        "- ESTIMATED_CLASS_CONSTANCY: sampled coefficient estimates are equal within tolerance; this is not an exact theorem certificate.",
        "- COEFFICIENT_MISMATCH_OBSERVED: sampled/realized coefficients differ inside one declared equivalence class; escalate to repeated or score-based measurement.",
        "- INSUFFICIENT_REPETITION: coefficient-only observations do not repeat each trajectory enough to estimate conditional coefficients.",
        "- NOT_CERTIFIED: exact coefficient constancy failed, but no score vectors were supplied; this is not by itself a measured non-zero gap.",
        "- MEASURED_COMPATIBLE: a full exact supplied score model produced a gap within the declared numerical tolerance.",
        "- MEASURED_GAP: a non-zero quotient/fiber compatibility gap was directly measured from a full exact supplied score model.",
        "- MEASURED_PROJECTED_GAP: a non-zero gap was measured in a declared exact score projection/sketch.",
        "- PROJECTED_ZERO_NOT_CERTIFIED: the selected exact sketch saw zero gap, which does not certify the unobserved score space.",
        "- MEASURED_SAMPLE_GAP / MEASURED_PROJECTED_SAMPLE_GAP: a non-zero fixed-snapshot sample gap was observed; population uncertainty remains separate.",
        "- SAMPLED_ZERO_NOT_CERTIFIED / SAMPLED_PROJECTED_ZERO_NOT_CERTIFIED: a sampled audit saw no gap above tolerance, which is not a population-level certificate.",
        "",
        "QuotientGuard is a diagnostic. A measured gap does not by itself imply downstream reward degradation.",
        "",
        "Research basis: Minseong Sim (2026), Diagnosing Estimator Homomorphism Compatibility in Reinforcement Learning with Verifiable Rewards, DOI 10.5281/zenodo.22859044.",
        "",
    ]
    return "\n".join(lines)


def text_report(certificate: dict[str, Any]) -> str:
    """Compact terminal report for the common check/demo path."""
    summary = certificate["summary"]
    lines = [
        "QuotientGuard",
        f"snapshot: {certificate.get('policy_snapshot_id') or 'n/a'}",
        f"policy: {summary['policy_status']}",
        (
            "classes: {classes} | measured gaps: {gaps} | "
            "certified: {certified} | not certified: {uncertified}"
        ).format(
            classes=summary["class_count"],
            gaps=summary["measured_gap_count"],
            certified=summary["sufficiently_certified_count"],
            uncertified=summary["not_certified_count"],
        ),
    ]
    for row in certificate["classes"]:
        details = [row["compatibility_status"]]
        if row.get("coefficient_spread") is not None:
            details.append(f"coeff_spread={row['coefficient_spread']:.6g}")
        if row.get("gap_ratio") is not None:
            details.append(f"gap_ratio={row['gap_ratio']:.6g}")
        lines.append(f"- {row['class_id']}: " + " | ".join(details))
    lines.append(
        "Interpretation: compatibility diagnostic only; no downstream reward/quality claim."
    )
    return "\n".join(lines)


def comparison_text(comparison: dict[str, Any]) -> str:
    summary = comparison["summary"]
    lines = [
        "QuotientGuard comparison",
        f"policy: {summary['policy_status']}",
        (
            "classes: {classes} | new gaps: {new} | removed gaps: {removed} | "
            "evidence downgrades: {downgrades}"
        ).format(
            classes=summary["class_count"],
            new=summary["new_measured_gap_count"],
            removed=summary["removed_measured_gap_count"],
            downgrades=summary["evidence_downgrade_count"],
        ),
        f"estimator changed: {str(comparison['estimator_changed']).lower()}",
    ]
    for row in comparison["classes"]:
        flags = []
        if row["new_measured_gap"]:
            flags.append("NEW_GAP")
        if row["removed_measured_gap"]:
            flags.append("REMOVED_GAP")
        if row["evidence_downgrade"]:
            flags.append("EVIDENCE_DOWNGRADE")
        suffix = f" [{' '.join(flags)}]" if flags else ""
        lines.append(
            f"- {row['class_id']}: {row['baseline_status']} -> "
            f"{row['candidate_status']}{suffix}"
        )
    return "\n".join(lines)
