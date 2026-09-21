"""Mechanism-level explanations for supported trainer inputs."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from .adapters.trl_sequence import (
    effective_coefficient_trace,
    validate_trl_config,
    validate_trl_source_provenance,
)
from .provenance import sha256_json
from .version import __version__


def explain_trl_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema_version") != "trl-sequence-v2":
        raise ValueError("explain currently supports trl-sequence-v2 inputs")

    source = validate_trl_source_provenance(payload)
    config = payload.get("config", {})
    parsed = validate_trl_config(config)
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("samples must be a non-empty list")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample_index, sample in enumerate(samples):
        trajectory_id = str(sample.get("trajectory_id", "")).strip()
        class_id = str(sample.get("class_id", "")).strip()
        if not trajectory_id or not class_id:
            raise ValueError("every sample requires class_id and trajectory_id")

        trace = effective_coefficient_trace(sample, config)
        no_clip_value = (
            trace.sequence_ratio
            * trace.advantage
            / trace.denominator
            / trace.common_divisor
        )
        grouped[class_id].append(
            {
                "sample_index": sample_index,
                "trajectory_id": trajectory_id,
                **trace.to_mapping(),
                "counterfactual_no_clipping_coefficient": no_clip_value,
            }
        )

    classes = []
    for class_id in sorted(grouped):
        traces = grouped[class_id]
        branches = {row["branch"] for row in traces}
        lengths = {row["completion_length"] for row in traces}
        denominators = {row["denominator"] for row in traces}
        ratios = {row["sequence_ratio"] for row in traces}
        coefficients = {row["value"] for row in traces}
        no_clip_values = {
            row["counterfactual_no_clipping_coefficient"] for row in traces
        }

        mechanisms = []
        if len(branches) > 1:
            mechanisms.append("clipping_branch_differs")
        if len(lengths) > 1:
            mechanisms.append("completion_length_differs")
        if len(denominators) > 1:
            mechanisms.append("normalization_denominator_differs")
        if len(ratios) > 1:
            mechanisms.append("sequence_ratio_differs")
        if len(coefficients) > 1:
            mechanisms.append("effective_coefficient_differs")

        classes.append(
            {
                "class_id": class_id,
                "mechanisms_observed": mechanisms,
                "traces": traces,
                "counterfactuals": {
                    "no_clipping": {
                        "coefficients": [
                            row["counterfactual_no_clipping_coefficient"]
                            for row in traces
                        ],
                        "coefficient_mismatch_persists": len(no_clip_values) > 1,
                    }
                },
                "interpretation": (
                    "Mechanical differences only. Counterfactuals are diagnostic "
                    "and are not percentage causal attributions."
                ),
            }
        )

    result: dict[str, Any] = {
        "schema_version": "quotientguard.explanation.v1",
        "tool": {"name": "QuotientGuard", "version": __version__},
        "source": source,
        "policy_snapshot_id": payload.get("policy_snapshot_id"),
        "config": parsed,
        "classes": classes,
    }
    result["explanation_sha256"] = sha256_json(result)
    return result


def explanation_text(result: dict[str, Any]) -> str:
    lines = [
        "QuotientGuard explanation",
        f"snapshot: {result.get('policy_snapshot_id') or 'n/a'}",
    ]
    for class_row in result["classes"]:
        lines.append(f"- class {class_row['class_id']}")
        mechanisms = class_row["mechanisms_observed"]
        lines.append(
            "  observed: " + (", ".join(mechanisms) if mechanisms else "no trace differences")
        )
        for trace in class_row["traces"]:
            lines.append(
                "  {trajectory}: branch={branch} ratio={ratio:.6g} "
                "length={length} denominator={denominator:.6g} H={value:.6g}".format(
                    trajectory=trace["trajectory_id"],
                    branch=trace["branch"],
                    ratio=trace["sequence_ratio"],
                    length=trace["completion_length"],
                    denominator=trace["denominator"],
                    value=trace["value"],
                )
            )
        no_clip = class_row["counterfactuals"]["no_clipping"]
        lines.append(
            "  no-clipping counterfactual mismatch persists: "
            + str(no_clip["coefficient_mismatch_persists"]).lower()
        )
    lines.append("No causal percentage attribution or downstream-quality claim is made.")
    return "\n".join(lines)
