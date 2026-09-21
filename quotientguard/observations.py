"""Aggregate fixed-snapshot trainer observations into a coefficient receipt.

This compatibility layer estimates representative-conditioned coefficients from
sample means. Score vectors are intentionally not averaged into the receipt:
E[H] * E[s] is not generally E[H*s]. Use audit_sampled_observations() for
sampled score/gap measurement.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .schema import normalize_observations


def observations_to_receipt(
    observations: list[dict[str, Any]],
    *,
    run_id: str | None = None,
    estimator: dict[str, Any] | None = None,
    equivalence_contract: dict[str, Any] | None = None,
    score_kind: str = "sketch",
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if score_kind not in {"full", "sketch"}:
        raise ValueError("score_kind must be 'full' or 'sketch'")

    normalized = normalize_observations(
        observations,
        default_score_kind=score_kind,
    )
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    source_had_scores = all(obs.score is not None for obs in normalized)

    for obs in normalized:
        grouped[obs.class_id][obs.trajectory_id].append(obs.to_mapping())

    classes = []
    for class_id in sorted(grouped):
        trajectories = grouped[class_id]
        class_total = sum(len(rows) for rows in trajectories.values())
        representatives = []

        for trajectory_id in sorted(trajectories):
            rows = trajectories[trajectory_id]
            coefficients = [float(row["realized_coefficient"]) for row in rows]
            representatives.append(
                {
                    "representative_id": trajectory_id,
                    "trajectory_id": trajectory_id,
                    "conditional_probability": len(rows) / class_total,
                    "effective_coefficient": sum(coefficients) / len(coefficients),
                    "metadata": {
                        "observation_count": len(rows),
                        "batch_count": len({row["batch_id"] for row in rows}),
                        "coefficient_min": min(coefficients),
                        "coefficient_max": max(coefficients),
                    },
                }
            )

        if len(representatives) < 2:
            raise ValueError(
                f"class {class_id!r} has only one observed trajectory; at least two are required"
            )

        classes.append(
            {
                "class_id": class_id,
                "coefficient_evidence": "sampled",
                "representatives": representatives,
            }
        )

    receipt: dict[str, Any] = {
        "schema_version": "quotientguard.receipt.v2",
        "policy_snapshot_id": normalized[0].policy_snapshot_id,
        "run_id": run_id,
        "estimator": estimator or {},
        "equivalence_contract": equivalence_contract or {},
        "score_space": {"kind": "none"},
        "aggregation": {
            "source": "sampled_observations",
            "observation_count": len(normalized),
            "batch_count": len({obs.batch_id for obs in normalized}),
            "effective_coefficient_estimator": (
                "mean_realized_coefficient_by_trajectory"
            ),
            "conditional_probability_estimator": "empirical_within_class_frequency",
            "source_had_scores": source_had_scores,
            "score_handling": (
                "scores omitted; use audit_sampled_observations for E[H*s] gap estimation"
            ),
            "warning": (
                "sampled coefficient equality is not an exact class-constancy certificate"
            ),
        },
        "classes": classes,
    }
    if policy:
        receipt["policy"] = policy
    return receipt


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    observations = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"line {line_number} must contain a JSON object")
            observations.append(row)
    return observations
