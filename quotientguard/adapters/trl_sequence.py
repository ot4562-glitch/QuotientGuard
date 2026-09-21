"""Fail-closed TRL sequence-level estimator adapter.

Supported scientific reference:
TRL commit a98fa6a4428f9aae58dfb26d729d7437f662f27a,
sequence-level importance sampling, beta=0, no extra policy multiplier.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from ..mathops import ensure_finite_scalar
from ..observations import observations_to_receipt
from ..provenance import sha256_json
from ..source_manifests import (
    REFERENCE_TRL_COMMIT,
    REFERENCE_TRL_GRPO_SHA256,
)

SUPPORTED_LOSS_TYPES = {"grpo", "bnpo", "dr_grpo"}
ADAPTER_VERSION = "trl-sequence-v3"


@dataclass(frozen=True)
class CoefficientTrace:
    value: float
    branch: str
    numerator: float
    denominator: float
    common_divisor: float
    advantage: float
    sequence_ratio: float
    completion_length: int
    loss_type: str

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)


def validate_trl_source_provenance(payload: dict[str, Any]) -> dict[str, Any]:
    source = payload.get("source_provenance")
    if not isinstance(source, dict):
        raise ValueError(
            "TRL input requires source_provenance; unsupported source semantics must fail closed"
        )
    if source.get("framework") != "trl":
        raise ValueError("source_provenance.framework must be 'trl'")
    commit = str(source.get("source_commit", "")).strip()
    if commit != REFERENCE_TRL_COMMIT:
        raise ValueError(
            "unsupported TRL source commit; adapter is frozen to "
            f"{REFERENCE_TRL_COMMIT}"
        )
    source_hash = str(source.get("source_hash", "")).strip()
    if source_hash != REFERENCE_TRL_GRPO_SHA256:
        raise ValueError(
            "unsupported or missing TRL source hash; expected frozen "
            f"grpo_trainer.py SHA-256 {REFERENCE_TRL_GRPO_SHA256}"
        )
    return {
        "framework": "trl",
        "source_commit": commit,
        "source_hash": source_hash,
        "framework_version": source.get("framework_version"),
        "manifest_id": "trl-grpo-research-frozen-v1",
    }


def _active_surrogate(
    *,
    advantage: float,
    ratio: float,
    epsilon_low: float,
    epsilon_high: float,
    boundary_tolerance: float = 1e-12,
) -> tuple[float, str]:
    if ratio <= 0:
        raise ValueError("sequence_ratio must be positive")
    if epsilon_low < 0 or epsilon_high < 0:
        raise ValueError("clip epsilons must be non-negative")
    if advantage == 0.0:
        return 0.0, "zero_advantage"

    lower = 1.0 - epsilon_low
    upper = 1.0 + epsilon_high
    if advantage > 0.0:
        if ratio > upper + boundary_tolerance:
            return 0.0, "clipped_high"
        if abs(ratio - upper) <= boundary_tolerance:
            raise ValueError("sequence ratio lies on the upper clipping boundary")
    else:
        if ratio < lower - boundary_tolerance:
            return 0.0, "clipped_low"
        if abs(ratio - lower) <= boundary_tolerance:
            raise ValueError("sequence ratio lies on the lower clipping boundary")
    return ratio * advantage, "active_ratio"


def _normalization_denominator(
    *,
    loss_type: str,
    completion_length: int,
    batch_size: int,
    active_count: int | None,
    max_completion_length: int | None,
) -> float:
    if completion_length <= 0:
        raise ValueError("completion_length must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if loss_type == "grpo":
        return float(batch_size * completion_length)
    if loss_type == "bnpo":
        if active_count is None or active_count <= 0:
            raise ValueError("bnpo requires a positive active-count denominator")
        return float(active_count)
    if loss_type == "dr_grpo":
        if max_completion_length is None or max_completion_length <= 0:
            raise ValueError("dr_grpo requires positive max_completion_length")
        return float(batch_size * max_completion_length)
    raise ValueError(f"unsupported loss_type: {loss_type}")


def validate_trl_config(config: dict[str, Any]) -> dict[str, Any]:
    loss_type = str(config.get("loss_type", ""))
    if loss_type not in SUPPORTED_LOSS_TYPES:
        raise ValueError("unsupported loss_type")
    if config.get("importance_sampling_level", "sequence") != "sequence":
        raise ValueError("sequence-level importance sampling is required")
    if float(config.get("beta", 0.0)) != 0.0:
        raise ValueError("beta must be zero for this adapter")
    if config.get("extra_policy_multiplier", False):
        raise ValueError("extra policy multipliers are outside this adapter scope")

    epsilon_low = ensure_finite_scalar(config.get("epsilon_low", 0.2), "epsilon_low")
    epsilon_high = ensure_finite_scalar(config.get("epsilon_high", 0.2), "epsilon_high")
    batch_size = int(config["batch_size"])
    active_count = config.get("active_count")
    if active_count is not None:
        active_count = int(active_count)
    max_completion_length = config.get("max_completion_length")
    if max_completion_length is not None:
        max_completion_length = int(max_completion_length)
    common_divisor = ensure_finite_scalar(
        config.get("common_divisor", 1.0),
        "common_divisor",
    )
    if common_divisor <= 0:
        raise ValueError("common_divisor must be positive")

    return {
        "loss_type": loss_type,
        "epsilon_low": epsilon_low,
        "epsilon_high": epsilon_high,
        "batch_size": batch_size,
        "active_count": active_count,
        "max_completion_length": max_completion_length,
        "common_divisor": common_divisor,
    }


def effective_coefficient_trace(
    sample: dict[str, Any],
    config: dict[str, Any],
) -> CoefficientTrace:
    parsed = validate_trl_config(config)
    advantage = ensure_finite_scalar(sample.get("advantage"), "advantage")
    ratio = ensure_finite_scalar(sample.get("sequence_ratio"), "sequence_ratio")
    completion_length = int(sample.get("completion_length"))

    active_surrogate, branch = _active_surrogate(
        advantage=advantage,
        ratio=ratio,
        epsilon_low=parsed["epsilon_low"],
        epsilon_high=parsed["epsilon_high"],
    )
    denominator = _normalization_denominator(
        loss_type=parsed["loss_type"],
        completion_length=completion_length,
        batch_size=parsed["batch_size"],
        active_count=parsed["active_count"],
        max_completion_length=parsed["max_completion_length"],
    )
    value = active_surrogate / denominator / parsed["common_divisor"]
    return CoefficientTrace(
        value=value,
        branch=branch,
        numerator=active_surrogate,
        denominator=denominator,
        common_divisor=parsed["common_divisor"],
        advantage=advantage,
        sequence_ratio=ratio,
        completion_length=completion_length,
        loss_type=parsed["loss_type"],
    )


def trl_sequence_batch_to_receipt(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema_version") != "trl-sequence-v2":
        raise ValueError("schema_version must be trl-sequence-v2")

    source = validate_trl_source_provenance(payload)
    policy_snapshot_id = str(payload.get("policy_snapshot_id", "")).strip()
    batch_id = str(payload.get("batch_id", "")).strip()
    if not policy_snapshot_id:
        raise ValueError("TRL payload requires policy_snapshot_id")
    if not batch_id:
        raise ValueError("TRL payload requires batch_id")

    config = payload.get("config", {})
    parsed = validate_trl_config(config)
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("samples must be a non-empty list")

    observations = []
    traces = []
    branch_counts: dict[str, int] = {}
    for sample_index, sample in enumerate(samples):
        trajectory_id = str(sample.get("trajectory_id", "")).strip()
        if not trajectory_id:
            raise ValueError("each TRL sample requires trajectory_id")

        trace = effective_coefficient_trace(sample, config)
        branch_counts[trace.branch] = branch_counts.get(trace.branch, 0) + 1
        traces.append(
            {
                "sample_index": sample_index,
                "class_id": sample.get("class_id"),
                "trajectory_id": trajectory_id,
                **trace.to_mapping(),
            }
        )

        observation: dict[str, Any] = {
            "policy_snapshot_id": policy_snapshot_id,
            "batch_id": batch_id,
            "sample_index": sample_index,
            "class_id": sample.get("class_id"),
            "trajectory_id": trajectory_id,
            "realized_coefficient": trace.value,
        }
        if sample.get("score") is not None:
            observation["score"] = sample.get("score")
            observation["score_kind"] = payload.get("score_space", {}).get(
                "kind",
                "sketch",
            )
        observations.append(observation)

    estimator = {
        "name": "trl.GRPOTrainer",
        "framework": "trl",
        "adapter": "trl_sequence",
        "adapter_version": ADAPTER_VERSION,
        "loss_type": parsed["loss_type"],
        "importance_sampling_level": "sequence",
        "epsilon_low": parsed["epsilon_low"],
        "epsilon_high": parsed["epsilon_high"],
        "source_commit": source["source_commit"],
        "source_hash": source["source_hash"],
        "configuration_sha256": sha256_json(config),
    }

    receipt = observations_to_receipt(
        observations,
        run_id=payload.get("run_id"),
        estimator=estimator,
        equivalence_contract=payload.get("equivalence_contract", {}),
        score_kind=payload.get("score_space", {}).get("kind", "sketch"),
        policy=payload.get("policy"),
    )
    receipt["snapshot_metadata"] = dict(payload.get("snapshot_metadata") or {})
    receipt["adapter_provenance"] = {
        "scope": "sequence-level surrogate; beta zero; no extra policy multipliers",
        "source": source,
        "reference_trl_commit": REFERENCE_TRL_COMMIT,
        "branch_counts": branch_counts,
        "batch_size": parsed["batch_size"],
        "active_count": parsed["active_count"],
        "max_completion_length": parsed["max_completion_length"],
        "common_divisor": parsed["common_divisor"],
        "coefficient_traces": traces,
    }
    return receipt
