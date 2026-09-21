"""Explicit runtime capture bridge for supported TRL estimator state.

The bridge does not monkeypatch TRL. Call it from the actual loss/gradient
consumption point and provide the realized coefficient that production used.
In validation mode it reconstructs the supported coefficient and fails closed
if production and the frozen adapter disagree.
"""

from __future__ import annotations

from typing import Any

from ..sampled import audit_sampled_observations
from ..source_manifests import verified_installed_trl_provenance
from .trl_sequence import (
    ADAPTER_VERSION,
    effective_coefficient_trace,
    validate_trl_source_provenance,
)


_ALLOWED_CONSUMPTION_POINTS = {
    "loss_contribution",
    "gradient_contribution",
}


class TRLRuntimeCapture:
    def __init__(
        self,
        *,
        policy_snapshot_id: str,
        source_provenance: dict[str, Any],
        config: dict[str, Any],
        equivalence_contract: dict[str, Any] | None = None,
        score_kind: str = "sketch",
        snapshot_metadata: dict[str, Any] | None = None,
        validation_tolerance: float = 1e-12,
    ) -> None:
        if not policy_snapshot_id:
            raise ValueError("policy_snapshot_id is required")
        if validation_tolerance < 0:
            raise ValueError("validation_tolerance must be non-negative")

        self.policy_snapshot_id = policy_snapshot_id
        self.source = validate_trl_source_provenance(
            {"source_provenance": source_provenance}
        )
        self.config = dict(config)
        self.equivalence_contract = dict(equivalence_contract or {})
        self.score_kind = score_kind
        self.snapshot_metadata = dict(snapshot_metadata or {})
        self.validation_tolerance = validation_tolerance
        self._observations: list[dict[str, Any]] = []

    @classmethod
    def from_installed(
        cls,
        *,
        policy_snapshot_id: str,
        config: dict[str, Any],
        equivalence_contract: dict[str, Any] | None = None,
        score_kind: str = "sketch",
        snapshot_metadata: dict[str, Any] | None = None,
        validation_tolerance: float = 1e-12,
    ) -> "TRLRuntimeCapture":
        provenance = verified_installed_trl_provenance()
        return cls(
            policy_snapshot_id=policy_snapshot_id,
            source_provenance={
                "framework": "trl",
                "source_commit": provenance["source_commit"],
                "source_hash": provenance["source_hash"],
            },
            config=config,
            equivalence_contract=equivalence_contract,
            score_kind=score_kind,
            snapshot_metadata=snapshot_metadata,
            validation_tolerance=validation_tolerance,
        )

    def record_sequence_sample(
        self,
        *,
        batch_id: str,
        sample_index: int,
        class_id: str,
        trajectory_id: str,
        sample: dict[str, Any],
        realized_coefficient: float,
        state_consuming_point: str,
        score: list[float] | tuple[float, ...] | None = None,
    ) -> None:
        if state_consuming_point not in _ALLOWED_CONSUMPTION_POINTS:
            raise ValueError(
                "state_consuming_point must identify the actual loss/gradient contribution"
            )
        trace = effective_coefficient_trace(sample, self.config)
        observed = float(realized_coefficient)
        if abs(observed - trace.value) > self.validation_tolerance:
            raise ValueError(
                "runtime coefficient does not match the frozen supported TRL reconstruction"
            )

        row: dict[str, Any] = {
            "policy_snapshot_id": self.policy_snapshot_id,
            "batch_id": batch_id,
            "sample_index": sample_index,
            "class_id": class_id,
            "trajectory_id": trajectory_id,
            "realized_coefficient": observed,
            "metadata": {
                "state_consuming_point": state_consuming_point,
                "coefficient_trace": trace.to_mapping(),
            },
        }
        if score is not None:
            row["score"] = list(score)
            row["score_kind"] = self.score_kind
        self._observations.append(row)

    @property
    def observation_count(self) -> int:
        return len(self._observations)

    def observations(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self._observations]

    def certificate(
        self,
        *,
        run_id: str | None = None,
        policy: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self._observations:
            raise ValueError("no runtime observations captured")
        estimator = {
            "name": "trl.GRPOTrainer",
            "framework": "trl",
            "adapter": "trl_runtime_capture",
            "adapter_version": ADAPTER_VERSION,
            "source_commit": self.source["source_commit"],
            "source_hash": self.source["source_hash"],
            "source_manifest": self.source["manifest_id"],
            "coefficient_semantics": "frozen_reconstructed_and_runtime_validated",
            "reconstruction_validation": True,
        }
        return audit_sampled_observations(
            self._observations,
            run_id=run_id,
            estimator=estimator,
            equivalence_contract=self.equivalence_contract,
            score_kind=self.score_kind,
            policy=policy,
            snapshot_metadata=self.snapshot_metadata,
        )
