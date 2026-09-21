"""Schema v2 objects and fixed-snapshot invariants."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .mathops import ensure_finite_scalar, ensure_vector
from .provenance import sha256_json


@dataclass(frozen=True)
class Observation:
    policy_snapshot_id: str
    batch_id: str
    sample_index: int
    class_id: str
    trajectory_id: str
    realized_coefficient: float
    score: tuple[float, ...] | None = None
    score_kind: str = "none"
    metadata: dict[str, Any] | None = None

    @classmethod
    def from_mapping(
        cls,
        row: Mapping[str, Any],
        *,
        default_score_kind: str = "sketch",
    ) -> "Observation":
        policy_snapshot_id = str(row.get("policy_snapshot_id", "")).strip()
        batch_id = str(row.get("batch_id", "")).strip()
        class_id = str(row.get("class_id", "")).strip()
        trajectory_id = str(row.get("trajectory_id", "")).strip()

        if not policy_snapshot_id:
            raise ValueError("observation requires policy_snapshot_id")
        if not batch_id:
            raise ValueError("observation requires batch_id")
        if not class_id:
            raise ValueError("observation requires class_id")
        if not trajectory_id:
            raise ValueError("observation requires trajectory_id")
        if "sample_index" not in row:
            raise ValueError("observation requires sample_index")

        sample_index = int(row["sample_index"])
        if sample_index < 0:
            raise ValueError("sample_index must be non-negative")

        coefficient = ensure_finite_scalar(
            row["realized_coefficient"],
            "realized_coefficient",
        )

        raw_metadata = row.get("metadata")
        if raw_metadata is None:
            metadata = None
        else:
            if not isinstance(raw_metadata, Mapping):
                raise ValueError("observation metadata must be a mapping")
            metadata = copy.deepcopy(dict(raw_metadata))
            # Validate that metadata can participate in deterministic source hashing.
            sha256_json(metadata)

        raw_score = row.get("score")
        if raw_score is None:
            score = None
            score_kind = "none"
        else:
            score = tuple(ensure_vector(raw_score, "score"))
            score_kind = str(row.get("score_kind", default_score_kind))
            if score_kind not in {"full", "sketch"}:
                raise ValueError(
                    "score_kind must be 'full' or 'sketch' when score is present"
                )

        return cls(
            policy_snapshot_id=policy_snapshot_id,
            batch_id=batch_id,
            sample_index=sample_index,
            class_id=class_id,
            trajectory_id=trajectory_id,
            realized_coefficient=coefficient,
            score=score,
            score_kind=score_kind,
            metadata=metadata,
        )

    def to_mapping(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "policy_snapshot_id": self.policy_snapshot_id,
            "batch_id": self.batch_id,
            "sample_index": self.sample_index,
            "class_id": self.class_id,
            "trajectory_id": self.trajectory_id,
            "realized_coefficient": self.realized_coefficient,
        }
        if self.score is not None:
            row["score"] = list(self.score)
            row["score_kind"] = self.score_kind
        if self.metadata is not None:
            row["metadata"] = copy.deepcopy(self.metadata)
        return row


@dataclass(frozen=True)
class EstimatorFingerprint:
    framework: str
    adapter_name: str
    adapter_version: str
    config_hash: str
    framework_version: str | None = None
    source_commit: str | None = None
    source_hash: str | None = None

    def to_mapping(self) -> dict[str, Any]:
        return {
            "framework": self.framework,
            "framework_version": self.framework_version,
            "source_commit": self.source_commit,
            "source_hash": self.source_hash,
            "adapter_name": self.adapter_name,
            "adapter_version": self.adapter_version,
            "config_hash": self.config_hash,
        }

    @property
    def sha256(self) -> str:
        return sha256_json(self.to_mapping())


def normalize_observations(
    rows: Sequence[Mapping[str, Any]],
    *,
    default_score_kind: str = "sketch",
) -> list[Observation]:
    if not rows:
        raise ValueError("at least one observation is required")

    observations = [
        Observation.from_mapping(row, default_score_kind=default_score_kind)
        for row in rows
    ]

    snapshots = {obs.policy_snapshot_id for obs in observations}
    if len(snapshots) != 1:
        raise ValueError(
            "observations from different policy snapshots cannot be pooled into one audit"
        )

    score_presence = [obs.score is not None for obs in observations]
    if any(score_presence) and not all(score_presence):
        raise ValueError("observation stream mixes rows with and without score vectors")

    if all(score_presence):
        score_kinds = {obs.score_kind for obs in observations}
        if len(score_kinds) != 1:
            raise ValueError("observation stream mixes score kinds")
        dimensions = {len(obs.score or ()) for obs in observations}
        if len(dimensions) != 1:
            raise ValueError("observation stream mixes score dimensions")

    batch_sample_keys = {(obs.batch_id, obs.sample_index) for obs in observations}
    if len(batch_sample_keys) != len(observations):
        raise ValueError("duplicate (batch_id, sample_index) observation")

    return observations
