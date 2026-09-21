"""Framework-neutral explicit runtime capture.

This path never reconstructs a framework loss. The caller provides the scalar
coefficient actually consumed at the loss/gradient contribution point.
Framework metadata is provenance only.
"""

from __future__ import annotations

from typing import Any, Callable

from ..sampled import audit_sampled_observations


_ALLOWED_CONSUMPTION_POINTS = {
    "loss_contribution",
    "gradient_contribution",
}


class FrameworkRuntimeCapture:
    def __init__(
        self,
        *,
        framework: str,
        policy_snapshot_id: str,
        framework_version: str | None = None,
        source_commit: str | None = None,
        source_hash: str | None = None,
        adapter_name: str = "explicit_runtime_capture",
        adapter_version: str = "1",
        equivalence_contract: dict[str, Any] | None = None,
        score_kind: str = "sketch",
        snapshot_metadata: dict[str, Any] | None = None,
    ) -> None:
        if not framework.strip():
            raise ValueError("framework is required")
        if not policy_snapshot_id.strip():
            raise ValueError("policy_snapshot_id is required")
        if not any((framework_version, source_commit, source_hash)):
            raise ValueError(
                "explicit runtime capture requires at least one framework "
                "provenance field: framework_version, source_commit, or source_hash"
            )
        if score_kind not in {"full", "sketch"}:
            raise ValueError("score_kind must be 'full' or 'sketch'")

        self.framework = framework
        self.policy_snapshot_id = policy_snapshot_id
        self.framework_version = framework_version
        self.source_commit = source_commit
        self.source_hash = source_hash
        self.adapter_name = adapter_name
        self.adapter_version = adapter_version
        self.equivalence_contract = dict(equivalence_contract or {})
        self.score_kind = score_kind
        self.snapshot_metadata = dict(snapshot_metadata or {})
        self._observations: list[dict[str, Any]] = []

    def record(
        self,
        *,
        batch_id: str,
        sample_index: int,
        class_id: str,
        trajectory_id: str,
        realized_coefficient: float,
        state_consuming_point: str,
        score: list[float] | tuple[float, ...] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if state_consuming_point not in _ALLOWED_CONSUMPTION_POINTS:
            raise ValueError(
                "state_consuming_point must identify the actual loss/gradient contribution"
            )
        row_metadata = dict(metadata or {})
        row_metadata["state_consuming_point"] = state_consuming_point
        row_metadata["coefficient_semantics"] = "explicit_runtime_value"

        row: dict[str, Any] = {
            "policy_snapshot_id": self.policy_snapshot_id,
            "batch_id": batch_id,
            "sample_index": sample_index,
            "class_id": class_id,
            "trajectory_id": trajectory_id,
            "realized_coefficient": float(realized_coefficient),
            "metadata": row_metadata,
        }
        if score is not None:
            row["score"] = list(score)
            row["score_kind"] = self.score_kind
        self._observations.append(row)

    def callback(self) -> Callable[..., None]:
        """Return a keyword-only callback suitable for trainer integration."""
        def _callback(**kwargs: Any) -> None:
            self.record(**kwargs)

        return _callback

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
            "name": self.framework,
            "framework": self.framework,
            "framework_version": self.framework_version,
            "source_commit": self.source_commit,
            "source_hash": self.source_hash,
            "adapter": self.adapter_name,
            "adapter_version": self.adapter_version,
            "coefficient_semantics": "explicit_runtime_value",
            "reconstruction_validation": False,
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
