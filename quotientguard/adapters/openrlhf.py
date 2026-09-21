"""Stable explicit-runtime integration surface for OpenRLHF."""

from __future__ import annotations

from typing import Any

from .runtime_base import FrameworkRuntimeCapture


class OpenRLHFRuntimeCapture(FrameworkRuntimeCapture):
    def __init__(
        self,
        *,
        policy_snapshot_id: str,
        framework_version: str | None = None,
        source_commit: str | None = None,
        source_hash: str | None = None,
        equivalence_contract: dict[str, Any] | None = None,
        score_kind: str = "sketch",
        snapshot_metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            framework="openrlhf",
            policy_snapshot_id=policy_snapshot_id,
            framework_version=framework_version,
            source_commit=source_commit,
            source_hash=source_hash,
            adapter_name="openrlhf_explicit_runtime",
            adapter_version="1",
            equivalence_contract=equivalence_contract,
            score_kind=score_kind,
            snapshot_metadata=snapshot_metadata,
        )
