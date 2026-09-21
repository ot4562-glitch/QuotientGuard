"""Simple front-door input detection for QuotientGuard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .adapters.trl_sequence import trl_sequence_batch_to_receipt
from .core import audit_receipt
from .observations import read_jsonl
from .sampled import audit_sampled_observations


def check_payload(payload: dict[str, Any]) -> dict[str, Any]:
    schema = payload.get("schema_version")
    if schema == "quotientguard.receipt.v2":
        return audit_receipt(payload)
    if schema == "trl-sequence-v2":
        return audit_receipt(trl_sequence_batch_to_receipt(payload))
    if schema == "quotientguard.observations.v2":
        observations = payload.get("observations")
        if not isinstance(observations, list):
            raise ValueError("quotientguard.observations.v2 requires observations list")
        return audit_sampled_observations(
            observations,
            run_id=payload.get("run_id"),
            estimator=payload.get("estimator"),
            equivalence_contract=payload.get("equivalence_contract"),
            score_kind=payload.get("score_space", {}).get("kind", "sketch"),
            policy=payload.get("policy"),
            snapshot_metadata=payload.get("snapshot_metadata"),
        )
    raise ValueError(
        "unsupported input schema; expected quotientguard.receipt.v2, "
        "quotientguard.observations.v2, or trl-sequence-v2"
    )


def check_path(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".jsonl":
        return audit_sampled_observations(read_jsonl(path))

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("JSON input must contain an object")
    return check_payload(payload)
