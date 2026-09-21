"""Frozen source manifests for source-sensitive reconstructed adapters."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from typing import Any

REFERENCE_TRL_COMMIT = "a98fa6a4428f9aae58dfb26d729d7437f662f27a"
REFERENCE_TRL_GRPO_SHA256 = (
    "e9f5ad165c620c8da1eceade84b00e405c76691d9af363681e8f7cbbfae8a709"
)

SOURCE_MANIFESTS: dict[str, dict[str, Any]] = {
    "trl-grpo-research-frozen-v1": {
        "framework": "trl",
        "component": "trl/trainer/grpo_trainer.py",
        "source_commit": REFERENCE_TRL_COMMIT,
        "sha256": REFERENCE_TRL_GRPO_SHA256,
        "validation_evidence": {
            "paper_doi": "10.5281/zenodo.22859044",
            "actual_trainer_result": "F8_TFGL_ACTUAL_TRL_TRAINER_RESULT_V0_3",
            "actual_trainer_valid_seeds": "5/5",
        },
    }
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source_file(
    path: str | Path,
    *,
    manifest_id: str = "trl-grpo-research-frozen-v1",
) -> dict[str, Any]:
    if manifest_id not in SOURCE_MANIFESTS:
        raise ValueError(f"unknown source manifest: {manifest_id}")
    manifest = SOURCE_MANIFESTS[manifest_id]
    actual_sha256 = sha256_file(path)
    expected_sha256 = str(manifest["sha256"])
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "source hash mismatch for "
            f"{manifest_id}: expected {expected_sha256}, got {actual_sha256}"
        )
    return {
        "manifest_id": manifest_id,
        "framework": manifest["framework"],
        "component": manifest["component"],
        "source_commit": manifest["source_commit"],
        "source_hash": actual_sha256,
        "source_file": str(Path(path)),
        "verified": True,
    }


def discover_installed_trl_grpo_source() -> Path:
    spec = importlib.util.find_spec("trl.trainer.grpo_trainer")
    if spec is None or spec.origin is None:
        raise RuntimeError("installed TRL GRPOTrainer source could not be located")
    return Path(spec.origin)


def verified_installed_trl_provenance() -> dict[str, Any]:
    path = discover_installed_trl_grpo_source()
    return verify_source_file(path)
