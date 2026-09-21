"""Built-in research-grounded demos."""

from __future__ import annotations

from typing import Any

from .adapters.trl_sequence import REFERENCE_TRL_COMMIT, trl_sequence_batch_to_receipt
from .source_manifests import REFERENCE_TRL_GRPO_SHA256
from .core import audit_receipt


def paper_a1_payload() -> dict[str, Any]:
    return {
        "schema_version": "trl-sequence-v2",
        "policy_snapshot_id": "paper-a1-step2",
        "batch_id": "paper-a1-fixed-batch",
        "run_id": "paper-a1-equal-length-screen",
        "source_provenance": {
            "framework": "trl",
            "source_commit": REFERENCE_TRL_COMMIT,
            "source_hash": REFERENCE_TRL_GRPO_SHA256,
        },
        "config": {
            "loss_type": "grpo",
            "importance_sampling_level": "sequence",
            "beta": 0.0,
            "epsilon_low": 0.2,
            "epsilon_high": 0.2,
            "batch_size": 8,
            "common_divisor": 1.0,
        },
        "equivalence_contract": {
            "name": "paper-a1-answer-4",
            "version": "1",
            "description": (
                "Two one-token representatives accepted as answer 4 in the "
                "paper's equal-length A1 experiment."
            ),
            "research_doi": "10.5281/zenodo.22859044",
        },
        "samples": [
            {
                "class_id": "accepted-answer-4",
                "trajectory_id": "token-19:4",
                "advantage": 1.0,
                "sequence_ratio": 1.3488,
                "completion_length": 1,
            },
            {
                "class_id": "accepted-answer-4",
                "trajectory_id": "token-604:space-4",
                "advantage": 1.0,
                "sequence_ratio": 0.8181,
                "completion_length": 1,
            },
        ],
    }


def run_paper_a1_demo() -> tuple[dict[str, Any], dict[str, Any]]:
    payload = paper_a1_payload()
    receipt = trl_sequence_batch_to_receipt(payload)
    return receipt, audit_receipt(receipt)
