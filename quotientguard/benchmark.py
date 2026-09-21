"""Local microbenchmarks for QuotientGuard audit overhead."""

from __future__ import annotations

import statistics
import time
from typing import Any

from .sampled import audit_sampled_observations
from .version import __version__


def _synthetic_observations(
    sample_count: int,
    score_dimension: int,
) -> list[dict[str, Any]]:
    if sample_count < 2:
        raise ValueError("sample_count must be at least 2")
    if score_dimension < 2:
        raise ValueError("score_dimension must be at least 2")

    rows = []
    for index in range(sample_count):
        trajectory = "a" if index % 2 == 0 else "b"
        score = [0.0] * score_dimension
        score[0 if trajectory == "a" else 1] = 1.0
        rows.append(
            {
                "policy_snapshot_id": "benchmark-theta",
                "batch_id": f"batch-{index // 8}",
                "sample_index": index % 8,
                "class_id": "benchmark-class",
                "trajectory_id": trajectory,
                "realized_coefficient": 1.0 if trajectory == "a" else 1.1,
                "score": score,
                "score_kind": "sketch",
            }
        )
    return rows


def benchmark_sampled_audit(
    *,
    sample_count: int = 256,
    score_dimension: int = 64,
    repeats: int = 7,
) -> dict[str, Any]:
    if repeats <= 0:
        raise ValueError("repeats must be positive")

    observations = _synthetic_observations(sample_count, score_dimension)
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        audit_sampled_observations(
            observations,
            run_id="local-overhead-benchmark",
            score_kind="sketch",
        )
        durations.append(time.perf_counter() - start)

    median_seconds = statistics.median(durations)
    return {
        "schema_version": "quotientguard.benchmark.v1",
        "tool_version": __version__,
        "benchmark": "sampled_fixed_snapshot_audit",
        "sample_count": sample_count,
        "score_dimension": score_dimension,
        "repeats": repeats,
        "median_ms": median_seconds * 1000.0,
        "min_ms": min(durations) * 1000.0,
        "max_ms": max(durations) * 1000.0,
        "observations_per_second_at_median": (
            sample_count / median_seconds if median_seconds > 0 else None
        ),
        "scope_note": (
            "Measures QuotientGuard audit runtime only; it is not an end-to-end "
            "trainer overhead percentage."
        ),
    }
