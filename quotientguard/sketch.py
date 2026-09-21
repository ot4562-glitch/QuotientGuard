"""Deterministic low-dimensional linear score sketches."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable

from .mathops import ensure_vector


@dataclass(frozen=True)
class HashProjection:
    dimension: int
    seed: str = "quotientguard-hash-projection-v1"

    def __post_init__(self) -> None:
        if self.dimension <= 0:
            raise ValueError("sketch dimension must be positive")
        if not self.seed:
            raise ValueError("sketch seed must not be empty")

    def _bucket_sign(self, coordinate: int) -> tuple[int, float]:
        digest = hashlib.sha256(
            f"{self.seed}:{coordinate}".encode("utf-8")
        ).digest()
        bucket = int.from_bytes(digest[:8], "big") % self.dimension
        sign = 1.0 if (digest[8] & 1) == 0 else -1.0
        return bucket, sign

    def project(self, values: Iterable[float]) -> list[float]:
        vector = ensure_vector(values, "score")
        out = [0.0] * self.dimension
        for coordinate, value in enumerate(vector):
            bucket, sign = self._bucket_sign(coordinate)
            out[bucket] += sign * value
        return out

    def metadata(self, source_dimension: int) -> dict[str, Any]:
        if source_dimension <= 0:
            raise ValueError("source_dimension must be positive")
        return {
            "kind": "sketch",
            "method": "deterministic_hash_sign_v1",
            "seed": self.seed,
            "source_dimension": source_dimension,
            "sketch_dimension": self.dimension,
            "semantics": (
                "non-zero projection proves a non-zero source component; "
                "zero projection does not certify the source vector is zero"
            ),
        }


def sketch_observations(
    observations: list[dict[str, Any]],
    *,
    dimension: int,
    seed: str = "quotientguard-hash-projection-v1",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not observations:
        raise ValueError("at least one observation is required")
    projection = HashProjection(dimension=dimension, seed=seed)

    source_dimension = None
    output = []
    for row in observations:
        score = row.get("score")
        if score is None:
            raise ValueError("every observation must contain a score vector")
        score_vector = ensure_vector(score, "score")
        if source_dimension is None:
            source_dimension = len(score_vector)
        elif len(score_vector) != source_dimension:
            raise ValueError("all score vectors must share one source dimension")

        transformed = dict(row)
        transformed["score"] = projection.project(score_vector)
        transformed["score_kind"] = "sketch"
        output.append(transformed)

    assert source_dimension is not None
    return output, projection.metadata(source_dimension)
