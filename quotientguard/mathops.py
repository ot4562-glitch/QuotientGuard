"""Small dependency-free vector helpers used by QuotientGuard."""

from __future__ import annotations

import math
from typing import Iterable, Sequence


def ensure_finite_scalar(value: float, label: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return value


def ensure_vector(values: Iterable[float], label: str) -> list[float]:
    vector = [ensure_finite_scalar(v, label) for v in values]
    if not vector:
        raise ValueError(f"{label} must not be empty")
    return vector


def add(a: Sequence[float], b: Sequence[float]) -> list[float]:
    _same_dim(a, b)
    return [x + y for x, y in zip(a, b)]


def sub(a: Sequence[float], b: Sequence[float]) -> list[float]:
    _same_dim(a, b)
    return [x - y for x, y in zip(a, b)]


def scale(a: Sequence[float], c: float) -> list[float]:
    return [c * x for x in a]


def zeros(dim: int) -> list[float]:
    if dim <= 0:
        raise ValueError("vector dimension must be positive")
    return [0.0] * dim


def norm(a: Sequence[float]) -> float:
    return math.sqrt(sum(x * x for x in a))


def max_abs(a: Sequence[float]) -> float:
    return max((abs(x) for x in a), default=0.0)


def weighted_mean(vectors: Sequence[Sequence[float]], weights: Sequence[float]) -> list[float]:
    if len(vectors) != len(weights) or not vectors:
        raise ValueError("vectors and weights must have the same non-zero length")
    dim = len(vectors[0])
    out = zeros(dim)
    for vector, weight in zip(vectors, weights):
        if len(vector) != dim:
            raise ValueError("all score vectors must have the same dimension")
        for j, value in enumerate(vector):
            out[j] += weight * value
    return out


def _same_dim(a: Sequence[float], b: Sequence[float]) -> None:
    if len(a) != len(b):
        raise ValueError("vector dimensions differ")
