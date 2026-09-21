"""Verifier-output equivalence contracts."""

from __future__ import annotations

from typing import Any, Callable

from .base import EquivalenceContract


def VerifierContract(
    verifier: Callable[[str], Any],
    *,
    name: str = "verifier-output",
    version: str = "1",
    description: str = "",
) -> EquivalenceContract:
    return EquivalenceContract(
        name=name,
        version=version,
        description=description or "Equivalence defined by verifier output.",
        classifier=verifier,
    )


def IntegerAnswerContract(
    *,
    name: str = "integer-answer",
    version: str = "1",
) -> EquivalenceContract:
    def classify(text: str) -> int:
        normalized = text.strip()
        if not normalized:
            raise ValueError("empty completion has no integer answer")
        return int(normalized)

    return VerifierContract(
        classify,
        name=name,
        version=version,
        description="Completions are equivalent when they parse to the same integer.",
    )
