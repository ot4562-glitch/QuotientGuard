"""Built-in exact decoded-text equivalence contract."""

from __future__ import annotations

from .base import EquivalenceContract


def DecodedTextContract(
    *,
    name: str = "decoded-text",
    version: str = "1",
) -> EquivalenceContract:
    return EquivalenceContract(
        name=name,
        version=version,
        description="Exact decoded-text equality.",
        classifier=lambda text: text,
    )
