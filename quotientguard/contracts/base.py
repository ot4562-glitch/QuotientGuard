"""Equivalence contracts for quotient-class assignment."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from ..provenance import sha256_json


def canonical_class_id(value: Any) -> str:
    """Return a stable, type-preserving class id for JSON-like verifier output."""
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
    except TypeError as exc:
        raise TypeError("equivalence classifier output must be JSON-serializable") from exc


@dataclass(frozen=True)
class EquivalenceContract:
    name: str
    version: str
    classifier: Callable[[str], Any]
    description: str = ""

    def classify(self, text: str) -> str:
        return canonical_class_id(self.classifier(text))

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
        }

    @property
    def sha256(self) -> str:
        return sha256_json(self.metadata())
