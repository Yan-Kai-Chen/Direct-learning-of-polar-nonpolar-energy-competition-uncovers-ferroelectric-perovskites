"""Stable public interfaces between operation planning and crystal generation.

This module intentionally defines only the information exchanged between the
published PolarGen stages. Production diffusion models, training records, and
crystal-specific numerical kernels are not part of this interface.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence


@dataclass(frozen=True)
class PlannedOperation:
    """One graph-derived operation selected by language-graph fusion."""

    operation_id: str
    name: str
    rank: int
    direction: str
    confidence: float
    magnitude: float
    interval: tuple[float, float]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PlannedOperation":
        interval = tuple(float(item) for item in value["target_interval"])
        if len(interval) != 2 or interval[0] > interval[1]:
            raise ValueError("target_interval must contain ordered lower/upper values")
        confidence = float(value["confidence"])
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between zero and one")
        rank = int(value["rank"])
        if rank < 1:
            raise ValueError("rank must be positive")
        return cls(
            operation_id=str(value["operation_id"]),
            name=str(value["name"]),
            rank=rank,
            direction=str(value["direction"]),
            confidence=confidence,
            magnitude=float(value["magnitude"]),
            interval=(interval[0], interval[1]),
        )


@dataclass(frozen=True)
class OperationPlan:
    """Top-k structural-operation plan for a known nonpolar parent."""

    request_id: str
    parent_id: str
    operations: tuple[PlannedOperation, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "OperationPlan":
        if value.get("schema_version") != "polargen_operation_plan_v1":
            raise ValueError("unsupported operation-plan schema")
        operations = tuple(
            PlannedOperation.from_mapping(item)
            for item in value["selected_operations"]
        )
        if not 1 <= len(operations) <= 3:
            raise ValueError("the public plan contains one to three operations")
        if tuple(item.rank for item in operations) != tuple(
            range(1, len(operations) + 1)
        ):
            raise ValueError("operations must be ordered by consecutive rank")
        return cls(
            request_id=str(value["request_id"]),
            parent_id=str(value["parent"]["structure_id"]),
            operations=operations,
        )


def load_operation_plan(path: str | Path) -> OperationPlan:
    """Load a public operation plan from JSON."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return OperationPlan.from_mapping(payload)


class CrystalGenerationBackend(Protocol):
    """Protocol implemented by a private or independently supplied backend."""

    def generate(
        self,
        parent: Mapping[str, Any],
        plan: OperationPlan,
        *,
        num_samples: int,
    ) -> Sequence[Mapping[str, Any]]:
        """Generate candidate structures without consuming language hidden states."""


__all__ = [
    "CrystalGenerationBackend",
    "OperationPlan",
    "PlannedOperation",
    "load_operation_plan",
]
