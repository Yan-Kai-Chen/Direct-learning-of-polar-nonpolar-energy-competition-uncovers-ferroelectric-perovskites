"""Runtime state in asymmetric-unit and Cartesian dual spaces."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias

from polarevolve.crystal.contracts import ContractError


Vector3: TypeAlias = tuple[float, float, float]
Matrix3: TypeAlias = tuple[Vector3, Vector3, Vector3]


def _finite_float(value: object, *, field_name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ContractError(f"{field_name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ContractError(f"{field_name} must be finite")
    return result


def _normalize_matrix3(matrix: Sequence[Sequence[float]]) -> Matrix3:
    if not isinstance(matrix, (list, tuple)) or len(matrix) != 3:
        raise ContractError("lattice.matrix must be a 3x3 matrix")
    rows: list[Vector3] = []
    for row in matrix:
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            raise ContractError("lattice.matrix must be a 3x3 matrix")
        values = tuple(_finite_float(value, field_name="lattice.matrix") for value in row)
        rows.append((values[0], values[1], values[2]))
    return rows[0], rows[1], rows[2]


def _determinant(matrix: Matrix3) -> float:
    a, b, c = matrix
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


@dataclass(frozen=True)
class LatticeState:
    matrix: Matrix3

    def __post_init__(self) -> None:
        matrix = _normalize_matrix3(self.matrix)
        if _determinant(matrix) <= 1.0e-12:
            raise ContractError("lattice.matrix must have a positive finite determinant")
        object.__setattr__(self, "matrix", matrix)


@dataclass(frozen=True)
class OrbitLayout:
    orbit_ids: tuple[str, ...]
    free_dimensions: tuple[int, ...]
    parameter_offsets: tuple[int, ...]

    def __post_init__(self) -> None:
        orbit_ids = tuple(self.orbit_ids)
        dimensions = tuple(self.free_dimensions)
        offsets = tuple(self.parameter_offsets)
        if not orbit_ids or len(orbit_ids) != len(dimensions):
            raise ContractError("orbit layout ids and dimensions must be non-empty and aligned")
        if len(set(orbit_ids)) != len(orbit_ids):
            raise ContractError("orbit layout ids must be unique")
        if any(not isinstance(value, int) or not 0 <= value <= 3 for value in dimensions):
            raise ContractError("orbit free dimensions must lie in 0..3")
        if len(offsets) != len(orbit_ids) + 1 or offsets[0] != 0:
            raise ContractError("parameter_offsets must start at zero and contain N+1 values")
        for index, dimension in enumerate(dimensions):
            if offsets[index + 1] - offsets[index] != dimension:
                raise ContractError("parameter_offsets do not match orbit free dimensions")
        object.__setattr__(self, "orbit_ids", orbit_ids)
        object.__setattr__(self, "free_dimensions", dimensions)
        object.__setattr__(self, "parameter_offsets", offsets)

    @property
    def num_parameters(self) -> int:
        return self.parameter_offsets[-1]


@dataclass(frozen=True)
class ASUState:
    parameters: tuple[float, ...]
    lattice: LatticeState
    layout: OrbitLayout

    def __post_init__(self) -> None:
        parameters = tuple(
            _finite_float(value, field_name="asu.parameters") for value in self.parameters
        )
        if not isinstance(self.lattice, LatticeState):
            raise ContractError("asu.lattice must be a LatticeState")
        if not isinstance(self.layout, OrbitLayout):
            raise ContractError("asu.layout must be an OrbitLayout")
        if len(parameters) != self.layout.num_parameters:
            raise ContractError("ASU parameter count does not match OrbitLayout")
        object.__setattr__(self, "parameters", parameters)


__all__ = ["ASUState", "LatticeState", "OrbitLayout"]
