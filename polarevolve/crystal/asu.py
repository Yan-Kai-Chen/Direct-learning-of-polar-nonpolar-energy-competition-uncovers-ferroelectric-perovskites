"""Canonical ASU parameter normalization and orbit expansion."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from polarevolve.crystal.symmetry import WyckoffGauge
from polarevolve.crystal.contracts import ContractError


def _fraction_array(values: Sequence[Sequence[object]] | Sequence[object]) -> np.ndarray:
    if values and isinstance(values[0], (tuple, list)):
        return np.asarray(
            [[float(value) for value in row] for row in values], dtype=np.float64
        )
    return np.asarray([float(value) for value in values], dtype=np.float64)


def canonicalize_parameters(
    gauge: WyckoffGauge, parameters: Sequence[float]
) -> np.ndarray:
    values = np.asarray(parameters, dtype=np.float64)
    dimension = gauge.free_dimension
    if values.shape != (dimension,) or not np.isfinite(values).all():
        raise ContractError(f"parameters must be finite with shape ({dimension},)")
    if dimension == 0:
        return np.empty((0,), dtype=np.float64)
    period = _fraction_array(gauge.parameter_period_basis)
    coordinates = np.linalg.solve(period, values)
    return period @ (coordinates - np.floor(coordinates))


def expand_orbit(
    gauge: WyckoffGauge,
    parameters: Sequence[float],
    *,
    wrap: bool = True,
) -> np.ndarray:
    q = canonicalize_parameters(gauge, parameters)
    points = []
    for member in gauge.member_maps:
        point = _fraction_array(member.origin) + _fraction_array(member.basis) @ q
        points.append(point - np.floor(point) if wrap else point)
    output = np.asarray(points, dtype=np.float64)
    if output.shape != (gauge.multiplicity, 3):
        raise ContractError("expanded orbit shape contradicts its multiplicity")
    return output


__all__ = ["canonicalize_parameters", "expand_orbit"]
