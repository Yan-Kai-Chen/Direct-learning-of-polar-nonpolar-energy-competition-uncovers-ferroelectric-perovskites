"""Rotation-free Hall-invariant coordinates for bounded lattice generation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from polarevolve.crystal.symmetry import WyckoffDatabase


_SYMMETRIC_BASIS = (
    np.diag([1.0, 0.0, 0.0]),
    np.diag([0.0, 1.0, 0.0]),
    np.diag([0.0, 0.0, 1.0]),
    np.array([[0.0, 2.0**-0.5, 0.0], [2.0**-0.5, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    np.array([[0.0, 0.0, 2.0**-0.5], [0.0, 0.0, 0.0], [2.0**-0.5, 0.0, 0.0]]),
    np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0**-0.5], [0.0, 2.0**-0.5, 0.0]]),
)


@dataclass(frozen=True)
class HallMetricFrame:
    hall_number: int
    rotations: np.ndarray
    reference_metric: np.ndarray
    reference_sqrt: np.ndarray
    reference_inverse_sqrt: np.ndarray
    shape_basis: np.ndarray
    maximum_rotation_orthogonality_error: float
    reference_invariance_error: float

    @property
    def shape_dimension(self) -> int:
        return int(self.shape_basis.shape[0])


@dataclass(frozen=True)
class LatticeCoordinates:
    log_volume_per_atom: float
    shape_coefficients: tuple[float, ...]


@dataclass(frozen=True)
class LatticeBounds:
    minimum_volume_per_atom: float = 1.0
    maximum_volume_per_atom: float = 100.0
    maximum_aspect_ratio: float = 4.5

    def __post_init__(self) -> None:
        if not 0.0 < self.minimum_volume_per_atom < self.maximum_volume_per_atom:
            raise ValueError("volume-per-atom bounds must satisfy 0 < min < max")
        if self.maximum_aspect_ratio <= 1.0:
            raise ValueError("maximum_aspect_ratio must be greater than one")


@dataclass(frozen=True)
class LatticeEncodingDiagnostics:
    volume_per_atom: float
    aspect_ratio: float
    metric_invariance_error: float
    shape_projection_error: float
    metric_roundtrip_error: float


@dataclass(frozen=True)
class LatticeDecodeDiagnostics:
    volume_per_atom: float
    aspect_ratio: float
    volume_was_clipped: bool
    shape_scale: float
    metric_invariance_error: float


def _symmetric_matrix_function(matrix: np.ndarray, function) -> np.ndarray:
    values, vectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
    if not np.isfinite(values).all() or float(values.min()) <= 0.0:
        raise ValueError("lattice metric must be finite and positive definite")
    return vectors @ np.diag(function(values)) @ vectors.T


def _unit_determinant(metric: np.ndarray) -> np.ndarray:
    determinant = float(np.linalg.det(metric))
    if not math.isfinite(determinant) or determinant <= 0.0:
        raise ValueError("metric determinant must be finite and positive")
    return metric / determinant ** (1.0 / 3.0)


def lattice_aspect_ratio(metric: np.ndarray) -> float:
    values = np.linalg.eigvalsh(0.5 * (metric + metric.T))
    if not np.isfinite(values).all() or float(values.min()) <= 0.0:
        raise ValueError("metric must be finite and positive definite")
    return float(np.sqrt(values.max() / values.min()))


def metric_invariance_error(metric: np.ndarray, rotations: np.ndarray) -> float:
    scale = max(float(np.linalg.norm(metric)), np.finfo(np.float64).tiny)
    return max(
        float(np.linalg.norm(rotation.T @ metric @ rotation - metric)) / scale
        for rotation in rotations
    )


def _general_position_rotations(
    database: WyckoffDatabase, hall_number: int
) -> np.ndarray:
    general = max(
        (
            entry
            for entry in database.entries_for_hall(hall_number)
            if entry.free_dimension == 3
        ),
        key=lambda entry: entry.multiplicity,
    )
    period = np.asarray(general.parameter_period_basis, dtype=np.float64)
    reference = np.asarray(general.member_maps[0].basis, dtype=np.float64) @ period
    inverse_reference = np.linalg.inv(reference)
    unique: dict[tuple[float, ...], np.ndarray] = {}
    for member in general.member_maps:
        rotation = np.asarray(member.basis, dtype=np.float64) @ period @ inverse_reference
        unique.setdefault(tuple(np.round(rotation, 12).reshape(-1)), rotation)
    return np.stack(tuple(unique.values()))


def _canonical_null_basis(constraints: np.ndarray, *, tolerance: float) -> np.ndarray:
    _, singular_values, right = np.linalg.svd(constraints, full_matrices=True)
    scale = float(singular_values.max()) if singular_values.size else 1.0
    rank = int(np.count_nonzero(singular_values > tolerance * max(scale, 1.0)))
    null = right[rank:].T
    projector = null @ null.T
    columns: list[np.ndarray] = []
    for axis in range(projector.shape[0]):
        vector = projector[:, axis].copy()
        for existing in columns:
            vector -= existing * float(existing @ vector)
        norm = float(np.linalg.norm(vector))
        if norm <= tolerance:
            continue
        vector /= norm
        pivot = int(np.argmax(np.abs(vector)))
        if vector[pivot] < 0.0:
            vector *= -1.0
        columns.append(vector)
    if len(columns) != null.shape[1]:
        raise RuntimeError("failed to construct a canonical Hall metric basis")
    return np.stack(columns, axis=1) if columns else np.zeros((6, 0))


def build_hall_metric_frame(
    database: WyckoffDatabase,
    hall_number: int,
    *,
    tolerance: float = 1.0e-10,
) -> HallMetricFrame:
    rotations = _general_position_rotations(database, int(hall_number))
    reference = _unit_determinant(
        np.mean([rotation.T @ rotation for rotation in rotations], axis=0)
    )
    reference_sqrt = _symmetric_matrix_function(reference, np.sqrt)
    reference_inverse_sqrt = _symmetric_matrix_function(reference, lambda value: value**-0.5)
    orthogonal = np.stack(
        [reference_sqrt @ rotation @ reference_inverse_sqrt for rotation in rotations]
    )
    orthogonality_error = max(
        float(np.linalg.norm(rotation.T @ rotation - np.eye(3)))
        for rotation in orthogonal
    )
    if orthogonality_error > 1.0e-8:
        raise ValueError("Hall rotations did not become orthogonal in the reference metric")

    basis = np.stack(_SYMMETRIC_BASIS)
    rows = [np.asarray([np.trace(item) for item in basis], dtype=np.float64)]
    for rotation in orthogonal:
        rows.extend(
            np.stack([rotation.T @ item @ rotation - item for item in basis], axis=-1)
            .reshape(9, 6)
        )
    coefficients = _canonical_null_basis(np.vstack(rows), tolerance=tolerance)
    matrices = np.stack(
        [sum(coefficients[index, column] * basis[index] for index in range(6))
         for column in range(coefficients.shape[1])]
    ) if coefficients.shape[1] else np.zeros((0, 3, 3), dtype=np.float64)
    return HallMetricFrame(
        hall_number=int(hall_number),
        rotations=rotations,
        reference_metric=reference,
        reference_sqrt=reference_sqrt,
        reference_inverse_sqrt=reference_inverse_sqrt,
        shape_basis=matrices,
        maximum_rotation_orthogonality_error=orthogonality_error,
        reference_invariance_error=metric_invariance_error(reference, rotations),
    )


def _shape_factor(
    frame: HallMetricFrame, coefficients: np.ndarray, scale: float
) -> np.ndarray:
    log_shape = np.einsum("d,dij->ij", coefficients * scale, frame.shape_basis)
    values, vectors = np.linalg.eigh(0.5 * (log_shape + log_shape.T))
    if not np.isfinite(values).all():
        raise ValueError("log metric must be finite")
    with np.errstate(over="ignore"):
        return frame.reference_sqrt @ vectors @ np.diag(np.exp(0.5 * values))


def _shape_aspect_ratio(
    frame: HallMetricFrame, coefficients: np.ndarray, scale: float
) -> float:
    factor = _shape_factor(frame, coefficients, scale)
    if not np.isfinite(factor).all():
        return math.inf
    singular_values = np.linalg.svd(factor, compute_uv=False)
    if float(singular_values.min()) <= 0.0:
        return math.inf
    return float(singular_values.max() / singular_values.min())


def _shape_metric(
    frame: HallMetricFrame, coefficients: np.ndarray, scale: float
) -> np.ndarray:
    factor = _shape_factor(frame, coefficients, scale)
    return factor @ factor.T


def decode_lattice_coordinates(
    coordinates: LatticeCoordinates,
    *,
    num_atoms: int,
    frame: HallMetricFrame,
    bounds: LatticeBounds | None = None,
) -> tuple[np.ndarray, LatticeDecodeDiagnostics]:
    if num_atoms <= 0:
        raise ValueError("num_atoms must be positive")
    coefficients = np.asarray(coordinates.shape_coefficients, dtype=np.float64)
    if coefficients.shape != (frame.shape_dimension,) or not np.isfinite(coefficients).all():
        raise ValueError("shape coefficients do not match the Hall metric frame")
    raw_vpa = math.exp(float(coordinates.log_volume_per_atom))
    if not math.isfinite(raw_vpa) or raw_vpa <= 0.0:
        raise ValueError("volume per atom must be finite and positive")
    volume_per_atom = raw_vpa
    volume_was_clipped = False
    shape_scale = 1.0
    if bounds is not None:
        volume_per_atom = min(
            max(raw_vpa, bounds.minimum_volume_per_atom),
            bounds.maximum_volume_per_atom,
        )
        volume_was_clipped = volume_per_atom != raw_vpa
        if lattice_aspect_ratio(frame.reference_metric) > bounds.maximum_aspect_ratio:
            raise ValueError("Hall reference metric exceeds the configured aspect bound")
        if _shape_aspect_ratio(frame, coefficients, 1.0) > bounds.maximum_aspect_ratio:
            low, high = 0.0, 1.0
            for _ in range(48):
                middle = 0.5 * (low + high)
                if (
                    _shape_aspect_ratio(frame, coefficients, middle)
                    <= bounds.maximum_aspect_ratio
                ):
                    low = middle
                else:
                    high = middle
            shape_scale = low
    normalized_metric = _unit_determinant(_shape_metric(frame, coefficients, shape_scale))
    volume = volume_per_atom * float(num_atoms)
    metric = normalized_metric * volume ** (2.0 / 3.0)
    lattice = _symmetric_matrix_function(metric, np.sqrt)
    diagnostics = LatticeDecodeDiagnostics(
        volume_per_atom=volume_per_atom,
        aspect_ratio=lattice_aspect_ratio(metric),
        volume_was_clipped=volume_was_clipped,
        shape_scale=shape_scale,
        metric_invariance_error=metric_invariance_error(metric, frame.rotations),
    )
    return lattice, diagnostics


def encode_lattice_coordinates(
    lattice: np.ndarray,
    *,
    num_atoms: int,
    frame: HallMetricFrame,
) -> tuple[LatticeCoordinates, LatticeEncodingDiagnostics]:
    matrix = np.asarray(lattice, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or num_atoms <= 0:
        raise ValueError("lattice must be finite 3x3 and num_atoms must be positive")
    metric = matrix @ matrix.T
    determinant = float(np.linalg.det(metric))
    if determinant <= 0.0:
        raise ValueError("lattice metric must have a positive determinant")
    volume = math.sqrt(determinant)
    normalized = metric / determinant ** (1.0 / 3.0)
    whitened = frame.reference_inverse_sqrt @ normalized @ frame.reference_inverse_sqrt
    log_shape = _symmetric_matrix_function(whitened, np.log)
    coefficients = np.asarray(
        [float(np.sum(log_shape * item)) for item in frame.shape_basis]
    )
    projected = np.einsum("d,dij->ij", coefficients, frame.shape_basis)
    coordinates = LatticeCoordinates(
        log_volume_per_atom=math.log(volume / float(num_atoms)),
        shape_coefficients=tuple(float(value) for value in coefficients),
    )
    decoded, _ = decode_lattice_coordinates(
        coordinates, num_atoms=num_atoms, frame=frame
    )
    decoded_metric = decoded @ decoded.T
    scale = max(float(np.linalg.norm(metric)), np.finfo(np.float64).tiny)
    diagnostics = LatticeEncodingDiagnostics(
        volume_per_atom=volume / float(num_atoms),
        aspect_ratio=lattice_aspect_ratio(metric),
        metric_invariance_error=metric_invariance_error(metric, frame.rotations),
        shape_projection_error=float(np.linalg.norm(log_shape - projected))
        / max(float(np.linalg.norm(log_shape)), np.finfo(np.float64).tiny),
        metric_roundtrip_error=float(np.linalg.norm(decoded_metric - metric)) / scale,
    )
    return coordinates, diagnostics


__all__ = [
    "HallMetricFrame",
    "LatticeBounds",
    "LatticeCoordinates",
    "LatticeDecodeDiagnostics",
    "LatticeEncodingDiagnostics",
    "build_hall_metric_frame",
    "decode_lattice_coordinates",
    "encode_lattice_coordinates",
    "lattice_aspect_ratio",
    "metric_invariance_error",
]
