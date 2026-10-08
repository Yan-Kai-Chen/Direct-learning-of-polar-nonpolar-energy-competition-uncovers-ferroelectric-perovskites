"""Geometry metrics shared by Version9 sampling audits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

if TYPE_CHECKING:
    from polarevolve.data.batch import PackedASUBatch


@dataclass(frozen=True)
class GeometryMetrics:
    finite: bool
    volume: float
    volume_per_atom: float
    lattice_aspect_ratio: float
    minimum_pair_distance: float


def geometry_validity(geometry: GeometryMetrics, thresholds) -> tuple[bool, bool, bool]:
    """Separate lattice-cell validity from generated-coordinate validity."""

    lattice_valid = bool(
        geometry.finite
        and geometry.lattice_aspect_ratio <= thresholds.maximum_aspect_ratio
        and thresholds.minimum_volume_per_atom
        <= geometry.volume_per_atom
        <= thresholds.maximum_volume_per_atom
    )
    coordinates_valid = bool(
        geometry.finite
        and geometry.minimum_pair_distance >= thresholds.minimum_distance
    )
    return lattice_valid, coordinates_valid, lattice_valid and coordinates_valid


def geometry_metrics(
    lattice: Sequence[Sequence[float]],
    fractional: Sequence[Sequence[float]],
) -> GeometryMetrics:
    matrix = np.asarray(lattice, dtype=np.float64)
    coordinates = np.asarray(fractional, dtype=np.float64)
    finite = bool(
        matrix.shape == (3, 3)
        and coordinates.ndim == 2
        and coordinates.shape[1:] == (3,)
        and len(coordinates) > 0
        and np.isfinite(matrix).all()
        and np.isfinite(coordinates).all()
    )
    if not finite:
        return GeometryMetrics(False, float("nan"), float("nan"), float("nan"), float("nan"))
    volume = float(abs(np.linalg.det(matrix)))
    singular = np.linalg.svd(matrix, compute_uv=False)
    aspect = float(singular.max() / singular.min()) if singular.min() > 0.0 else float("inf")
    if singular.min() <= 1e-12:
        return GeometryMetrics(True, volume, volume / len(coordinates), aspect, 0.)
    from pymatgen.core import Lattice
    cell = Lattice(matrix)
    bound = float(np.linalg.norm(matrix, axis=-1).min()) * (1 + 1e-10)
    images = cell.get_points_in_sphere([[0, 0, 0]], [0, 0, 0], bound)
    minimum = min(float(distance) for _, distance, _, image in images if np.any(image != 0))
    distances = cell.get_all_distances(coordinates, coordinates)
    np.fill_diagonal(distances, np.inf)
    minimum = min(minimum, float(distances.min()))
    return GeometryMetrics(
        True,
        volume,
        volume / len(coordinates),
        aspect,
        minimum,
    )


def paired_periodic_rmsd(
    lattice: Sequence[Sequence[float]],
    generated_fractional: Sequence[Sequence[float]],
    target_fractional: Sequence[Sequence[float]],
) -> float:
    matrix = np.asarray(lattice, dtype=np.float64)
    generated = np.asarray(generated_fractional, dtype=np.float64)
    target = np.asarray(target_fractional, dtype=np.float64)
    if generated.shape != target.shape or generated.ndim != 2 or generated.shape[1] != 3:
        raise ValueError("paired periodic RMSD requires aligned [atoms,3] coordinates")
    _, squared = minimum_image_displacements(generated - target, matrix)
    return float(np.sqrt(squared.mean()))


def minimum_image_displacements(
    fractional_displacements: np.ndarray,
    lattice: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    from pymatgen.core import Lattice
    from pymatgen.util.coord import pbc_shortest_vectors
    values = np.asarray(fractional_displacements, dtype=np.float64)
    if not values.size:
        return values.copy(), np.empty(values.shape[:-1])
    vectors, squared = pbc_shortest_vectors(Lattice(lattice), [[0, 0, 0]],
                                           values.reshape(-1, 3), return_d2=True)
    best = vectors[0] @ np.linalg.inv(lattice)
    return best.reshape(values.shape), squared[0].reshape(values.shape[:-1])


def _linear_sum_assignment(cost: np.ndarray) -> np.ndarray:
    """Return the minimum-cost column for each row of a square matrix."""

    values = np.asarray(cost, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("assignment cost must be a square matrix")
    size = values.shape[0]
    if size == 0:
        return np.empty((0,), dtype=np.int64)
    if not np.isfinite(values).all():
        raise ValueError("assignment cost must be finite")

    _, assignment = linear_sum_assignment(values)
    return assignment


def _species_assignment(
    lattice: np.ndarray,
    generated: np.ndarray,
    target: np.ndarray,
    generated_species: np.ndarray,
    target_species: np.ndarray,
    shift: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    assignment = np.empty(len(generated), dtype=np.int64)
    residuals = np.empty_like(generated)
    for species in np.unique(generated_species):
        generated_indices = np.flatnonzero(generated_species == species)
        target_indices = np.flatnonzero(target_species == species)
        delta = (
            generated[generated_indices, None, :]
            + shift
            - target[None, target_indices, :]
        )
        minimum_images, squared = minimum_image_displacements(delta, lattice)
        local_assignment = _linear_sum_assignment(squared)
        assignment[generated_indices] = target_indices[local_assignment]
        residuals[generated_indices] = minimum_images[
            np.arange(len(generated_indices)), local_assignment
        ]
    return assignment, residuals


def species_permutation_aligned_rmsd(
    lattice: Sequence[Sequence[float]],
    generated_fractional: Sequence[Sequence[float]],
    generated_atomic_numbers: Sequence[int],
    target_fractional: Sequence[Sequence[float]],
    target_atomic_numbers: Sequence[int],
) -> float:
    """Periodic RMSD minimized over one translation and same-species permutations."""

    matrix = np.asarray(lattice, dtype=np.float64)
    generated = np.asarray(generated_fractional, dtype=np.float64)
    target = np.asarray(target_fractional, dtype=np.float64)
    generated_species = np.asarray(generated_atomic_numbers, dtype=np.int64)
    target_species = np.asarray(target_atomic_numbers, dtype=np.int64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("lattice must be a finite 3x3 matrix")
    if abs(float(np.linalg.det(matrix))) <= np.finfo(np.float64).eps:
        raise ValueError("lattice must be nonsingular")
    if (
        generated.shape != target.shape
        or generated.ndim != 2
        or generated.shape[1:] != (3,)
        or len(generated) == 0
    ):
        raise ValueError("permutation-aligned RMSD requires nonempty aligned [atoms,3] arrays")
    if generated_species.shape != (len(generated),) or target_species.shape != (
        len(target),
    ):
        raise ValueError("atomic numbers must contain one entry per atom")
    if not np.isfinite(generated).all() or not np.isfinite(target).all():
        raise ValueError("fractional coordinates must be finite")
    generated_counts = {
        int(species): int(np.count_nonzero(generated_species == species))
        for species in np.unique(generated_species)
    }
    target_counts = {
        int(species): int(np.count_nonzero(target_species == species))
        for species in np.unique(target_species)
    }
    if generated_counts != target_counts:
        raise ValueError("generated and target compositions differ")

    anchor_species = min(generated_counts, key=lambda species: (generated_counts[species], species))
    generated_anchor = generated[generated_species == anchor_species]
    target_anchor = target[target_species == anchor_species]
    seeds = [np.zeros(3, dtype=np.float64)]
    seeds.extend(
        target_point - generated_point
        for generated_point in generated_anchor
        for target_point in target_anchor
    )
    unique_seeds: dict[tuple[float, float, float], np.ndarray] = {}
    for seed in seeds:
        wrapped = np.asarray(seed, dtype=np.float64) - np.round(seed)
        unique_seeds.setdefault(tuple(np.round(wrapped, decimals=12)), wrapped)

    best = np.inf
    for seed in unique_seeds.values():
        shift = seed.copy()
        for _ in range(12):
            _, residuals = _species_assignment(
                matrix,
                generated,
                target,
                generated_species,
                target_species,
                shift,
            )
            correction = residuals.mean(axis=0)
            shift = shift - correction
            shift -= np.round(shift)
            if float(np.linalg.norm(correction @ matrix)) < 1.0e-12:
                break
        _, residuals = _species_assignment(
            matrix,
            generated,
            target,
            generated_species,
            target_species,
            shift,
        )
        squared = np.sum((residuals @ matrix) ** 2, axis=1)
        best = min(best, float(np.sqrt(squared.mean())))
    return best


def global_translation_aligned_rmsd(
    lattice: Sequence[Sequence[float]],
    generated_fractional: Sequence[Sequence[float]],
    target_fractional: Sequence[Sequence[float]],
) -> float:
    """Paired periodic RMSD after quotienting one common fractional translation."""

    matrix = np.asarray(lattice, dtype=np.float64)
    generated = np.asarray(generated_fractional, dtype=np.float64)
    target = np.asarray(target_fractional, dtype=np.float64)
    if generated.shape != target.shape or generated.ndim != 2 or generated.shape[1] != 3:
        raise ValueError("translation-aligned RMSD requires aligned [atoms,3] coordinates")
    if len(generated) == 0:
        raise ValueError("translation-aligned RMSD requires at least one atom")
    delta = generated - target
    delta -= np.round(delta)
    best = float("inf")
    seeds = np.concatenate((np.zeros((1, 3), dtype=np.float64), delta), axis=0)
    for seed in seeds:
        shift = seed.copy()
        for _ in range(12):
            residual = delta - shift
            residual, _ = minimum_image_displacements(residual, matrix)
            correction = residual.mean(axis=0)
            shift += correction
            shift -= np.round(shift)
            if float(np.linalg.norm(correction)) < 1.0e-12:
                break
        residual = delta - shift
        residual, _ = minimum_image_displacements(residual, matrix)
        squared = np.sum((residual @ matrix) ** 2, axis=1)
        best = min(best, float(np.sqrt(squared.mean())))
    return best


def legal_translation_gauge_rmsd(
    batch: "PackedASUBatch",
    normalized_u: Any,
) -> float:
    """RMSD after removing only Hall/ASU-legal common translations."""

    from polarevolve.diffusion.metric import project_translation_quotient_tangent

    value = normalized_u.to(
        device=batch.clean_u.device, dtype=batch.clean_u.dtype
    )
    if batch.batch_size != 1:
        raise ValueError("legal gauge RMSD currently requires one structure")
    if value.shape != batch.clean_u.shape:
        raise ValueError("normalized_u must match packed ASU parameters")
    delta = (value - batch.clean_u + 0.5).remainder(1.0) - 0.5
    quotient_delta = project_translation_quotient_tangent(batch, delta)
    aligned_u = (batch.clean_u + quotient_delta).remainder(1.0)
    generated = batch.expand(aligned_u).detach().cpu().numpy()
    target = batch.expand(batch.clean_u).detach().cpu().numpy()
    return paired_periodic_rmsd(
        batch.lattice[0].detach().cpu().numpy(), generated, target
    )
