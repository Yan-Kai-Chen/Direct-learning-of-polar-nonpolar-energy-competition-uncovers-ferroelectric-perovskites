"""Deterministic polar-child to non-polar-parent channel discovery.

This first FE2-A owner scans a frozen symmetry-tolerance ladder.  It only
admits a higher-symmetry result when the operations detected for the child
are a subset of the candidate operations in the same input basis.  A full
maximal-supergroup graph remains a later asset-backed extension.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from typing import Any, Sequence

import numpy as np
import spglib
from scipy.optimize import linear_sum_assignment

from polarevolve.crystal.contracts import ContractError, HardCondition
from polarevolve.crystal.io import atomic_symbol
from polarevolve.crystal.modes import match_species_permutation
from polarevolve.crystal.periodic import minimum_image_numpy
from polarevolve.crystal.program import OrbitSpec, compile_hard_condition
from polarevolve.crystal.supergroup import (
    POLAR_POINT_GROUPS,
    CompiledSupergroupRelation,
    OccupiedOrbit,
    OrbitMerge,
    check_orbit_merging,
)
from polarevolve.crystal.symmetry import GroupDatabase, WyckoffDatabase


def _matrix_tuple(values: np.ndarray) -> tuple[tuple[float, float, float], ...]:
    return tuple(tuple(float(item) for item in row) for row in values)


@dataclass(frozen=True)
class ParentChannelProposal:
    """One symmetry-legal parent hypothesis found from a polar child."""

    proposal_id: str
    child_hall_number: int
    child_space_group: int
    parent_hall_number: int
    parent_space_group: int
    parent_point_group: str
    detection_tolerance_angstrom: float
    child_operation_count: int
    parent_operation_count: int
    transformation_matrix: tuple[tuple[float, float, float], ...]
    origin_shift: tuple[float, float, float]
    parent_lattice: tuple[tuple[float, float, float], ...]
    parent_fractional: tuple[tuple[float, float, float], ...]
    parent_atomic_numbers: tuple[int, ...]
    hard_condition: HardCondition
    aligned_parent_fractional: tuple[tuple[float, float, float], ...] | None
    alignment_rmsd_angstrom: float | None
    discovery_method: str = "symmetry_tolerance_ascent_v1"
    relation_graph_depth: int | None = None
    relation_group_index: int | None = None
    relation_hall_path: tuple[int, ...] = ()
    common_cell_atom_mapping: tuple[tuple[int, int], ...] = ()
    orbit_merges: tuple[OrbitMerge, ...] = ()
    projection_rmsd_angstrom: float | None = None
    projection_residual_kind: str | None = None
    lattice_metric_strain: float | None = None

    @property
    def supports_direct_od(self) -> bool:
        return self.aligned_parent_fractional is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "fe2_parent_channel_proposal_v2",
            "proposal_id": self.proposal_id,
            "child_hall_number": self.child_hall_number,
            "child_space_group": self.child_space_group,
            "parent_hall_number": self.parent_hall_number,
            "parent_space_group": self.parent_space_group,
            "parent_point_group": self.parent_point_group,
            "detection_tolerance_angstrom": self.detection_tolerance_angstrom,
            "operation_counts": {
                "child": self.child_operation_count,
                "parent": self.parent_operation_count,
            },
            "transformation_matrix": self.transformation_matrix,
            "origin_shift": self.origin_shift,
            "parent_lattice": self.parent_lattice,
            "parent_fractional": self.parent_fractional,
            "parent_atomic_numbers": self.parent_atomic_numbers,
            "hard_condition": self.hard_condition.to_dict(),
            "aligned_parent_fractional": self.aligned_parent_fractional,
            "alignment_rmsd_angstrom": self.alignment_rmsd_angstrom,
            "supports_direct_od": self.supports_direct_od,
            "discovery_method": self.discovery_method,
            "relation": (
                None
                if self.relation_graph_depth is None
                else {
                    "graph_depth": self.relation_graph_depth,
                    "group_index": self.relation_group_index,
                    "hall_path": list(self.relation_hall_path),
                }
            ),
            "common_cell_atom_mapping": [
                list(pair) for pair in self.common_cell_atom_mapping
            ],
            "orbit_merges": [
                {
                    "species": row.species,
                    "child_letters": list(row.child_letters),
                    "parent_letter": row.parent_letter,
                    "child_atom_count": row.child_atom_count,
                    "parent_atom_count": row.parent_atom_count,
                }
                for row in self.orbit_merges
            ],
            "projection_rmsd_angstrom": self.projection_rmsd_angstrom,
            "projection_residual_kind": self.projection_residual_kind,
            "lattice_metric_strain": self.lattice_metric_strain,
        }


def _cell(
    lattice: Sequence[Sequence[float]],
    fractional: Sequence[Sequence[float]],
    atomic_numbers: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lattice_array = np.asarray(lattice, dtype=np.float64)
    fractional_array = np.asarray(fractional, dtype=np.float64)
    numbers = np.asarray(atomic_numbers, dtype=np.int64)
    if lattice_array.shape != (3, 3) or not np.isfinite(lattice_array).all():
        raise ContractError("parent search lattice must be a finite 3x3 matrix")
    if np.linalg.det(lattice_array) <= 1.0e-12:
        raise ContractError("parent search lattice must have positive volume")
    if fractional_array.ndim != 2 or fractional_array.shape[1:] != (3,):
        raise ContractError("parent search coordinates must have shape [N,3]")
    if len(numbers) != len(fractional_array) or not len(numbers):
        raise ContractError("parent search atom types must align with coordinates")
    if not np.isfinite(fractional_array).all() or np.any(numbers <= 0):
        raise ContractError("parent search structure contains invalid values")
    return lattice_array, fractional_array % 1.0, numbers


def _operation_subset(child: Any, parent: Any, *, tolerance: float = 1.0e-6) -> bool:
    parent_ops = list(zip(parent.rotations, parent.translations))
    for child_rotation, child_translation in zip(child.rotations, child.translations):
        found = False
        for parent_rotation, parent_translation in parent_ops:
            if not np.array_equal(child_rotation, parent_rotation):
                continue
            delta = np.asarray(child_translation) - np.asarray(parent_translation)
            delta -= np.round(delta)
            if float(np.max(np.abs(delta))) <= tolerance:
                found = True
                break
        if not found:
            return False
    return True


def _orbit_specs(dataset: Any, *, expected_hall_number: int) -> tuple[OrbitSpec, ...]:
    numbers = np.asarray(dataset.std_types, dtype=np.int64)
    positions = np.asarray(dataset.std_positions, dtype=np.float64)
    standardized = spglib.get_symmetry_dataset(
        (np.asarray(dataset.std_lattice), positions, numbers), symprec=1.0e-5
    )
    if standardized is None:
        raise ContractError("candidate parent standard cell has no symmetry dataset")
    if int(standardized.hall_number) != expected_hall_number:
        raise ContractError("candidate parent standard cell changed Hall setting")
    occurrences: dict[tuple[str, str], int] = {}
    specs: list[OrbitSpec] = []
    equivalents = np.asarray(standardized.equivalent_atoms, dtype=np.int64)
    wyckoffs = tuple(str(value).lower() for value in standardized.wyckoffs)
    for representative in sorted(set(equivalents.tolist())):
        indices = np.flatnonzero(equivalents == representative)
        species = set(numbers[indices].tolist())
        letters = {wyckoffs[index] for index in indices}
        if len(species) != 1 or len(letters) != 1:
            raise ContractError("candidate parent orbit changes species or Wyckoff letter")
        element = atomic_symbol(species.pop())
        letter = letters.pop()
        key = (element, letter)
        occurrences[key] = occurrences.get(key, 0) + 1
        specs.append(OrbitSpec(element, letter, occurrences[key]))
    return tuple(specs)


def _same_cell_alignment(
    dataset: Any,
    child_fractional: np.ndarray,
    child_numbers: np.ndarray,
    child_lattice: np.ndarray,
) -> tuple[tuple[tuple[float, float, float], ...] | None, float | None]:
    parent = np.asarray(dataset.std_positions, dtype=np.float64)
    parent_numbers = np.asarray(dataset.std_types, dtype=np.int64)
    transform = np.asarray(dataset.transformation_matrix, dtype=np.float64)
    if (
        parent.shape != child_fractional.shape
        or sorted(parent_numbers.tolist()) != sorted(child_numbers.tolist())
        or not np.allclose(transform, np.eye(3), atol=1.0e-8, rtol=0.0)
    ):
        return None, None
    aligned = np.empty_like(child_fractional)
    squared_residuals: list[float] = []
    for number in np.unique(child_numbers):
        child_indices = np.flatnonzero(child_numbers == number)
        parent_indices = np.flatnonzero(parent_numbers == number)
        delta = child_fractional[child_indices, None, :] - parent[None, parent_indices, :]
        delta = minimum_image_numpy(delta, child_lattice)
        cartesian = delta @ child_lattice
        cost = np.sum(cartesian * cartesian, axis=-1)
        rows, columns = linear_sum_assignment(cost)
        aligned[child_indices[rows]] = parent[parent_indices[columns]]
        squared_residuals.extend(cost[rows, columns].tolist())
    rmsd = math.sqrt(float(np.mean(squared_residuals)))
    return _matrix_tuple(aligned), rmsd


def _determinant_fraction(
    matrix: tuple[tuple[Fraction, ...], ...],
) -> Fraction:
    a, b, c = matrix
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


class ParentTranslationAtomCountError(ContractError):
    """A valid translation embedding cannot preserve the species inventory."""

    def __init__(self, details: dict[str, Any]) -> None:
        self.details = details
        super().__init__(
            "parent_translation_atom_count_incompatible: "
            + ",".join(details["incompatible_elements"])
        )


@lru_cache(maxsize=530)
def _hall_translations(hall: int) -> frozenset[tuple[Fraction, ...]]:
    operations = spglib.get_symmetry_from_database(hall)
    if operations is None:
        raise ContractError("parent relation Hall operations are unavailable")
    return frozenset(
        tuple(Fraction(float(value)).limit_denominator(24) % 1 for value in shift)
        for rotation, shift in zip(operations["rotations"], operations["translations"])
        if np.array_equal(rotation, np.eye(3, dtype=int))
    )


def relation_atom_count_check(
    numbers: np.ndarray, relation: CompiledSupergroupRelation
) -> dict[str, Any]:
    """Check a witness in its declared Hall cells, before any coordinate folding."""

    transform = relation.transform_parent_from_child
    determinant = _determinant_fraction(transform)
    if determinant <= 0:
        raise ContractError("parent relation transform must have positive determinant")
    child_translations = _hall_translations(relation.child_hall)
    parent_translations = _hall_translations(relation.parent_hall)
    # T maps every child lattice generator into the parent's full translation lattice.
    generators = child_translations | {(1, 0, 0), (0, 1, 0), (0, 0, 1)}
    for generator in generators:
        mapped = tuple(
            sum(transform[row][column] * generator[column] for column in range(3)) % 1
            for row in range(3)
        )
        if mapped not in parent_translations:
            raise ContractError("parent relation does not contain the child translation lattice")
    child_centering, parent_centering = len(child_translations), len(parent_translations)
    translation_index = determinant * Fraction(parent_centering, child_centering)
    if translation_index.denominator != 1 or translation_index < 1:
        raise ContractError("parent relation translation index must be a positive integer")
    counts = {atomic_symbol(int(species)): count for species, count in sorted(Counter(numbers).items())}
    if any(count % child_centering for count in counts.values()):
        raise ContractError("child species counts do not close the declared Hall cell")
    primitive_counts = {
        element: Fraction(count, 1) / (determinant * parent_centering)
        for element, count in counts.items()
    }
    details = {
        "schema_version": "parent_translation_atom_count_v1",
        "relation": relation.to_dict(),
        "determinant": str(determinant),
        "child_centering": child_centering,
        "parent_centering": parent_centering,
        "translation_index": int(translation_index),
        "species_counts": counts,
        "expected_parent_primitive_counts": {
            element: str(count) for element, count in primitive_counts.items()
        },
        "incompatible_elements": sorted(
            element for element, count in primitive_counts.items() if count.denominator != 1
        ),
    }
    if details["incompatible_elements"]:
        raise ParentTranslationAtomCountError(details)
    details["expected_parent_cell_counts"] = {
        element: int(count * parent_centering) for element, count in primitive_counts.items()
    }
    return details


def relation_common_cell(
    lattice: np.ndarray,
    fractional: np.ndarray,
    numbers: np.ndarray,
    relation: CompiledSupergroupRelation,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Express a periodic child structure in the relation's parent cell."""

    accounting = relation_atom_count_check(numbers, relation)
    expected_species_counts = accounting["expected_parent_cell_counts"]
    expected_count = sum(expected_species_counts.values())
    transform_exact = relation.transform_parent_from_child

    denominator = 1
    for row in transform_exact:
        for value in row:
            denominator = math.lcm(denominator, value.denominator)
    if denominator > 24:
        raise ContractError("parent relation translation quotient is unbounded")

    shifts: list[tuple[Fraction, Fraction, Fraction]] = []
    for integer in itertools.product(range(denominator), repeat=3):
        shift = tuple(
            sum(
                transform_exact[row][column] * integer[column]
                for column in range(3)
            )
            % 1
            for row in range(3)
        )
        if shift not in shifts:
            shifts.append(shift)

    transform = np.asarray(
        [[float(value) for value in row] for row in transform_exact],
        dtype=np.float64,
    )
    origin = np.asarray(
        [float(value) for value in relation.origin_parent_from_child],
        dtype=np.float64,
    )
    parent_lattice = np.linalg.inv(transform).T @ lattice
    parent_fractional: list[np.ndarray] = []
    parent_numbers: list[int] = []
    for shift_exact in shifts:
        shift = np.asarray([float(value) for value in shift_exact], dtype=np.float64)
        transformed = (fractional @ transform.T + origin + shift) % 1.0
        for coordinate, species in zip(transformed, numbers):
            duplicate = False
            for previous, previous_species in zip(parent_fractional, parent_numbers):
                if int(species) != previous_species:
                    continue
                delta = coordinate - previous
                delta -= np.round(delta)
                if float(np.max(np.abs(delta))) <= 1.0e-7:
                    duplicate = True
                    break
            if not duplicate:
                parent_fractional.append(coordinate)
                parent_numbers.append(int(species))
    fold_rmsd = 0.0
    if len(parent_fractional) > expected_count:
        folded_fractional: list[np.ndarray] = []
        folded_numbers: list[int] = []
        squared_residuals: list[float] = []
        for species in sorted(set(parent_numbers)):
            coordinates = [
                coordinate
                for coordinate, value in zip(parent_fractional, parent_numbers)
                if value == species
            ]
            target_count = expected_species_counts[atomic_symbol(species)]
            clusters = [[coordinate] for coordinate in coordinates]
            while len(clusters) > target_count:
                centers = []
                for cluster in clusters:
                    anchor = cluster[0]
                    unwrapped = [
                        anchor + minimum_image_numpy(value - anchor, parent_lattice)
                        for value in cluster
                    ]
                    centers.append(np.mean(unwrapped, axis=0) % 1.0)
                best = None
                for left in range(len(clusters)):
                    for right in range(left + 1, len(clusters)):
                        delta = centers[left] - centers[right]
                        delta = minimum_image_numpy(delta, parent_lattice)
                        distance = float(np.sum((delta @ parent_lattice) ** 2))
                        key = (distance, left, right)
                        if best is None or key < best:
                            best = key
                if best is None:
                    raise ContractError("parent relation cannot fold periodic sites")
                _, left, right = best
                clusters[left].extend(clusters.pop(right))
            for cluster in clusters:
                anchor = cluster[0]
                unwrapped = [
                    anchor + minimum_image_numpy(value - anchor, parent_lattice)
                    for value in cluster
                ]
                center = np.mean(unwrapped, axis=0) % 1.0
                folded_fractional.append(center)
                folded_numbers.append(species)
                for value in cluster:
                    delta = value - center
                    delta = minimum_image_numpy(delta, parent_lattice)
                    squared_residuals.append(
                        float(np.sum((delta @ parent_lattice) ** 2))
                    )
        parent_fractional = folded_fractional
        parent_numbers = folded_numbers
        fold_rmsd = math.sqrt(float(np.mean(squared_residuals)))
    actual_counts = {atomic_symbol(species): count for species, count in Counter(parent_numbers).items()}
    if actual_counts != expected_species_counts:
        raise ContractError(
            "parent relation common-cell atom count does not match its determinant: "
            f"found={actual_counts} expected={expected_species_counts} "
            f"determinant={accounting['determinant']}"
        )
    return (
        parent_lattice,
        np.asarray(parent_fractional, dtype=np.float64),
        np.asarray(parent_numbers, dtype=np.int64),
        fold_rmsd,
    )


def _occupied_orbits(dataset: Any, numbers: np.ndarray) -> tuple[OccupiedOrbit, ...]:
    equivalents = np.asarray(dataset.equivalent_atoms, dtype=np.int64)
    letters = tuple(str(value).lower() for value in dataset.wyckoffs)
    if len(equivalents) != len(numbers) or len(letters) != len(numbers):
        raise ContractError("symmetry orbit labels do not align with the common cell")
    orbits = []
    for representative in sorted(set(equivalents.tolist())):
        indices = np.flatnonzero(equivalents == representative)
        species = {int(numbers[index]) for index in indices}
        orbit_letters = {letters[index] for index in indices}
        if len(species) != 1 or len(orbit_letters) != 1:
            raise ContractError("occupied orbit changes species or Wyckoff letter")
        orbits.append(
            OccupiedOrbit(
                species=species.pop(),
                letter=orbit_letters.pop(),
                atom_indices=frozenset(int(index) for index in indices),
            )
        )
    return tuple(orbits)


def project_lattice_metric(lattice: np.ndarray, rotations: np.ndarray) -> np.ndarray:
    metric = lattice @ lattice.T
    projected_metric = np.mean(
        [rotation.T @ metric @ rotation for rotation in rotations], axis=0
    )
    projected_metric = 0.5 * (projected_metric + projected_metric.T)
    eigenvalues, eigenvectors = np.linalg.eigh(projected_metric)
    if float(np.min(eigenvalues)) <= 1.0e-10:
        raise ContractError("target group produces a non-positive lattice metric")
    square_root = (
        eigenvectors * np.sqrt(eigenvalues)[None, :]
    ) @ eigenvectors.T
    left, _, right = np.linalg.svd(square_root.T @ lattice)
    orientation = left @ right
    if np.linalg.det(orientation) < 0:
        left[:, -1] *= -1.0
        orientation = left @ right
    return square_root @ orientation


def _operation_residual_rmsd(
    lattice: np.ndarray,
    fractional: np.ndarray,
    numbers: np.ndarray,
    rotations: np.ndarray,
    translations: np.ndarray,
) -> float:
    squared: list[float] = []
    for rotation, translation in zip(rotations, translations):
        transformed = fractional @ rotation.T + translation
        permutation, _ = match_species_permutation(
            transformed, fractional, numbers, lattice
        )
        delta = transformed - fractional[permutation]
        delta = minimum_image_numpy(delta, lattice)
        squared.extend(np.sum((delta @ lattice) ** 2, axis=-1).tolist())
    return math.sqrt(float(np.mean(squared)))


def _project_positions_to_group(
    lattice: np.ndarray,
    fractional: np.ndarray,
    numbers: np.ndarray,
    rotations: np.ndarray,
    translations: np.ndarray,
    *,
    iterations: int,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    projected_lattice = project_lattice_metric(lattice, rotations)
    current = fractional.copy()
    atom_count = len(current)
    for _ in range(iterations):
        constraint_rows: list[np.ndarray] = []
        targets: list[float] = []
        for rotation, translation in zip(rotations, translations):
            transformed = current @ rotation.T + translation
            permutation, _ = match_species_permutation(
                transformed, current, numbers, projected_lattice
            )
            for source, target in enumerate(permutation):
                delta = transformed[source] - current[target]
                image = np.rint(delta - minimum_image_numpy(delta, projected_lattice))
                for dimension in range(3):
                    row = np.zeros(3 * atom_count, dtype=np.float64)
                    row[3 * target + dimension] = 1.0
                    row[3 * source : 3 * source + 3] -= rotation[dimension]
                    constraint_rows.append(row)
                    targets.append(float(translation[dimension] - image[dimension]))
        constraints = np.asarray(constraint_rows, dtype=np.float64)
        target_values = np.asarray(targets, dtype=np.float64)
        flattened = current.reshape(-1)
        correction = np.linalg.lstsq(
            constraints,
            target_values - constraints @ flattened,
            rcond=1.0e-10,
        )[0]
        current = (flattened + correction).reshape(atom_count, 3) % 1.0

    delta = fractional - current
    delta = minimum_image_numpy(delta, projected_lattice)
    projection_rmsd = math.sqrt(
        float(np.mean(np.sum((delta @ projected_lattice) ** 2, axis=-1)))
    )
    metric = lattice @ lattice.T
    projected_metric = projected_lattice @ projected_lattice.T
    lattice_strain = float(
        np.linalg.norm(projected_metric - metric) / np.linalg.norm(metric)
    )
    return projected_lattice, current, projection_rmsd, lattice_strain


def discover_parent_channels(
    *,
    lattice: Sequence[Sequence[float]],
    fractional: Sequence[Sequence[float]],
    atomic_numbers: Sequence[int],
    group_database: GroupDatabase,
    wyckoff_database: WyckoffDatabase,
    child_symmetry_tolerance_angstrom: float = 1.0e-3,
    parent_tolerances_angstrom: Sequence[float] = (0.01, 0.03, 0.05, 0.1, 0.2),
) -> tuple[ParentChannelProposal, ...]:
    """Discover non-polar higher-symmetry channels on a fixed tolerance ladder."""

    cell = _cell(lattice, fractional, atomic_numbers)
    lattice_array, fractional_array, numbers = cell
    tolerances = tuple(float(value) for value in parent_tolerances_angstrom)
    if child_symmetry_tolerance_angstrom <= 0.0 or not tolerances:
        raise ContractError("symmetry tolerances must be positive")
    if any(value <= child_symmetry_tolerance_angstrom for value in tolerances):
        raise ContractError("parent tolerances must exceed the child tolerance")
    if any(right <= left for left, right in zip(tolerances, tolerances[1:])):
        raise ContractError("parent tolerances must be strictly increasing")
    child = spglib.get_symmetry_dataset(cell, symprec=child_symmetry_tolerance_angstrom)
    if child is None:
        raise ContractError("polar child has no symmetry dataset")
    child_setting = group_database.setting(int(child.hall_number))
    if child_setting.point_group not in POLAR_POINT_GROUPS:
        raise ContractError("parent search requires a polar child point group")

    proposals: list[ParentChannelProposal] = []
    seen_halls: set[int] = set()
    for tolerance in tolerances:
        candidate = spglib.get_symmetry_dataset(cell, symprec=tolerance)
        if candidate is None or int(candidate.hall_number) in seen_halls:
            continue
        parent_setting = group_database.setting(int(candidate.hall_number))
        if (
            parent_setting.point_group in POLAR_POINT_GROUPS
            or len(candidate.rotations) <= len(child.rotations)
            or not _operation_subset(child, candidate)
        ):
            continue
        specs = _orbit_specs(
            candidate, expected_hall_number=int(candidate.hall_number)
        )
        compiled = compile_hard_condition(
            condition_id=f"fe2-parent-hall-{candidate.hall_number}",
            hall_number=int(candidate.hall_number),
            orbit_specs=specs,
            base_cell_representation="conventional",
            group_database=group_database,
            wyckoff_database=wyckoff_database,
        )
        parent_lattice = np.asarray(candidate.std_lattice, dtype=np.float64)
        parent_fractional = np.asarray(candidate.std_positions, dtype=np.float64) % 1.0
        parent_numbers = tuple(int(value) for value in candidate.std_types)
        if compiled.hard.group_num_atoms != len(parent_numbers):
            raise ContractError("candidate parent program does not close its standard cell")
        aligned, alignment_rmsd = _same_cell_alignment(
            candidate, fractional_array, numbers, lattice_array
        )
        identity_payload = {
            "child_hall": int(child.hall_number),
            "parent_hall": int(candidate.hall_number),
            "tolerance": tolerance,
            "program": [orbit.site_id for orbit in compiled.hard.wyckoff_orbits],
        }
        digest = hashlib.sha256(
            json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:16]
        proposals.append(
            ParentChannelProposal(
                proposal_id=f"fe2-parent-{digest}",
                child_hall_number=int(child.hall_number),
                child_space_group=int(child.number),
                parent_hall_number=int(candidate.hall_number),
                parent_space_group=int(candidate.number),
                parent_point_group=parent_setting.point_group,
                detection_tolerance_angstrom=tolerance,
                child_operation_count=len(child.rotations),
                parent_operation_count=len(candidate.rotations),
                transformation_matrix=_matrix_tuple(np.asarray(candidate.transformation_matrix)),
                origin_shift=tuple(float(value) for value in candidate.origin_shift),
                parent_lattice=_matrix_tuple(parent_lattice),
                parent_fractional=_matrix_tuple(parent_fractional),
                parent_atomic_numbers=parent_numbers,
                hard_condition=compiled.hard,
                aligned_parent_fractional=aligned,
                alignment_rmsd_angstrom=alignment_rmsd,
            )
        )
        seen_halls.add(int(candidate.hall_number))
    return tuple(proposals)


def compile_relation_parent_channel(
    *,
    lattice: Sequence[Sequence[float]],
    fractional: Sequence[Sequence[float]],
    atomic_numbers: Sequence[int],
    relation: CompiledSupergroupRelation,
    group_database: GroupDatabase,
    wyckoff_database: WyckoffDatabase,
    child_symmetry_tolerance_angstrom: float = 1.0e-3,
    standardization_tolerances_angstrom: Sequence[float] = (
        0.01,
        0.02,
        0.03,
        0.05,
        0.08,
        0.1,
        0.15,
        0.2,
        0.3,
        0.4,
        0.6,
        0.8,
        1.0,
        1.2,
        1.5,
    ),
    maximum_projection_rmsd_angstrom: float = 1.5,
    projection_iterations: int = 4,
    exact_tolerance_angstrom: float = 1.0e-5,
) -> ParentChannelProposal:
    """Compile one exact group relation into a coordinate-bearing parent channel.

    Gold parent labels are not inputs to this function. The requested parent is
    carried by the independently compiled relation. Every returned structure is
    re-detected at ``exact_tolerance_angstrom`` and has an explicit common-cell
    orbit-merge proof.
    """

    source = _cell(lattice, fractional, atomic_numbers)
    tolerances = tuple(float(value) for value in standardization_tolerances_angstrom)
    if (
        child_symmetry_tolerance_angstrom <= 0.0
        or exact_tolerance_angstrom <= 0.0
        or maximum_projection_rmsd_angstrom <= 0.0
        or projection_iterations < 1
    ):
        raise ContractError("relation projection tolerances and iterations must be positive")
    if not tolerances or any(value <= 0.0 for value in tolerances):
        raise ContractError("relation standardization tolerances must be positive")
    if any(right <= left for left, right in zip(tolerances, tolerances[1:])):
        raise ContractError("relation standardization tolerances must increase")

    standardized = spglib.standardize_cell(
        source,
        to_primitive=False,
        no_idealize=True,
        symprec=child_symmetry_tolerance_angstrom,
    )
    if standardized is None:
        raise ContractError("polar child cannot be standardized for relation projection")
    child_lattice, child_fractional, child_numbers = (
        np.asarray(standardized[0], dtype=np.float64),
        np.asarray(standardized[1], dtype=np.float64) % 1.0,
        np.asarray(standardized[2], dtype=np.int64),
    )
    child_dataset = spglib.get_symmetry_dataset(
        (child_lattice, child_fractional, child_numbers),
        symprec=child_symmetry_tolerance_angstrom,
    )
    if child_dataset is None or int(child_dataset.hall_number) != relation.child_hall:
        raise ContractError("polar child Hall setting does not match the compiled relation")

    (
        common_lattice,
        common_fractional,
        common_numbers,
        cell_fold_rmsd,
    ) = relation_common_cell(child_lattice, child_fractional, child_numbers, relation)
    common_child = spglib.get_symmetry_dataset(
        (common_lattice, common_fractional, common_numbers),
        symprec=child_symmetry_tolerance_angstrom,
    )
    if common_child is None:
        raise ContractError("rebased polar child has no symmetry dataset")

    parent_dataset = None
    projection_rmsd = None
    lattice_strain = None
    method = ""
    for tolerance in tolerances:
        candidate = spglib.get_symmetry_dataset(
            (common_lattice, common_fractional, common_numbers), symprec=tolerance
        )
        if candidate is None or int(candidate.number) != relation.parent_sg:
            continue
        exact = spglib.get_symmetry_dataset(
            (
                np.asarray(candidate.std_lattice, dtype=np.float64),
                np.asarray(candidate.std_positions, dtype=np.float64),
                np.asarray(candidate.std_types, dtype=np.int64),
            ),
            symprec=exact_tolerance_angstrom,
        )
        if (
            exact is None
            or int(exact.number) != relation.parent_sg
        ):
            continue
        projection_rmsd = math.hypot(
            cell_fold_rmsd,
            _operation_residual_rmsd(
            common_lattice,
            common_fractional,
            common_numbers,
            np.asarray(candidate.rotations, dtype=np.float64),
            np.asarray(candidate.translations, dtype=np.float64),
            ),
        )
        projected_lattice = project_lattice_metric(
            common_lattice, np.asarray(candidate.rotations, dtype=np.float64)
        )
        lattice_metric = common_lattice @ common_lattice.T
        projected_metric = projected_lattice @ projected_lattice.T
        lattice_strain = float(
            np.linalg.norm(projected_metric - lattice_metric)
            / np.linalg.norm(lattice_metric)
        )
        parent_dataset = candidate
        method = "relation_target_standardization_v1"
        break

    common_parent = parent_dataset
    if parent_dataset is None:
        operations = spglib.get_symmetry_from_database(relation.parent_hall)
        if operations is None:
            raise ContractError("compiled relation parent Hall has no operation table")
        projected_lattice, projected_fractional, group_rmsd, lattice_strain = (
            _project_positions_to_group(
                common_lattice,
                common_fractional,
                common_numbers,
                np.asarray(operations["rotations"], dtype=np.float64),
                np.asarray(operations["translations"], dtype=np.float64),
                iterations=projection_iterations,
            )
        )
        projection_rmsd = math.hypot(cell_fold_rmsd, group_rmsd)
        common_parent = spglib.get_symmetry_dataset(
            (projected_lattice, projected_fractional, common_numbers),
            symprec=exact_tolerance_angstrom,
        )
        if common_parent is None or int(common_parent.number) != relation.parent_sg:
            detected = (
                "none"
                if common_parent is None
                else f"SG{int(common_parent.number)}/Hall{int(common_parent.hall_number)}"
            )
            raise ContractError(
                f"operation-constrained projection missed the target SG: {detected}"
            )
        parent_dataset = common_parent
        method = "relation_group_projection_v1"

    if projection_rmsd is None or lattice_strain is None or common_parent is None:
        raise ContractError("relation projection did not produce complete diagnostics")
    if not math.isfinite(projection_rmsd) or not math.isfinite(lattice_strain):
        raise ContractError("relation projection diagnostics are non-finite")
    if projection_rmsd > maximum_projection_rmsd_angstrom:
        raise ContractError("relation projection exceeds the displacement budget")

    atom_mapping = {index: index for index in range(len(common_numbers))}
    orbit_merges = check_orbit_merging(
        child_orbits=_occupied_orbits(common_child, common_numbers),
        parent_orbits=_occupied_orbits(common_parent, common_numbers),
        atom_mapping=atom_mapping,
    )
    parent_hall = int(parent_dataset.hall_number)
    specs = _orbit_specs(parent_dataset, expected_hall_number=parent_hall)
    compiled = compile_hard_condition(
        condition_id=(
            f"fe2-relation-hall-{relation.child_hall}-{parent_hall}"
        ),
        hall_number=parent_hall,
        orbit_specs=specs,
        base_cell_representation="conventional",
        group_database=group_database,
        wyckoff_database=wyckoff_database,
    )
    parent_lattice = np.asarray(parent_dataset.std_lattice, dtype=np.float64)
    parent_fractional = np.asarray(
        parent_dataset.std_positions, dtype=np.float64
    ) % 1.0
    parent_numbers = tuple(int(value) for value in parent_dataset.std_types)
    if compiled.hard.group_num_atoms != len(parent_numbers):
        raise ContractError("relation parent program does not close its standard cell")

    identity_payload = {
        "child_hall": relation.child_hall,
        "relation_parent_hall": relation.parent_hall,
        "standard_parent_hall": parent_hall,
        "hall_path": [relation.child_hall]
        + [edge.parent_hall for edge in relation.edges],
        "method": method,
        "program": [orbit.site_id for orbit in compiled.hard.wyckoff_orbits],
    }
    digest = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    child_operations = spglib.get_symmetry_from_database(relation.child_hall)
    parent_operations = spglib.get_symmetry_from_database(parent_hall)
    if child_operations is None or parent_operations is None:
        raise ContractError("relation endpoint operation table is unavailable")
    setting_transform = np.asarray(
        parent_dataset.transformation_matrix, dtype=np.float64
    )
    setting_origin = np.asarray(parent_dataset.origin_shift, dtype=np.float64)
    relation_transform = np.asarray(
        [
            [float(value) for value in row]
            for row in relation.transform_parent_from_child
        ],
        dtype=np.float64,
    )
    relation_origin = np.asarray(
        [float(value) for value in relation.origin_parent_from_child],
        dtype=np.float64,
    )
    standard_transform = setting_transform @ relation_transform
    standard_origin = setting_transform @ relation_origin + setting_origin
    return ParentChannelProposal(
        proposal_id=f"fe2-relation-{digest}",
        child_hall_number=relation.child_hall,
        child_space_group=relation.child_sg,
        parent_hall_number=parent_hall,
        parent_space_group=relation.parent_sg,
        parent_point_group=relation.parent_point_group,
        detection_tolerance_angstrom=exact_tolerance_angstrom,
        child_operation_count=len(child_operations["rotations"]),
        parent_operation_count=len(parent_operations["rotations"]),
        transformation_matrix=_matrix_tuple(standard_transform),
        origin_shift=tuple(float(value) for value in standard_origin),
        parent_lattice=_matrix_tuple(parent_lattice),
        parent_fractional=_matrix_tuple(parent_fractional),
        parent_atomic_numbers=parent_numbers,
        hard_condition=compiled.hard,
        aligned_parent_fractional=None,
        alignment_rmsd_angstrom=None,
        discovery_method=method,
        relation_graph_depth=relation.graph_depth,
        relation_group_index=relation.group_index,
        relation_hall_path=(relation.child_hall,)
        + tuple(edge.parent_hall for edge in relation.edges),
        common_cell_atom_mapping=tuple(atom_mapping.items()),
        orbit_merges=orbit_merges,
        projection_rmsd_angstrom=projection_rmsd,
        projection_residual_kind=(
            "operation_mismatch_plus_cell_fold_rms"
            if method == "relation_target_standardization_v1"
            else "coordinate_projection_plus_cell_fold_rms"
        ),
        lattice_metric_strain=lattice_strain,
    )


__all__ = [
    "POLAR_POINT_GROUPS",
    "ParentChannelProposal",
    "ParentTranslationAtomCountError",
    "compile_relation_parent_channel",
    "discover_parent_channels",
    "project_lattice_metric",
    "relation_atom_count_check",
    "relation_common_cell",
]
