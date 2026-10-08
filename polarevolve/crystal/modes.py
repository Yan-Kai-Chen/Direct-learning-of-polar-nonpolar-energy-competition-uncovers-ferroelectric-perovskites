"""Finite-group displacement projectors for symmetry-breaking crystal modes.

This module owns only the representation-space mathematics.  Relation search,
Wyckoff splitting, ASU pullback and task-specific OD readouts remain with their
existing owners.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from polarevolve.crystal.contracts import ContractError
from polarevolve.crystal.periodic import minimum_image_numpy


@dataclass(frozen=True)
class AtomicDisplacementRepresentation:
    """Orthogonal Cartesian displacement operators in one common cell."""

    operators: np.ndarray
    permutations: np.ndarray
    maximum_mapping_residual_angstrom: float
    maximum_orthogonality_residual: float


@dataclass(frozen=True)
class SymmetrizedPositions:
    """Positions projected onto the exact fixed set of a finite space group."""

    fractional: np.ndarray
    maximum_displacement_angstrom: float
    maximum_mapping_residual_angstrom: float


@dataclass(frozen=True)
class SymmetryModeDecomposition:
    """Projectors and deterministic bases for one G -> H mode decomposition."""

    translation_projector: np.ndarray
    parent_invariant_projector: np.ndarray
    subgroup_invariant_projector: np.ndarray
    symmetry_breaking_projector: np.ndarray
    primary_projector: np.ndarray
    secondary_projector: np.ndarray
    primary_basis: np.ndarray
    secondary_basis: np.ndarray

    @property
    def symmetry_breaking_dimension(self) -> int:
        return int(round(float(np.trace(self.symmetry_breaking_projector))))

    @property
    def primary_dimension(self) -> int:
        return self.primary_basis.shape[1]

    @property
    def secondary_dimension(self) -> int:
        return self.secondary_basis.shape[1]


def _finite_array(value: object, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ContractError(f"{name} contains non-finite values")
    return array


def _validate_orthogonal_operators(operators: object, *, tolerance: float) -> np.ndarray:
    array = _finite_array(operators, name="representation operators")
    if array.ndim != 3 or array.shape[1] != array.shape[2] or not len(array):
        raise ContractError("representation operators must have shape [G,N,N]")
    identity = np.eye(array.shape[1], dtype=np.float64)
    residual = max(
        float(np.linalg.norm(operator.T @ operator - identity, ord=2)) for operator in array
    )
    if residual > tolerance:
        raise ContractError(f"representation operators are not orthogonal: residual={residual:.3e}")
    return array


def _validate_projector(projector: np.ndarray, *, name: str, tolerance: float) -> np.ndarray:
    symmetric = 0.5 * (projector + projector.T)
    symmetry_residual = float(np.linalg.norm(projector - projector.T, ord=2))
    idempotence_residual = float(np.linalg.norm(symmetric @ symmetric - symmetric, ord=2))
    if max(symmetry_residual, idempotence_residual) > tolerance:
        raise ContractError(
            f"{name} is not an orthogonal projector: "
            f"symmetry={symmetry_residual:.3e}, idempotence={idempotence_residual:.3e}"
        )
    return symmetric


def translation_projector(atom_count: int) -> np.ndarray:
    """Return the Cartesian projector onto rigid translations of one cell."""

    if atom_count <= 0:
        raise ContractError("atom count must be positive")
    basis = np.zeros((3 * atom_count, 3), dtype=np.float64)
    scale = 1.0 / np.sqrt(float(atom_count))
    for atom in range(atom_count):
        basis[3 * atom : 3 * atom + 3] = np.eye(3) * scale
    return basis @ basis.T


def projector_basis(projector: object, *, tolerance: float = 1.0e-9) -> np.ndarray:
    """Build a coordinate-ordered orthonormal basis without labelling degeneracies."""

    matrix = _validate_projector(
        _finite_array(projector, name="projector"),
        name="projector",
        tolerance=tolerance,
    )
    vectors: list[np.ndarray] = []
    for column in matrix.T:
        vector = column.copy()
        for previous in vectors:
            vector -= previous * float(previous @ vector)
        norm = float(np.linalg.norm(vector))
        if norm > tolerance:
            vectors.append(vector / norm)
    if not vectors:
        return np.zeros((matrix.shape[0], 0), dtype=np.float64)
    basis = np.stack(vectors, axis=1)
    rank = int(round(float(np.trace(matrix))))
    if basis.shape[1] != rank:
        raise ContractError("projector rank is numerically ambiguous")
    return basis


def reynolds_projector(
    operators: object,
    *,
    quotient_projector: object | None = None,
    tolerance: float = 1.0e-9,
) -> np.ndarray:
    """Average a finite orthogonal representation, optionally in a quotient."""

    representation = _validate_orthogonal_operators(operators, tolerance=tolerance)
    projector = representation.mean(axis=0)
    if quotient_projector is not None:
        quotient = _validate_projector(
            _finite_array(quotient_projector, name="quotient projector"),
            name="quotient projector",
            tolerance=tolerance,
        )
        if quotient.shape != projector.shape:
            raise ContractError("quotient and representation dimensions differ")
        projector = quotient @ projector @ quotient
    return _validate_projector(projector, name="Reynolds average", tolerance=tolerance)


@dataclass(frozen=True)
class RealIsotypicSector:
    """One real-irreducible type: its isotypic projector and irreducible copies."""

    projector: np.ndarray
    copies: tuple[np.ndarray, ...]
    character: np.ndarray
    frobenius_schur_norm: float


def _cluster_eigenspaces(
    values: np.ndarray, vectors: np.ndarray, *, tolerance: float
) -> list[np.ndarray]:
    order = np.argsort(values)
    values, vectors = values[order], vectors[:, order]
    scale = max(1.0, float(np.max(np.abs(values), initial=0.0)))
    groups: list[np.ndarray] = []
    start = 0
    for index in range(1, len(values) + 1):
        if index == len(values) or values[index] - values[index - 1] > tolerance * scale:
            groups.append(vectors[:, start:index])
            start = index
    return groups


class _GroupAction:
    """Apply a finite orthogonal representation to column blocks.

    Atomic displacement representations are block permutations
    ``D[perm[b], b] = B[b]``; they are applied in O(|G| N) per column instead of
    dense O(|G| N^2), which keeps supercell groups tractable. Any other
    orthogonal representation falls back to dense products.
    """

    def __init__(self, operators: np.ndarray) -> None:
        self.dense = operators
        self.permutations: np.ndarray | None = None
        self.blocks: np.ndarray | None = None
        size = operators.shape[1]
        if size % 3:
            return
        atoms = size // 3
        tiled = operators.reshape(len(operators), atoms, 3, atoms, 3)
        weight = np.linalg.norm(tiled, axis=(2, 4))
        permutations = np.argmax(weight, axis=1)
        columns = np.arange(atoms)
        blocks = tiled[np.arange(len(operators))[:, None], permutations, :, columns[None, :], :]
        rebuilt = np.zeros_like(tiled)
        rebuilt[np.arange(len(operators))[:, None], permutations, :, columns[None, :], :] = blocks
        if not np.array_equal(rebuilt, tiled):
            return
        if any(len(np.unique(row)) != atoms for row in permutations):
            return
        self.permutations = permutations
        self.blocks = blocks

    def apply(self, matrix: np.ndarray) -> np.ndarray:
        """Return ``D(g) @ matrix`` for every g, shape [G, N, K]."""

        if self.permutations is None or self.blocks is None:
            return np.matmul(self.dense, matrix)
        count, atoms = self.permutations.shape
        fields = matrix.reshape(atoms, 3, -1)
        moved = np.einsum("gbij,bjk->gbik", self.blocks, fields)
        result = np.empty_like(moved)
        result[np.arange(count)[:, None], self.permutations] = moved
        return result.reshape(count, 3 * atoms, -1)

    def congruence_average(self, symmetric: np.ndarray) -> np.ndarray:
        """Return ``|G|^-1 sum_g D(g) X D(g)^T``."""

        if self.permutations is None or self.blocks is None:
            total = np.zeros_like(symmetric)
            for operator in self.dense:
                total += operator @ symmetric @ operator.T
            return total / float(len(self.dense))
        atoms = self.permutations.shape[1]
        tiled = symmetric.reshape(atoms, 3, atoms, 3)
        total = np.zeros_like(tiled)
        for permutation, block in zip(self.permutations, self.blocks):
            moved = np.einsum("bij,bjdk,dlk->bidl", block, tiled, block, optimize=True)
            total[np.ix_(permutation, range(3), permutation, range(3))] += moved
        return total.reshape(symmetric.shape) / float(len(self.permutations))


def real_isotypic_sectors(
    operators: object,
    *,
    quotient_projector: object,
    seed: int = 20260926,
    tolerance: float = 1.0e-7,
) -> tuple[RealIsotypicSector, ...]:
    """Decompose an orthogonal representation into real-irreducible isotypic sectors.

    Character tables are not required. The eigenspaces of a generic symmetric
    element of the commutant, ``|G|^-1 sum_g D(g) X D(g)^T``, restricted to the
    quotient, are real-irreducible invariant subspaces; copies of one
    irreducible type share their character and are summed into an isotypic
    projector. This covers supercell (folded) wave vectors because pure
    translations are ordinary group elements here. Each copy is checked for
    invariance and for real irreducibility through the Frobenius-Schur norm
    ``|G|^-1 sum chi^2``, which is 1, 2 or 4 for real, complex or quaternionic
    irreducible types. The quotient must commute with the representation.
    """

    representation = _finite_array(operators, name="representation operators")
    if (
        representation.ndim != 3
        or representation.shape[1] != representation.shape[2]
        or not len(representation)
    ):
        raise ContractError("representation operators must have shape [G,N,N]")
    quotient = _validate_projector(
        _finite_array(quotient_projector, name="quotient projector"),
        name="quotient projector",
        tolerance=tolerance,
    )
    if quotient.shape != representation.shape[1:]:
        raise ContractError("quotient and representation dimensions differ")
    action = _GroupAction(representation)
    if action.blocks is not None:
        orthogonality = float(
            np.max(np.abs(np.einsum("gbji,gbjk->gbik", action.blocks, action.blocks) - np.eye(3)))
        )
    else:
        orthogonality = float(
            np.max(
                np.abs(
                    np.matmul(representation.transpose(0, 2, 1), representation)
                    - np.eye(representation.shape[1])
                )
            )
        )
    if orthogonality > tolerance:
        raise ContractError(f"representation operators are not orthogonal: {orthogonality:.3e}")
    space = projector_basis(quotient, tolerance=tolerance)
    if space.shape[1] == 0:
        return ()
    moved_space = action.apply(space)
    commutation = float(np.max(np.abs(moved_space - np.matmul(quotient, moved_space)), initial=0.0))
    if commutation > 1.0e-6:
        raise ContractError(f"quotient is not invariant under the group: {commutation:.3e}")
    generator = np.random.default_rng(seed)
    noise = generator.standard_normal(quotient.shape)
    averaged = action.congruence_average(0.5 * (noise + noise.T))
    commutant = space.T @ averaged @ space
    commutant = 0.5 * (commutant + commutant.T)
    values, vectors = np.linalg.eigh(commutant)
    copies = _cluster_eigenspaces(values, vectors, tolerance=1.0e-6)

    typed: list[tuple[np.ndarray, list[np.ndarray]]] = []
    for copy in copies:
        lifted = space @ copy
        moved = action.apply(lifted)
        blocks = np.matmul(lifted.T, moved)
        leakage = float(np.max(np.abs(moved - np.matmul(lifted, blocks))))
        if leakage > 1.0e-6:
            raise ContractError(f"commutant eigenspace is not invariant: {leakage:.3e}")
        character = np.trace(blocks, axis1=1, axis2=2)
        norm = float(np.mean(character**2))
        if min(abs(norm - value) for value in (1.0, 2.0, 4.0)) > 1.0e-6:
            raise ContractError(
                f"commutant eigenspace is not real-irreducible (Frobenius-Schur {norm:.6f}); "
                "accidental degeneracy"
            )
        for known, members in typed:
            if np.allclose(known, character, atol=1.0e-6):
                members.append(lifted)
                break
        else:
            typed.append((character, [lifted]))

    sectors: list[RealIsotypicSector] = []
    for character, members in typed:
        joined = np.concatenate(members, axis=1)
        sectors.append(
            RealIsotypicSector(
                projector=joined @ joined.T,
                copies=tuple(members),
                character=character,
                frobenius_schur_norm=float(np.mean(character**2)),
            )
        )
    closure = float(np.max(np.abs(sum(item.projector for item in sectors) - quotient)))
    if closure > 1.0e-6:
        raise ContractError(f"real isotypic sectors do not close: {closure:.3e}")
    return tuple(sectors)


def irreducible_matrices(operators: object, copy: object) -> np.ndarray:
    """Real matrices ``C^T D(g) C`` of one irreducible copy with orthonormal columns."""

    basis = _finite_array(copy, name="irreducible copy")
    return np.matmul(basis.T, _GroupAction(_finite_array(operators, name="operators")).apply(basis))


@dataclass(frozen=True)
class WavevectorStar:
    """Wave vectors carried by one real-irreducible copy under pure translations.

    ``wavevectors`` are fractional in the reciprocal basis of
    ``primitive_basis``, whose rows are the primitive translations of the group
    in working-cell fractional coordinates.
    """

    wavevectors: tuple[tuple[float, float, float], ...]
    multiplicities: tuple[int, ...]
    primitive_basis: np.ndarray

    @property
    def is_gamma(self) -> bool:
        return self.wavevectors == ((0.0, 0.0, 0.0),)


def _integer_row_basis(generators: np.ndarray) -> np.ndarray:
    """Triangular basis of the integer lattice spanned by full-rank generator rows."""

    rows = [np.asarray(row, dtype=np.int64) for row in generators]
    basis: list[np.ndarray] = []
    for column in range(3):
        while True:
            active = [row for row in rows if row[column] != 0]
            if len(active) <= 1:
                break
            pivot = min(active, key=lambda row: abs(int(row[column])))
            rows = [
                row if row is pivot else row - (int(row[column]) // int(pivot[column])) * pivot
                for row in rows
            ]
        if not active:
            raise ContractError("translation generators are rank deficient")
        basis.append(active[0])
        rows = [row for row in rows if row is not active[0]]
    return np.stack(basis)


def pure_translation_wavevectors(
    character: object,
    rotations: object,
    translations: object,
    *,
    tolerance: float = 1.0e-6,
) -> WavevectorStar:
    """Resolve the wave-vector star of a real-irreducible copy.

    ``rotations``/``translations`` are the fractional operations of the working
    cell and ``character`` is the copy's character on them. The pure
    translations ``t`` form a finite abelian group ``T``; its characters are
    ``exp(2 pi i q.t)`` for integer ``q`` in the working-cell reciprocal basis,
    and each multiplicity follows from character orthogonality. Wave vectors
    are reported in the reciprocal basis of the primitive translation lattice
    generated by the working cell and ``T``.
    """

    chi = np.asarray(character, dtype=np.float64)
    rotation = np.asarray(rotations, dtype=np.int64)
    shift = np.asarray(translations, dtype=np.float64)
    if (
        rotation.ndim != 3
        or rotation.shape[1:] != (3, 3)
        or shift.shape != (len(rotation), 3)
        or chi.shape != (len(rotation),)
    ):
        raise ContractError("wave-vector inputs have incompatible shapes")
    pure = np.flatnonzero(np.all(rotation == np.eye(3, dtype=np.int64), axis=(1, 2)))
    vectors = shift[pure] - np.round(shift[pure])
    lengths = np.linalg.norm(vectors, axis=1) if len(pure) else np.ones(1)
    if float(lengths.min()) > tolerance:
        raise ContractError("operation list has no identity")
    order = len(pure)
    scaled = np.round(vectors * order)
    if float(np.max(np.abs(scaled - vectors * order))) > tolerance * order:
        raise ContractError(f"pure translations are not in the 1/{order} lattice")
    generators = np.concatenate([order * np.eye(3, dtype=np.int64), scaled.astype(np.int64)])
    primitive = _integer_row_basis(generators) / float(order)
    if round(abs(float(np.linalg.det(primitive))) * order) != 1:
        raise ContractError(f"pure translations do not form a group of order {order}")
    dimension = float(chi[pure][int(np.argmin(lengths))])

    characters: dict[tuple[int, ...], np.ndarray] = {}
    for q in np.ndindex(order, order, order):
        phase = np.round(np.mod(vectors @ np.asarray(q, dtype=np.float64), 1.0) * order)
        characters.setdefault(
            tuple(int(value) % order for value in phase), np.asarray(q, dtype=np.float64)
        )
    if len(characters) != order:
        raise ContractError(f"translation group of order {order} has {len(characters)} characters")
    stars: list[tuple[tuple[float, float, float], int]] = []
    for q in characters.values():
        overlap = np.mean(chi[pure] * np.exp(-2.0j * np.pi * (vectors @ q)))
        count = int(round(float(overlap.real)))
        if abs(overlap - count) > tolerance:
            raise ContractError(f"non-integral translation multiplicity {overlap:.6f}")
        if count:
            k = np.mod(primitive @ q, 1.0)
            k[np.isclose(k, 1.0, atol=tolerance)] = 0.0
            stars.append((tuple(float(round(value, 10)) for value in k), count))
    if sum(count for _, count in stars) != round(dimension):
        raise ContractError("translation multiplicities do not sum to the copy dimension")
    stars.sort()
    return WavevectorStar(
        wavevectors=tuple(item[0] for item in stars),
        multiplicities=tuple(item[1] for item in stars),
        primitive_basis=primitive,
    )


def intersect_projectors(left: object, right: object, *, tolerance: float = 1.0e-8) -> np.ndarray:
    """Return the intersection of two commuting orthogonal projectors."""

    left_projector = _validate_projector(
        _finite_array(left, name="left projector"),
        name="left projector",
        tolerance=tolerance,
    )
    right_projector = _validate_projector(
        _finite_array(right, name="right projector"),
        name="right projector",
        tolerance=tolerance,
    )
    if left_projector.shape != right_projector.shape:
        raise ContractError("projectors have different dimensions")
    commutator = float(
        np.linalg.norm(
            left_projector @ right_projector - right_projector @ left_projector,
            ord=2,
        )
    )
    if commutator > tolerance:
        raise ContractError(f"mode projectors do not commute: residual={commutator:.3e}")
    return _validate_projector(
        left_projector @ right_projector,
        name="projector intersection",
        tolerance=tolerance,
    )


def decompose_symmetry_modes(
    parent_operators: object,
    subgroup_operators: object,
    *,
    atom_count: int,
    primary_isotypic_projector: object | None = None,
    tolerance: float = 1.0e-8,
) -> SymmetryModeDecomposition:
    """Decompose H-invariant/G-breaking displacements after translation quotient."""

    parent = _validate_orthogonal_operators(parent_operators, tolerance=tolerance)
    subgroup = _validate_orthogonal_operators(subgroup_operators, tolerance=tolerance)
    if parent.shape[1:] != subgroup.shape[1:] or parent.shape[1] != 3 * atom_count:
        raise ContractError("group operators and atom count use different dimensions")
    for operator in subgroup:
        if min(float(np.linalg.norm(operator - item, ord=2)) for item in parent) > tolerance:
            raise ContractError("subgroup representation is not embedded in parent group")

    translations = translation_projector(atom_count)
    quotient = np.eye(3 * atom_count, dtype=np.float64) - translations
    parent_invariant = reynolds_projector(parent, quotient_projector=quotient, tolerance=tolerance)
    subgroup_invariant = reynolds_projector(
        subgroup, quotient_projector=quotient, tolerance=tolerance
    )
    containment = float(
        np.linalg.norm(subgroup_invariant @ parent_invariant - parent_invariant, ord=2)
    )
    if containment > tolerance:
        raise ContractError(
            f"parent invariant space is not contained in subgroup space: {containment:.3e}"
        )
    breaking = _validate_projector(
        subgroup_invariant - parent_invariant,
        name="symmetry-breaking projector",
        tolerance=tolerance,
    )
    if primary_isotypic_projector is None:
        primary = np.zeros_like(breaking)
    else:
        isotypic = _validate_projector(
            _finite_array(primary_isotypic_projector, name="primary isotypic projector"),
            name="primary isotypic projector",
            tolerance=tolerance,
        )
        if isotypic.shape != breaking.shape:
            raise ContractError("primary projector and displacement dimensions differ")
        primary = intersect_projectors(breaking, isotypic, tolerance=tolerance)
    secondary = _validate_projector(
        breaking - primary, name="secondary-mode projector", tolerance=tolerance
    )
    return SymmetryModeDecomposition(
        translation_projector=translations,
        parent_invariant_projector=parent_invariant,
        subgroup_invariant_projector=subgroup_invariant,
        symmetry_breaking_projector=breaking,
        primary_projector=primary,
        secondary_projector=secondary,
        primary_basis=projector_basis(primary, tolerance=tolerance),
        secondary_basis=projector_basis(secondary, tolerance=tolerance),
    )


def atomic_displacement_representation(
    cartesian_rotations: object, permutations: object, *, tolerance: float = 1.0e-8
) -> np.ndarray:
    """Construct D(g) when each source atom maps to ``permutations[g, source]``."""

    rotations = _finite_array(cartesian_rotations, name="Cartesian rotations")
    mapping = np.asarray(permutations, dtype=np.int64)
    if rotations.ndim != 3 or rotations.shape[1:] != (3, 3):
        raise ContractError("Cartesian rotations must have shape [G,3,3]")
    if mapping.ndim != 2 or mapping.shape[0] != len(rotations):
        raise ContractError("permutations must have shape [G,N]")
    atom_count = mapping.shape[1]
    expected = np.arange(atom_count)
    if any(not np.array_equal(np.sort(row), expected) for row in mapping):
        raise ContractError("every operation must contain one atom permutation")
    operators = np.zeros((len(rotations), 3 * atom_count, 3 * atom_count))
    for operation, (rotation, permutation) in enumerate(zip(rotations, mapping)):
        for source, target in enumerate(permutation):
            operators[operation, 3 * target : 3 * target + 3, 3 * source : 3 * source + 3] = (
                rotation
            )
    return _validate_orthogonal_operators(operators, tolerance=tolerance)


def match_species_permutation(
    transformed: object,
    reference: object,
    atomic_numbers: object,
    lattice: object,
) -> tuple[np.ndarray, float]:
    """Match transformed atoms to same-species reference atoms in one cell."""

    moved = _finite_array(transformed, name="transformed coordinates")
    target = _finite_array(reference, name="reference coordinates")
    numbers = np.asarray(atomic_numbers, dtype=np.int64)
    cell = _finite_array(lattice, name="lattice")
    if moved.shape != target.shape or moved.ndim != 2 or moved.shape[1:] != (3,):
        raise ContractError("transformed and reference coordinates must have shape [N,3]")
    if numbers.shape != (len(target),) or np.any(numbers <= 0):
        raise ContractError("atomic numbers must align with coordinates")
    if cell.shape != (3, 3) or abs(float(np.linalg.det(cell))) <= 1.0e-12:
        raise ContractError("lattice must be a nonsingular 3x3 matrix")

    permutation = np.empty(len(target), dtype=np.int64)
    maximum_residual = 0.0
    for species in np.unique(numbers):
        indices = np.flatnonzero(numbers == species)
        delta = moved[indices, None, :] - target[None, indices, :]
        delta = minimum_image_numpy(delta, cell)
        cost = np.sum((delta @ cell) ** 2, axis=-1)
        rows, columns = linear_sum_assignment(cost)
        selected = np.sqrt(cost[rows, columns])
        if len(selected):
            maximum_residual = max(maximum_residual, float(np.max(selected)))
        permutation[indices[rows]] = indices[columns]
    return permutation, maximum_residual


def _validated_operations(
    lattice: object,
    fractional: object,
    atomic_numbers: object,
    rotations: object,
    translations: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cell = _finite_array(lattice, name="lattice")
    positions = _finite_array(fractional, name="fractional coordinates") % 1.0
    numbers = np.asarray(atomic_numbers, dtype=np.int64)
    rotations_array = np.asarray(rotations, dtype=np.int64)
    translations_array = _finite_array(translations, name="translations")
    if cell.shape != (3, 3) or float(np.linalg.det(cell)) <= 1.0e-12:
        raise ContractError("lattice must be a positive-volume 3x3 matrix")
    if positions.ndim != 2 or positions.shape[1:] != (3,):
        raise ContractError("fractional coordinates must have shape [N,3]")
    if numbers.shape != (len(positions),) or np.any(numbers <= 0):
        raise ContractError("atomic numbers must align with coordinates")
    if (
        rotations_array.ndim != 3
        or rotations_array.shape[1:] != (3, 3)
        or translations_array.shape != (len(rotations_array), 3)
    ):
        raise ContractError("fractional operations have incompatible shapes")
    return cell, positions, numbers, rotations_array, translations_array


def _operation_permutations(
    cell: np.ndarray,
    positions: np.ndarray,
    numbers: np.ndarray,
    rotations: np.ndarray,
    translations: np.ndarray,
) -> tuple[np.ndarray, float]:
    permutations: list[np.ndarray] = []
    maximum_residual = 0.0
    for rotation, translation in zip(rotations, translations):
        transformed = positions @ rotation.T + translation
        permutation, residual = match_species_permutation(transformed, positions, numbers, cell)
        maximum_residual = max(maximum_residual, residual)
        permutations.append(permutation)
    return np.stack(permutations), maximum_residual


def symmetrize_positions(
    lattice: object,
    fractional: object,
    atomic_numbers: object,
    rotations: object,
    translations: object,
    *,
    mapping_tolerance_angstrom: float,
) -> SymmetrizedPositions:
    """Project positions onto the fixed set of the given finite group.

    Every atom receives the minimum-image average of the images that the group
    maps onto it. The result is exactly invariant up to lattice translations,
    and its per-atom displacement is bounded by twice the per-atom noise when
    the input is an exactly symmetric structure plus bounded noise.
    """

    cell, positions, numbers, rotations_array, translations_array = _validated_operations(
        lattice, fractional, atomic_numbers, rotations, translations
    )
    if mapping_tolerance_angstrom <= 0.0:
        raise ContractError("mapping tolerance must be positive")
    permutations, maximum_residual = _operation_permutations(
        cell, positions, numbers, rotations_array, translations_array
    )
    if maximum_residual > mapping_tolerance_angstrom:
        raise ContractError(
            "space-group operation does not map the declared structure: "
            f"residual={maximum_residual:.3e} A"
        )
    correction = np.zeros_like(positions)
    for rotation, translation, permutation in zip(
        rotations_array, translations_array, permutations
    ):
        image = positions @ rotation.T + translation
        delta = image - positions[permutation]
        delta = minimum_image_numpy(delta, cell)
        correction[permutation] += delta
    correction /= float(len(rotations_array))
    idealized = positions + correction
    displacement = float(np.max(np.linalg.norm(correction @ cell, axis=1), initial=0.0))
    return SymmetrizedPositions(
        fractional=idealized % 1.0,
        maximum_displacement_angstrom=displacement,
        maximum_mapping_residual_angstrom=maximum_residual,
    )


def serialization_displacement_bound(source_lattice: object, *, decimals: int) -> float:
    """Worst-case per-atom Cartesian error of symmetrizing rounded coordinates.

    Rounding each fractional coordinate to ``decimals`` places moves an atom by
    at most ``0.5e-decimals * sum_i |a_i|``. Symmetrizing the rounded
    structure moves an atom by at most twice that bound.
    """

    cell = _finite_array(source_lattice, name="source lattice")
    if cell.shape != (3, 3) or decimals <= 0:
        raise ContractError("source lattice must be 3x3 and decimals positive")
    return 2.0 * 0.5 * 10.0 ** (-int(decimals)) * float(np.sum(np.linalg.norm(cell, axis=1)))


def representation_from_fractional_operations(
    lattice: object,
    fractional: object,
    atomic_numbers: object,
    rotations: object,
    translations: object,
    *,
    mapping_tolerance_angstrom: float = 1.0e-5,
    orthogonality_tolerance: float = 1.0e-9,
) -> AtomicDisplacementRepresentation:
    """Map spglib-style operations onto Cartesian atomic displacements.

    The lattice must carry the metric symmetry of every supplied operation.
    A strained lattice is rejected instead of being repaired: strain is a
    separate order-parameter channel, not representation roundoff.
    """

    cell, positions, numbers, rotations_array, translations_array = _validated_operations(
        lattice, fractional, atomic_numbers, rotations, translations
    )
    if mapping_tolerance_angstrom <= 0.0:
        raise ContractError("mapping tolerance must be positive")
    if not 0.0 < orthogonality_tolerance <= 1.0e-7:
        raise ContractError("orthogonality tolerance must lie in (0, 1e-7]")
    permutations, maximum_residual = _operation_permutations(
        cell, positions, numbers, rotations_array, translations_array
    )
    if maximum_residual > mapping_tolerance_angstrom:
        raise ContractError(
            "space-group operation does not map the declared structure: "
            f"residual={maximum_residual:.3e} A"
        )

    inverse_cartesian = np.linalg.inv(cell.T)
    cartesian_rotations = np.stack(
        [cell.T @ rotation @ inverse_cartesian for rotation in rotations_array]
    )
    identity = np.eye(3, dtype=np.float64)
    orthogonality = max(
        float(np.linalg.norm(rotation.T @ rotation - identity, ord=2))
        for rotation in cartesian_rotations
    )
    if orthogonality > orthogonality_tolerance:
        raise ContractError(
            "representation operators are not orthogonal in the supplied lattice "
            f"(metric-symmetry residual={orthogonality:.3e}); use the reference "
            "lattice of the group that owns these operations"
        )
    operators = atomic_displacement_representation(
        cartesian_rotations, permutations, tolerance=orthogonality_tolerance
    )
    return AtomicDisplacementRepresentation(
        operators=operators,
        permutations=permutations,
        maximum_mapping_residual_angstrom=maximum_residual,
        maximum_orthogonality_residual=orthogonality,
    )


__all__ = [
    "AtomicDisplacementRepresentation",
    "RealIsotypicSector",
    "SymmetrizedPositions",
    "SymmetryModeDecomposition",
    "WavevectorStar",
    "atomic_displacement_representation",
    "decompose_symmetry_modes",
    "intersect_projectors",
    "irreducible_matrices",
    "match_species_permutation",
    "projector_basis",
    "pure_translation_wavevectors",
    "real_isotypic_sectors",
    "reynolds_projector",
    "representation_from_fractional_operations",
    "serialization_displacement_bound",
    "symmetrize_positions",
    "translation_projector",
]
