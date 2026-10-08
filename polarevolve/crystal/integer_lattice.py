"""Exact integer-lattice helpers for Version9 cell contracts."""

from __future__ import annotations

from collections.abc import Sequence


class IntegerLatticeError(ValueError):
    """Raised when an integer lattice transform is malformed."""


IntegerMatrix3 = tuple[
    tuple[int, int, int],
    tuple[int, int, int],
    tuple[int, int, int],
]
IDENTITY_HNF: IntegerMatrix3 = ((1, 0, 0), (0, 1, 0), (0, 0, 1))


def normalize_integer_matrix(
    matrix: Sequence[Sequence[int]], *, field_name: str
) -> IntegerMatrix3:
    if not isinstance(matrix, (list, tuple)) or len(matrix) != 3:
        raise IntegerLatticeError(f"{field_name} must be a 3x3 integer matrix")
    rows: list[tuple[int, int, int]] = []
    for row in matrix:
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            raise IntegerLatticeError(f"{field_name} must be a 3x3 integer matrix")
        if any(not isinstance(value, int) or isinstance(value, bool) for value in row):
            raise IntegerLatticeError(f"{field_name} must contain only integers")
        rows.append((int(row[0]), int(row[1]), int(row[2])))
    return rows[0], rows[1], rows[2]


def determinant_3x3(matrix: Sequence[Sequence[int]]) -> int:
    value = normalize_integer_matrix(matrix, field_name="matrix")
    a, b, c = value
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def validate_row_hnf(matrix: Sequence[Sequence[int]]) -> IntegerMatrix3:
    value = normalize_integer_matrix(matrix, field_name="output_hnf")
    if value[1][0] != 0 or value[2][0] != 0 or value[2][1] != 0:
        raise IntegerLatticeError("output_hnf must be upper triangular")
    if any(value[index][index] <= 0 for index in range(3)):
        raise IntegerLatticeError("output_hnf diagonal entries must be positive")
    for row in range(3):
        for column in range(row + 1, 3):
            if not 0 <= value[row][column] < value[column][column]:
                raise IntegerLatticeError(
                    "output_hnf upper entries must satisfy 0 <= H[i,j] < H[j,j]"
                )
    return value


def supercell_translation_set(matrix: Sequence[Sequence[int]]) -> tuple[tuple[int, int, int], ...]:
    """Enumerate the interior lattice points of an integer supercell matrix.

    The matrix maps the base lattice to the supercell lattice under the
    row-vector convention (``lattice_super = matrix @ lattice_base``) and must
    have positive determinant.  Returns exactly ``det(matrix)`` integer
    translations, sorted lexicographically, so every consumer agrees on
    supercell copy order.  Exact rational arithmetic keeps the membership
    test free of tolerance choices.
    """

    from fractions import Fraction

    value = normalize_integer_matrix(matrix, field_name="supercell_matrix")
    count = determinant_3x3(value)
    if count <= 0:
        raise IntegerLatticeError("supercell matrix must have positive determinant")
    a, b, c = value
    cofactor = (
        (b[1] * c[2] - b[2] * c[1], -(b[0] * c[2] - b[2] * c[0]), b[0] * c[1] - b[1] * c[0]),
        (-(a[1] * c[2] - a[2] * c[1]), a[0] * c[2] - a[2] * c[0], -(a[0] * c[1] - a[1] * c[0])),
        (a[1] * b[2] - a[2] * b[1], -(a[0] * b[2] - a[2] * b[0]), a[0] * b[1] - a[1] * b[0]),
    )
    adjugate = tuple(tuple(cofactor[column][row] for column in range(3)) for row in range(3))
    bounds = [sum(abs(value[row][column]) for row in range(3)) + 1 for column in range(3)]
    points: list[tuple[int, int, int]] = []
    for t1 in range(-bounds[0], bounds[0] + 1):
        for t2 in range(-bounds[1], bounds[1] + 1):
            for t3 in range(-bounds[2], bounds[2] + 1):
                t = (t1, t2, t3)
                x = tuple(
                    Fraction(sum(t[j] * adjugate[j][i] for j in range(3)), count)
                    for i in range(3)
                )
                if all(0 <= component < 1 for component in x):
                    points.append(t)
    points.sort()
    if len(points) != count:
        raise IntegerLatticeError("supercell translation enumeration does not close the determinant")
    return tuple(points)


__all__ = [
    "IDENTITY_HNF",
    "IntegerLatticeError",
    "IntegerMatrix3",
    "determinant_3x3",
    "normalize_integer_matrix",
    "supercell_translation_set",
    "validate_row_hnf",
]
