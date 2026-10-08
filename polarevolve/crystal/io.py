"""Minimal, explicit structure serialization for Version9 artifacts."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Sequence

import numpy as np

_ELEMENTS = (
    "X", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg",
    "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn",
    "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb",
    "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In",
    "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm",
    "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta",
    "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At",
    "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk",
    "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt",
    "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
)


def atomic_symbol(atomic_number: int) -> str:
    value = int(atomic_number)
    if not 1 <= value < len(_ELEMENTS):
        raise ValueError(f"unsupported atomic number: {value}")
    return _ELEMENTS[value]


def safe_stem(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    if not stem:
        raise ValueError("artifact stem is empty after sanitization")
    return stem


def _angle(left: np.ndarray, right: np.ndarray) -> float:
    cosine = float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right)))
    return math.degrees(math.acos(min(1.0, max(-1.0, cosine))))


def write_p1_cif(
    path: str | Path,
    *,
    identifier: str,
    lattice: Sequence[Sequence[float]],
    fractional: Sequence[Sequence[float]],
    atomic_numbers: Sequence[int],
    target_hall_number: int,
    target_space_group_number: int,
) -> None:
    """Write a P1 coordinate carrier without claiming a post-hoc SG assignment."""

    matrix = np.asarray(lattice, dtype=np.float64)
    coordinates = np.remainder(np.asarray(fractional, dtype=np.float64), 1.0)
    numbers = tuple(int(value) for value in atomic_numbers)
    if matrix.shape != (3, 3) or coordinates.shape != (len(numbers), 3):
        raise ValueError("CIF lattice, coordinates, and atom types are inconsistent")
    if not np.isfinite(matrix).all() or not np.isfinite(coordinates).all():
        raise ValueError("CIF data must be finite")
    a, b, c = (float(np.linalg.norm(row)) for row in matrix)
    alpha, beta, gamma = (
        _angle(matrix[1], matrix[2]),
        _angle(matrix[0], matrix[2]),
        _angle(matrix[0], matrix[1]),
    )
    lines = [
        f"data_{safe_stem(identifier)}",
        "_audit_creation_method 'GT-SGE Version9 exact-ASU coordinate carrier'",
        "_symmetry_space_group_name_H-M 'P 1'",
        "_symmetry_Int_Tables_number 1",
        f"_gt_sge_target_hall_number {int(target_hall_number)}",
        f"_gt_sge_target_space_group_number {int(target_space_group_number)}",
        f"_cell_length_a {a:.10f}",
        f"_cell_length_b {b:.10f}",
        f"_cell_length_c {c:.10f}",
        f"_cell_angle_alpha {alpha:.10f}",
        f"_cell_angle_beta {beta:.10f}",
        f"_cell_angle_gamma {gamma:.10f}",
        "loop_",
        "_symmetry_equiv_pos_site_id",
        "_symmetry_equiv_pos_as_xyz",
        "1 'x, y, z'",
        "loop_",
        "_atom_site_type_symbol",
        "_atom_site_label",
        "_atom_site_symmetry_multiplicity",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z",
        "_atom_site_occupancy",
    ]
    counts: dict[str, int] = {}
    for number, xyz in zip(numbers, coordinates):
        symbol = atomic_symbol(number)
        index = counts.get(symbol, 0)
        counts[symbol] = index + 1
        lines.append(
            f"{symbol} {symbol}{index} 1 {xyz[0]:.10f} {xyz[1]:.10f} {xyz[2]:.10f} 1"
        )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="ascii")


__all__ = ["atomic_symbol", "safe_stem", "write_p1_cif"]
