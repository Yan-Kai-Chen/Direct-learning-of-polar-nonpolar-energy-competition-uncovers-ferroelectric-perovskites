"""Deterministic supercell expansion shared by the FE1 sidecar builder and objective.

The sidecar stores one row-HNF repeat matrix and its translation set per
pair (see ``crystal.integer_lattice.supercell_translation_set``); the
reference polar structure is stored in the atom order produced by
:func:`expand_supercell` on the parent group cell.  Training recomputes the
same expansion so the OD17/OD20 per-atom correspondences hold by
construction.
"""

from __future__ import annotations

import torch


def expand_supercell(
    fractional: torch.Tensor,
    matrix: torch.Tensor,
    translations: torch.Tensor,
) -> torch.Tensor:
    """Expand fractional coordinates into the supercell defined by ``matrix``.

    Row-vector convention: the supercell lattice is ``matrix @ lattice`` and
    the output fractional coordinates are expressed in the supercell frame.
    Copy order follows ``translations`` exactly, with the input atom order
    preserved inside each copy.  All arithmetic is differentiable in
    ``fractional``.
    """

    if fractional.ndim != 2 or fractional.shape[1] != 3:
        raise ValueError("fractional must have shape [N,3]")
    if matrix.shape != (3, 3) or translations.ndim != 2 or translations.shape[1] != 3:
        raise ValueError("matrix must be [3,3] and translations [K,3]")
    compute_dtype = torch.promote_types(fractional.dtype, torch.float32)
    inverse = torch.linalg.inv(matrix.to(dtype=compute_dtype, device=fractional.device))
    translations = translations.to(dtype=compute_dtype, device=fractional.device)
    blocks = [
        (fractional.to(compute_dtype) + shift) @ inverse for shift in translations
    ]
    return torch.remainder(torch.cat(blocks, dim=0), 1.0)


__all__ = ["expand_supercell"]
