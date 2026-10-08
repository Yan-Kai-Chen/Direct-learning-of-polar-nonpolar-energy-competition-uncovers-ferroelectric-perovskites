"""Shared periodic geometry features for crystal graph models."""

from __future__ import annotations

import torch

from polarevolve.data.batch import PackedASUBatch
from polarevolve.crystal.periodic import PERIODIC_IMAGES


def complete_periodic_geometry(batch, fractional, *, radial_centers, cutoff):
    atoms = torch.arange(batch.num_atoms, device=fractional.device)
    source = torch.cat((batch.edge_index[0], atoms))
    target = torch.cat((batch.edge_index[1], atoms))
    structure = batch.atom_to_structure[source]
    source, target, edge = PERIODIC_IMAGES.edges(
        fractional, batch.lattice, source, target, structure, cutoff
    )
    distance = edge.norm(dim=-1).clamp_min(1.0e-8)
    direction = edge / distance[:, None]
    width = cutoff / max(int(radial_centers.numel()) - 1, 1)
    radial = torch.exp(-0.5 * (
        (distance[:, None] - radial_centers.to(distance)[None]) / width
    ).square())
    envelope = 0.5 * (torch.cos(torch.pi * distance / cutoff) + 1)
    return source, target, direction, radial * envelope[:, None], envelope


def lattice_invariants(batch: PackedASUBatch) -> torch.Tensor:
    """Seven rotation-invariant, volume-normalized lattice features."""

    gram = batch.lattice @ batch.lattice.transpose(1, 2)
    volume = torch.linalg.det(batch.lattice).abs().clamp_min(1.0e-8)
    normalized = gram / volume.pow(2.0 / 3.0)[:, None, None]
    atom_count = (batch.atom_ptr[1:] - batch.atom_ptr[:-1]).to(volume.dtype)
    return torch.stack(
        (
            normalized[:, 0, 0], normalized[:, 1, 1], normalized[:, 2, 2],
            normalized[:, 0, 1], normalized[:, 0, 2], normalized[:, 1, 2],
            torch.log(volume / atom_count.clamp_min(1.0)),
        ),
        dim=-1,
    )


def periodic_edge_geometry(
    batch: PackedASUBatch,
    fractional: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    periodic_offsets: torch.Tensor,
    radial_centers: torch.Tensor,
    cutoff: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Minimum-image directions, radial features and cutoff envelope."""

    delta = fractional[target] - fractional[source]
    candidates = delta[:, None, :] + periodic_offsets.to(delta)[None]
    compute_dtype = torch.float64 if fractional.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=fractional.device.type, enabled=False):
        cartesian = torch.einsum(
            "eki,eij->ekj", candidates.to(compute_dtype),
            batch.lattice[batch.edge_to_structure].to(compute_dtype),
        )
    squared = cartesian.square().sum(dim=-1)
    # Degenerate minimum images need a stable fractional-index tie break under
    # Cartesian rotation; FP32 roundoff must not select a different bond image.
    minimum = squared.min(dim=1, keepdim=True).values
    tolerance = 8 * torch.finfo(squared.dtype).eps * minimum.abs().clamp_min(1.0)
    chosen = (squared <= minimum + tolerance).to(torch.int32).argmax(dim=1)
    edge = cartesian[torch.arange(len(cartesian), device=delta.device), chosen]
    distance = torch.linalg.vector_norm(edge, dim=-1).clamp_min(1.0e-8)
    direction = edge / distance[:, None]
    width = cutoff / max(int(radial_centers.numel()) - 1, 1)
    radial = torch.exp(
        -0.5 * ((distance[:, None] - radial_centers.to(distance)[None]) / width).square()
    )
    envelope = torch.where(
        distance < cutoff,
        0.5 * (torch.cos(torch.pi * distance / cutoff) + 1.0),
        torch.zeros_like(distance),
    )
    return direction, radial * envelope[:, None], envelope


__all__ = ["lattice_invariants", "periodic_edge_geometry"]
