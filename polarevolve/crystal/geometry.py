"""Shared geometry thresholds and device-side final-state delivery screening."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, kw_only=True)
class GeometryThresholds:
    minimum_distance: float = 0.7
    maximum_aspect_ratio: float = 6.0
    minimum_volume_per_atom: float = 1.0
    maximum_volume_per_atom: float = 100.0


def device_geometry_screen(lattice: Any, fractional: Any, config: GeometryThresholds) -> dict:
    """Final-state delivery filter using complete, bounded periodic distances."""
    import torch
    from polarevolve.crystal.periodic import PERIODIC_IMAGES

    matrix = lattice.detach().to(dtype=torch.float64)
    coordinates = fractional.detach().to(device=matrix.device, dtype=torch.float64)
    if matrix.shape != (3, 3) or coordinates.ndim != 2 or coordinates.shape[1:] != (3,) or not len(coordinates):
        raise ValueError("geometry screen requires lattice [3,3] and nonempty coordinates [atoms,3]")
    finite = torch.isfinite(matrix).all() & torch.isfinite(coordinates).all()
    safe_matrix = torch.where(finite, matrix, torch.eye(3, device=matrix.device, dtype=matrix.dtype))
    safe_coordinates = torch.where(finite, coordinates, torch.zeros_like(coordinates))
    volume = torch.linalg.det(safe_matrix).abs()
    singular = torch.linalg.svdvals(safe_matrix)
    aspect = torch.where(singular.min() > 0, singular.max() / singular.min(), torch.inf)
    search_matrix = torch.where(singular.min() > 1e-12, safe_matrix,
                                torch.eye(3, device=matrix.device, dtype=matrix.dtype))
    bound = float(search_matrix.norm(dim=-1).min().cpu()) * (1 + 1e-10)
    offsets = PERIODIC_IMAGES.shifts(search_matrix, bound)
    self_distance = torch.linalg.vector_norm(offsets @ search_matrix, dim=-1)
    self_distance = self_distance.masked_fill((offsets == 0).all(dim=-1), torch.inf).min()
    pairs = torch.triu_indices(len(coordinates), len(coordinates), offset=1, device=matrix.device)
    displacement = safe_coordinates[pairs[1]] - safe_coordinates[pairs[0]]
    distances = PERIODIC_IMAGES.nearest(displacement, search_matrix.expand(len(displacement), -1, -1)).norm(dim=-1)
    minimum = torch.cat((self_distance.reshape(1), distances.reshape(-1))).min()
    vpa = volume / len(coordinates)
    eligible = (finite & (vpa >= config.minimum_volume_per_atom) & (vpa <= config.maximum_volume_per_atom)
                & (aspect <= config.maximum_aspect_ratio) & (minimum >= config.minimum_distance))
    values = torch.stack((finite, volume, vpa, aspect, minimum, eligible)).cpu().tolist()
    return {
        "schema_version": "gt_sge_delivery_geometry_v2",
        "device": str(matrix.device), "finite": bool(values[0]),
        "volume": values[1] if values[0] else None, "volume_per_atom": values[2] if values[0] else None,
        "lattice_aspect_ratio": values[3] if values[0] else None,
        "minimum_pair_distance": values[4] if values[0] else None,
        "eligible": bool(values[5]), "rejection_reason": (
            None if values[5] else "geometry_threshold" if values[0] else "nonfinite"
        ),
        "thresholds": {name: getattr(config, name) for name in (
            "minimum_distance", "maximum_aspect_ratio", "minimum_volume_per_atom", "maximum_volume_per_atom")},
    }
