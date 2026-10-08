"""Complete periodic image geometry for row-vector crystal lattices."""

from __future__ import annotations

from collections import OrderedDict

import torch


class PeriodicImages:
    """Cache only lattice-dependent search bounds, never moving atom geometry."""

    def __init__(self, capacity: int = 32, chunk_size: int = 131072):
        if capacity <= 0 or chunk_size <= 0:
            raise ValueError("periodic cache capacity and chunk size must be positive")
        self.capacity = capacity
        self.chunk_size = chunk_size
        self.cache = OrderedDict()

    def shifts(self, lattice: torch.Tensor, cutoff: float) -> torch.Tensor:
        if cutoff <= 0:
            raise ValueError("periodic cutoff must be positive")
        key = (id(lattice), lattice._version, float(cutoff))
        cached = self.cache.get(key)
        if cached is not None and cached[0] is lattice:
            self.cache.move_to_end(key)
            return cached[1]
        # Centered fractional differences are in [-1/2,1/2]. Reciprocal
        # columns bound every component of a Cartesian vector inside the ball.
        inverse = torch.linalg.inv(lattice.detach().double())
        bounds = torch.ceil(cutoff * inverse.norm(dim=-2) + 0.5)
        bounds = bounds.reshape(-1, 3).amax(dim=0).to(torch.int64).cpu().tolist()
        axes = [torch.arange(-b, b + 1, device=lattice.device) for b in bounds]
        shifts = torch.cartesian_prod(*axes).to(lattice.dtype)
        self.cache[key] = (lattice, shifts)
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
        return shifts

    def edges(self, fractional, lattice, source, target, structure, cutoff):
        """Return all directed image edges and differentiable Cartesian vectors."""
        dtype = torch.float64 if fractional.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=fractional.device.type, enabled=False):
            cell = lattice.to(dtype)
            shifts = self.shifts(lattice, cutoff).to(dtype)
            delta = fractional[target].to(dtype) - fractional[source].to(dtype)
            centered = delta - torch.round(delta)
            shift_count = len(shifts)
            total = len(source) * shift_count
            left, right, vectors = [], [], []
            for start in range(0, total, self.chunk_size):
                flat = torch.arange(
                    start, min(start + self.chunk_size, total), device=fractional.device
                )
                pair = torch.div(flat, shift_count, rounding_mode="floor")
                image = flat.remainder(shift_count)
                displacement = centered[pair] + shifts[image]
                vector = (displacement @ cell if cell.ndim == 2 else torch.einsum(
                    "ni,nij->nj", displacement, cell[structure[pair]]
                ))
                valid = vector.square().sum(-1) < cutoff * cutoff
                valid &= ~((source[pair] == target[pair]) & (shifts[image] == 0).all(-1))
                left.append(source[pair][valid])
                right.append(target[pair][valid])
                vectors.append(vector[valid])
            if not total:
                return source[:0], target[:0], fractional.new_empty((0, 3), dtype=dtype)
            return torch.cat(left), torch.cat(right), torch.cat(vectors)

    def nearest(self, delta: torch.Tensor, lattice: torch.Tensor) -> torch.Tensor:
        """Exact closest image, bounded by a known centered-image candidate."""
        dtype = torch.float64 if delta.dtype == torch.float64 else torch.float32
        if not len(delta):
            return delta.to(dtype)
        with torch.autocast(device_type=delta.device.type, enabled=False):
            cell = lattice.to(dtype)
            centered = delta.to(dtype) - delta.to(dtype).round()
            initial = torch.einsum("ni,nij->nj", centered, cell)
            # This scalar is a search bound only. Gradients flow through the
            # selected Cartesian image; ties use the lexicographic shift order.
            radius = max(float(initial.detach().norm(dim=-1).max().cpu()), 1.0e-8)
            shifts = self.shifts(lattice, radius).to(dtype)
            best = initial.square().sum(-1)
            result = initial
            for start in range(0, len(shifts), 64):
                candidates = torch.einsum(
                    "nki,nij->nkj", centered[:, None] + shifts[None, start:start + 64], cell
                )
                squared, index = candidates.square().sum(-1).min(-1)
                chosen = candidates[torch.arange(len(delta), device=delta.device), index]
                tolerance = 8 * torch.finfo(dtype).eps * best.abs().clamp_min(1)
                replace = squared < best - tolerance
                result = torch.where(replace[:, None], chosen, result)
                best = torch.minimum(best, squared)
            return result


PERIODIC_IMAGES = PeriodicImages()


def minimum_image_numpy(delta, lattice):
    """CPU crystallographic matching, using the mature closest-image solver."""
    import numpy as np
    from pymatgen.core import Lattice
    from pymatgen.util.coord import pbc_shortest_vectors
    values = np.asarray(delta, dtype=np.float64)
    cell = np.asarray(lattice, dtype=np.float64)
    if not values.size:
        return values.copy()
    vectors = pbc_shortest_vectors(Lattice(cell), np.zeros((1, 3)), values.reshape(-1, 3))[0]
    best = (vectors @ np.linalg.inv(cell)).reshape(values.shape)
    centered = values - np.round(values)
    initial_squared = np.sum((centered @ cell) ** 2, axis=-1)
    best_squared = np.sum(vectors ** 2, axis=-1).reshape(values.shape[:-1])
    # Preserve the centered representative only when it is an equally short
    # image; deterministic ties must not change an operation's atom mapping.
    tied = np.abs(initial_squared - best_squared) <= 8 * np.finfo(float).eps * np.maximum(1, best_squared)
    return np.where(tied[..., None], centered, best)

