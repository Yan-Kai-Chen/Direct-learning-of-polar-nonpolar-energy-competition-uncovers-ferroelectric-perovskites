"""Cartesian dual-metric operations for packed ASU parameters."""

from __future__ import annotations

import torch

from polarevolve.data.batch import PackedASUBatch

FULL_ASU_STATE_V1 = "full_asu_v1"
TRANSLATION_QUOTIENT_V1 = "translation_quotient_v1"
STATE_QUOTIENT_CONTRACTS = (FULL_ASU_STATE_V1, TRANSLATION_QUOTIENT_V1)
MEMBER_SUM_CARTESIAN_V1 = "member_sum_cartesian_v1"
MEMBER_MEAN_CARTESIAN_V1 = "member_mean_cartesian_v1"
DIFFUSION_METRIC_CONTRACTS = (
    MEMBER_SUM_CARTESIAN_V1,
    MEMBER_MEAN_CARTESIAN_V1,
)
NATIVE_MEMBER_SUM_SCORE_V1 = "member_sum_native_covector_v1"
CONVENTIONAL_HALL_CELL_V1 = "conventional_hall_group_cell_v1"


def metric_contract_metadata(diffusion_metric: str) -> dict[str, str]:
    """Return the versioned checkpoint identity for one diffusion metric."""

    if diffusion_metric not in DIFFUSION_METRIC_CONTRACTS:
        raise ValueError(f"unsupported diffusion metric: {diffusion_metric!r}")
    return {
        "geometry_metric": MEMBER_SUM_CARTESIAN_V1,
        "diffusion_metric": diffusion_metric,
        "score_preconditioner": NATIVE_MEMBER_SUM_SCORE_V1,
        "cell_representation": CONVENTIONAL_HALL_CELL_V1,
    }


def orbit_metric_view(
    batch: PackedASUBatch, metric_convention: str
) -> torch.Tensor:
    """View the unique member-sum batch metric under an explicit convention."""

    if metric_convention == MEMBER_SUM_CARTESIAN_V1:
        return batch.orbit_metric
    if metric_convention != MEMBER_MEAN_CARTESIAN_V1:
        raise ValueError(f"unsupported metric convention: {metric_convention!r}")
    multiplicity = batch.orbit_multiplicities.to(
        device=batch.orbit_metric.device, dtype=batch.orbit_metric.dtype
    )
    if bool((multiplicity <= 0).any().item()):
        raise ValueError("orbit multiplicities must be positive")
    return batch.orbit_metric / multiplicity[:, None, None]


def parameter_multiplicity(batch: PackedASUBatch) -> torch.Tensor:
    orbit = torch.repeat_interleave(
        torch.arange(batch.num_orbits, device=batch.lattice.device),
        batch.orbit_dimensions,
    )
    if orbit.shape != batch.parameter_shape:
        raise ValueError("orbit dimensions do not span packed ASU parameters")
    return batch.orbit_multiplicities[orbit].to(batch.lattice.dtype)


def diffusion_to_native_covector(
    batch: PackedASUBatch,
    covector: torch.Tensor,
    diffusion_metric: str,
) -> torch.Tensor:
    """Map a diffusion-score covector to the model's member-sum native form."""

    if covector.shape != batch.parameter_shape:
        raise ValueError("covector must match packed ASU parameters")
    if diffusion_metric == MEMBER_SUM_CARTESIAN_V1:
        return covector
    if diffusion_metric == MEMBER_MEAN_CARTESIAN_V1:
        return covector * parameter_multiplicity(batch).to(covector.dtype)
    raise ValueError(f"unsupported diffusion metric: {diffusion_metric!r}")


def active_orbit_indices(
    batch: PackedASUBatch, dimension: int
) -> tuple[torch.Tensor, torch.Tensor]:
    orbits = torch.nonzero(batch.orbit_dimensions == dimension, as_tuple=False).reshape(-1)
    parameters = batch.u_ptr[orbits, None] + torch.arange(
        dimension, device=batch.lattice.device, dtype=torch.long
    )[None, :]
    return orbits, parameters


def integer_dual_quadratic(
    inverse_metric: torch.Tensor, vectors: torch.Tensor
) -> torch.Tensor:
    """Evaluate k^T G^-1 k for inverse metrics and integer vectors."""

    metric = torch.as_tensor(inverse_metric)
    values = torch.as_tensor(vectors, device=metric.device, dtype=metric.dtype)
    if metric.ndim != 3 or metric.shape[1] != metric.shape[2]:
        raise ValueError("inverse metrics must have shape [items,d,d]")
    if values.ndim != 2 or values.shape[1] != metric.shape[1]:
        raise ValueError("integer vectors must have shape [vectors,d]")
    return torch.einsum("ki,nij,kj->nk", values, metric, values)


def metric_inverse_multiply(
    batch: PackedASUBatch,
    covector: torch.Tensor,
    *,
    metric_convention: str = MEMBER_SUM_CARTESIAN_V1,
) -> torch.Tensor:
    if covector.shape != batch.parameter_shape:
        raise ValueError("covector must match packed ASU parameters")
    output = torch.zeros_like(covector)
    compute_dtype = torch.float64 if covector.dtype == torch.float64 else torch.float32
    for dimension in (1, 2, 3):
        orbits, parameters = active_orbit_indices(batch, dimension)
        if orbits.numel() == 0:
            continue
        metric = orbit_metric_view(batch, metric_convention)[
            orbits, :dimension, :dimension
        ].to(compute_dtype)
        solved = torch.linalg.solve(
            metric, covector[parameters].to(compute_dtype).unsqueeze(-1)
        ).squeeze(-1)
        output[parameters] = solved.to(output.dtype)
    return output


def metric_multiply(
    batch: PackedASUBatch,
    tangent: torch.Tensor,
    *,
    metric_convention: str = MEMBER_SUM_CARTESIAN_V1,
) -> torch.Tensor:
    if tangent.shape != batch.parameter_shape:
        raise ValueError("tangent must match packed ASU parameters")
    output = torch.zeros_like(tangent)
    compute_dtype = torch.float64 if tangent.dtype == torch.float64 else torch.float32
    for dimension in (1, 2, 3):
        orbits, parameters = active_orbit_indices(batch, dimension)
        if orbits.numel() == 0:
            continue
        metric = orbit_metric_view(batch, metric_convention)[
            orbits, :dimension, :dimension
        ].to(compute_dtype)
        output[parameters] = torch.einsum(
            "nij,nj->ni", metric, tangent[parameters].to(compute_dtype)
        ).to(output.dtype)
    return output


def _translation_metric_basis(
    batch: PackedASUBatch, metric_convention: str
) -> torch.Tensor:
    return torch.stack(
        [
            metric_multiply(
                batch,
                batch.translation_basis[:, axis],
                metric_convention=metric_convention,
            )
            for axis in range(3)
        ],
        dim=1,
    )


def _translation_coefficients(
    batch: PackedASUBatch,
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    products = left[:, :, None] * right[:, None, :]
    totals = products.new_zeros((batch.batch_size, 3, 3))
    totals.index_add_(0, batch.parameter_to_structure, products)
    return totals


def _solve_translation_systems(
    batch: PackedASUBatch,
    gram: torch.Tensor,
    rhs: torch.Tensor,
) -> torch.Tensor:
    """Solve the 1D/2D/3D quotient systems in three batched operations."""

    coefficients = torch.zeros_like(rhs)
    compute_dtype = torch.float64 if rhs.dtype == torch.float64 else torch.float32
    for dimension in (1, 2, 3):
        selected = batch.translation_dimensions == dimension
        solved = torch.linalg.solve(
            gram[selected, :dimension, :dimension].to(compute_dtype),
            rhs[selected, :dimension].to(compute_dtype).unsqueeze(-1),
        ).squeeze(-1)
        coefficients[selected, :dimension] = solved.to(rhs.dtype)
    return coefficients


def project_translation_quotient_tangent(
    batch: PackedASUBatch,
    tangent: torch.Tensor,
    *,
    metric_convention: str = MEMBER_SUM_CARTESIAN_V1,
) -> torch.Tensor:
    """Remove legal common-translation gauge from a primal ASU tangent."""

    if tangent.shape != batch.parameter_shape:
        raise ValueError("tangent must match packed ASU parameters")
    if not bool((batch.translation_dimensions > 0).any().item()):
        return tangent
    basis = batch.translation_basis.to(tangent.dtype)
    metric_basis = _translation_metric_basis(batch, metric_convention).to(tangent.dtype)
    weighted = metric_multiply(
        batch, tangent, metric_convention=metric_convention
    )
    gram = _translation_coefficients(batch, basis, metric_basis)
    rhs = tangent.new_zeros((batch.batch_size, 3))
    rhs.index_add_(0, batch.parameter_to_structure, basis * weighted[:, None])
    coefficients = _solve_translation_systems(batch, gram, rhs)
    correction = torch.sum(
        basis * coefficients[batch.parameter_to_structure], dim=1
    )
    return tangent - correction


def project_translation_quotient_covector(
    batch: PackedASUBatch,
    covector: torch.Tensor,
    *,
    metric_convention: str = MEMBER_SUM_CARTESIAN_V1,
) -> torch.Tensor:
    """Project a covector into the annihilator of legal translations."""

    if covector.shape != batch.parameter_shape:
        raise ValueError("covector must match packed ASU parameters")
    if not bool((batch.translation_dimensions > 0).any().item()):
        return covector
    basis = batch.translation_basis.to(covector.dtype)
    metric_basis = _translation_metric_basis(batch, metric_convention).to(
        covector.dtype
    )
    gram = _translation_coefficients(batch, basis, metric_basis)
    rhs = covector.new_zeros((batch.batch_size, 3))
    rhs.index_add_(0, batch.parameter_to_structure, basis * covector[:, None])
    coefficients = _solve_translation_systems(batch, gram, rhs)
    correction = torch.sum(
        metric_basis * coefficients[batch.parameter_to_structure], dim=1
    )
    return covector - correction


def apply_state_quotient_tangent(
    batch: PackedASUBatch,
    tangent: torch.Tensor,
    state_quotient: str,
    *,
    metric_convention: str = MEMBER_SUM_CARTESIAN_V1,
) -> torch.Tensor:
    if state_quotient == FULL_ASU_STATE_V1:
        return tangent
    if state_quotient == TRANSLATION_QUOTIENT_V1:
        return project_translation_quotient_tangent(
            batch, tangent, metric_convention=metric_convention
        )
    raise ValueError(f"unsupported state quotient: {state_quotient!r}")


def apply_state_quotient_covector(
    batch: PackedASUBatch,
    covector: torch.Tensor,
    state_quotient: str,
    *,
    metric_convention: str = MEMBER_SUM_CARTESIAN_V1,
) -> torch.Tensor:
    if state_quotient == FULL_ASU_STATE_V1:
        return covector
    if state_quotient == TRANSLATION_QUOTIENT_V1:
        return project_translation_quotient_covector(
            batch, covector, metric_convention=metric_convention
        )
    raise ValueError(f"unsupported state quotient: {state_quotient!r}")


def metric_whiten_noise(
    batch: PackedASUBatch,
    standard_noise: torch.Tensor,
    *,
    metric_convention: str = MEMBER_SUM_CARTESIAN_V1,
) -> torch.Tensor:
    if standard_noise.shape != batch.parameter_shape:
        raise ValueError("standard noise must match packed ASU parameters")
    output = torch.zeros_like(standard_noise)
    compute_dtype = (
        torch.float64 if standard_noise.dtype == torch.float64 else torch.float32
    )
    for dimension in (1, 2, 3):
        orbits, parameters = active_orbit_indices(batch, dimension)
        if orbits.numel() == 0:
            continue
        metric = orbit_metric_view(batch, metric_convention)[
            orbits, :dimension, :dimension
        ].to(compute_dtype)
        cholesky, info = torch.linalg.cholesky_ex(metric)
        if bool((info != 0).any().item()):
            raise FloatingPointError("orbit Cartesian metric is not positive definite")
        output[parameters] = torch.linalg.solve_triangular(
            cholesky.transpose(-1, -2),
            standard_noise[parameters].to(compute_dtype).unsqueeze(-1),
            upper=True,
        ).squeeze(-1).to(output.dtype)
    return output


def tangent_cartesian_inner(
    batch: PackedASUBatch,
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    """Return the member-sum Cartesian inner product for each structure."""

    if left.shape != batch.parameter_shape or right.shape != batch.parameter_shape:
        raise ValueError("tangents must match packed ASU parameters")
    compute_dtype = (
        torch.float64
        if torch.float64 in (left.dtype, right.dtype, batch.orbit_metric.dtype)
        else torch.float32
    )
    with torch.autocast(device_type=left.device.type, enabled=False):
        orbit_inner = torch.zeros(
            batch.num_orbits, device=left.device, dtype=compute_dtype
        )
        for dimension in (1, 2, 3):
            orbits, parameters = active_orbit_indices(batch, dimension)
            if orbits.numel() == 0:
                continue
            metrics = batch.orbit_metric[
                orbits, :dimension, :dimension
            ].to(compute_dtype)
            orbit_inner[orbits] = torch.einsum(
                "ni,nij,nj->n",
                left[parameters].to(compute_dtype),
                metrics,
                right[parameters].to(compute_dtype),
            )
        structure_inner = torch.zeros(
            batch.batch_size, device=left.device, dtype=compute_dtype
        )
        structure_inner.index_add_(0, batch.orbit_to_structure, orbit_inner)
    return structure_inner


def tangent_cartesian_mean_square(
    batch: PackedASUBatch, tangent: torch.Tensor
) -> torch.Tensor:
    """Return the per-structure member-sum Cartesian mean square."""

    structure_squared = tangent_cartesian_inner(batch, tangent, tangent)
    atom_counts = (batch.atom_ptr[1:] - batch.atom_ptr[:-1]).to(tangent.dtype)
    return structure_squared / atom_counts.clamp_min(1.0)


def tangent_cartesian_rms(
    batch: PackedASUBatch, tangent: torch.Tensor
) -> torch.Tensor:
    return torch.sqrt(tangent_cartesian_mean_square(batch, tangent).clamp_min(0.0))


def limit_cartesian_rms(
    batch: PackedASUBatch,
    tangent: torch.Tensor,
    maximum_rms_angstrom: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if maximum_rms_angstrom <= 0.0:
        raise ValueError("maximum Cartesian RMS must be positive")
    rms = tangent_cartesian_rms(batch, tangent)
    scale = torch.clamp(
        tangent.new_tensor(maximum_rms_angstrom) / rms.clamp_min(1.0e-12),
        max=1.0,
    )
    return (
        tangent * scale[batch.parameter_to_structure],
        rms,
        rms > maximum_rms_angstrom,
    )


__all__ = [
    "CONVENTIONAL_HALL_CELL_V1",
    "DIFFUSION_METRIC_CONTRACTS",
    "FULL_ASU_STATE_V1",
    "MEMBER_MEAN_CARTESIAN_V1",
    "MEMBER_SUM_CARTESIAN_V1",
    "NATIVE_MEMBER_SUM_SCORE_V1",
    "STATE_QUOTIENT_CONTRACTS",
    "TRANSLATION_QUOTIENT_V1",
    "apply_state_quotient_covector",
    "apply_state_quotient_tangent",
    "diffusion_to_native_covector",
    "integer_dual_quadratic",
    "limit_cartesian_rms",
    "metric_inverse_multiply",
    "metric_multiply",
    "metric_contract_metadata",
    "metric_whiten_noise",
    "orbit_metric_view",
    "parameter_multiplicity",
    "project_translation_quotient_covector",
    "project_translation_quotient_tangent",
    "tangent_cartesian_inner",
    "tangent_cartesian_mean_square",
    "tangent_cartesian_rms",
]
