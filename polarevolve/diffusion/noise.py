"""Wrapped-normal corruption and negative-score targets."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from polarevolve.data.batch import PackedASUBatch
from polarevolve.diffusion.metric import (
    MEMBER_MEAN_CARTESIAN_V1,
    MEMBER_SUM_CARTESIAN_V1,
    TRANSLATION_QUOTIENT_V1,
    active_orbit_indices,
    apply_state_quotient_covector,
    apply_state_quotient_tangent,
    diffusion_to_native_covector,
    integer_dual_quadratic,
    metric_whiten_noise,
    orbit_metric_view,
)
from polarevolve.diffusion.periodic import integer_vectors
from polarevolve.diffusion.schedule import certified_fourier_search_radius


@dataclass(frozen=True)
class MetricVENoiseSample:
    noisy_u: torch.Tensor
    target_negative_score: torch.Tensor
    sigma_by_structure: torch.Tensor
    standard_noise: torch.Tensor


def wrap_unit_interval(values: torch.Tensor) -> torch.Tensor:
    return torch.remainder(values, 1.0)


def periodic_unit_delta(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return torch.remainder(left - right + 0.5, 1.0) - 0.5


def wrapped_metric_negative_score(
    batch: PackedASUBatch,
    displacement: torch.Tensor,
    sigma_by_structure: torch.Tensor,
    *,
    image_radius: int = 4,
    maximum_radius: int = 24,
    fourier_switch: float = 0.05,
    metric_convention: str = MEMBER_SUM_CARTESIAN_V1,
) -> torch.Tensor:
    """Wrapped-normal negative score from finite stable primal/dual sums."""

    if displacement.shape != batch.parameter_shape:
        raise ValueError("displacement must match packed ASU parameters")
    sigma = torch.as_tensor(
        sigma_by_structure, device=batch.lattice.device, dtype=batch.lattice.dtype
    ).reshape(-1)
    if sigma.shape != (batch.batch_size,) or bool((sigma <= 0).any().item()):
        raise ValueError("sigma must contain one positive value per structure")
    if image_radius < 1 or maximum_radius < image_radius or not 0.0 < fourier_switch < 1.0:
        raise ValueError("invalid wrapped-normal numerical bounds")
    output = torch.zeros_like(displacement)
    compute_dtype = (
        torch.float64 if displacement.dtype == torch.float64 else torch.float32
    )
    for dimension in (1, 2, 3):
        orbits, parameters = active_orbit_indices(batch, dimension)
        if orbits.numel() == 0:
            continue
        metric = orbit_metric_view(batch, metric_convention)[
            orbits, :dimension, :dimension
        ].to(compute_dtype)
        inverse = torch.linalg.inv(metric)
        local_sigma = sigma[batch.orbit_to_structure[orbits]].to(compute_dtype)
        radii = certified_fourier_search_radius(
            metric, minimum_radius=image_radius
        )
        if bool((radii > maximum_radius).any().item()):
            required = int(radii.max().item())
            clipped = int((radii > maximum_radius).count_nonzero().item())
            raise ValueError(
                "wrapped-normal finite Fourier support is below the conservative "
                f"shortest-vector bound: {clipped} orbit(s) require radius up to "
                f"{required}, above maximum_radius={maximum_radius}"
            )
        for radius_tensor in torch.unique(radii):
            radius = int(radius_tensor.item())
            local = torch.nonzero(radii == radius_tensor, as_tuple=False).reshape(-1)
            vectors = torch.tensor(
                integer_vectors(dimension, radius),
                device=displacement.device,
                dtype=compute_dtype,
            )
            dual_quadratic = integer_dual_quadratic(inverse[local], vectors)
            coefficient = torch.exp(
                -2.0
                * math.pi**2
                * local_sigma[local].square()
                * dual_quadratic.min(dim=1).values
            )
            use_fourier = coefficient <= fourier_switch
            use_primal = ~use_fourier
            if bool(use_fourier.any().item()):
                chosen_local = torch.nonzero(use_fourier, as_tuple=False).reshape(-1)
                chosen = local[chosen_local]
                delta = displacement[parameters[chosen]].to(compute_dtype)
                quadratic = dual_quadratic[use_fourier]
                weights = torch.exp(
                    -2.0
                    * math.pi**2
                    * local_sigma[chosen, None].square()
                    * quadratic
                )
                phase = 2.0 * math.pi * torch.einsum("ni,ki->nk", delta, vectors)
                density = 1.0 + torch.sum(weights * torch.cos(phase), dim=1)
                numerator = 2.0 * math.pi * torch.einsum(
                    "nk,nk,ki->ni", weights, torch.sin(phase), vectors
                )
                valid = torch.isfinite(density) & (density > 1.0e-10)
                if bool(valid.any().item()):
                    output[parameters[chosen[valid]]] = (
                        numerator[valid] / density[valid, None]
                    ).to(output.dtype)
                if bool((~valid).any().item()):
                    use_primal[chosen_local[~valid]] = True
            if bool(use_primal.any().item()):
                chosen = local[use_primal]
                axis = torch.arange(
                    -radius,
                    radius + 1,
                    device=displacement.device,
                    dtype=compute_dtype,
                )
                images = torch.cartesian_prod(*([axis] * dimension)).reshape(-1, dimension)
                shifted = (
                    displacement[parameters[chosen]].to(compute_dtype)[:, None, :]
                    + images[None, :, :]
                )
                metric_shifted = torch.einsum(
                    "nij,nkj->nki", metric[chosen], shifted
                )
                quadratic = torch.einsum("nki,nki->nk", shifted, metric_shifted)
                log_weights = -0.5 * quadratic / local_sigma[chosen, None].square()
                weights = torch.softmax(log_weights, dim=1)
                score = torch.sum(
                    weights[..., None]
                    * metric_shifted
                    / local_sigma[chosen, None, None].square(),
                    dim=1,
                )
                output[parameters[chosen]] = score.to(output.dtype)
    if not bool(torch.isfinite(output).all().item()):
        raise FloatingPointError("wrapped-normal score is non-finite")
    return output


def corrupt_parameters(
    batch: PackedASUBatch,
    sigma_by_structure: torch.Tensor,
    *,
    standard_noise: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    state_quotient: str = TRANSLATION_QUOTIENT_V1,
    diffusion_metric: str = MEMBER_SUM_CARTESIAN_V1,
) -> MetricVENoiseSample:
    sigma = torch.as_tensor(
        sigma_by_structure, device=batch.lattice.device, dtype=batch.lattice.dtype
    ).reshape(-1)
    if sigma.numel() == 1:
        sigma = sigma.expand(batch.batch_size)
    if sigma.shape != (batch.batch_size,) or bool((sigma <= 0).any().item()):
        raise ValueError("sigma must contain one positive value per structure")
    noise = (
        torch.randn(
            batch.parameter_shape,
            device=batch.lattice.device,
            dtype=batch.lattice.dtype,
            generator=generator,
        )
        if standard_noise is None
        else torch.as_tensor(
            standard_noise, device=batch.lattice.device, dtype=batch.lattice.dtype
        )
    )
    if noise.shape != batch.parameter_shape:
        raise ValueError("standard noise must match packed ASU parameters")
    whitened = metric_whiten_noise(
        batch, noise, metric_convention=diffusion_metric
    )
    if diffusion_metric == MEMBER_MEAN_CARTESIAN_V1:
        whitened = apply_state_quotient_tangent(
            batch,
            whitened,
            state_quotient,
            metric_convention=diffusion_metric,
        )
    noisy = wrap_unit_interval(
        batch.require_target() + sigma[batch.parameter_to_structure] * whitened
    )
    diffusion_score = wrapped_metric_negative_score(
        batch,
        periodic_unit_delta(noisy, batch.require_target()),
        sigma,
        metric_convention=diffusion_metric,
    )
    diffusion_score = apply_state_quotient_covector(
        batch,
        diffusion_score,
        state_quotient,
        metric_convention=diffusion_metric,
    )
    target = diffusion_to_native_covector(batch, diffusion_score, diffusion_metric)
    return MetricVENoiseSample(noisy, target, sigma, noise)


def analytic_oracle_negative_score(
    batch: PackedASUBatch,
    u: torch.Tensor,
    sigma_by_structure: torch.Tensor,
    *,
    state_quotient: str = TRANSLATION_QUOTIENT_V1,
    diffusion_metric: str = MEMBER_SUM_CARTESIAN_V1,
) -> torch.Tensor:
    """Exact target-conditioned score for coordinate-recovery diagnostics only."""

    if u.shape != batch.parameter_shape:
        raise ValueError("oracle state must match packed ASU parameters")
    diffusion_score = wrapped_metric_negative_score(
        batch,
        periodic_unit_delta(u, batch.require_target()),
        sigma_by_structure,
        metric_convention=diffusion_metric,
    )
    diffusion_score = apply_state_quotient_covector(
        batch,
        diffusion_score,
        state_quotient,
        metric_convention=diffusion_metric,
    )
    return diffusion_to_native_covector(batch, diffusion_score, diffusion_metric)


__all__ = [
    "MetricVENoiseSample",
    "analytic_oracle_negative_score",
    "corrupt_parameters",
    "periodic_unit_delta",
    "wrap_unit_interval",
    "wrapped_metric_negative_score",
]
