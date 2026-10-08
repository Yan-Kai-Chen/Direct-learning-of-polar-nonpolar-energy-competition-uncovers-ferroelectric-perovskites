"""Explicit VE reverse-process updates in the ASU Cartesian metric."""

from __future__ import annotations

import torch

from polarevolve.data.batch import PackedASUBatch
from polarevolve.diffusion.metric import (
    MEMBER_SUM_CARTESIAN_V1,
    TRANSLATION_QUOTIENT_V1,
    apply_state_quotient_tangent,
    metric_inverse_multiply,
    metric_whiten_noise,
)

LEGACY_MEAN = "legacy_mean"
PROBABILITY_FLOW_ODE = "probability_flow_ode"
REVERSE_SDE = "reverse_sde"
SAMPLER_INTEGRATORS = (LEGACY_MEAN, PROBABILITY_FLOW_ODE, REVERSE_SDE)


def _variance_decrement(
    batch: PackedASUBatch,
    sigma: torch.Tensor,
    next_sigma: torch.Tensor,
) -> torch.Tensor:
    current = torch.as_tensor(
        sigma, device=batch.lattice.device, dtype=batch.lattice.dtype
    )
    following = torch.as_tensor(
        next_sigma, device=batch.lattice.device, dtype=batch.lattice.dtype
    )
    if current.ndim == 0:
        delta = current.square() - following.square()
        delta = delta.expand(batch.parameter_shape)
    else:
        current = current.reshape(-1)
        following = following.reshape(-1)
        if current.shape != (batch.batch_size,) or following.shape != current.shape:
            raise ValueError("sigma vectors must match batch size")
        delta = (current.square() - following.square())[batch.parameter_to_structure]
    if bool(((delta < -1.0e-8) | ~torch.isfinite(delta)).any().item()):
        raise ValueError("reverse schedule must not increase sigma")
    return delta.clamp_min(0.0)


def reverse_process_update(
    batch: PackedASUBatch,
    negative_score: torch.Tensor,
    sigma: torch.Tensor,
    next_sigma: torch.Tensor,
    *,
    integrator: str,
    generator: torch.Generator | None = None,
    standard_noise: torch.Tensor | None = None,
    state_quotient: str = TRANSLATION_QUOTIENT_V1,
    diffusion_metric: str = MEMBER_SUM_CARTESIAN_V1,
) -> torch.Tensor:
    """Return one legacy, probability-flow, or Euler-Maruyama VE update.

    The network predicts the negative score covector. The two mathematically
    complete paths are the half-drift probability-flow ODE and the full-drift
    reverse SDE with metric-whitened noise. ``legacy_mean`` preserves the
    former production behavior only as an explicit diagnostic baseline.
    """

    if integrator not in SAMPLER_INTEGRATORS:
        raise ValueError(f"unsupported sampler integrator: {integrator!r}")
    if negative_score.shape != batch.parameter_shape:
        raise ValueError("negative score must match packed ASU parameters")
    delta = _variance_decrement(batch, sigma, next_sigma)
    drift_scale = 0.5 if integrator == PROBABILITY_FLOW_ODE else 1.0
    update = apply_state_quotient_tangent(
        batch,
        -drift_scale * delta * metric_inverse_multiply(batch, negative_score),
        state_quotient,
        metric_convention=diffusion_metric,
    )
    if integrator != REVERSE_SDE:
        if standard_noise is not None:
            raise ValueError("standard_noise is only valid for reverse_sde")
        return update
    noise = (
        torch.randn(
            batch.parameter_shape,
            device=batch.lattice.device,
            dtype=batch.lattice.dtype,
            generator=generator,
        )
        if standard_noise is None
        else torch.as_tensor(
            standard_noise,
            device=batch.lattice.device,
            dtype=batch.lattice.dtype,
        )
    )
    if noise.shape != batch.parameter_shape:
        raise ValueError("standard noise must match packed ASU parameters")
    stochastic = apply_state_quotient_tangent(
        batch,
        torch.sqrt(delta)
        * metric_whiten_noise(
            batch, noise, metric_convention=diffusion_metric
        ),
        state_quotient,
        metric_convention=diffusion_metric,
    )
    return update + stochastic


__all__ = [
    "LEGACY_MEAN",
    "PROBABILITY_FLOW_ODE",
    "REVERSE_SDE",
    "SAMPLER_INTEGRATORS",
    "reverse_process_update",
]
