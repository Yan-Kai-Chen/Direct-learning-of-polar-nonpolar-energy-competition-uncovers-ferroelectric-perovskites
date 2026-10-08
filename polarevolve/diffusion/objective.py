"""Dual-metric score objective and clean-parameter estimate."""

from __future__ import annotations

import torch

from polarevolve.data.batch import PackedASUBatch
from polarevolve.diffusion.metric import (
    MEMBER_MEAN_CARTESIAN_V1,
    MEMBER_SUM_CARTESIAN_V1,
    TRANSLATION_QUOTIENT_V1,
    apply_state_quotient_tangent,
    metric_inverse_multiply,
)
from polarevolve.diffusion.noise import wrap_unit_interval


def scale_score_covector(
    score: torch.Tensor,
    batch: PackedASUBatch,
    sigma_by_structure: torch.Tensor,
) -> torch.Tensor:
    """Map a physical score covector to the sigma-scaled training convention."""

    if score.shape != batch.parameter_shape:
        raise ValueError("score must match packed ASU parameters")
    sigma = torch.as_tensor(
        sigma_by_structure, device=score.device, dtype=score.dtype
    ).reshape(-1)
    if sigma.shape != (batch.batch_size,) or bool((sigma <= 0).any().item()):
        raise ValueError("sigma must contain one positive value per structure")
    return score * sigma[batch.parameter_to_structure]


def dual_metric_covector_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    batch: PackedASUBatch,
    *,
    diffusion_metric: str = MEMBER_SUM_CARTESIAN_V1,
) -> torch.Tensor:
    """Structure-balanced dual-metric MSE without additional sigma weighting."""

    if prediction.shape != target.shape or prediction.shape != batch.parameter_shape:
        raise ValueError("covectors must match packed ASU parameters")
    if batch.num_parameters == 0:
        return prediction.sum() * 0.0
    error = prediction - target
    contribution = error * metric_inverse_multiply(batch, error)
    totals = prediction.new_zeros(batch.batch_size)
    totals.index_add_(0, batch.parameter_to_structure, contribution)
    counts = prediction.new_zeros(batch.batch_size)
    if diffusion_metric == MEMBER_MEAN_CARTESIAN_V1:
        active_atoms = batch.orbit_multiplicities.to(prediction.dtype) * (
            batch.orbit_dimensions > 0
        ).to(prediction.dtype)
        counts.index_add_(0, batch.orbit_to_structure, active_atoms)
    elif diffusion_metric == MEMBER_SUM_CARTESIAN_V1:
        counts.index_add_(
            0, batch.parameter_to_structure, torch.ones_like(contribution)
        )
    else:
        raise ValueError(f"unsupported diffusion metric: {diffusion_metric!r}")
    active = counts > 0
    return (totals[active] / counts[active]).mean()


def dual_metric_score_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    batch: PackedASUBatch,
    sigma_by_structure: torch.Tensor,
    *,
    diffusion_metric: str = MEMBER_SUM_CARTESIAN_V1,
) -> torch.Tensor:
    """Structure-balanced NCSN loss with fixed sigma-squared weighting."""

    scaled_prediction = scale_score_covector(
        prediction, batch, sigma_by_structure
    )
    scaled_target = scale_score_covector(target, batch, sigma_by_structure)
    return dual_metric_covector_loss(
        scaled_prediction,
        scaled_target,
        batch,
        diffusion_metric=diffusion_metric,
    )


def dual_metric_score_calibration(
    prediction: torch.Tensor,
    target: torch.Tensor,
    batch: PackedASUBatch,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return structure-balanced dual-metric calibration vectors.

    Each returned entry belongs to one active structure.  Keeping structures as
    the reduction unit matches the score loss and prevents a large ASU from
    dominating a batch-level cosine through its parameter count.
    """

    if prediction.shape != target.shape or prediction.shape != batch.parameter_shape:
        raise ValueError("score tensors must match packed ASU parameters")
    if batch.num_parameters == 0:
        empty = prediction.new_empty((0,))
        return empty, empty
    prediction_primal = metric_inverse_multiply(batch, prediction)
    target_primal = metric_inverse_multiply(batch, target)
    structure = batch.parameter_to_structure
    dot = prediction.new_zeros(batch.batch_size)
    prediction_squared = prediction.new_zeros(batch.batch_size)
    target_squared = prediction.new_zeros(batch.batch_size)
    dot.index_add_(0, structure, prediction * target_primal)
    prediction_squared.index_add_(0, structure, prediction * prediction_primal)
    target_squared.index_add_(0, structure, target * target_primal)
    prediction_norm = torch.sqrt(prediction_squared.clamp_min(0.0))
    target_norm = torch.sqrt(target_squared.clamp_min(0.0))
    epsilon = prediction.new_tensor(1.0e-12)
    active = target_norm > epsilon
    cosine = dot[active] / (prediction_norm[active] * target_norm[active]).clamp_min(
        epsilon
    )
    ratio = prediction_norm[active] / target_norm[active]
    return cosine.clamp(min=-1.0, max=1.0), ratio


def estimate_clean_parameters(
    batch: PackedASUBatch,
    noisy_u: torch.Tensor,
    negative_score: torch.Tensor,
    sigma_by_structure: torch.Tensor,
    *,
    state_quotient: str = TRANSLATION_QUOTIENT_V1,
    diffusion_metric: str = MEMBER_SUM_CARTESIAN_V1,
) -> torch.Tensor:
    sigma = torch.as_tensor(
        sigma_by_structure, device=noisy_u.device, dtype=noisy_u.dtype
    ).reshape(-1)
    update = apply_state_quotient_tangent(
        batch,
        sigma[batch.parameter_to_structure].square()
        * metric_inverse_multiply(batch, negative_score),
        state_quotient,
        metric_convention=diffusion_metric,
    )
    return wrap_unit_interval(noisy_u - update)


__all__ = [
    "dual_metric_covector_loss",
    "dual_metric_score_calibration",
    "dual_metric_score_loss",
    "estimate_clean_parameters",
    "scale_score_covector",
]
