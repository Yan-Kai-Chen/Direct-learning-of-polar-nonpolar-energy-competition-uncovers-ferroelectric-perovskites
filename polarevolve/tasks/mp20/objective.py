"""MP20 score objective with hard symmetry and optional physics regularization."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from polarevolve.data.batch import PackedASUBatch
from polarevolve.diffusion.metric import (
    DIFFUSION_METRIC_CONTRACTS,
    MEMBER_SUM_CARTESIAN_V1,
    TRANSLATION_QUOTIENT_V1,
    apply_state_quotient_tangent,
    limit_cartesian_rms,
    metric_whiten_noise,
    orbit_metric_view,
    tangent_cartesian_mean_square,
    tangent_cartesian_rms,
)
from polarevolve.diffusion.noise import (
    analytic_oracle_negative_score,
    corrupt_parameters,
    periodic_unit_delta,
    wrap_unit_interval,
)
from polarevolve.diffusion.objective import (
    dual_metric_score_calibration,
    dual_metric_score_loss,
    estimate_clean_parameters,
)
from polarevolve.diffusion.schedule import (
    FIXED_SIGMA_WINDOW_V1,
    TERMINAL_ADAPTIVE_SIGMA_V1,
    TRAINING_SIGMA_MODES,
    SigmaSchedule,
    minimum_terminal_sigma,
)
from polarevolve.diffusion.update import PROBABILITY_FLOW_ODE, reverse_process_update
from polarevolve.models.score import HardConditionScoreNetwork
from polarevolve.guidance.physics import PhysicsGuidance, PhysicsGuidanceConfig


@dataclass(frozen=True)
class MP20StepOutput:
    loss: torch.Tensor
    score_loss: torch.Tensor
    rollout_score_loss: torch.Tensor
    rollout_x0_cartesian_mse: torch.Tensor
    one_step_cartesian_rmsd: torch.Tensor
    input_perturbation_fraction: torch.Tensor
    input_perturbation_rmsd: torch.Tensor
    rollout_clipped_fraction: torch.Tensor
    sigma_mean: torch.Tensor
    sigma_max_mean: torch.Tensor
    sigma_by_structure: torch.Tensor
    score_dual_cosine_by_structure: torch.Tensor
    score_dual_norm_ratio_by_structure: torch.Tensor
    identity_cartesian_rmsd: torch.Tensor
    denoising_rmsd_improvement: torch.Tensor
    denoising_relative_improvement: torch.Tensor
    physics_loss: torch.Tensor
    physics_overlap_excess: torch.Tensor
    physics_bond_radius_excess: torch.Tensor
    physics_bond_valence_excess: torch.Tensor
    physics_bond_valence_applicable_fraction: torch.Tensor
    physics_coordination_excess: torch.Tensor
    physics_coordination_coverage: torch.Tensor
    predicted_u0: torch.Tensor | None = None

    @property
    def score_dual_cosine(self) -> torch.Tensor:
        if self.score_dual_cosine_by_structure.numel() == 0:
            return self.loss.detach() * 0.0
        return self.score_dual_cosine_by_structure.mean()

    @property
    def score_dual_norm_ratio(self) -> torch.Tensor:
        if self.score_dual_norm_ratio_by_structure.numel() == 0:
            return self.loss.detach() * 0.0
        return self.score_dual_norm_ratio_by_structure.mean()


class MP20ScoreObjective(nn.Module):
    """Own the MP20 loss while delegating all diffusion math to process.py."""

    def __init__(
        self,
        *,
        model: HardConditionScoreNetwork,
        schedule: SigmaSchedule,
        terminal_mixing_tolerance: float = 1.0e-3,
        sigma_sampling_mode: str = TERMINAL_ADAPTIVE_SIGMA_V1,
        diffusion_metric: str = MEMBER_SUM_CARTESIAN_V1,
        input_perturbation_probability: float = 0.0,
        input_perturbation_scale: float = 0.0,
        rollout_steps: int = 1,
        rollout_score_weight: float = 0.0,
        rollout_x0_weight: float = 0.0,
        rollout_maximum_step_rms_angstrom: float = 0.12,
        physics_auxiliary_weight: float = 0.0,
        physics_config: PhysicsGuidanceConfig = PhysicsGuidanceConfig(),
    ) -> None:
        super().__init__()
        self.model = model
        self.schedule = schedule
        self.terminal_mixing_tolerance = float(terminal_mixing_tolerance)
        self.sigma_sampling_mode = str(sigma_sampling_mode)
        self.diffusion_metric = str(diffusion_metric)
        self.input_perturbation_probability = float(input_perturbation_probability)
        self.input_perturbation_scale = float(input_perturbation_scale)
        self.rollout_steps = int(rollout_steps)
        self.rollout_score_weight = float(rollout_score_weight)
        self.rollout_x0_weight = float(rollout_x0_weight)
        self.rollout_maximum_step_rms_angstrom = float(rollout_maximum_step_rms_angstrom)
        self.physics_auxiliary_weight = float(physics_auxiliary_weight)
        self.physics_guidance = (
            PhysicsGuidance(physics_config) if self.physics_auxiliary_weight > 0.0 else None
        )
        if not 0.0 <= self.input_perturbation_probability <= 1.0:
            raise ValueError("input perturbation probability must lie in [0,1]")
        if self.sigma_sampling_mode not in TRAINING_SIGMA_MODES:
            raise ValueError("unsupported training sigma sampling mode")
        if self.diffusion_metric not in DIFFUSION_METRIC_CONTRACTS:
            raise ValueError("unsupported diffusion metric")
        if self.input_perturbation_scale < 0.0:
            raise ValueError("input perturbation scale must be non-negative")
        if self.rollout_steps not in {1, 2}:
            raise ValueError("rollout_steps must be one or two")
        if self.rollout_score_weight < 0.0 or self.rollout_x0_weight < 0.0:
            raise ValueError("rollout loss weights must be non-negative")
        if self.rollout_maximum_step_rms_angstrom <= 0.0:
            raise ValueError("rollout maximum step RMS must be positive")
        if self.physics_auxiliary_weight < 0.0:
            raise ValueError("physics auxiliary weight must be non-negative")

    def training_sigmas(
        self, batch: PackedASUBatch, *, generator: torch.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        maximum = batch.clean_u.new_full((batch.batch_size,), self.schedule.sigma_max)
        if self.sigma_sampling_mode != FIXED_SIGMA_WINDOW_V1:
            required = minimum_terminal_sigma(
                batch,
                tolerance=self.terminal_mixing_tolerance,
                orbit_metric=orbit_metric_view(batch, self.diffusion_metric),
            )
            maximum = torch.maximum(required, maximum)
            violating = torch.nonzero(
                maximum > self.schedule.sigma_max_cap, as_tuple=False
            ).reshape(-1)
            if violating.numel() > 0:
                indices = violating[:8].tolist()
                details = ", ".join(
                    f"{batch.material_ids[index]}:required={float(required[index]):.6g}"
                    for index in indices
                )
                remaining = int(violating.numel()) - len(indices)
                if remaining > 0:
                    details = f"{details}, ... ({remaining} more)"
                raise ValueError(
                    "training record requires sigma above "
                    f"sigma_max_cap={self.schedule.sigma_max_cap:.6g}; {details}"
                )
        uniform = torch.rand(
            (batch.batch_size,),
            device=batch.clean_u.device,
            dtype=batch.clean_u.dtype,
            generator=generator,
        )
        sigma = torch.exp(
            torch.log(batch.clean_u.new_tensor(self.schedule.sigma_min))
            + uniform
            * (torch.log(maximum) - torch.log(batch.clean_u.new_tensor(self.schedule.sigma_min)))
        )
        return sigma, maximum

    @staticmethod
    def _one_step_rmsd(batch: PackedASUBatch, predicted_u0: torch.Tensor) -> torch.Tensor:
        delta = periodic_unit_delta(predicted_u0, batch.clean_u)
        return tangent_cartesian_rms(
            batch,
            apply_state_quotient_tangent(batch, delta, TRANSLATION_QUOTIENT_V1),
        )

    def _input_perturbation(
        self,
        batch: PackedASUBatch,
        sigma: torch.Tensor,
        *,
        generator: torch.Generator | None,
        standard_noise: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        zeros = torch.zeros_like(batch.clean_u)
        scalar_zero = batch.clean_u.new_zeros(())
        if (
            self.input_perturbation_probability == 0.0
            or self.input_perturbation_scale == 0.0
            or batch.num_parameters == 0
        ):
            return zeros, scalar_zero, scalar_zero
        mask = (
            torch.rand(
                (batch.batch_size,),
                device=batch.clean_u.device,
                dtype=batch.clean_u.dtype,
                generator=generator,
            )
            < self.input_perturbation_probability
        )
        raw = (
            torch.randn(
                batch.clean_u.shape,
                device=batch.clean_u.device,
                dtype=batch.clean_u.dtype,
                generator=generator,
            )
            if standard_noise is None
            else torch.as_tensor(
                standard_noise,
                device=batch.clean_u.device,
                dtype=batch.clean_u.dtype,
            )
        )
        if raw.shape != batch.clean_u.shape:
            raise ValueError("input perturbation noise must match ASU parameters")
        scale = sigma * self.input_perturbation_scale * mask.to(sigma.dtype)
        tangent = apply_state_quotient_tangent(
            batch,
            scale[batch.parameter_to_structure]
            * metric_whiten_noise(batch, raw, metric_convention=self.diffusion_metric),
            TRANSLATION_QUOTIENT_V1,
            metric_convention=self.diffusion_metric,
        )
        return (
            tangent,
            mask.float().mean(),
            tangent_cartesian_rms(batch, tangent).mean(),
        )

    def _rollout_state(
        self,
        batch: PackedASUBatch,
        noisy_u: torch.Tensor,
        sigma: torch.Tensor,
        maximum: torch.Tensor,
        condition: torch.Tensor | None,
        atom_reference_fractional: torch.Tensor | None,
        atom_reference_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        current_u = noisy_u.detach().float()
        current_sigma = sigma.detach().float()
        clipped_total = current_u.new_zeros(())
        with (
            torch.no_grad(),
            torch.autocast(
                device_type=batch.clean_u.device.type,
                enabled=False,
            ),
        ):
            for _ in range(self.rollout_steps):
                next_sigma = self.schedule.adjacent_lower(
                    current_sigma,
                    maximum,
                )
                score = self.model(
                    batch,
                    current_u,
                    current_sigma,
                    condition=condition,
                    atom_reference_fractional=atom_reference_fractional,
                    atom_reference_mask=atom_reference_mask,
                ).float()
                proposed = reverse_process_update(
                    batch,
                    score,
                    current_sigma,
                    next_sigma,
                    integrator=PROBABILITY_FLOW_ODE,
                    state_quotient=TRANSLATION_QUOTIENT_V1,
                    diffusion_metric=self.diffusion_metric,
                )
                update, _, clipped = limit_cartesian_rms(
                    batch,
                    proposed,
                    self.rollout_maximum_step_rms_angstrom,
                )
                clipped_total += clipped.float().mean()
                current_u = wrap_unit_interval(current_u + update)
                current_sigma = next_sigma
        return current_u.detach(), current_sigma.detach(), clipped_total / self.rollout_steps

    def forward(
        self,
        batch: PackedASUBatch,
        *,
        generator: torch.Generator | None = None,
        sigma_by_structure: torch.Tensor | None = None,
        standard_noise: torch.Tensor | None = None,
        input_perturbation_enabled: bool = True,
        input_perturbation_standard_noise: torch.Tensor | None = None,
        rollout_enabled: bool = False,
        physics_weight_scale: float = 1.0,
        condition: torch.Tensor | None = None,
        atom_reference_fractional: torch.Tensor | None = None,
        atom_reference_mask: torch.Tensor | None = None,
    ) -> MP20StepOutput:
        if not 0.0 <= physics_weight_scale <= 1.0:
            raise ValueError("physics weight scale must lie in [0,1]")
        if sigma_by_structure is None:
            sigma, maximum = self.training_sigmas(batch, generator=generator)
        else:
            sigma = torch.as_tensor(
                sigma_by_structure,
                device=batch.clean_u.device,
                dtype=batch.clean_u.dtype,
            ).reshape(-1)
            maximum = torch.maximum(
                sigma,
                sigma.new_full(sigma.shape, self.schedule.sigma_max),
            )
        device_type = batch.clean_u.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            if input_perturbation_enabled:
                perturbation, perturbed_fraction, perturbation_rmsd = self._input_perturbation(
                    batch,
                    sigma.float(),
                    generator=generator,
                    standard_noise=input_perturbation_standard_noise,
                )
            else:
                perturbation = torch.zeros_like(batch.clean_u)
                perturbed_fraction = batch.clean_u.new_zeros(())
                perturbation_rmsd = batch.clean_u.new_zeros(())
            noise = corrupt_parameters(
                batch,
                sigma.float(),
                standard_noise=standard_noise,
                generator=generator,
                state_quotient=TRANSLATION_QUOTIENT_V1,
                diffusion_metric=self.diffusion_metric,
            )
            model_u = wrap_unit_interval(noise.noisy_u + perturbation)
        prediction = self.model(
            batch,
            model_u,
            sigma,
            condition=condition,
            atom_reference_fractional=atom_reference_fractional,
            atom_reference_mask=atom_reference_mask,
        )
        with torch.autocast(device_type=device_type, enabled=False):
            prediction_fp32 = prediction.float()
            score_loss = dual_metric_score_loss(
                prediction_fp32,
                noise.target_negative_score.float(),
                batch,
                sigma.float(),
                diffusion_metric=self.diffusion_metric,
            )
            score_cosine, score_norm_ratio = dual_metric_score_calibration(
                prediction_fp32,
                noise.target_negative_score.float(),
                batch,
            )
            predicted_u0 = estimate_clean_parameters(
                batch,
                model_u,
                prediction_fp32,
                sigma.float(),
                state_quotient=TRANSLATION_QUOTIENT_V1,
                diffusion_metric=self.diffusion_metric,
            )
            rmsd = self._one_step_rmsd(batch, predicted_u0)
            identity_rmsd = self._one_step_rmsd(batch, model_u)
            identity_mean = identity_rmsd.mean()
            rmsd_mean = rmsd.mean()
            denoising_improvement = identity_mean - rmsd_mean
            denoising_relative_improvement = denoising_improvement / identity_mean.clamp_min(
                1.0e-12
            )
        rollout_score_loss = prediction_fp32.sum() * 0.0
        rollout_x0_mse = prediction_fp32.sum() * 0.0
        rollout_clipped = prediction_fp32.new_zeros(())
        if rollout_enabled:
            rollout_u, rollout_sigma, rollout_clipped = self._rollout_state(
                batch,
                model_u,
                sigma.float(),
                maximum.float(),
                condition,
                atom_reference_fractional,
                atom_reference_mask,
            )
            with torch.autocast(device_type=device_type, enabled=False):
                rollout_target = analytic_oracle_negative_score(
                    batch,
                    rollout_u,
                    rollout_sigma,
                    state_quotient=TRANSLATION_QUOTIENT_V1,
                    diffusion_metric=self.diffusion_metric,
                )
            rollout_prediction = self.model(
                batch,
                rollout_u,
                rollout_sigma,
                condition=condition,
                atom_reference_fractional=atom_reference_fractional,
                atom_reference_mask=atom_reference_mask,
            )
            with torch.autocast(device_type=device_type, enabled=False):
                rollout_prediction_fp32 = rollout_prediction.float()
                rollout_score_loss = dual_metric_score_loss(
                    rollout_prediction_fp32,
                    rollout_target.float(),
                    batch,
                    rollout_sigma,
                    diffusion_metric=self.diffusion_metric,
                )
                rollout_u0 = estimate_clean_parameters(
                    batch,
                    rollout_u,
                    rollout_prediction_fp32,
                    rollout_sigma,
                    state_quotient=TRANSLATION_QUOTIENT_V1,
                    diffusion_metric=self.diffusion_metric,
                )
                rollout_x0_mse = tangent_cartesian_mean_square(
                    batch,
                    apply_state_quotient_tangent(
                        batch,
                        periodic_unit_delta(rollout_u0, batch.clean_u),
                        TRANSLATION_QUOTIENT_V1,
                    ),
                ).mean()
        physics_loss = prediction_fp32.sum() * 0.0
        physics_overlap = prediction_fp32.sum() * 0.0
        physics_bond = prediction_fp32.sum() * 0.0
        physics_bvs = prediction_fp32.sum() * 0.0
        physics_bvs_fraction = prediction_fp32.new_zeros(())
        physics_cn = prediction_fp32.sum() * 0.0
        physics_cn_coverage = prediction_fp32.new_zeros(())
        if self.physics_guidance is not None:
            with torch.autocast(device_type=device_type, enabled=False):
                physics = self.physics_guidance.target_relative_loss(
                    batch, predicted_u0.float(), sigma.float()
                )
            physics_loss = physics.loss
            physics_overlap = physics.overlap_excess
            physics_bond = physics.bond_radius_excess
            physics_bvs = physics.bond_valence_excess
            physics_bvs_fraction = physics.bond_valence_applicable_fraction
            physics_cn = physics.coordination_excess
            physics_cn_coverage = physics.coordination_coverage
        loss = (
            score_loss
            + self.rollout_score_weight * rollout_score_loss
            + self.rollout_x0_weight * rollout_x0_mse
            + self.physics_auxiliary_weight * physics_weight_scale * physics_loss
        )
        finite_values = (
            loss,
            score_loss,
            rollout_score_loss,
            rollout_x0_mse,
            rmsd,
            perturbation_rmsd,
            identity_rmsd,
            denoising_improvement,
            denoising_relative_improvement,
            physics_loss,
            physics_overlap,
            physics_bond,
            physics_bvs,
        )
        finite = torch.stack(tuple(torch.isfinite(value).all() for value in finite_values)).all()
        if not bool(finite.item()):
            raise FloatingPointError("MP20 score objective is non-finite")
        return MP20StepOutput(
            loss=loss,
            score_loss=score_loss,
            rollout_score_loss=rollout_score_loss,
            rollout_x0_cartesian_mse=rollout_x0_mse,
            one_step_cartesian_rmsd=rmsd_mean,
            input_perturbation_fraction=perturbed_fraction,
            input_perturbation_rmsd=perturbation_rmsd,
            rollout_clipped_fraction=rollout_clipped,
            sigma_mean=sigma.mean(),
            sigma_max_mean=maximum.mean(),
            sigma_by_structure=sigma.detach(),
            score_dual_cosine_by_structure=score_cosine,
            score_dual_norm_ratio_by_structure=score_norm_ratio,
            identity_cartesian_rmsd=identity_mean,
            denoising_rmsd_improvement=denoising_improvement,
            denoising_relative_improvement=denoising_relative_improvement,
            physics_loss=physics_loss,
            physics_overlap_excess=physics_overlap,
            physics_bond_radius_excess=physics_bond,
            physics_bond_valence_excess=physics_bvs,
            physics_bond_valence_applicable_fraction=physics_bvs_fraction,
            physics_coordination_excess=physics_cn,
            physics_coordination_coverage=physics_cn_coverage,
            predicted_u0=predicted_u0,
        )


__all__ = ["MP20ScoreObjective", "MP20StepOutput"]
