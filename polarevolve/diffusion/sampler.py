"""The single production sampler for the hard-conditioned ASU process."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol

import torch

from polarevolve.data.batch import PackedASUBatch
from polarevolve.diffusion.metric import (
    DIFFUSION_METRIC_CONTRACTS,
    MEMBER_SUM_CARTESIAN_V1,
    STATE_QUOTIENT_CONTRACTS,
    TRANSLATION_QUOTIENT_V1,
    apply_state_quotient_tangent,
    limit_cartesian_rms,
    metric_inverse_multiply,
    orbit_metric_view,
)
from polarevolve.diffusion.noise import wrap_unit_interval
from polarevolve.diffusion.objective import estimate_clean_parameters
from polarevolve.diffusion.schedule import (
    SigmaSchedule,
    minimum_terminal_sigma,
    terminal_mixing_coefficients,
)
from polarevolve.diffusion.update import (
    LEGACY_MEAN,
    SAMPLER_INTEGRATORS,
    reverse_process_update,
)

GUIDANCE_TELEMETRY_CONTRACT = "gt_sge_guidance_sigma_telemetry_v1"
GUIDANCE_TELEMETRY_BINS = 10


class NegativeScoreModel(Protocol):
    def __call__(
        self,
        batch: PackedASUBatch,
        u: torch.Tensor,
        sigma_by_structure: torch.Tensor,
    ) -> torch.Tensor: ...


class GuidanceCovector(Protocol):
    """Return dE/du in the native ASU dual space for the current state."""

    def __call__(
        self,
        batch: PackedASUBatch,
        u: torch.Tensor,
        sigma_by_structure: torch.Tensor,
    ) -> torch.Tensor: ...


@dataclass(frozen=True)
class SamplerConfig:
    schedule: SigmaSchedule = SigmaSchedule()
    integrator: str = LEGACY_MEAN
    maximum_step_rms_angstrom: float = 0.12
    terminal_mixing_tolerance: float = 1.0e-3
    state_quotient: str = TRANSLATION_QUOTIENT_V1
    diffusion_metric: str = MEMBER_SUM_CARTESIAN_V1
    guidance_step_scale: float = 0.0
    maximum_guidance_step_rms_angstrom: float = 0.02

    def __post_init__(self) -> None:
        if self.integrator not in SAMPLER_INTEGRATORS:
            raise ValueError(f"unsupported sampler integrator: {self.integrator!r}")
        if self.maximum_step_rms_angstrom <= 0.0:
            raise ValueError("maximum step RMS must be positive")
        if not 0.0 < self.terminal_mixing_tolerance < 1.0:
            raise ValueError("terminal mixing tolerance must lie in (0,1)")
        if self.state_quotient not in STATE_QUOTIENT_CONTRACTS:
            raise ValueError("unsupported sampler state quotient")
        if self.diffusion_metric not in DIFFUSION_METRIC_CONTRACTS:
            raise ValueError("unsupported sampler diffusion metric")
        if self.guidance_step_scale < 0.0:
            raise ValueError("guidance step scale must be non-negative")
        if self.maximum_guidance_step_rms_angstrom <= 0.0:
            raise ValueError("maximum guidance step RMS must be positive")


@dataclass(frozen=True)
class ASUSamplingResult:
    normalized_u: torch.Tensor
    full_fractional: torch.Tensor
    finite: bool
    clipped_reverse_step_fraction: float = 0.0
    maximum_proposed_step_rms_angstrom: float = 0.0
    final_denoise_clipped_fraction: float = 0.0
    state_quotient: str = TRANSLATION_QUOTIENT_V1
    guidance_applied_structure_step_fraction: float = 0.0
    guidance_clipped_structure_step_fraction: float = 0.0
    maximum_proposed_guidance_rms_angstrom: float = 0.0
    mean_proposed_guidance_rms_angstrom: float = 0.0
    mean_applied_guidance_rms_angstrom: float = 0.0
    guidance_sigma_bins: tuple["GuidanceSigmaBin", ...] = ()
    guidance_clipped_fraction_by_structure: tuple[float, ...] = ()


@dataclass(frozen=True)
class GuidanceSigmaBin:
    bin_index: int
    sigma_lower: float
    sigma_upper: float
    structure_steps: int
    applied_structure_steps: int
    clipped_structure_steps: int
    mean_proposed_rms_angstrom: float
    mean_applied_rms_angstrom: float
    maximum_proposed_rms_angstrom: float


@dataclass(frozen=True)
class _GuidanceTelemetrySummary:
    applied_fraction: float
    clipped_fraction: float
    maximum_proposed_rms: float
    mean_proposed_rms: float
    mean_applied_rms: float
    sigma_bins: tuple[GuidanceSigmaBin, ...]
    clipped_fraction_by_structure: tuple[float, ...] = ()


class _GuidanceTelemetry:
    """Accumulate fixed-bin guidance statistics without per-step host sync."""

    def __init__(self, config: SamplerConfig, reference: torch.Tensor, batch_size: int) -> None:
        self.maximum_applied_rms = float(config.maximum_guidance_step_rms_angstrom)
        self.boundaries = torch.logspace(
            math.log10(config.schedule.sigma_min),
            math.log10(config.schedule.sigma_max_cap),
            GUIDANCE_TELEMETRY_BINS + 1,
            device=reference.device,
            dtype=torch.float32,
        )
        self.structure_steps = torch.zeros(
            GUIDANCE_TELEMETRY_BINS, device=reference.device, dtype=torch.long
        )
        self.applied_steps = torch.zeros_like(self.structure_steps)
        self.clipped_steps = torch.zeros_like(self.structure_steps)
        self.proposed_sum = torch.zeros(
            GUIDANCE_TELEMETRY_BINS, device=reference.device, dtype=torch.float64
        )
        self.applied_sum = torch.zeros_like(self.proposed_sum)
        self.proposed_max = torch.zeros_like(self.proposed_sum)
        self.clipped_by_structure = torch.zeros(batch_size, device=reference.device, dtype=torch.long)
        self.calls = 0

    def record(
        self,
        sigma_by_structure: torch.Tensor,
        proposed_rms: torch.Tensor,
        clipped: torch.Tensor,
    ) -> None:
        sigma = sigma_by_structure.detach().float().reshape(-1)
        proposed = proposed_rms.detach().double().reshape(-1)
        clipped = clipped.detach().reshape(-1)
        if sigma.shape != proposed.shape or sigma.shape != clipped.shape:
            raise ValueError("guidance telemetry requires one value per structure")
        indices = torch.bucketize(sigma, self.boundaries[1:-1], right=False)
        ones = torch.ones_like(indices, dtype=torch.long)
        applied = proposed > 0.0
        applied_rms = proposed.clamp_max(self.maximum_applied_rms)
        self.structure_steps.index_add_(0, indices, ones)
        self.applied_steps.index_add_(0, indices, applied.to(torch.long))
        self.clipped_steps.index_add_(0, indices, clipped.to(torch.long))
        self.proposed_sum.index_add_(0, indices, proposed)
        self.applied_sum.index_add_(0, indices, applied_rms)
        self.proposed_max.scatter_reduce_(
            0, indices, proposed, reduce="amax", include_self=True
        )
        self.clipped_by_structure += clipped.to(torch.long)
        self.calls += 1

    def summary(self) -> _GuidanceTelemetrySummary:
        boundaries = self.boundaries.detach().cpu().tolist()
        counts = self.structure_steps.detach().cpu().tolist()
        applied = self.applied_steps.detach().cpu().tolist()
        clipped = self.clipped_steps.detach().cpu().tolist()
        proposed_sum = self.proposed_sum.detach().cpu().tolist()
        applied_sum = self.applied_sum.detach().cpu().tolist()
        proposed_max = self.proposed_max.detach().cpu().tolist()
        bins = tuple(
            GuidanceSigmaBin(
                bin_index=index,
                sigma_lower=float(boundaries[index]),
                sigma_upper=float(boundaries[index + 1]),
                structure_steps=int(counts[index]),
                applied_structure_steps=int(applied[index]),
                clipped_structure_steps=int(clipped[index]),
                mean_proposed_rms_angstrom=(
                    float(proposed_sum[index] / counts[index]) if counts[index] else 0.0
                ),
                mean_applied_rms_angstrom=(
                    float(applied_sum[index] / counts[index]) if counts[index] else 0.0
                ),
                maximum_proposed_rms_angstrom=float(proposed_max[index]),
            )
            for index in range(GUIDANCE_TELEMETRY_BINS)
        )
        total = sum(counts)
        return _GuidanceTelemetrySummary(
            applied_fraction=sum(applied) / total if total else 0.0,
            clipped_fraction=sum(clipped) / total if total else 0.0,
            maximum_proposed_rms=max(proposed_max, default=0.0),
            mean_proposed_rms=sum(proposed_sum) / total if total else 0.0,
            mean_applied_rms=sum(applied_sum) / total if total else 0.0,
            sigma_bins=bins,
            clipped_fraction_by_structure=tuple(
                (self.clipped_by_structure.double() / max(self.calls, 1)).cpu().tolist()
            ),
        )


@dataclass(frozen=True)
class SamplerStepObservation:
    """Read-only tensors for an optional online diagnostic consumer."""

    phase: str
    u: torch.Tensor
    sigma_by_structure: torch.Tensor
    next_sigma_by_structure: torch.Tensor
    negative_score: torch.Tensor
    proposed_update: torch.Tensor
    applied_update: torch.Tensor
    clipped_by_structure: torch.Tensor


class SamplerStepObserver(Protocol):
    def __call__(self, observation: SamplerStepObservation) -> None: ...


def _score(
    model: NegativeScoreModel,
    batch: PackedASUBatch,
    u: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    value = model(batch, u, sigma)
    if value.shape != u.shape or not bool(torch.isfinite(value).all().item()):
        raise FloatingPointError("score model returned an invalid negative-score covector")
    return value


def _guidance_update(
    guidance: GuidanceCovector,
    batch: PackedASUBatch,
    u: torch.Tensor,
    sigma: torch.Tensor,
    config: SamplerConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    covector = guidance(batch, u, sigma)
    if covector.shape != u.shape or not bool(torch.isfinite(covector).all().item()):
        raise FloatingPointError("guidance returned an invalid ASU covector")
    proposed = apply_state_quotient_tangent(
        batch,
        -config.guidance_step_scale
        * metric_inverse_multiply(
            batch, covector, metric_convention=config.diffusion_metric
        ),
        config.state_quotient,
        metric_convention=config.diffusion_metric,
    )
    return limit_cartesian_rms(
        batch, proposed, config.maximum_guidance_step_rms_angstrom
    )


@torch.no_grad()
def sample(
    *,
    model: NegativeScoreModel,
    batch: PackedASUBatch,
    config: SamplerConfig,
    generator: torch.Generator | None = None,
    initial_u: torch.Tensor | None = None,
    initial_sigma_by_structure: torch.Tensor | None = None,
    step_observer: SamplerStepObserver | None = None,
    guidance: GuidanceCovector | None = None,
) -> ASUSamplingResult:
    """Run one explicitly selected reverse process on fixed cache lattices."""

    if (guidance is None) != (config.guidance_step_scale == 0.0):
        raise ValueError(
            "guidance callback and a positive guidance step scale must be enabled together"
        )
    if initial_sigma_by_structure is None:
        diffusion_orbit_metric = orbit_metric_view(batch, config.diffusion_metric)
        required_terminal = minimum_terminal_sigma(
            batch,
            tolerance=config.terminal_mixing_tolerance,
            orbit_metric=diffusion_orbit_metric,
        )
        starting_sigma = torch.maximum(
            required_terminal,
            batch.lattice.new_full((batch.batch_size,), config.schedule.sigma_max),
        )
        cap_violations = torch.nonzero(
            starting_sigma > config.schedule.sigma_max_cap, as_tuple=False
        ).reshape(-1)
        if cap_violations.numel() > 0:
            index = int(cap_violations[0])
            raise ValueError(
                "required terminal sigma exceeds "
                f"sigma_max_cap={config.schedule.sigma_max_cap:.6g}; "
                f"{batch.material_ids[index]}:required={float(required_terminal[index]):.6g}"
            )
        levels = config.schedule.descending(
            device=batch.lattice.device,
            dtype=batch.lattice.dtype,
            sigma_max_by_structure=starting_sigma,
        )
        terminal_sigma = levels[0]
        mixing = terminal_mixing_coefficients(
            batch, terminal_sigma, orbit_metric=diffusion_orbit_metric
        )
        mixing_violations = torch.nonzero(
            mixing > config.terminal_mixing_tolerance, as_tuple=False
        ).reshape(-1)
        if mixing_violations.numel() > 0:
            index = int(mixing_violations[0])
            raise ValueError(
                "terminal sigma does not mix every active ASU torus; "
                f"{batch.material_ids[index]}:sigma={float(terminal_sigma[index]):.6g},"
                f" coefficient={float(mixing[index]):.6g},"
                f" tolerance={config.terminal_mixing_tolerance:.6g}"
            )
    else:
        if initial_u is None:
            raise ValueError("an explicit initial sigma requires an explicit initial state")
        starting_sigma = torch.as_tensor(
            initial_sigma_by_structure,
            device=batch.lattice.device,
            dtype=batch.lattice.dtype,
        ).reshape(-1)
        if starting_sigma.numel() == 1:
            starting_sigma = starting_sigma.expand(batch.batch_size)
        if starting_sigma.shape != (batch.batch_size,) or not bool(
            torch.isfinite(starting_sigma).all().item()
        ):
            raise ValueError("initial sigma must be finite with one value per structure")
        if bool((starting_sigma > config.schedule.sigma_max_cap).any().item()):
            raise ValueError("initial sigma exceeds the configured sigma_max_cap")
        levels = config.schedule.descending(
            device=batch.lattice.device,
            dtype=batch.lattice.dtype,
            sigma_max_by_structure=starting_sigma,
        )
    if initial_u is None:
        u = torch.rand(
            batch.parameter_shape,
            device=batch.lattice.device,
            dtype=batch.lattice.dtype,
            generator=generator,
        )
    else:
        u = wrap_unit_interval(
            torch.as_tensor(
                initial_u, device=batch.lattice.device, dtype=batch.lattice.dtype
            )
        )
        if u.shape != batch.parameter_shape:
            raise ValueError("initial_u does not match packed ASU parameters")
    clipped_steps = torch.zeros((), device=batch.lattice.device, dtype=torch.long)
    maximum_proposed_rms = batch.lattice.new_zeros(())
    guidance_telemetry = (
        _GuidanceTelemetry(config, batch.lattice, batch.batch_size)
        if guidance is not None
        else None
    )
    for source, target in zip(levels[:-1], levels[1:]):
        score = _score(model, batch, u, source)
        proposed_update = reverse_process_update(
            batch,
            score,
            source,
            target,
            integrator=config.integrator,
            generator=generator,
            state_quotient=config.state_quotient,
            diffusion_metric=config.diffusion_metric,
        )
        if guidance is not None:
            guidance_update, guidance_rms, guidance_was_clipped = _guidance_update(
                guidance, batch, u, source, config
            )
            proposed_update = proposed_update + guidance_update
            guidance_telemetry.record(source, guidance_rms, guidance_was_clipped)
        update, proposed_rms, clipped = limit_cartesian_rms(
            batch, proposed_update, config.maximum_step_rms_angstrom
        )
        clipped_steps += clipped.sum()
        maximum_proposed_rms = torch.maximum(
            maximum_proposed_rms, proposed_rms.max()
        )
        if step_observer is not None:
            step_observer(
                SamplerStepObservation(
                    phase="reverse",
                    u=u,
                    sigma_by_structure=source,
                    next_sigma_by_structure=target,
                    negative_score=score,
                    proposed_update=proposed_update,
                    applied_update=update,
                    clipped_by_structure=clipped,
                )
            )
        u = wrap_unit_interval(u + update)
    final_sigma = levels[-1]
    final_score = _score(model, batch, u, final_sigma)
    denoised = estimate_clean_parameters(
        batch,
        u,
        final_score,
        final_sigma,
        state_quotient=config.state_quotient,
        diffusion_metric=config.diffusion_metric,
    )
    proposed_final_update = torch.remainder(denoised - u + 0.5, 1.0) - 0.5
    proposed_final_update = apply_state_quotient_tangent(
        batch,
        proposed_final_update,
        config.state_quotient,
        metric_convention=config.diffusion_metric,
    )
    if guidance is not None:
        guidance_update, guidance_rms, guidance_was_clipped = _guidance_update(
            guidance, batch, u, final_sigma, config
        )
        proposed_final_update = proposed_final_update + guidance_update
        guidance_telemetry.record(final_sigma, guidance_rms, guidance_was_clipped)
    final_update, _, final_clipped = limit_cartesian_rms(
        batch, proposed_final_update, config.maximum_step_rms_angstrom
    )
    if step_observer is not None:
        step_observer(
            SamplerStepObservation(
                phase="final_denoise",
                u=u,
                sigma_by_structure=final_sigma,
                next_sigma_by_structure=torch.zeros_like(final_sigma),
                negative_score=final_score,
                proposed_update=proposed_final_update,
                applied_update=final_update,
                clipped_by_structure=final_clipped,
            )
        )
    u = wrap_unit_interval(u + final_update)
    finite = bool(torch.isfinite(u).all().item())
    maximum_proposed_rms = float(maximum_proposed_rms.item())
    guidance_summary = (
        guidance_telemetry.summary()
        if guidance_telemetry is not None
        else _GuidanceTelemetrySummary(0.0, 0.0, 0.0, 0.0, 0.0, ())
    )
    return ASUSamplingResult(
        u,
        batch.expand(u),
        finite,
        clipped_reverse_step_fraction=(
            int(clipped_steps.item())
            / max(batch.batch_size * (len(levels) - 1), 1)
        ),
        maximum_proposed_step_rms_angstrom=maximum_proposed_rms,
        final_denoise_clipped_fraction=float(final_clipped.float().mean().item()),
        state_quotient=config.state_quotient,
        guidance_applied_structure_step_fraction=(
            guidance_summary.applied_fraction
        ),
        guidance_clipped_structure_step_fraction=(
            guidance_summary.clipped_fraction
        ),
        maximum_proposed_guidance_rms_angstrom=(
            guidance_summary.maximum_proposed_rms
        ),
        mean_proposed_guidance_rms_angstrom=guidance_summary.mean_proposed_rms,
        mean_applied_guidance_rms_angstrom=guidance_summary.mean_applied_rms,
        guidance_sigma_bins=guidance_summary.sigma_bins,
        guidance_clipped_fraction_by_structure=guidance_summary.clipped_fraction_by_structure,
    )


__all__ = [
    "ASUSamplingResult",
    "GUIDANCE_TELEMETRY_CONTRACT",
    "GuidanceCovector",
    "GuidanceSigmaBin",
    "NegativeScoreModel",
    "SamplerConfig",
    "SamplerStepObservation",
    "SamplerStepObserver",
    "sample",
]
