"""Variance-exploding schedule and terminal mixing contract."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from polarevolve.data.batch import PackedASUBatch
from polarevolve.diffusion.metric import active_orbit_indices, integer_dual_quadratic
from polarevolve.diffusion.periodic import integer_shell, integer_vectors


DEFAULT_SIGMA_MAX_CAP = 180.0
_TERMINAL_SIGMA_NUMERICAL_MARGIN = 1.0001
TERMINAL_ADAPTIVE_SIGMA_V1 = "terminal_adaptive_v1"
FIXED_SIGMA_WINDOW_V1 = "fixed_window_v1"
TRAINING_SIGMA_MODES = (
    TERMINAL_ADAPTIVE_SIGMA_V1,
    FIXED_SIGMA_WINDOW_V1,
)


@dataclass(frozen=True)
class SigmaSchedule:
    sigma_min: float = 0.01
    sigma_max: float = 1.0
    sigma_max_cap: float = DEFAULT_SIGMA_MAX_CAP
    levels: int = 128

    def __post_init__(self) -> None:
        if not 0.0 < float(self.sigma_min) < float(self.sigma_max):
            raise ValueError("sigma schedule requires 0 < sigma_min < sigma_max")
        if float(self.sigma_max_cap) < float(self.sigma_max):
            raise ValueError("sigma_max_cap must not be below sigma_max")
        if not isinstance(self.levels, int) or self.levels < 2:
            raise ValueError("sigma schedule requires at least two levels")

    def descending(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        sigma_max_by_structure: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if sigma_max_by_structure is None:
            return torch.logspace(
                math.log10(float(self.sigma_max)),
                math.log10(float(self.sigma_min)),
                int(self.levels),
                device=device,
                dtype=dtype,
            )
        maximum = torch.as_tensor(
            sigma_max_by_structure, device=device, dtype=dtype
        ).reshape(-1)
        if not bool(torch.isfinite(maximum).all().item()) or bool(
            (maximum < self.sigma_max).any().item()
        ):
            raise ValueError("adaptive sigma maxima must be finite and >= sigma_max")
        progress = torch.linspace(1.0, 0.0, self.levels, device=device, dtype=dtype)
        return torch.exp(
            math.log(float(self.sigma_min))
            + progress[:, None]
            * (torch.log(maximum)[None] - math.log(float(self.sigma_min)))
        )

    def adjacent_lower(
        self,
        sigma_by_structure: torch.Tensor,
        sigma_max_by_structure: torch.Tensor,
    ) -> torch.Tensor:
        """Return the next level from the production structure-wise log grid."""

        sigma = torch.as_tensor(sigma_by_structure).reshape(-1)
        maximum = torch.as_tensor(
            sigma_max_by_structure, device=sigma.device, dtype=sigma.dtype
        ).reshape(-1)
        if sigma.shape != maximum.shape or sigma.numel() == 0:
            raise ValueError("sigma and adaptive maxima must have matching non-empty shapes")
        if not bool(torch.isfinite(sigma).all().item()) or not bool(
            torch.isfinite(maximum).all().item()
        ):
            raise ValueError("sigma grid values must be finite")
        if bool((sigma <= 0.0).any().item()) or bool(
            (maximum < self.sigma_max).any().item()
        ):
            raise ValueError("invalid sigma or adaptive maximum")
        if bool((sigma > maximum * 1.00001).any().item()):
            raise ValueError("current sigma exceeds its structure-wise maximum")
        log_step = (
            math.log(float(self.sigma_min)) - torch.log(maximum)
        ) / (self.levels - 1)
        next_sigma = torch.exp(torch.log(sigma) + log_step)
        return torch.maximum(
            next_sigma,
            sigma.new_full(sigma.shape, float(self.sigma_min)),
        )


@dataclass(frozen=True)
class TerminalMixingRateAudit:
    """Finite Fourier search result with an explicit certification boundary."""

    rates: torch.Tensor
    orbit_rates: torch.Tensor
    sufficient_radii: torch.Tensor
    searched_radii: torch.Tensor
    radius_clipped: torch.Tensor

    @property
    def certified(self) -> bool:
        return not bool(self.radius_clipped.any().item())


def certified_fourier_search_radius(
    metric: torch.Tensor,
    *,
    minimum_radius: int = 4,
) -> torch.Tensor:
    """Return a conservative sufficient box radius for the shortest dual vector."""

    if minimum_radius < 1:
        raise ValueError("minimum Fourier radius must be positive")
    value = torch.as_tensor(metric)
    if value.ndim != 3 or value.shape[1] != value.shape[2]:
        raise ValueError("Fourier metrics must have shape [items,d,d]")
    if not bool(torch.isfinite(value).all().item()) or bool(
        (torch.linalg.eigvalsh(value) <= 0.0).any().item()
    ):
        raise ValueError("Fourier metrics must be finite and positive definite")
    # For Q=G^-1, a coordinate unit vector gives q_best <= lambda_max(Q).
    # Any integer vector outside this box has norm >= R+1, so
    # R >= sqrt(cond(G)) is a sufficient certificate for the global minimum.
    required = torch.ceil(torch.sqrt(torch.linalg.cond(value))).to(torch.long)
    return required.clamp(min=minimum_radius)


def terminal_mixing_rate_audit(
    batch: PackedASUBatch,
    *,
    orbit_metric: torch.Tensor | None = None,
    fourier_radius: int = 4,
    maximum_radius: int = 12,
) -> TerminalMixingRateAudit:
    """Return finite-search rates and whether every orbit search is certified."""

    if fourier_radius < 1 or maximum_radius < fourier_radius:
        raise ValueError("invalid terminal-mixing Fourier radii")
    metric_source = batch.orbit_metric if orbit_metric is None else orbit_metric
    metric_source = torch.as_tensor(
        metric_source, device=batch.lattice.device, dtype=batch.lattice.dtype
    )
    if metric_source.shape != batch.orbit_metric.shape or not bool(
        torch.isfinite(metric_source).all().item()
    ):
        raise ValueError("terminal-mixing metric must be finite with shape [orbits,3,3]")
    orbit_rates = batch.lattice.new_full((batch.num_orbits,), torch.inf)
    sufficient_radii = torch.zeros(
        batch.num_orbits, device=batch.lattice.device, dtype=torch.long
    )
    searched_radii = torch.zeros_like(sufficient_radii)
    radius_clipped = torch.zeros(
        batch.num_orbits, device=batch.lattice.device, dtype=torch.bool
    )
    compute_dtype = torch.float64 if batch.lattice.dtype == torch.float64 else torch.float32
    for dimension in (1, 2, 3):
        orbits, _ = active_orbit_indices(batch, dimension)
        if orbits.numel() == 0:
            continue
        metric = metric_source[orbits, :dimension, :dimension].to(compute_dtype)
        local_sufficient = certified_fourier_search_radius(
            metric, minimum_radius=fourier_radius
        )
        sufficient_radii[orbits] = local_sufficient
        inverse = torch.linalg.inv(metric)
        lower_eigenvalue = torch.linalg.eigvalsh(inverse)[:, 0]
        best = inverse.new_full((len(orbits),), torch.inf)
        unresolved = torch.ones(len(orbits), device=inverse.device, dtype=torch.bool)
        margin = 64.0 * torch.finfo(compute_dtype).eps
        for radius in range(fourier_radius, maximum_radius + 1):
            local = torch.nonzero(unresolved, as_tuple=False).reshape(-1)
            if local.numel() == 0:
                break
            integer_values = (
                integer_vectors(dimension, radius)
                if radius == fourier_radius
                else integer_shell(dimension, radius)
            )
            vectors = torch.tensor(
                integer_values,
                device=batch.lattice.device,
                dtype=compute_dtype,
            )
            shell_best = integer_dual_quadratic(
                inverse[local], vectors
            ).min(dim=1).values
            best[local] = torch.minimum(best[local], shell_best)
            outside_lower = lower_eigenvalue[local] * float((radius + 1) ** 2)
            completed = local[outside_lower > best[local] * (1.0 + margin)]
            if completed.numel():
                orbit_rates[orbits[completed]] = best[completed].to(orbit_rates.dtype)
                searched_radii[orbits[completed]] = radius
                unresolved[completed] = False
        remaining = torch.nonzero(unresolved, as_tuple=False).reshape(-1)
        if remaining.numel():
            orbit_rates[orbits[remaining]] = best[remaining].to(
                orbit_rates.dtype
            )
            searched_radii[orbits[remaining]] = maximum_radius
            radius_clipped[orbits[remaining]] = True
    structures = batch.lattice.new_full((batch.batch_size,), torch.inf)
    for structure in range(batch.batch_size):
        selected = batch.orbit_to_structure == structure
        if bool(selected.any().item()):
            structures[structure] = orbit_rates[selected].min()
    return TerminalMixingRateAudit(
        rates=structures,
        orbit_rates=orbit_rates,
        sufficient_radii=sufficient_radii,
        searched_radii=searched_radii,
        radius_clipped=radius_clipped,
    )


def terminal_mixing_rates(
    batch: PackedASUBatch,
    *,
    orbit_metric: torch.Tensor | None = None,
    fourier_radius: int = 4,
    maximum_radius: int = 12,
    require_certified: bool = True,
) -> torch.Tensor:
    """Return the certified slowest non-constant Fourier rate per structure."""

    audit = terminal_mixing_rate_audit(
        batch,
        orbit_metric=orbit_metric,
        fourier_radius=fourier_radius,
        maximum_radius=maximum_radius,
    )
    if require_certified and not audit.certified:
        sufficient = int(audit.sufficient_radii.max().item())
        clipped = int(audit.radius_clipped.count_nonzero().item())
        raise ValueError(
            "terminal-mixing Fourier search is uncertified: "
            f"{clipped} orbit(s) remain unresolved at maximum_radius={maximum_radius}; "
            f"a conservative sufficient radius is at most {sufficient}"
        )
    return audit.rates


def terminal_mixing_coefficients(
    batch: PackedASUBatch,
    sigma_by_structure: torch.Tensor,
    *,
    orbit_metric: torch.Tensor | None = None,
    fourier_radius: int = 4,
    maximum_radius: int = 12,
    require_certified: bool = True,
) -> torch.Tensor:
    """Leading non-constant Fourier coefficient for each structure."""

    rates = terminal_mixing_rates(
        batch,
        orbit_metric=orbit_metric,
        fourier_radius=fourier_radius,
        maximum_radius=maximum_radius,
        require_certified=require_certified,
    )
    return terminal_mixing_coefficients_from_rates(rates, sigma_by_structure)


def terminal_mixing_coefficients_from_rates(
    rates: torch.Tensor,
    sigma_by_structure: torch.Tensor,
) -> torch.Tensor:
    """Evaluate terminal Fourier coefficients from verified positive rates."""

    rates = torch.as_tensor(rates)
    sigma = torch.as_tensor(
        sigma_by_structure, device=rates.device, dtype=rates.dtype
    ).reshape(-1)
    if sigma.numel() == 1:
        sigma = sigma.expand(rates.numel())
    if sigma.shape != rates.shape or bool((sigma <= 0).any().item()):
        raise ValueError("sigma must contain one positive value per structure")
    if bool(torch.isnan(rates).any().item()) or bool((rates <= 0.0).any().item()):
        raise ValueError("terminal-mixing rates must be positive or infinite")
    return torch.exp(-2.0 * math.pi**2 * sigma.square() * rates)


def minimum_terminal_sigma_from_rates(
    rates: torch.Tensor,
    *,
    tolerance: float = 1.0e-3,
) -> torch.Tensor:
    """Evaluate the minimum terminal sigma from verified Fourier rates."""

    if not 0.0 < tolerance < 1.0:
        raise ValueError("terminal mixing tolerance must lie in (0,1)")
    rates = torch.as_tensor(rates)
    if bool(torch.isnan(rates).any().item()) or bool((rates <= 0.0).any().item()):
        raise ValueError("terminal-mixing rates must be positive or infinite")
    required = torch.sqrt(
        -math.log(float(tolerance)) / (2.0 * math.pi**2 * rates)
    )
    return required * _TERMINAL_SIGMA_NUMERICAL_MARGIN


def minimum_terminal_sigma(
    batch: PackedASUBatch,
    *,
    tolerance: float = 1.0e-3,
    orbit_metric: torch.Tensor | None = None,
    fourier_radius: int = 4,
    maximum_radius: int = 12,
    require_certified: bool = True,
) -> torch.Tensor:
    """Minimum structure-wise Cartesian sigma satisfying the mixing gate."""

    if not 0.0 < tolerance < 1.0:
        raise ValueError("terminal mixing tolerance must lie in (0,1)")
    if fourier_radius < 1 or maximum_radius < fourier_radius:
        raise ValueError("invalid terminal-mixing Fourier radii")
    rates = terminal_mixing_rates(
        batch,
        orbit_metric=orbit_metric,
        fourier_radius=fourier_radius,
        maximum_radius=maximum_radius,
        require_certified=require_certified,
    )
    return minimum_terminal_sigma_from_rates(rates, tolerance=tolerance)


__all__ = [
    "DEFAULT_SIGMA_MAX_CAP",
    "FIXED_SIGMA_WINDOW_V1",
    "SigmaSchedule",
    "TERMINAL_ADAPTIVE_SIGMA_V1",
    "TerminalMixingRateAudit",
    "TRAINING_SIGMA_MODES",
    "certified_fourier_search_radius",
    "minimum_terminal_sigma",
    "minimum_terminal_sigma_from_rates",
    "terminal_mixing_coefficients",
    "terminal_mixing_coefficients_from_rates",
    "terminal_mixing_rate_audit",
    "terminal_mixing_rates",
]
