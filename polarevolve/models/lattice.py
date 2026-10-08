"""Permutation-invariant conditional distribution over C2L lattice coordinates."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from polarevolve.data.packing import (
    LATTICE_COORDINATE_DIMENSION,
    LatticeConditionBatch,
)
from polarevolve.models.config import LatticeModelConfig


@dataclass(frozen=True)
class LatticeMixtureOutput:
    logits: torch.Tensor
    means: torch.Tensor
    log_scales: torch.Tensor


def lattice_mixture_log_probability(
    output: LatticeMixtureOutput,
    target: torch.Tensor,
    active_mask: torch.Tensor,
) -> torch.Tensor:
    """Return per-structure log probability over Hall-active coordinates."""

    if target.shape != active_mask.shape or target.ndim != 2:
        raise ValueError("lattice target and mask must share shape [batch,coordinates]")
    residual = (target[:, None, :] - output.means) / output.log_scales.exp()
    log_density = -0.5 * (
        residual.square() + 2.0 * output.log_scales + math.log(2.0 * math.pi)
    )
    log_density = (log_density * active_mask[:, None, :]).sum(dim=-1)
    return torch.logsumexp(
        torch.log_softmax(output.logits, dim=-1) + log_density, dim=-1
    )


class HardConditionLatticeNetwork(nn.Module):
    """Predict a diagonal Gaussian mixture from hard crystal conditions only."""

    def __init__(self, config: LatticeModelConfig) -> None:
        super().__init__()
        self.config = config
        hidden = config.hidden_dim
        self.element_embedding = nn.Embedding(119, hidden)
        self.multiplicity_embedding = nn.Embedding(257, hidden)
        self.dimension_embedding = nn.Embedding(4, hidden)
        self.wyckoff_embedding = nn.Embedding(27, hidden)
        self.hall_embedding = nn.Embedding(531, hidden)
        self.space_group_embedding = nn.Embedding(231, hidden)
        self.orbit_encoder = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.scalar_projection = nn.Sequential(
            nn.Linear(3, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        trunk: list[nn.Module] = []
        for _ in range(config.layers):
            trunk.extend((nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.SiLU()))
        self.trunk = nn.Sequential(*trunk)
        components = config.mixture_components
        self.logit_head = nn.Linear(hidden, components)
        self.mean_head = nn.Linear(
            hidden, components * LATTICE_COORDINATE_DIMENSION
        )
        self.log_scale_head = nn.Linear(
            hidden, components * LATTICE_COORDINATE_DIMENSION
        )

    def forward(self, condition: LatticeConditionBatch) -> LatticeMixtureOutput:
        orbit = (
            self.element_embedding(condition.orbit_atomic_numbers)
            + self.multiplicity_embedding(condition.orbit_multiplicities)
            + self.dimension_embedding(condition.orbit_dimensions)
            + self.wyckoff_embedding(condition.orbit_letter_indices)
        )
        orbit = self.orbit_encoder(orbit)
        weights = condition.orbit_multiplicities.to(orbit.dtype)
        pooled = orbit.new_zeros((condition.batch_size, orbit.shape[-1]))
        pooled.index_add_(
            0,
            condition.orbit_to_structure,
            orbit * weights[:, None],
        )
        pooled = pooled / condition.atom_counts.to(orbit.dtype).clamp_min(1.0)[:, None]
        orbit_counts = torch.bincount(
            condition.orbit_to_structure,
            minlength=condition.batch_size,
        ).to(orbit.dtype)
        scalar = torch.stack(
            (
                torch.log(condition.atom_counts.to(orbit.dtype)),
                torch.log1p(orbit_counts),
                condition.shape_dimensions.to(orbit.dtype) / 5.0,
            ),
            dim=-1,
        )
        hidden = self.trunk(
            pooled
            + self.hall_embedding(condition.hall_numbers)
            + self.space_group_embedding(condition.space_group_numbers)
            + self.scalar_projection(scalar)
        )
        shape = (
            condition.batch_size,
            self.config.mixture_components,
            LATTICE_COORDINATE_DIMENSION,
        )
        return LatticeMixtureOutput(
            logits=self.logit_head(hidden),
            means=self.mean_head(hidden).reshape(shape),
            log_scales=self.log_scale_head(hidden).reshape(shape).clamp(
                self.config.minimum_log_scale,
                self.config.maximum_log_scale,
            ),
        )


__all__ = [
    "HardConditionLatticeNetwork",
    "LatticeMixtureOutput",
    "lattice_mixture_log_probability",
]
