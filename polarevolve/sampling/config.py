"""Typed configuration for the MP20 coordinate-pilot sampling run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from polarevolve.crystal.lattice import LatticeBounds
from polarevolve.data import contracts as data_contracts
from polarevolve.data.contracts import INDEPENDENT_LATTICE_MODE
from polarevolve.diffusion.sampler import SamplerConfig

@dataclass(frozen=True)
class SamplingRunConfig:
    checkpoint: Path
    run_id: str
    lattice_checkpoint: Path | None = None
    lattice_replay_samples: Path | None = None
    queries: Path | None = None
    query_cif_policy: str = "geometry_eligible"
    program_selection: Path | None = None
    lattices_per_program: int = 2
    split: str = "test"
    records: int = 32
    candidates: int = 4
    seed: int = 20260807
    evaluation_panel: Path | None = None
    cache_relative: str = data_contracts.DEFAULT_CACHE_RELATIVE
    group_assets_relative: str = data_contracts.DEFAULT_GROUP_ASSETS_RELATIVE
    wyckoff_assets_relative: str = data_contracts.DEFAULT_WYCKOFF_ASSETS_RELATIVE
    physics_guidance_enabled: bool = False
    physics_prior: Path | None = None
    overlap_repair_steps: int = 0
    lattice_minimum_volume_per_atom: float = LatticeBounds.minimum_volume_per_atom
    lattice_maximum_volume_per_atom: float = LatticeBounds.maximum_volume_per_atom
    lattice_maximum_aspect_ratio: float = LatticeBounds.maximum_aspect_ratio
    sampler: SamplerConfig = SamplerConfig()

    def __post_init__(self) -> None:
        if not isinstance(self.overlap_repair_steps, int) or not 0 <= self.overlap_repair_steps <= 8:
            raise ValueError("overlap repair steps must be an integer in [0,8]; zero disables repair")
        if self.query_cif_policy not in {"all", "geometry_eligible"}:
            raise ValueError("query CIF policy must be all or geometry_eligible")
        if self.queries is not None and (
            self.lattice_checkpoint is None or self.evaluation_panel is not None
            or self.lattice_replay_samples is not None or self.lattices_per_program <= 0
        ):
            raise ValueError("query search requires an independent lattice checkpoint and no cache panel/replay")
        if self.program_selection is not None and self.queries is None:
            raise ValueError("program selection is only valid for query search")
        if not self.run_id.strip():
            raise ValueError("run_id must be non-empty")
        if self.split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        if self.records <= 0 or self.candidates <= 0:
            raise ValueError("records and candidates must be positive")
        if self.physics_guidance_enabled != (self.sampler.guidance_step_scale > 0.0):
            raise ValueError(
                "physics guidance and a positive sampler guidance scale must be enabled together"
            )
        if self.lattice_replay_samples is not None and self.lattice_checkpoint is None:
            raise ValueError("lattice replay requires the matching lattice checkpoint")
        if not (
            0.0
            < self.lattice_minimum_volume_per_atom
            < self.lattice_maximum_volume_per_atom
        ) or self.lattice_maximum_aspect_ratio < 1.0:
            raise ValueError("invalid independent-lattice bounds")


__all__ = [
    "INDEPENDENT_LATTICE_MODE",
    "SamplingRunConfig",
]
