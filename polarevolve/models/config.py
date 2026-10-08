"""Score-model configuration with the only model defaults."""

from __future__ import annotations

from dataclasses import asdict, dataclass


PHYSICAL_SCORE_V1 = "physical_score_v1"
SIGMA_SCALED_SCORE_V1 = "sigma_scaled_score_v1"
SCORE_PARAMETERIZATIONS = (PHYSICAL_SCORE_V1, SIGMA_SCALED_SCORE_V1)
PERIODIC_PRECISION_V1 = "geometry_fp32_or_fp64_mlp_amp_fp64_reduction_v1"
@dataclass(frozen=True)
class ScoreModelConfig:
    hidden_dim: int = 256
    time_dim: int = 64
    radial_basis: int = 48
    layers: int = 4
    cutoff: float = 8.0
    score_parameterization: str = SIGMA_SCALED_SCORE_V1
    condition_dim: int = 0
    atom_condition_dim: int = 0
    equivariant_reference: bool = False
    backbone_version: int = 1

    def __post_init__(self) -> None:
        if self.backbone_version not in (1, 2):
            raise ValueError("unsupported score backbone version")
        if self.hidden_dim < 32 or self.time_dim < 8 or self.time_dim % 2:
            raise ValueError("invalid score model dimensions")
        if self.radial_basis < 4 or self.layers < 1 or self.cutoff <= 0.0:
            raise ValueError("invalid score model radial/depth configuration")
        if self.score_parameterization not in SCORE_PARAMETERIZATIONS:
            raise ValueError(f"score_parameterization must be one of {SCORE_PARAMETERIZATIONS}")
        if self.condition_dim < 0 or self.atom_condition_dim < 0:
            raise ValueError("condition dimensions must be non-negative")
        if self.equivariant_reference and self.atom_condition_dim != 2:
            raise ValueError("equivariant reference requires distance and mask (dimension 2)")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class LatticeModelConfig:
    """Independent C2L lattice distribution model defaults."""

    hidden_dim: int = 128
    layers: int = 3
    mixture_components: int = 4
    minimum_log_scale: float = -5.0
    maximum_log_scale: float = 2.0

    def __post_init__(self) -> None:
        if self.hidden_dim < 32 or self.layers < 1 or self.mixture_components < 1:
            raise ValueError("invalid lattice model width, depth, or mixture count")
        if self.minimum_log_scale >= self.maximum_log_scale:
            raise ValueError("lattice log-scale bounds must be ordered")

    def to_dict(self) -> dict:
        return asdict(self)
