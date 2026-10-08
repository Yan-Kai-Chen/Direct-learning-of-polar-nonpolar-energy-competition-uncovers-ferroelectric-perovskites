"""Training configuration with the only optimization defaults."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

from polarevolve.data import contracts as data_contracts
from polarevolve.diffusion.metric import (
    DIFFUSION_METRIC_CONTRACTS,
    MEMBER_SUM_CARTESIAN_V1,
)
from polarevolve.diffusion.schedule import (
    TERMINAL_ADAPTIVE_SIGMA_V1,
    TRAINING_SIGMA_MODES,
)


@dataclass(frozen=True)
class TrainingRunConfig:
    run_id: str
    cache_relative: str = data_contracts.DEFAULT_CACHE_RELATIVE
    group_assets_relative: str = data_contracts.DEFAULT_GROUP_ASSETS_RELATIVE
    wyckoff_assets_relative: str = data_contracts.DEFAULT_WYCKOFF_ASSETS_RELATIVE
    resume: Path | None = None
    initialize_from: Path | None = None
    terminal_mixing_tolerance: float = 1.0e-3
    diffusion_metric: str = MEMBER_SUM_CARTESIAN_V1
    allow_cache_transfer: bool = False

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id must be non-empty")
        if self.resume is not None and self.initialize_from is not None:
            raise ValueError("resume and initialize_from are mutually exclusive")
        if self.allow_cache_transfer and self.initialize_from is None:
            raise ValueError("cache transfer is only valid for model-only initialization")
        if self.terminal_mixing_tolerance <= 0.0:
            raise ValueError("terminal_mixing_tolerance must be positive")
        if self.diffusion_metric not in DIFFUSION_METRIC_CONTRACTS:
            raise ValueError("unsupported diffusion metric")


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 20260807
    max_steps: int = 2_000
    data_epochs: int | None = None
    evaluation_interval_epochs: int = 1
    calibration_interval_epochs: int = 0
    epoch_mode: str = "legacy_stream"
    warmup_epochs: int = 1
    resident_epoch: bool = False
    replay_cache_relative: str | None = None
    replay_manifest_sha256: str | None = None
    checkpoint_minutes: float = 30.0
    snapshot_interval_epochs: int = 25
    ema_half_life_epochs: float = 0.0
    batch_size: int = 4
    val_batch_size: int = 4
    num_workers: int = 8
    gradient_accumulation: int = 1
    learning_rate: float = 2.0e-4
    weight_decay: float = 1.0e-5
    warmup_steps: int = 100
    minimum_lr_ratio: float = 0.05
    gradient_clip_norm: float = 1.0
    precision: str = "bf16"
    log_every: int = 10
    mem_debug: bool = False
    validate_every: int = 100
    validation_batches: int = 8
    validation_mode: str = "legacy_batches"
    save_every: int = 100
    train_limit: int | None = 32
    val_limit: int | None = 32
    active_only: bool = False
    sigma_sampling_mode: str = TERMINAL_ADAPTIVE_SIGMA_V1
    sigma_strata: tuple[float, ...] = ()
    fixed_training_noise_seed: int | None = None
    input_perturbation_probability: float = 0.0
    input_perturbation_scale: float = 0.0
    rollout_interval: int = 0
    rollout_steps: int = 1
    rollout_score_weight: float = 0.0
    rollout_x0_weight: float = 0.0
    rollout_maximum_step_rms_angstrom: float = 0.12
    physics_auxiliary_weight: float = 0.0
    physics_warmup_epochs: int = 10
    physics_prior: str | None = None
    physics_calibration: str | None = None
    physics_sigma_full_strength: float = 0.10
    physics_sigma_cutoff: float = 0.50

    def __post_init__(self) -> None:
        if self.calibration_interval_epochs < 0:
            raise ValueError("calibration interval must be nonnegative")
        if self.calibration_interval_epochs and (
            self.validation_mode != "full_unique"
            or self.evaluation_interval_epochs <= 0
            or self.calibration_interval_epochs % self.evaluation_interval_epochs
        ):
            raise ValueError("calibration interval requires full_unique and a whole validation interval")
        integer_positive = (
            self.max_steps,
            self.evaluation_interval_epochs,
            self.batch_size,
            self.val_batch_size,
            self.gradient_accumulation,
            self.log_every,
            self.validate_every,
            self.validation_batches,
            self.save_every,
        )
        if any(value <= 0 for value in integer_positive) or self.num_workers < 0:
            raise ValueError("training counts must be positive and workers non-negative")
        if self.data_epochs is not None and self.data_epochs <= 0:
            raise ValueError("data_epochs must be positive or omitted")
        if self.epoch_mode not in {"legacy_stream", "finite_unique"}:
            raise ValueError("unsupported epoch mode")
        if self.epoch_mode == "finite_unique" and (not self.active_only or self.data_epochs is None):
            raise ValueError("finite_unique requires active_only and an explicit epoch count")
        if self.warmup_epochs <= 0 or self.checkpoint_minutes <= 0 or self.ema_half_life_epochs < 0 or self.snapshot_interval_epochs <= 0:
            raise ValueError("invalid epoch warmup, checkpoint interval or EMA half-life")
        if self.resident_epoch and self.epoch_mode != "finite_unique":
            raise ValueError("resident_epoch requires finite_unique")
        if (self.replay_cache_relative is None) != (self.replay_manifest_sha256 is None):
            raise ValueError("replay requires both a cache path and its manifest SHA")
        if self.replay_cache_relative is not None:
            digest = self.replay_manifest_sha256
            if self.epoch_mode != "finite_unique" or self.validation_mode != "full_unique":
                raise ValueError("balanced replay requires finite_unique/full_unique")
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("invalid replay manifest SHA")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("invalid optimizer configuration")
        if not 0.0 < self.minimum_lr_ratio <= 1.0:
            raise ValueError("minimum_lr_ratio must lie in (0,1]")
        if self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm must be positive")
        if self.precision not in {"bf16", "fp32"}:
            raise ValueError("precision must be bf16 or fp32")
        if self.validation_mode not in {"legacy_batches", "full_unique"}:
            raise ValueError("unsupported validation mode")
        for name, value in (("train_limit", self.train_limit), ("val_limit", self.val_limit)):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive or omitted")
        if self.sigma_sampling_mode not in TRAINING_SIGMA_MODES:
            raise ValueError("unsupported training sigma sampling mode")
        if any(value <= 0.0 for value in self.sigma_strata):
            raise ValueError("sigma strata must contain only positive values")
        if self.fixed_training_noise_seed is not None and self.train_limit != 1:
            raise ValueError("fixed training noise is restricted to the N=1 diagnostic")
        if not 0.0 <= self.input_perturbation_probability <= 1.0:
            raise ValueError("input_perturbation_probability must lie in [0,1]")
        if self.input_perturbation_scale < 0.0:
            raise ValueError("input_perturbation_scale must be non-negative")
        if self.rollout_interval < 0:
            raise ValueError("rollout_interval must be non-negative")
        if self.rollout_steps not in {1, 2}:
            raise ValueError("rollout_steps must be one or two")
        if self.rollout_score_weight < 0.0 or self.rollout_x0_weight < 0.0:
            raise ValueError("rollout loss weights must be non-negative")
        if self.rollout_maximum_step_rms_angstrom <= 0.0:
            raise ValueError("rollout maximum step RMS must be positive")
        if self.physics_auxiliary_weight < 0.0 or self.physics_warmup_epochs <= 0:
            raise ValueError("physics weight must be non-negative and warmup epochs positive")
        if not 0.0 < self.physics_sigma_full_strength < self.physics_sigma_cutoff:
            raise ValueError("physics sigma gate requires 0 < full_strength < cutoff")
        rollout_active = self.rollout_score_weight > 0.0 or self.rollout_x0_weight > 0.0
        if (self.rollout_interval > 0) != rollout_active:
            raise ValueError(
                "rollout interval and at least one rollout loss weight must be enabled together"
            )

    def to_dict(self) -> dict:
        return asdict(self)


def cosine_lr_factor(step: int, config: TrainConfig) -> float:
    if step < config.warmup_steps:
        return max((step + 1) / max(config.warmup_steps, 1), 1.0e-8)
    span = max(config.max_steps - config.warmup_steps, 1)
    progress = min(max((step - config.warmup_steps) / span, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.minimum_lr_ratio + (1.0 - config.minimum_lr_ratio) * cosine


@dataclass(frozen=True)
class LatticeTrainConfig:
    """Independent full-distribution lattice-training defaults."""

    seed: int = 20260903
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-5
    gradient_clip_norm: float = 5.0
    audit_candidates: int = 4
    data_epochs: int = 30
    batch_size: int = 256
    val_batch_size: int = 512
    num_workers: int = 4
    early_stopping_patience: int = 8

    def __post_init__(self) -> None:
        counts = (
            self.audit_candidates,
            self.data_epochs,
            self.batch_size,
            self.val_batch_size,
            self.early_stopping_patience,
        )
        if any(value <= 0 for value in counts) or self.num_workers < 0:
            raise ValueError("lattice training counts must be positive")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("invalid lattice optimizer configuration")
        if self.gradient_clip_norm <= 0.0:
            raise ValueError("lattice gradient clip norm must be positive")

    def to_dict(self) -> dict:
        return asdict(self)
__all__ = ["LatticeTrainConfig", "TrainConfig", "TrainingRunConfig", "cosine_lr_factor"]
