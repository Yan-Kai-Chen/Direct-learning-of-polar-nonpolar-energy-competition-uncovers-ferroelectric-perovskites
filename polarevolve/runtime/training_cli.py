"""Shared command-line arguments for score-training entry points."""

from __future__ import annotations

import argparse

from polarevolve.diffusion.schedule import TRAINING_SIGMA_MODES, SigmaSchedule
from polarevolve.models.config import SCORE_PARAMETERIZATIONS, ScoreModelConfig
from polarevolve.training.config import TrainConfig

_TRAIN_FIELDS = (
    "seed", "max_steps", "evaluation_interval_epochs", "batch_size", "val_batch_size",
    "num_workers", "gradient_accumulation", "learning_rate", "weight_decay",
    "warmup_steps", "minimum_lr_ratio", "gradient_clip_norm", "log_every",
    "validate_every", "validation_batches", "save_every",
    "warmup_epochs", "checkpoint_minutes", "ema_half_life_epochs", "snapshot_interval_epochs",
)
_MODEL_FIELDS = ("hidden_dim", "time_dim", "radial_basis", "layers", "cutoff", "backbone_version")
_SIGMA_FIELDS = (
    ("sigma_min", "sigma_min"), ("sigma_max", "sigma_max"),
    ("sigma_max_cap", "sigma_max_cap"), ("levels", "sigma_levels"),
)

def add_common_training_arguments(
    parser: argparse.ArgumentParser, train: TrainConfig,
    model: ScoreModelConfig, sigma: SigmaSchedule,
) -> None:
    parser.add_argument("--data-epochs", type=int, default=train.data_epochs or 0)
    for name in _TRAIN_FIELDS:
        default = getattr(train, name)
        parser.add_argument(f"--{name.replace('_', '-')}", type=type(default), default=default)
    parser.add_argument("--train-limit", type=int, default=train.train_limit or 0)
    parser.add_argument("--val-limit", type=int, default=train.val_limit or 0)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default=train.precision)
    parser.add_argument("--validation-mode", choices=("legacy_batches", "full_unique"),
                        default=train.validation_mode)
    parser.add_argument("--epoch-mode", choices=("legacy_stream", "finite_unique"), default=train.epoch_mode)
    parser.add_argument("--resident-epoch", action="store_true", default=train.resident_epoch)
    parser.add_argument("--sigma-sampling-mode", choices=TRAINING_SIGMA_MODES,
                        default=train.sigma_sampling_mode)
    parser.add_argument(
        "--sigma-strata", default=",".join(str(value) for value in train.sigma_strata)
    )
    parser.add_argument(
        "--fixed-training-noise-seed", type=int,
        default=-1 if train.fixed_training_noise_seed is None else train.fixed_training_noise_seed,
    )
    for name in ("input_perturbation_probability", "input_perturbation_scale"):
        parser.add_argument(f"--{name.replace('_', '-')}", type=float, default=getattr(train, name))
    for name in _MODEL_FIELDS:
        default = getattr(model, name)
        parser.add_argument(f"--{name.replace('_', '-')}", type=type(default), default=default)
    for field, argument in _SIGMA_FIELDS:
        default = getattr(sigma, field)
        parser.add_argument(f"--{argument.replace('_', '-')}", type=type(default), default=default)
    parser.add_argument(
        "--score-parameterization", choices=SCORE_PARAMETERIZATIONS,
        default=model.score_parameterization,
    )


def common_train_kwargs(args: argparse.Namespace) -> dict:
    direct = (*_TRAIN_FIELDS, "precision", "validation_mode", "epoch_mode", "resident_epoch", "input_perturbation_probability",
              "input_perturbation_scale")
    return {name: getattr(args, name) for name in direct} | {
        "data_epochs": args.data_epochs or None,
        "train_limit": args.train_limit or None,
        "val_limit": args.val_limit or None,
        "sigma_sampling_mode": args.sigma_sampling_mode,
        "sigma_strata": tuple(
            float(item.strip()) for item in args.sigma_strata.split(",") if item.strip()
        ),
        "fixed_training_noise_seed": (
            args.fixed_training_noise_seed if args.fixed_training_noise_seed >= 0 else None
        ),
    }


def model_config_from_args(
    args: argparse.Namespace, *, condition_dim: int = 0, atom_condition_dim: int = 0
) -> ScoreModelConfig:
    values = {name: getattr(args, name) for name in _MODEL_FIELDS}
    return ScoreModelConfig(
        **values,
        score_parameterization=args.score_parameterization,
        condition_dim=condition_dim,
        atom_condition_dim=atom_condition_dim,
    )


def sigma_schedule_from_args(args: argparse.Namespace) -> SigmaSchedule:
    values = {field: getattr(args, argument) for field, argument in _SIGMA_FIELDS}
    return SigmaSchedule(**values)


__all__ = ["add_common_training_arguments", "common_train_kwargs",
           "model_config_from_args", "sigma_schedule_from_args"]
