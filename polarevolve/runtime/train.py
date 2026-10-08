"""CLI for distributed MP20 score training."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from polarevolve import asset_root
from polarevolve.data.contracts import ExternalRoots
from polarevolve.diffusion.schedule import SigmaSchedule
from polarevolve.models.config import ScoreModelConfig
from polarevolve.runtime.training_cli import (
    add_common_training_arguments,
    common_train_kwargs,
    model_config_from_args,
    sigma_schedule_from_args,
)
from polarevolve.training.config import TrainConfig, TrainingRunConfig
from polarevolve.training.engine import run_training


def _parser() -> argparse.ArgumentParser:
    train = TrainConfig()
    model = ScoreModelConfig()
    sigma = SigmaSchedule()
    run = TrainingRunConfig(run_id="defaults")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=os.environ.get("POLAREVOLVE_DATA_ROOT"))
    parser.add_argument("--asset-root", default=os.environ.get("POLAREVOLVE_ASSET_ROOT", str(asset_root())))
    parser.add_argument("--output-root", default=os.environ.get("POLAREVOLVE_OUTPUT_ROOT"))
    parser.add_argument("--cache-relative", default=run.cache_relative)
    parser.add_argument("--group-assets-relative", default=run.group_assets_relative)
    parser.add_argument("--wyckoff-assets-relative", default=run.wyckoff_assets_relative)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--diffusion-metric", default=run.diffusion_metric)
    parser.add_argument("--resume")
    parser.add_argument(
        "--initialize-from",
        help="Load compatible model weights but reset optimizer, scheduler, and step.",
    )
    parser.add_argument(
        "--allow-cache-transfer",
        action="store_true",
        help="With --initialize-from, accept weights trained on a different ASU cache.",
    )
    add_common_training_arguments(parser, train, model, sigma)
    parser.add_argument(
        "--active-only",
        action="store_true",
        default=train.active_only,
        help="Train and validate only records containing non-0D ASU parameters.",
    )
    parser.add_argument("--rollout-interval", type=int, default=train.rollout_interval)
    parser.add_argument(
        "--mem-debug",
        action="store_true",
        default=os.environ.get("POLAREVOLVE_MEM_DEBUG") == "1",
        help="Log per-step CUDA memory diagnostics (alloc/peak/reserved and batch shape).",
    )
    parser.add_argument("--rollout-steps", type=int, default=train.rollout_steps)
    parser.add_argument("--rollout-score-weight", type=float, default=train.rollout_score_weight)
    parser.add_argument("--rollout-x0-weight", type=float, default=train.rollout_x0_weight)
    parser.add_argument(
        "--rollout-maximum-step-rms",
        type=float,
        default=train.rollout_maximum_step_rms_angstrom,
    )
    parser.add_argument(
        "--physics-auxiliary-weight", type=float, default=train.physics_auxiliary_weight
    )
    parser.add_argument("--physics-warmup-epochs", type=int, default=train.physics_warmup_epochs)
    parser.add_argument("--physics-prior")
    parser.add_argument("--physics-calibration")
    parser.add_argument("--replay-cache-relative")
    parser.add_argument("--replay-manifest-sha256")
    parser.add_argument(
        "--physics-sigma-full-strength",
        type=float,
        default=train.physics_sigma_full_strength,
    )
    parser.add_argument(
        "--physics-sigma-cutoff",
        type=float,
        default=train.physics_sigma_cutoff,
    )
    parser.add_argument(
        "--terminal-mixing-tolerance",
        type=float,
        default=run.terminal_mixing_tolerance,
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    roots = ExternalRoots(args.data_root, args.asset_root, args.output_root)
    run_training(
        roots=roots,
        run=TrainingRunConfig(
            run_id=args.run_id,
            cache_relative=args.cache_relative,
            group_assets_relative=args.group_assets_relative,
            wyckoff_assets_relative=args.wyckoff_assets_relative,
            resume=Path(args.resume).expanduser() if args.resume else None,
            initialize_from=(
                Path(args.initialize_from).expanduser() if args.initialize_from else None
            ),
            terminal_mixing_tolerance=args.terminal_mixing_tolerance,
            diffusion_metric=args.diffusion_metric,
            allow_cache_transfer=args.allow_cache_transfer,
        ),
        train_config=TrainConfig(
            **common_train_kwargs(args),
            active_only=args.active_only,
            mem_debug=args.mem_debug,
            rollout_interval=args.rollout_interval,
            rollout_steps=args.rollout_steps,
            rollout_score_weight=args.rollout_score_weight,
            rollout_x0_weight=args.rollout_x0_weight,
            rollout_maximum_step_rms_angstrom=args.rollout_maximum_step_rms,
            physics_auxiliary_weight=args.physics_auxiliary_weight,
            physics_warmup_epochs=args.physics_warmup_epochs,
            physics_prior=args.physics_prior,
            physics_calibration=args.physics_calibration,
            replay_cache_relative=args.replay_cache_relative,
            replay_manifest_sha256=args.replay_manifest_sha256,
            physics_sigma_full_strength=args.physics_sigma_full_strength,
            physics_sigma_cutoff=args.physics_sigma_cutoff,
        ),
        model_config=model_config_from_args(args),
        sigma_schedule=sigma_schedule_from_args(args),
    )


if __name__ == "__main__":
    main()
