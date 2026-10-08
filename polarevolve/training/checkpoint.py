"""Strict Version9 checkpoint loading shared by training and sampling."""

from __future__ import annotations

import os
import hashlib
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import numpy as np

from polarevolve.diffusion.schedule import SigmaSchedule
from polarevolve.diffusion.metric import (
    FULL_ASU_STATE_V1,
    MEMBER_SUM_CARTESIAN_V1,
    TRANSLATION_QUOTIENT_V1,
    metric_contract_metadata,
)
from polarevolve.crystal.symmetry import sha256_file
from polarevolve.models.config import PHYSICAL_SCORE_V1, PERIODIC_PRECISION_V1, ScoreModelConfig
from polarevolve.models.score import HardConditionScoreNetwork
from polarevolve.runtime.distributed import FIXED_PARAMETER_MEAN_V1, run_on_primary_and_broadcast
from polarevolve.runtime.filesystem import atomic_json
from polarevolve.training.data import LOADER_RNG_V1

LEGACY_CHECKPOINT_SCHEMA = "gt_sge_mp20_hard_only_checkpoint_v1"
SCORE_CHECKPOINT_SCHEMA = "gt_sge_mp20_hard_only_checkpoint_v2"
QUOTIENT_CHECKPOINT_SCHEMA = "gt_sge_mp20_hard_only_checkpoint_v3"
CHECKPOINT_SCHEMA = "gt_sge_mp20_hard_only_checkpoint_v4"
BACKBONE_CHECKPOINT_SCHEMA = "gt_sge_mp20_hard_only_checkpoint_v5"
SUPPORTED_CHECKPOINT_SCHEMAS = (
    LEGACY_CHECKPOINT_SCHEMA,
    SCORE_CHECKPOINT_SCHEMA,
    QUOTIENT_CHECKPOINT_SCHEMA,
    CHECKPOINT_SCHEMA,
    BACKBONE_CHECKPOINT_SCHEMA,
)
_SAMPLING_PAYLOAD_FIELDS = (
    "physics_identity",
    "task_identity",
    "schema_version",
    "state_quotient",
    "global_step",
    "model",
    "model_config",
    "sigma_schedule",
    "cache_manifest_sha256",
    "geometry_metric",
    "diffusion_metric",
    "score_preconditioner",
    "cell_representation",
    "backbone_contract",
)


@dataclass(frozen=True)
class ValidationBest:
    loss: float = float("inf")
    one_step_rmsd: float = float("inf")


@dataclass(frozen=True)
class TrainingStartup:
    step: int
    validation_best: ValidationBest
    run_manifest: dict
    checkpoint_initialization: dict
    ema_state: dict | None = None


def atomic_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def model_state_sha256(state: dict) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(f"{name}|{tensor.dtype}|{tuple(tensor.shape)}\n".encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def capture_rank_rng() -> dict:
    algorithm, values, position, has_gauss, gaussian = np.random.get_state()
    return {"python": random.getstate(), "numpy": (algorithm, values.tolist(), position, has_gauss, gaussian),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None}


def training_payload(
    *,
    model: HardConditionScoreNetwork,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    validation_best: ValidationBest,
    model_config: ScoreModelConfig,
    train_config: object,
    sigma_schedule: SigmaSchedule,
    cache_manifest_sha256: str,
    diffusion_metric: str,
    initialization: dict | None = None,
    physics_identity: dict | None = None,
    task_identity: dict | None = None,
) -> dict:
    if model.score_parameterization != model_config.score_parameterization:
        raise ValueError("model and checkpoint score parameterizations disagree")
    return {
        "schema_version": (BACKBONE_CHECKPOINT_SCHEMA
                           if model.backbone_version == 2 else CHECKPOINT_SCHEMA),
        "state_quotient": TRANSLATION_QUOTIENT_V1,
        "global_step": int(step),
        "best_validation_loss": float(validation_best.loss),
        "best_validation_one_step_rmsd": float(validation_best.one_step_rmsd),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "model_config": model_config.to_dict(),
        "train_config": train_config.to_dict(),
        "gradient_reduction_contract": (FIXED_PARAMETER_MEAN_V1
                                        if train_config.epoch_mode == "finite_unique" else "DDP_buckets_legacy"),
        "loader_rng_contract": LOADER_RNG_V1,
        "sigma_schedule": asdict(sigma_schedule),
        "cache_manifest_sha256": cache_manifest_sha256,
        **metric_contract_metadata(diffusion_metric),
        "initialization": initialization or {"mode": "from_scratch"},
        "physics_identity": physics_identity or {"enabled": False},
        "task_identity": task_identity or {"task": "mp20_coordinate_diffusion"},
        "backbone_contract": {
            "version": model.backbone_version,
            "graph": "complete_periodic_v2" if model.backbone_version == 2 else "minimum_image_v1",
            "aggregation": "one_plus_envelope_sum_v2" if model.backbone_version == 2 else "degree_v1",
            **({"precision": PERIODIC_PRECISION_V1} if model.backbone_version == 2 else {}),
        },
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }


def restore_training_state(
    payload: dict,
    *,
    model: HardConditionScoreNetwork,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    model_config: ScoreModelConfig,
    sigma_schedule: SigmaSchedule,
    cache_manifest_sha256: str,
    diffusion_metric: str,
    validation_mode: str = "legacy_batches",
    epoch_mode: str = "legacy_stream",
    rank: int = 0,
) -> tuple[int, ValidationBest]:
    previous_mode = payload.get("train_config", {}).get("validation_mode", "legacy_batches")
    if previous_mode != validation_mode:
        raise ValueError("validation selection changed; use model-only initialization")
    if payload.get("train_config", {}).get("epoch_mode", "legacy_stream") != epoch_mode:
        raise ValueError("training epoch convention changed; use model-only initialization")
    validate_training_identity(
        payload,
        model_config=model_config,
        sigma_schedule=sigma_schedule,
        cache_manifest_sha256=cache_manifest_sha256,
        diffusion_metric=diffusion_metric,
    )
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    torch.set_rng_state(payload["torch_rng_state"])
    if torch.cuda.is_available() and payload.get("cuda_rng_state_all") is not None:
        torch.cuda.set_rng_state_all(payload["cuda_rng_state_all"])
    if "rank_rng_states" in payload:
        state = payload["rank_rng_states"][rank]
        random.setstate(state["python"])
        algorithm, values, position, has_gauss, gaussian = state["numpy"]
        np.random.set_state((algorithm, np.asarray(values, dtype=np.uint32), position, has_gauss, gaussian))
        torch.set_rng_state(state["torch"])
        if state["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(state["cuda"])
    return int(payload["global_step"]), ValidationBest(
        loss=float(payload.get("best_validation_loss", float("inf"))),
        one_step_rmsd=float(payload.get("best_validation_one_step_rmsd", float("inf"))),
    )


def initialize_model_state(
    payload: dict,
    *,
    model: HardConditionScoreNetwork,
    model_config: ScoreModelConfig,
    cache_manifest_sha256: str,
    diffusion_metric: str,
    allow_condition_extension: bool = False,
    allow_cache_mismatch: bool = False,
) -> None:
    """Load model weights only while keeping a new optimization/schedule run."""

    if payload.get("schema_version") not in {
        QUOTIENT_CHECKPOINT_SCHEMA,
        CHECKPOINT_SCHEMA,
        BACKBONE_CHECKPOINT_SCHEMA,
    }:
        raise ValueError("only v3/v4/v5 checkpoints can initialize current training")
    if payload.get("state_quotient") != TRANSLATION_QUOTIENT_V1:
        raise ValueError("initialization checkpoint uses a different state quotient")
    source_config = _model_config_from_payload(payload)
    source_values = source_config.to_dict()
    target_values = model_config.to_dict()
    same_config = source_values == target_values
    source_condition_dim = int(source_values.pop("condition_dim", 0))
    target_condition_dim = int(target_values.pop("condition_dim", 0))
    source_atom_condition_dim = int(source_values.pop("atom_condition_dim", 0))
    target_atom_condition_dim = int(target_values.pop("atom_condition_dim", 0))
    condition_extension = (
        allow_condition_extension
        and source_condition_dim in (0, target_condition_dim)
        and source_atom_condition_dim in (0, target_atom_condition_dim)
        and (source_condition_dim, source_atom_condition_dim)
        != (target_condition_dim, target_atom_condition_dim)
        and source_values == target_values
    )
    if not same_config and not condition_extension:
        raise ValueError("initialization model configuration differs from the current run")
    if (
        payload["cache_manifest_sha256"] != cache_manifest_sha256
        and not allow_cache_mismatch
    ):
        raise ValueError("initialization cache manifest differs from the current run")
    if condition_extension:
        incompatible = model.load_state_dict(payload["model"], strict=False)
        expected_missing = {
            name
            for name in model.state_dict()
            if (
                source_condition_dim == 0
                and name.startswith("condition_projection.")
            )
            or (
                source_atom_condition_dim == 0
                and name.startswith("atom_condition_projection.")
            )
        }
        if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
            raise ValueError("checkpoint trunk does not match the condition-aware score network")
        for projection in (model.condition_projection, model.atom_condition_projection):
            if projection is not None:
                output = projection[-1]
                torch.nn.init.zeros_(output.weight)
                torch.nn.init.zeros_(output.bias)
        return
    source_metric = diffusion_metric_from_payload(payload)
    if (
        same_config
        and payload["schema_version"] in {CHECKPOINT_SCHEMA, BACKBONE_CHECKPOINT_SCHEMA}
        and source_metric == diffusion_metric
    ):
        model.load_state_dict(payload["model"], strict=True)
        return
    state = {
        name: value
        for name, value in payload["model"].items()
        if not name.startswith("edge_output.")
    }
    incompatible = model.load_state_dict(state, strict=False)
    expected_missing = {name for name in model.state_dict() if name.startswith("edge_output.")}
    if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
        raise ValueError("checkpoint trunk does not match the current score network")
    for module in model.edge_output.modules():
        reset = getattr(module, "reset_parameters", None)
        if reset is not None:
            reset()


def prepare_training_startup(
    *,
    resume_path: Path | None,
    initialize_path: Path | None,
    model: HardConditionScoreNetwork,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    model_config: ScoreModelConfig,
    sigma_schedule: SigmaSchedule,
    cache_manifest_sha256: str,
    diffusion_metric: str,
    rank: int,
    physics_identity: dict | None = None,
    task_identity: dict | None = None,
    allow_condition_extension: bool = False,
    allow_cache_mismatch_on_initialize: bool = False,
    validation_mode: str = "legacy_batches",
    epoch_mode: str = "legacy_stream",
    resume_train_config: object | None = None,
    world_size: int = 1,
) -> TrainingStartup:
    """Resolve mutually exclusive scratch, resume, and model-only startup modes."""

    if resume_path is None and initialize_path is None:
        scratch = {"mode": "from_scratch"}
        return TrainingStartup(0, ValidationBest(), scratch, scratch)
    path = resume_path if resume_path is not None else initialize_path
    mode = "resume" if resume_path is not None else "initialize_model_only"
    payload = run_on_primary_and_broadcast(
        lambda: load_payload(path),
        rank=rank,
        operation=f"{mode} checkpoint load",
    )
    if not isinstance(payload, dict):
        raise RuntimeError("rank 0 did not broadcast a valid checkpoint payload")
    digest = run_on_primary_and_broadcast(
        lambda: sha256_file(path),
        rank=rank,
        operation=f"{mode} checkpoint hash",
    )
    if resume_path is not None:
        if epoch_mode == "finite_unique":
            if payload.get("loader_rng_contract") != LOADER_RNG_V1:
                raise ValueError("finite epoch loader RNG contract changed on resume")
            if payload.get("gradient_reduction_contract") != FIXED_PARAMETER_MEAN_V1:
                raise ValueError("finite epoch gradient reduction contract changed on resume")
            fields = ("batch_size", "gradient_accumulation", "seed", "data_epochs", "warmup_steps",
                      "learning_rate", "minimum_lr_ratio", "weight_decay", "precision", "ema_half_life_epochs",
                      "replay_cache_relative", "replay_manifest_sha256")
            expected = resume_train_config.to_dict() if resume_train_config is not None else {}
            if any(payload["train_config"].get(key) != expected.get(key) for key in fields):
                raise ValueError("finite epoch optimization contract changed on resume")
            if payload.get("data_cursor", {}).get("world_size") != world_size:
                raise ValueError("finite epoch world size changed on resume")
            if world_size > 1 and len(payload.get("rank_rng_states", [])) != world_size:
                raise ValueError("finite DDP resume requires RNG states for every rank")
        if physics_identity is not None:
            previous = payload.get(
                "physics_identity",
                {"enabled": payload.get("train_config", {}).get("physics_auxiliary_weight", 0) > 0},
            )
            if previous != physics_identity:
                raise ValueError("physics objective/prior changed; use model-only initialization")
        if task_identity is not None and payload.get("task_identity") != task_identity:
            raise ValueError("task/condition identity changed; use model-only initialization")
        step, selected = restore_training_state(
            payload,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            model_config=model_config,
            sigma_schedule=sigma_schedule,
            cache_manifest_sha256=cache_manifest_sha256,
            diffusion_metric=diffusion_metric,
            validation_mode=validation_mode,
            epoch_mode=epoch_mode,
            rank=rank,
        )
        startup = {
            "mode": mode,
            "checkpoint": str(path),
            "checkpoint_sha256": digest,
            "source_global_step": int(payload["global_step"]),
        }
        initialization = dict(payload.get("initialization", {"mode": "from_scratch"}))
        return TrainingStartup(step, selected, startup, initialization, payload.get("ema_state"))
    initialize_model_state(
        payload,
        model=model,
        model_config=model_config,
        cache_manifest_sha256=cache_manifest_sha256,
        diffusion_metric=diffusion_metric,
        allow_condition_extension=allow_condition_extension,
        allow_cache_mismatch=allow_cache_mismatch_on_initialize,
    )
    startup = {
        "mode": mode,
        "checkpoint": str(path),
        "checkpoint_sha256": digest,
        "source_global_step": int(payload["global_step"]),
        "source_sigma_schedule": dict(payload["sigma_schedule"]),
        "source_cache_manifest_sha256": str(payload["cache_manifest_sha256"]),
        "cache_transfer": str(payload["cache_manifest_sha256"]) != cache_manifest_sha256,
    }
    return TrainingStartup(0, ValidationBest(), startup, startup)


def persist_training_checkpoints(
    *,
    output_dir: Path,
    payload: dict,
    step: int,
    validated: bool,
    validation_loss: float,
    validation_one_step_rmsd: float,
    previous_best: ValidationBest,
    retain_snapshot: bool = True,
) -> ValidationBest:
    """Persist explicit selection targets and immutable validation snapshots."""

    loss_improved = validated and validation_loss < previous_best.loss
    rmsd_improved = validated and validation_one_step_rmsd < previous_best.one_step_rmsd
    updated = ValidationBest(
        loss=validation_loss if loss_improved else previous_best.loss,
        one_step_rmsd=(validation_one_step_rmsd if rmsd_improved else previous_best.one_step_rmsd),
    )
    payload["best_validation_loss"] = float(updated.loss)
    payload["best_validation_one_step_rmsd"] = float(updated.one_step_rmsd)
    atomic_checkpoint(output_dir / "last.pt", payload)
    if not validated:
        return updated

    snapshot = output_dir / "checkpoints" / f"step_{step:06d}.pt"
    if retain_snapshot:
        atomic_checkpoint(snapshot, payload)
    if loss_improved:
        atomic_checkpoint(output_dir / "best_loss.pt", payload)
        atomic_checkpoint(output_dir / "best.pt", payload)
    if rmsd_improved:
        atomic_checkpoint(output_dir / "best_one_step_rmsd.pt", payload)
    atomic_json(
        output_dir / "checkpoint_manifest.json",
        {
            "schema_version": "gt_sge_checkpoint_selection_v2",
            "state_quotient": TRANSLATION_QUOTIENT_V1,
            **{key: payload[key] for key in metric_contract_metadata(payload["diffusion_metric"])},
            "latest_validation_step": int(step),
            "validation_noise_policy": "fixed_across_checkpoints",
            "selectors": {
                "validation_loss": {
                    "value": float(updated.loss),
                    "checkpoint": "best_loss.pt",
                    "compatibility_alias": "best.pt",
                },
                "validation_one_step_rmsd_angstrom": {
                    "value": float(updated.one_step_rmsd),
                    "checkpoint": "best_one_step_rmsd.pt",
                },
            },
            "validation_snapshots": [
                path.relative_to(output_dir).as_posix()
                for path in sorted((output_dir / "checkpoints").glob("step_*.pt"))
            ],
        },
    )
    return updated


def load_payload(path: str | Path) -> dict:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"checkpoint is missing: {resolved}")
    payload = torch.load(resolved, map_location="cpu")
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") not in SUPPORTED_CHECKPOINT_SCHEMAS
    ):
        raise ValueError("unsupported Version9 checkpoint schema")
    required = {
        "global_step",
        "model",
        "model_config",
        "sigma_schedule",
        "cache_manifest_sha256",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"checkpoint is missing required fields: {missing}")
    model_config = payload["model_config"]
    if not isinstance(model_config, dict):
        raise ValueError("checkpoint model_config must be a mapping")
    if payload["schema_version"] in {
        SCORE_CHECKPOINT_SCHEMA,
        QUOTIENT_CHECKPOINT_SCHEMA,
        CHECKPOINT_SCHEMA,
        BACKBONE_CHECKPOINT_SCHEMA,
    }:
        if "score_parameterization" not in model_config:
            raise ValueError("checkpoint omits score_parameterization")
    elif "score_parameterization" in model_config:
        raise ValueError("legacy checkpoint unexpectedly declares new score semantics")
    if payload["schema_version"] in {
        QUOTIENT_CHECKPOINT_SCHEMA,
        CHECKPOINT_SCHEMA,
        BACKBONE_CHECKPOINT_SCHEMA,
    }:
        if payload.get("state_quotient") != TRANSLATION_QUOTIENT_V1:
            raise ValueError("quotient checkpoint omits the translation quotient contract")
    elif "state_quotient" in payload:
        raise ValueError("pre-quotient checkpoint unexpectedly declares state_quotient")
    if payload["schema_version"] in {
        CHECKPOINT_SCHEMA,
        BACKBONE_CHECKPOINT_SCHEMA,
    }:
        expected_contract = metric_contract_metadata(str(payload.get("diffusion_metric", "")))
        if any(payload.get(key) != value for key, value in expected_contract.items()):
            raise ValueError("v4 checkpoint has an incomplete metric contract")
    if payload["schema_version"] == BACKBONE_CHECKPOINT_SCHEMA:
        expected = {"version": 2, "graph": "complete_periodic_v2",
                    "aggregation": "one_plus_envelope_sum_v2", "precision": PERIODIC_PRECISION_V1}
        if model_config.get("backbone_version") != 2 or payload.get("backbone_contract") != expected:
            raise ValueError("v5 checkpoint has an incomplete periodic backbone contract")
    elif model_config.get("backbone_version", 1) != 1:
        raise ValueError("periodic backbone requires a v5 checkpoint")
    return payload


def sampling_payload(payload: dict) -> dict:
    """Discard training-only fields from a payload already validated by ``load_payload``."""

    return {name: payload[name] for name in _SAMPLING_PAYLOAD_FIELDS if name in payload}


def load_sampling_payload(path: str | Path) -> dict:
    """Strictly load a checkpoint, then discard training-only rank payload."""

    return sampling_payload(load_payload(path))


def diffusion_metric_from_payload(payload: dict) -> str:
    if payload.get("schema_version") in {CHECKPOINT_SCHEMA, BACKBONE_CHECKPOINT_SCHEMA}:
        return str(payload["diffusion_metric"])
    return MEMBER_SUM_CARTESIAN_V1


def state_quotient_from_payload(payload: dict) -> str:
    if payload.get("schema_version") in {
        QUOTIENT_CHECKPOINT_SCHEMA,
        CHECKPOINT_SCHEMA,
        BACKBONE_CHECKPOINT_SCHEMA,
    }:
        return TRANSLATION_QUOTIENT_V1
    return FULL_ASU_STATE_V1


def _model_config_from_payload(payload: dict) -> ScoreModelConfig:
    values = dict(payload["model_config"])
    if payload["schema_version"] == LEGACY_CHECKPOINT_SCHEMA:
        values["score_parameterization"] = PHYSICAL_SCORE_V1
    return ScoreModelConfig(**values)


def model_from_payload(
    payload: dict,
    *,
    cache_manifest_sha256: str,
) -> tuple[HardConditionScoreNetwork, ScoreModelConfig, SigmaSchedule]:
    if payload["cache_manifest_sha256"] != cache_manifest_sha256:
        raise ValueError("checkpoint cache manifest differs from the active cache")
    model_config = _model_config_from_payload(payload)
    schedule = SigmaSchedule(**payload["sigma_schedule"])
    model = HardConditionScoreNetwork(**model_config.to_dict())
    model.load_state_dict(payload["model"], strict=True)
    return model, model_config, schedule


def validate_training_identity(
    payload: dict,
    *,
    model_config: ScoreModelConfig,
    sigma_schedule: SigmaSchedule,
    cache_manifest_sha256: str,
    diffusion_metric: str,
) -> None:
    expected_schema = BACKBONE_CHECKPOINT_SCHEMA if model_config.backbone_version == 2 else CHECKPOINT_SCHEMA
    if payload.get("schema_version") != expected_schema:
        raise ValueError(
            "pre-v4 checkpoints are diagnostic-only and cannot resume the explicit metric contract"
        )
    if payload.get("state_quotient") != TRANSLATION_QUOTIENT_V1:
        raise ValueError("checkpoint state quotient differs from the current run")
    expected_metric = metric_contract_metadata(diffusion_metric)
    if expected_metric != {key: payload.get(key) for key in expected_metric}:
        raise ValueError("checkpoint metric contract differs from the current run")
    if _model_config_from_payload(payload).to_dict() != model_config.to_dict():
        raise ValueError("checkpoint model configuration differs from the current run")
    if payload["sigma_schedule"] != asdict(sigma_schedule):
        raise ValueError("checkpoint sigma schedule differs from the current run")
    if payload["cache_manifest_sha256"] != cache_manifest_sha256:
        raise ValueError("checkpoint cache manifest differs from the current run")


__all__ = [
    "CHECKPOINT_SCHEMA",
    "BACKBONE_CHECKPOINT_SCHEMA",
    "LEGACY_CHECKPOINT_SCHEMA",
    "SCORE_CHECKPOINT_SCHEMA",
    "QUOTIENT_CHECKPOINT_SCHEMA",
    "ValidationBest",
    "TrainingStartup",
    "atomic_checkpoint",
    "diffusion_metric_from_payload",
    "initialize_model_state",
    "load_payload",
    "load_sampling_payload",
    "model_from_payload",
    "persist_training_checkpoints",
    "prepare_training_startup",
    "restore_training_state",
    "sampling_payload",
    "state_quotient_from_payload",
    "training_payload",
    "validate_training_identity",
]
