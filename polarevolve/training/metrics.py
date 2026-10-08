"""Validation and append-only metric persistence for training runs."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from polarevolve.data.provenance import source_fingerprint
from polarevolve.diffusion.metric import TRANSLATION_QUOTIENT_V1
from polarevolve.runtime.distributed import (
    FIXED_PARAMETER_MEAN_V1,
    gather_named_1d_across_ranks,
    mean_named_scalars_across_ranks,
    raise_if_any_rank_failed,
)
from polarevolve.runtime.filesystem import atomic_json
from polarevolve.training.data import LOADER_RNG_V1, ResidentValidationLoader, effective_worker_count

CALIBRATION_SIGMAS = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
CALIBRATION_STATISTICS = ("mean", "median", "p10", "p90")
TRAIN_METRIC_ATTRIBUTES = (
    ("loss", "loss"),
    ("score_loss", "score_loss"),
    ("one_step_rmsd", "one_step_cartesian_rmsd"),
    ("identity_rmsd", "identity_cartesian_rmsd"),
    ("denoising_rmsd_improvement", "denoising_rmsd_improvement"),
    ("denoising_relative_improvement", "denoising_relative_improvement"),
    ("rollout_score_loss", "rollout_score_loss"),
    ("rollout_x0_mse", "rollout_x0_cartesian_mse"),
    ("input_perturbation_fraction", "input_perturbation_fraction"),
    ("input_perturbation_rmsd", "input_perturbation_rmsd"),
    ("rollout_clipped_fraction", "rollout_clipped_fraction"),
    ("score_dual_cosine", "score_dual_cosine"),
    ("score_dual_norm_ratio", "score_dual_norm_ratio"),
    ("physics_loss", "physics_loss"),
    ("physics_coordination_excess", "physics_coordination_excess"),
    ("physics_coordination_coverage", "physics_coordination_coverage"),
    ("physics_overlap_excess", "physics_overlap_excess"),
    ("physics_bond_radius_excess", "physics_bond_radius_excess"),
    ("physics_bond_valence_excess", "physics_bond_valence_excess"),
    (
        "physics_bond_valence_applicable_fraction",
        "physics_bond_valence_applicable_fraction",
    ),
)
TRAIN_METRIC_NAMES = tuple(name for name, _ in TRAIN_METRIC_ATTRIBUTES)
VALIDATION_METRIC_NAMES = (
    "loss",
    "score_loss",
    "one_step_rmsd",
    "identity_rmsd",
    "denoising_rmsd_improvement",
    "denoising_relative_improvement",
    "rollout_score_loss",
    "rollout_x0_mse",
    "rollout_clipped_fraction",
    "physics_loss",
    "physics_coordination_excess",
    "physics_coordination_coverage",
    "physics_overlap_excess",
    "physics_bond_radius_excess",
    "physics_bond_valence_excess",
    "physics_bond_valence_applicable_fraction",
)
_OBJECTIVE_ATTRIBUTE_BY_METRIC = dict(TRAIN_METRIC_ATTRIBUTES)
_FULL_UNIQUE_NOISE_CONTRACT = "sym1_full_unique_noise_v1"


def _sigma_label(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def calibration_metric_names() -> tuple[str, ...]:
    names = []
    for sigma in CALIBRATION_SIGMAS:
        label = _sigma_label(sigma)
        for metric in ("cosine", "norm_ratio"):
            for statistic in CALIBRATION_STATISTICS:
                middle = "" if statistic == "mean" else f"_{statistic}"
                names.append(f"calibration_{metric}{middle}_sigma_{label}")
    return tuple(names)


def objective_metric_tensors(
    output: object,
    names: tuple[str, ...] = TRAIN_METRIC_NAMES,
) -> dict[str, torch.Tensor]:
    return {name: getattr(output, _OBJECTIVE_ATTRIBUTE_BY_METRIC[name]) for name in names}


def zero_metric_tensors(names: tuple[str, ...], *, device: torch.device) -> dict[str, torch.Tensor]:
    return {name: torch.zeros((), device=device) for name in names}


def empty_validation_metrics() -> dict[str, float]:
    return {name: float("nan") for name in (*VALIDATION_METRIC_NAMES, *calibration_metric_names())}


def parameter_update_metrics(model, optimizer, previous: dict) -> dict:
    """Layer-level actual AdamW updates; no inference from optimizer-step labels."""
    groups = {}
    for name, parameter in model.named_parameters():
        key = ".".join(name.split(".")[:2]) if name.split(".")[0] in {"message_layers", "update_layers", "norms"} else name.split(".")[0]
        state = optimizer.state.get(parameter, {})
        zero = parameter.new_zeros(())
        values = [parameter.detach().float().square().sum(),
                  (parameter.detach() - previous[name]).float().square().sum(),
                  parameter.grad.detach().float().square().sum() if parameter.grad is not None else zero,
                  state["exp_avg"].float().square().sum() if "exp_avg" in state else zero,
                  state["exp_avg_sq"].float().sum() if "exp_avg_sq" in state else zero]
        packed = torch.stack(values)
        groups[key] = groups.get(key, torch.zeros_like(packed)) + packed
    result = {}
    for key, packed in groups.items():
        norm, update, gradient, first_moment, second_moment = packed.sqrt().cpu().tolist()
        result.update({f"{key}_parameter_norm": norm, f"{key}_update_norm": update,
                       f"{key}_relative_update": update / max(norm, 1e-12),
                       f"{key}_clipped_gradient_norm": gradient,
                       f"{key}_first_moment_norm": first_moment,
                       f"{key}_second_moment_root_sum": second_moment})
    return result


def training_history_row(
    *,
    step: int,
    train_metrics: Mapping[str, torch.Tensor],
    validation_metrics: Mapping[str, float],
    learning_rate: float,
    gradient_norm: float,
    gradient_clipped: float,
) -> dict[str, float | int]:
    row: dict[str, float | int] = {"step": step}
    row.update((f"train_{name}", float(train_metrics[name])) for name in TRAIN_METRIC_NAMES)
    row.update((f"val_{name}", float(validation_metrics[name])) for name in VALIDATION_METRIC_NAMES)
    row.update(
        (f"val_{name}", float(validation_metrics[name])) for name in calibration_metric_names()
    )
    row.update(
        learning_rate=float(learning_rate),
        gradient_norm=float(gradient_norm),
        gradient_clipped=float(gradient_clipped),
    )
    return row


def _calibration_statistics(value: torch.Tensor) -> dict[str, torch.Tensor]:
    if value.numel() == 0:
        missing = torch.full((), float("nan"), device=value.device)
        return {name: missing.clone() for name in CALIBRATION_STATISTICS}
    return {
        "mean": value.mean(),
        "median": value.median(),
        "p10": torch.quantile(value, 0.1),
        "p90": torch.quantile(value, 0.9),
    }


def append_metrics(path: Path, row: dict[str, float | int | str]) -> None:
    exists = path.is_file()
    if exists:
        with path.open(newline="", encoding="utf-8") as handle:
            if next(csv.reader(handle), None) != list(row):
                raise ValueError(
                    "metrics schema changed; preserve history and use a new run directory"
                )
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def reconcile_metrics_for_resume(path: Path, checkpoint_step: int) -> int:
    """Atomically discard metric rows newer than the resumed checkpoint."""

    if not path.is_file():
        raise FileNotFoundError(f"resume metrics are missing: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if not fieldnames or "step" not in fieldnames:
        raise ValueError("resume metrics must contain a step column")
    steps = [int(row["step"]) for row in rows]
    if steps != sorted(steps):
        raise ValueError("resume metrics are not ordered by step")
    retained = [row for row, step in zip(rows, steps) if step <= checkpoint_step]
    if not retained or int(retained[-1]["step"]) != checkpoint_step:
        raise ValueError("resume checkpoint step is absent from metrics history")
    removed = len(rows) - len(retained)
    if removed == 0:
        return 0
    temporary = path.with_suffix(path.suffix + ".resume.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(retained)
    temporary.replace(path)
    return removed


def write_training_run_manifest(
    *,
    output_dir: Path,
    cache_root: Path,
    group_assets: Path,
    wyckoff_assets: Path,
    manifest: Any,
    physics_identity: dict,
    physics_config: Any,
    objective: Any,
    model: torch.nn.Module,
    model_config: Any,
    train_config: Any,
    sigma_schedule: Any,
    startup: Any,
    data_plan: Any,
    training_budget: Any,
    world_size: int,
) -> None:
    """Persist the admitted training identities and partition plan."""
    physics = objective.physics_guidance

    def worker_counts(plan, batch):
        return [
            effective_worker_count(train_config.num_workers, rank_records=count, batch_size=batch)
            for count in plan.records_per_rank
        ]

    atomic_json(
        output_dir / "run_manifest.json",
        {
            "schema_version": "gt_sge_run_manifest_v3",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "task": "structure_score_balanced_replay" if train_config.replay_cache_relative else "mp20_hard_conditioned_asu_score",
            "source_fingerprint": source_fingerprint(),
            "lattice_mode": "verified_cache_lattice_coordinate_pilot",
            "soft_guidance_enabled": train_config.physics_auxiliary_weight > 0.0,
            "physics_guidance": {
                "contract": physics_identity.get("contract"),
                "mode": "target_relative_training_auxiliary",
                "enabled": train_config.physics_auxiliary_weight > 0.0,
                "config": asdict(physics_config),
                "provenance": physics.provenance if physics is not None else None,
            },
            "state_quotient": TRANSLATION_QUOTIENT_V1,
            "world_size": world_size,
            "gradient_reduction_contract": (FIXED_PARAMETER_MEAN_V1
                                            if train_config.epoch_mode == "finite_unique" else "DDP_buckets_legacy"),
            "loader_rng_contract": LOADER_RNG_V1,
            "effective_batch_size": train_config.batch_size
            * world_size
            * train_config.gradient_accumulation * (2 if train_config.replay_cache_relative else 1),
            "replay": {"enabled": train_config.replay_cache_relative is not None,
                       "cache_relative": train_config.replay_cache_relative,
                       "manifest_sha256": train_config.replay_manifest_sha256,
                       "loss_contract": "equal_separate_structure_means_v1",
                       "epoch_clock": "primary_structures_only"},
            "training_budget": asdict(training_budget),
            "cache_manifest_sha256": manifest.manifest_sha256,
            "paths": {
                "cache_root": str(cache_root.resolve()),
                "group_assets": str(group_assets.resolve()),
                "wyckoff_assets": str(wyckoff_assets.resolve()),
                "output_dir": str(output_dir.resolve()),
            },
            "model": model_config.to_dict(),
            "model_parameter_count": sum(p.numel() for p in model.parameters()),
            "training": train_config.to_dict(),
            "sigma_schedule": asdict(sigma_schedule),
            "startup": startup.run_manifest,
            "partition_plans": {
                "train": {
                    **asdict(data_plan.train),
                    "raw_available_records": manifest.split_count("train"),
                    "active_available_records": data_plan.active_train_records,
                    "workers_per_rank": worker_counts(data_plan.train, train_config.batch_size),
                },
                "val": {
                    **asdict(data_plan.val),
                    "raw_available_records": manifest.split_count("val"),
                    "active_available_records": data_plan.active_val_records,
                    "workers_per_rank": worker_counts(data_plan.val, train_config.val_batch_size),
                },
            },
        },
    )


@torch.no_grad()
def validate_objective(
    module: torch.nn.Module,
    loader: DataLoader,
    *,
    batches: int,
    device: torch.device,
    precision: str,
    seed: int,
    rank: int,
    rollout_enabled: bool,
    sigma_strata: tuple[float, ...] = (),
    fixed_noise_seed: int | None = None,
    validation_mode: str = "legacy_batches",
    calibration_enabled: bool = True,
) -> dict[str, float]:
    if validation_mode not in {"legacy_batches", "full_unique"}:
        raise ValueError("validation_mode must be legacy_batches or full_unique")
    if validation_mode == "full_unique":
        return _validate_full_unique(
            module, loader, device=device, precision=precision, seed=seed, rank=rank,
            rollout_enabled=rollout_enabled, sigma_strata=sigma_strata,
            fixed_noise_seed=fixed_noise_seed,
            calibration_enabled=calibration_enabled,
        )
    module.eval()
    generator = torch.Generator(device=device).manual_seed(seed + 50_003 * rank)
    fixed_noise_generator = (
        torch.Generator(device=device).manual_seed(fixed_noise_seed + 90_001 * rank)
        if fixed_noise_seed is not None
        else None
    )
    iterator = iter(loader)
    totals = zero_metric_tensors(VALIDATION_METRIC_NAMES, device=device)
    calibration_batch = None
    for batch_index in range(batches):
        batch = next(iterator).to(device, non_blocking=device.type == "cuda")
        if calibration_batch is None:
            calibration_batch = batch
        sigma_by_structure = None
        if sigma_strata:
            start = batch_index * batch.batch_size
            sigma_by_structure = torch.tensor(
                [
                    sigma_strata[(start + index) % len(sigma_strata)]
                    for index in range(batch.batch_size)
                ],
                device=device,
                dtype=batch.clean_u.dtype,
            )
        standard_noise = None
        if fixed_noise_generator is not None:
            standard_noise = torch.randn(
                batch.clean_u.shape,
                device=device,
                dtype=batch.clean_u.dtype,
                generator=fixed_noise_generator,
            )
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=precision == "bf16" and device.type == "cuda",
        ):
            output = module(
                batch,
                generator=generator,
                sigma_by_structure=sigma_by_structure,
                standard_noise=standard_noise,
                input_perturbation_enabled=False,
                rollout_enabled=rollout_enabled,
            )
        for name, value in objective_metric_tensors(output, VALIDATION_METRIC_NAMES).items():
            totals[name] += value.detach().float()
    if calibration_batch is None:
        raise RuntimeError("validation loader did not yield a calibration batch")
    standard_noise = torch.linspace(
        -0.75,
        0.75,
        calibration_batch.num_parameters,
        device=device,
        dtype=calibration_batch.clean_u.dtype,
    )
    local_calibration = {}
    for sigma_value in CALIBRATION_SIGMAS:
        sigma = calibration_batch.clean_u.new_full((calibration_batch.batch_size,), sigma_value)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=precision == "bf16" and device.type == "cuda",
        ):
            output = module(
                calibration_batch,
                sigma_by_structure=sigma,
                standard_noise=standard_noise,
                input_perturbation_enabled=False,
                rollout_enabled=False,
            )
        label = _sigma_label(sigma_value)
        for metric, values in (
            ("cosine", output.score_dual_cosine_by_structure),
            ("norm_ratio", output.score_dual_norm_ratio_by_structure),
        ):
            local_calibration[f"{metric}:{label}"] = values
    gathered_calibration = gather_named_1d_across_ranks(local_calibration)
    for sigma_value in CALIBRATION_SIGMAS:
        label = _sigma_label(sigma_value)
        for metric in ("cosine", "norm_ratio"):
            for statistic, value in _calibration_statistics(
                gathered_calibration[f"{metric}:{label}"]
            ).items():
                middle = "" if statistic == "mean" else f"_{statistic}"
                totals[f"calibration_{metric}{middle}_sigma_{label}"] = value
    reduced = mean_named_scalars_across_ranks(
        {
            name: value if name.startswith("calibration_") else value / batches
            for name, value in totals.items()
        }
    )
    module.train()
    return {name: float(value.cpu()) for name, value in reduced.items()}


def _validation_id_seed(seed: int, material_id: str, stream: str) -> int:
    payload = json.dumps(
        [_FULL_UNIQUE_NOISE_CONTRACT, seed, material_id, stream], separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def _full_unique_corruption_inputs(batch, objective, *, seed, sigma_strata, fixed_noise_seed):
    noise, sigma_draws = [], []
    for material_id, layout in zip(batch.material_ids, batch.layouts):
        noise_rng = torch.Generator().manual_seed(_validation_id_seed(
            seed if fixed_noise_seed is None else fixed_noise_seed, material_id, "noise",
        ))
        noise.append(torch.randn(layout.num_parameters, generator=noise_rng,
                                 dtype=batch.clean_u.dtype))
        sigma_rng = torch.Generator().manual_seed(_validation_id_seed(seed, material_id, "sigma"))
        if sigma_strata:
            index = int(torch.randint(len(sigma_strata), (), generator=sigma_rng))
            sigma_draws.append(torch.tensor(sigma_strata[index], dtype=batch.clean_u.dtype))
        else:
            sigma_draws.append(torch.rand((), generator=sigma_rng, dtype=batch.clean_u.dtype))
    sigma = torch.stack(sigma_draws).to(batch.clean_u)
    if not sigma_strata:
        # Keep adaptive maxima and cap admission at the existing objective owner.
        _, maximum = objective.training_sigmas(
            batch, generator=torch.Generator(device=batch.clean_u.device).manual_seed(seed),
        )
        log_minimum = torch.log(batch.clean_u.new_tensor(objective.schedule.sigma_min))
        sigma = torch.exp(log_minimum + sigma * (torch.log(maximum) - log_minimum))
    return sigma, torch.cat(noise).to(batch.clean_u)


def _fixed_validation_inputs(loader, index, batch, objective, *, seed, sigma_strata, fixed_noise_seed):
    cache = loader.fixed_validation_inputs if isinstance(loader, ResidentValidationLoader) else None
    key = None if cache is None else (id(objective), seed, sigma_strata, fixed_noise_seed, tuple(asdict(objective.schedule).items()),
           objective.diffusion_metric, objective.sigma_sampling_mode, objective.terminal_mixing_tolerance,
           batch.material_ids, tuple((name, id(value), value._version) for name, value in vars(batch).items()
                                     if isinstance(value, torch.Tensor)))
    if cache is not None and index in cache and cache[index][0] == key:
        return cache[index][1:]
    sigma, noise = _full_unique_corruption_inputs(
        batch, objective, seed=seed, sigma_strata=sigma_strata, fixed_noise_seed=fixed_noise_seed,
    )
    probe = torch.cat([torch.linspace(-0.75, 0.75, layout.num_parameters, device=batch.clean_u.device,
                                     dtype=batch.clean_u.dtype) for layout in batch.layouts])
    if cache is not None:
        cache[index] = (key, sigma, noise, probe)
    return sigma, noise, probe


def _validate_full_unique(
    module: torch.nn.Module, loader: DataLoader, *, device: torch.device, precision: str,
    seed: int, rank: int, rollout_enabled: bool, sigma_strata: tuple[float, ...],
    fixed_noise_seed: int | None,
    calibration_enabled: bool = True,
) -> dict[str, float]:
    """Reduce finite real shards; the MP20 eval forward has no collectives."""
    if getattr(loader.dataset, "repeat", False) or loader.drop_last:
        raise ValueError("full_unique requires repeat=False and drop_last=False")
    objective = module.module if isinstance(module, DistributedDataParallel) else module
    was_training = module.training
    module.eval()
    try:
        # Bypass DDP forward broadcasts on uneven shards, including empty ranks.
        if isinstance(module, DistributedDataParallel) and module.broadcast_buffers:
            for buffer in objective.buffers():
                dist.broadcast(buffer, src=0, group=module.process_group)
        generator = torch.Generator(device=device).manual_seed(seed)
        totals = zero_metric_tensors(VALIDATION_METRIC_NAMES, device=device)
        counts = zero_metric_tensors(("samples", "active_samples", "physics_gate"), device=device)
        gated_metrics = {
            "physics_loss", "physics_overlap_excess", "physics_bond_radius_excess",
            "physics_bond_valence_excess", "physics_coordination_excess",
        }
        active_metrics = {"score_loss", "rollout_score_loss"}
        calibration = {
            f"{metric}:{_sigma_label(sigma)}": []
            for sigma in CALIBRATION_SIGMAS for metric in ("cosine", "norm_ratio")
        }

        def forward(batch, **kwargs):
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16,
                enabled=precision == "bf16" and device.type == "cuda",
            ):
                return objective(batch, input_perturbation_enabled=False, **kwargs)

        local_error = None
        try:
            for batch_index, batch in enumerate(loader):
                batch = batch.to(device, non_blocking=device.type == "cuda")
                sigma, noise, probe = _fixed_validation_inputs(
                    loader, batch_index, batch, objective, seed=seed, sigma_strata=sigma_strata,
                    fixed_noise_seed=fixed_noise_seed,
                )
                output = forward(batch, generator=generator, sigma_by_structure=sigma,
                                 standard_noise=noise, rollout_enabled=rollout_enabled)
                active_count = sum(layout.num_parameters > 0 for layout in batch.layouts)
                gate = (
                    objective.physics_guidance.sigma_gate(output.sigma_by_structure).sum()
                    if objective.physics_guidance is not None else batch.clean_u.new_zeros(())
                )
                counts["samples"] += batch.batch_size
                counts["active_samples"] += active_count
                counts["physics_gate"] += gate
                for name, value in objective_metric_tensors(output, VALIDATION_METRIC_NAMES).items():
                    weight = (gate.clamp_min(1.0) if name in gated_metrics
                              else active_count if name in active_metrics else batch.batch_size)
                    totals[name] += value.detach().float() * weight

                # Per-structure fixed probes are invariant to batch/rank packing.
                for sigma_value in CALIBRATION_SIGMAS if calibration_enabled else ():
                    output = forward(
                        batch, sigma_by_structure=batch.clean_u.new_full(
                            (batch.batch_size,), sigma_value,
                        ), standard_noise=probe, rollout_enabled=False,
                    )
                    label = _sigma_label(sigma_value)
                    calibration[f"cosine:{label}"].append(output.score_dual_cosine_by_structure)
                    calibration[f"norm_ratio:{label}"].append(
                        output.score_dual_norm_ratio_by_structure,
                    )
        except Exception as exc:
            local_error = f"rank={rank} {type(exc).__name__}: {exc}"
        raise_if_any_rank_failed(local_error, operation="full_unique validation")

        # One fixed scalar collective, with zero numerators/counts on empty ranks.
        names = (*totals, *counts)
        packed = torch.stack(tuple(totals.values()) + tuple(counts.values()))
        if dist.is_initialized():
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        reduced = dict(zip(names, packed.unbind()))
        if reduced["samples"] == 0:
            raise RuntimeError("no admitted validation records in full_unique traversal")
        for name in totals:
            denominator = (reduced["physics_gate"] if name in gated_metrics
                           else reduced["active_samples"] if name in active_metrics
                           else reduced["samples"])
            totals[name] = reduced[name] / denominator.clamp_min(1.0)
        totals["loss"] = (
            totals["score_loss"] + objective.rollout_score_weight * totals["rollout_score_loss"]
            + objective.rollout_x0_weight * totals["rollout_x0_mse"]
            + objective.physics_auxiliary_weight * totals["physics_loss"]
        )
        totals["denoising_relative_improvement"] = (
            totals["denoising_rmsd_improvement"] / totals["identity_rmsd"].clamp_min(1.0e-12)
        )
        gathered = gather_named_1d_across_ranks({
            name: torch.cat(values) if values else torch.empty(0, device=device)
            for name, values in calibration.items()
        }) if calibration_enabled else {}
        if not calibration_enabled:
            totals.update({name: torch.tensor(float("nan"), device=device)
                           for name in calibration_metric_names()})
        for name, values in gathered.items():
            metric, label = name.split(":")
            for statistic, value in _calibration_statistics(values).items():
                middle = "" if statistic == "mean" else f"_{statistic}"
                totals[f"calibration_{metric}{middle}_sigma_{label}"] = value
        return {name: float(value.cpu()) for name, value in totals.items()}
    finally:
        module.train(was_training)


__all__ = [
    "CALIBRATION_SIGMAS",
    "CALIBRATION_STATISTICS",
    "TRAIN_METRIC_NAMES",
    "VALIDATION_METRIC_NAMES",
    "append_metrics",
    "calibration_metric_names",
    "empty_validation_metrics",
    "objective_metric_tensors",
    "training_history_row",
    "validate_objective",
    "write_training_run_manifest",
    "zero_metric_tensors",
]
