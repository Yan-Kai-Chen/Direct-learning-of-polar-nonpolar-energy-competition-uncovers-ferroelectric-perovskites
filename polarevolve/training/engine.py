"""Distributed MP20 hard-conditioned score training."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import time

import torch
from torch.distributed import is_initialized
from torch.nn.parallel import DistributedDataParallel

from polarevolve.data.cache import ASUCacheManifest, load_cache_manifest
from polarevolve.crystal.symmetry import WyckoffDatabase
from polarevolve.data.contracts import ExternalRoots
from polarevolve.diffusion.schedule import SigmaSchedule
from polarevolve.models.config import ScoreModelConfig
from polarevolve.models.score import HardConditionScoreNetwork
from polarevolve.guidance.physics import PhysicsGuidanceConfig
from polarevolve.runtime.distributed import (
    distributed_device,
    is_primary,
    mean_named_scalars_across_ranks,
    mean_parameter_gradients_across_ranks,
    maybe_no_sync,
    run_on_primary_and_broadcast,
    seed_everything,
    shutdown_distributed,
)
from polarevolve.training.checkpoint import (
    ValidationBest,
    atomic_checkpoint,
    capture_rank_rng,
    model_state_sha256,
    persist_training_checkpoints,
    prepare_training_startup,
    training_payload,
)
from polarevolve.tasks.mp20 import MP20ScoreObjective
from polarevolve.training.config import TrainConfig, TrainingRunConfig, cosine_lr_factor
from polarevolve.training.physics import admit_physics_training
from polarevolve.training.data import (
    build_training_loaders,
    build_loader,
    FiniteEpochLoader,
    ResidentValidationLoader,
    plan_training_data,
    prepare_training_manifest,
    resolve_training_budget,
    stratified_sigma_values,
)
from polarevolve.runtime.filesystem import atomic_json
from polarevolve.training.metrics import (
    TRAIN_METRIC_NAMES,
    append_metrics,
    empty_validation_metrics,
    objective_metric_tensors,
    parameter_update_metrics,
    reconcile_metrics_for_resume,
    training_history_row,
    validate_objective,
    write_training_run_manifest,
    zero_metric_tensors,
)


def run_training(
    *,
    roots: ExternalRoots,
    run: TrainingRunConfig,
    train_config: TrainConfig,
    model_config: ScoreModelConfig,
    sigma_schedule: SigmaSchedule,
) -> None:
    owns_process_group = not is_initialized()
    device, rank, world_size, local_rank = distributed_device()
    seed_everything(train_config.seed, rank)
    mem_debug = train_config.mem_debug and device.type == "cuda"
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    cache_root = Path(roots.data_root) / run.cache_relative
    group_assets = Path(roots.asset_root) / run.group_assets_relative
    wyckoff_assets = Path(roots.asset_root) / run.wyckoff_assets_relative
    output_dir = Path(roots.output_root) / "mp20" / run.run_id
    resume_path = run.resume.resolve() if run.resume else None
    initialize_path = run.initialize_from.resolve() if run.initialize_from else None
    manifest = run_on_primary_and_broadcast(
        lambda: prepare_training_manifest(
            cache_root=cache_root,
            wyckoff_assets=wyckoff_assets,
            output_dir=output_dir,
            resume_path=resume_path,
        ),
        rank=rank,
        operation="training run preparation",
    )
    if not isinstance(manifest, ASUCacheManifest):
        raise RuntimeError("rank 0 did not broadcast a valid cache manifest")
    data_plan = run_on_primary_and_broadcast(
        lambda: plan_training_data(manifest, config=train_config, world_size=world_size),
        rank=rank,
        operation="training data plan",
    )
    data_plan.train.require_all_ranks_nonempty()
    if train_config.validation_mode == "legacy_batches":
        data_plan.val.require_all_ranks_nonempty()
    train_config, training_budget = resolve_training_budget(
        train_config,
        selected_train_records=data_plan.train.selected_records,
        world_size=world_size,
    )
    if train_config.sigma_strata and (
        min(train_config.sigma_strata) < sigma_schedule.sigma_min
        or max(train_config.sigma_strata) > sigma_schedule.sigma_max_cap
    ):
        raise ValueError("sigma strata must lie inside the configured sigma schedule")
    train_config, calibration_sha = run_on_primary_and_broadcast(
        lambda: admit_physics_training(train_config, initialize_path, manifest.manifest_sha256),
        rank=rank,
        operation="physics objective admission",
    )
    physics_config = PhysicsGuidanceConfig(
        prior_path=train_config.physics_prior,
        sigma_full_strength=train_config.physics_sigma_full_strength,
        sigma_cutoff=train_config.physics_sigma_cutoff,
    )
    model = HardConditionScoreNetwork(**model_config.to_dict()).to(device)
    if model.backbone_version == 2 and train_config.physics_prior is not None and resume_path is None and initialize_path is None:
        import json
        calibration = run_on_primary_and_broadcast(
            lambda: json.loads(Path(train_config.physics_calibration).read_text(encoding="utf-8")),
            rank=rank, operation="scratch physics calibration identity",
        )
        scratch_matches = run_on_primary_and_broadcast(
            lambda: (calibration.get("model_state_sha256") == model_state_sha256(model.state_dict())
                     and calibration.get("model_config") == model_config.to_dict()
                     and calibration.get("sigma_schedule") == asdict(sigma_schedule)),
            rank=rank, operation="scratch parameter identity",
        )
        if not scratch_matches:
            raise ValueError("physics calibration does not bind the exact randomly initialized backbone")
    objective = MP20ScoreObjective(
        model=model,
        schedule=sigma_schedule,
        terminal_mixing_tolerance=run.terminal_mixing_tolerance,
        sigma_sampling_mode=train_config.sigma_sampling_mode,
        diffusion_metric=run.diffusion_metric,
        input_perturbation_probability=train_config.input_perturbation_probability,
        input_perturbation_scale=train_config.input_perturbation_scale,
        rollout_steps=train_config.rollout_steps,
        rollout_score_weight=train_config.rollout_score_weight,
        rollout_x0_weight=train_config.rollout_x0_weight,
        rollout_maximum_step_rms_angstrom=train_config.rollout_maximum_step_rms_angstrom,
        physics_auxiliary_weight=train_config.physics_auxiliary_weight,
        physics_config=physics_config,
    ).to(device)
    physics_identity = {"enabled": objective.physics_guidance is not None}
    if objective.physics_guidance is not None:
        physics_identity.update(
            contract=objective.physics_guidance.contract,
            config=asdict(physics_config),
            provenance=objective.physics_guidance.provenance,
            auxiliary_weight=train_config.physics_auxiliary_weight,
            warmup_epochs=train_config.physics_warmup_epochs,
            calibration_sha256=calibration_sha,
        )
        prior = objective.physics_guidance.chemistry_prior
        if (
            prior is not None
            and prior.identity["cache_manifest_sha256"] != manifest.manifest_sha256
        ):
            raise ValueError("physics prior and training cache identity differ")
    optimizer = torch.optim.AdamW(
        objective.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: cosine_lr_factor(step, train_config)
    )
    resolved_startup = prepare_training_startup(
        resume_path=resume_path,
        initialize_path=initialize_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        model_config=model_config,
        sigma_schedule=sigma_schedule,
        cache_manifest_sha256=manifest.manifest_sha256,
        diffusion_metric=run.diffusion_metric,
        rank=rank,
        physics_identity=physics_identity,
        allow_cache_mismatch_on_initialize=run.allow_cache_transfer,
        validation_mode=train_config.validation_mode,
        epoch_mode=train_config.epoch_mode,
        resume_train_config=train_config,
        world_size=world_size,
    )
    start_step = resolved_startup.step
    validation_best = resolved_startup.validation_best
    if resume_path is not None:
        removed_rows = run_on_primary_and_broadcast(
            lambda: reconcile_metrics_for_resume(output_dir / "metrics.csv", start_step),
            rank=rank,
            operation="resume metrics reconciliation",
        )
        if train_config.replay_cache_relative is not None:
            run_on_primary_and_broadcast(
                lambda: reconcile_metrics_for_resume(output_dir / "replay_metrics.csv", start_step),
                rank=rank, operation="replay metrics reconciliation",
            )
        if is_primary(rank):
            print(
                f"[resume] checkpoint_step={start_step} "
                f"discarded_post_checkpoint_metric_rows={removed_rows}",
                flush=True,
            )
    module: torch.nn.Module = objective
    if world_size > 1:
        module = DistributedDataParallel(
            objective, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False
        )
    train_loader, val_loader = build_training_loaders(
        manifest=manifest,
        group_assets=group_assets,
        wyckoff_assets=wyckoff_assets,
        config=train_config,
        plan=data_plan,
        rank=rank,
        world_size=world_size,
    )
    if is_primary(rank):
        atomic_json(output_dir / "training_status.json", {"status": "preparing_data",
                    "timestamp_unix": time.time(), "global_step": start_step,
                    "world_size": world_size, "max_steps": train_config.max_steps})
    replay_loader = None
    if train_config.replay_cache_relative is not None:
        replay_manifest = run_on_primary_and_broadcast(
            lambda: load_cache_manifest(Path(roots.data_root) / train_config.replay_cache_relative,
                                       wyckoff_database=WyckoffDatabase(wyckoff_assets)),
            rank=rank, operation="balanced replay cache admission",
        )
        if replay_manifest.manifest_sha256 != train_config.replay_manifest_sha256:
            raise ValueError("balanced replay cache identity changed")
        replay_plan = plan_training_data(replay_manifest, config=train_config, world_size=world_size)
        replay_loader = build_loader(
            manifest=replay_manifest, group_assets=group_assets, wyckoff_assets=wyckoff_assets,
            split="train", rank=0, world_size=1, batch_size=train_config.batch_size,
            workers=train_config.num_workers, seed=train_config.seed, repeat=False, shuffle=False,
            global_limit=None, rank_record_count=replay_plan.train.available_records,
            require_trainable_parameters=True, drop_last=False,
        )
    if train_config.epoch_mode == "finite_unique":
        train_loader = FiniteEpochLoader(
            train_loader, config=train_config, records=training_budget.selected_train_records,
            rank=rank, world_size=world_size, device=device, start_step=start_step, replay_loader=replay_loader,
        )
    if train_config.resident_epoch and device.type == "cuda" and train_config.validation_mode == "full_unique":
        val_loader = ResidentValidationLoader(val_loader, device)
    ema = {name: value.detach().clone() for name, value in model.state_dict().items()}
    ema_best = float("inf")
    if resolved_startup.ema_state is not None:
        state = resolved_startup.ema_state
        if set(state["model"]) != set(ema):
            raise ValueError("EMA checkpoint keys differ from the model")
        ema = {name: state["model"][name].to(value) for name, value in ema.items()}
        ema_best = state["best_loss"]
    run_on_primary_and_broadcast(
        lambda: write_training_run_manifest(
            output_dir=output_dir,
            cache_root=cache_root,
            group_assets=group_assets,
            wyckoff_assets=wyckoff_assets,
            manifest=manifest,
            physics_identity=physics_identity,
            physics_config=physics_config,
            objective=objective,
            model=model,
            model_config=model_config,
            train_config=train_config,
            sigma_schedule=sigma_schedule,
            startup=resolved_startup,
            data_plan=data_plan,
            training_budget=training_budget,
            world_size=world_size,
        ),
        rank=rank,
        operation="run manifest write",
    )
    if is_primary(rank):
        print(
            "[train] data_view="
            f"{'active_only' if train_config.active_only else 'all_records'} "
            f"selected={training_budget.selected_train_records} "
            f"steps_per_epoch={training_budget.steps_per_data_epoch} "
            f"optimizer_steps={training_budget.optimizer_steps} "
            f"equivalent_epochs={training_budget.equivalent_data_epochs:.3f}",
            flush=True,
        )
    train_iterator = iter(train_loader)
    generator = torch.Generator(device=device)
    module.train()
    optimizer.zero_grad(set_to_none=True)
    last_checkpoint = time.monotonic()
    completed_samples = (start_step // training_budget.steps_per_data_epoch * training_budget.selected_train_records
                         + start_step % training_budget.steps_per_data_epoch * training_budget.primary_batch_size)
    for step in range(start_step + 1, train_config.max_steps + 1):
        step_started = time.monotonic()
        step_totals = zero_metric_tensors(TRAIN_METRIC_NAMES, device=device)
        replay_enabled = train_config.replay_cache_relative is not None
        if replay_enabled:
            step_totals.update(fe_loss=torch.zeros((), device=device), replay_loss=torch.zeros((), device=device))
        rollout_enabled = (
            train_config.rollout_interval > 0 and step % train_config.rollout_interval == 0
        )
        micro_steps = train_config.gradient_accumulation * (2 if replay_enabled else 1)
        for micro_step in range(micro_steps):
            generator.manual_seed(
                train_config.seed
                + 70_001 * rank
                + step * micro_steps
                + micro_step
            )
            item = next(train_iterator)
            if train_config.epoch_mode == "finite_unique":
                batch, loss_weight = item
            else:
                batch, loss_weight = item, 1.0 / train_config.gradient_accumulation
            batch = batch.to(device, non_blocking=device.type == "cuda")
            if mem_debug and is_primary(rank):
                print(
                    f"[mem] step={step} micro={micro_step} "
                    f"structures={batch.batch_size} atoms={batch.num_atoms} "
                    f"params={batch.num_parameters} edges={batch.edge_index.shape[1]} "
                    f"alloc={torch.cuda.memory_allocated() / 2**30:.2f}GiB "
                    f"peak={torch.cuda.max_memory_allocated() / 2**30:.2f}GiB "
                    f"reserved={torch.cuda.memory_reserved() / 2**30:.2f}GiB",
                    flush=True,
                )
                torch.cuda.reset_peak_memory_stats()
            no_sync = train_config.epoch_mode == "finite_unique" or micro_step + 1 < micro_steps
            with maybe_no_sync(module, no_sync):
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=train_config.precision == "bf16" and device.type == "cuda",
                ):
                    sigma_by_structure = stratified_sigma_values(
                        train_config.sigma_strata,
                        batch_size=batch.batch_size,
                        step=step,
                        micro_step=micro_step,
                        rank=rank,
                        world_size=world_size,
                        gradient_accumulation=micro_steps,
                        device=device,
                        dtype=batch.clean_u.dtype,
                    )
                    standard_noise = None
                    if train_config.fixed_training_noise_seed is not None:
                        fixed_generator = torch.Generator(device=device).manual_seed(
                            train_config.fixed_training_noise_seed + 10_007 * rank
                        )
                        standard_noise = torch.randn(
                            batch.clean_u.shape,
                            device=device,
                            dtype=batch.clean_u.dtype,
                            generator=fixed_generator,
                        )
                    output = module(
                        batch,
                        generator=generator,
                        sigma_by_structure=sigma_by_structure,
                        standard_noise=standard_noise,
                        input_perturbation_enabled=True,
                        rollout_enabled=rollout_enabled,
                        physics_weight_scale=min(
                            1.0,
                            step
                            / max(
                                train_config.physics_warmup_epochs
                                * training_budget.steps_per_data_epoch,
                                1,
                            ),
                        ),
                    )
                    loss = output.loss * loss_weight
                loss.backward()
            for name, value in objective_metric_tensors(output).items():
                step_totals[name] += value.detach().float() * loss_weight
            if replay_enabled:
                step_totals["replay_loss" if micro_step % 2 else "fe_loss"] += output.loss.detach().float() * loss_weight * 2
        if train_config.epoch_mode == "finite_unique" and world_size > 1:
            mean_parameter_gradients_across_ranks(objective.parameters())
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            objective.parameters(), train_config.gradient_clip_norm, error_if_nonfinite=True
        )
        reduced = mean_named_scalars_across_ranks(
            {
                **step_totals,
                "gradient_clipped": (gradient_norm > train_config.gradient_clip_norm).float(),
            }
        )
        gradient_clipped = reduced.pop("gradient_clipped")
        monitor_update = model.backbone_version == 2 and is_primary(rank) and (step == 1 or step % train_config.log_every == 0)
        previous_parameters = {name: p.detach().clone() for name, p in model.named_parameters()} if monitor_update else None
        optimizer.step()
        scheduler.step()
        samples = training_budget.primary_batch_size
        if train_config.epoch_mode == "finite_unique":
            remaining = training_budget.selected_train_records - ((step - 1) % training_budget.steps_per_data_epoch) * samples
            samples = min(samples, remaining)
        completed_samples += samples
        if train_config.ema_half_life_epochs:
            decay = 0.5 ** (samples / (train_config.ema_half_life_epochs * training_budget.selected_train_records))
            with torch.no_grad():
                for name, value in model.state_dict().items():
                    if value.is_floating_point():
                        ema[name].lerp_(value, 1 - decay)
                    else:
                        ema[name].copy_(value)
        if monitor_update:
            append_metrics(output_dir / "parameter_updates.csv", {
                "step": step, **parameter_update_metrics(model, optimizer, previous_parameters),
            })
        optimizer.zero_grad(set_to_none=True)
        if is_primary(rank) and (step == 1 or step % train_config.log_every == 0):
            print(
                f"[train] step={step}/{train_config.max_steps} "
                f"loss={float(reduced['loss']):.6g} "
                f"score={float(reduced['score_loss']):.6g} "
                f"one_step_rmsd={float(reduced['one_step_rmsd']):.6g}A "
                f"identity_rmsd={float(reduced['identity_rmsd']):.6g}A "
                f"denoise_gain={float(reduced['denoising_relative_improvement']):.3%} "
                f"rollout_score={float(reduced['rollout_score_loss']):.6g} "
                f"rollout_x0_mse={float(reduced['rollout_x0_mse']):.6g}A2 "
                f"physics={float(reduced['physics_loss']):.6g} "
                f"grad_norm={float(gradient_norm):.6g} "
                f"clipped={bool(gradient_clipped.item())} "
                f"lr={optimizer.param_groups[0]['lr']:.6g}",
                flush=True,
            )
        val_metrics = empty_validation_metrics()
        validated = step % train_config.validate_every == 0 or step == train_config.max_steps
        calibration_enabled = (
            not train_config.calibration_interval_epochs or step == train_config.max_steps
            or step % (train_config.calibration_interval_epochs * training_budget.steps_per_data_epoch) == 0
        )
        raw_validation_seconds = ema_validation_seconds = 0.0
        if validated:
            validation_started = time.monotonic()
            val_metrics = validate_objective(
                module,
                val_loader,
                batches=train_config.validation_batches,
                device=device,
                precision=train_config.precision,
                seed=train_config.seed,
                rank=rank,
                rollout_enabled=train_config.rollout_interval > 0,
                sigma_strata=train_config.sigma_strata,
                fixed_noise_seed=train_config.fixed_training_noise_seed,
                validation_mode=train_config.validation_mode,
                calibration_enabled=calibration_enabled,
            )
            raw_validation_seconds = time.monotonic() - validation_started
            if is_primary(rank):
                print(
                    f"[val] step={step} loss={val_metrics['loss']:.6g} "
                    f"score={val_metrics['score_loss']:.6g} "
                    f"one_step_rmsd={val_metrics['one_step_rmsd']:.6g}A "
                    f"identity_rmsd={val_metrics['identity_rmsd']:.6g}A "
                    f"denoise_gain={val_metrics['denoising_relative_improvement']:.3%} "
                    f"rollout_score={val_metrics['rollout_score_loss']:.6g} "
                    f"rollout_x0_mse={val_metrics['rollout_x0_mse']:.6g}A2",
                    f"physics={val_metrics['physics_loss']:.6g}",
                    flush=True,
                )
        ema_metrics = None
        if validated and train_config.ema_half_life_epochs:
            validation_started = time.monotonic()
            raw = {name: value.detach().clone() for name, value in model.state_dict().items()}
            try:
                model.load_state_dict(ema, strict=True)
                ema_metrics = validate_objective(
                    module, val_loader, batches=train_config.validation_batches,
                    device=device, precision=train_config.precision, seed=train_config.seed,
                    rank=rank, sigma_strata=train_config.sigma_strata,
                    validation_mode=train_config.validation_mode,
                    rollout_enabled=train_config.rollout_interval > 0,
                    fixed_noise_seed=train_config.fixed_training_noise_seed,
                    calibration_enabled=calibration_enabled,
                )
            finally:
                model.load_state_dict(raw, strict=True)
            ema_validation_seconds = time.monotonic() - validation_started
        if validated and is_primary(rank):
            print(f"[val timing] step={step} raw_seconds={raw_validation_seconds:.3f} "
                  f"ema_seconds={ema_validation_seconds:.3f} calibration_enabled={calibration_enabled} "
                  f"batch_per_rank={train_config.val_batch_size}", flush=True)

        save_due = time.monotonic() - last_checkpoint >= train_config.checkpoint_minutes * 60
        checkpoint_due = run_on_primary_and_broadcast(
            lambda: validated or step % train_config.save_every == 0 or save_due,
            rank=rank, operation="checkpoint schedule",
        )
        rank_rng_states = None
        if checkpoint_due and train_config.epoch_mode == "finite_unique":
            rank_rng_states = [capture_rank_rng()]
            if world_size > 1:
                import torch.distributed as dist
                state = rank_rng_states[0]
                rank_rng_states = [None] * world_size
                dist.all_gather_object(rank_rng_states, state)

        def persist_step() -> ValidationBest:
            nonlocal ema_best
            if validated:
                append_metrics(output_dir / "validation_timing.csv", {
                    "step": step, "raw_seconds": raw_validation_seconds,
                    "ema_seconds": ema_validation_seconds, "calibration_enabled": calibration_enabled,
                    "validation_batch_per_rank": train_config.val_batch_size,
                    "validation_records": data_plan.val.selected_records,
                })
            atomic_json(output_dir / "training_status.json", {
                "schema_version": "feb1_training_status_v1", "global_step": step,
                "epoch_mode": train_config.epoch_mode,
                "samples_seen": completed_samples,
                "replay_samples_seen": completed_samples if replay_enabled else 0,
                "total_samples_seen": completed_samples * (2 if replay_enabled else 1),
                "data_epochs_completed": completed_samples / training_budget.selected_train_records,
                "world_size": world_size, "effective_batch_size": training_budget.effective_batch_size,
                "step_seconds": time.monotonic() - step_started,
                "timestamp_unix": time.time(), "max_steps": train_config.max_steps,
                "peak_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                "status": "completed" if step == train_config.max_steps else "running",
            })
            append_metrics(
                output_dir / "metrics.csv",
                training_history_row(
                    step=step,
                    train_metrics=reduced,
                    validation_metrics=val_metrics,
                    learning_rate=optimizer.param_groups[0]["lr"],
                    gradient_norm=float(gradient_norm),
                    gradient_clipped=float(gradient_clipped),
                ),
            )
            if replay_enabled:
                append_metrics(output_dir / "replay_metrics.csv", {
                    "step": step, "fe_loss": reduced["fe_loss"], "replay_loss": reduced["replay_loss"],
                    "fe_samples_seen": completed_samples, "replay_samples_seen": completed_samples,
                })
            if checkpoint_due:
                payload = training_payload(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=step,
                    validation_best=validation_best,
                    model_config=model_config,
                    train_config=train_config,
                    sigma_schedule=sigma_schedule,
                    cache_manifest_sha256=manifest.manifest_sha256,
                    diffusion_metric=run.diffusion_metric,
                    initialization=resolved_startup.checkpoint_initialization,
                    physics_identity=physics_identity,
                )
                payload["data_cursor"] = {"samples_seen": completed_samples, "world_size": world_size,
                                          "steps_per_epoch": training_budget.steps_per_data_epoch,
                                          "replay_samples_seen": completed_samples if replay_enabled else 0,
                                          "replay_manifest_sha256": train_config.replay_manifest_sha256}
                if rank_rng_states is not None:
                    payload["rank_rng_states"] = rank_rng_states
                if train_config.ema_half_life_epochs:
                    ema_best = min(ema_best, ema_metrics["loss"] if ema_metrics else float("inf"))
                    payload["ema_state"] = {"model": ema, "best_loss": ema_best}
                    if ema_metrics is not None:
                        append_metrics(output_dir / "ema_metrics.csv", {"step": step, **ema_metrics})
                        ema_payload = payload | {"model": ema, "weight_view": "ema"}
                        atomic_checkpoint(output_dir / "ema_last.pt", ema_payload)
                        if ema_metrics["loss"] == ema_best:
                            atomic_checkpoint(output_dir / "ema_best_loss.pt", ema_payload)
                return persist_training_checkpoints(
                    output_dir=output_dir,
                    payload=payload,
                    step=step,
                    validated=validated,
                    validation_loss=val_metrics["loss"],
                    validation_one_step_rmsd=val_metrics["one_step_rmsd"],
                    previous_best=validation_best,
                    retain_snapshot=step % (train_config.snapshot_interval_epochs * training_budget.steps_per_data_epoch) == 0 or step == train_config.max_steps,
                )
            return validation_best

        validation_best = run_on_primary_and_broadcast(
            persist_step,
            rank=rank,
            operation=f"step {step} metrics/checkpoint write",
        )
        if checkpoint_due:
            last_checkpoint = time.monotonic()
    if is_primary(rank):
        print(f"[OK] training complete: {output_dir}", flush=True)
    if owns_process_group:
        shutdown_distributed()


__all__ = ["run_training"]
