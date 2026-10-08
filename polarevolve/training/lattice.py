"""Independent C2L lattice training and strict checkpoint contract."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Mapping, Sequence

import torch

from polarevolve.crystal.lattice import HallMetricFrame
from polarevolve.crystal.symmetry import sha256_file
from polarevolve.data.cache import ASUCacheManifest
from polarevolve.data.packing import (
    LatticeNormalizer,
    LatticeTargetBatch,
    pack_lattice_records,
)
from polarevolve.data.provenance import source_fingerprint
from polarevolve.models.config import LatticeModelConfig
from polarevolve.models.lattice import (
    HardConditionLatticeNetwork,
    LatticeMixtureOutput,
    lattice_mixture_log_probability,
)
from polarevolve.runtime.filesystem import atomic_json
from polarevolve.training.checkpoint import atomic_checkpoint
from polarevolve.training.config import LatticeTrainConfig
from polarevolve.training.data import build_loader
from polarevolve.training.metrics import append_metrics

LATTICE_CHECKPOINT_SCHEMA = "gt_sge_c2l_lattice_checkpoint_v1"
LATTICE_TARGET_CONTRACT = "hall_invariant_log_metric_v1"


def lattice_mixture_nll(
    output: LatticeMixtureOutput,
    target: torch.Tensor,
    active_mask: torch.Tensor,
) -> torch.Tensor:
    """Diagonal Gaussian-mixture NLL over Hall-active coordinates only."""

    return -lattice_mixture_log_probability(output, target, active_mask).mean()


def oracle_component_rmse(
    output: LatticeMixtureOutput,
    target: LatticeTargetBatch,
    normalizer: LatticeNormalizer,
) -> torch.Tensor:
    """Diagnostic target-aware best-component error in physical C2L coordinates."""

    means = normalizer.denormalize(output.means)
    squared = (
        (means - target.coordinates[:, None, :]).square()
        * target.active_mask[:, None, :]
    ).sum(dim=-1)
    count = target.active_mask.sum(dim=-1).clamp_min(1)
    return (squared / count[:, None]).min(dim=1).values.sqrt().mean()


def _losses(
    model: HardConditionLatticeNetwork,
    batch: LatticeTargetBatch,
    normalizer: LatticeNormalizer,
) -> tuple[torch.Tensor, torch.Tensor]:
    output = model(batch.condition)
    normalized = normalizer.normalize(batch.coordinates)
    return (
        lattice_mixture_nll(output, normalized, batch.active_mask),
        oracle_component_rmse(output, batch, normalizer),
    )


def _payload(
    *,
    model: HardConditionLatticeNetwork,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    best_train_nll: float,
    model_config: LatticeModelConfig,
    train_config: LatticeTrainConfig,
    normalizer: LatticeNormalizer,
    cache_manifest_sha256: str,
    selection_split: str = "train",
    best_selection_nll: float | None = None,
) -> dict:
    selection_nll = best_train_nll if best_selection_nll is None else best_selection_nll
    return {
        "schema_version": LATTICE_CHECKPOINT_SCHEMA,
        "target_contract": LATTICE_TARGET_CONTRACT,
        "global_step": int(step),
        "best_train_nll": float(best_train_nll),
        "selection_split": selection_split,
        "best_selection_nll": float(selection_nll),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "model_config": model_config.to_dict(),
        "train_config": train_config.to_dict(),
        "normalizer": normalizer.to_dict(),
        "cache_manifest_sha256": cache_manifest_sha256,
        "coordinate_checkpoint": None,
    }


def load_lattice_payload(
    path: str | Path,
    *,
    cache_manifest_sha256: str,
) -> tuple[dict, str]:
    """Load and validate a C2L checkpoint once at the process boundary."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"C2L lattice checkpoint is missing: {resolved}")
    payload = torch.load(resolved, map_location="cpu")
    if not isinstance(payload, dict) or payload.get("schema_version") != LATTICE_CHECKPOINT_SCHEMA:
        raise ValueError("unsupported C2L lattice checkpoint schema")
    required = {
        "target_contract",
        "global_step",
        "model",
        "model_config",
        "normalizer",
        "cache_manifest_sha256",
        "coordinate_checkpoint",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"C2L lattice checkpoint is missing fields: {missing}")
    if payload.get("target_contract") != LATTICE_TARGET_CONTRACT:
        raise ValueError("lattice checkpoint target contract differs from C2L")
    if payload.get("cache_manifest_sha256") != cache_manifest_sha256:
        raise ValueError("lattice checkpoint cache manifest differs from the active cache")
    if payload.get("coordinate_checkpoint") is not None:
        raise ValueError("independent lattice checkpoint must not embed a coordinate checkpoint")
    return payload, sha256_file(resolved)


def lattice_model_from_payload(
    payload: Mapping[str, object],
) -> tuple[HardConditionLatticeNetwork, LatticeNormalizer]:
    """Rebuild the strict lattice inference objects from a validated payload."""

    config = LatticeModelConfig(**payload["model_config"])
    model = HardConditionLatticeNetwork(config)
    model.load_state_dict(payload["model"], strict=True)
    return model, LatticeNormalizer.from_dict(payload["normalizer"])


def load_lattice_checkpoint(
    path: str | Path,
    *,
    cache_manifest_sha256: str,
) -> tuple[HardConditionLatticeNetwork, LatticeNormalizer, dict, str]:
    payload, digest = load_lattice_payload(
        path, cache_manifest_sha256=cache_manifest_sha256
    )
    model, normalizer = lattice_model_from_payload(payload)
    return model, normalizer, payload, digest


def materialize_lattice_split(
    *,
    manifest: ASUCacheManifest,
    group_asset_root: Path,
    wyckoff_asset_root: Path,
    frames: Mapping[int, HallMetricFrame],
    split: str,
    batch_size: int,
    workers: int,
    seed: int,
) -> tuple[LatticeTargetBatch, ...]:
    expected = manifest.split_count(split)
    loader = build_loader(
        manifest=manifest,
        group_assets=group_asset_root,
        wyckoff_assets=wyckoff_asset_root,
        split=split,
        rank=0,
        world_size=1,
        batch_size=batch_size,
        workers=workers,
        seed=seed,
        repeat=False,
        shuffle=False,
        global_limit=None,
        rank_record_count=expected,
        collate_fn=partial(pack_lattice_records, frames=frames),
        drop_last=False,
        pin_memory=False,
    )
    # Retained worker tensors must own their storage; otherwise every batch keeps
    # multiprocessing file descriptors alive for the lifetime of the run.
    batches = tuple(batch.to("cpu", copy=True) for batch in loader)
    actual = sum(batch.condition.batch_size for batch in batches)
    if actual != expected:
        raise RuntimeError(
            f"C2L split {split} materialized {actual} records, expected {expected}"
        )
    return batches


@torch.no_grad()
def evaluate_lattice_batches(
    model: HardConditionLatticeNetwork,
    batches: Sequence[LatticeTargetBatch],
    normalizer: LatticeNormalizer,
    device: torch.device,
) -> dict[str, float | int]:
    model.eval()
    total_nll = 0.0
    total_rmse = 0.0
    records = 0
    for cpu_batch in batches:
        batch = cpu_batch.to(device)
        nll, rmse = _losses(model, batch, normalizer)
        count = batch.condition.batch_size
        total_nll += float(nll) * count
        total_rmse += float(rmse) * count
        records += count
    if records == 0:
        raise ValueError("lattice evaluation requires non-empty batches")
    return {
        "records": records,
        "nll": total_nll / records,
        "oracle_component_rmse": total_rmse / records,
    }


def run_lattice_distribution_training(
    *,
    manifest: ASUCacheManifest,
    group_asset_root: Path,
    wyckoff_asset_root: Path,
    frames: Mapping[int, HallMetricFrame],
    output_dir: Path,
    model_config: LatticeModelConfig,
    train_config: LatticeTrainConfig,
    device: torch.device,
) -> dict:
    """Fit the C2L model on the complete train split and select on validation."""

    if output_dir.exists():
        raise FileExistsError(f"C2L-2 training output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    torch.manual_seed(train_config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(train_config.seed)
    train_batches = materialize_lattice_split(
        manifest=manifest,
        group_asset_root=group_asset_root,
        wyckoff_asset_root=wyckoff_asset_root,
        frames=frames,
        split="train",
        batch_size=train_config.batch_size,
        workers=train_config.num_workers,
        seed=train_config.seed,
    )
    val_batches = materialize_lattice_split(
        manifest=manifest,
        group_asset_root=group_asset_root,
        wyckoff_asset_root=wyckoff_asset_root,
        frames=frames,
        split="val",
        batch_size=train_config.val_batch_size,
        workers=train_config.num_workers,
        seed=train_config.seed,
    )
    normalizer = LatticeNormalizer.fit_batches(train_batches).to(device)
    model = HardConditionLatticeNetwork(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    maximum_steps = train_config.data_epochs * len(train_batches)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=maximum_steps,
        eta_min=0.05 * train_config.learning_rate,
    )
    initial_val = evaluate_lattice_batches(model, val_batches, normalizer, device)
    best_val_nll = float("inf")
    best_train_nll = float("inf")
    best_epoch = 0
    stale_epochs = 0
    global_step = 0
    generator = torch.Generator().manual_seed(train_config.seed)
    for epoch in range(1, train_config.data_epochs + 1):
        model.train()
        train_nll_sum = 0.0
        train_records = 0
        gradient_norm = 0.0
        for index in torch.randperm(len(train_batches), generator=generator).tolist():
            batch = train_batches[index].to(device)
            optimizer.zero_grad(set_to_none=True)
            nll, _ = _losses(model, batch, normalizer)
            nll.backward()
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), train_config.gradient_clip_norm
                )
            )
            optimizer.step()
            scheduler.step()
            count = batch.condition.batch_size
            train_nll_sum += float(nll.detach()) * count
            train_records += count
            global_step += 1
        train_nll = train_nll_sum / train_records
        validation = evaluate_lattice_batches(model, val_batches, normalizer, device)
        val_nll = float(validation["nll"])
        if not math.isfinite(train_nll) or not math.isfinite(val_nll):
            raise FloatingPointError("C2L-2 lattice optimization became non-finite")
        improved = val_nll < best_val_nll
        if improved:
            best_val_nll = val_nll
            best_train_nll = train_nll
            best_epoch = epoch
            stale_epochs = 0
            atomic_checkpoint(
                output_dir / "best_val_nll.pt",
                _payload(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=global_step,
                    best_train_nll=best_train_nll,
                    model_config=model_config,
                    train_config=train_config,
                    normalizer=normalizer,
                    cache_manifest_sha256=manifest.manifest_sha256,
                    selection_split="val",
                    best_selection_nll=best_val_nll,
                ),
            )
        else:
            stale_epochs += 1
        append_metrics(
            output_dir / "metrics.csv",
            {
                "epoch": epoch,
                "step": global_step,
                "train_nll": train_nll,
                "val_nll": val_nll,
                "val_oracle_component_rmse": validation[
                    "oracle_component_rmse"
                ],
                "learning_rate": scheduler.get_last_lr()[0],
                "gradient_norm": gradient_norm,
            },
        )
        print(
            f"[c2l2] epoch={epoch}/{train_config.data_epochs} "
            f"train_nll={train_nll:.6f} val_nll={val_nll:.6f} "
            f"best_epoch={best_epoch}",
            flush=True,
        )
        if stale_epochs >= train_config.early_stopping_patience:
            break
    atomic_checkpoint(
        output_dir / "last.pt",
        _payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=global_step,
            best_train_nll=best_train_nll,
            model_config=model_config,
            train_config=train_config,
            normalizer=normalizer,
            cache_manifest_sha256=manifest.manifest_sha256,
            selection_split="val",
            best_selection_nll=best_val_nll,
        ),
    )
    summary = {
        "schema_version": "gt_sge_c2l_lattice_distribution_training_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "evidence_state": "exploratory",
        "scientific_scope": "C2L-2 full train with validation-only selection",
        "production_connected": False,
        "source_fingerprint": source_fingerprint(),
        "cache_manifest_sha256": manifest.manifest_sha256,
        "model_config": model_config.to_dict(),
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "train_config": train_config.to_dict(),
        "split_records": {
            "train": manifest.split_count("train"),
            "val": manifest.split_count("val"),
            "test_seen": 0,
        },
        "initial_validation": initial_val,
        "best_validation_nll": best_val_nll,
        "best_train_nll": best_train_nll,
        "best_epoch": best_epoch,
        "completed_epochs": epoch,
        "global_step": global_step,
        "checkpoint": "best_val_nll.pt",
    }
    summary["checkpoint_sha256"] = sha256_file(output_dir / "best_val_nll.pt")
    atomic_json(output_dir / "training_summary.json", summary)
    return summary


__all__ = [
    "LATTICE_CHECKPOINT_SCHEMA",
    "LATTICE_TARGET_CONTRACT",
    "LatticeNormalizer",
    "evaluate_lattice_batches",
    "lattice_mixture_nll",
    "load_lattice_checkpoint",
    "load_lattice_payload",
    "lattice_model_from_payload",
    "oracle_component_rmse",
    "materialize_lattice_split",
    "run_lattice_distribution_training",
]
