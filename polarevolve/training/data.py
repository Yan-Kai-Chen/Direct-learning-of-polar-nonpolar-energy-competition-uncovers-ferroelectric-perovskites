"""Training-owned DataLoader construction for verified ASU caches."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from polarevolve.crystal.contracts import ContractError
from polarevolve.crystal.symmetry import WyckoffDatabase
from polarevolve.data.cache import (
    ASUCacheManifest,
    count_cache_records,
    load_cache_manifest,
)
from polarevolve.data.packing import pack_decoded_records
from polarevolve.data.dataset import ASUCacheDataset
from polarevolve.training.config import TrainConfig

LOADER_RNG_V1 = "isolated_loader_rng_v1"


@dataclass(frozen=True)
class DistributedRecordPlan:
    split: str
    available_records: int
    selected_records: int
    world_size: int
    records_per_rank: tuple[int, ...]

    def __post_init__(self) -> None:
        valid_world = self.world_size > 0 and len(self.records_per_rank) == self.world_size
        valid_count = 0 <= self.selected_records <= self.available_records
        if not valid_world or self.available_records < 0 or not valid_count:
            raise ContractError("distributed record plan has invalid counts")
        if sum(self.records_per_rank) != self.selected_records:
            raise ContractError("distributed record plan does not conserve selected records")

    def require_all_ranks_nonempty(self) -> None:
        empty = [rank for rank, count in enumerate(self.records_per_rank) if not count]
        if empty:
            raise ContractError(
                f"split {self.split!r} leaves DDP ranks without records: {empty}; "
                "increase the global limit or reduce world_size"
            )


def distributed_record_count_plan(
    *, split: str, available_records: int, world_size: int, global_limit: int | None
) -> DistributedRecordPlan:
    if world_size <= 0 or available_records < 0:
        raise ContractError("record plan requires non-negative records and world size")
    if global_limit is not None and global_limit <= 0:
        raise ContractError("global_limit must be positive when provided")
    selected = min(available_records, global_limit) if global_limit else available_records
    per_rank = tuple(
        0 if selected <= rank else 1 + (selected - 1 - rank) // world_size
        for rank in range(world_size)
    )
    return DistributedRecordPlan(split, available_records, selected, world_size, per_rank)


@dataclass(frozen=True)
class TrainingBudget:
    selected_train_records: int
    steps_per_data_epoch: int
    requested_data_epochs: int | None
    optimizer_steps: int
    effective_batch_size: int
    samples_seen: int
    equivalent_data_epochs: float
    primary_batch_size: int
    replay_samples_seen: int


@dataclass(frozen=True)
class TrainingDataPlan:
    train: DistributedRecordPlan
    val: DistributedRecordPlan
    active_train_records: int
    active_val_records: int


def prepare_training_manifest(
    *, cache_root: Path, wyckoff_assets: Path, output_dir: Path, resume_path: Path | None
) -> ASUCacheManifest:
    """Create the run directory and verify the cache before distributed training."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "metrics.csv").exists() and resume_path is None:
        raise FileExistsError(
            "run directory already contains metrics; use a new run-id or --resume"
        )
    return load_cache_manifest(cache_root, wyckoff_database=WyckoffDatabase(wyckoff_assets))


def plan_training_data(
    manifest: ASUCacheManifest,
    *,
    config: TrainConfig,
    world_size: int,
) -> TrainingDataPlan:
    active = {
        split: count_cache_records(
            manifest,
            split=split,
            require_trainable_parameters=True,
        )
        for split in ("train", "val")
    }
    available = {
        split: active[split] if config.active_only else manifest.split_count(split)
        for split in ("train", "val")
    }
    return TrainingDataPlan(
        train=distributed_record_count_plan(
            split="train",
            available_records=available["train"],
            world_size=world_size,
            global_limit=config.train_limit,
        ),
        val=distributed_record_count_plan(
            split="val",
            available_records=available["val"],
            world_size=world_size,
            global_limit=config.val_limit,
        ),
        active_train_records=active["train"],
        active_val_records=active["val"],
    )


def resolve_training_budget(
    config: TrainConfig,
    *,
    selected_train_records: int,
    world_size: int,
) -> tuple[TrainConfig, TrainingBudget]:
    """Resolve epoch requests to the one authoritative optimizer-step budget."""

    if selected_train_records <= 0 or world_size <= 0:
        raise ValueError("training budget requires records and a positive world size")
    primary_batch = config.batch_size * world_size * config.gradient_accumulation
    effective_batch = primary_batch * (2 if config.replay_cache_relative else 1)
    steps_per_epoch = math.ceil(selected_train_records / primary_batch)
    resolved = config
    if config.data_epochs is not None:
        epoch_interval_steps = config.evaluation_interval_epochs * steps_per_epoch
        resolved = replace(
            config,
            max_steps=config.data_epochs * steps_per_epoch,
            warmup_steps=max(config.warmup_steps, config.warmup_epochs * steps_per_epoch),
            validate_every=epoch_interval_steps,
            save_every=epoch_interval_steps,
        )
    samples_seen = resolved.max_steps * primary_batch
    if config.epoch_mode == "finite_unique":
        samples_seen = config.data_epochs * selected_train_records
    budget = TrainingBudget(
        selected_train_records=selected_train_records,
        steps_per_data_epoch=steps_per_epoch,
        requested_data_epochs=config.data_epochs,
        optimizer_steps=resolved.max_steps,
        effective_batch_size=effective_batch,
        samples_seen=samples_seen,
        equivalent_data_epochs=samples_seen / selected_train_records,
        primary_batch_size=primary_batch,
        replay_samples_seen=samples_seen if config.replay_cache_relative else 0,
    )
    return resolved, budget


def effective_worker_count(requested_workers: int, *, rank_records: int, batch_size: int) -> int:
    if requested_workers == 0:
        return 0
    return min(requested_workers, max(rank_records // batch_size, 1))


def stratified_sigma_values(
    strata: tuple[float, ...],
    *,
    batch_size: int,
    step: int,
    micro_step: int,
    rank: int,
    world_size: int,
    gradient_accumulation: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    """Cycle explicit sigma strata deterministically across the global batch."""

    if not strata:
        return None
    global_micro_step = (
        (step - 1) * gradient_accumulation * world_size + micro_step * world_size + rank
    )
    start = global_micro_step * batch_size
    values = [strata[(start + index) % len(strata)] for index in range(batch_size)]
    return torch.tensor(values, device=device, dtype=dtype)


def build_loader(
    *,
    manifest: ASUCacheManifest,
    group_assets: Path,
    wyckoff_assets: Path,
    split: str,
    rank: int,
    world_size: int,
    batch_size: int,
    workers: int,
    seed: int,
    repeat: bool,
    shuffle: bool,
    global_limit: int | None,
    rank_record_count: int,
    require_trainable_parameters: bool = False,
    collate_fn: Callable = pack_decoded_records,
    drop_last: bool = True,
    pin_memory: bool | None = None,
) -> DataLoader:
    effective_workers = effective_worker_count(
        workers,
        rank_records=rank_record_count,
        batch_size=batch_size,
    )
    dataset = ASUCacheDataset(
        manifest=manifest,
        group_asset_root=group_assets,
        wyckoff_asset_root=wyckoff_assets,
        split=split,
        rank=rank,
        world_size=world_size,
        shuffle=shuffle,
        seed=seed,
        repeat=repeat,
        global_limit=global_limit,
        require_trainable_parameters=require_trainable_parameters,
    )
    use_pin_memory = torch.cuda.is_available() if pin_memory is None else pin_memory
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "num_workers": effective_workers,
        "collate_fn": collate_fn,
        "pin_memory": use_pin_memory,
        "drop_last": drop_last,
        "persistent_workers": effective_workers > 0,
        "generator": torch.Generator().manual_seed(seed + 70_001 * rank),
    }
    if effective_workers > 0:
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


def build_training_loaders(
    *,
    manifest: ASUCacheManifest,
    group_assets: Path,
    wyckoff_assets: Path,
    config: TrainConfig,
    plan: TrainingDataPlan,
    rank: int,
    world_size: int,
) -> tuple[DataLoader, DataLoader]:
    """Build the paired train/validation loaders from one admitted data plan."""
    full_validation = config.validation_mode == "full_unique"
    shared = dict(
        manifest=manifest,
        group_assets=group_assets,
        wyckoff_assets=wyckoff_assets,
        rank=rank,
        world_size=world_size,
        workers=config.num_workers,
        require_trainable_parameters=config.active_only,
    )
    specs = (
        ("train", config.batch_size, config.seed, True, config.train_limit, plan.train),
        ("val", config.val_batch_size, config.seed + 1, False, config.val_limit, plan.val),
    )
    return tuple(
        build_loader(
            **(shared | ({"rank": 0, "world_size": 1} if split == "train" and config.epoch_mode == "finite_unique" else {})),
            split=split, batch_size=batch, seed=seed, shuffle=shuffle,
            repeat=not (full_validation and split == "val") and not (split == "train" and config.epoch_mode == "finite_unique"),
            drop_last=not (full_validation and split == "val"),
            global_limit=limit, rank_record_count=record_plan.records_per_rank[rank],
        ) for split, batch, seed, shuffle, limit, record_plan in specs
    )


def finite_batch_indices(order, *, update: int, micro: int, rank: int,
                         world_size: int, batch_size: int, accumulation: int):
    """Assign every record once; an empty rank executes a zero-weight dummy."""
    effective = world_size * batch_size * accumulation
    start = update * effective + (micro * world_size + rank) * batch_size
    indices = order[start:start + batch_size]
    total = min(effective, max(len(order) - update * effective, 0))
    return indices or [order[0]], len(indices), total


class FiniteEpochLoader:
    """Decode immutable records once, then prefetch one deterministic epoch."""

    def __init__(self, loader, *, config: TrainConfig, records: int, rank: int,
                 world_size: int, device: torch.device, start_step: int, replay_loader=None):
        loader.dataset.shuffle = False
        self.records = tuple(DataLoader(loader.dataset, batch_size=None, num_workers=config.num_workers,
                                       generator=loader.generator))
        if len(self.records) != records:
            raise ValueError("finite epoch dataset differs from the frozen record count")
        self.config, self.rank, self.world_size = config, rank, world_size
        self.device, self.start_step = device, start_step
        self.steps = math.ceil(records / (config.batch_size * world_size * config.gradient_accumulation))
        self.replay_records = ()
        self.replay_orders = {}
        if replay_loader is not None:
            replay_loader.dataset.shuffle = False
            self.replay_records = tuple(DataLoader(replay_loader.dataset, batch_size=None,
                                                   num_workers=config.num_workers,
                                                   generator=replay_loader.generator))
            if not self.replay_records:
                raise ValueError("balanced replay pool is empty")

    def replay_indices(self, epoch, update, micro, valid):
        config = self.config
        start = epoch * len(self.records) + update * config.batch_size * self.world_size * config.gradient_accumulation
        start += (micro * self.world_size + self.rank) * config.batch_size
        indices = []
        for cursor in range(start, start + valid):
            cycle, index = divmod(cursor, len(self.replay_records))
            if cycle not in self.replay_orders:
                generator = torch.Generator().manual_seed(config.seed + 2_000_003 * cycle + 97)
                self.replay_orders[cycle] = torch.randperm(len(self.replay_records), generator=generator).tolist()
            indices.append(self.replay_orders[cycle][index])
        return indices or [0]

    def __iter__(self):
        config = self.config
        first_epoch, first_update = divmod(self.start_step, self.steps)
        for epoch in range(first_epoch, config.data_epochs):
            generator = torch.Generator().manual_seed(config.seed + 1_000_003 * epoch)
            order = torch.randperm(len(self.records), generator=generator).tolist()
            assignments = [finite_batch_indices(
                order, update=update, micro=micro, rank=self.rank,
                world_size=self.world_size, batch_size=config.batch_size,
                accumulation=config.gradient_accumulation,
            ) for update in range(first_update if epoch == first_epoch else 0, self.steps)
              for micro in range(config.gradient_accumulation)]
            records = self.records
            if self.replay_records:
                offset = len(records)
                records += self.replay_records
                paired = []
                first = first_update if epoch == first_epoch else 0
                for index, assignment in enumerate(assignments):
                    update, micro = divmod(index, config.gradient_accumulation)
                    indices = self.replay_indices(epoch, first + update, micro, assignment[1])
                    paired.extend((assignment, ([offset + i for i in indices], assignment[1], assignment[2])))
                assignments = paired
            loader = DataLoader(
                records, batch_sampler=[item[0] for item in assignments],
                collate_fn=pack_decoded_records, num_workers=config.num_workers,
                pin_memory=self.device.type == "cuda",
                generator=torch.Generator().manual_seed(config.seed + 1_000_003 * epoch + 70_001 * self.rank + 97),
            )
            if config.resident_epoch and self.device.type == "cuda":
                batches = []
                allocated = torch.cuda.memory_allocated(self.device)
                free, _ = torch.cuda.mem_get_info(self.device)
                for batch in loader:
                    batches.append(batch.to(self.device, non_blocking=True))
                    if torch.cuda.memory_allocated(self.device) - allocated > free * 0.3:
                        raise RuntimeError("resident epoch exceeds 30% of free VRAM; reduce residency, not batch identity")
            else:
                batches = loader
            for batch, (_, valid, total) in zip(batches, assignments, strict=True):
                yield batch, self.world_size * valid / total / (2 if self.replay_records else 1)


class ResidentValidationLoader:
    """Reuse immutable full-unique batches across raw/EMA validation passes."""

    def __init__(self, loader, device):
        self.fixed_validation_inputs = {}
        self.dataset, self.drop_last = loader.dataset, loader.drop_last
        if self.dataset.repeat or self.drop_last:
            raise ValueError("validation residency requires full_unique shards")
        allocated = torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
        free = torch.cuda.mem_get_info(device)[0] if device.type == "cuda" else float("inf")
        self.batches = []
        for batch in loader:
            self.batches.append(batch.to(device, non_blocking=True))
            if device.type == "cuda" and torch.cuda.memory_allocated(device) - allocated > free * 0.3:
                raise RuntimeError("validation residency exceeds 30% of free VRAM")

    def __iter__(self):
        return iter(self.batches)
