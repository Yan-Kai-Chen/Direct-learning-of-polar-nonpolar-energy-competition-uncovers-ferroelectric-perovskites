"""Deterministic streaming dataset for verified ASU caches."""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import IterableDataset, get_worker_info

from polarevolve.crystal.lattice import HallMetricFrame, build_hall_metric_frame
from polarevolve.crystal.symmetry import GroupDatabase, WyckoffDatabase
from polarevolve.data.cache import (
    ASUCacheManifest,
    DecodedASURecord,
    decode_cache_record,
    load_cache_manifest,
)
from polarevolve.data.index import (
    build_cache_split_index,
    iter_indexed_cache_records,
)


class ASUCacheDataset(IterableDataset[DecodedASURecord]):
    """Shard cache records across DDP ranks and DataLoader workers exactly once."""

    def __init__(
        self,
        *,
        manifest: ASUCacheManifest,
        group_asset_root: str | Path,
        wyckoff_asset_root: str | Path,
        split: str,
        rank: int = 0,
        world_size: int = 1,
        shuffle: bool = False,
        seed: int = 0,
        repeat: bool = False,
        global_limit: int | None = None,
        require_trainable_parameters: bool = False,
        shuffle_buffer_size: int = 512,
    ) -> None:
        super().__init__()
        self.manifest = manifest
        self.group_asset_root = Path(group_asset_root)
        self.wyckoff_asset_root = Path(wyckoff_asset_root)
        self.split = split
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.repeat = bool(repeat)
        self.global_limit = global_limit
        self.require_trainable_parameters = bool(require_trainable_parameters)
        self.shuffle_buffer_size = int(shuffle_buffer_size)
        if self.world_size <= 0 or not 0 <= self.rank < self.world_size:
            raise ValueError("rank must lie in [0,world_size)")
        self.cache_index = build_cache_split_index(manifest, split=split)

    def __iter__(self):
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        workers = 1 if worker is None else worker.num_workers
        group_database = GroupDatabase(self.group_asset_root)
        wyckoff_database = WyckoffDatabase(self.wyckoff_asset_root)
        epoch = 0
        while True:
            emitted = 0
            records = iter_indexed_cache_records(
                self.cache_index,
                shuffle=self.shuffle,
                seed=self.seed,
                epoch=epoch,
                rank=self.rank,
                world_size=self.world_size,
                global_limit=self.global_limit,
                require_trainable_parameters=self.require_trainable_parameters,
                shuffle_buffer_size=self.shuffle_buffer_size,
                worker_id=worker_id,
                worker_count=workers,
            )
            for raw in records:
                emitted += 1
                yield decode_cache_record(
                    raw,
                    group_database=group_database,
                    wyckoff_database=wyckoff_database,
                )
            if not self.repeat:
                return
            if emitted == 0:
                return
            epoch += 1


def load_lattice_distribution_context(
    *,
    cache_root: str | Path,
    wyckoff_asset_root: str | Path,
) -> tuple[ASUCacheManifest, dict[int, HallMetricFrame]]:
    """Load the verified cache identity and all Hall lattice frames once."""

    wyckoff = WyckoffDatabase(wyckoff_asset_root)
    manifest = load_cache_manifest(cache_root, wyckoff_database=wyckoff)
    frames = {
        hall: build_hall_metric_frame(wyckoff, hall) for hall in range(1, 531)
    }
    return manifest, frames


__all__ = [
    "ASUCacheDataset",
    "load_lattice_distribution_context",
]
