"""Lightweight cache offsets for worker-local JSON parsing."""

from __future__ import annotations

import random
from contextlib import ExitStack
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any, Iterator, Mapping

from polarevolve.crystal.contracts import ContractError
from polarevolve.data.cache import (
    ASUCacheManifest,
    buffered_shuffle,
    parse_cache_record_line,
    record_has_trainable_parameters,
)


@dataclass(frozen=True)
class CacheRecordLocation:
    artifact_index: int
    byte_offset: int
    line_number: int
    trainable: bool


@dataclass(frozen=True)
class CacheSplitIndex:
    manifest_sha256: str
    split: str
    paths: tuple[Path, ...]
    records_by_artifact: tuple[tuple[CacheRecordLocation, ...], ...]
    record_count: int


def build_cache_split_index(
    manifest: ASUCacheManifest, *, split: str
) -> CacheSplitIndex:
    """Scan one split once before DataLoader workers are created."""

    paths = manifest.split_artifacts(split)
    records_by_artifact: list[tuple[CacheRecordLocation, ...]] = []
    record_count = 0
    for artifact_index, path in enumerate(paths):
        locations: list[CacheRecordLocation] = []
        with path.open("rb") as handle:
            line_number = 0
            while True:
                byte_offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                line_number += 1
                if not line.strip():
                    continue
                record = parse_cache_record_line(
                    line, path=path, line_number=line_number
                )
                locations.append(
                    CacheRecordLocation(
                        artifact_index=artifact_index,
                        byte_offset=byte_offset,
                        line_number=line_number,
                        trainable=record_has_trainable_parameters(record),
                    )
                )
        record_count += len(locations)
        records_by_artifact.append(tuple(locations))
    expected = manifest.split_count(split)
    if record_count != expected:
        raise ContractError(
            f"cache split index count mismatch for {split}: "
            f"expected {expected}, found {record_count}"
        )
    return CacheSplitIndex(
        manifest_sha256=manifest.manifest_sha256,
        split=split,
        paths=paths,
        records_by_artifact=tuple(records_by_artifact),
        record_count=record_count,
    )


def iter_indexed_locations(
    index: CacheSplitIndex,
    *,
    shuffle: bool = False,
    seed: int = 0,
    epoch: int = 0,
    rank: int = 0,
    world_size: int = 1,
    worker_id: int = 0,
    worker_count: int = 1,
    limit: int | None = None,
    global_limit: int | None = None,
    require_trainable_parameters: bool = False,
    shuffle_buffer_size: int = 512,
) -> Iterator[CacheRecordLocation]:
    """Reproduce canonical rank/worker ordering without reparsing other records."""

    if world_size <= 0 or not 0 <= rank < world_size:
        raise ContractError("rank must lie in [0, world_size)")
    if worker_count <= 0 or not 0 <= worker_id < worker_count:
        raise ContractError("worker_id must lie in [0,worker_count)")
    if limit is not None and limit <= 0:
        return
    if global_limit is not None and global_limit <= 0:
        return
    if shuffle_buffer_size <= 0:
        raise ContractError("shuffle_buffer_size must be positive")
    rng = random.Random(int(seed) + 1_000_003 * int(epoch))

    def eligible(artifact_order: list[int]) -> Iterator[CacheRecordLocation]:
        for artifact_index in artifact_order:
            for location in index.records_by_artifact[artifact_index]:
                if require_trainable_parameters and not location.trainable:
                    continue
                yield location

    artifact_order = list(range(len(index.paths)))
    if global_limit is not None:
        locations: Iterator[CacheRecordLocation] = islice(
            eligible(artifact_order), global_limit
        )
        if shuffle:
            locations = buffered_shuffle(
                locations,
                rng=rng,
                buffer_size=min(shuffle_buffer_size, global_limit),
            )
    else:
        if shuffle:
            rng.shuffle(artifact_order)
        locations = eligible(artifact_order)
        if shuffle:
            locations = buffered_shuffle(
                locations, rng=rng, buffer_size=shuffle_buffer_size
            )

    rank_records = 0
    for selected_index, location in enumerate(locations):
        if selected_index % world_size != rank:
            continue
        if limit is not None and rank_records >= limit:
            return
        local_index = rank_records
        rank_records += 1
        if local_index % worker_count == worker_id:
            yield location


def iter_indexed_cache_records(
    index: CacheSplitIndex, **selection: Any
) -> Iterator[Mapping[str, Any]]:
    """Read only records assigned to one rank-local DataLoader worker."""

    with ExitStack() as stack:
        handles = {}
        for location in iter_indexed_locations(index, **selection):
            handle = handles.get(location.artifact_index)
            if handle is None:
                handle = stack.enter_context(
                    index.paths[location.artifact_index].open("rb")
                )
                handles[location.artifact_index] = handle
            handle.seek(location.byte_offset)
            line = handle.readline()
            if not line:
                raise ContractError("cache record offset points past end of artifact")
            yield parse_cache_record_line(
                line,
                path=index.paths[location.artifact_index],
                line_number=location.line_number,
            )


__all__ = [
    "CacheRecordLocation",
    "CacheSplitIndex",
    "build_cache_split_index",
    "iter_indexed_cache_records",
    "iter_indexed_locations",
]
