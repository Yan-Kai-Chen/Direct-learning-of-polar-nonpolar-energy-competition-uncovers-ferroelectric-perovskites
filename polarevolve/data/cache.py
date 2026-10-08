"""Verified MP20 ASU cache reader and crystal-domain decoder."""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any, TypeVar

import numpy as np

from polarevolve.crystal.asu import expand_orbit
from polarevolve.crystal.program import OrbitSpec, compile_hard_condition
from polarevolve.crystal.symmetry import (
    GroupDatabase,
    WyckoffDatabase,
    WyckoffGauge,
    sha256_file,
)
from polarevolve.crystal.contracts import ContractError, HardCondition
from polarevolve.crystal.state import ASUState, LatticeState, OrbitLayout

CACHE_MANIFEST_SCHEMA = "mp20_asu_cache_manifest_v1"
CACHE_RECORD_SCHEMA = "mp20_asu_cache_record_v1"
T = TypeVar("T")


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{name} must be an object")
    return value


def _safe_relative(value: object) -> str:
    relative = Path(str(value))
    if relative.is_absolute() or ".." in relative.parts:
        raise ContractError("cache manifest contains an unsafe artifact path")
    return relative.as_posix()


@dataclass(frozen=True)
class ASUCacheManifest:
    root: Path
    artifacts: tuple[tuple[str, str], ...]
    split_counts: tuple[tuple[str, int], ...]
    cached_records: int
    manifest_sha256: str

    def split_count(self, split: str) -> int:
        try:
            return dict(self.split_counts)[split]
        except KeyError as exc:
            raise ContractError(f"cache manifest does not contain split {split!r}") from exc

    def split_artifacts(self, split: str) -> tuple[Path, ...]:
        if split not in {"train", "val", "test"}:
            raise ContractError("cache split must be train, val, or test")
        prefix = f"{split}/"
        return tuple(
            self.root / relative
            for relative, _ in self.artifacts
            if relative.startswith(prefix) and relative.endswith(".jsonl")
        )


def record_has_trainable_parameters(raw_record: Mapping[str, Any]) -> bool:
    """Return whether a cache record contains at least one non-0D orbit."""

    raw_orbits = raw_record.get("orbits")
    if not isinstance(raw_orbits, list) or not raw_orbits:
        raise ContractError("cache record orbits must be a non-empty list")
    dimensions: list[int] = []
    for raw_orbit in raw_orbits:
        orbit = _mapping(raw_orbit, name="record.orbit")
        try:
            dimension = int(orbit["free_dimension"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractError("cache orbit free_dimension is malformed") from exc
        if dimension not in {0, 1, 2, 3}:
            raise ContractError("cache orbit free_dimension must lie in [0,3]")
        dimensions.append(dimension)
    return any(dimension > 0 for dimension in dimensions)


def parse_cache_record_line(
    line: str | bytes, *, path: Path, line_number: int
) -> Mapping[str, Any]:
    """Parse one non-empty cache line with a stable artifact location error."""

    try:
        value = json.loads(line)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"invalid cache JSON at {path}:{line_number}") from exc
    return _mapping(value, name="cache record")


@dataclass(frozen=True)
class CachedOrbit:
    orbit_id: str
    element: str
    atomic_number: int
    occurrence: int
    gauge: WyckoffGauge
    parameters_q0: tuple[float, ...]
    member_to_group_atom: tuple[int, ...]


@dataclass(frozen=True)
class DecodedASURecord:
    material_id: str
    split: str
    space_group_number: int
    state: ASUState
    hard_condition: HardCondition
    orbit_gauges: tuple[WyckoffGauge, ...]
    group_atom_types: tuple[int, ...]
    group_fractional: tuple[tuple[float, float, float], ...]
    member_to_group_atom: tuple[tuple[int, ...], ...]

    @property
    def layout(self) -> OrbitLayout:
        return self.state.layout


def _verify_database_identity(raw: Mapping[str, Any], database: WyckoffDatabase) -> None:
    expected = _mapping(raw.get("wyckoff_database"), name="wyckoff_database")
    actual = database.provenance
    comparisons = {
        "database_version": actual.database_version,
        "source_version": actual.source_version,
        "license_spdx": actual.license_spdx,
    }
    for key, value in comparisons.items():
        if value is not None and str(expected.get(key)) != str(value):
            raise ContractError(f"cache and Wyckoff assets disagree on {key}")
    expected_artifacts = expected.get("artifacts")
    if isinstance(expected_artifacts, Mapping):
        normalized = {str(key): str(value) for key, value in expected_artifacts.items()}
        if normalized != dict(actual.artifacts):
            raise ContractError("cache and Wyckoff asset hash manifests disagree")


def load_cache_manifest(root: str | Path, *, wyckoff_database: WyckoffDatabase) -> ASUCacheManifest:
    resolved = Path(root).resolve()
    manifest_path = resolved / "CACHE_MANIFEST.json"
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read ASU cache manifest: {manifest_path}") from exc
    manifest = _mapping(raw, name="cache manifest")
    if manifest.get("schema_version") != CACHE_MANIFEST_SCHEMA:
        raise ContractError("unsupported ASU cache manifest schema")
    _verify_database_identity(manifest, wyckoff_database)
    raw_artifacts = _mapping(manifest.get("artifacts"), name="cache artifacts")
    artifacts: list[tuple[str, str]] = []
    for raw_relative, raw_digest in sorted(raw_artifacts.items()):
        relative = _safe_relative(raw_relative)
        digest = str(raw_digest).lower()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ContractError(f"invalid cache SHA-256 for {relative}")
        path = (resolved / relative).resolve()
        try:
            path.relative_to(resolved)
        except ValueError as exc:
            raise ContractError("cache artifact escapes the cache root") from exc
        if not path.is_file():
            raise ContractError(f"cache artifact is missing: {path}")
        actual = sha256_file(path)
        if actual != digest:
            raise ContractError(
                f"cache artifact hash mismatch for {relative}: expected {digest}, got {actual}"
            )
        artifacts.append((relative, digest))
    split_counts_raw = _mapping(manifest.get("split_counts"), name="split_counts")
    split_counts = tuple(
        (split, int(_mapping(values, name=f"split_counts.{split}").get("cached", 0)))
        for split, values in sorted(split_counts_raw.items())
    )
    cached_records = int(manifest.get("cached_records", sum(value for _, value in split_counts)))
    if cached_records != sum(value for _, value in split_counts):
        raise ContractError("cache manifest total does not equal its split counts")
    return ASUCacheManifest(
        root=resolved,
        artifacts=tuple(artifacts),
        split_counts=split_counts,
        cached_records=cached_records,
        manifest_sha256=sha256_file(manifest_path),
    )


def buffered_shuffle(records: Iterable[T], *, rng: random.Random, buffer_size: int) -> Iterator[T]:
    """Shuffle a stream with the production bounded-memory algorithm."""

    buffer: list[T] = []
    for record in records:
        if len(buffer) < buffer_size:
            buffer.append(record)
            continue
        index = rng.randrange(len(buffer))
        yield buffer[index]
        buffer[index] = record
    rng.shuffle(buffer)
    yield from buffer


def iter_cache_records(
    manifest: ASUCacheManifest,
    *,
    split: str,
    shuffle: bool = False,
    seed: int = 0,
    epoch: int = 0,
    rank: int = 0,
    world_size: int = 1,
    limit: int | None = None,
    global_limit: int | None = None,
    selected_material_ids: frozenset[str] | None = None,
    require_trainable_parameters: bool = False,
    shuffle_buffer_size: int = 512,
) -> Iterator[Mapping[str, Any]]:
    if world_size <= 0 or not 0 <= rank < world_size:
        raise ContractError("rank must lie in [0, world_size)")
    if limit is not None and limit <= 0:
        return
    if global_limit is not None and global_limit <= 0:
        return
    if selected_material_ids is not None and not selected_material_ids:
        raise ContractError("selected_material_ids must be non-empty when provided")
    if selected_material_ids is not None and global_limit is not None:
        raise ContractError("selected_material_ids and global_limit are mutually exclusive")
    if shuffle_buffer_size <= 0:
        raise ContractError("shuffle_buffer_size must be positive")
    canonical_paths = list(manifest.split_artifacts(split))
    rng = random.Random(int(seed) + 1_000_003 * int(epoch))

    def stream(paths: Iterable[Path]) -> Iterator[Mapping[str, Any]]:
        for path in paths:
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    yield parse_cache_record_line(line, path=path, line_number=line_number)

    def eligible_records(paths: Iterable[Path]) -> Iterator[Mapping[str, Any]]:
        for record in stream(paths):
            if selected_material_ids is not None:
                source = _mapping(record.get("source"), name="record.source")
                if str(source.get("material_id", "")) not in selected_material_ids:
                    continue
            if require_trainable_parameters and not record_has_trainable_parameters(record):
                continue
            yield record

    if global_limit is not None:
        # A diagnostic limit denotes one stable canonical subset. Epoch shuffling
        # may change its order, but must never change its membership.
        records: Iterable[Mapping[str, Any]] = islice(
            eligible_records(canonical_paths), global_limit
        )
        if shuffle:
            records = buffered_shuffle(
                records, rng=rng, buffer_size=min(shuffle_buffer_size, global_limit)
            )
    else:
        paths = list(canonical_paths)
        if shuffle:
            rng.shuffle(paths)
        records = eligible_records(paths)
        if shuffle:
            records = buffered_shuffle(records, rng=rng, buffer_size=shuffle_buffer_size)
    emitted = 0
    selected_index = 0
    for record in records:
        assigned_rank = selected_index % world_size
        selected_index += 1
        if assigned_rank != rank:
            continue
        yield record
        emitted += 1
        if limit is not None and emitted >= limit:
            return


def count_cache_records(
    manifest: ASUCacheManifest,
    *,
    split: str,
    require_trainable_parameters: bool = False,
) -> int:
    """Count verified records after applying the production data-view filter."""

    return sum(
        1
        for _ in iter_cache_records(
            manifest,
            split=split,
            require_trainable_parameters=require_trainable_parameters,
        )
    )


def _lattice(raw: Any) -> LatticeState:
    array = np.asarray(raw, dtype=np.float64)
    if array.shape != (3, 3) or not np.isfinite(array).all():
        raise ContractError("cache group lattice must be a finite 3x3 matrix")
    return LatticeState(tuple(tuple(float(value) for value in row) for row in array))


def decode_cache_record(
    raw_record: Mapping[str, Any],
    *,
    group_database: GroupDatabase,
    wyckoff_database: WyckoffDatabase,
    geometry_tolerance_angstrom: float = 1.0e-5,
) -> DecodedASURecord:
    if raw_record.get("schema_version") != CACHE_RECORD_SCHEMA:
        raise ContractError("unsupported ASU cache record schema")
    source = _mapping(raw_record.get("source"), name="record.source")
    symmetry = _mapping(raw_record.get("symmetry"), name="record.symmetry")
    group_cell = _mapping(raw_record.get("group_cell"), name="record.group_cell")
    hall_number = int(symmetry["hall_number"])
    setting = group_database.setting(hall_number)
    if str(symmetry["setting_id"]) != setting.setting_id:
        raise ContractError("cache record and Hall setting IDs disagree")
    if int(symmetry["space_group_number"]) != setting.space_group_number:
        raise ContractError("cache record and Hall space-group numbers disagree")
    lattice = _lattice(group_cell.get("lattice"))
    lattice_array = np.asarray(lattice.matrix, dtype=np.float64)
    atom_types = tuple(int(value) for value in group_cell.get("atom_types", ()))
    fractional_array = np.asarray(group_cell.get("fractional_coordinates"), dtype=np.float64)
    atom_count = int(group_cell.get("atom_count", -1))
    if atom_count <= 0 or len(atom_types) != atom_count:
        raise ContractError("cache group atom count does not match atom types")
    if fractional_array.shape != (atom_count, 3) or not np.isfinite(fractional_array).all():
        raise ContractError("cache group fractional coordinates are malformed")

    cached_orbits: list[CachedOrbit] = []
    specs: list[OrbitSpec] = []
    all_members: list[int] = []
    parameters: list[float] = []
    offsets = [0]
    for raw_orbit in raw_record.get("orbits", ()):
        orbit = _mapping(raw_orbit, name="record.orbit")
        gauge = wyckoff_database.entry(hall_number, str(orbit["wyckoff_letter"]))
        if gauge.setting_id != setting.setting_id:
            raise ContractError("cache orbit and Wyckoff setting IDs disagree")
        if str(orbit["table_entry_hash"]) != gauge.table_entry_hash:
            raise ContractError("cache orbit Wyckoff entry hash mismatch")
        if (
            int(orbit["multiplicity"]) != gauge.multiplicity
            or int(orbit["free_dimension"]) != gauge.free_dimension
        ):
            raise ContractError("cache orbit dimensions disagree with Wyckoff assets")
        q0 = tuple(float(value) for value in orbit.get("parameters_q0", ()))
        if len(q0) != gauge.free_dimension or not np.isfinite(q0).all():
            raise ContractError("cache orbit ASU parameters are malformed")
        members = tuple(int(value) for value in orbit.get("member_to_group_atom", ()))
        if len(members) != gauge.multiplicity or any(
            index < 0 or index >= atom_count for index in members
        ):
            raise ContractError("cache orbit member map is malformed")
        atomic_number = int(orbit["atomic_number"])
        if any(atom_types[index] != atomic_number for index in members):
            raise ContractError("cache orbit member map changes atomic species")
        expected = expand_orbit(gauge, q0)
        observed = fractional_array[np.asarray(members)]
        displacement = expected - observed
        displacement -= np.round(displacement)
        error = np.linalg.norm(displacement @ lattice_array, axis=1)
        if float(error.max(initial=0.0)) > geometry_tolerance_angstrom:
            raise ContractError("cache orbit geometry does not match its exact gauge")
        occurrence = int(orbit.get("occurrence", 0)) + 1
        element = str(orbit["element"])
        cached_orbits.append(
            CachedOrbit(
                orbit_id=str(orbit["orbit_id"]),
                element=element,
                atomic_number=atomic_number,
                occurrence=occurrence,
                gauge=gauge,
                parameters_q0=q0,
                member_to_group_atom=members,
            )
        )
        specs.append(OrbitSpec(element, gauge.letter, occurrence))
        all_members.extend(members)
        parameters.extend(q0)
        offsets.append(len(parameters))
    if not cached_orbits or sorted(all_members) != list(range(atom_count)):
        raise ContractError("cache orbit members must form one full atom permutation")
    compiled = compile_hard_condition(
        condition_id=str(source["material_id"]),
        hall_number=hall_number,
        orbit_specs=tuple(specs),
        base_cell_representation="conventional",
        group_database=group_database,
        wyckoff_database=wyckoff_database,
    )
    if compiled.hard.base_num_atoms != atom_count:
        raise ContractError("compiled hard condition does not close the group cell")
    layout = OrbitLayout(
        orbit_ids=tuple(orbit.site_id for orbit in compiled.hard.wyckoff_orbits),
        free_dimensions=tuple(item.gauge.free_dimension for item in cached_orbits),
        parameter_offsets=tuple(offsets),
    )
    state = ASUState(tuple(parameters), lattice, layout)
    return DecodedASURecord(
        material_id=str(source["material_id"]),
        split=str(source["split"]),
        space_group_number=setting.space_group_number,
        state=state,
        hard_condition=compiled.hard,
        orbit_gauges=tuple(item.gauge for item in cached_orbits),
        group_atom_types=atom_types,
        group_fractional=tuple(tuple(float(value) for value in row) for row in fractional_array),
        member_to_group_atom=tuple(item.member_to_group_atom for item in cached_orbits),
    )


__all__ = [
    "ASUCacheManifest",
    "CACHE_MANIFEST_SCHEMA",
    "CACHE_RECORD_SCHEMA",
    "CachedOrbit",
    "DecodedASURecord",
    "buffered_shuffle",
    "count_cache_records",
    "decode_cache_record",
    "iter_cache_records",
    "load_cache_manifest",
    "parse_cache_record_line",
    "record_has_trainable_parameters",
]
