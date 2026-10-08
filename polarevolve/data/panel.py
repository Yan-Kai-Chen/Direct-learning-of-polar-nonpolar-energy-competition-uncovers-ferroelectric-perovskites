"""Cache-bound deterministic evaluation-panel contracts."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from polarevolve.crystal.contracts import ContractError
from polarevolve.data.cache import ASUCacheManifest, iter_cache_records

EVALUATION_PANEL_SCHEMA = "gt_sge_evaluation_panel_v1"


def crystal_system_from_space_group(space_group_number: int) -> str:
    """Return the conventional crystal system for an International SG number."""

    if not 1 <= space_group_number <= 230:
        raise ContractError("space-group number must lie in [1,230]")
    upper_bounds = (2, 15, 74, 142, 167, 194, 230)
    names = (
        "triclinic",
        "monoclinic",
        "orthorhombic",
        "tetragonal",
        "trigonal",
        "hexagonal",
        "cubic",
    )
    return next(name for bound, name in zip(upper_bounds, names) if space_group_number <= bound)


@dataclass(frozen=True)
class EvaluationPanelEntry:
    material_id: str
    split: str
    hall_number: int
    space_group_number: int
    crystal_system: str
    maximum_free_dimension: int
    maximum_orbit_multiplicity: int
    atom_count: int
    atom_count_quantile: str
    zero_dimensional_only: bool

    def __post_init__(self) -> None:
        if not self.material_id or self.split not in {"train", "val", "test"}:
            raise ContractError("evaluation panel entry has an invalid identity")
        expected_system = crystal_system_from_space_group(self.space_group_number)
        if self.crystal_system != expected_system:
            raise ContractError("evaluation panel crystal-system label is inconsistent")
        if not 1 <= self.hall_number <= 530:
            raise ContractError("evaluation panel Hall number must lie in [1,530]")
        if self.maximum_free_dimension not in {0, 1, 2, 3}:
            raise ContractError("evaluation panel free dimension must lie in [0,3]")
        if self.maximum_orbit_multiplicity <= 0 or self.atom_count <= 0:
            raise ContractError("evaluation panel counts must be positive")
        if self.atom_count_quantile not in {"q1", "q2", "q3", "q4"}:
            raise ContractError("evaluation panel atom-count quantile is invalid")
        if self.zero_dimensional_only != (self.maximum_free_dimension == 0):
            raise ContractError("evaluation panel 0D label is inconsistent")

    def to_dict(self) -> dict[str, Any]:
        return {
            "material_id": self.material_id,
            "split": self.split,
            "hall_number": self.hall_number,
            "space_group_number": self.space_group_number,
            "crystal_system": self.crystal_system,
            "maximum_free_dimension": self.maximum_free_dimension,
            "maximum_orbit_multiplicity": self.maximum_orbit_multiplicity,
            "atom_count": self.atom_count,
            "atom_count_quantile": self.atom_count_quantile,
            "zero_dimensional_only": self.zero_dimensional_only,
        }


def _panel_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class EvaluationPanel:
    cache_manifest_sha256: str
    split: str
    seed: int
    requested_records: int
    available_records: int
    atom_count_quantile_boundaries: tuple[int, int, int]
    entries: tuple[EvaluationPanelEntry, ...]
    selection_sha256: str

    def __post_init__(self) -> None:
        digest = self.cache_manifest_sha256.lower()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ContractError("evaluation panel cache SHA-256 is malformed")
        if self.split not in {"train", "val", "test"}:
            raise ContractError("evaluation panel split is invalid")
        if self.requested_records != len(self.entries) or not (
            0 < self.requested_records <= self.available_records
        ):
            raise ContractError("evaluation panel record counts are inconsistent")
        if (
            tuple(sorted(self.atom_count_quantile_boundaries))
            != self.atom_count_quantile_boundaries
        ):
            raise ContractError("evaluation panel atom-count boundaries must be sorted")
        identifiers = [entry.material_id for entry in self.entries]
        if len(identifiers) != len(set(identifiers)):
            raise ContractError("evaluation panel material IDs must be unique")
        if any(entry.split != self.split for entry in self.entries):
            raise ContractError("evaluation panel entries mix cache splits")
        if self.selection_sha256 != _panel_digest(self._payload()):
            raise ContractError("evaluation panel selection SHA-256 mismatch")

    @property
    def material_ids(self) -> frozenset[str]:
        return frozenset(entry.material_id for entry in self.entries)

    @property
    def entries_by_material(self) -> dict[str, EvaluationPanelEntry]:
        return {entry.material_id: entry for entry in self.entries}

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": EVALUATION_PANEL_SCHEMA,
            "cache_manifest_sha256": self.cache_manifest_sha256,
            "split": self.split,
            "seed": self.seed,
            "requested_records": self.requested_records,
            "available_records": self.available_records,
            "atom_count_quantile_boundaries": list(self.atom_count_quantile_boundaries),
            "entries": [entry.to_dict() for entry in self.entries],
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "selection_sha256": self.selection_sha256}


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{name} must be an object")
    return value


def _record_features(raw_record: Mapping[str, Any]) -> dict[str, Any]:
    source = _mapping(raw_record.get("source"), name="record.source")
    symmetry = _mapping(raw_record.get("symmetry"), name="record.symmetry")
    group_cell = _mapping(raw_record.get("group_cell"), name="record.group_cell")
    raw_orbits = raw_record.get("orbits")
    if not isinstance(raw_orbits, list) or not raw_orbits:
        raise ContractError("evaluation panel requires a non-empty orbit list")
    dimensions: list[int] = []
    multiplicities: list[int] = []
    for raw_orbit in raw_orbits:
        orbit = _mapping(raw_orbit, name="record.orbit")
        dimensions.append(int(orbit["free_dimension"]))
        multiplicities.append(int(orbit["multiplicity"]))
    if any(value not in {0, 1, 2, 3} for value in dimensions):
        raise ContractError("evaluation panel encountered an invalid free dimension")
    if any(value <= 0 for value in multiplicities):
        raise ContractError("evaluation panel encountered an invalid multiplicity")
    space_group_number = int(symmetry["space_group_number"])
    return {
        "material_id": str(source["material_id"]),
        "split": str(source["split"]),
        "hall_number": int(symmetry["hall_number"]),
        "space_group_number": space_group_number,
        "crystal_system": crystal_system_from_space_group(space_group_number),
        "maximum_free_dimension": max(dimensions),
        "maximum_orbit_multiplicity": max(multiplicities),
        "atom_count": int(group_cell["atom_count"]),
    }


def _quantile_boundaries(atom_counts: list[int]) -> tuple[int, int, int]:
    ordered = sorted(atom_counts)
    if not ordered:
        raise ContractError("cannot stratify an empty cache split")
    indices = [max(0, (len(ordered) * part + 3) // 4 - 1) for part in (1, 2, 3)]
    return tuple(ordered[min(index, len(ordered) - 1)] for index in indices)


def _quantile_label(atom_count: int, boundaries: tuple[int, int, int]) -> str:
    for index, boundary in enumerate(boundaries, start=1):
        if atom_count <= boundary:
            return f"q{index}"
    return "q4"


def _stable_key(seed: int, *parts: object) -> str:
    value = ":".join((str(seed), *(str(part) for part in parts)))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _allocate_stratum_quotas(
    groups: Mapping[tuple[Any, ...], list[dict[str, Any]]], requested: int, seed: int
) -> dict[tuple[Any, ...], int]:
    keys = list(groups)
    quotas = {key: 0 for key in keys}
    if requested >= len(keys):
        for key in keys:
            quotas[key] = 1
    else:
        ranked = sorted(
            keys,
            key=lambda key: (-len(groups[key]), _stable_key(seed, "stratum", *key)),
        )
        for key in ranked[:requested]:
            quotas[key] = 1
    remaining = requested - sum(quotas.values())
    while remaining:
        capacities = {key: len(groups[key]) - quotas[key] for key in keys}
        total_capacity = sum(max(0, value) for value in capacities.values())
        if total_capacity < remaining:
            raise ContractError("evaluation panel quota allocation exhausted the split")
        ideals = {
            key: remaining * max(0, capacity) / total_capacity
            for key, capacity in capacities.items()
        }
        grants = {key: min(capacities[key], int(ideals[key])) for key in keys}
        granted = sum(grants.values())
        if granted == 0:
            key = max(
                (item for item in keys if capacities[item] > 0),
                key=lambda item: (
                    ideals[item],
                    capacities[item],
                    _stable_key(seed, "remainder", *item),
                ),
            )
            grants[key] = 1
            granted = 1
        for key, grant in grants.items():
            quotas[key] += grant
        remaining -= granted
    return quotas


def _select_within_stratum(
    records: list[dict[str, Any]], *, count: int, seed: int
) -> list[dict[str, Any]]:
    buckets: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        buckets[(record["hall_number"], record["maximum_orbit_multiplicity"])].append(record)
    for values in buckets.values():
        values.sort(key=lambda item: _stable_key(seed, item["material_id"]))
    bucket_order = sorted(
        buckets,
        key=lambda key: _stable_key(seed, "hall-multiplicity", *key),
    )
    selected: list[dict[str, Any]] = []
    while len(selected) < count:
        emitted = False
        for key in bucket_order:
            if buckets[key]:
                selected.append(buckets[key].pop(0))
                emitted = True
                if len(selected) == count:
                    break
        if not emitted:
            raise ContractError("evaluation panel stratum quota exceeds its population")
    return selected


def build_evaluation_panel(
    manifest: ASUCacheManifest, *, split: str, requested_records: int | None, seed: int
) -> EvaluationPanel:
    """Build a deterministic, representative coverage panel from one cache split."""

    features = [_record_features(raw) for raw in iter_cache_records(manifest, split=split)]
    requested_records = len(features) if requested_records is None else requested_records
    if requested_records <= 0:
        raise ContractError("evaluation panel size must be positive")
    if requested_records > len(features):
        raise ContractError(
            f"evaluation panel requests {requested_records} records from a {len(features)}-record split"
        )
    boundaries = _quantile_boundaries([item["atom_count"] for item in features])
    for item in features:
        item["atom_count_quantile"] = _quantile_label(item["atom_count"], boundaries)
        item["zero_dimensional_only"] = item["maximum_free_dimension"] == 0
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for item in features:
        primary = (
            item["zero_dimensional_only"],
            item["crystal_system"],
            item["maximum_free_dimension"],
            item["atom_count_quantile"],
        )
        groups[primary].append(item)
    quotas = _allocate_stratum_quotas(groups, requested_records, seed)
    selected: list[dict[str, Any]] = []
    for key in sorted(groups, key=lambda value: _stable_key(seed, "primary", *value)):
        selected.extend(_select_within_stratum(groups[key], count=quotas[key], seed=seed))
    entries = tuple(
        EvaluationPanelEntry(**item)
        for item in sorted(selected, key=lambda value: value["material_id"])
    )
    payload = {
        "schema_version": EVALUATION_PANEL_SCHEMA,
        "cache_manifest_sha256": manifest.manifest_sha256,
        "split": split,
        "seed": int(seed),
        "requested_records": requested_records,
        "available_records": len(features),
        "atom_count_quantile_boundaries": list(boundaries),
        "entries": [entry.to_dict() for entry in entries],
    }
    return EvaluationPanel(
        cache_manifest_sha256=manifest.manifest_sha256,
        split=split,
        seed=int(seed),
        requested_records=requested_records,
        available_records=len(features),
        atom_count_quantile_boundaries=boundaries,
        entries=entries,
        selection_sha256=_panel_digest(payload),
    )


def load_evaluation_panel(
    path: str | Path,
    *,
    expected_cache_manifest_sha256: str | None = None,
    expected_split: str | None = None,
) -> EvaluationPanel:
    resolved = Path(path).expanduser().resolve()
    try:
        raw = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read evaluation panel: {resolved}") from exc
    value = _mapping(raw, name="evaluation panel")
    if value.get("schema_version") != EVALUATION_PANEL_SCHEMA:
        raise ContractError("unsupported evaluation panel schema")
    raw_entries = value.get("entries")
    if not isinstance(raw_entries, list):
        raise ContractError("evaluation panel entries must be a list")
    panel = EvaluationPanel(
        cache_manifest_sha256=str(value["cache_manifest_sha256"]),
        split=str(value["split"]),
        seed=int(value["seed"]),
        requested_records=int(value["requested_records"]),
        available_records=int(value["available_records"]),
        atom_count_quantile_boundaries=tuple(
            int(item) for item in value["atom_count_quantile_boundaries"]
        ),
        entries=tuple(EvaluationPanelEntry(**entry) for entry in raw_entries),
        selection_sha256=str(value["selection_sha256"]),
    )
    if (
        expected_cache_manifest_sha256 is not None
        and panel.cache_manifest_sha256 != expected_cache_manifest_sha256
    ):
        raise ContractError("evaluation panel was built from a different cache manifest")
    if expected_split is not None and panel.split != expected_split:
        raise ContractError("evaluation panel split does not match the requested split")
    return panel


__all__ = [
    "EVALUATION_PANEL_SCHEMA",
    "EvaluationPanel",
    "EvaluationPanelEntry",
    "build_evaluation_panel",
    "crystal_system_from_space_group",
    "load_evaluation_panel",
]
