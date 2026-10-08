"""FE1 condition sidecar schema and packed condition batch.

One sidecar row per admitted transition pair carries the gold OD target
vector (planner 18 order), the aligned polar reference structure in the
parent supercell frame, and the repeat contract linking the parent group
cell to that frame.  The module owns schema validation; tensor packing for
training produces :class:`FerroConditionBatch`.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from polarevolve.crystal.symmetry import sha256_file
from polarevolve.tasks.ferroelectric.od_registry import (
    CORRESPONDENCE_IDS,
    PLANNER_OD_IDS,
)

CONDITION_SCHEMA = "fe1_transition_condition_v1"
CHILD_CONDITION_SCHEMA = "fe3_parent_to_child_condition_v1"
CONDITION_MANIFEST_SCHEMA = "fe1_transition_condition_manifest_v1"
CONDITION_ENCODING = "child_geometry_plus_source_sg_v2"
CHILD_CONDITION_ENCODING = "parent_sg_plus_selected_od_top3_mask_v1"
SPACE_GROUP_COUNT = 230
CONDITION_DIMENSION = SPACE_GROUP_COUNT
CHILD_CONDITION_DIMENSION = SPACE_GROUP_COUNT + len(PLANNER_OD_IDS)
ATOM_CONDITION_DIMENSION = 4
DEFAULT_CORRESPONDENCE_MAX_SITE_RMS_ANGSTROM = 1.0


def verify_condition_asset(
    root: str | Path, *, expected_cache_sha256: str
) -> tuple[Path, str]:
    """Verify the sidecar manifest and return its JSONL path and identity."""

    root = Path(root).resolve()
    manifest_path = root / "CONDITION_MANIFEST.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read FE1 condition manifest: {manifest_path}") from exc
    if manifest.get("schema_version") != CONDITION_MANIFEST_SCHEMA:
        raise ValueError("unsupported FE1 condition manifest schema")
    condition_schema = manifest.get("condition_schema")
    if condition_schema not in {
        CONDITION_SCHEMA,
        CHILD_CONDITION_SCHEMA,
    }:
        raise ValueError("condition manifest and record schemas disagree")
    expected_encoding = (
        CHILD_CONDITION_ENCODING
        if condition_schema == CHILD_CONDITION_SCHEMA
        else CONDITION_ENCODING
    )
    actual_encoding = manifest.get("condition_encoding")
    legacy_encoding_omitted = (
        condition_schema == CONDITION_SCHEMA and actual_encoding is None
    )
    if actual_encoding != expected_encoding and not legacy_encoding_omitted:
        raise ValueError("condition manifest encoding differs from the schema contract")
    if tuple(manifest.get("od_order", ())) != PLANNER_OD_IDS:
        raise ValueError("condition manifest OD order differs from the planner contract")
    if manifest.get("cache_manifest_sha256") != expected_cache_sha256:
        raise ValueError("condition sidecar and FE1 cache identities disagree")
    relative = "conditions.jsonl"
    expected = manifest.get("artifacts", {}).get(relative)
    path = root / relative
    if not isinstance(expected, str) or sha256_file(path) != expected:
        raise ValueError(f"condition artifact hash mismatch: {path}")
    return path, sha256_file(manifest_path)


@dataclass(frozen=True)
class FerroConditionRecord:
    pair_id: str
    split: str
    parent_material_id: str
    polar_material_id: str
    source_space_group: int
    parent_space_group: int
    od_gold: tuple[float, ...]
    od_mask: tuple[int, ...]
    repeat_matrix: tuple[tuple[int, int, int], ...]
    translations: tuple[tuple[int, int, int], ...]
    ref_lattice: tuple[tuple[float, float, float], ...]
    ref_frac: tuple[tuple[float, float, float], ...]
    ref_role_ids: tuple[int, ...]
    ref_atom_types: tuple[int, ...]
    site_rms_angstrom: float
    schema_version: str = CONDITION_SCHEMA


def _float_matrix(value: Any, *, shape: tuple[int, ...], name: str) -> tuple:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"condition {name} must be a nested sequence")
    flat = tuple(float(item) for row in value for item in row)
    expected = math.prod(shape)
    if len(flat) != expected or not all(math.isfinite(item) for item in flat):
        raise ValueError(f"condition {name} must contain {expected} finite numbers")
    width = shape[-1]
    return tuple(tuple(flat[index:index + width]) for index in range(0, expected, width))


def parse_condition_record(raw: Mapping[str, Any], *, path: Path, line_number: int) -> FerroConditionRecord:
    try:
        schema_version = str(raw.get("schema_version"))
        if schema_version not in {CONDITION_SCHEMA, CHILD_CONDITION_SCHEMA}:
            raise ValueError("unsupported condition schema")
        expected_encoding = (
            CHILD_CONDITION_ENCODING
            if schema_version == CHILD_CONDITION_SCHEMA
            else CONDITION_ENCODING
        )
        if raw.get("condition_encoding", expected_encoding) != expected_encoding:
            raise ValueError("condition record encoding differs from the schema contract")
        od_gold = tuple(float(value) for value in raw["od_gold"])
        od_mask = tuple(int(value) for value in raw["od_mask"])
        if len(od_gold) != len(PLANNER_OD_IDS) or len(od_mask) != len(PLANNER_OD_IDS):
            raise ValueError("od_gold/od_mask must match the planner 18 layout")
        if any(value not in (0, 1) for value in od_mask):
            raise ValueError("od_mask must be binary")
        if not all(math.isfinite(value) for value in od_gold):
            raise ValueError("od_gold must be finite")
        source_space_group = int(raw["source_space_group"])
        parent_space_group = int(raw["parent_space_group"])
        if not 1 <= source_space_group <= SPACE_GROUP_COUNT:
            raise ValueError("source_space_group must lie in [1,230]")
        if not 1 <= parent_space_group <= SPACE_GROUP_COUNT:
            raise ValueError("parent_space_group must lie in [1,230]")
        repeat = tuple(
            tuple(int(item) for item in row) for row in raw["repeat_matrix"]
        )
        if len(repeat) != 3 or any(len(row) != 3 for row in repeat):
            raise ValueError("repeat_matrix must be 3x3")
        translations = tuple(
            tuple(int(item) for item in row) for row in raw["translations"]
        )
        ref_frac = _float_matrix(raw["ref_frac"], shape=(len(raw["ref_role_ids"]), 3), name="ref_frac")
        roles = tuple(int(value) for value in raw["ref_role_ids"])
        atom_types = tuple(int(value) for value in raw["ref_atom_types"])
        if len(roles) != len(atom_types) or len(roles) != len(ref_frac):
            raise ValueError("reference roles/types/positions must have equal length")
        if any(role not in (0, 1, 2) for role in roles):
            raise ValueError("reference role ids must lie in {0,1,2} (A/B/X)")
        site_rms = float(raw["alignment"]["site_rms_angstrom"])
        if not math.isfinite(site_rms) or site_rms < 0.0:
            raise ValueError("alignment site_rms_angstrom must be finite and non-negative")
        return FerroConditionRecord(
            pair_id=str(raw["pair_id"]),
            split=str(raw["split"]),
            parent_material_id=str(raw["parent_material_id"]),
            polar_material_id=str(raw["polar_material_id"]),
            source_space_group=source_space_group,
            parent_space_group=parent_space_group,
            od_gold=od_gold,
            od_mask=od_mask,
            repeat_matrix=repeat,
            translations=translations,
            ref_lattice=_float_matrix(raw["ref_lattice"], shape=(3, 3), name="ref_lattice"),
            ref_frac=ref_frac,
            ref_role_ids=roles,
            ref_atom_types=atom_types,
            site_rms_angstrom=site_rms,
            schema_version=schema_version,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid condition record at {path}:{line_number}: {exc}") from exc


def load_condition_records(
    path: str | Path, *, split: str | None = None
) -> tuple[FerroConditionRecord, ...]:
    path = Path(path)
    records: list[FerroConditionRecord] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = parse_condition_record(
                json.loads(line), path=path, line_number=line_number
            )
            if record.pair_id in seen:
                raise ValueError(f"duplicate condition pair_id {record.pair_id!r}")
            seen.add(record.pair_id)
            if split is None or record.split == split:
                records.append(record)
    return tuple(records)


@dataclass(frozen=True)
class FerroConditionBatch:
    """Packed per-pair condition tensors aligned with a PackedASUBatch."""

    od_gold: torch.Tensor
    od_mask: torch.Tensor
    ref_frac: torch.Tensor
    ref_lattice: torch.Tensor
    ref_ptr: torch.Tensor
    ref_role_ids: torch.Tensor
    repeat_matrix: torch.Tensor
    translations: torch.Tensor
    translation_ptr: torch.Tensor
    source_space_group: torch.Tensor
    site_rms_angstrom: torch.Tensor
    child_reference_frac: torch.Tensor
    child_reference_mask: torch.Tensor
    pair_ids: tuple[str, ...]
    schema_versions: tuple[str, ...]

    @property
    def batch_size(self) -> int:
        return len(self.pair_ids)

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> "FerroConditionBatch":
        return FerroConditionBatch(
            od_gold=self.od_gold.to(device=device, non_blocking=non_blocking),
            od_mask=self.od_mask.to(device=device, non_blocking=non_blocking),
            ref_frac=self.ref_frac.to(device=device, non_blocking=non_blocking),
            ref_lattice=self.ref_lattice.to(device=device, non_blocking=non_blocking),
            ref_ptr=self.ref_ptr.to(device=device, non_blocking=non_blocking),
            ref_role_ids=self.ref_role_ids.to(device=device, non_blocking=non_blocking),
            repeat_matrix=self.repeat_matrix.to(device=device, non_blocking=non_blocking),
            translations=self.translations.to(device=device, non_blocking=non_blocking),
            translation_ptr=self.translation_ptr.to(device=device, non_blocking=non_blocking),
            source_space_group=self.source_space_group.to(device=device, non_blocking=non_blocking),
            site_rms_angstrom=self.site_rms_angstrom.to(
                device=device, non_blocking=non_blocking
            ),
            child_reference_frac=self.child_reference_frac.to(
                device=device, non_blocking=non_blocking
            ),
            child_reference_mask=self.child_reference_mask.to(
                device=device, non_blocking=non_blocking
            ),
            pair_ids=self.pair_ids,
            schema_versions=self.schema_versions,
        )

    def condition_vector(self) -> torch.Tensor:
        """Return only inference-available categorical transition conditions."""

        space_group = torch.nn.functional.one_hot(
            self.source_space_group - 1, num_classes=SPACE_GROUP_COUNT
        ).to(self.od_gold.dtype)
        schemas = set(self.schema_versions)
        if schemas == {CONDITION_SCHEMA}:
            return space_group
        if schemas == {CHILD_CONDITION_SCHEMA}:
            return torch.cat((space_group, self.od_mask), dim=-1)
        raise ValueError("a condition batch cannot mix transition schemas")


def pack_condition_records(
    records: tuple[FerroConditionRecord, ...] | list[FerroConditionRecord],
    *,
    od_applicability_masks: list[torch.Tensor] | None = None,
    child_reference_fractional: list[torch.Tensor] | None = None,
    correspondence_max_site_rms_angstrom: float = (
        DEFAULT_CORRESPONDENCE_MAX_SITE_RMS_ANGSTROM
    ),
) -> FerroConditionBatch:
    if not records:
        raise ValueError("cannot pack an empty condition record list")
    ref_counts = [len(record.ref_frac) for record in records]
    translation_counts = [len(record.translations) for record in records]
    ref_ptr = [0]
    translation_ptr = [0]
    for count in ref_counts:
        ref_ptr.append(ref_ptr[-1] + count)
    for count in translation_counts:
        translation_ptr.append(translation_ptr[-1] + count)
    if correspondence_max_site_rms_angstrom <= 0.0:
        raise ValueError("correspondence RMS threshold must be positive")
    effective_masks = [list(record.od_mask) for record in records]
    if od_applicability_masks is not None:
        if len(od_applicability_masks) != len(records):
            raise ValueError("OD applicability masks must align with condition records")
        for mask, applicable in zip(effective_masks, od_applicability_masks):
            if tuple(applicable.shape) != (len(PLANNER_OD_IDS),):
                raise ValueError("OD applicability mask has the wrong shape")
            mask[:] = [int(left and bool(right)) for left, right in zip(mask, applicable)]
    correspondence_indices = tuple(PLANNER_OD_IDS.index(item) for item in CORRESPONDENCE_IDS)
    for record, mask in zip(records, effective_masks):
        if record.site_rms_angstrom > correspondence_max_site_rms_angstrom:
            for index in correspondence_indices:
                mask[index] = 0
    if child_reference_fractional is None:
        child_reference = torch.empty((0, 3), dtype=torch.float32)
        child_reference_mask = torch.empty((0,), dtype=torch.float32)
    else:
        if len(child_reference_fractional) != len(records):
            raise ValueError("child references must align with condition records")
        for value in child_reference_fractional:
            if value.ndim != 2 or value.shape[1] != 3:
                raise ValueError("each child reference must have shape [atoms,3]")
        child_reference = torch.cat(
            [value.to(dtype=torch.float32) for value in child_reference_fractional]
        )
        child_reference_mask = torch.ones(
            child_reference.shape[0], dtype=torch.float32
        )
    return FerroConditionBatch(
        od_gold=torch.tensor([record.od_gold for record in records], dtype=torch.float32),
        od_mask=torch.tensor(effective_masks, dtype=torch.float32),
        ref_frac=torch.tensor(
            [position for record in records for position in record.ref_frac],
            dtype=torch.float32,
        ),
        ref_lattice=torch.tensor(
            [record.ref_lattice for record in records], dtype=torch.float32
        ),
        ref_ptr=torch.tensor(ref_ptr, dtype=torch.long),
        ref_role_ids=torch.tensor(
            [role for record in records for role in record.ref_role_ids], dtype=torch.long
        ),
        repeat_matrix=torch.tensor(
            [record.repeat_matrix for record in records], dtype=torch.float32
        ),
        translations=torch.tensor(
            [shift for record in records for shift in record.translations],
            dtype=torch.float32,
        ),
        translation_ptr=torch.tensor(translation_ptr, dtype=torch.long),
        source_space_group=torch.tensor(
            [record.source_space_group for record in records], dtype=torch.long
        ),
        site_rms_angstrom=torch.tensor(
            [record.site_rms_angstrom for record in records], dtype=torch.float32
        ),
        child_reference_frac=child_reference,
        child_reference_mask=child_reference_mask,
        pair_ids=tuple(record.pair_id for record in records),
        schema_versions=tuple(record.schema_version for record in records),
    )


__all__ = [
    "ATOM_CONDITION_DIMENSION",
    "CHILD_CONDITION_DIMENSION",
    "CHILD_CONDITION_ENCODING",
    "CHILD_CONDITION_SCHEMA",
    "CONDITION_DIMENSION",
    "CONDITION_ENCODING",
    "CONDITION_MANIFEST_SCHEMA",
    "CONDITION_SCHEMA",
    "DEFAULT_CORRESPONDENCE_MAX_SITE_RMS_ANGSTROM",
    "SPACE_GROUP_COUNT",
    "FerroConditionBatch",
    "FerroConditionRecord",
    "load_condition_records",
    "pack_condition_records",
    "parse_condition_record",
    "verify_condition_asset",
]
