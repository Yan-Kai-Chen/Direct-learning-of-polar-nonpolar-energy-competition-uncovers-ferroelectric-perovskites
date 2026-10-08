"""Versioned FE2 group-relation ranking records and verified asset loading."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import spglib

from polarevolve.crystal.supergroup import CompiledSupergroupRelation
from polarevolve.crystal.symmetry import sha256_file

RELATION_RECORD_SCHEMA = "fe2_relation_record_v1"
RELATION_MANIFEST_SCHEMA = "fe2_relation_manifest_v1"
_SPLITS = ("train", "val", "test")


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(value: object, *, name: str) -> str:
    digest = str(value).lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"{name} must contain 64 lowercase hexadecimal digits")
    return digest


def _positive_int(value: object, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class RelationCandidateRecord:
    """One exact group-level relation representative for a candidate parent SG."""

    candidate_id: str
    parent_space_group: int
    parent_hall_number: int
    parent_point_group: str
    graph_depth: int
    group_index: int
    hall_path: tuple[int, ...]
    transform_parent_from_child: tuple[tuple[str, str, str], ...]
    origin_parent_from_child: tuple[str, str, str]
    edge_kinds: tuple[str, ...]
    child_operation_count: int
    parent_operation_count: int

    def __post_init__(self) -> None:
        if not self.candidate_id or not self.parent_point_group:
            raise ValueError("relation candidate identities must be non-empty")
        if not 1 <= self.parent_space_group <= 230:
            raise ValueError("candidate parent space group must lie in [1,230]")
        if not 1 <= self.parent_hall_number <= 530:
            raise ValueError("candidate parent Hall number must lie in [1,530]")
        _positive_int(self.graph_depth, name="graph_depth")
        _positive_int(self.group_index, name="group_index")
        if len(self.hall_path) != self.graph_depth + 1:
            raise ValueError("candidate Hall path length contradicts graph depth")
        if self.hall_path[-1] != self.parent_hall_number:
            raise ValueError("candidate Hall path does not terminate at its parent")
        if len(self.transform_parent_from_child) != 3 or any(
            len(row) != 3 for row in self.transform_parent_from_child
        ):
            raise ValueError("candidate transform must be a rational 3x3 matrix")
        if len(self.origin_parent_from_child) != 3:
            raise ValueError("candidate origin must contain three rational values")
        if len(self.edge_kinds) != self.graph_depth:
            raise ValueError("candidate edge kinds must follow the complete path")
        _positive_int(self.child_operation_count, name="child_operation_count")
        _positive_int(self.parent_operation_count, name="parent_operation_count")

    @classmethod
    def from_relation(
        cls, relation: CompiledSupergroupRelation
    ) -> "RelationCandidateRecord":
        child = spglib.get_symmetry_from_database(relation.child_hall)
        parent = spglib.get_symmetry_from_database(relation.parent_hall)
        if child is None or parent is None:
            raise ValueError("relation endpoint operation table is unavailable")
        payload = relation.to_dict()
        return cls(
            candidate_id=f"fe2-group-relation-{_canonical_sha256(payload)[:16]}",
            parent_space_group=relation.parent_sg,
            parent_hall_number=relation.parent_hall,
            parent_point_group=relation.parent_point_group,
            graph_depth=relation.graph_depth,
            group_index=relation.group_index,
            hall_path=(relation.child_hall, *(edge.parent_hall for edge in relation.edges)),
            transform_parent_from_child=tuple(
                tuple(str(value) for value in row)
                for row in relation.transform_parent_from_child
            ),
            origin_parent_from_child=tuple(
                str(value) for value in relation.origin_parent_from_child
            ),
            edge_kinds=tuple(edge.kind for edge in relation.edges),
            child_operation_count=len(child["rotations"]),
            parent_operation_count=len(parent["rotations"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "parent_space_group": self.parent_space_group,
            "parent_hall_number": self.parent_hall_number,
            "parent_point_group": self.parent_point_group,
            "relation": {
                "graph_depth": self.graph_depth,
                "group_index": self.group_index,
                "hall_path": list(self.hall_path),
                "transform_parent_from_child": [
                    list(row) for row in self.transform_parent_from_child
                ],
                "origin_parent_from_child": list(self.origin_parent_from_child),
                "edge_kinds": list(self.edge_kinds),
            },
            "operation_counts": {
                "child": self.child_operation_count,
                "parent": self.parent_operation_count,
            },
        }


@dataclass(frozen=True)
class RelationTrainingRecord:
    """One child with gold-free group candidates and a later-attached parent label."""

    pair_id: str
    split: str
    formula: str
    polar_material_id: str
    nonpolar_material_id: str
    source_record_index: int
    source_structure_sha256: str
    child_space_group: int
    child_hall_number: int
    prototype_key: str
    candidates: tuple[RelationCandidateRecord, ...]
    positive_candidate_index: int
    supervision_parent_space_group: int

    def __post_init__(self) -> None:
        if self.split not in _SPLITS:
            raise ValueError("relation record split must be train, val, or test")
        if not all(
            value
            for value in (
                self.pair_id,
                self.formula,
                self.polar_material_id,
                self.nonpolar_material_id,
                self.prototype_key,
            )
        ):
            raise ValueError("relation record identities must be non-empty")
        if self.source_record_index < 0:
            raise ValueError("source record index must be non-negative")
        _sha256(self.source_structure_sha256, name="source_structure_sha256")
        if not 1 <= self.child_space_group <= 230 or not 1 <= self.child_hall_number <= 530:
            raise ValueError("relation record child symmetry identity is invalid")
        if not self.candidates:
            raise ValueError("relation record must contain group-legal candidates")
        parent_groups = [item.parent_space_group for item in self.candidates]
        if len(parent_groups) != len(set(parent_groups)):
            raise ValueError("relation record must keep at most one relation per parent SG")
        if not 0 <= self.positive_candidate_index < len(self.candidates):
            raise ValueError("positive candidate index lies outside the candidate set")
        positive = self.candidates[self.positive_candidate_index]
        if positive.parent_space_group != self.supervision_parent_space_group:
            raise ValueError("positive candidate does not match the supervision parent SG")

    @property
    def hard_negative_count(self) -> int:
        return len(self.candidates) - 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": RELATION_RECORD_SCHEMA,
            "pair_id": self.pair_id,
            "split": self.split,
            "formula": self.formula,
            "polar_material_id": self.polar_material_id,
            "nonpolar_material_id": self.nonpolar_material_id,
            "source_record_index": self.source_record_index,
            "source_structure_sha256": self.source_structure_sha256,
            "child": {
                "space_group": self.child_space_group,
                "hall_number": self.child_hall_number,
                "prototype_key": self.prototype_key,
            },
            "candidates": [item.to_dict() for item in self.candidates],
            "supervision": {
                "positive_candidate_index": self.positive_candidate_index,
                "parent_space_group": self.supervision_parent_space_group,
                "hard_negative_count": self.hard_negative_count,
            },
        }


def parse_relation_candidate(raw: Mapping[str, Any]) -> RelationCandidateRecord:
    relation = raw["relation"]
    operations = raw["operation_counts"]
    return RelationCandidateRecord(
        candidate_id=str(raw["candidate_id"]),
        parent_space_group=int(raw["parent_space_group"]),
        parent_hall_number=int(raw["parent_hall_number"]),
        parent_point_group=str(raw["parent_point_group"]),
        graph_depth=int(relation["graph_depth"]),
        group_index=int(relation["group_index"]),
        hall_path=tuple(int(value) for value in relation["hall_path"]),
        transform_parent_from_child=tuple(
            tuple(str(value) for value in row)
            for row in relation["transform_parent_from_child"]
        ),
        origin_parent_from_child=tuple(
            str(value) for value in relation["origin_parent_from_child"]
        ),
        edge_kinds=tuple(str(value) for value in relation["edge_kinds"]),
        child_operation_count=int(operations["child"]),
        parent_operation_count=int(operations["parent"]),
    )


def parse_relation_record(raw: Mapping[str, Any]) -> RelationTrainingRecord:
    if raw.get("schema_version") != RELATION_RECORD_SCHEMA:
        raise ValueError("unsupported FE2 relation-record schema")
    child = raw["child"]
    supervision = raw["supervision"]
    return RelationTrainingRecord(
        pair_id=str(raw["pair_id"]),
        split=str(raw["split"]),
        formula=str(raw["formula"]),
        polar_material_id=str(raw["polar_material_id"]),
        nonpolar_material_id=str(raw["nonpolar_material_id"]),
        source_record_index=int(raw["source_record_index"]),
        source_structure_sha256=str(raw["source_structure_sha256"]),
        child_space_group=int(child["space_group"]),
        child_hall_number=int(child["hall_number"]),
        prototype_key=str(child["prototype_key"]),
        candidates=tuple(parse_relation_candidate(item) for item in raw["candidates"]),
        positive_candidate_index=int(supervision["positive_candidate_index"]),
        supervision_parent_space_group=int(supervision["parent_space_group"]),
    )


def assign_relation_splits(
    rows: Sequence[tuple[str, str, str]], *, seed: int
) -> dict[str, str]:
    """Assign formula/prototype connected components to deterministic splits."""

    identifiers = [row[0] for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("relation split input contains duplicate record IDs")
    parent = list(range(len(rows)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    owners: dict[tuple[str, str], int] = {}
    for index, (_, formula, prototype) in enumerate(rows):
        for key in (("formula", formula), ("prototype", prototype)):
            if key in owners:
                union(index, owners[key])
            else:
                owners[key] = index
    components: dict[int, list[str]] = {}
    for index, identifier in enumerate(identifiers):
        components.setdefault(find(index), []).append(identifier)
    ordered = sorted(
        components.values(),
        key=lambda values: (
            -len(values),
            hashlib.sha256(f"{seed}|{'|'.join(sorted(values))}".encode()).hexdigest(),
        ),
    )
    targets = {
        "train": 0.70 * len(rows),
        "val": 0.15 * len(rows),
        "test": 0.15 * len(rows),
    }
    counts = dict.fromkeys(_SPLITS, 0)
    assigned: dict[str, str] = {}
    for component in ordered:
        split = max(
            _SPLITS,
            key=lambda name: (
                (targets[name] - counts[name]) / max(targets[name], 1.0),
                -_SPLITS.index(name),
            ),
        )
        counts[split] += len(component)
        assigned.update({identifier: split for identifier in component})
    return assigned


def load_relation_records(
    root: str | Path, *, split: str
) -> tuple[RelationTrainingRecord, ...]:
    """Load one split after verifying the manifest and JSONL artifact hash."""

    if split not in _SPLITS:
        raise ValueError("relation-record split must be train, val, or test")
    root = Path(root).resolve()
    manifest = json.loads((root / "RELATION_MANIFEST.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != RELATION_MANIFEST_SCHEMA:
        raise ValueError("unsupported FE2 relation manifest schema")
    if manifest.get("record_schema") != RELATION_RECORD_SCHEMA:
        raise ValueError("FE2 relation manifest and record schemas disagree")
    artifact = manifest.get("artifacts", {}).get("records.jsonl", {})
    path = root / "records.jsonl"
    if _sha256(artifact.get("sha256"), name="records artifact SHA-256") != sha256_file(path):
        raise ValueError("FE2 relation record artifact hash mismatch")
    records = tuple(
        parse_relation_record(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if len(records) != int(artifact.get("records", -1)):
        raise ValueError("FE2 relation manifest record count mismatch")
    identifiers = [record.pair_id for record in records]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("FE2 relation asset contains duplicate pair IDs")
    return tuple(record for record in records if record.split == split)


__all__ = [
    "RELATION_MANIFEST_SCHEMA",
    "RELATION_RECORD_SCHEMA",
    "RelationCandidateRecord",
    "RelationTrainingRecord",
    "assign_relation_splits",
    "load_relation_records",
    "parse_relation_candidate",
    "parse_relation_record",
]
