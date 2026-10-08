"""Verified maximal-supergroup graph reader and parent-channel legality checkers.

The graph asset (`assets/symmetry/supergroup_graph_v1`) is generated offline by
complete subgroup enumeration of the finite groups F_g = G/T_conv with spglib;
this module only loads the hash-verified asset, enumerates candidate parent
channels, and validates operation-subset and Wyckoff orbit-merge legality.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from polarevolve.crystal.contracts import ContractError
from polarevolve.crystal.symmetry import AssetProvenance

GRAPH_SCHEMA_VERSION = "supergroup_graph_v1"
EDGE_KINDS = frozenset({"t", "k_same_cell"})


def _fraction(value: Any) -> Fraction:
    if isinstance(value, bool):
        raise ContractError("boolean values are not rational coordinates")
    try:
        return Fraction(str(value))
    except (ValueError, ZeroDivisionError) as exc:
        raise ContractError(f"invalid rational value: {value!r}") from exc


@dataclass(frozen=True)
class SupergroupEdge:
    """One verified maximal-subgroup witness embedding (child into parent)."""

    child_hall: int
    child_sg: int
    parent_hall: int
    parent_sg: int
    kind: str
    index: int
    transform_parent_from_child: tuple[tuple[Fraction, ...], ...]
    origin_parent_from_child: tuple[Fraction, ...]


@dataclass(frozen=True)
class ParentSpaceGroupCandidate:
    """One group-level parent candidate ranked by shortest supergroup depth.

    This object proves only that a parent space-group type is reachable through
    the verified graph.  It deliberately carries no coordinates or Wyckoff
    program; those belong to the structure-channel owner in ``parent.py``.
    """

    space_group: int
    point_group: str
    graph_depth: int
    shortest_path: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "space_group": self.space_group,
            "point_group": self.point_group,
            "graph_depth": self.graph_depth,
            "shortest_path": list(self.shortest_path),
        }


@dataclass(frozen=True)
class CompiledSupergroupRelation:
    """One Hall-continuous path with an exact composed coordinate transform."""

    child_hall: int
    child_sg: int
    parent_hall: int
    parent_sg: int
    parent_point_group: str
    graph_depth: int
    group_index: int
    transform_parent_from_child: tuple[tuple[Fraction, ...], ...]
    origin_parent_from_child: tuple[Fraction, ...]
    edges: tuple[SupergroupEdge, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "child_hall": self.child_hall,
            "child_sg": self.child_sg,
            "parent_hall": self.parent_hall,
            "parent_sg": self.parent_sg,
            "parent_point_group": self.parent_point_group,
            "graph_depth": self.graph_depth,
            "group_index": self.group_index,
            "transform_parent_from_child": [
                [str(value) for value in row]
                for row in self.transform_parent_from_child
            ],
            "origin_parent_from_child": [
                str(value) for value in self.origin_parent_from_child
            ],
            "edge_kinds": [edge.kind for edge in self.edges],
            "hall_path": [self.child_hall]
            + [edge.parent_hall for edge in self.edges],
            "space_group_path": [self.child_sg]
            + [edge.parent_sg for edge in self.edges],
        }


_RATIONAL_IDENTITY = (
    (Fraction(1), Fraction(0), Fraction(0)),
    (Fraction(0), Fraction(1), Fraction(0)),
    (Fraction(0), Fraction(0), Fraction(1)),
)
_RATIONAL_ZERO = (Fraction(0), Fraction(0), Fraction(0))


def _compose_relation(
    transform: tuple[tuple[Fraction, ...], ...],
    origin: tuple[Fraction, ...],
    edge: SupergroupEdge,
) -> tuple[tuple[tuple[Fraction, ...], ...], tuple[Fraction, ...]]:
    edge_matrix = edge.transform_parent_from_child
    composed_matrix = tuple(
        tuple(
            sum(edge_matrix[row][inner] * transform[inner][column] for inner in range(3))
            for column in range(3)
        )
        for row in range(3)
    )
    composed_origin = tuple(
        sum(edge_matrix[row][inner] * origin[inner] for inner in range(3))
        + edge.origin_parent_from_child[row]
        for row in range(3)
    )
    return composed_matrix, composed_origin


class SupergroupGraph:
    """Read-only maximal-supergroup relations loaded from a verified asset root."""

    def __init__(self, root: str | Path) -> None:
        self.provenance = AssetProvenance.load(root)
        payload = self._payload(self.provenance.root / "supergroup_graph.json")
        if payload.get("schema_version") != GRAPH_SCHEMA_VERSION:
            raise ContractError("unsupported supergroup graph schema")
        edges: dict[int, list[SupergroupEdge]] = {}
        point_groups: dict[int, str] = {}
        for raw in payload.get("edges", ()):
            edge = self._edge(raw)
            for sg, pg in (
                (edge.child_sg, str(raw["child_point_group"])),
                (edge.parent_sg, str(raw["parent_point_group"])),
            ):
                previous = point_groups.setdefault(sg, pg)
                if previous != pg:
                    raise ContractError(f"inconsistent point group recorded for SG {sg}")
            edges.setdefault(edge.child_sg, []).append(edge)
        if not edges:
            raise ContractError("supergroup graph contains no edges")
        self._edges_by_child = {
            child_sg: tuple(items) for child_sg, items in edges.items()
        }
        edges_by_hall: dict[int, list[SupergroupEdge]] = {}
        space_group_by_hall: dict[int, int] = {}
        for items in edges.values():
            for edge in items:
                edges_by_hall.setdefault(edge.child_hall, []).append(edge)
                for hall, space_group in (
                    (edge.child_hall, edge.child_sg),
                    (edge.parent_hall, edge.parent_sg),
                ):
                    previous = space_group_by_hall.setdefault(hall, space_group)
                    if previous != space_group:
                        raise ContractError(f"Hall {hall} has inconsistent space groups")
        self._edges_by_child_hall = {
            hall: tuple(
                sorted(
                    items,
                    key=lambda edge: (
                        edge.parent_sg,
                        edge.parent_hall,
                        edge.kind,
                        edge.index,
                    ),
                )
            )
            for hall, items in edges_by_hall.items()
        }
        self._space_group_by_hall = space_group_by_hall
        self._point_groups = point_groups

    @staticmethod
    def _payload(path: Path) -> Mapping[str, Any]:
        import json

        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ContractError(f"cannot read JSON asset: {path}") from exc
        if not isinstance(value, Mapping):
            raise ContractError("supergroup graph root must be an object")
        return value

    @staticmethod
    def _edge(raw: Any) -> SupergroupEdge:
        if not isinstance(raw, Mapping):
            raise ContractError("supergroup edge entries must be objects")
        child_hall, parent_hall = int(raw["child_hall"]), int(raw["parent_hall"])
        child_sg, parent_sg = int(raw["child_sg"]), int(raw["parent_sg"])
        if not (1 <= child_hall <= 530 and 1 <= parent_hall <= 530):
            raise ContractError("supergroup edge Hall number out of range")
        if not (1 <= child_sg <= 230 and 1 <= parent_sg <= 230):
            raise ContractError("supergroup edge space-group number out of range")
        kind = str(raw["kind"])
        if kind not in EDGE_KINDS:
            raise ContractError(f"unknown supergroup edge kind: {kind}")
        index = int(raw["index"])
        if index < 2:
            raise ContractError("supergroup edge index must be at least 2")
        transform = tuple(
            tuple(_fraction(value) for value in row)
            for row in raw["transform_parent_from_child"]
        )
        if len(transform) != 3 or any(len(row) != 3 for row in transform):
            raise ContractError("supergroup edge transform must be 3x3")
        a, b, c = transform
        det = (
            a[0] * (b[1] * c[2] - b[2] * c[1])
            - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0])
        )
        if det <= 0:
            raise ContractError("supergroup edge transform must have positive determinant")
        origin = tuple(_fraction(value) for value in raw["origin_parent_from_child"])
        if len(origin) != 3:
            raise ContractError("supergroup edge origin must have three components")
        return SupergroupEdge(
            child_hall=child_hall,
            child_sg=child_sg,
            parent_hall=parent_hall,
            parent_sg=parent_sg,
            kind=kind,
            index=index,
            transform_parent_from_child=transform,
            origin_parent_from_child=origin,
        )

    def point_group(self, space_group: int) -> str:
        try:
            return self._point_groups[int(space_group)]
        except (KeyError, ValueError) as exc:
            raise ContractError(f"space group absent from graph: {space_group}") from exc

    def maximal_supergroup_edges(self, child_sg: int) -> tuple[SupergroupEdge, ...]:
        return self._edges_by_child.get(int(child_sg), ())

    def compiled_relations(
        self,
        child_hall: int,
        *,
        parent_space_group: int | None = None,
        nonpolar_only: bool = False,
        maximum_depth: int = 8,
        maximum_witnesses_per_hall: int = 1,
        maximum_extra_depth: int = 0,
    ) -> tuple[CompiledSupergroupRelation, ...]:
        """Compile shortest Hall-continuous paths into exact relation witnesses.

        The type-level ancestor search is sufficient for recall accounting but
        cannot be used to transform coordinates. This traversal retains Hall
        continuity and composes every basis/origin map with exact rationals.
        """

        child = int(child_hall)
        if child not in self._space_group_by_hall:
            raise ContractError(f"Hall number absent from graph: {child_hall}")
        if maximum_depth < 1:
            raise ContractError("maximum relation depth must be positive")
        if maximum_witnesses_per_hall < 1:
            raise ContractError("maximum relation witnesses per Hall must be positive")
        if maximum_extra_depth < 0:
            raise ContractError("maximum extra relation depth cannot be negative")
        target = None if parent_space_group is None else int(parent_space_group)
        if target is not None and not 1 <= target <= 230:
            raise ContractError(f"space group out of range: {parent_space_group}")

        start = (child, (), _RATIONAL_IDENTITY, _RATIONAL_ZERO, 1)
        queue = [start]
        minimum_depth = {child: 0}
        witness_keys = {child: {(_RATIONAL_IDENTITY, _RATIONAL_ZERO)}}
        witnesses = {child: [start]}
        cursor = 0
        while cursor < len(queue):
            hall, path, transform, origin, group_index = queue[cursor]
            cursor += 1
            if len(path) >= maximum_depth:
                continue
            for edge in self._edges_by_child_hall.get(hall, ()):
                if edge.parent_hall in {hall, *(item.child_hall for item in path)}:
                    continue
                composed = _compose_relation(transform, origin, edge)
                depth = len(path) + 1
                first_depth = minimum_depth.setdefault(edge.parent_hall, depth)
                if depth > first_depth + maximum_extra_depth:
                    continue
                key = (composed[0], tuple(value % 1 for value in composed[1]))
                keys = witness_keys.setdefault(edge.parent_hall, set())
                if key in keys:
                    continue
                items = witnesses.setdefault(edge.parent_hall, [])
                if len(items) >= maximum_witnesses_per_hall:
                    continue
                state = (
                    edge.parent_hall,
                    path + (edge,),
                    composed[0],
                    composed[1],
                    group_index * edge.index,
                )
                keys.add(key)
                items.append(state)
                queue.append(state)

        child_sg = self._space_group_by_hall[child]
        relations = []
        for parent_hall, items in witnesses.items():
            if parent_hall == child:
                continue
            for _, path, transform, origin, group_index in items:
                parent_sg = self._space_group_by_hall[parent_hall]
                point_group = self._point_groups[parent_sg]
                if target is not None and parent_sg != target:
                    continue
                if nonpolar_only and point_group in POLAR_POINT_GROUPS:
                    continue
                relations.append(
                    CompiledSupergroupRelation(
                        child_hall=child,
                        child_sg=child_sg,
                        parent_hall=parent_hall,
                        parent_sg=parent_sg,
                        parent_point_group=point_group,
                        graph_depth=len(path),
                        group_index=group_index,
                        transform_parent_from_child=transform,
                        origin_parent_from_child=origin,
                        edges=path,
                    )
                )
        return tuple(
            sorted(
                relations,
                key=lambda item: (
                    item.graph_depth,
                    item.parent_sg,
                    item.parent_hall,
                    item.transform_parent_from_child,
                    item.origin_parent_from_child,
                ),
            )
        )

    def ancestor_candidates(
        self, child_sg: int, *, nonpolar_only: bool = False
    ) -> tuple[ParentSpaceGroupCandidate, ...]:
        """Return deterministic shortest-depth supergroup candidates.

        The traversal is over space-group *types*.  Hall-setting witnesses are
        retained in the asset and validated independently, while coordinate-
        bearing channel construction remains a separate gate.
        """

        child = int(child_sg)
        if not 1 <= child <= 230:
            raise ContractError(f"space group out of range: {child_sg}")
        paths: dict[int, tuple[int, ...]] = {child: (child,)}
        queue = [child]
        cursor = 0
        while cursor < len(queue):
            node = queue[cursor]
            cursor += 1
            edges = sorted(
                self._edges_by_child.get(node, ()),
                key=lambda edge: (
                    edge.parent_sg,
                    edge.parent_hall,
                    edge.child_hall,
                    edge.kind,
                    edge.index,
                ),
            )
            for edge in edges:
                parent = edge.parent_sg
                if parent == child or parent in paths:
                    continue
                paths[parent] = paths[node] + (parent,)
                queue.append(parent)
        candidates = []
        for space_group, path in paths.items():
            if space_group == child:
                continue
            point_group = self._point_groups.get(space_group)
            if point_group is None:
                raise ContractError(f"space group absent from graph: {space_group}")
            if nonpolar_only and point_group in POLAR_POINT_GROUPS:
                continue
            candidates.append(
                ParentSpaceGroupCandidate(
                    space_group=space_group,
                    point_group=point_group,
                    graph_depth=len(path) - 1,
                    shortest_path=path,
                )
            )
        return tuple(
            sorted(candidates, key=lambda item: (item.graph_depth, item.space_group))
        )

    def ancestors(self, child_sg: int, *, nonpolar_only: bool = False) -> frozenset[int]:
        """All parent SGs reachable through maximal-supergroup chains."""

        return frozenset(
            candidate.space_group
            for candidate in self.ancestor_candidates(
                child_sg, nonpolar_only=nonpolar_only
            )
        )


POLAR_POINT_GROUPS = frozenset(
    {"1", "2", "m", "mm2", "4", "4mm", "3", "3m", "6", "6mm"}
)


def operation_subset_conjugated(
    child_rotations: Sequence[Sequence[Sequence[float]]],
    child_translations: Sequence[Sequence[float]],
    parent_rotations: Sequence[Sequence[Sequence[float]]],
    parent_translations: Sequence[Sequence[float]],
    transform_parent_from_child: Sequence[Sequence[float]],
    origin_parent_from_child: Sequence[float],
    *,
    tolerance: float = 1.0e-6,
) -> bool:
    """True when every child operation lands in the parent group under (M, o).

    The basis map is x_parent = M @ x_child + o in fractional coordinates, so a
    child operation (R, t) conjugates to R' = M R M^-1 with translation
    t' = M t + o - R' o; it must equal a parent operation modulo integer
    translations.
    """

    matrix = np.asarray(transform_parent_from_child, dtype=np.float64)
    origin = np.asarray(origin_parent_from_child, dtype=np.float64)
    if matrix.shape != (3, 3) or origin.shape != (3,):
        raise ContractError("conjugated subset check requires a 3x3 transform and 3-origin")
    parent_by_rotation: dict[tuple[int, ...], list[np.ndarray]] = {}
    for rotation, translation in zip(parent_rotations, parent_translations):
        key = tuple(np.rint(rotation).astype(int).flat)
        parent_by_rotation.setdefault(key, []).append(np.asarray(translation, dtype=np.float64))
    inverse = np.linalg.inv(matrix)
    for rotation, translation in zip(child_rotations, child_translations):
        conj_rotation = matrix @ np.asarray(rotation, dtype=np.float64) @ inverse
        if not np.allclose(conj_rotation, np.rint(conj_rotation), atol=tolerance, rtol=0.0):
            return False
        key = tuple(np.rint(conj_rotation).astype(int).flat)
        if key not in parent_by_rotation:
            return False
        conj_translation = (
            matrix @ np.asarray(translation, dtype=np.float64) + origin - conj_rotation @ origin
        )
        found = False
        for parent_translation in parent_by_rotation[key]:
            delta = (conj_translation - parent_translation) % 1.0
            if float(np.min(np.abs(np.minimum(delta, 1.0 - delta)))) <= tolerance and float(
                np.max(np.abs(delta - np.round(delta)))
            ) <= tolerance:
                found = True
                break
        if not found:
            return False
    return True


@dataclass(frozen=True)
class OccupiedOrbit:
    """One species-resolved occupied Wyckoff orbit on a structure's atom list."""

    species: int
    letter: str
    atom_indices: frozenset[int]


@dataclass(frozen=True)
class OrbitMerge:
    """One legal inverse-splitting row: child orbits merging into a parent orbit."""

    species: int
    child_letters: tuple[str, ...]
    parent_letter: str
    child_atom_count: int
    parent_atom_count: int


def check_orbit_merging(
    *,
    child_orbits: Sequence[OccupiedOrbit],
    parent_orbits: Sequence[OccupiedOrbit],
    atom_mapping: Mapping[int, int],
) -> tuple[OrbitMerge, ...]:
    """Validate that child orbits merge into parent orbits without residue.

    Every child orbit must map into exactly one parent orbit of the same
    species, and every parent orbit must be fully covered by mapped child
    atoms. The returned rows record the legal inverse-splitting pattern.
    """

    parent_of: dict[int, tuple[int, OccupiedOrbit]] = {}
    for parent_index, orbit in enumerate(parent_orbits):
        for atom in orbit.atom_indices:
            if atom in parent_of:
                raise ContractError("parent orbits must partition their atom list")
            parent_of[atom] = (parent_index, orbit)
    child_atoms: set[int] = set()
    for orbit in child_orbits:
        if child_atoms & orbit.atom_indices:
            raise ContractError("child orbits must partition their atom list")
        child_atoms |= set(orbit.atom_indices)
    missing = child_atoms - set(atom_mapping)
    if missing:
        raise ContractError("atom mapping does not cover every child atom")

    # Pass 1: resolve every child orbit to exactly one parent orbit of the same
    # species, and accumulate parent-atom coverage.
    parent_index_of_child: dict[int, int] = {}
    covered_parent_atoms: set[int] = set()
    for child_index, orbit in enumerate(child_orbits):
        targets = set()
        for atom in orbit.atom_indices:
            target = atom_mapping[atom]
            if target not in parent_of:
                raise ContractError("atom mapping targets an atom outside parent orbits")
            targets.add(parent_of[target][0])
        if len(targets) != 1:
            raise ContractError("child orbit splits across multiple parent orbits")
        (parent_index,) = targets
        if parent_orbits[parent_index].species != orbit.species:
            raise ContractError("orbit merging changes species")
        parent_index_of_child[child_index] = parent_index
        covered_parent_atoms |= {atom_mapping[atom] for atom in orbit.atom_indices}
    if covered_parent_atoms != set(parent_of):
        raise ContractError("parent orbits are not fully covered by child atoms")

    # Pass 2: aggregate merge rows per (species, parent letter).
    rows: dict[tuple[int, str], dict[str, Any]] = {}
    for child_index, orbit in enumerate(child_orbits):
        parent_orbit = parent_orbits[parent_index_of_child[child_index]]
        key = (orbit.species, parent_orbit.letter)
        row = rows.setdefault(
            key,
            {
                "child_letters": [],
                "child_atom_count": 0,
                "parent_atoms": set(),
            },
        )
        row["child_letters"].append(orbit.letter)
        row["child_atom_count"] += len(orbit.atom_indices)
        row["parent_atoms"] |= set(parent_orbit.atom_indices)
    return tuple(
        OrbitMerge(
            species=species,
            child_letters=tuple(row["child_letters"]),
            parent_letter=parent_letter,
            child_atom_count=row["child_atom_count"],
            parent_atom_count=len(row["parent_atoms"]),
        )
        for (species, parent_letter), row in sorted(rows.items())
    )


__all__ = [
    "EDGE_KINDS",
    "GRAPH_SCHEMA_VERSION",
    "CompiledSupergroupRelation",
    "OccupiedOrbit",
    "OrbitMerge",
    "ParentSpaceGroupCandidate",
    "POLAR_POINT_GROUPS",
    "SupergroupEdge",
    "SupergroupGraph",
    "check_orbit_merging",
    "operation_subset_conjugated",
]
