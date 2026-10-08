from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from ase.io import read as read_structure
from ase.neighborlist import neighbor_list


Graph = dict[str, Any]


@dataclass(frozen=True)
class GraphBuildConfig:
    r_max: float = 6.0
    max_neighbors: int = 64
    self_loops: bool = False
    k_angle: int = 12
    max_triplets_per_center: int = 200

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "GraphBuildConfig":
        fields = cls.__dataclass_fields__
        return cls(**{key: value for key, value in values.items() if key in fields})


def normalize_structure_id(value: object) -> str:
    text = os.path.basename(str(value).strip())
    lowered = text.lower()
    for suffix in (".cif.gz", ".cif"):
        if lowered.endswith(suffix):
            text = text[: -len(suffix)]
            break
    return text.strip()


def build_cif_index(structure_dir: str | Path) -> dict[str, Path]:
    root = Path(structure_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Structure directory does not exist: {root}")
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name.lower().endswith((".cif", ".cif.gz"))
    )
    index: dict[str, Path] = {}
    for path in paths:
        index.setdefault(normalize_structure_id(path.name).lower(), path)
    if not index:
        raise FileNotFoundError(f"No CIF files found under: {root}")
    return index


def resolve_cif_path(structure_id: object, index: dict[str, Path]) -> Path:
    key = normalize_structure_id(structure_id).lower()
    try:
        return index[key]
    except KeyError as exc:
        raise FileNotFoundError(
            f"No exact CIF filename match for structure id: {structure_id}"
        ) from exc


def _unique_rows(matrix: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(matrix)
    viewed = contiguous.view(
        np.dtype((np.void, contiguous.dtype.itemsize * contiguous.shape[1]))
    )
    _, first = np.unique(viewed, return_index=True)
    return np.sort(first)


def _empty_graph(
    z: np.ndarray,
    positions: np.ndarray,
    cell: np.ndarray,
    pbc: np.ndarray,
) -> Graph:
    return {
        "z": z,
        "pos": positions,
        "cell": cell,
        "pbc": pbc,
        "edge_index": np.zeros((2, 0), dtype=np.int64),
        "edge_vec": np.zeros((0, 3), dtype=np.float32),
        "edge_dist": np.zeros((0,), dtype=np.float32),
        "edge_shift": np.zeros((0, 3), dtype=np.float32),
        "triplet_center": np.zeros((0,), dtype=np.int64),
        "triplet_e1": np.zeros((0,), dtype=np.int64),
        "triplet_e2": np.zeros((0,), dtype=np.int64),
        "triplet_cos": np.zeros((0,), dtype=np.float32),
    }


def build_periodic_angle_graph(
    atoms: Any,
    config: GraphBuildConfig = GraphBuildConfig(),
) -> Graph:
    """Build a periodic neighbor graph with explicit angular triplets."""

    positions = atoms.get_positions().astype(np.float32)
    z = atoms.get_atomic_numbers().astype(np.int64)
    cell = atoms.get_cell().array.astype(np.float32)
    pbc = np.asarray(atoms.get_pbc(), dtype=np.bool_)
    i, j, shifts, _ = neighbor_list(
        "ijSd",
        atoms,
        cutoff=config.r_max,
        self_interaction=config.self_loops,
    )
    if len(i) == 0:
        return _empty_graph(z, positions, cell, pbc)

    i = i.astype(np.int64)
    j = j.astype(np.int64)
    shifts = shifts.astype(np.int64)
    cart_shifts = (shifts.astype(np.float32) @ cell).astype(np.float32)
    vectors = (positions[j] + cart_shifts - positions[i]).astype(np.float32)
    distances = np.linalg.norm(vectors, axis=1).astype(np.float32)

    if config.max_neighbors > 0:
        order = np.lexsort((distances, i))
        i, j = i[order], j[order]
        shifts, vectors, distances = (
            shifts[order],
            vectors[order],
            distances[order],
        )
        keep = np.zeros(len(i), dtype=bool)
        start = 0
        while start < len(i):
            end = start + 1
            while end < len(i) and i[end] == i[start]:
                end += 1
            keep[start : start + min(config.max_neighbors, end - start)] = True
            start = end
        i, j = i[keep], j[keep]
        shifts, vectors, distances = shifts[keep], vectors[keep], distances[keep]

    primary_edges = len(i)
    centers: list[np.ndarray] = []
    edge_1: list[np.ndarray] = []
    edge_2: list[np.ndarray] = []
    cosines: list[np.ndarray] = []
    if primary_edges >= 2:
        ordered = np.argsort(i, kind="mergesort")
        sorted_centers = i[ordered]
        start = 0
        while start < primary_edges:
            center = int(sorted_centers[start])
            end = start + 1
            while end < primary_edges and int(sorted_centers[end]) == center:
                end += 1
            selected_edges = ordered[start:end]
            if len(selected_edges) > config.k_angle:
                nearest = np.argsort(
                    distances[selected_edges],
                    kind="mergesort",
                )[: config.k_angle]
                selected_edges = selected_edges[nearest]
            if len(selected_edges) >= 2:
                unit = vectors[selected_edges] / (
                    distances[selected_edges, None] + 1e-12
                )
                cosine_matrix = (unit @ unit.T).astype(np.float32)
                first, second = np.triu_indices(len(selected_edges), k=1)
                first = first[: config.max_triplets_per_center]
                second = second[: config.max_triplets_per_center]
                centers.append(
                    np.full(len(first), center, dtype=np.int64)
                )
                edge_1.append(selected_edges[first].astype(np.int64))
                edge_2.append(selected_edges[second].astype(np.int64))
                cosines.append(cosine_matrix[first, second].astype(np.float32))
            start = end

    if centers:
        triplet_center = np.concatenate(centers)
        triplet_e1 = np.concatenate(edge_1)
        triplet_e2 = np.concatenate(edge_2)
        triplet_cos = np.concatenate(cosines)
    else:
        triplet_center = np.zeros(0, dtype=np.int64)
        triplet_e1 = np.zeros(0, dtype=np.int64)
        triplet_e2 = np.zeros(0, dtype=np.int64)
        triplet_cos = np.zeros(0, dtype=np.float32)

    i_all = np.concatenate([i, j]).astype(np.int64)
    j_all = np.concatenate([j, i]).astype(np.int64)
    shift_all = np.concatenate([shifts, -shifts]).astype(np.int64)
    vector_all = np.concatenate([vectors, -vectors]).astype(np.float32)
    distance_all = np.concatenate([distances, distances]).astype(np.float32)
    keys = np.stack(
        [
            i_all,
            j_all,
            shift_all[:, 0],
            shift_all[:, 1],
            shift_all[:, 2],
        ],
        axis=1,
    )
    first_unique = _unique_rows(keys)
    remap = -np.ones(len(i_all), dtype=np.int64)
    remap[first_unique] = np.arange(len(first_unique), dtype=np.int64)

    if triplet_e1.size:
        new_e1 = remap[triplet_e1]
        new_e2 = remap[triplet_e2]
        valid = (new_e1 >= 0) & (new_e2 >= 0)
        triplet_center = triplet_center[valid]
        triplet_e1 = new_e1[valid].astype(np.int64)
        triplet_e2 = new_e2[valid].astype(np.int64)
        triplet_cos = triplet_cos[valid].astype(np.float32)

    i_all = i_all[first_unique]
    j_all = j_all[first_unique]
    shift_all = shift_all[first_unique]
    graph = {
        "z": z,
        "pos": positions,
        "cell": cell,
        "pbc": pbc,
        "edge_index": np.stack([i_all, j_all]),
        "edge_vec": vector_all[first_unique],
        "edge_dist": distance_all[first_unique],
        "edge_shift": (shift_all.astype(np.float32) @ cell).astype(np.float32),
        "triplet_center": triplet_center,
        "triplet_e1": triplet_e1,
        "triplet_e2": triplet_e2,
        "triplet_cos": triplet_cos,
    }
    validate_graph(graph)
    return graph


def validate_graph(graph: Graph) -> None:
    nodes = len(graph["z"])
    edges = int(graph["edge_index"].shape[1])
    triplets = len(graph["triplet_cos"])
    if graph["edge_index"].shape != (2, edges):
        raise ValueError("edge_index must have shape (2, E)")
    if graph["edge_vec"].shape != (edges, 3):
        raise ValueError("edge_vec must have shape (E, 3)")
    if graph["edge_dist"].shape != (edges,):
        raise ValueError("edge_dist must have shape (E,)")
    if edges and (
        graph["edge_index"].min() < 0 or graph["edge_index"].max() >= nodes
    ):
        raise ValueError("edge_index contains an invalid node index")
    for key in ("triplet_center", "triplet_e1", "triplet_e2"):
        if graph[key].shape != (triplets,):
            raise ValueError(f"{key} is inconsistent with triplet_cos")
    if triplets and (
        graph["triplet_e1"].max() >= edges
        or graph["triplet_e2"].max() >= edges
    ):
        raise ValueError("Triplet edge index is out of bounds")


class GraphStore:
    """Resolve exact CIF filenames and keep graph caches outside source code."""

    def __init__(
        self,
        structure_dir: str | Path,
        cache_dir: str | Path,
        config: GraphBuildConfig = GraphBuildConfig(),
    ) -> None:
        self.structure_dir = Path(structure_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.index = build_cif_index(self.structure_dir)

    def cache_path(self, structure_id: object) -> Path:
        return self.cache_dir / f"{normalize_structure_id(structure_id)}.joblib"

    def get(self, structure_id: object) -> Graph:
        path = self.cache_path(structure_id)
        if path.is_file():
            graph = joblib.load(path)
            validate_graph(graph)
            return graph
        cif_path = resolve_cif_path(structure_id, self.index)
        graph = build_periodic_angle_graph(
            read_structure(cif_path),
            self.config,
        )
        graph["id"] = normalize_structure_id(structure_id)
        graph["natoms"] = int(len(graph["z"]))
        joblib.dump(graph, path, compress=0)
        return graph
