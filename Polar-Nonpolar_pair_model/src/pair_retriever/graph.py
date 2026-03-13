# -*- coding: utf-8 -*-
from __future__ import annotations

import glob
import os
import re
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from torch_geometric.data import Data


def norm_id(x: str) -> str:
    return str(x).strip().lower()


def build_cif_index(struct_dir: str) -> Dict[str, str]:
    if not os.path.isdir(struct_dir):
        raise ValueError(f"struct_dir does not exist: {struct_dir}")

    cifs = []
    for pat in ["**/*.cif", "**/*.CIF", "**/*.cif.gz", "**/*.CIF.gz"]:
        cifs += glob.glob(os.path.join(struct_dir, pat), recursive=True)

    if len(cifs) == 0:
        raise ValueError(f"No CIF files found under: {struct_dir}")

    pat_id = re.compile(r"(mp-\d+|mvc-\d+)", re.IGNORECASE)
    idx = {}
    for p in cifs:
        m = pat_id.search(os.path.basename(p))
        if m:
            idx.setdefault(m.group(1).lower(), p)

    if len(idx) == 0:
        raise ValueError("Cannot extract mp-xxxx or mvc-xxxx from CIF filenames.")
    return idx


def load_structure_from_index(mpid: str, cif_index: Dict[str, str]) -> Optional[Structure]:
    k = norm_id(mpid)
    path = cif_index.get(k)
    if path is None:
        return None
    try:
        return Structure.from_file(path)
    except Exception:
        return None


def safe_reduced_formula(s: Structure) -> Optional[str]:
    try:
        return s.composition.reduced_formula
    except Exception:
        return None


def calc_opcount(s: Structure, symprec: float) -> Optional[int]:
    try:
        sga = SpacegroupAnalyzer(s, symprec=float(symprec))
        ds = sga.get_symmetry_dataset()
        if ds is not None and "rotations" in ds:
            return int(len(ds["rotations"]))
        ops = sga.get_symmetry_operations(cartesian=False)
        return int(len(ops))
    except Exception:
        return None


def _parse_neighbor_list(ret):
    arrs = [np.asarray(x) for x in ret]
    int_1d = [a for a in arrs if a.ndim == 1 and np.issubdtype(a.dtype, np.integer)]
    float_1d = [a for a in arrs if a.ndim == 1 and np.issubdtype(a.dtype, np.floating)]

    if len(int_1d) >= 2 and len(float_1d) >= 1:
        i_idx = int_1d[0].astype(np.int64, copy=False).reshape(-1)
        j_idx = int_1d[1].astype(np.int64, copy=False).reshape(-1)
        dists = float_1d[0].astype(np.float32, copy=False).reshape(-1)
        m = min(len(i_idx), len(j_idx), len(dists))
        return i_idx[:m], j_idx[:m], dists[:m]

    raise ValueError(
        f"Cannot parse get_neighbor_list return shapes/dtypes: "
        f"{[(a.shape, str(a.dtype)) for a in arrs]}"
    )


def structure_to_graph(s: Structure, cutoff: float, max_neighbors: int) -> Data:
    coords = np.asarray([site.coords for site in s.sites], dtype=np.float32)
    z = np.asarray([int(site.specie.Z) for site in s.sites], dtype=np.int64)

    try:
        ret = s.get_neighbor_list(r=cutoff, numerical_tol=1e-8)
        i_idx, j_idx, dists = _parse_neighbor_list(ret)
    except Exception:
        i_idx, j_idx, dists = [], [], []
        for i, site in enumerate(s.sites):
            nbrs = s.get_neighbors(site, cutoff)
            for nb in nbrs:
                j = int(getattr(nb, "index", -1))
                if j >= 0:
                    i_idx.append(i)
                    j_idx.append(j)
                    dists.append(float(nb.distance))
        i_idx = np.asarray(i_idx, dtype=np.int64)
        j_idx = np.asarray(j_idx, dtype=np.int64)
        dists = np.asarray(dists, dtype=np.float32)

    if dists.size == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 1), dtype=torch.float32)
    else:
        order = np.lexsort((dists, i_idx))
        i_idx = i_idx[order]
        j_idx = j_idx[order]
        dists = dists[order]

        kept_i, kept_j, kept_d = [], [], []
        cnt = {}
        for i, j, d in zip(i_idx, j_idx, dists):
            c = cnt.get(int(i), 0)
            if c >= max_neighbors:
                continue
            cnt[int(i)] = c + 1
            kept_i.append(int(i))
            kept_j.append(int(j))
            kept_d.append(float(d))

        ii = np.array(kept_i + kept_j, dtype=np.int64)
        jj = np.array(kept_j + kept_i, dtype=np.int64)
        dd = np.array(kept_d + kept_d, dtype=np.float32)

        edge_index = torch.tensor(np.vstack([ii, jj]), dtype=torch.long)
        edge_attr = torch.tensor(dd[:, None], dtype=torch.float32)

    return Data(
        z=torch.tensor(z, dtype=torch.long),
        pos=torch.tensor(coords, dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=int(len(z)),
    )


class GraphCache:
    def __init__(self, cache_dir: str, cutoff: float, max_neighbors: int):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cutoff = float(cutoff)
        self.max_neighbors = int(max_neighbors)

    def get(self, mpid: str, cif_index: Dict[str, str]) -> Optional[Data]:
        k = norm_id(mpid)
        cache_path = self.cache_dir / f"{k}.pt"

        if cache_path.is_file():
            try:
                return torch.load(cache_path, map_location="cpu")
            except Exception:
                pass

        s = load_structure_from_index(k, cif_index)
        if s is None:
            return None

        g = structure_to_graph(s, cutoff=self.cutoff, max_neighbors=self.max_neighbors)
        torch.save(g, cache_path)
        return g