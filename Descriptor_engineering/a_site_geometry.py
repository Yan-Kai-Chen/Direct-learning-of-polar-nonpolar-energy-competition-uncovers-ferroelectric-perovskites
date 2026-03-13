# -*- coding: utf-8 -*-
"""
Public-safe A-site geometry descriptor module.

This version exposes only a minimal, interpretable A-site local-geometry workflow:
- load structure
- identify A and X species
- collect local A-centered X environment
- compute simple geometric summary statistics
- write polar / npolar / delta descriptors

Detailed neighbor-shell rules and internal thresholds are intentionally abstracted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pymatgen.core import Structure


# =========================================================
# Basic utilities
# =========================================================
def _safe_norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def _as_array(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def _nan_stats(arr: List[float], prefix: str) -> Dict[str, float]:
    a = np.asarray(arr, dtype=float)
    if a.size == 0:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
        }
    return {
        f"{prefix}_mean": float(np.nanmean(a)),
        f"{prefix}_std": float(np.nanstd(a)),
        f"{prefix}_min": float(np.nanmin(a)),
        f"{prefix}_max": float(np.nanmax(a)),
    }


def _normalize_mpid(x: Any) -> str:
    return str(x).strip().lower()


def _build_cif_index(structure_dir: Path, suffixes=(".cif", ".CIF")) -> Dict[str, Path]:
    idx = {}
    for p in structure_dir.rglob("*"):
        if p.suffix in suffixes:
            idx[p.stem.strip().lower()] = p
    return idx


def _load_structure(mpid: str, cif_index: Dict[str, Path]) -> Optional[Structure]:
    key = _normalize_mpid(mpid)
    path = cif_index.get(key)
    if path is None:
        return None
    try:
        return Structure.from_file(str(path))
    except Exception:
        return None


# =========================================================
# Public-safe A-site geometry calculation
# =========================================================
def _collect_A_and_X_sites(struct: Structure, A_symbol: str, X_symbol: str) -> Tuple[List[int], List[int]]:
    a_idx, x_idx = [], []
    for i, site in enumerate(struct.sites):
        sp = str(site.specie)
        if sp == A_symbol:
            a_idx.append(i)
        elif sp == X_symbol:
            x_idx.append(i)
    return a_idx, x_idx


def _get_local_X_neighbors(
    struct: Structure,
    a_index: int,
    X_symbol: str,
    rules: Dict[str, Any],
) -> List[Tuple[int, float]]:
    """
    Public-safe simplified local-neighbor collection.

    We deliberately keep only a minimal interface here:
    - radius-based neighbor search
    - filter by X species
    - keep nearest few according to public placeholder settings
    """
    search_radius = float(rules.get("public_search_radius", 4.0))
    max_neighbors = int(rules.get("public_max_neighbors", 12))

    site = struct[a_index]
    neighs = struct.get_neighbors(site, search_radius)

    out = []
    for nb in neighs:
        j = int(nb.index)
        sp = str(nb.specie)
        if sp != X_symbol:
            continue
        out.append((j, float(nb.nn_distance)))

    out = sorted(out, key=lambda x: x[1])
    return out[:max_neighbors]


def _compute_A_geometry_for_structure(
    struct: Structure,
    A_symbol: str,
    X_symbol: str,
    rules: Dict[str, Any],
) -> Dict[str, float]:
    """
    Compute a small public-safe feature set:
    - A_offcenter_mean/max
    - AX_dist_mean/std/min/max
    - A_CN_mean
    - A_cage_asph_mean (simple radial spread proxy)
    """
    a_idx, _ = _collect_A_and_X_sites(struct, A_symbol, X_symbol)

    if len(a_idx) == 0:
        return {
            "A_offcenter_mean": np.nan,
            "A_offcenter_max": np.nan,
            "AX_dist_mean": np.nan,
            "AX_dist_std": np.nan,
            "AX_dist_min": np.nan,
            "AX_dist_max": np.nan,
            "A_CN_mean": np.nan,
            "A_cage_asph_mean": np.nan,
        }

    all_offcenter = []
    all_cn = []
    all_asph = []
    all_dists = []

    for i in a_idx:
        neighs = _get_local_X_neighbors(struct, i, X_symbol, rules)
        if len(neighs) == 0:
            continue

        A_pos = _as_array(struct[i].coords)
        X_pos = np.vstack([_as_array(struct[j].coords) for j, _ in neighs])
        dists = np.asarray([d for _, d in neighs], dtype=float)

        cage_center = X_pos.mean(axis=0)
        offcenter = _safe_norm(A_pos - cage_center)

        # simple asphericity proxy from radial dispersion
        asph = float(np.std(dists)) if dists.size > 0 else np.nan

        all_offcenter.append(offcenter)
        all_cn.append(float(len(neighs)))
        all_asph.append(asph)
        all_dists.extend(dists.tolist())

    if len(all_offcenter) == 0:
        return {
            "A_offcenter_mean": np.nan,
            "A_offcenter_max": np.nan,
            "AX_dist_mean": np.nan,
            "AX_dist_std": np.nan,
            "AX_dist_min": np.nan,
            "AX_dist_max": np.nan,
            "A_CN_mean": np.nan,
            "A_cage_asph_mean": np.nan,
        }

    return {
        "A_offcenter_mean": float(np.nanmean(all_offcenter)),
        "A_offcenter_max": float(np.nanmax(all_offcenter)),
        "AX_dist_mean": float(np.nanmean(all_dists)) if len(all_dists) else np.nan,
        "AX_dist_std": float(np.nanstd(all_dists)) if len(all_dists) else np.nan,
        "AX_dist_min": float(np.nanmin(all_dists)) if len(all_dists) else np.nan,
        "AX_dist_max": float(np.nanmax(all_dists)) if len(all_dists) else np.nan,
        "A_CN_mean": float(np.nanmean(all_cn)),
        "A_cage_asph_mean": float(np.nanmean(all_asph)),
    }


def _prefix_feature_dict(d: Dict[str, float], prefix: str) -> Dict[str, float]:
    return {f"{prefix}{k}": v for k, v in d.items()}


def _delta_feature_dict(
    polar_d: Dict[str, float],
    npolar_d: Dict[str, float],
    delta_prefix: str = "d_",
) -> Dict[str, float]:
    out = {}
    keys = sorted(set(polar_d.keys()) & set(npolar_d.keys()))
    for k in keys:
        pk = polar_d[k]
        nk = npolar_d[k]
        name = k.replace("polar_", delta_prefix, 1) if k.startswith("polar_") else f"{delta_prefix}{k}"
        try:
            out[name] = float(pk) - float(nk)
        except Exception:
            out[name] = np.nan
    return out


# =========================================================
# Public API
# =========================================================
def run_a_site_geometry_stage(
    df: pd.DataFrame,
    structure_dir: Path,
    rules: Dict[str, Any],
) -> pd.DataFrame:
    out = df.copy()
    structure_dir = Path(structure_dir)
    cif_index = _build_cif_index(structure_dir)

    required_cols = rules.get(
        "required_input_cols",
        ["Polar_mpid", "NPolar_mpid", "$A_{site}$", "$X_{site}$"],
    )
    for c in required_cols:
        if c not in out.columns:
            raise ValueError(f"Missing required column for A-site geometry stage: {c}")

    polar_feat_rows = []
    npolar_feat_rows = []
    delta_feat_rows = []

    strict_mode = bool(rules.get("strict_mode", False))

    for row in out.itertuples(index=False):
        polar_mpid = getattr(row, "Polar_mpid")
        npolar_mpid = getattr(row, "NPolar_mpid")
        A_symbol = getattr(row, "$A_{site}$")
        X_symbol = getattr(row, "$X_{site}$")

        p_struct = _load_structure(polar_mpid, cif_index)
        n_struct = _load_structure(npolar_mpid, cif_index)

        if p_struct is None or n_struct is None:
            if strict_mode:
                raise RuntimeError(
                    f"Cannot load structure for pair ({polar_mpid}, {npolar_mpid})."
                )
            p_feat = {
                "polar_A_offcenter_mean": np.nan,
                "polar_A_offcenter_max": np.nan,
                "polar_AX_dist_mean": np.nan,
                "polar_AX_dist_std": np.nan,
                "polar_AX_dist_min": np.nan,
                "polar_AX_dist_max": np.nan,
                "polar_A_CN_mean": np.nan,
                "polar_A_cage_asph_mean": np.nan,
            }
            n_feat = {
                "npolar_A_offcenter_mean": np.nan,
                "npolar_A_offcenter_max": np.nan,
                "npolar_AX_dist_mean": np.nan,
                "npolar_AX_dist_std": np.nan,
                "npolar_AX_dist_min": np.nan,
                "npolar_AX_dist_max": np.nan,
                "npolar_A_CN_mean": np.nan,
                "npolar_A_cage_asph_mean": np.nan,
            }
            d_feat = {
                "d_A_offcenter_mean": np.nan,
                "d_A_offcenter_max": np.nan,
                "d_AX_dist_mean": np.nan,
                "d_AX_dist_std": np.nan,
                "d_AX_dist_min": np.nan,
                "d_AX_dist_max": np.nan,
                "d_A_CN_mean": np.nan,
                "d_A_cage_asph_mean": np.nan,
            }
        else:
            p_raw = _compute_A_geometry_for_structure(p_struct, str(A_symbol), str(X_symbol), rules)
            n_raw = _compute_A_geometry_for_structure(n_struct, str(A_symbol), str(X_symbol), rules)

            p_feat = _prefix_feature_dict(p_raw, "polar_")
            n_feat = _prefix_feature_dict(n_raw, "npolar_")
            d_feat = _delta_feature_dict(p_feat, n_feat, delta_prefix="d_")

        polar_feat_rows.append(p_feat)
        npolar_feat_rows.append(n_feat)
        delta_feat_rows.append(d_feat)

    out = pd.concat(
        [
            out.reset_index(drop=True),
            pd.DataFrame(polar_feat_rows),
            pd.DataFrame(npolar_feat_rows),
            pd.DataFrame(delta_feat_rows),
        ],
        axis=1,
    )

    return out