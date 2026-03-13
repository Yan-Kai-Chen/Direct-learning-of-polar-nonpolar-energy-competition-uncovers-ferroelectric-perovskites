# -*- coding: utf-8 -*-
"""
Public-safe B-site geometry / off-centering descriptor module.

This version exposes a minimal, interpretable B-site local-geometry workflow:
- load structure
- identify B and X species
- collect local B-centered X environment
- compute simple geometric summary statistics
- write polar / npolar / delta descriptors

Detailed octahedral selection rules and internal thresholds are intentionally abstracted.
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
# Public-safe B-site geometry calculation
# =========================================================
def _collect_B_and_X_sites(struct: Structure, B_symbol: str, X_symbol: str) -> Tuple[List[int], List[int]]:
    b_idx, x_idx = [], []
    for i, site in enumerate(struct.sites):
        sp = str(site.specie)
        if sp == B_symbol:
            b_idx.append(i)
        elif sp == X_symbol:
            x_idx.append(i)
    return b_idx, x_idx


def _get_local_X_neighbors(
    struct: Structure,
    b_index: int,
    X_symbol: str,
    rules: Dict[str, Any],
) -> List[Tuple[int, float]]:
    """
    Public-safe simplified local-neighbor collection for B-site.
    """
    search_radius = float(rules.get("public_search_radius", 3.2))
    max_neighbors = int(rules.get("public_max_neighbors", 8))

    site = struct[b_index]
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


def _compute_B_geometry_for_structure(
    struct: Structure,
    B_symbol: str,
    X_symbol: str,
    rules: Dict[str, Any],
) -> Dict[str, float]:
    """
    Compute a small public-safe feature set:
    - B_offcenter_mean/max
    - B_offcenter_frac_gt_public
    - BX_dist_mean/std/min/max
    - B_CN_mean
    """
    b_idx, _ = _collect_B_and_X_sites(struct, B_symbol, X_symbol)

    if len(b_idx) == 0:
        return {
            "B_offcenter_mean": np.nan,
            "B_offcenter_max": np.nan,
            "B_offcenter_frac_gt_public": np.nan,
            "BX_dist_mean": np.nan,
            "BX_dist_std": np.nan,
            "BX_dist_min": np.nan,
            "BX_dist_max": np.nan,
            "B_CN_mean": np.nan,
        }

    all_offcenter = []
    all_cn = []
    all_dists = []

    frac_threshold = float(rules.get("public_offcenter_threshold", 0.10))

    for i in b_idx:
        neighs = _get_local_X_neighbors(struct, i, X_symbol, rules)
        if len(neighs) == 0:
            continue

        B_pos = _as_array(struct[i].coords)
        X_pos = np.vstack([_as_array(struct[j].coords) for j, _ in neighs])
        dists = np.asarray([d for _, d in neighs], dtype=float)

        cage_center = X_pos.mean(axis=0)
        offcenter = _safe_norm(B_pos - cage_center)

        all_offcenter.append(offcenter)
        all_cn.append(float(len(neighs)))
        all_dists.extend(dists.tolist())

    if len(all_offcenter) == 0:
        return {
            "B_offcenter_mean": np.nan,
            "B_offcenter_max": np.nan,
            "B_offcenter_frac_gt_public": np.nan,
            "BX_dist_mean": np.nan,
            "BX_dist_std": np.nan,
            "BX_dist_min": np.nan,
            "BX_dist_max": np.nan,
            "B_CN_mean": np.nan,
        }

    all_offcenter = np.asarray(all_offcenter, dtype=float)
    all_dists = np.asarray(all_dists, dtype=float) if len(all_dists) else np.asarray([], dtype=float)
    all_cn = np.asarray(all_cn, dtype=float)

    return {
        "B_offcenter_mean": float(np.nanmean(all_offcenter)),
        "B_offcenter_max": float(np.nanmax(all_offcenter)),
        "B_offcenter_frac_gt_public": float(np.mean(all_offcenter > frac_threshold)),
        "BX_dist_mean": float(np.nanmean(all_dists)) if all_dists.size else np.nan,
        "BX_dist_std": float(np.nanstd(all_dists)) if all_dists.size else np.nan,
        "BX_dist_min": float(np.nanmin(all_dists)) if all_dists.size else np.nan,
        "BX_dist_max": float(np.nanmax(all_dists)) if all_dists.size else np.nan,
        "B_CN_mean": float(np.nanmean(all_cn)),
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
def run_b_site_geometry_stage(
    df: pd.DataFrame,
    structure_dir: Path,
    rules: Dict[str, Any],
) -> pd.DataFrame:
    out = df.copy()
    structure_dir = Path(structure_dir)
    cif_index = _build_cif_index(structure_dir)

    required_cols = rules.get(
        "required_input_cols",
        ["Polar_mpid", "NPolar_mpid", "$B_{site}$", "$X_{site}$"],
    )
    for c in required_cols:
        if c not in out.columns:
            raise ValueError(f"Missing required column for B-site geometry stage: {c}")

    polar_feat_rows = []
    npolar_feat_rows = []
    delta_feat_rows = []

    strict_mode = bool(rules.get("strict_mode", False))

    for row in out.itertuples(index=False):
        polar_mpid = getattr(row, "Polar_mpid")
        npolar_mpid = getattr(row, "NPolar_mpid")
        B_symbol = getattr(row, "$B_{site}$")
        X_symbol = getattr(row, "$X_{site}$")

        p_struct = _load_structure(polar_mpid, cif_index)
        n_struct = _load_structure(npolar_mpid, cif_index)

        if p_struct is None or n_struct is None:
            if strict_mode:
                raise RuntimeError(
                    f"Cannot load structure for pair ({polar_mpid}, {npolar_mpid})."
                )
            p_feat = {
                "polar_B_offcenter_mean": np.nan,
                "polar_B_offcenter_max": np.nan,
                "polar_B_offcenter_frac_gt_public": np.nan,
                "polar_BX_dist_mean": np.nan,
                "polar_BX_dist_std": np.nan,
                "polar_BX_dist_min": np.nan,
                "polar_BX_dist_max": np.nan,
                "polar_B_CN_mean": np.nan,
            }
            n_feat = {
                "npolar_B_offcenter_mean": np.nan,
                "npolar_B_offcenter_max": np.nan,
                "npolar_B_offcenter_frac_gt_public": np.nan,
                "npolar_BX_dist_mean": np.nan,
                "npolar_BX_dist_std": np.nan,
                "npolar_BX_dist_min": np.nan,
                "npolar_BX_dist_max": np.nan,
                "npolar_B_CN_mean": np.nan,
            }
            d_feat = {
                "d_B_offcenter_mean": np.nan,
                "d_B_offcenter_max": np.nan,
                "d_B_offcenter_frac_gt_public": np.nan,
                "d_BX_dist_mean": np.nan,
                "d_BX_dist_std": np.nan,
                "d_BX_dist_min": np.nan,
                "d_BX_dist_max": np.nan,
                "d_B_CN_mean": np.nan,
            }
        else:
            p_raw = _compute_B_geometry_for_structure(p_struct, str(B_symbol), str(X_symbol), rules)
            n_raw = _compute_B_geometry_for_structure(n_struct, str(B_symbol), str(X_symbol), rules)

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