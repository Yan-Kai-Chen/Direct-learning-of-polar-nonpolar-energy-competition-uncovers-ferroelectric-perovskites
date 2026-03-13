# -*- coding: utf-8 -*-
"""
Public-safe Ewald / electrostatic descriptor module.

This version exposes a minimal electrostatic workflow:
- load structure
- assign oxidation states using a public callback interface
- compute Ewald total energy per atom
- summarize site-wise electrostatic energies for A/B/X sites
- write polar / npolar / delta descriptors

Detailed oxidation-state assignment logic is intentionally abstracted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from pymatgen.analysis.ewald import EwaldSummation
from pymatgen.core import Structure


# =========================================================
# Basic utilities
# =========================================================
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


def _site_energy_stats(vals: List[float], prefix: str) -> Dict[str, float]:
    a = np.asarray(vals, dtype=float)
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
# Oxidation-state interface
# =========================================================
def _assign_oxidation_states_public(
    struct: Structure,
    A_symbol: str,
    B_symbol: str,
    X_symbol: str,
    rules: Dict[str, Any],
) -> Optional[Structure]:
    """
    Public-safe oxidation-state assignment wrapper.

    A callback must be supplied through:
        rules["oxidation_state_fn"]

    Callback signature:
        oxidation_state_fn(structure, A_symbol, B_symbol, X_symbol) -> dict[element_symbol, oxidation_state]

    Returns a copied structure with oxidation states assigned, or None on failure.
    """
    oxi_fn = rules.get("oxidation_state_fn", None)
    if oxi_fn is None or not callable(oxi_fn):
        return None

    try:
        oxi_map = oxi_fn(
            structure=struct,
            A_symbol=str(A_symbol),
            B_symbol=str(B_symbol),
            X_symbol=str(X_symbol),
        )
        if not isinstance(oxi_map, dict) or len(oxi_map) == 0:
            return None

        s = struct.copy()
        s.add_oxidation_state_by_element(oxi_map)
        return s
    except Exception:
        return None


# =========================================================
# Ewald feature calculation
# =========================================================
def _compute_ewald_features_for_structure(
    struct: Structure,
    A_symbol: str,
    B_symbol: str,
    X_symbol: str,
    rules: Dict[str, Any],
) -> Dict[str, float]:
    s_oxi = _assign_oxidation_states_public(struct, A_symbol, B_symbol, X_symbol, rules)
    if s_oxi is None:
        return {
            "Ewald_eV_per_atom": np.nan,
            "Ewald_A_siteE_mean": np.nan,
            "Ewald_A_siteE_std": np.nan,
            "Ewald_A_siteE_min": np.nan,
            "Ewald_A_siteE_max": np.nan,
            "Ewald_B_siteE_mean": np.nan,
            "Ewald_B_siteE_std": np.nan,
            "Ewald_B_siteE_min": np.nan,
            "Ewald_B_siteE_max": np.nan,
            "Ewald_X_siteE_mean": np.nan,
            "Ewald_X_siteE_std": np.nan,
            "Ewald_X_siteE_min": np.nan,
            "Ewald_X_siteE_max": np.nan,
        }

    try:
        ew = EwaldSummation(s_oxi)
        total_e = float(ew.total_energy)
        site_e = np.asarray(ew.site_energies, dtype=float)

        A_vals, B_vals, X_vals = [], [], []
        for i, site in enumerate(s_oxi.sites):
            sym = str(site.specie.element) if hasattr(site.specie, "element") else str(site.specie)
            if sym == A_symbol:
                A_vals.append(float(site_e[i]))
            elif sym == B_symbol:
                B_vals.append(float(site_e[i]))
            elif sym == X_symbol:
                X_vals.append(float(site_e[i]))

        out = {
            "Ewald_eV_per_atom": total_e / float(len(s_oxi)),
        }
        out.update(_site_energy_stats(A_vals, "Ewald_A_siteE"))
        out.update(_site_energy_stats(B_vals, "Ewald_B_siteE"))
        out.update(_site_energy_stats(X_vals, "Ewald_X_siteE"))
        return out

    except Exception:
        return {
            "Ewald_eV_per_atom": np.nan,
            "Ewald_A_siteE_mean": np.nan,
            "Ewald_A_siteE_std": np.nan,
            "Ewald_A_siteE_min": np.nan,
            "Ewald_A_siteE_max": np.nan,
            "Ewald_B_siteE_mean": np.nan,
            "Ewald_B_siteE_std": np.nan,
            "Ewald_B_siteE_min": np.nan,
            "Ewald_B_siteE_max": np.nan,
            "Ewald_X_siteE_mean": np.nan,
            "Ewald_X_siteE_std": np.nan,
            "Ewald_X_siteE_min": np.nan,
            "Ewald_X_siteE_max": np.nan,
        }


# =========================================================
# Public API
# =========================================================
def run_ewald_stage(
    df: pd.DataFrame,
    structure_dir: Path,
    rules: Dict[str, Any],
) -> pd.DataFrame:
    out = df.copy()
    structure_dir = Path(structure_dir)
    cif_index = _build_cif_index(structure_dir)

    required_cols = rules.get(
        "required_input_cols",
        ["Polar_mpid", "NPolar_mpid", "$A_{site}$", "$B_{site}$", "$X_{site}$"],
    )
    for c in required_cols:
        if c not in out.columns:
            raise ValueError(f"Missing required column for Ewald stage: {c}")

    polar_feat_rows = []
    npolar_feat_rows = []
    delta_feat_rows = []

    strict_mode = bool(rules.get("strict_mode", False))

    for row in out.itertuples(index=False):
        polar_mpid = getattr(row, "Polar_mpid")
        npolar_mpid = getattr(row, "NPolar_mpid")
        A_symbol = getattr(row, "$A_{site}$")
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
                "polar_Ewald_eV_per_atom": np.nan,
                "polar_Ewald_A_siteE_mean": np.nan,
                "polar_Ewald_A_siteE_std": np.nan,
                "polar_Ewald_A_siteE_min": np.nan,
                "polar_Ewald_A_siteE_max": np.nan,
                "polar_Ewald_B_siteE_mean": np.nan,
                "polar_Ewald_B_siteE_std": np.nan,
                "polar_Ewald_B_siteE_min": np.nan,
                "polar_Ewald_B_siteE_max": np.nan,
                "polar_Ewald_X_siteE_mean": np.nan,
                "polar_Ewald_X_siteE_std": np.nan,
                "polar_Ewald_X_siteE_min": np.nan,
                "polar_Ewald_X_siteE_max": np.nan,
            }
            n_feat = {
                "npolar_Ewald_eV_per_atom": np.nan,
                "npolar_Ewald_A_siteE_mean": np.nan,
                "npolar_Ewald_A_siteE_std": np.nan,
                "npolar_Ewald_A_siteE_min": np.nan,
                "npolar_Ewald_A_siteE_max": np.nan,
                "npolar_Ewald_B_siteE_mean": np.nan,
                "npolar_Ewald_B_siteE_std": np.nan,
                "npolar_Ewald_B_siteE_min": np.nan,
                "npolar_Ewald_B_siteE_max": np.nan,
                "npolar_Ewald_X_siteE_mean": np.nan,
                "npolar_Ewald_X_siteE_std": np.nan,
                "npolar_Ewald_X_siteE_min": np.nan,
                "npolar_Ewald_X_siteE_max": np.nan,
            }
            d_feat = {
                "d_Ewald_eV_per_atom": np.nan,
                "d_Ewald_A_siteE_mean": np.nan,
                "d_Ewald_A_siteE_std": np.nan,
                "d_Ewald_A_siteE_min": np.nan,
                "d_Ewald_A_siteE_max": np.nan,
                "d_Ewald_B_siteE_mean": np.nan,
                "d_Ewald_B_siteE_std": np.nan,
                "d_Ewald_B_siteE_min": np.nan,
                "d_Ewald_B_siteE_max": np.nan,
                "d_Ewald_X_siteE_mean": np.nan,
                "d_Ewald_X_siteE_std": np.nan,
                "d_Ewald_X_siteE_min": np.nan,
                "d_Ewald_X_siteE_max": np.nan,
            }
        else:
            p_raw = _compute_ewald_features_for_structure(p_struct, str(A_symbol), str(B_symbol), str(X_symbol), rules)
            n_raw = _compute_ewald_features_for_structure(n_struct, str(A_symbol), str(B_symbol), str(X_symbol), rules)

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