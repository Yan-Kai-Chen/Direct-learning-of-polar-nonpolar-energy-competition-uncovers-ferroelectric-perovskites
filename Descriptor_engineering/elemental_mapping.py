# -*- coding: utf-8 -*-
"""
Public-safe elemental property mapping module.

High-level workflow is public:
- load element property table
- lookup A/B/X symbols
- map selected elemental properties to pair table

Sensitive property selection / naming / fallback rules
should stay inside the private rule dictionary.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# =========================================================
# Utilities
# =========================================================
def _normalize_colnames(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def _find_first_existing_col(df: pd.DataFrame, candidates: List[str], what: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Cannot find {what}. Tried {candidates}. Available columns: {list(df.columns)[:50]}")


def _safe_series_map(
    left_series: pd.Series,
    lookup_df: pd.DataFrame,
    key_col: str,
    value_col: str,
) -> pd.Series:
    lut = lookup_df.set_index(key_col)[value_col].to_dict()
    return left_series.map(lut)


def _validate_mapping_spec(mapping_spec: List[Dict[str, Any]]) -> None:
    if not isinstance(mapping_spec, list) or len(mapping_spec) == 0:
        raise ValueError("ELEMENT_RULES['mapping_spec'] must be a non-empty list of mapping definitions.")

    required = {"site", "source_col", "out_col"}
    for i, item in enumerate(mapping_spec):
        if not isinstance(item, dict):
            raise ValueError(f"mapping_spec[{i}] must be a dict.")
        missing = required - set(item.keys())
        if missing:
            raise ValueError(f"mapping_spec[{i}] missing required keys: {missing}")


def _pick_site_symbol_col(df: pd.DataFrame, site: str, rules: Dict[str, Any]) -> str:
    site_col_candidates = rules.get(
        "site_symbol_col_candidates",
        {
            "A": ["$A_{site}$", "A_site_symbol"],
            "B": ["$B_{site}$", "B_site_symbol"],
            "X": ["$X_{site}$", "X_site_symbol"],
        },
    )
    candidates = site_col_candidates.get(site, [])
    return _find_first_existing_col(df, candidates, f"{site}-site symbol column")


# =========================================================
# Public API
# =========================================================
def run_elemental_mapping_stage(
    df: pd.DataFrame,
    prop_df: pd.DataFrame,
    rules: Dict[str, Any],
) -> pd.DataFrame:
    """
    Map selected elemental properties from an elemental property table
    onto A/B/X site symbols.

    Parameters
    ----------
    df : pd.DataFrame
        Pair table after site assignment.
    prop_df : pd.DataFrame
        Element property table.
    rules : dict
        Private mapping rules.

    Returns
    -------
    pd.DataFrame
        Input pair table with new elemental-property columns added.
    """
    out = _normalize_colnames(df)
    prop = _normalize_colnames(prop_df)

    symbol_col_candidates = rules.get("element_symbol_col_candidates", ["Sym", "symbol", "Element"])
    symbol_col = _find_first_existing_col(prop, symbol_col_candidates, "element-symbol key column")

    mapping_spec = rules.get("mapping_spec", [])
    _validate_mapping_spec(mapping_spec)

    # normalize key
    prop = prop.copy()
    prop[symbol_col] = prop[symbol_col].astype(str).str.strip()

    # optional fill defaults
    default_fill_value = rules.get("default_fill_value", np.nan)
    strict_mode = bool(rules.get("strict_mode", True))

    for item in mapping_spec:
        site = str(item["site"]).strip()
        source_col = str(item["source_col"]).strip()
        out_col = str(item["out_col"]).strip()

        site_symbol_col = _pick_site_symbol_col(out, site, rules)

        if source_col not in prop.columns:
            if strict_mode:
                raise ValueError(
                    f"Element property source column {source_col!r} not found in property table."
                )
            out[out_col] = default_fill_value
            continue

        mapped = _safe_series_map(
            left_series=out[site_symbol_col].astype(str).str.strip(),
            lookup_df=prop,
            key_col=symbol_col,
            value_col=source_col,
        )

        out[out_col] = mapped

        # numeric cast when requested or possible
        cast_numeric = bool(item.get("cast_numeric", True))
        if cast_numeric:
            out[out_col] = pd.to_numeric(out[out_col], errors="coerce")

        # fill missing if requested
        if "fill_value" in item:
            out[out_col] = out[out_col].fillna(item["fill_value"])
        else:
            out[out_col] = out[out_col].fillna(default_fill_value)

    return out