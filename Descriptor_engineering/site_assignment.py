# -*- coding: utf-8 -*-
"""
Public-safe A/B/X site assignment module.

High-level workflow is public:
- parse formula
- assign A/B/X via private rule callback
- write site-level composition metadata

Sensitive assignment criteria should stay inside the private rule callback.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from pymatgen.core.composition import Composition
from pymatgen.core.periodic_table import Element


# =========================================================
# Utilities
# =========================================================
def _is_missing(x: Any) -> bool:
    if pd.isna(x):
        return True
    s = str(x).strip().lower()
    return s in {"", "nan", "none", "null"}


def _ordered_unique_symbols_from_formula(formula: str) -> List[str]:
    """
    Extract ordered unique element symbols from a formula string.
    Keeps first appearance order, e.g. 'BaTiO3' -> ['Ba', 'Ti', 'O'].
    """
    if not isinstance(formula, str) or not formula.strip():
        return []

    elems = re.findall(r"[A-Z][a-z]?", formula.strip())
    out = []
    seen = set()
    for e in elems:
        if e not in seen:
            out.append(e)
            seen.add(e)
    return out


def _build_fraction_map(comp: Composition, symbols: List[str]) -> Dict[str, float]:
    out = {}
    for s in symbols:
        try:
            out[s] = float(comp.get_atomic_fraction(s))
        except Exception:
            out[s] = np.nan
    return out


def _element_mass(sym: str) -> float:
    try:
        return float(Element(sym).atomic_mass)
    except Exception:
        return np.nan


def _element_mendeleev(sym: str) -> float:
    try:
        return float(Element(sym).mendeleev_no)
    except Exception:
        return np.nan


def _pick_formula_col(df: pd.DataFrame, rules: Dict[str, Any]) -> str:
    candidates = rules.get(
        "formula_col_candidates",
        ["Polar_pretty_formula", "pretty_formula", "formula", "Formula"],
    )
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"Cannot find formula column. Tried: {candidates}. "
        f"Available columns: {list(df.columns)[:50]}"
    )


def _validate_assignment(
    symbols_in_formula: List[str],
    assign_out: Tuple[str, str, str],
) -> Tuple[str, str, str]:
    if not isinstance(assign_out, (tuple, list)) or len(assign_out) != 3:
        raise ValueError(
            "Private site-assignment callback must return exactly 3 symbols: (A, B, X)."
        )

    A, B, X = [str(x).strip() for x in assign_out]

    if len({A, B, X}) != 3:
        raise ValueError(f"A/B/X assignment must be distinct, got: {(A, B, X)}")

    missing = [s for s in [A, B, X] if s not in symbols_in_formula]
    if missing:
        raise ValueError(
            f"Assigned symbols {missing} are not present in formula symbols {symbols_in_formula}"
        )

    return A, B, X


# =========================================================
# Public API
# =========================================================
def run_site_assignment_stage(
    df: pd.DataFrame,
    rules: Dict[str, Any],
) -> pd.DataFrame:
    """
    Run site assignment on the input pair table.

    Expected:
    - formula column exists (usually Polar_pretty_formula)
    - private rule callback is provided in rules["site_assignment_fn"]

    Returns:
    - original df with added A/B/X site metadata columns
    """
    out = df.copy()
    formula_col = _pick_formula_col(out, rules)

    site_assignment_fn = rules.get("site_assignment_fn", None)
    if site_assignment_fn is None or not callable(site_assignment_fn):
        raise ValueError(
            "SITE_RULES must provide a callable `site_assignment_fn(symbols, composition, formula)`."
        )

    A_site, B_site, X_site = [], [], []
    A_wt, B_wt, X_wt = [], [], []
    A_mass, B_mass, X_mass = [], [], []
    A_mend, B_mend, X_mend = [], [], []

    strict_mode = bool(rules.get("strict_mode", True))

    for i, formula in enumerate(out[formula_col].tolist()):
        if _is_missing(formula):
            vals = [np.nan] * 12
        else:
            try:
                comp = Composition(str(formula).strip())
                ordered_symbols = _ordered_unique_symbols_from_formula(str(formula).strip())

                # private rule decides how to assign A/B/X
                assign_out = site_assignment_fn(
                    symbols=ordered_symbols,
                    composition=comp,
                    formula=str(formula).strip(),
                )
                A, B, X = _validate_assignment(ordered_symbols, assign_out)

                frac_map = _build_fraction_map(comp, ordered_symbols)

                vals = [
                    A,
                    B,
                    X,
                    frac_map.get(A, np.nan),
                    frac_map.get(B, np.nan),
                    frac_map.get(X, np.nan),
                    _element_mass(A),
                    _element_mass(B),
                    _element_mass(X),
                    _element_mendeleev(A),
                    _element_mendeleev(B),
                    _element_mendeleev(X),
                ]
            except Exception as e:
                if strict_mode:
                    raise RuntimeError(f"Site assignment failed at row {i}, formula={formula!r}: {e}") from e
                vals = [np.nan] * 12

        A_site.append(vals[0]); B_site.append(vals[1]); X_site.append(vals[2])
        A_wt.append(vals[3]);   B_wt.append(vals[4]);   X_wt.append(vals[5])
        A_mass.append(vals[6]); B_mass.append(vals[7]); X_mass.append(vals[8])
        A_mend.append(vals[9]); B_mend.append(vals[10]); X_mend.append(vals[11])

    # -----------------------------------------------------
    # Write columns using your original notebook schema
    # -----------------------------------------------------
    out["$A_{site}$"] = A_site
    out["$B_{site}$"] = B_site
    out["$X_{site}$"] = X_site

    out["$A_{wt}$"] = pd.to_numeric(A_wt, errors="coerce")
    out["$B_{wt}$"] = pd.to_numeric(B_wt, errors="coerce")
    out["$X_{wt}$"] = pd.to_numeric(X_wt, errors="coerce")

    out["$M_A$"] = pd.to_numeric(A_mass, errors="coerce")
    out["$M_B$"] = pd.to_numeric(B_mass, errors="coerce")
    out["$M_X$"] = pd.to_numeric(X_mass, errors="coerce")

    out["$A_{Mend}$"] = pd.to_numeric(A_mend, errors="coerce")
    out["$B_{Mend}$"] = pd.to_numeric(B_mend, errors="coerce")
    out["$X_{Mend}$"] = pd.to_numeric(X_mend, errors="coerce")

    out["$M_{A_wt}$"] = (out["$M_A$"] * out["$A_{wt}$"]).abs()
    out["$M_{B_wt}$"] = (out["$M_B$"] * out["$B_{wt}$"]).abs()
    out["$M_{X_wt}$"] = (out["$M_X$"] * out["$X_{wt}$"]).abs()

    out["$A_{Mend_wt}$"] = (out["$A_{Mend}$"] * out["$A_{wt}$"]).abs()
    out["$B_{Mend_wt}$"] = (out["$B_{Mend}$"] * out["$B_{wt}$"]).abs()
    out["$X_{Mend_wt}$"] = (out["$X_{Mend}$"] * out["$X_{wt}$"]).abs()

    # Optional generic aliases for downstream code readability
    if bool(rules.get("add_alias_cols", True)):
        out["A_site_symbol"] = out["$A_{site}$"]
        out["B_site_symbol"] = out["$B_{site}$"]
        out["X_site_symbol"] = out["$X_{site}$"]

        out["A_site_fraction"] = out["$A_{wt}$"]
        out["B_site_fraction"] = out["$B_{wt}$"]
        out["X_site_fraction"] = out["$X_{wt}$"]

    return out