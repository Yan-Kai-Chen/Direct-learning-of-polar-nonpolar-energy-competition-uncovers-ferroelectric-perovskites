from __future__ import annotations

from pymatgen.core import Element


def assign_abx(*, symbols, composition, formula):
    """Interpretable public fallback for three-element ABX compositions."""

    del composition, formula
    if len(symbols) != 3:
        raise ValueError(
            "The default rule supports exactly three unique elements; "
            "provide private_rules_local.py for mixed-site compositions."
        )
    elements = [Element(symbol) for symbol in symbols]
    x_symbol = max(
        elements,
        key=lambda element: (
            float(element.X) if element.X is not None else float("-inf")
        ),
    ).symbol
    cations = [element for element in elements if element.symbol != x_symbol]

    def radius(element):
        value = element.atomic_radius or element.atomic_radius_calculated
        return float(value) if value is not None else 0.0

    a_element = max(cations, key=radius)
    b_element = next(
        element for element in cations if element.symbol != a_element.symbol
    )
    return a_element.symbol, b_element.symbol, x_symbol


def oxidation_state_guess(*, structure, A_symbol, B_symbol, X_symbol):
    del A_symbol, B_symbol, X_symbol
    guesses = structure.composition.oxi_state_guesses()
    return dict(guesses[0]) if guesses else {}


SITE_RULES = {
    "formula_col_candidates": [
        "Polar_pretty_formula",
        "pretty_formula",
        "formula",
    ],
    "site_assignment_fn": assign_abx,
    "strict_mode": True,
    "add_alias_cols": True,
}

ELEMENT_RULES = {
    "element_symbol_col_candidates": ["symbol", "Symbol", "Element", "Sym"],
    "mapping_spec": [
        {
            "site": "A",
            "source_col": "electronegativity",
            "out_col": "A_electronegativity",
        },
        {
            "site": "B",
            "source_col": "electronegativity",
            "out_col": "B_electronegativity",
        },
        {
            "site": "X",
            "source_col": "electronegativity",
            "out_col": "X_electronegativity",
        },
    ],
    "strict_mode": True,
}

A_GEOM_RULES = {
    "public_search_radius": 4.0,
    "public_max_neighbors": 12,
    "strict_mode": False,
}

B_GEOM_RULES = {
    "public_search_radius": 3.5,
    "public_max_neighbors": 8,
    "public_offcenter_threshold": 0.10,
    "strict_mode": False,
}

EWALD_RULES = {
    "oxidation_state_fn": oxidation_state_guess,
    "strict_mode": False,
}

DERIVED_RULES = {
    "operations": [
        {
            "output": "AX_chi_mismatch",
            "left": "A_electronegativity",
            "right": "X_electronegativity",
            "operation": "absolute_difference",
        }
    ]
}

EXPORT_RULES = {
    "public_keep_cols": None,
    "strict_mode": True,
}
