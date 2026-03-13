# -*- coding: utf-8 -*-
"""
Public example template for local private descriptor rules.

IMPORTANT:
- Copy this file to `private_rules_local.py` locally.
- Fill in your own private thresholds / selected columns / matching rules.
- Do NOT upload `private_rules_local.py`.
"""

SITE_RULES = {
    "required_input_cols": ["Polar_pretty_formula"],
    "public_output_cols": [
        "A_site_symbol",
        "B_site_symbol",
        "X_site_symbol",
        "A_site_fraction",
        "B_site_fraction",
        "X_site_fraction",
    ],
}

ELEMENT_RULES = {
    "public_property_cols": [
        "A_prop_1",
        "B_prop_1",
        "X_prop_1",
    ],
}

A_GEOM_RULES = {
    "public_output_cols": [
        "polar_A_geom_1",
        "npolar_A_geom_1",
        "d_A_geom_1",
    ],
}

B_GEOM_RULES = {
    "public_output_cols": [
        "polar_B_geom_1",
        "npolar_B_geom_1",
        "d_B_geom_1",
    ],
}

EWALD_RULES = {
    "public_output_cols": [
        "polar_Ewald_1",
        "npolar_Ewald_1",
        "d_Ewald_1",
    ],
}

DERIVED_RULES = {
    "public_output_cols": [
        "derived_feature_1",
        "derived_feature_2",
    ],
}

EXPORT_RULES = {
    "public_keep_cols": [
        "Polar_mpid",
        "NPolar_mpid",
        "A_site_symbol",
        "B_site_symbol",
        "X_site_symbol",
        "polar_A_geom_1",
        "npolar_A_geom_1",
        "d_A_geom_1",
        "polar_B_geom_1",
        "npolar_B_geom_1",
        "d_B_geom_1",
        "polar_Ewald_1",
        "npolar_Ewald_1",
        "d_Ewald_1",
        "derived_feature_1",
        "derived_feature_2",
    ]
}