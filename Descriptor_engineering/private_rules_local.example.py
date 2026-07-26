"""Copy to private_rules_local.py and edit locally when overrides are needed."""

from default_rules import (
    A_GEOM_RULES,
    B_GEOM_RULES,
    DERIVED_RULES,
    ELEMENT_RULES,
    EWALD_RULES,
    EXPORT_RULES,
    SITE_RULES,
)

# Make copies before editing nested settings in a real override.
SITE_RULES = dict(SITE_RULES)
ELEMENT_RULES = dict(ELEMENT_RULES)
A_GEOM_RULES = dict(A_GEOM_RULES)
B_GEOM_RULES = dict(B_GEOM_RULES)
EWALD_RULES = dict(EWALD_RULES)
DERIVED_RULES = dict(DERIVED_RULES)
EXPORT_RULES = dict(EXPORT_RULES)
