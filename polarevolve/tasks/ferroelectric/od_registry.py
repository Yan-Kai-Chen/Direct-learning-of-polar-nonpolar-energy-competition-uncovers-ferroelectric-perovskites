"""FE1 order-parameter (OD) registry: the planner's 18 operation IDs.

Three OD enumerations exist upstream and must not be conflated (user ruling
2026-09-18, recorded in RFC FE1 section 2.4):

- the planner registry (``operation_registry.json``) holds **18** IDs:
  OD01-OD13 plus OD17/OD18/OD19/OD20/OD22 — this module's authoritative set;
- the legacy differentiable registry holds 22 (OD01-OD22);
- the target schema holds 25 (adding the OD23/24/25 Ewald family).

All 18 planner ODs are computable supervision targets.  Only OD01 and OD06
are admitted for sampling-time guidance by default; the other 16 are
``supervised_only`` until an FE1-distribution calibration is authorized.
"""

from __future__ import annotations

from dataclasses import dataclass

PLANNER_OD_IDS: tuple[str, ...] = (
    "OD01", "OD02", "OD03", "OD04", "OD05", "OD06", "OD07", "OD08", "OD09",
    "OD10", "OD11", "OD12", "OD13", "OD17", "OD18", "OD19", "OD20", "OD22",
)

# ODs whose per-atom minimum-image deltas require the reference and candidate
# to share atom count and order (guaranteed by the sidecar alignment contract).
CORRESPONDENCE_IDS: tuple[str, ...] = ("OD17", "OD20")


@dataclass(frozen=True)
class ODSpec:
    operation_id: str
    short_name: str
    anchor: str
    value_semantics: str
    scale: float


_SPECS = (
    ("OD01", "BX_distortion_contrast", "B_oct", "signed_change", 0.05),
    ("OD02", "B_offcentering_contrast", "B_oct", "signed_change", 0.20),
    ("OD03", "A_cage_anisotropy_contrast", "A_cage", "signed_change", 0.10),
    ("OD04", "A_offcentering_contrast", "A_cage", "signed_change", 0.20),
    ("OD05", "polar_AX_distortion", "A_cage", "absolute_target", 0.10),
    ("OD06", "polar_BX_distortion", "B_oct", "absolute_target", 0.10),
    ("OD07", "B_coordination_angular_distortion", "B_oct", "absolute_target", 0.20),
    ("OD08", "A_polyhedral_volume_distortion", "A_cage", "signed_change", 0.30),
    ("OD09", "B_polyhedral_volume_distortion", "B_oct", "signed_change", 0.30),
    ("OD10", "bridge_multiplicity_heterogeneity", "global", "signed_change", 0.20),
    ("OD11", "framework_strain_heterogeneity", "global", "signed_change", 0.10),
    ("OD12", "corner_sharing_angular_disorder", "B_oct", "signed_change", 10.0),
    ("OD13", "B_site_local_environment_heterogeneity", "B_oct", "signed_change", 0.10),
    ("OD17", "global_inversion_breaking_proxy", "global", "absolute_target", 0.20),
    ("OD18", "A_B_sublattice_distortion_contrast", "global", "signed_change", 0.20),
    ("OD19", "axial_distortion_anisotropy", "global", "signed_change", 0.30),
    ("OD20", "deformation_concentration_HHI", "global", "absolute_target", 0.20),
    ("OD22", "framework_orientational_misalignment", "global", "signed_change", 0.30),
)

OD_SPECS: tuple[ODSpec, ...] = tuple(
    ODSpec(
        operation_id=operation_id,
        short_name=short_name,
        anchor=anchor,
        value_semantics=value_semantics,
        scale=scale,
    )
    for operation_id, short_name, anchor, value_semantics, scale in _SPECS
)

OD_SCALES: dict[str, float] = {spec.operation_id: spec.scale for spec in OD_SPECS}
B_OCT_OPERATION_IDS: tuple[str, ...] = tuple(
    spec.operation_id for spec in OD_SPECS if spec.anchor == "B_oct"
)

if tuple(spec.operation_id for spec in OD_SPECS) != PLANNER_OD_IDS:
    raise ValueError("OD spec order must match the planner enumeration")
if len(OD_SPECS) != 18:
    raise ValueError("the planner OD contract holds exactly 18 operations")

__all__ = [
    "B_OCT_OPERATION_IDS",
    "CORRESPONDENCE_IDS",
    "OD_SCALES",
    "OD_SPECS",
    "PLANNER_OD_IDS",
    "ODSpec",
]
