"""Public names and value semantics for the 18 PolarGen operations.

The registry supports operation planning only. Production guidance-energy
implementations and their executable-status mapping are intentionally absent.
"""

from __future__ import annotations

from dataclasses import dataclass


CORE_OPERATION_IDS = (
    "OD01",
    "OD02",
    "OD03",
    "OD04",
    "OD05",
    "OD06",
    "OD07",
    "OD08",
    "OD09",
    "OD10",
    "OD11",
    "OD12",
    "OD13",
    "OD17",
    "OD18",
    "OD19",
    "OD20",
    "OD22",
)


@dataclass(frozen=True)
class StructuralOperationSpec:
    operation_id: str
    name: str
    value_semantics: str
    site_role: str | None


def _spec(
    operation_id: str,
    name: str,
    value_semantics: str,
    site_role: str | None,
) -> StructuralOperationSpec:
    if value_semantics not in {"signed_change", "absolute_target"}:
        raise ValueError(f"invalid operation semantics: {value_semantics}")
    return StructuralOperationSpec(
        operation_id=operation_id,
        name=name,
        value_semantics=value_semantics,
        site_role=site_role,
    )


STRUCTURAL_OPERATIONS = {
    "OD01": _spec("OD01", "B-X distortion contrast", "signed_change", "B"),
    "OD02": _spec(
        "OD02", "B-site off-centering contrast", "signed_change", "B"
    ),
    "OD03": _spec(
        "OD03", "A-cage anisotropy contrast", "signed_change", "A"
    ),
    "OD04": _spec(
        "OD04", "A-site off-centering contrast", "signed_change", "A"
    ),
    "OD05": _spec("OD05", "Polar A-X distortion", "absolute_target", "A"),
    "OD06": _spec("OD06", "Polar B-X distortion", "absolute_target", "B"),
    "OD07": _spec(
        "OD07",
        "B-site coordination angular distortion",
        "absolute_target",
        "B",
    ),
    "OD08": _spec(
        "OD08", "A-site polyhedral volume distortion", "signed_change", "A"
    ),
    "OD09": _spec(
        "OD09", "B-site polyhedral volume distortion", "signed_change", "B"
    ),
    "OD10": _spec(
        "OD10", "Bridge multiplicity heterogeneity", "signed_change", None
    ),
    "OD11": _spec(
        "OD11", "Framework strain heterogeneity", "signed_change", None
    ),
    "OD12": _spec(
        "OD12", "Corner-sharing angular disorder", "signed_change", "B"
    ),
    "OD13": _spec(
        "OD13", "B-site local environment heterogeneity", "signed_change", "B"
    ),
    "OD17": _spec(
        "OD17", "Global inversion-breaking proxy", "absolute_target", None
    ),
    "OD18": _spec(
        "OD18", "A-B sublattice distortion contrast", "signed_change", None
    ),
    "OD19": _spec(
        "OD19", "Axial distortion anisotropy", "signed_change", None
    ),
    "OD20": _spec(
        "OD20", "Deformation concentration HHI", "absolute_target", None
    ),
    "OD22": _spec(
        "OD22", "Framework orientational misalignment", "signed_change", None
    ),
}


def operation_spec(operation_id: str) -> StructuralOperationSpec:
    key = str(operation_id).upper()
    try:
        return STRUCTURAL_OPERATIONS[key]
    except KeyError as exc:
        raise ValueError(f"unknown PolarGen operation {key!r}") from exc


__all__ = [
    "CORE_OPERATION_IDS",
    "STRUCTURAL_OPERATIONS",
    "StructuralOperationSpec",
    "operation_spec",
]
