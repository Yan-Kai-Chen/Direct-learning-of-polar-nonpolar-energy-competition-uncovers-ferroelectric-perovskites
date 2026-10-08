"""Immutable crystallographic contracts at package boundaries."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal, TypeAlias

from polarevolve.crystal.integer_lattice import (
    IDENTITY_HNF,
    IntegerMatrix3,
    determinant_3x3,
    validate_row_hnf,
)


class ContractError(ValueError):
    """Raised when a package boundary object is ambiguous or inconsistent."""


CellRepresentation: TypeAlias = Literal["primitive", "conventional"]
OrbitReduction: TypeAlias = Literal["member_sum"]

_ELEMENT_RE = re.compile(r"^[A-Z][a-z]?$")
_WYCKOFF_RE = re.compile(r"^[a-z]+$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _nonempty(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{field_name} must be a non-empty string")
    return value.strip()


def _positive_int(value: object, *, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ContractError(f"{field_name} must be a positive integer")
    return int(value)


def _normalize_composition(
    value: Mapping[str, int] | Sequence[tuple[str, int]],
) -> tuple[tuple[str, int], ...]:
    items = value.items() if isinstance(value, Mapping) else value
    normalized: dict[str, int] = {}
    for element, count in items:
        if not isinstance(element, str) or not _ELEMENT_RE.fullmatch(element):
            raise ContractError(f"invalid element symbol in reduced_composition: {element!r}")
        normalized[element] = _positive_int(
            count, field_name=f"reduced_composition[{element}]"
        )
    if not normalized:
        raise ContractError("reduced_composition must not be empty")
    return tuple(sorted(normalized.items()))


@dataclass(frozen=True)
class GaugeAssetRef:
    name: str
    version: str
    sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _nonempty(self.name, field_name="gauge.name"))
        object.__setattr__(
            self, "version", _nonempty(self.version, field_name="gauge.version")
        )
        digest = str(self.sha256).lower()
        if not _SHA256_RE.fullmatch(digest):
            raise ContractError("gauge.sha256 must contain 64 lowercase hex digits")
        object.__setattr__(self, "sha256", digest)


@dataclass(frozen=True)
class WyckoffOrbit:
    site_id: str
    element: str
    letter: str
    occurrence: int
    multiplicity_conventional: int
    multiplicity_base: int
    free_dimension: int
    orbit_reduction: OrbitReduction = "member_sum"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "site_id", _nonempty(self.site_id, field_name="orbit.site_id")
        )
        if not _ELEMENT_RE.fullmatch(self.element):
            raise ContractError(f"invalid orbit element: {self.element!r}")
        if not _WYCKOFF_RE.fullmatch(self.letter):
            raise ContractError(f"invalid Wyckoff letter: {self.letter!r}")
        _positive_int(self.occurrence, field_name="orbit.occurrence")
        _positive_int(
            self.multiplicity_conventional,
            field_name="orbit.multiplicity_conventional",
        )
        _positive_int(self.multiplicity_base, field_name="orbit.multiplicity_base")
        if (
            not isinstance(self.free_dimension, int)
            or isinstance(self.free_dimension, bool)
            or not 0 <= self.free_dimension <= 3
        ):
            raise ContractError("orbit.free_dimension must be an integer in 0..3")
        if self.orbit_reduction != "member_sum":
            raise ContractError("orbit reductions must sum all symmetry-related members")


@dataclass(frozen=True)
class HardCondition:
    condition_id: str
    hall_number: int
    setting_id: str
    centering_index: int
    base_cell_representation: CellRepresentation
    group_num_atoms: int
    base_num_atoms: int
    output_num_atoms: int
    reduced_composition: tuple[tuple[str, int], ...]
    formula_unit_scale_base: int
    wyckoff_orbits: tuple[WyckoffOrbit, ...]
    gauge_asset: GaugeAssetRef
    output_hnf: IntegerMatrix3 = IDENTITY_HNF
    schema_version: str = "gt_sge_hard_condition_v1"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "condition_id",
            _nonempty(self.condition_id, field_name="condition_id"),
        )
        if not isinstance(self.hall_number, int) or not 1 <= self.hall_number <= 530:
            raise ContractError("hall_number must be an integer in 1..530")
        object.__setattr__(
            self, "setting_id", _nonempty(self.setting_id, field_name="setting_id")
        )
        centering = _positive_int(self.centering_index, field_name="centering_index")
        if self.base_cell_representation not in {"primitive", "conventional"}:
            raise ContractError(
                "base_cell_representation must be primitive or conventional"
            )
        group_atoms = _positive_int(self.group_num_atoms, field_name="group_num_atoms")
        base_atoms = _positive_int(self.base_num_atoms, field_name="base_num_atoms")
        output_atoms = _positive_int(self.output_num_atoms, field_name="output_num_atoms")
        formula_scale = _positive_int(
            self.formula_unit_scale_base, field_name="formula_unit_scale_base"
        )
        composition = _normalize_composition(self.reduced_composition)
        orbits = tuple(self.wyckoff_orbits)
        if not orbits or any(not isinstance(item, WyckoffOrbit) for item in orbits):
            raise ContractError("wyckoff_orbits must contain at least one WyckoffOrbit")
        if not isinstance(self.gauge_asset, GaugeAssetRef):
            raise ContractError("gauge_asset must be a GaugeAssetRef")
        hnf = validate_row_hnf(self.output_hnf)

        expected_group_atoms = (
            base_atoms * centering
            if self.base_cell_representation == "primitive"
            else base_atoms
        )
        if group_atoms != expected_group_atoms:
            raise ContractError(
                "group/base atom counts contradict centering_index and cell representation"
            )
        if output_atoms != determinant_3x3(hnf) * base_atoms:
            raise ContractError(
                "output_num_atoms must equal det(output_hnf) * base_num_atoms"
            )
        if sum(count for _, count in composition) * formula_scale != base_atoms:
            raise ContractError(
                "reduced_composition and formula_unit_scale_base contradict base_num_atoms"
            )

        seen_ids: set[str] = set()
        base_by_element: Counter[str] = Counter()
        conventional_total = 0
        for orbit in orbits:
            if orbit.site_id in seen_ids:
                raise ContractError(f"duplicate orbit site_id: {orbit.site_id}")
            seen_ids.add(orbit.site_id)
            expected_base = (
                orbit.multiplicity_conventional // centering
                if self.base_cell_representation == "primitive"
                else orbit.multiplicity_conventional
            )
            if (
                self.base_cell_representation == "primitive"
                and orbit.multiplicity_conventional % centering
            ):
                raise ContractError(
                    f"orbit {orbit.site_id} multiplicity is not divisible by centering_index"
                )
            if orbit.multiplicity_base != expected_base:
                raise ContractError(
                    f"orbit {orbit.site_id} conventional/base multiplicities disagree"
                )
            base_by_element[orbit.element] += orbit.multiplicity_base
            conventional_total += orbit.multiplicity_conventional

        expected_by_element = {
            element: count * formula_scale for element, count in composition
        }
        if dict(base_by_element) != expected_by_element:
            raise ContractError(
                f"orbit composition {dict(base_by_element)} != expected {expected_by_element}"
            )
        if conventional_total != group_atoms:
            raise ContractError(
                "sum of conventional orbit multiplicities must equal group_num_atoms"
            )

        object.__setattr__(self, "reduced_composition", composition)
        object.__setattr__(self, "wyckoff_orbits", orbits)
        object.__setattr__(self, "output_hnf", hnf)

    @property
    def element_counts(self) -> tuple[tuple[str, int], ...]:
        return tuple(
            (element, count * self.formula_unit_scale_base)
            for element, count in self.reduced_composition
        )

    @property
    def output_supercell_factor(self) -> int:
        return determinant_3x3(self.output_hnf)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["reduced_composition"] = dict(self.reduced_composition)
        payload["element_counts"] = dict(self.element_counts)
        payload["output_hnf"] = [list(row) for row in self.output_hnf]
        return payload


__all__ = ["ContractError", "GaugeAssetRef", "HardCondition", "WyckoffOrbit"]
