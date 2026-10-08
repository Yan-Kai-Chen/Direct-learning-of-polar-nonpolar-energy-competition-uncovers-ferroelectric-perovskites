"""Compile authoritative Hall/Wyckoff inputs into closed hard conditions."""

from __future__ import annotations

import math
from dataclasses import dataclass

from polarevolve.crystal.integer_lattice import (
    IDENTITY_HNF,
    IntegerMatrix3,
    determinant_3x3,
)
from polarevolve.crystal.symmetry import GroupDatabase, HallSetting, WyckoffDatabase
from polarevolve.crystal.contracts import (
    CellRepresentation,
    ContractError,
    GaugeAssetRef,
    HardCondition,
    WyckoffOrbit,
)


@dataclass(frozen=True)
class OrbitSpec:
    element: str
    letter: str
    occurrence: int = 1


@dataclass(frozen=True)
class CompiledHardCondition:
    hard: HardCondition
    hall: HallSetting


def compile_hard_condition(
    *,
    condition_id: str,
    hall_number: int,
    orbit_specs: tuple[OrbitSpec, ...],
    base_cell_representation: CellRepresentation,
    output_hnf: IntegerMatrix3 = IDENTITY_HNF,
    group_database: GroupDatabase,
    wyckoff_database: WyckoffDatabase,
) -> CompiledHardCondition:
    if not orbit_specs:
        raise ContractError("at least one Wyckoff orbit specification is required")
    hall = group_database.setting(hall_number)
    centering = hall.centering_index
    orbits: list[WyckoffOrbit] = []
    base_counts: dict[str, int] = {}
    seen: set[tuple[str, str, int]] = set()
    occupied_fixed: set[str] = set()
    for spec in orbit_specs:
        key = (spec.element, spec.letter.lower(), int(spec.occurrence))
        if key in seen:
            raise ContractError(f"duplicate orbit specification: {key}")
        seen.add(key)
        gauge = wyckoff_database.entry(hall_number, spec.letter)
        if gauge.free_dimension == 0:
            if gauge.letter in occupied_fixed:
                raise ContractError(f"fixed Wyckoff position {gauge.letter} is occupied twice")
            occupied_fixed.add(gauge.letter)
        if gauge.setting_id != hall.setting_id:
            raise ContractError("Hall and Wyckoff setting IDs disagree")
        conventional = gauge.multiplicity
        if base_cell_representation == "primitive":
            if conventional % centering:
                raise ContractError(
                    f"Wyckoff multiplicity {conventional} is not divisible by centering {centering}"
                )
            base = conventional // centering
        elif base_cell_representation == "conventional":
            base = conventional
        else:
            raise ContractError("base cell representation must be primitive or conventional")
        site_id = f"{spec.element}_{conventional}{gauge.letter}_{spec.occurrence}"
        orbits.append(
            WyckoffOrbit(
                site_id=site_id,
                element=spec.element,
                letter=gauge.letter,
                occurrence=spec.occurrence,
                multiplicity_conventional=conventional,
                multiplicity_base=base,
                free_dimension=gauge.free_dimension,
            )
        )
        base_counts[spec.element] = base_counts.get(spec.element, 0) + base
    formula_scale = math.gcd(*base_counts.values())
    reduced = tuple(
        sorted((element, count // formula_scale) for element, count in base_counts.items())
    )
    base_atoms = sum(base_counts.values())
    group_atoms = sum(orbit.multiplicity_conventional for orbit in orbits)
    gauge_digest = wyckoff_database.provenance.artifact_sha256("wyckoff_gauges.json")
    source_version = (
        wyckoff_database.provenance.source_version
        or wyckoff_database.provenance.database_version
    )
    hard = HardCondition(
        condition_id=condition_id,
        hall_number=hall_number,
        setting_id=hall.setting_id,
        centering_index=centering,
        base_cell_representation=base_cell_representation,
        group_num_atoms=group_atoms,
        base_num_atoms=base_atoms,
        output_num_atoms=base_atoms * determinant_3x3(output_hnf),
        reduced_composition=reduced,
        formula_unit_scale_base=formula_scale,
        wyckoff_orbits=tuple(orbits),
        gauge_asset=GaugeAssetRef(
            name=wyckoff_database.provenance.database_version,
            version=source_version,
            sha256=gauge_digest,
        ),
        output_hnf=output_hnf,
    )
    return CompiledHardCondition(hard=hard, hall=hall)


__all__ = ["CompiledHardCondition", "OrbitSpec", "compile_hard_condition"]
