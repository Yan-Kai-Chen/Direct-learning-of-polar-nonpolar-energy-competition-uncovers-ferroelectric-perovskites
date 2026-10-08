"""Build :class:`PackedASUBatch` tensors from verified cache records."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Sequence

import numpy as np
import torch

from polarevolve.crystal.lattice import (
    HallMetricFrame,
    encode_lattice_coordinates,
)
from polarevolve.data.batch import PackedASUBatch, lattice_dependent_geometry
from polarevolve.data.cache import DecodedASURecord
from polarevolve.crystal.contracts import HardCondition
from polarevolve.crystal.state import OrbitLayout
from polarevolve.crystal.symmetry import WyckoffDatabase, WyckoffGauge
from pymatgen.core import Element
from itertools import accumulate

LATTICE_COORDINATE_DIMENSION = 6


@dataclass(frozen=True)
class ConditionRecord:
    """Hard input only: deliberately has no state, target or lattice."""
    hard_condition: HardCondition
    space_group_number: int
    orbit_gauges: tuple[WyckoffGauge, ...]
    group_atom_types: tuple[int, ...]
    member_to_group_atom: tuple[tuple[int, ...], ...]
    layout: OrbitLayout
    split: str = "query"

    @property
    def material_id(self) -> str:
        return self.hard_condition.condition_id

    @classmethod
    def build(cls, hard: HardCondition, space_group: int, wyckoff: WyckoffDatabase):
        gauges = tuple(wyckoff.entry(hard.hall_number, o.letter) for o in hard.wyckoff_orbits)
        atoms, members = [], []
        for orbit, gauge in zip(hard.wyckoff_orbits, gauges):
            start = len(atoms)
            atoms.extend([Element(orbit.element).Z] * gauge.multiplicity)
            members.append(tuple(range(start, len(atoms))))
        dimensions = tuple(g.free_dimension for g in gauges)
        return cls(hard, space_group, gauges, tuple(atoms), tuple(members),
                   OrbitLayout(tuple(o.site_id for o in hard.wyckoff_orbits), dimensions,
                               tuple(accumulate(dimensions, initial=0))))


@dataclass(frozen=True)
class LatticeConditionBatch:
    """Inference-safe hard conditions for independent lattice prediction."""

    hall_numbers: torch.Tensor
    space_group_numbers: torch.Tensor
    atom_counts: torch.Tensor
    orbit_to_structure: torch.Tensor
    orbit_atomic_numbers: torch.Tensor
    orbit_multiplicities: torch.Tensor
    orbit_dimensions: torch.Tensor
    orbit_letter_indices: torch.Tensor
    shape_dimensions: torch.Tensor

    def __post_init__(self) -> None:
        batch_size = int(self.hall_numbers.numel())
        structure_fields = (
            self.space_group_numbers,
            self.atom_counts,
            self.shape_dimensions,
        )
        if self.hall_numbers.shape != (batch_size,) or any(
            value.shape != (batch_size,) for value in structure_fields
        ):
            raise ValueError("lattice structure conditions must be one-dimensional")
        orbit_count = int(self.orbit_to_structure.numel())
        orbit_fields = (
            self.orbit_atomic_numbers,
            self.orbit_multiplicities,
            self.orbit_dimensions,
            self.orbit_letter_indices,
        )
        if any(value.shape != (orbit_count,) for value in orbit_fields):
            raise ValueError("lattice orbit conditions must share one length")
        if batch_size == 0 or orbit_count == 0:
            raise ValueError("lattice condition batches must be non-empty")
        if bool((self.atom_counts <= 0).any().item()) or bool(
            (self.shape_dimensions < 0).any().item()
        ) or bool((self.shape_dimensions > 5).any().item()):
            raise ValueError("invalid lattice condition dimensions")
        range_checks = (
            (self.hall_numbers >= 1).all(),
            (self.hall_numbers <= 530).all(),
            (self.space_group_numbers >= 1).all(),
            (self.space_group_numbers <= 230).all(),
            (self.orbit_atomic_numbers >= 1).all(),
            (self.orbit_atomic_numbers <= 118).all(),
            (self.orbit_multiplicities >= 1).all(),
            (self.orbit_multiplicities <= 256).all(),
            (self.orbit_dimensions >= 0).all(),
            (self.orbit_dimensions <= 3).all(),
            (self.orbit_letter_indices >= 1).all(),
            (self.orbit_letter_indices <= 26).all(),
        )
        if not all(bool(value.item()) for value in range_checks):
            raise ValueError("lattice condition values exceed their contract")
        if bool((self.orbit_to_structure < 0).any().item()) or bool(
            (self.orbit_to_structure >= batch_size).any().item()
        ):
            raise ValueError("lattice orbit-to-structure index is out of bounds")
        represented_atoms = torch.zeros_like(self.atom_counts)
        represented_atoms.index_add_(
            0, self.orbit_to_structure, self.orbit_multiplicities
        )
        if not torch.equal(represented_atoms, self.atom_counts):
            raise ValueError("Wyckoff multiplicities do not match the atom counts")

    @property
    def batch_size(self) -> int:
        return int(self.hall_numbers.numel())

    def to(
        self, device: torch.device | str, *, copy: bool = False
    ) -> "LatticeConditionBatch":
        return LatticeConditionBatch(
            **{
                item.name: getattr(self, item.name).to(device=device, copy=copy)
                for item in fields(self)
            }
        )

    def active_coordinate_mask(self) -> torch.Tensor:
        """Return the Hall-determined mask for volume plus shape coordinates."""

        coordinate = torch.arange(
            LATTICE_COORDINATE_DIMENSION,
            device=self.shape_dimensions.device,
        )
        return coordinate[None, :] <= self.shape_dimensions[:, None]


@dataclass(frozen=True)
class LatticeTargetBatch:
    condition: LatticeConditionBatch
    coordinates: torch.Tensor
    active_mask: torch.Tensor
    lattices: torch.Tensor

    def __post_init__(self) -> None:
        expected = (self.condition.batch_size, LATTICE_COORDINATE_DIMENSION)
        if self.coordinates.shape != expected or self.active_mask.shape != expected:
            raise ValueError("lattice targets must have shape [batch,6]")
        if self.active_mask.dtype != torch.bool:
            raise ValueError("lattice target mask must be boolean")
        if not torch.equal(
            self.active_mask, self.condition.active_coordinate_mask()
        ):
            raise ValueError("lattice target mask differs from the Hall frame")
        if self.lattices.shape != (self.condition.batch_size, 3, 3):
            raise ValueError("target lattices must have shape [batch,3,3]")
        if not bool(torch.isfinite(self.coordinates).all().item()) or not bool(
            torch.isfinite(self.lattices).all().item()
        ):
            raise FloatingPointError("lattice targets contain non-finite values")

    def to(
        self, device: torch.device | str, *, copy: bool = False
    ) -> "LatticeTargetBatch":
        return LatticeTargetBatch(
            condition=self.condition.to(device, copy=copy),
            coordinates=self.coordinates.to(device=device, copy=copy),
            active_mask=self.active_mask.to(device=device, copy=copy),
            lattices=self.lattices.to(device=device, copy=copy),
        )


@dataclass(frozen=True)
class LatticeNormalizer:
    """Train-fitted coordinate transform shared by lattice consumers."""

    mean: torch.Tensor
    scale: torch.Tensor

    @classmethod
    def fit_batches(cls, batches: Sequence[LatticeTargetBatch]) -> "LatticeNormalizer":
        if not batches:
            raise ValueError("normalizer requires at least one lattice batch")
        coordinates = torch.cat([batch.coordinates for batch in batches])
        mask = torch.cat([batch.active_mask for batch in batches])
        weights = mask.to(coordinates.dtype)
        count = weights.sum(dim=0)
        mean = (coordinates * weights).sum(dim=0) / count.clamp_min(1.0)
        variance = ((coordinates - mean).square() * weights).sum(dim=0)
        variance = variance / count.clamp_min(1.0)
        return cls(
            torch.where(count > 0, mean, torch.zeros_like(mean)),
            torch.where(
                count > 1,
                variance.sqrt().clamp_min(1.0e-4),
                torch.ones_like(mean),
            ),
        )

    @classmethod
    def fit(cls, target: LatticeTargetBatch) -> "LatticeNormalizer":
        return cls.fit_batches((target,))

    def normalize(self, coordinates: torch.Tensor) -> torch.Tensor:
        return (coordinates - self.mean.to(coordinates)) / self.scale.to(coordinates)

    def denormalize(self, coordinates: torch.Tensor) -> torch.Tensor:
        return coordinates * self.scale.to(coordinates) + self.mean.to(coordinates)

    def to(self, device: torch.device | str) -> "LatticeNormalizer":
        return LatticeNormalizer(self.mean.to(device), self.scale.to(device))

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "mean": self.mean.detach().cpu().tolist(),
            "scale": self.scale.detach().cpu().tolist(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "LatticeNormalizer":
        return cls(
            torch.tensor(value["mean"], dtype=torch.float32),
            torch.tensor(value["scale"], dtype=torch.float32),
        )


def pack_lattice_conditions(
    records: Sequence[DecodedASURecord | ConditionRecord],
    *,
    frames: Mapping[int, HallMetricFrame],
) -> LatticeConditionBatch:
    """Pack only inference-safe hard conditions for lattice generation."""

    if not records:
        raise ValueError("at least one decoded ASU record is required")
    columns = {field.name: [] for field in fields(LatticeConditionBatch)}
    for structure, record in enumerate(records):
        hall = record.hard_condition.hall_number
        frame = frames[hall]
        if frame.hall_number != hall:
            raise ValueError("Hall metric frame key and payload disagree")
        values = {"hall_numbers": hall, "space_group_numbers": record.space_group_number,
                  "atom_counts": len(record.group_atom_types), "shape_dimensions": frame.shape_dimension}
        for name, value in values.items():
            columns[name].append(value)
        for gauge, members in zip(record.orbit_gauges, record.member_to_group_atom):
            if len(gauge.letter) != 1 or not "a" <= gauge.letter <= "z":
                raise ValueError("lattice model requires one-letter Wyckoff labels")
            values = {"orbit_to_structure": structure,
                      "orbit_atomic_numbers": record.group_atom_types[members[0]],
                      "orbit_multiplicities": gauge.multiplicity, "orbit_dimensions": gauge.free_dimension,
                      "orbit_letter_indices": ord(gauge.letter) - ord("a") + 1}
            for name, value in values.items():
                columns[name].append(value)
    return LatticeConditionBatch(**{name: torch.tensor(values, dtype=torch.long)
                                    for name, values in columns.items()})


def pack_lattice_records(
    records: Sequence[DecodedASURecord],
    *,
    frames: Mapping[int, HallMetricFrame],
    dtype: torch.dtype = torch.float32,
) -> LatticeTargetBatch:
    """Add supervised lattice coordinates to target-free hard conditions."""

    condition = pack_lattice_conditions(records, frames=frames)
    target_coordinates: list[list[float]] = []
    lattices: list[tuple[tuple[float, float, float], ...]] = []
    for record in records:
        hall = int(record.hard_condition.hall_number)
        frame = frames[hall]
        atom_count = len(record.group_atom_types)
        encoded, _ = encode_lattice_coordinates(
            np.asarray(record.state.lattice.matrix, dtype=np.float64),
            num_atoms=atom_count,
            frame=frame,
        )
        coordinate = [0.0] * LATTICE_COORDINATE_DIMENSION
        coordinate[0] = encoded.log_volume_per_atom
        coordinate[1 : 1 + frame.shape_dimension] = encoded.shape_coefficients
        target_coordinates.append(coordinate)
        lattices.append(record.state.lattice.matrix)
    return LatticeTargetBatch(
        condition=condition,
        coordinates=torch.tensor(target_coordinates, dtype=dtype),
        active_mask=condition.active_coordinate_mask(),
        lattices=torch.tensor(lattices, dtype=dtype),
    )


def pack_decoded_records(
    records: Sequence[DecodedASURecord | ConditionRecord],
    *,
    dtype: torch.dtype = torch.float32,
    lattice_override: Sequence[Sequence[Sequence[float]]] | None = None,
    include_clean_target: bool = True,
) -> PackedASUBatch:
    """Pack decoded records and precompute their immutable graph topology."""

    if not records:
        raise ValueError("at least one decoded ASU record is required")
    if lattice_override is not None and len(lattice_override) != len(records):
        raise ValueError("lattice override must contain one cell per record")
    if any(isinstance(r, ConditionRecord) for r in records) and (
        include_clean_target or lattice_override is None
    ):
        raise ValueError("condition-only packing requires a lattice override and no clean target")
    clean_u: list[float] = []
    u_ptr = [0]
    orbit_ptr = [0]
    atom_ptr = [0]
    parameter_to_structure: list[int] = []
    orbit_to_structure: list[int] = []
    orbit_atomic_numbers: list[int] = []
    orbit_multiplicities: list[int] = []
    orbit_dimensions: list[int] = []
    orbit_letter_indices: list[int] = []
    orbit_local_indices: list[int] = []
    member_to_orbit: list[int] = []
    member_to_structure: list[int] = []
    member_to_atom: list[int] = []
    member_parameter_index: list[list[int]] = []
    member_parameter_mask: list[list[bool]] = []
    member_origin: list[list[float]] = []
    member_basis_u: list[np.ndarray] = []
    member_operation_indices: list[int] = []
    lattices: list[tuple[tuple[float, float, float], ...]] = []
    atom_types: list[int] = []
    hall_numbers: list[int] = []
    space_group_numbers: list[int] = []

    for structure, record in enumerate(records):
        atom_offset = atom_ptr[-1]
        orbit_offset = orbit_ptr[-1]
        lattice = np.asarray(
            record.state.lattice.matrix
            if lattice_override is None
            else lattice_override[structure],
            dtype=np.float64,
        )
        if lattice.shape != (3, 3) or not np.isfinite(lattice).all():
            raise ValueError("packed lattice must be a finite 3x3 matrix")
        lattices.append(tuple(tuple(float(value) for value in row) for row in lattice))
        atom_types.extend(record.group_atom_types)
        hall_numbers.append(record.hard_condition.hall_number)
        space_group_numbers.append(record.space_group_number)
        for local_orbit, (gauge, member_map) in enumerate(
            zip(record.orbit_gauges, record.member_to_group_atom)
        ):
            orbit = orbit_offset + local_orbit
            dimension = gauge.free_dimension
            period = np.asarray(gauge.parameter_period_basis, dtype=np.float64)
            if dimension and include_clean_target:
                q_start = record.layout.parameter_offsets[local_orbit]
                q_stop = record.layout.parameter_offsets[local_orbit + 1]
                q0 = np.asarray(
                    record.state.parameters[q_start:q_stop], dtype=np.float64
                )
                normalized = np.linalg.solve(period, q0) % 1.0
            else:
                normalized = np.zeros(dimension, dtype=np.float64)
            parameter_start = len(clean_u)
            clean_u.extend(float(value) for value in normalized)
            parameter_to_structure.extend([structure] * dimension)
            u_ptr.append(len(clean_u))
            orbit_to_structure.append(structure)
            orbit_atomic_numbers.append(int(record.group_atom_types[member_map[0]]))
            orbit_multiplicities.append(gauge.multiplicity)
            orbit_dimensions.append(dimension)
            if len(gauge.letter) != 1 or not "a" <= gauge.letter <= "z":
                raise ValueError(
                    "production score model requires one-letter Wyckoff labels"
                )
            orbit_letter_indices.append(ord(gauge.letter) - ord("a") + 1)
            orbit_local_indices.append(local_orbit + 1)
            padded_basis: list[np.ndarray] = []
            for member in gauge.member_maps:
                basis_q = np.asarray(member.basis, dtype=np.float64)
                basis_u = np.zeros((3, 3), dtype=np.float64)
                if dimension:
                    basis_u[:, :dimension] = basis_q @ period
                padded_basis.append(basis_u)
            for member_index, (member, atom_index, basis_u) in enumerate(
                zip(gauge.member_maps, member_map, padded_basis)
            ):
                indices = [-1, -1, -1]
                mask = [False, False, False]
                for axis in range(dimension):
                    indices[axis] = parameter_start + axis
                    mask[axis] = True
                member_to_orbit.append(orbit)
                member_to_structure.append(structure)
                member_to_atom.append(atom_offset + atom_index)
                member_parameter_index.append(indices)
                member_parameter_mask.append(mask)
                member_origin.append([float(value) for value in member.origin])
                member_basis_u.append(basis_u)
                member_operation_indices.append(member_index + 1)
        orbit_ptr.append(orbit_offset + len(record.orbit_gauges))
        atom_ptr.append(atom_offset + len(record.group_atom_types))

    member_to_atom_array = np.asarray(member_to_atom, dtype=np.int64)
    atom_counts = np.diff(np.asarray(atom_ptr, dtype=np.int64))
    atom_to_structure_array = np.repeat(
        np.arange(len(records), dtype=np.int64), atom_counts
    )
    atom_to_orbit_array = np.empty(len(atom_types), dtype=np.int64)
    atom_to_orbit_array[member_to_atom_array] = np.asarray(
        member_to_orbit, dtype=np.int64
    )
    atom_member_indices_array = np.empty(len(atom_types), dtype=np.int64)
    atom_member_indices_array[member_to_atom_array] = np.asarray(
        member_operation_indices, dtype=np.int64
    )
    edge_sources: list[np.ndarray] = []
    edge_targets: list[np.ndarray] = []
    for start, stop in zip(atom_ptr[:-1], atom_ptr[1:]):
        atoms = np.arange(start, stop, dtype=np.int64)
        source = np.repeat(atoms, len(atoms))
        target = np.tile(atoms, len(atoms))
        keep = source != target
        edge_sources.append(source[keep])
        edge_targets.append(target[keep])
    edge_source = np.concatenate(edge_sources)
    edge_target = np.concatenate(edge_targets)
    edge_index = np.stack((edge_source, edge_target))
    edge_to_structure = atom_to_structure_array[edge_source]
    edge_same_orbit = atom_to_orbit_array[edge_source] == atom_to_orbit_array[edge_target]
    atom_degree = np.bincount(edge_source, minlength=len(atom_types))

    columns = {
        "lattice": lattices, "atom_types": atom_types, "hall_numbers": hall_numbers,
        "space_group_numbers": space_group_numbers, "u_ptr": u_ptr,
        "orbit_ptr": orbit_ptr, "atom_ptr": atom_ptr,
        "atom_to_structure": atom_to_structure_array, "atom_to_orbit": atom_to_orbit_array,
        "atom_member_indices": atom_member_indices_array,
        "edge_index": edge_index, "edge_to_structure": edge_to_structure,
        "edge_same_orbit": edge_same_orbit, "atom_degree": atom_degree,
        "parameter_to_structure": parameter_to_structure, "orbit_to_structure": orbit_to_structure,
        "orbit_atomic_numbers": orbit_atomic_numbers, "orbit_multiplicities": orbit_multiplicities,
        "orbit_dimensions": orbit_dimensions, "orbit_letter_indices": orbit_letter_indices,
        "orbit_local_indices": orbit_local_indices,
        "member_to_orbit": member_to_orbit, "member_to_structure": member_to_structure,
        "member_to_atom": member_to_atom, "member_parameter_index": member_parameter_index,
        "member_parameter_mask": member_parameter_mask, "member_origin": member_origin,
        "member_basis_u": np.asarray(member_basis_u),
        "member_operation_indices": member_operation_indices,
    }
    float_columns = {"lattice", "member_origin", "member_basis_u"}
    bool_columns = {"edge_same_orbit", "member_parameter_mask"}
    tensors = {
        name: torch.tensor(values, dtype=(
            dtype if name in float_columns else torch.bool if name in bool_columns else torch.long
        ))
        for name, values in columns.items()
    }
    geometry_names = (
        "lattice", "atom_ptr", "parameter_to_structure", "orbit_dimensions",
        "member_to_orbit", "member_to_structure", "member_to_atom",
        "member_parameter_index", "member_parameter_mask", "member_basis_u",
    )
    metric, translations, dimensions = lattice_dependent_geometry(
        **{name: tensors[name] for name in geometry_names}
    )
    return PackedASUBatch(
        **tensors,
        clean_u=torch.tensor(clean_u, dtype=dtype) if include_clean_target else None,
        orbit_metric=metric, translation_basis=translations, translation_dimensions=dimensions,
        material_ids=tuple(record.material_id for record in records),
        layouts=tuple(record.layout for record in records),
        hard_conditions=tuple(record.hard_condition for record in records),
    )


__all__ = [
    "LATTICE_COORDINATE_DIMENSION",
    "LatticeConditionBatch",
    "LatticeNormalizer",
    "LatticeTargetBatch",
    "pack_decoded_records",
    "pack_lattice_conditions",
    "pack_lattice_records",
]
