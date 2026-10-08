"""Torch packing for verified ASU records.

This module is intentionally not imported by :mod:`polarevolve.data`; CPU-only
contract and audit tools therefore do not acquire a PyTorch dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Callable
import torch

from polarevolve.crystal.contracts import HardCondition
from polarevolve.crystal.state import OrbitLayout

TORCH_BATCH_SCHEMA = "gt_sge_packed_asu_batch_v5"


def lattice_dependent_geometry(
    *,
    lattice: torch.Tensor,
    atom_ptr: torch.Tensor,
    parameter_to_structure: torch.Tensor,
    orbit_dimensions: torch.Tensor,
    member_to_orbit: torch.Tensor,
    member_to_structure: torch.Tensor,
    member_to_atom: torch.Tensor,
    member_parameter_index: torch.Tensor,
    member_parameter_mask: torch.Tensor,
    member_basis_u: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rebuild every Cartesian-metric tensor owned by a lattice."""

    compute_dtype = (
        torch.float32
        if lattice.device.type == "cuda" or lattice.dtype in {torch.float16, torch.bfloat16}
        else torch.float64
    )
    cell = lattice.to(dtype=compute_dtype)
    bases = member_basis_u.to(dtype=compute_dtype)
    orbit_metric = torch.zeros(
        (orbit_dimensions.numel(), 3, 3), device=lattice.device, dtype=compute_dtype
    )
    for orbit, dimension in enumerate(orbit_dimensions.tolist()):
        if dimension == 0:
            continue
        members = torch.nonzero(member_to_orbit == orbit, as_tuple=False).reshape(-1)
        jacobian = torch.einsum(
            "mij,mjk->mik",
            cell[member_to_structure[members]].transpose(1, 2),
            bases[members, :, :dimension],
        )
        orbit_metric[orbit, :dimension, :dimension] = torch.einsum(
            "mcp,mcq->pq", jacobian, jacobian
        )

    translation_basis = torch.zeros(
        (parameter_to_structure.numel(), 3),
        device=lattice.device,
        dtype=compute_dtype,
    )
    translation_dimensions = torch.zeros(
        lattice.shape[0], device=lattice.device, dtype=torch.long
    )
    for structure in range(lattice.shape[0]):
        parameters = torch.nonzero(
            parameter_to_structure == structure, as_tuple=False
        ).reshape(-1)
        if not parameters.numel():
            continue
        expected = torch.arange(
            int(parameters[0]),
            int(parameters[0]) + parameters.numel(),
            device=parameters.device,
        )
        if not torch.equal(parameters, expected):
            raise ValueError("structure parameters must be contiguous")
        members = torch.nonzero(
            member_to_structure == structure, as_tuple=False
        ).reshape(-1)
        atom_count = int(atom_ptr[structure + 1] - atom_ptr[structure])
        jacobian = torch.zeros(
            (atom_count, 3, parameters.numel()),
            device=lattice.device,
            dtype=compute_dtype,
        )
        for member in members.tolist():
            axes = torch.nonzero(
                member_parameter_mask[member], as_tuple=False
            ).reshape(-1)
            if not axes.numel():
                continue
            atom = int(member_to_atom[member] - atom_ptr[structure])
            local_parameters = member_parameter_index[member, axes] - parameters[0]
            member_jacobian = cell[structure].T @ bases[member]
            jacobian[atom, :, local_parameters] = member_jacobian[:, axes]
        flat = jacobian.reshape(-1, parameters.numel())
        centered = (jacobian - jacobian.mean(dim=0, keepdim=True)).reshape_as(flat)
        _, singular_values, right_h = torch.linalg.svd(centered, full_matrices=True)
        scale = float(singular_values.max()) if singular_values.numel() else 0.0
        tolerance = max(centered.shape) * torch.finfo(compute_dtype).eps * max(scale, 1.0)
        rank = int(torch.count_nonzero(singular_values > tolerance))
        kernel = right_h[rank:].T
        dimension = int(kernel.shape[1])
        if dimension > 3:
            raise ValueError("legal common-translation gauge exceeds three dimensions")
        if dimension == 0:
            continue
        gram = kernel.T @ (flat.T @ flat) @ kernel
        eigenvalues, eigenvectors = torch.linalg.eigh(gram)
        if bool((eigenvalues <= torch.finfo(compute_dtype).eps).any().item()):
            raise ValueError("translation gauge is singular in the Cartesian metric")
        basis = kernel @ eigenvectors @ torch.diag(eigenvalues.rsqrt())
        for column in range(dimension):
            pivot = int(torch.argmax(torch.abs(basis[:, column])))
            if float(basis[pivot, column]) < 0.0:
                basis[:, column] *= -1.0
        translation_basis[parameters, :dimension] = basis
        translation_dimensions[structure] = dimension
    return (
        orbit_metric.to(dtype=lattice.dtype),
        translation_basis.to(dtype=lattice.dtype),
        translation_dimensions,
    )


@dataclass(frozen=True)
class PackedASUBatch:
    clean_u: torch.Tensor | None
    u_ptr: torch.Tensor
    orbit_ptr: torch.Tensor
    atom_ptr: torch.Tensor
    atom_to_structure: torch.Tensor
    atom_to_orbit: torch.Tensor
    atom_member_indices: torch.Tensor
    edge_index: torch.Tensor
    edge_to_structure: torch.Tensor
    edge_same_orbit: torch.Tensor
    atom_degree: torch.Tensor
    parameter_to_structure: torch.Tensor
    orbit_to_structure: torch.Tensor
    orbit_atomic_numbers: torch.Tensor
    orbit_multiplicities: torch.Tensor
    orbit_dimensions: torch.Tensor
    orbit_letter_indices: torch.Tensor
    orbit_local_indices: torch.Tensor
    orbit_metric: torch.Tensor
    translation_basis: torch.Tensor
    translation_dimensions: torch.Tensor
    member_to_orbit: torch.Tensor
    member_to_structure: torch.Tensor
    member_to_atom: torch.Tensor
    member_parameter_index: torch.Tensor
    member_parameter_mask: torch.Tensor
    member_origin: torch.Tensor
    member_basis_u: torch.Tensor
    member_operation_indices: torch.Tensor
    lattice: torch.Tensor
    atom_types: torch.Tensor
    hall_numbers: torch.Tensor
    space_group_numbers: torch.Tensor
    material_ids: tuple[str, ...]
    layouts: tuple[OrbitLayout, ...]
    hard_conditions: tuple[HardCondition, ...]
    schema_version: str = TORCH_BATCH_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != TORCH_BATCH_SCHEMA:
            raise ValueError("unsupported packed ASU batch schema")
        if self.clean_u is not None and (
            self.clean_u.shape != self.parameter_shape or not self.clean_u.is_floating_point()
        ):
            raise ValueError("clean_u must be a one-dimensional floating tensor")
        if self.edge_index.ndim != 2 or self.edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2,edges]")
        edge_count = self.edge_index.shape[1]
        shape_groups = (
            ((self.batch_size, 3, 3), ("lattice",)),
            ((self.num_orbits + 1,), ("u_ptr",)),
            ((self.batch_size + 1,), ("orbit_ptr", "atom_ptr")),
            ((self.num_atoms,), ("atom_to_structure", "atom_to_orbit", "atom_member_indices", "atom_degree")),
            ((edge_count,), ("edge_to_structure", "edge_same_orbit")),
            ((self.num_orbits, 3, 3), ("orbit_metric",)),
            ((self.num_parameters, 3), ("translation_basis",)),
            ((self.batch_size,), ("translation_dimensions",)),
            ((self.num_atoms, 3), ("member_origin", "member_parameter_index", "member_parameter_mask")),
            ((self.num_atoms, 3, 3), ("member_basis_u",)),
        )
        for shape, names in shape_groups:
            for name in names:
                if getattr(self, name).shape != shape:
                    raise ValueError(f"{name} must have shape {shape}")
        if self.edge_same_orbit.dtype != torch.bool:
            raise ValueError("edge_same_orbit must be boolean")
        if int(self.u_ptr[0]) != 0 or int(self.u_ptr[-1]) != self.num_parameters:
            raise ValueError("u_ptr does not span packed ASU parameters")
        if not torch.equal(self.u_ptr[1:] - self.u_ptr[:-1], self.orbit_dimensions):
            raise ValueError("u_ptr increments do not match orbit dimensions")
        if bool((self.translation_dimensions < 0).any().item()) or bool(
            (self.translation_dimensions > 3).any().item()
        ):
            raise ValueError("translation gauge dimensions must lie in [0,3]")
        if (
            len(self.layouts) != self.batch_size
            or len(self.hard_conditions) != self.batch_size
        ):
            raise ValueError("Python metadata does not match batch size")
        if (self.clean_u is not None and not bool(torch.isfinite(self.clean_u).all().item())) or not bool(
            torch.isfinite(self.lattice).all().item()
        ) or not bool(torch.isfinite(self.translation_basis).all().item()):
            raise FloatingPointError("packed ASU batch contains non-finite values")
        expected_atoms = torch.arange(
            self.num_atoms, device=self.member_to_atom.device, dtype=torch.long
        )
        if not torch.equal(torch.sort(self.member_to_atom).values, expected_atoms):
            raise ValueError("member_to_atom must be a complete atom permutation")
        for orbit, dimension in enumerate(self.orbit_dimensions.tolist()):
            active = self.orbit_metric[orbit, :dimension, :dimension].float()
            if dimension == 0:
                if bool(torch.count_nonzero(self.orbit_metric[orbit]).item()):
                    raise ValueError("0D orbit metric must be exactly zero")
            elif int(torch.linalg.cholesky_ex(active).info.item()) != 0:
                raise ValueError("active orbit metric must be positive definite")

    @property
    def batch_size(self) -> int:
        return len(self.material_ids)

    @property
    def num_orbits(self) -> int:
        return int(self.orbit_to_structure.numel())

    @property
    def num_parameters(self) -> int:
        return int(self.parameter_to_structure.numel())

    @property
    def parameter_shape(self) -> tuple[int]:
        return (self.num_parameters,)

    def require_target(self) -> torch.Tensor:
        if self.clean_u is None:
            raise ValueError("this operation requires a supervised ASU target")
        return self.clean_u

    @property
    def num_atoms(self) -> int:
        return int(self.atom_types.numel())

    def _map_tensors(
        self, transform: Callable[[torch.Tensor], torch.Tensor]
    ) -> "PackedASUBatch":
        values = {
            item.name: (
                transform(getattr(self, item.name))
                if isinstance(getattr(self, item.name), torch.Tensor)
                else getattr(self, item.name)
            )
            for item in fields(self)
        }
        copied = object.__new__(type(self))
        for name, value in values.items():
            object.__setattr__(copied, name, value)
        return copied

    def pin_memory(self) -> "PackedASUBatch":
        """Pin every CPU tensor so DataLoader transfers can be asynchronous."""

        return self._map_tensors(
            lambda value: value.pin_memory() if value.device.type == "cpu" else value
        )

    def to(
        self, device: torch.device | str, *, non_blocking: bool = False
    ) -> "PackedASUBatch":
        return self._map_tensors(
            lambda value: value.to(device=device, non_blocking=non_blocking)
        )

    def with_lattice(self, lattice: torch.Tensor) -> "PackedASUBatch":
        value = torch.as_tensor(
            lattice, device=self.lattice.device, dtype=self.lattice.dtype
        )
        if value.shape != self.lattice.shape or not bool(
            torch.isfinite(value).all().item()
        ):
            raise ValueError(
                "replacement lattice must be finite with shape [batch,3,3]"
            )
        metric, translation_basis, translation_dimensions = lattice_dependent_geometry(
            lattice=value,
            atom_ptr=self.atom_ptr,
            parameter_to_structure=self.parameter_to_structure,
            orbit_dimensions=self.orbit_dimensions,
            member_to_orbit=self.member_to_orbit,
            member_to_structure=self.member_to_structure,
            member_to_atom=self.member_to_atom,
            member_parameter_index=self.member_parameter_index,
            member_parameter_mask=self.member_parameter_mask,
            member_basis_u=self.member_basis_u,
        )
        return replace(
            self,
            lattice=value,
            orbit_metric=metric,
            translation_basis=translation_basis,
            translation_dimensions=translation_dimensions,
        )

    def expand(self, u: torch.Tensor) -> torch.Tensor:
        if u.shape != self.parameter_shape:
            raise ValueError("ASU parameter shape does not match batch")
        member_u = torch.zeros((self.num_atoms, 3), device=u.device, dtype=u.dtype)
        mask = self.member_parameter_mask
        member_u[mask] = u[self.member_parameter_index[mask]]
        with torch.autocast(device_type=u.device.type, enabled=False):
            member_fractional = torch.remainder(
                self.member_origin
                + torch.einsum("mij,mj->mi", self.member_basis_u, member_u),
                1.0,
            )
        output = torch.empty_like(member_fractional)
        output[self.member_to_atom] = member_fractional
        return output

    def pullback(self, atom_cartesian_covectors: torch.Tensor) -> torch.Tensor:
        if atom_cartesian_covectors.shape != (self.num_atoms, 3):
            raise ValueError("Cartesian covectors must have shape [atoms,3]")
        compute_dtype = torch.promote_types(self.lattice.dtype, atom_cartesian_covectors.dtype)
        if compute_dtype in {torch.float16, torch.bfloat16}:
            compute_dtype = torch.float32
        with torch.autocast(
            device_type=atom_cartesian_covectors.device.type, enabled=False
        ):
            member_covectors = atom_cartesian_covectors[self.member_to_atom].to(
                dtype=compute_dtype
            )
            jacobian = torch.einsum(
                "mij,mjk->mik",
                self.lattice[self.member_to_structure]
                .transpose(1, 2)
                .to(dtype=compute_dtype),
                self.member_basis_u.to(dtype=compute_dtype),
            )
            contribution = torch.einsum("mcp,mc->mp", jacobian, member_covectors)
            output = torch.zeros(
                self.parameter_shape,
                device=self.lattice.device,
                dtype=compute_dtype,
            )
            mask = self.member_parameter_mask
            output.index_add_(
                0,
                self.member_parameter_index[mask],
                contribution[mask],
            )
        return output

    def pushforward(self, tangent: torch.Tensor) -> torch.Tensor:
        """Expand one ASU tangent into aligned atom Cartesian displacements."""

        if tangent.shape != self.parameter_shape:
            raise ValueError("ASU tangent must match packed parameters")
        member_u = torch.zeros(
            (self.num_atoms, 3), device=tangent.device, dtype=tangent.dtype
        )
        mask = self.member_parameter_mask
        member_u[mask] = tangent[self.member_parameter_index[mask]]
        member_fractional = torch.einsum(
            "mij,mj->mi", self.member_basis_u, member_u
        )
        member_cartesian = torch.einsum(
            "mi,mij->mj",
            member_fractional,
            self.lattice[self.member_to_structure],
        )
        output = torch.empty_like(member_cartesian)
        output[self.member_to_atom] = member_cartesian
        return output

__all__ = ["PackedASUBatch", "TORCH_BATCH_SCHEMA", "lattice_dependent_geometry"]
