"""Differentiable, target-coordinate-free chemistry guidance for MP20.

Hall number and Wyckoff occupancy remain hard conditions.  This module only
scores expanded coordinates and returns Cartesian gradients that can be pulled
back to the existing ASU state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from importlib.metadata import version

import torch
import torch.nn as nn

from polarevolve.data.batch import PackedASUBatch
from polarevolve.guidance.chemistry import (
    CHEMISTRY_CONTRACT, ChemistryPrior, PeriodicNeighborhood, periodic_coordination,
    periodic_energies,
    neutral_oxidation_assignments as _neutral_oxidation_assignments,
)

PHYSICS_GUIDANCE_CONTRACT = "mp20_soft_physics_v1"
RUNTIME_GUIDANCE_ALLOWED_INPUTS = (
    "current_asu_parameters",
    "fixed_lattice",
    "hard_atomic_numbers",
    "hard_reduced_composition",
    "sigma",
)
RUNTIME_GUIDANCE_FORBIDDEN_INPUTS = (
    "clean_asu_target",
    "reference_coordinates",
    "material_id",
    "audit_labels",
)


@dataclass(frozen=True)
class PhysicsGuidanceConfig:
    prior_path: str | None = None
    coordination_weight: float = 0.2
    preference_weight: float = 1.0
    overlap_weight: float = 1.0
    bond_radius_weight: float = 0.15
    bond_valence_weight: float = 0.10
    target_relative_margin: float = 0.02
    sigma_full_strength: float = 0.10
    sigma_cutoff: float = 0.50

    def __post_init__(self) -> None:
        weights = (
            self.overlap_weight,
            self.bond_radius_weight,
            self.bond_valence_weight,
            self.coordination_weight,
            self.preference_weight,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in weights) or not any(weights):
            raise ValueError("physics guidance requires non-negative, non-zero weights")
        if self.target_relative_margin < 0.0:
            raise ValueError("target-relative margin must be non-negative")
        if not 0.0 < self.sigma_full_strength < self.sigma_cutoff:
            raise ValueError("physics sigma gate requires 0 < full_strength < cutoff")


@dataclass(frozen=True)
class PhysicsEnergy:
    total_by_structure: torch.Tensor
    overlap_by_structure: torch.Tensor
    bond_radius_by_structure: torch.Tensor
    bond_valence_by_structure: torch.Tensor
    bond_valence_applicable: torch.Tensor
    coordination_by_structure: torch.Tensor
    preference_by_structure: torch.Tensor
    coordination_coverage: torch.Tensor
    bond_prior_coverage: torch.Tensor


@dataclass(frozen=True)
class PhysicsLoss:
    loss: torch.Tensor
    overlap_excess: torch.Tensor
    bond_radius_excess: torch.Tensor
    bond_valence_excess: torch.Tensor
    bond_valence_applicable_fraction: torch.Tensor
    coordination_excess: torch.Tensor
    coordination_coverage: torch.Tensor


@lru_cache(maxsize=512)
def _neutral_oxidation_assignment(
    atomic_numbers: tuple[int, ...],
) -> tuple[tuple[int, float], ...] | None:
    """Return a unique common-oxidation-state assignment, otherwise mask BVS."""

    solutions = _neutral_oxidation_assignments(atomic_numbers)
    return solutions[0] if len(solutions) == 1 else None




class PhysicsGuidance(nn.Module):
    """Evaluate one shared physical energy for training and reverse sampling."""

    def __init__(self, config: PhysicsGuidanceConfig = PhysicsGuidanceConfig()) -> None:
        super().__init__()
        self.config = config
        self.chemistry_prior = ChemistryPrior(config.prior_path) if config.prior_path else None
        self.neighborhood = PeriodicNeighborhood()
        self.local_preferences = ()
        self.contract = CHEMISTRY_CONTRACT if self.chemistry_prior else PHYSICS_GUIDANCE_CONTRACT
        radii = torch.ones(119, dtype=torch.float32)
        bv_r = torch.zeros(119, dtype=torch.float32)
        bv_c = torch.zeros(119, dtype=torch.float32)
        electronegativity = torch.zeros(119, dtype=torch.float32)
        bv_valid = torch.zeros(119, dtype=torch.bool)
        from pymatgen.analysis.bond_valence import BV_PARAMS
        from pymatgen.core import Element

        for atomic_number in range(1, 119):
            element = Element.from_Z(atomic_number)
            raw_radius = element.atomic_radius
            radii[atomic_number] = float(raw_radius) if raw_radius is not None else 1.0
            electronegativity[atomic_number] = float(element.data.get("X") or 0.0)
            parameters = BV_PARAMS.get(element)
            if parameters is not None:
                bv_r[atomic_number] = float(parameters["r"])
                bv_c[atomic_number] = float(parameters["c"])
                bv_valid[atomic_number] = True
        self._bv_parameter_numbers = frozenset(
            index for index, valid in enumerate(bv_valid.tolist()) if valid
        )
        self._atomic_number_by_symbol = {
            Element.from_Z(atomic_number).symbol: atomic_number
            for atomic_number in range(1, 119)
        }
        self._symbol_by_atomic_number = {
            atomic_number: symbol for symbol, atomic_number in self._atomic_number_by_symbol.items()
        }
        self._oxidation_assignments: dict[
            tuple[int, ...], tuple[tuple[int, float], ...] | None
        ] = {}
        self._pullback_audit_observations = 0
        self._maximum_pullback_residual = 0.0
        self.register_buffer("atomic_radii", radii, persistent=False)
        self.register_buffer("bv_r", bv_r, persistent=False)
        self.register_buffer("bv_c", bv_c, persistent=False)
        self.register_buffer("electronegativity", electronegativity, persistent=False)
        self.register_buffer("bv_valid", bv_valid, persistent=False)
        self.register_buffer(
            "periodic_shifts",
            torch.cartesian_prod(
                torch.tensor([-1.0, 0.0, 1.0]),
                torch.tensor([-1.0, 0.0, 1.0]),
                torch.tensor([-1.0, 0.0, 1.0]),
            ),
            persistent=False,
        )
        self.provenance = {
            "contract": self.contract,
            "bond_valence_parameters": "pymatgen.analysis.bond_valence.BV_PARAMS",
            "pymatgen_version": version("pymatgen"),
            "charge_policy": "unique_common_oxidation_state_neutral_assignment",
            "runtime_input_contract": {
                "allowed": list(RUNTIME_GUIDANCE_ALLOWED_INPUTS),
                "forbidden": list(RUNTIME_GUIDANCE_FORBIDDEN_INPUTS),
            },
            "coordination_term": {
                "enabled": False,
                "reason": "no_leak_free_coordination_prior_in_mp20_soft_physics_v1",
            },
        }
        if self.chemistry_prior is not None:
            self.provenance.update(
                prior=self.chemistry_prior.identity,
                charge_policy="uniform_marginal_over_common_element_neutral_assignments",
                coordination_term={"enabled": True, "missing_policy": "mask_unsupported_elements"},
                bond_term={"enabled": True, "missing_policy": "mask_unsupported_element_pairs"},
                jt_term={"enabled": False, "reason": "no_validated_electronic_state_labels"},
                property_term={"enabled": False, "reason": "no_calibrated_property_predictor"},
                normalization="per_atom_dimensionless; BVS divided by absolute oxidation state",
            )

    @property
    def runtime_diagnostics(self) -> dict[str, float | int | str]:
        return {
            "pullback_contract": "cartesian_gradient_then_exact_asu_jacobian_transpose",
            "pullback_audit_observations": self._pullback_audit_observations,
            "maximum_pullback_residual": self._maximum_pullback_residual,
            "nonfinite_guidance_failures_in_completed_run": 0,
        }

    def sigma_gate(self, sigma_by_structure: torch.Tensor) -> torch.Tensor:
        sigma = sigma_by_structure.float().clamp_min(1.0e-12)
        lower = torch.log(sigma.new_tensor(self.config.sigma_full_strength))
        upper = torch.log(sigma.new_tensor(self.config.sigma_cutoff))
        fraction = ((upper - torch.log(sigma)) / (upper - lower)).clamp(0.0, 1.0)
        return fraction.square() * (3.0 - 2.0 * fraction)

    def _pair_distances(
        self,
        fractional: torch.Tensor,
        lattice: torch.Tensor,
        pairs: torch.Tensor,
    ) -> torch.Tensor:
        delta = fractional[pairs[:, 1]] - fractional[pairs[:, 0]]
        images = delta[:, None, :] + self.periodic_shifts.to(delta)[None, :, :]
        cartesian = torch.einsum("pki,ij->pkj", images, lattice)
        return torch.linalg.vector_norm(cartesian, dim=-1).min(dim=1).values.clamp_min(1.0e-6)

    def _structure_energy(
        self,
        fractional: torch.Tensor,
        atomic_numbers: torch.Tensor,
        lattice: torch.Tensor,
        oxidation_assignment: tuple[tuple[int, float], ...] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        zero = fractional.sum() * 0.0
        if fractional.shape[0] < 2:
            return zero, zero, zero, False
        pairs = torch.combinations(
            torch.arange(fractional.shape[0], device=fractional.device), r=2
        )
        distance = self._pair_distances(fractional, lattice, pairs)
        left, right = pairs[:, 0], pairs[:, 1]
        radii = self.atomic_radii[atomic_numbers].to(fractional.dtype)
        reference = (radii[left] + radii[right]).clamp_min(0.5)
        overlap_limit = 0.55 * reference
        overlap = torch.nn.functional.softplus((overlap_limit - distance) / 0.08)
        overlap_energy = overlap.square().mean()

        unlike = atomic_numbers[left] != atomic_numbers[right]
        neighbor = torch.sigmoid((1.25 * reference - distance) / 0.12).detach()
        bond_terms = neighbor * unlike.to(distance.dtype) * ((distance - reference) / reference).square()
        bond_energy = bond_terms.sum() / (neighbor * unlike).sum().clamp_min(1.0)

        bvs_applicable = (
            oxidation_assignment is not None and len(oxidation_assignment) > 1
        )
        if not bvs_applicable:
            return overlap_energy, bond_energy, zero, False
        left, right = left[unlike], right[unlike]
        distance = distance[unlike]
        r_left, r_right = self.bv_r[atomic_numbers[left]], self.bv_r[atomic_numbers[right]]
        c_left, c_right = self.bv_c[atomic_numbers[left]], self.bv_c[atomic_numbers[right]]
        numerator = r_left * r_right * (torch.sqrt(c_left) - torch.sqrt(c_right)).square()
        denominator = (c_left * r_left + c_right * r_right).clamp_min(1.0e-6)
        bond_r = r_left + r_right - numerator / denominator
        valence = torch.exp(((bond_r - distance) / 0.31).clamp(-20.0, 20.0))
        x = self.electronegativity[atomic_numbers]
        sign = torch.where(x[left] < x[right], 1.0, -1.0).to(valence.dtype)
        sums = torch.zeros(fractional.shape[0], device=fractional.device, dtype=fractional.dtype)
        sums.index_add_(0, left, sign * valence)
        sums.index_add_(0, right, -sign * valence)
        target_lookup = torch.zeros_like(self.bv_r, dtype=fractional.dtype)
        for atomic_number, value in oxidation_assignment:
            target_lookup[atomic_number] = value
        target = target_lookup[atomic_numbers]
        return overlap_energy, bond_energy, (sums - target).square().mean(), True

    def _oxidation_assignment(
        self, batch: PackedASUBatch, structure: int
    ) -> tuple[tuple[int, float], ...] | None:
        numbers = tuple(
            sorted(
                self._atomic_number_by_symbol[symbol]
                for symbol, count in batch.hard_conditions[
                    structure
                ].reduced_composition
                for _ in range(count)
            )
        )
        if numbers not in self._oxidation_assignments:
            assignment = _neutral_oxidation_assignment(numbers)
            if assignment is not None and any(
                number not in self._bv_parameter_numbers for number, _ in assignment
            ):
                assignment = None
            self._oxidation_assignments[numbers] = assignment
        return self._oxidation_assignments[numbers]

    def energy_from_fractional(
        self, batch: PackedASUBatch, fractional: torch.Tensor
    ) -> PhysicsEnergy:
        if fractional.shape != (batch.num_atoms, 3):
            raise ValueError("fractional coordinates must have shape [atoms,3]")
        overlap, bond, bvs, applicable, coordination, preference, coverage, bond_coverage = ([] for _ in range(8))
        atom_start = 0
        for structure, condition in enumerate(batch.hard_conditions):
            atom_stop = atom_start + condition.group_num_atoms
            indices = slice(atom_start, atom_stop)
            if self.chemistry_prior is None:
                values = self._structure_energy(
                    fractional[indices], batch.atom_types[indices], batch.lattice[structure],
                    self._oxidation_assignment(batch, structure),
                )
                zero = fractional[indices].sum() * 0
                values = (*values, zero, zero, zero, zero)
            else:
                numbers = tuple(self._atomic_number_by_symbol[e]
                                for e, count in batch.hard_conditions[structure].reduced_composition
                                for _ in range(count))
                values = periodic_energies(
                    self, fractional[indices], batch.atom_types[indices], batch.lattice[structure],
                    _neutral_oxidation_assignments(numbers),
                )
            overlap.append(values[0])
            bond.append(values[1])
            bvs.append(values[2])
            applicable.append(values[3])
            coordination.append(values[4])
            preference.append(values[5])
            coverage.append(values[6])
            bond_coverage.append(values[7])
            atom_start = atom_stop
        if atom_start != batch.num_atoms:
            raise ValueError("hard-condition atom counts do not span the packed batch")
        overlap_tensor = torch.stack(overlap)
        bond_tensor = torch.stack(bond)
        bvs_tensor = torch.stack(bvs)
        applicable_tensor = torch.tensor(applicable, device=fractional.device, dtype=torch.bool)
        total = (
            self.config.overlap_weight * overlap_tensor
            + self.config.bond_radius_weight * bond_tensor
            + self.config.bond_valence_weight
            * bvs_tensor
            * applicable_tensor.to(bvs_tensor.dtype)
            + self.config.coordination_weight * torch.stack(coordination)
            + self.config.preference_weight * torch.stack(preference)
        )
        return PhysicsEnergy(
            total, overlap_tensor, bond_tensor, bvs_tensor, applicable_tensor,
            torch.stack(coordination), torch.stack(preference), torch.stack(coverage),
            torch.stack(bond_coverage),
        )

    def energy(self, batch: PackedASUBatch, u: torch.Tensor) -> PhysicsEnergy:
        return self.energy_from_fractional(batch, batch.expand(u))

    def summarize(self, batch, u):
        with torch.no_grad():
            energy = self.energy(batch, u)
            fractional = batch.expand(u)
            coordination_values = []
            atom_start = 0
            for structure, condition in enumerate(batch.hard_conditions):
                atom_stop = atom_start + condition.group_num_atoms
                indices = slice(atom_start, atom_stop)
                numbers = batch.atom_types[indices]
                reduced_numbers = tuple(
                    self._atomic_number_by_symbol[element]
                    for element, count in condition.reduced_composition
                    for _ in range(count)
                )
                coordination = periodic_coordination(
                    self,
                    fractional[indices],
                    numbers,
                    batch.lattice[structure],
                    _neutral_oxidation_assignments(tuple(sorted(reduced_numbers))),
                )[-1]
                coordination_values.append({
                    self._symbol_by_atomic_number[int(number)]: coordination[numbers == number]
                    .detach().cpu().tolist()
                    for number in torch.unique(numbers).tolist()
                })
                atom_start = atom_stop
        summary = {name: tensor.detach().cpu().tolist() for name, tensor in vars(energy).items()}
        summary["coordination_values_by_element"] = coordination_values
        return summary

    def target_relative_loss(
        self,
        batch: PackedASUBatch,
        predicted_u0: torch.Tensor,
        sigma_by_structure: torch.Tensor,
    ) -> PhysicsLoss:
        predicted = self.energy(batch, predicted_u0)
        with torch.no_grad():
            target = self.energy(batch, batch.require_target())
        margin = self.config.target_relative_margin
        overlap = torch.relu(predicted.overlap_by_structure - target.overlap_by_structure - margin)
        bond = torch.relu(predicted.bond_radius_by_structure - target.bond_radius_by_structure - margin)
        bvs = torch.relu(predicted.bond_valence_by_structure - target.bond_valence_by_structure - margin)
        bvs = bvs * predicted.bond_valence_applicable.to(bvs.dtype)
        cn = torch.relu(predicted.coordination_by_structure - target.coordination_by_structure - margin)
        combined = (
            self.config.overlap_weight * overlap
            + self.config.bond_radius_weight * bond
            + self.config.bond_valence_weight * bvs
            + self.config.coordination_weight * cn
        )
        gate = self.sigma_gate(sigma_by_structure).to(combined.dtype)
        denominator = gate.sum().clamp_min(1.0)
        return PhysicsLoss(
            loss=(gate * combined).sum() / denominator,
            overlap_excess=(gate * overlap).sum() / denominator,
            bond_radius_excess=(gate * bond).sum() / denominator,
            bond_valence_excess=(gate * bvs).sum() / denominator,
            bond_valence_applicable_fraction=(
                predicted.bond_valence_applicable.float().mean()
            ),
            coordination_excess=(gate * cn).sum() / denominator,
            coordination_coverage=predicted.coordination_coverage.mean(),
        )

    def cartesian_pullback_covector(
        self,
        batch: PackedASUBatch,
        u: torch.Tensor,
        sigma_by_structure: torch.Tensor,
    ) -> torch.Tensor:
        """Return J_Phi^T grad_x E after a Cartesian-gradient boundary."""

        if u.shape != batch.parameter_shape:
            raise ValueError("guidance state must match packed ASU parameters")
        if batch.num_parameters == 0:
            return torch.zeros_like(u)

        with torch.enable_grad(), torch.autocast(
            device_type=u.device.type, enabled=False
        ):
            audit_pullback = self._pullback_audit_observations == 0
            asu = u.detach().float().requires_grad_(audit_pullback)
            fractional = batch.expand(asu)
            if not audit_pullback:
                fractional = fractional.requires_grad_(True)
            energy = self.energy_from_fractional(batch, fractional)
            gate = self.sigma_gate(sigma_by_structure).to(energy.total_by_structure)
            objective = (gate * energy.total_by_structure).sum()
            if audit_pullback:
                fractional_covector, direct_asu_covector = torch.autograd.grad(
                    objective, (fractional, asu)
                )
            else:
                fractional_covector = torch.autograd.grad(objective, fractional)[0]
            lattice = batch.lattice[batch.atom_to_structure].float()
            cartesian_covector = torch.linalg.solve(
                lattice, fractional_covector.unsqueeze(-1)
            ).squeeze(-1)
            pulled_back = batch.pullback(cartesian_covector)
            if audit_pullback:
                residual = torch.max(torch.abs(pulled_back - direct_asu_covector))
                self._maximum_pullback_residual = float(residual.detach().item())
                self._pullback_audit_observations = 1
        if not bool(torch.isfinite(pulled_back).all().item()):
            raise FloatingPointError("physics guidance produced a non-finite ASU covector")
        return pulled_back.to(u.dtype)


__all__ = [
    "PHYSICS_GUIDANCE_CONTRACT",
    "RUNTIME_GUIDANCE_ALLOWED_INPUTS",
    "RUNTIME_GUIDANCE_FORBIDDEN_INPUTS",
    "PhysicsEnergy",
    "PhysicsGuidance",
    "PhysicsGuidanceConfig",
    "PhysicsLoss",
]
