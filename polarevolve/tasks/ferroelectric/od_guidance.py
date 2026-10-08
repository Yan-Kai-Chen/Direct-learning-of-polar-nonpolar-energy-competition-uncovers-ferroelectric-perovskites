"""Experimental orbit-space OD guidance (FE4, non-default).

The child state is the Wyckoff-ASU torus of a fixed Hall/Wyckoff program, so an
OD target is enforced by a damped Gauss-Newton step in the orbit metric rather
than in 3N Cartesian space.  Updates stay inside the program manifold (exact
child symmetry), never move 0D orbits and act only in the row space of the
whitened OD Jacobian; the null space is left to the score prior.  See
``docs/history/fe_orbit_od_guidance.md`` sections 2--3 for the
selection rules and the operator; the promotion/deletion gate is FE4-E1.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import torch

from polarevolve.data.batch import PackedASUBatch
from polarevolve.diffusion.metric import (
    TRANSLATION_QUOTIENT_V1,
    apply_state_quotient_covector,
    metric_inverse_multiply,
)
from polarevolve.tasks.ferroelectric.od_features import (
    ALIGNMENT_FREE_OD_IDS,
    StructureDescriptors,
    complete_shell_shifts,
    alignment_free_od,
    structure_descriptors,
)
from polarevolve.tasks.ferroelectric.od_registry import OD_SCALES

ROLE_A, ROLE_B, ROLE_X, ROLE_NONE = 0, 1, 2, -1
LATTICE_ONLY_OD_IDS: tuple[str, ...] = ("OD19",)
A_ROLE_OD_IDS: tuple[str, ...] = ("OD03", "OD04", "OD05", "OD08", "OD18")
# Candidate-side terms that vanish identically when every center of a role is
# symmetry-equivalent (rule B) or sits on a 0D, hence non-polar, site (rule C).
HETEROGENEITY_OD_IDS: dict[int, tuple[str, ...]] = {ROLE_A: ("OD03",), ROLE_B: ("OD10", "OD13")}
OFFCENTER_OD_IDS: dict[int, tuple[str, ...]] = {ROLE_A: ("OD04",), ROLE_B: ("OD02",)}
# Rule E (algebraic dependence): OD06 = OD01 + mother constant and
# OD18 = |OD04| - |OD02| identically, so their rows add no rank and only
# re-weight B distortion or reintroduce the |.| cusp.
DEPENDENT_OD_IDS: dict[str, tuple[str, ...]] = {"OD06": ("OD01",), "OD18": ("OD02", "OD04")}


@dataclass(frozen=True)
class SelectionRules:
    """Program-determined OD support of one child structure."""

    frozen: Mapping[str, str]
    dependent: Mapping[str, str]
    active: tuple[str, ...]
    orbit_count_by_role: Mapping[int, int]
    polar_orbit_count_by_role: Mapping[int, int]


def role_ids_from_atomic_numbers(
    atom_types: torch.Tensor, roles_by_atomic_number: Mapping[int, int]
) -> torch.Tensor:
    """Map atomic numbers to A/B/X roles; unassigned species get ``ROLE_NONE``."""

    roles = torch.full_like(atom_types.reshape(-1).long(), ROLE_NONE)
    for atomic_number, role in roles_by_atomic_number.items():
        if role not in (ROLE_A, ROLE_B, ROLE_X):
            raise ValueError("roles must be A=0, B=1 or X=2")
        roles[atom_types.reshape(-1).long() == int(atomic_number)] = int(role)
    return roles


def _structure_atoms(batch: PackedASUBatch, structure: int) -> slice:
    return slice(int(batch.atom_ptr[structure]), int(batch.atom_ptr[structure + 1]))


def orbit_selection_rules(
    batch: PackedASUBatch, structure: int, role_ids: torch.Tensor
) -> SelectionRules:
    """Apply orbit-constancy, non-polar-site and algebraic-dependence rules."""

    atoms = _structure_atoms(batch, structure)
    roles = role_ids.reshape(-1).long().to(batch.atom_to_orbit.device)
    if roles.numel() != atoms.stop - atoms.start:
        raise ValueError("role ids must cover exactly the structure atoms")
    orbits = batch.atom_to_orbit[atoms]
    frozen: dict[str, str] = {name: "lattice_only" for name in LATTICE_ONLY_OD_IDS}
    counts: dict[int, int] = {}
    polar: dict[int, int] = {}
    for role in (ROLE_A, ROLE_B, ROLE_X):
        role_orbits = torch.unique(orbits[roles == role])
        counts[role] = int(role_orbits.numel())
        polar[role] = int((batch.orbit_dimensions[role_orbits] > 0).sum())
    if counts[ROLE_B] == 0 or counts[ROLE_X] == 0:
        raise ValueError("OD guidance requires B and X roles in the child")
    if counts[ROLE_A] == 0:
        frozen.update({name: "role_absent" for name in A_ROLE_OD_IDS})
    for role in (ROLE_A, ROLE_B):
        if counts[role] == 1:
            for name in HETEROGENEITY_OD_IDS[role]:
                frozen.setdefault(name, "single_orbit_heterogeneity")
        if counts[role] and polar[role] == 0:
            for name in OFFCENTER_OD_IDS[role]:
                frozen.setdefault(name, "nonpolar_site_offcenter")
    if "OD02" in frozen and "OD04" in frozen:
        frozen.setdefault("OD18", "nonpolar_site_offcenter")
    dependent = {
        name: "algebraic_dependence"
        for name, basis in DEPENDENT_OD_IDS.items()
        if name not in frozen and any(item not in frozen for item in basis)
    }
    active = tuple(
        name for name in ALIGNMENT_FREE_OD_IDS if name not in frozen and name not in dependent
    )
    return SelectionRules(frozen, dependent, active, counts, polar)


@dataclass(frozen=True)
class ODTarget:
    """Oracle or planner OD target for one structure of a packed batch."""

    values: Mapping[str, float]
    mother: StructureDescriptors
    role_ids: torch.Tensor
    active: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.active or set(self.active) - set(ALIGNMENT_FREE_OD_IDS):
            raise ValueError("active ODs must be a non-empty alignment-free subset")
        missing = set(self.active) - set(self.values)
        if missing:
            raise ValueError(f"OD target is missing active values: {sorted(missing)}")


def whitened_residual(
    fractional: torch.Tensor, lattice: torch.Tensor, target: ODTarget
) -> torch.Tensor:
    """Return ``(OD - OD*) / scale`` over the target's active set."""

    values = alignment_free_od(
        target.mother, structure_descriptors(fractional, lattice, target.role_ids,
                                             periodic_contract=target.mother.periodic_contract)
    )
    return torch.stack(
        [
            (values[name] - float(target.values[name])) / OD_SCALES[name]
            for name in target.active
        ]
    )


def _residual_function(target: ODTarget, periodic_shifts=None) -> Callable:
    """Whitened residual of one topology as a pure function of (frac, lattice, values)."""

    scales = [float(OD_SCALES[name]) for name in target.active]

    def residual(frac: torch.Tensor, lattice: torch.Tensor, values: torch.Tensor):
        ods = alignment_free_od(target.mother, structure_descriptors(
            frac, lattice, target.role_ids, periodic_contract=target.mother.periodic_contract,
            periodic_shifts=periodic_shifts,
        ))
        stacked = torch.stack(
            [(ods[name] - values[row]) / scales[row] for row, name in enumerate(target.active)]
        )
        return stacked, stacked

    return residual


def _target_values(target: ODTarget, device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [float(target.values[name]) for name in target.active], device=device, dtype=torch.float32
    )


def _group(
    batch: PackedASUBatch,
    template: ODTarget,
    structures: Sequence[int],
    targets: Sequence[ODTarget],
    device: torch.device,
) -> tuple[ODTarget, torch.Tensor, torch.Tensor, torch.Tensor]:
    starts = batch.atom_ptr[list(structures)].to(device)
    count = int(batch.atom_ptr[structures[0] + 1] - batch.atom_ptr[structures[0]])
    atoms = starts[:, None] + torch.arange(count, device=device)
    values = torch.stack([_target_values(target, device) for target in targets])
    return template, torch.tensor(list(structures), device=device), values, atoms


def _jacobian_rows(
    batch: PackedASUBatch,
    state: torch.Tensor,
    groups: Sequence[tuple[ODTarget, torch.Tensor, torch.Tensor, torch.Tensor]],
    slots: int,
    *,
    state_quotient: str,
    metric_convention: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Residuals ``[B, slots]`` and quotient-projected ASU Jacobian rows ``[slots, P]``.

    Each group holds structures with one topology (same target template, roles and
    atom count): ``(template, structure_indices[R], target_values[R, K], atoms[R, n])``.  The
    per-structure Jacobian is one ``vmap(jacrev)`` per group and the chain rule
    through ``batch.expand`` is one VJP per slot, so the cost does not grow with
    the number of guided structures.  Structures own disjoint parameter blocks,
    hence row ``k`` stacks slot ``k`` of every structure.
    """

    with torch.enable_grad(), torch.autocast(device_type=state.device.type, enabled=False):
        state = state.detach().to(batch.lattice.dtype)
        fractional, pullback = torch.func.vjp(batch.expand, state)
        residuals = torch.zeros((batch.batch_size, slots), device=state.device, dtype=torch.float64)
        cotangent = fractional.new_zeros((slots, *fractional.shape))
        for template, structures, values, atoms in groups:
            shifts = None
            if template.mother.periodic_contract == "complete_images_v2":
                shifts = complete_shell_shifts(fractional[atoms], batch.lattice[structures], template.role_ids)
            jacobian, residual = torch.func.vmap(
                torch.func.jacrev(_residual_function(template, shifts), has_aux=True)
            )(fractional[atoms].detach(), batch.lattice[structures], values)
            rows = len(template.active)
            residuals[structures, :rows] = residual.to(torch.float64)
            cotangent[:rows, atoms.reshape(-1)] = (
                jacobian.permute(1, 0, 2, 3).reshape(rows, -1, 3).to(cotangent.dtype)
            )
        gradients = [pullback(cotangent[row])[0] for row in range(slots)]
    rows = torch.stack(
        [
            apply_state_quotient_covector(
                batch, gradient, state_quotient, metric_convention=metric_convention
            ).to(torch.float64)
            for gradient in gradients
        ]
    )
    return residuals, rows


def od_jacobian(
    batch: PackedASUBatch,
    u: torch.Tensor,
    structure: int,
    target: ODTarget,
    *,
    state_quotient: str = TRANSLATION_QUOTIENT_V1,
    metric_convention: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the whitened residual and its exact ASU Jacobian (rows are covectors)."""

    group = _group(batch, target, [structure], [target], u.device)
    residuals, rows = _jacobian_rows(
        batch, u, [group], len(target.active),
        state_quotient=state_quotient, metric_convention=metric_convention,
    )
    return residuals[structure], rows


def metric_gram(
    batch: PackedASUBatch, jacobian: torch.Tensor, *, metric_convention: str
) -> torch.Tensor:
    """Return ``J G^-1 J^T`` in the requested diffusion metric."""

    inverse = torch.stack(
        [
            metric_inverse_multiply(batch, row.to(torch.float64), metric_convention=metric_convention)
            for row in jacobian
        ]
    )
    return jacobian @ inverse.T


def identifiability(
    batch: PackedASUBatch,
    u: torch.Tensor,
    structure: int,
    target: ODTarget,
    *,
    metric_convention: str,
    relative_tolerance: float = 1.0e-6,
) -> dict[str, object]:
    """Singular values of ``J G^-1/2`` and per-orbit Jacobian block norms."""

    _, jacobian = od_jacobian(
        batch, u, structure, target, metric_convention=metric_convention
    )
    gram = metric_gram(batch, jacobian, metric_convention=metric_convention)
    singular = torch.linalg.eigvalsh(0.5 * (gram + gram.T)).clamp_min(0.0).sqrt()
    singular = torch.flip(singular, dims=(0,))
    scale = float(singular.max()) if singular.numel() else 0.0
    rank = int((singular > relative_tolerance * max(scale, 1.0e-30)).sum()) if scale else 0
    orbit_norms: dict[int, list[float]] = {}
    first = int(batch.orbit_ptr[structure])
    for orbit in range(first, int(batch.orbit_ptr[structure + 1])):
        start, stop = int(batch.u_ptr[orbit]), int(batch.u_ptr[orbit + 1])
        if stop > start:
            block = jacobian[:, start:stop]
            orbit_norms[orbit - first] = torch.linalg.vector_norm(block, dim=1).tolist()
    return {
        "active": list(target.active),
        "singular_values": singular.tolist(),
        "rank": rank,
        "row_norms": torch.linalg.vector_norm(jacobian, dim=1).tolist(),
        "orbit_row_norms": orbit_norms,
    }


Denoiser = Callable[[PackedASUBatch, torch.Tensor, torch.Tensor], torch.Tensor]


class OrbitODGuidance:
    """GuidanceCovector returning ``J^T (J G^-1 J^T + lambda I)^-1 r_eff`` per structure.

    ``lambda = damping * mean(eig) + s^2 / sigma^2``.  With tolerance ``s = 0`` this
    is the FE4 damped Gauss-Newton (Levenberg-Marquardt) solve.  With ``s > 0`` the
    OD is a soft observation with whitened noise ``s`` under the diffusion prior
    ``sigma^2 G^-1`` (Kalman damping), and ``dead_zone`` soft-thresholds the
    residual, ``r_eff = r * max(0, 1 - s / rms(r))``, so a structure whose OD is
    already within tolerance is not pushed (FE4c plan section 2).

    With sampler guidance scale ``eta`` the applied step is the minimum-norm
    Levenberg-Marquardt update that removes a fraction ``eta`` of the linearized
    OD residual per reverse step.  ``targets`` holds one entry per packed
    structure; ``None`` leaves a structure unguided and ``scales`` multiplies a
    structure's covector (a per-structure ``eta`` ratio), so replicas with
    different lanes share one sampler batch.  ``denoiser`` optionally evaluates
    the OD at a Tweedie estimate of the clean state (identity-Jacobian
    approximation).  All structures are solved together without host syncs.
    """

    def __init__(
        self,
        targets: Sequence[ODTarget | None],
        *,
        metric_convention: str,
        state_quotient: str = TRANSLATION_QUOTIENT_V1,
        sigma_full_strength: float = 0.05,
        sigma_cutoff: float = 0.30,
        damping: float = 1.0e-3,
        denoiser: Denoiser | None = None,
        scales: Sequence[float] | None = None,
        tolerance: Sequence[float] | None = None,
        dead_zone: Sequence[bool] | None = None,
    ) -> None:
        if not 0.0 < sigma_full_strength < sigma_cutoff or damping <= 0.0:
            raise ValueError("OD guidance needs 0 < sigma_full < sigma_cutoff and damping > 0")
        self.targets = tuple(targets)
        if scales is not None and len(scales) != len(self.targets):
            raise ValueError("OD guidance scales need one value per target")
        self.scales = None if scales is None else tuple(float(value) for value in scales)
        self.tolerance = self._per_structure(tolerance, "tolerance")
        self.dead_zone = tuple(bool(value) for value in self._per_structure(dead_zone, "dead_zone"))
        if any(value < 0.0 for value in self.tolerance):
            raise ValueError("OD tolerance must be >= 0")
        self.metric_convention = metric_convention
        self.state_quotient = state_quotient
        self.sigma_full_strength = sigma_full_strength
        self.sigma_cutoff = sigma_cutoff
        self.damping = damping
        self.denoiser = denoiser
        self._plan: tuple | None = None

    def _per_structure(self, values: Sequence[float] | None, name: str) -> tuple[float, ...]:
        if values is None:
            return (0.0,) * len(self.targets)
        if len(values) != len(self.targets):
            raise ValueError(f"OD guidance {name} needs one value per target")
        return tuple(float(value) for value in values)

    def sigma_gate(self, sigma: torch.Tensor) -> torch.Tensor:
        value = sigma.to(torch.float64).clamp_min(1.0e-12)
        lower = torch.log(value.new_tensor(self.sigma_full_strength))
        upper = torch.log(value.new_tensor(self.sigma_cutoff))
        fraction = ((upper - torch.log(value)) / (upper - lower)).clamp(0.0, 1.0)
        return fraction.square() * (3.0 - 2.0 * fraction)

    def _groups(self, batch: PackedASUBatch, device: torch.device):
        """Group structures by topology once per packed batch."""

        if self._plan is not None and self._plan[0] == id(batch):
            return self._plan[1:]
        counts = (batch.atom_ptr[1:] - batch.atom_ptr[:-1]).tolist()
        members: dict[tuple, list[int]] = {}
        for structure, target in enumerate(self.targets):
            if target is not None:
                key = (id(target.mother), id(target.role_ids), target.active, counts[structure])
                members.setdefault(key, []).append(structure)
        groups = [
            _group(batch, self.targets[indices[0]], indices, [self.targets[i] for i in indices], device)
            for indices in members.values()
        ]
        slots = max((len(target.active) for target in self.targets if target is not None), default=0)
        active = torch.zeros((batch.batch_size, max(slots, 1)), device=device, dtype=torch.float64)
        for structure, target in enumerate(self.targets):
            if target is not None:
                active[structure, : len(target.active)] = 1.0
        weight = torch.tensor(
            self.scales or [1.0] * batch.batch_size, device=device, dtype=torch.float64
        )
        tolerance = torch.tensor(self.tolerance, device=device, dtype=torch.float64)
        dead_zone = torch.tensor(self.dead_zone, device=device, dtype=torch.bool)
        self._plan = (id(batch), groups, slots, active, weight, tolerance, dead_zone)
        return groups, slots, active, weight, tolerance, dead_zone

    def __call__(
        self, batch: PackedASUBatch, u: torch.Tensor, sigma_by_structure: torch.Tensor
    ) -> torch.Tensor:
        if len(self.targets) != batch.batch_size:
            raise ValueError("OD guidance needs one target per structure")
        groups, slots, active, weight, tolerance, dead_zone = self._groups(batch, u.device)
        if not groups:
            return torch.zeros_like(u)
        state = u if self.denoiser is None else self.denoiser(batch, u, sigma_by_structure)
        residual, rows = _jacobian_rows(
            batch, state, groups, slots,
            state_quotient=self.state_quotient, metric_convention=self.metric_convention,
        )
        inverse = torch.stack(
            [metric_inverse_multiply(batch, row, metric_convention=self.metric_convention) for row in rows]
        )
        owner = batch.parameter_to_structure
        gram = torch.zeros((batch.batch_size, slots * slots), device=u.device, dtype=torch.float64)
        gram.index_add_(0, owner, (rows[:, None, :] * inverse[None, :, :]).reshape(slots * slots, -1).T)
        gram = gram.reshape(batch.batch_size, slots, slots) * active[:, :, None] * active[:, None, :]
        count = active.sum(dim=1)
        trace = torch.diagonal(gram, dim1=-2, dim2=-1).sum(dim=1) / count.clamp_min(1.0)
        sigma = sigma_by_structure.reshape(-1).to(torch.float64)
        ridge = self.damping * trace + tolerance.square() / sigma.square().clamp_min(1.0e-24) + 1.0e-12
        residual = residual * active
        rms = (residual.square().sum(dim=1) / count.clamp_min(1.0)).sqrt()
        shrink = torch.where(
            dead_zone, (1.0 - tolerance / rms.clamp_min(1.0e-30)).clamp_min(0.0), torch.ones_like(rms)
        )
        system = gram + torch.diag_embed(ridge[:, None] * active + (1.0 - active))
        coefficients = torch.linalg.solve(system, (residual * shrink[:, None]).unsqueeze(-1)).squeeze(-1)
        usable = (count > 0) & (trace > 0.0)
        scale = self.sigma_gate(sigma) * weight * usable
        covector = (rows * (coefficients * scale[:, None])[owner].T).sum(dim=0)
        if not bool(torch.isfinite(covector).all().item()):
            raise FloatingPointError("OD guidance produced a non-finite ASU covector")
        return covector.to(u.dtype)


__all__ = [
    "A_ROLE_OD_IDS",
    "LATTICE_ONLY_OD_IDS",
    "ODTarget",
    "OrbitODGuidance",
    "ROLE_A",
    "ROLE_B",
    "ROLE_NONE",
    "ROLE_X",
    "SelectionRules",
    "identifiability",
    "metric_gram",
    "od_jacobian",
    "orbit_selection_rules",
    "role_ids_from_atomic_numbers",
    "whitened_residual",
]
