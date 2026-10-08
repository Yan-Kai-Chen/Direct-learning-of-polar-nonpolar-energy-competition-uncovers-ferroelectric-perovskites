"""Differentiable ferroelectric OD observables and legal child-ASU modes.

Ported from the legacy ``ferro_transition/od_features.py`` implementation.
Contrasts are always evaluated as ``candidate - source``. Role assignment uses
``A=0, B=1, X=2`` with A-X cage coordination 12 and B-X octahedral
coordination 6 (perovskite-specific).  OD17 and OD20 consume per-atom
minimum-image deltas and therefore require both structures to share atom
count and order; the sidecar alignment contract guarantees this.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from polarevolve.data.batch import PackedASUBatch
from polarevolve.crystal.periodic import PERIODIC_IMAGES
from polarevolve.diffusion.metric import (
    MEMBER_SUM_CARTESIAN_V1,
    TRANSLATION_QUOTIENT_V1,
    apply_state_quotient_tangent,
    limit_cartesian_rms,
    metric_inverse_multiply,
)
from polarevolve.diffusion.noise import wrap_unit_interval
from polarevolve.tasks.ferroelectric.od_registry import (
    B_OCT_OPERATION_IDS,
    OD_SCALES,
    PLANNER_OD_IDS,
)

EPS = 1.0e-7
PERIODIC_IMAGE_RANGE = (-1.0, 0.0, 1.0)
OCTAHEDRAL_MIN_OPPOSITE_PAIRS = 2
OCTAHEDRAL_MIN_ORTHOGONAL_PAIRS = 8


def minimum_image_delta(target: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    return target - source - torch.round(target - source)


def _cartesian_delta(
    target_frac: torch.Tensor,
    source_frac: torch.Tensor,
    lattice: torch.Tensor,
    periodic_contract: str = "legacy_27_v1",
) -> torch.Tensor:
    if periodic_contract == "complete_images_v2":
        return PERIODIC_IMAGES.nearest(target_frac - source_frac, lattice.expand(len(source_frac), -1, -1))
    return minimum_image_delta(target_frac, source_frac) @ lattice.float()


def _role(role_ids: torch.Tensor, value: int) -> torch.Tensor:
    return torch.nonzero(role_ids.reshape(-1).long() == value, as_tuple=False).flatten()


def _nearest_shell(
    frac_coords: torch.Tensor,
    lattice: torch.Tensor,
    centers: torch.Tensor,
    ligands: torch.Tensor,
    coordination: int,
    periodic_shifts: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if centers.numel() == 0 or ligands.numel() == 0:
        empty = torch.zeros(
            (0, max(int(coordination), 1), 3), device=frac_coords.device, dtype=torch.float32
        )
        return empty, empty[..., 0]
    base_delta = minimum_image_delta(
        frac_coords[ligands][None, :, :],
        frac_coords[centers][:, None, :],
    )
    # A plain factory (not ``new_tensor``) keeps the shell search vmap-compatible
    # for batched OD guidance; the constant is identical.
    image_range = torch.tensor(
        PERIODIC_IMAGE_RANGE, device=frac_coords.device, dtype=frac_coords.dtype
    )
    shifts = (torch.cartesian_prod(image_range, image_range, image_range)
              if periodic_shifts is None else periodic_shifts)
    delta = (
        base_delta[:, :, None, :].float()
        + shifts[None, None, :, :].float()
    )
    cart = (delta @ lattice.float()).reshape(int(centers.numel()), -1, 3)
    distances = torch.linalg.vector_norm(cart, dim=-1)
    count = min(max(int(coordination), 1), int(cart.shape[1]))
    values, indices = torch.topk(distances, k=count, dim=1, largest=False)
    vectors = torch.gather(
        cart,
        1,
        indices.unsqueeze(-1).expand(-1, -1, 3),
    )
    return vectors, values


def complete_shell_shifts(frac, lattice, role_ids):
    """Bound complete k-nearest shells by existing candidate distances, before vmap."""
    roles = [_role(role_ids, value) for value in (0, 1, 2)]
    def radius(x, cell):
        values = [_nearest_shell(x, cell, roles[a], roles[b], count)[1].amax()
                  for a, b, count in ((0, 2, 12), (1, 2, 6), (2, 1, 2))
                  if roles[a].numel() and roles[b].numel()]
        return torch.stack(values).amax()
    if frac.ndim == 2:
        bound = radius(frac.detach(), lattice.detach())
    else:
        bound = torch.func.vmap(radius)(frac.detach(), lattice.detach()).amax()
    # The kth old candidate is an upper bound on the kth true neighbor.
    return PERIODIC_IMAGES.shifts(lattice, max(float(bound.cpu()) * (1 + 1e-6), 1e-6))


@dataclass(frozen=True)
class ShellFeatures:
    offcenter: torch.Tensor
    bond_distortion: torch.Tensor
    angular_distortion: torch.Tensor
    volume_proxy: torch.Tensor
    local_heterogeneity: torch.Tensor
    coordination_heterogeneity: torch.Tensor
    orientation_tensor: torch.Tensor


@dataclass(frozen=True)
class OrbitModeProjection:
    """Least-squares projection of an aligned displacement onto child ASU modes."""

    parameter_tangent: torch.Tensor
    projected_cartesian: torch.Tensor
    residual_cartesian: torch.Tensor
    source_rms_angstrom: torch.Tensor
    projected_rms_angstrom: torch.Tensor
    residual_rms_angstrom: torch.Tensor
    captured_fraction: torch.Tensor
    orbit_rms_angstrom: torch.Tensor


def _indexed_cartesian_rms(
    vectors: torch.Tensor, index: torch.Tensor, count: int
) -> torch.Tensor:
    squared = vectors.float().square().sum(dim=-1)
    totals = squared.new_zeros((count,))
    totals.index_add_(0, index, squared)
    populations = torch.bincount(index, minlength=count).to(totals.dtype)
    return torch.sqrt(totals / populations.clamp_min(1.0))


def project_cartesian_mode_to_asu(
    batch: PackedASUBatch,
    atom_cartesian_displacement: torch.Tensor,
    *,
    state_quotient: str = TRANSLATION_QUOTIENT_V1,
) -> OrbitModeProjection:
    """Project one aligned atomic mode into the legal Wyckoff-ASU tangent.

    The solve minimizes member-sum Cartesian residual. Applying the state
    quotient removes a representable common translation after projection.
    Gold displacements passed here are supervision or audit labels, never
    inference conditions.
    """

    displacement = torch.as_tensor(
        atom_cartesian_displacement,
        device=batch.lattice.device,
        dtype=batch.lattice.dtype,
    )
    if displacement.shape != (batch.num_atoms, 3):
        raise ValueError("atomic mode must have shape [atoms,3]")
    if not bool(torch.isfinite(displacement).all().item()):
        raise FloatingPointError("atomic mode contains non-finite values")
    covector = batch.pullback(displacement)
    tangent = metric_inverse_multiply(
        batch, covector, metric_convention=MEMBER_SUM_CARTESIAN_V1
    )
    tangent = apply_state_quotient_tangent(
        batch,
        tangent,
        state_quotient,
        metric_convention=MEMBER_SUM_CARTESIAN_V1,
    )
    projected = batch.pushforward(tangent)
    residual = displacement - projected
    source_rms = _indexed_cartesian_rms(
        displacement, batch.atom_to_structure, batch.batch_size
    )
    projected_rms = _indexed_cartesian_rms(
        projected, batch.atom_to_structure, batch.batch_size
    )
    residual_rms = _indexed_cartesian_rms(
        residual, batch.atom_to_structure, batch.batch_size
    )
    captured = torch.where(
        source_rms > 1.0e-12,
        1.0 - residual_rms.square() / source_rms.square().clamp_min(1.0e-24),
        torch.ones_like(source_rms),
    ).clamp(0.0, 1.0)
    return OrbitModeProjection(
        parameter_tangent=tangent,
        projected_cartesian=projected,
        residual_cartesian=residual,
        source_rms_angstrom=source_rms,
        projected_rms_angstrom=projected_rms,
        residual_rms_angstrom=residual_rms,
        captured_fraction=captured,
        orbit_rms_angstrom=_indexed_cartesian_rms(
            projected, batch.atom_to_orbit, batch.num_orbits
        ),
    )


def project_fractional_mode_to_asu(
    batch: PackedASUBatch,
    atom_fractional_displacement: torch.Tensor,
    *,
    state_quotient: str = TRANSLATION_QUOTIENT_V1,
) -> OrbitModeProjection:
    """Convert an aligned fractional mode to Cartesian units and project it."""

    fractional = torch.as_tensor(
        atom_fractional_displacement,
        device=batch.lattice.device,
        dtype=batch.lattice.dtype,
    )
    if fractional.shape != (batch.num_atoms, 3):
        raise ValueError("fractional mode must have shape [atoms,3]")
    cartesian = torch.einsum(
        "ni,nij->nj", fractional, batch.lattice[batch.atom_to_structure]
    )
    return project_cartesian_mode_to_asu(
        batch, cartesian, state_quotient=state_quotient
    )


def apply_bounded_orbit_mode(
    batch: PackedASUBatch,
    u: torch.Tensor,
    parameter_tangent: torch.Tensor,
    amplitude: float | torch.Tensor,
    *,
    maximum_rms_angstrom: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply a signed orbit-mode coefficient with a Cartesian RMS step cap."""

    if u.shape != batch.parameter_shape or parameter_tangent.shape != batch.parameter_shape:
        raise ValueError("state and orbit mode must match packed ASU parameters")
    coefficients = torch.as_tensor(amplitude, device=u.device, dtype=u.dtype).reshape(-1)
    if coefficients.numel() == 1:
        coefficients = coefficients.expand(batch.batch_size)
    if coefficients.shape != (batch.batch_size,) or not bool(
        torch.isfinite(coefficients).all().item()
    ):
        raise ValueError("mode amplitude must be finite and scalar or per-structure")
    step = parameter_tangent * coefficients[batch.parameter_to_structure]
    bounded, rms, clipped = limit_cartesian_rms(
        batch, step, maximum_rms_angstrom
    )
    return wrap_unit_interval(u + bounded), rms, clipped


def _shell_features(
    frac_coords: torch.Tensor,
    lattice: torch.Tensor,
    centers: torch.Tensor,
    ligands: torch.Tensor,
    coordination: int,
    periodic_shifts: torch.Tensor | None = None,
) -> ShellFeatures:
    vectors, distances = _nearest_shell(
        frac_coords, lattice, centers, ligands, coordination, periodic_shifts
    )
    zero = torch.zeros((), device=frac_coords.device, dtype=torch.float32)
    if vectors.numel() == 0:
        return ShellFeatures(
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
            zero.repeat(3, 3),
        )
    offcenter_each = torch.linalg.vector_norm(vectors.mean(dim=1), dim=-1)
    distance_mean = distances.mean(dim=1).clamp_min(EPS)
    distortion_each = distances.std(dim=1, unbiased=False) / distance_mean

    normalized = F.normalize(vectors, dim=-1, eps=EPS)
    cosine = normalized @ normalized.transpose(-1, -2)
    shell_size = int(cosine.shape[-1])
    upper = torch.triu(
        torch.ones(
            (shell_size, shell_size),
            device=cosine.device,
            dtype=torch.bool,
        ),
        diagonal=1,
    )
    pair_cos = cosine[:, upper]
    # Octahedral/cage shells favor 90 and 180 degree relationships.  This
    # bounded residual is stable for non-octahedral environments as well.
    angular_each = torch.minimum(pair_cos.abs(), (pair_cos + 1.0).abs()).mean(dim=1)

    covariance = (
        vectors.transpose(-1, -2) @ vectors / max(int(vectors.shape[1]), 1)
    )
    identity = torch.eye(3, device=covariance.device, dtype=covariance.dtype)
    regularized = covariance.float() + EPS * identity
    volume_each = torch.linalg.det(regularized).clamp_min(EPS).sqrt()
    orientation_tensor = covariance.float() / torch.diagonal(
        covariance.float(), dim1=-2, dim2=-1
    ).sum(dim=-1, keepdim=True).clamp_min(EPS).unsqueeze(-1)

    return ShellFeatures(
        offcenter=offcenter_each.mean(),
        bond_distortion=distortion_each.mean(),
        angular_distortion=angular_each.mean(),
        volume_proxy=volume_each.mean(),
        local_heterogeneity=distortion_each.std(unbiased=False),
        coordination_heterogeneity=distances.mean(dim=1).std(unbiased=False),
        orientation_tensor=orientation_tensor.mean(dim=0),
    )


def _axial_anisotropy(lattice: torch.Tensor) -> torch.Tensor:
    singular = torch.linalg.svdvals(lattice.float()).clamp_min(EPS)
    return singular.max() / singular.min() - 1.0


def _framework_observables(
    frac_coords: torch.Tensor,
    lattice: torch.Tensor,
    b_indices: torch.Tensor,
    x_indices: torch.Tensor,
    periodic_shifts: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return bridge-strain heterogeneity and B-X-B angular disorder.

    Every X site is paired with its two nearest periodic B images.  Framework
    strain is the RMS relative mismatch of the two B-X distances and angular
    disorder is ``180 deg - mean(B-X-B angle)``.  Both are single-structure
    observables, so signed transition features are formed only by subtracting
    the NP value from the candidate value.
    """

    vectors, distances = _nearest_shell(
        frac_coords,
        lattice,
        x_indices,
        b_indices,
        coordination=2,
        periodic_shifts=periodic_shifts,
    )
    zero = torch.zeros((), device=frac_coords.device, dtype=torch.float32)
    if vectors.numel() == 0 or int(vectors.shape[1]) < 2:
        return zero, zero

    pair_mean = distances[:, :2].mean(dim=1).clamp_min(EPS)
    relative_mismatch = (distances[:, 0] - distances[:, 1]) / pair_mean
    strain_heterogeneity = relative_mismatch.norm() / relative_mismatch.numel()**0.5

    normalized = F.normalize(vectors[:, :2, :], dim=-1, eps=EPS)
    cosine = (normalized[:, 0] * normalized[:, 1]).sum(dim=-1)
    # The clamp keeps the derivative finite at exactly collinear bridges.
    angle_radians = torch.acos(cosine.clamp(-1.0 + 1.0e-6, 1.0 - 1.0e-6))
    angle_degrees = torch.rad2deg(angle_radians)
    angular_disorder = (180.0 - angle_degrees).mean()
    return strain_heterogeneity, angular_disorder


def _displacement_hhi(
    source_frac: torch.Tensor,
    candidate_frac: torch.Tensor,
    candidate_lattice: torch.Tensor,
    periodic_contract: str = "legacy_27_v1",
) -> torch.Tensor:
    displacement = _cartesian_delta(candidate_frac, source_frac, candidate_lattice, periodic_contract)
    magnitude = displacement.square().sum(dim=-1).clamp_min(0.0)
    weights = magnitude / magnitude.sum().clamp_min(EPS)
    return weights.square().sum()


def _orientational_change(
    source: ShellFeatures, candidate: ShellFeatures
) -> torch.Tensor:
    return torch.linalg.matrix_norm(
        candidate.orientation_tensor - source.orientation_tensor
    ) / (3.0**0.5)


@dataclass(frozen=True)
class StructureDescriptors:
    """Single-structure shell statistics from which every alignment-free OD is formed."""

    a_shell: ShellFeatures
    b_shell: ShellFeatures
    framework_strain: torch.Tensor
    framework_disorder: torch.Tensor
    axial_anisotropy: torch.Tensor
    periodic_contract: str = "legacy_27_v1"


# OD17/OD20 use per-atom displacements and OD22 compares Cartesian shell frames;
# the remaining 15 planner ODs combine two independently computed descriptors.
ALIGNMENT_DEPENDENT_OD_IDS: tuple[str, ...] = ("OD17", "OD20", "OD22")
ALIGNMENT_FREE_OD_IDS: tuple[str, ...] = tuple(
    operation_id
    for operation_id in PLANNER_OD_IDS
    if operation_id not in ALIGNMENT_DEPENDENT_OD_IDS
)


def structure_descriptors(
    frac_coords: torch.Tensor,
    lattice: torch.Tensor,
    role_ids: torch.Tensor,
    *, periodic_contract: str = "legacy_27_v1",
    periodic_shifts: torch.Tensor | None = None,
) -> StructureDescriptors:
    """Compute the A/B/X shell statistics of one structure in its own lattice."""

    frac = frac_coords.float()
    cell = lattice.float()
    roles = role_ids.reshape(-1).long()
    if periodic_contract not in {"legacy_27_v1", "complete_images_v2"}:
        raise ValueError("unsupported OD periodic contract")
    if periodic_contract == "complete_images_v2" and periodic_shifts is None:
        periodic_shifts = complete_shell_shifts(frac, cell, roles)
    if periodic_contract == "legacy_27_v1" and periodic_shifts is not None:
        raise ValueError("complete shifts cannot be used under a legacy OD identity")
    a_indices = _role(roles, 0)
    b_indices = _role(roles, 1)
    x_indices = _role(roles, 2)
    if not b_indices.numel() or not x_indices.numel():
        raise ValueError("OD features require non-empty B and X role assignments")
    strain, disorder = _framework_observables(frac, cell, b_indices, x_indices, periodic_shifts)
    return StructureDescriptors(
        a_shell=_shell_features(frac, cell, a_indices, x_indices, coordination=12, periodic_shifts=periodic_shifts),
        b_shell=_shell_features(frac, cell, b_indices, x_indices, coordination=6, periodic_shifts=periodic_shifts),
        framework_strain=strain,
        framework_disorder=disorder,
        axial_anisotropy=_axial_anisotropy(cell),
        periodic_contract=periodic_contract,
    )


def alignment_free_od(
    source: StructureDescriptors, candidate: StructureDescriptors
) -> dict[str, torch.Tensor]:
    """Form the 15 alignment-free ODs as ``candidate - source`` contrasts.

    The source (non-polar parent) enters only through scalar descriptors, so no
    atom correspondence, common cell or frame alignment is required.
    """

    if source.periodic_contract != candidate.periodic_contract:
        raise ValueError("OD source and candidate periodic contracts differ")
    np_a, p_a = source.a_shell, candidate.a_shell
    np_b, p_b = source.b_shell, candidate.b_shell
    b_off_delta = p_b.offcenter - np_b.offcenter
    a_off_delta = p_a.offcenter - np_a.offcenter
    values = {
        "OD01": p_b.bond_distortion - np_b.bond_distortion,
        "OD02": b_off_delta,
        "OD03": p_a.coordination_heterogeneity - np_a.coordination_heterogeneity,
        "OD04": a_off_delta,
        "OD05": p_a.bond_distortion,
        "OD06": p_b.bond_distortion,
        "OD07": p_b.angular_distortion,
        "OD08": (p_a.volume_proxy - np_a.volume_proxy)
        / np_a.volume_proxy.abs().clamp_min(EPS),
        "OD09": (p_b.volume_proxy - np_b.volume_proxy)
        / np_b.volume_proxy.abs().clamp_min(EPS),
        "OD10": p_b.coordination_heterogeneity - np_b.coordination_heterogeneity,
        "OD11": candidate.framework_strain - source.framework_strain,
        "OD12": candidate.framework_disorder - source.framework_disorder,
        "OD13": p_b.local_heterogeneity - np_b.local_heterogeneity,
        "OD18": a_off_delta.abs() - b_off_delta.abs(),
        "OD19": candidate.axial_anisotropy - source.axial_anisotropy,
    }
    return {operation_id: values[operation_id] for operation_id in ALIGNMENT_FREE_OD_IDS}


def compute_od_features(
    source_frac: torch.Tensor,
    source_lattice: torch.Tensor,
    candidate_frac: torch.Tensor,
    candidate_lattice: torch.Tensor,
    role_ids: torch.Tensor,
    *, periodic_contract: str = "legacy_27_v1",
) -> dict[str, torch.Tensor]:
    """Compute the differentiable OD vector for one aligned pair.

    Only the 18 planner IDs are returned.  The source is the non-polar
    parent side and the candidate the polar subphase side; contrasts are
    ``candidate - source``.  OD17/OD20/OD22 additionally require the shared
    atom order and Cartesian frame guaranteed by the sidecar alignment.
    """

    source = structure_descriptors(source_frac, source_lattice, role_ids, periodic_contract=periodic_contract)
    candidate = structure_descriptors(candidate_frac, candidate_lattice, role_ids, periodic_contract=periodic_contract)
    values = alignment_free_od(source, candidate)
    b_off_delta = candidate.b_shell.offcenter - source.b_shell.offcenter
    a_off_delta = candidate.a_shell.offcenter - source.a_shell.offcenter
    displacement = _cartesian_delta(
        candidate_frac.float(), source_frac.float(), candidate_lattice.float(), periodic_contract
    )
    values["OD17"] = torch.linalg.vector_norm(displacement.mean(dim=0)) + 0.5 * (
        a_off_delta.abs() + b_off_delta.abs()
    )
    values["OD20"] = _displacement_hhi(
        source_frac.float(), candidate_frac.float(), candidate_lattice.float(), periodic_contract
    )
    values["OD22"] = _orientational_change(source.b_shell, candidate.b_shell)
    return {operation_id: values[operation_id] for operation_id in PLANNER_OD_IDS}


def od_applicability_mask(
    source_frac: torch.Tensor,
    source_lattice: torch.Tensor,
    role_ids: torch.Tensor,
) -> torch.Tensor:
    """Mask B-octahedral observables from non-octahedral clean-parent shells."""

    mask = source_frac.new_ones((len(PLANNER_OD_IDS),), dtype=torch.float32)
    b_indices, x_indices = _role(role_ids, 1), _role(role_ids, 2)
    vectors, _ = _nearest_shell(
        source_frac.float(), source_lattice.float(), b_indices, x_indices, 6
    )
    applicable = False
    if vectors.numel() and int(vectors.shape[1]) == 6:
        unit = F.normalize(vectors, dim=-1, eps=EPS)
        cosine = unit @ unit.transpose(-1, -2)
        upper = torch.triu(
            torch.ones((6, 6), device=cosine.device, dtype=torch.bool), diagonal=1
        )
        pairs = cosine[:, upper]
        opposite = (pairs < -0.8).sum(dim=1)
        orthogonal = (pairs.abs() < 0.25).sum(dim=1)
        valid = (opposite >= OCTAHEDRAL_MIN_OPPOSITE_PAIRS) & (
            orthogonal >= OCTAHEDRAL_MIN_ORTHOGONAL_PAIRS
        )
        applicable = bool(valid.all().item())
    if not applicable:
        for operation_id in B_OCT_OPERATION_IDS:
            mask[PLANNER_OD_IDS.index(operation_id)] = 0.0
    return mask


def od_vector(values: dict[str, torch.Tensor]) -> torch.Tensor:
    """Stack a per-pair OD mapping in the canonical planner order."""

    if set(values) != set(PLANNER_OD_IDS):
        raise ValueError("OD value mapping must cover exactly the planner 18")
    return torch.stack([values[operation_id] for operation_id in PLANNER_OD_IDS])


def normalized_squared_error(
    values: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """OD_SCALES-normalized squared error over the planner 18 vector."""

    scales = torch.as_tensor(
        [OD_SCALES[operation_id] for operation_id in PLANNER_OD_IDS],
        device=values.device,
        dtype=values.dtype,
    )
    residual = ((values - target) / scales).square()
    if mask is not None:
        active = mask.to(dtype=torch.bool, device=residual.device)
        return torch.where(active, residual, 0.0).sum() / active.sum().clamp_min(1)
    return residual.mean()


__all__ = [
    "ALIGNMENT_DEPENDENT_OD_IDS",
    "ALIGNMENT_FREE_OD_IDS",
    "EPS",
    "OrbitModeProjection",
    "ShellFeatures",
    "StructureDescriptors",
    "alignment_free_od",
    "apply_bounded_orbit_mode",
    "compute_od_features",
    "minimum_image_delta",
    "normalized_squared_error",
    "od_applicability_mask",
    "od_vector",
    "project_cartesian_mode_to_asu",
    "project_fractional_mode_to_asu",
    "structure_descriptors",
]
