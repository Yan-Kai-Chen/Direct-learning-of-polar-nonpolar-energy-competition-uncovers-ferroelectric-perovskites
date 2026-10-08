"""Periodic v2 chemistry descriptors and immutable train-only priors.

Directed center-neighbor edges include periodic self images. All energies are
per atom; the directed repulsion sum is halved. This is a dimensionless chemical
regularizer, not an interatomic potential or a formation-energy predictor.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from functools import lru_cache
from itertools import product
from pathlib import Path

import torch
from torch import nn
from polarevolve.crystal.periodic import PeriodicImages

CHEMISTRY_CONTRACT = "mp20_periodic_chemistry_v2"
PRIOR_SCHEMA = "mp20_training_chemistry_prior_v1"
CUTOFF = 4.5
DISTANCE_CENTERS = [0.1 * i for i in range(1, 46)]
CN_CENTERS = list(range(25))
CN_DEFINITION = "opposite_valence_if_ionic_else_all_radius_neighbors_v1"


@lru_cache(maxsize=512)
def neutral_oxidation_assignments(atomic_numbers):
    from pymatgen.core import Element

    counts = Counter(atomic_numbers)
    elements = tuple(sorted(counts))
    choices = []
    for atomic_number in elements:
        states = tuple(float(value) for value in Element.from_Z(atomic_number).common_oxidation_states)
        if not states:
            return ()
        choices.append(states)
    solutions = {
        tuple(zip(elements, values))
        for values in product(*choices)
        if abs(sum(counts[number] * value for number, value in zip(elements, values))) < 1.0e-8
    }
    return tuple(sorted(solutions))

def coordination_mask(numbers, left, right, assignments):
    if not assignments or len(assignments[0]) == 1:
        return torch.ones(len(left), device=numbers.device)
    masks = []
    for assignment in assignments:
        target = torch.zeros_like(numbers)
        for z, value in assignment:
            target = torch.where(numbers == z, int(value), target)
        masks.append((target[left] * target[right] < 0).float())
    return torch.stack(masks).mean(0)


def validate_preferences(values, elements=None):
    from pymatgen.core import Element

    if not isinstance(values, (list, tuple)):
        raise ValueError("soft_conditions must be a list")
    result = []
    for value in values:
        supported = {"kind", "condition_id", "element", "coordination", "minimum_fraction", "weight"}
        if not isinstance(value, dict) or set(value) - supported:
            raise ValueError("unsupported local preference; only soft coordination is available")
        kind = value.get("kind", "coordination")
        element = value.get("element")
        allowed = value.get("coordination", [])
        fraction = value.get("minimum_fraction", 1.0)
        weight = value.get("weight", 1.0)
        condition_id = value.get("condition_id", f"coordination_{str(element).lower()}")
        if kind != "coordination":
            raise ValueError("unsupported_target: only coordination soft conditions are available")
        if not Element.is_valid_symbol(element) or (elements is not None and element not in elements):
            raise ValueError("coordination preference element is absent or invalid")
        if not allowed or any(type(n) is not int or not 1 <= n <= 24 for n in allowed):
            raise ValueError("coordination must be a non-empty set of integers in [1,24]")
        if (type(fraction) not in (int, float) or not math.isfinite(fraction)
                or not 0 < fraction <= 1):
            raise ValueError("minimum_fraction must lie in (0,1]")
        if type(weight) not in (int, float) or not math.isfinite(weight) or weight <= 0:
            raise ValueError("condition weight must be finite and positive")
        identifier_characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
        if (not isinstance(condition_id, str) or not condition_id.strip()
                or any(character not in identifier_characters for character in condition_id)):
            raise ValueError("condition_id must use letters, digits, dot, dash or underscore")
        if any(p["element"] == element or p["condition_id"] == condition_id for p in result):
            raise ValueError("combine same-element coordination alternatives into one preference")
        result.append({"condition_id": condition_id, "kind": kind, "element": element,
                       "coordination": sorted(set(allowed)), "minimum_fraction": float(fraction),
                       "weight": float(weight)})
    return sorted(result, key=lambda item: item["condition_id"])


class PeriodicNeighborhood(PeriodicImages):
    """Cache exhaustive image candidates for a fixed lattice, never stale edges."""

    def __init__(self, max_entries=1024):
        if not isinstance(max_entries, int) or max_entries <= 0:
            raise ValueError("periodic-neighborhood cache capacity must be positive")
        self.max_entries = max_entries
        super().__init__(capacity=max_entries)

    def edges(self, fractional, lattice):
        atoms = torch.arange(len(fractional), device=lattice.device)
        left, right = torch.cartesian_prod(atoms, atoms).T
        left, right, vectors = super().edges(
            fractional, lattice, left, right, torch.zeros_like(left), CUTOFF,
        )
        return left, right, vectors.norm(dim=-1).clamp_min(1e-6)


def neighbor_weights(distance, reference):
    taper = ((CUTOFF - distance) / 0.5).clamp(0, 1)
    taper = taper.square() * (3 - 2 * taper)
    return torch.sigmoid((1.25 * reference - distance) / 0.12) * taper, taper


class ChemistryPrior(nn.Module):
    def __init__(self, path):
        super().__init__()
        raw = Path(path).read_bytes()
        data = json.loads(raw)
        if (data.get("schema_version") != PRIOR_SCHEMA or data.get("split") != "train"
                or data.get("kernel") != CHEMISTRY_CONTRACT or data.get("cutoff") != CUTOFF
                or data.get("cn_definition") != CN_DEFINITION
                or data.get("distance_centers") != DISTANCE_CENTERS or data.get("cn_centers") != CN_CENTERS
                or not data.get("records", 0) or len(data.get("cache_manifest_sha256", "")) != 64):
            raise ValueError("chemistry prior requires a versioned, nonempty training-only source")
        bond = torch.zeros(119, 119, len(DISTANCE_CENTERS))
        cn = torch.zeros(119, len(CN_CENTERS))
        for key, entry in data["bond"].items():
            a, b = map(int, key.split("-"))
            if not 1 <= a <= b <= 118:
                raise ValueError("invalid element pair in chemistry prior")
            bond[a, b] = bond[b, a] = self._probabilities(entry, len(DISTANCE_CENTERS))
        for key, entry in data["coordination"].items():
            if not 1 <= int(key) <= 118:
                raise ValueError("invalid coordination element in chemistry prior")
            cn[int(key)] = self._probabilities(entry, len(CN_CENTERS))
        for name, tensor in (("bond", bond), ("cn", cn),
                             ("distance_centers", torch.tensor(DISTANCE_CENTERS)),
                             ("cn_centers", torch.tensor(CN_CENTERS, dtype=torch.float32))):
            self.register_buffer(name, tensor, persistent=False)
        self.identity = {"sha256": hashlib.sha256(raw).hexdigest(),
                         "cache_manifest_sha256": data["cache_manifest_sha256"],
                         "records": data["records"], "split": "train", "kernel": CHEMISTRY_CONTRACT}

    @staticmethod
    def _probabilities(entry, size):
        histogram = torch.tensor(entry["histogram"], dtype=torch.float32)
        if histogram.shape != (size,) or not torch.isfinite(histogram).all() or (histogram < 0).any():
            raise ValueError("invalid chemistry histogram")
        if type(entry["support"]) is not int or entry["support"] < 0:
            raise ValueError("chemistry support must count training structures")
        if entry["support"] < 8 or histogram.sum() <= 0:
            return torch.zeros_like(histogram)
        # Small nonzero tails retain unobserved environments as soft alternatives.
        return 0.98 * histogram / histogram.sum() + 0.02 / size

    @staticmethod
    def mixture_penalty(value, probabilities, centers, width):
        supported = probabilities.sum(-1) > 0
        density = (probabilities * torch.exp(-0.5 * ((value[..., None] - centers) / width).square())).sum(-1)
        return -torch.log(density.clamp_min(1e-12)) * supported, supported


def build_prior(records, cache_sha256, chemistry):
    bond, cn = {}, {}
    support = defaultdict(int)
    count = 0
    for record in records:
        if record.split != "train":
            raise ValueError("validation/test records must never enter chemistry prior fitting")
        x = torch.tensor(record.group_fractional, dtype=torch.float64)
        lattice = torch.tensor(record.state.lattice.matrix, dtype=torch.float64)
        numbers = torch.tensor(record.group_atom_types)
        i, j, distance = chemistry.neighborhood.edges(x, lattice)
        reference = chemistry.atomic_radii[numbers[i]] + chemistry.atomic_radii[numbers[j]]
        weights, _ = neighbor_weights(distance, reference)
        assignments = neutral_oxidation_assignments(tuple(sorted(record.group_atom_types)))
        weights = weights * coordination_mask(numbers, i, j, assignments)
        coordination = torch.zeros(len(x), dtype=x.dtype).index_add(0, i, weights)
        for z in set(record.group_atom_types):
            values = coordination[numbers == z]
            histogram = torch.bincount(values.round().long().clamp(0, 24), minlength=25).double()
            key = str(z)
            cn[key] = cn.get(key, torch.zeros(25, dtype=x.dtype)) + histogram / histogram.sum()
            support["cn:" + key] += 1
        pair_types = torch.sort(torch.stack((numbers[i], numbers[j]), -1), -1).values
        for pair in torch.unique(pair_types, dim=0):
            mask = (pair_types == pair).all(-1)
            histogram = torch.bincount(
                (distance[mask] * 10 - 1).round().long().clamp(0, 44),
                weights=weights[mask], minlength=45)
            if histogram.sum() < 1e-5:
                continue
            key = "-".join(str(int(z)) for z in pair)
            bond[key] = bond.get(key, torch.zeros(45, dtype=x.dtype)) + histogram / histogram.sum()
            support["bond:" + key] += 1
        count += 1
        if count % 500 == 0:
            print(f"[prior] training structures={count}", flush=True)
    if not count:
        raise ValueError("cannot fit an empty prior")
    return {"schema_version": PRIOR_SCHEMA, "kernel": CHEMISTRY_CONTRACT,
            "split": "train", "records": count, "cache_manifest_sha256": cache_sha256,
            "cutoff": CUTOFF, "distance_centers": DISTANCE_CENTERS, "cn_centers": CN_CENTERS,
            "support_unit": "structures", "structure_weight": "one_per_element_or_pair",
            "cn_definition": CN_DEFINITION,
            "bond": {k: {"histogram": h.tolist(), "support": support["bond:" + k]}
                     for k, h in sorted(bond.items())},
            "coordination": {k: {"histogram": h.tolist(), "support": support["cn:" + k]}
                             for k, h in sorted(cn.items())}}

def periodic_coordination(owner, fractional, numbers, lattice, assignments):
    """Return the continuous CN descriptor used by training and runtime guidance."""
    left, right, distance = owner.neighborhood.edges(fractional, lattice)
    reference = (owner.atomic_radii[numbers[left]] + owner.atomic_radii[numbers[right]]).to(distance)
    neighbor, taper = neighbor_weights(distance, reference)
    neighbor = neighbor * coordination_mask(numbers, left, right, assignments)
    coordination = torch.zeros(
        len(fractional), device=distance.device, dtype=distance.dtype
    ).index_add(0, left, neighbor)
    return left, right, distance, reference, neighbor, taper, coordination


def periodic_energies(owner, fractional, numbers, lattice, assignments):
    """Shared training/runtime energy. No reference or material ID is consumed."""
    prior = owner.chemistry_prior
    left, right, distance, reference, neighbor, taper, cn = periodic_coordination(
        owner, fractional, numbers, lattice, assignments
    )
    count = len(fractional)
    zero = fractional.sum() * 0
    overlap = (torch.nn.functional.softplus((0.55 * reference - distance) / 0.08).square() * taper).sum() / (2 * count)
    bond, bond_supported = prior.mixture_penalty(
        distance, prior.bond[numbers[left], numbers[right]], prior.distance_centers, 0.15)
    bond = (neighbor * bond * bond_supported).sum() / (2 * count) + zero
    cn_energy, cn_supported = prior.mixture_penalty(cn, prior.cn[numbers], prior.cn_centers, 0.35)
    cn_energy = cn_energy.sum() / cn_supported.sum().clamp_min(1) + zero
    preferences = []
    for item in owner.local_preferences:
        selected = cn[numbers == owner._atomic_number_by_symbol[item["element"]]]
        if not len(selected):
            raise ValueError("requested coordination element is absent from candidate")
        alternatives = selected.new_tensor(item["coordination"])
        errors = (selected[:, None] - alternatives).square().min(-1).values
        k = math.ceil(item["minimum_fraction"] * len(selected))
        preferences.append((item["weight"], errors.sort().values[:k].mean()))
    preference = (sum(weight * value for weight, value in preferences)
                  / sum(weight for weight, _ in preferences) if preferences else zero)
    bvs = zero
    supported_assignments = [a for a in assignments if len(a) > 1 and all(
        z in owner._bv_parameter_numbers for z, _ in a)]
    if supported_assignments:
        unlike = numbers[left] != numbers[right]
        i, j, d = left[unlike], right[unlike], distance[unlike]
        r1, r2 = owner.bv_r[numbers[i]], owner.bv_r[numbers[j]]
        c1, c2 = owner.bv_c[numbers[i]], owner.bv_c[numbers[j]]
        r = r1 + r2 - r1 * r2 * (c1.sqrt() - c2.sqrt()).square() / (c1 * r1 + c2 * r2).clamp_min(1e-6)
        sign = torch.sign(owner.electronegativity[numbers[j]] - owner.electronegativity[numbers[i]])
        valence = torch.exp(((r - d) / 0.31).clamp(-20, 20)) * sign * taper[unlike]
        losses = []
        for assignment in supported_assignments:
            target = torch.zeros_like(cn)
            for z, value in assignment:
                target = torch.where(numbers == z, value, target)
            opposite = target[i] * target[j] < 0
            sums = torch.zeros_like(cn).index_add(0, i, (valence * opposite).to(cn))
            losses.append(((sums - target) / target.abs().clamp_min(1)).square().mean())
        losses = torch.stack(losses)
        bvs = -torch.logsumexp(-losses, 0) + math.log(len(losses))
    bond_coverage = (neighbor * bond_supported).sum() / neighbor.sum().clamp_min(1e-12)
    return (overlap, bond, bvs, bool(supported_assignments), cn_energy, preference,
            cn_supported.float().mean(), bond_coverage)
