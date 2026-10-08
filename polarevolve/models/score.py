"""Compact Cartesian-equivariant hard-condition score network."""

from __future__ import annotations

import itertools
import math

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from polarevolve.data.batch import PackedASUBatch
from polarevolve.crystal.periodic import PERIODIC_IMAGES
from polarevolve.models.config import (
    PHYSICAL_SCORE_V1,
    SCORE_PARAMETERIZATIONS,
    SIGMA_SCALED_SCORE_V1,
)
from polarevolve.models.geometry import (
    complete_periodic_geometry, lattice_invariants, periodic_edge_geometry,
)


class HardConditionScoreNetwork(nn.Module):
    """Predict a hard-conditioned negative-score covector."""

    output_convention = "negative_score_covector"

    def __init__(
        self,
        *,
        hidden_dim: int = 256,
        time_dim: int = 64,
        radial_basis: int = 48,
        layers: int = 4,
        cutoff: float = 8.0,
        score_parameterization: str = PHYSICAL_SCORE_V1,
        max_atomic_number: int = 118,
        max_orbit_multiplicity: int = 256,
        max_orbits_per_structure: int = 256,
        max_orbit_members: int = 256,
        condition_dim: int = 0,
        atom_condition_dim: int = 0,
        equivariant_reference: bool = False,
        backbone_version: int = 1,
    ) -> None:
        super().__init__()
        if hidden_dim < 32 or time_dim < 8 or time_dim % 2:
            raise ValueError("hidden_dim >= 32 and an even time_dim >= 8 are required")
        if radial_basis < 4 or layers < 1 or cutoff <= 0.0:
            raise ValueError("invalid score-network depth, radial basis, or cutoff")
        if score_parameterization not in SCORE_PARAMETERIZATIONS:
            raise ValueError(f"score_parameterization must be one of {SCORE_PARAMETERIZATIONS}")
        if condition_dim < 0 or atom_condition_dim < 0:
            raise ValueError("condition dimensions must be non-negative")
        self.hidden_dim = int(hidden_dim)
        self.time_dim = int(time_dim)
        self.cutoff = float(cutoff)
        self.score_parameterization = str(score_parameterization)
        self.condition_dim = int(condition_dim)
        self.atom_condition_dim = int(atom_condition_dim)
        if backbone_version not in (1, 2):
            raise ValueError("unsupported score backbone version")
        self.backbone_version = backbone_version
        if equivariant_reference and atom_condition_dim != 2:
            raise ValueError("equivariant reference requires atom_condition_dim=2")
        self.equivariant_reference = bool(equivariant_reference)
        self.element_embedding = nn.Embedding(max_atomic_number + 1, hidden_dim)
        self.hall_embedding = nn.Embedding(531, hidden_dim)
        self.space_group_embedding = nn.Embedding(231, hidden_dim)
        self.multiplicity_embedding = nn.Embedding(max_orbit_multiplicity + 1, hidden_dim)
        self.dimension_embedding = nn.Embedding(4, hidden_dim)
        self.wyckoff_embedding = nn.Embedding(27, hidden_dim)
        if backbone_version == 1:
            self.orbit_slot_embedding = nn.Embedding(max_orbits_per_structure + 1, hidden_dim)
            self.member_embedding = nn.Embedding(max_orbit_members + 1, hidden_dim)
        self.time_projection = nn.Linear(time_dim, hidden_dim)
        self.lattice_projection = nn.Sequential(
            nn.Linear(7, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.condition_projection = (
            nn.Sequential(
                nn.Linear(self.condition_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            if self.condition_dim
            else None
        )
        self.atom_condition_projection = (
            nn.Sequential(
                nn.Linear(self.atom_condition_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            if self.atom_condition_dim
            else None
        )
        edge_dim = radial_basis + 1
        self.message_layers = nn.ModuleList(
            nn.Sequential(
                nn.Linear(2 * hidden_dim + edge_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(layers)
        )
        self.update_layers = nn.ModuleList(
            nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(layers))
        self.edge_output = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False),
        )
        self.graph_norm = nn.LayerNorm(hidden_dim)
        self.reference_output = (
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))
            if self.equivariant_reference else None
        )
        self.atom_norm = nn.LayerNorm(hidden_dim)
        self.register_buffer("radial_centers", torch.linspace(0.0, float(cutoff), radial_basis))
        self.register_buffer(
            "time_frequencies",
            torch.exp(torch.linspace(0.0, math.log(1000.0), time_dim // 2)),
        )
        self.register_buffer(
            "periodic_offsets",
            torch.tensor(
                tuple(itertools.product((-1.0, 0.0, 1.0), repeat=3)),
                dtype=torch.float32,
            ),
        )

    def _time_embedding(self, sigma: torch.Tensor) -> torch.Tensor:
        phase = torch.log(sigma.clamp_min(1.0e-8))[:, None] * self.time_frequencies[None]
        return torch.cat((phase.sin(), phase.cos()), dim=-1)

    @staticmethod
    def _validate_embedding_indices(entries) -> None:
        invalid = torch.stack(
            tuple(
                ((values < 0) | (values >= embedding.num_embeddings)).any()
                for values, embedding, _ in entries
            )
        )
        if bool(invalid.any().item()):
            for failed, (_, _, name) in zip(invalid.tolist(), entries):
                if failed:
                    raise ValueError(f"{name} lies outside its embedding table")

    def forward(
        self,
        batch: PackedASUBatch,
        u: torch.Tensor,
        sigma_by_structure: torch.Tensor,
        *,
        condition: torch.Tensor | None = None,
        atom_reference_fractional: torch.Tensor | None = None,
        atom_reference_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if u.shape != batch.parameter_shape:
            raise ValueError("ASU parameters do not match the packed batch")
        sigma = torch.as_tensor(sigma_by_structure, device=u.device, dtype=u.dtype).reshape(-1)
        if sigma.shape != (batch.batch_size,) or bool((sigma <= 0).any().item()):
            raise ValueError("sigma must contain one positive value per structure")
        if self.condition_projection is None:
            if condition is not None:
                raise ValueError("condition was provided to a condition-free model")
        else:
            if condition is None:
                raise ValueError("this model requires a per-structure condition vector")
            condition = torch.as_tensor(condition, device=u.device, dtype=u.dtype)
            if condition.shape != (batch.batch_size, self.condition_dim):
                raise ValueError("condition must have shape [batch, condition_dim]")
        fractional = batch.expand(u)
        feature_dtype = self.element_embedding.weight.dtype
        atom_reference_features = None
        reference_vector = None
        if self.atom_condition_projection is None:
            if atom_reference_fractional is not None or atom_reference_mask is not None:
                raise ValueError("atom reference was provided to a reference-free model")
        else:
            if atom_reference_fractional is None or atom_reference_mask is None:
                raise ValueError("this model requires child-reference atom geometry")
            reference = torch.as_tensor(
                atom_reference_fractional, device=u.device, dtype=u.dtype
            )
            mask = torch.as_tensor(atom_reference_mask, device=u.device, dtype=u.dtype)
            if reference.shape != (batch.num_atoms, 3) or mask.shape != (batch.num_atoms,):
                raise ValueError("atom reference must have shapes [atoms,3] and [atoms]")
            delta = reference - fractional
            if self.backbone_version == 2:
                cartesian = PERIODIC_IMAGES.nearest(delta, batch.lattice[batch.atom_to_structure])
            else:
                delta = delta - torch.round(delta)
                cartesian = torch.einsum(
                    "ni,nij->nj", delta, batch.lattice[batch.atom_to_structure]
                )
            if self.equivariant_reference:
                reference_vector = cartesian * mask[:, None]
                atom_reference_features = torch.stack((cartesian.norm(dim=-1), mask), dim=-1)
            else:
                atom_reference_features = torch.cat((cartesian, mask[:, None]), dim=-1)
            if atom_reference_features.shape[1] != self.atom_condition_dim:
                raise ValueError("atom reference feature dimension differs from the model")
        source, target = batch.edge_index
        atom_to_orbit = batch.atom_to_orbit
        atom_to_structure = batch.atom_to_structure
        multiplicities = batch.orbit_multiplicities[atom_to_orbit]
        orbit_slots = batch.orbit_local_indices[atom_to_orbit]
        orbit_members = batch.atom_member_indices
        entries = [(multiplicities, self.multiplicity_embedding, "orbit multiplicity")]
        if self.backbone_version == 1:
            entries.extend((
                (orbit_slots, self.orbit_slot_embedding, "orbit slot"),
                (orbit_members, self.member_embedding, "orbit member"),
            ))
        self._validate_embedding_indices(entries)
        lattice_features = lattice_invariants(batch).to(feature_dtype)
        if self.backbone_version == 2:
            # The graph encodes cell shape. Volume per atom is independent of
            # lattice-basis choice; Gram entries are not.
            lattice_features = torch.cat((torch.zeros_like(lattice_features[:, :6]),
                                          lattice_features[:, 6:]), dim=-1)
        graph = self.graph_norm(
            self.hall_embedding(batch.hall_numbers)
            + self.space_group_embedding(batch.space_group_numbers)
            + self.lattice_projection(lattice_features)
        ).to(feature_dtype)
        if self.condition_projection is not None:
            graph = graph + self.condition_projection(condition.to(feature_dtype)).to(feature_dtype)
        atom_condition = (
            self.multiplicity_embedding(multiplicities)
            + self.dimension_embedding(batch.orbit_dimensions[atom_to_orbit])
            + self.wyckoff_embedding(batch.orbit_letter_indices[atom_to_orbit])
        )
        if self.backbone_version == 1:
            atom_condition = (atom_condition + self.orbit_slot_embedding(orbit_slots)
                              + self.member_embedding(orbit_members))
        atom_condition = self.atom_norm(atom_condition).to(feature_dtype)
        time = self.time_projection(self._time_embedding(sigma).to(feature_dtype)).to(feature_dtype)
        hidden = (
            self.element_embedding(batch.atom_types).to(feature_dtype)
            + graph[atom_to_structure]
            + atom_condition
            + time[atom_to_structure]
        )
        if atom_reference_features is not None:
            hidden = hidden + self.atom_condition_projection(
                atom_reference_features.to(feature_dtype)
            ).to(feature_dtype)
        if (source.numel() == 0 and self.backbone_version == 1) or batch.num_parameters == 0:
            parameter_zero = sum(parameter.reshape(-1)[0] * 0.0 for parameter in self.parameters())
            return torch.zeros_like(u) + parameter_zero
        if self.backbone_version == 2:
            source, target, direction, radial, envelope = complete_periodic_geometry(
                batch, fractional, radial_centers=self.radial_centers, cutoff=self.cutoff
            )
        else:
            direction, radial, envelope = periodic_edge_geometry(
                batch, fractional, source, target, periodic_offsets=self.periodic_offsets,
                radial_centers=self.radial_centers, cutoff=self.cutoff,
            )
        same_orbit = (atom_to_orbit[source] == atom_to_orbit[target]).to(u.dtype)[:, None]
        edge_features = torch.cat((radial, same_orbit), dim=-1)
        # The MLP stays autocast-enabled; order-sensitive sums and cancellation
        # in the covector head need a wider accumulator on the periodic graph.
        compute_dtype = (torch.float64 if self.backbone_version == 2 or u.dtype == torch.float64
                         else torch.float32)
        degree = batch.atom_degree.to(compute_dtype)
        if self.backbone_version == 2:
            degree = torch.ones(batch.num_atoms, device=u.device, dtype=compute_dtype)
            degree.index_add_(0, source, envelope.to(compute_dtype))
        for message_layer, update_layer, norm in zip(
            self.message_layers, self.update_layers, self.norms
        ):
            aggregate = torch.zeros_like(hidden, dtype=compute_dtype)
            chunk_size = 32768 if self.backbone_version == 2 else max(len(source), 1)
            for start in range(0, len(source), chunk_size):
                sl = slice(start, start + chunk_size)
                args = (hidden, source[sl], target[sl], edge_features[sl], message_layer)
                if self.backbone_version == 2 and torch.is_grad_enabled():
                    message = checkpoint(self._edge_values, *args, use_reentrant=False)
                else:
                    message = self._edge_values(*args)
                message = message.to(compute_dtype) * envelope[sl, None]
                aggregate.index_add_(0, source[sl], message)
            updated = hidden + update_layer(
                torch.cat((hidden, (aggregate / degree.clamp_min(1.0)[:, None]).to(hidden)), dim=-1)
            )
            hidden = norm(updated)
        outputs = []
        for start in range(0, len(source), max(chunk_size, 1)):
            sl = slice(start, start + chunk_size)
            args = (hidden, source[sl], target[sl], edge_features[sl], self.edge_output)
            if self.backbone_version == 2 and torch.is_grad_enabled():
                outputs.append(checkpoint(self._edge_values, *args, use_reentrant=False))
            else:
                outputs.append(self._edge_values(*args))
        scalar = (torch.cat(outputs).squeeze(-1) if outputs
                  else hidden.new_empty((0,)))
        atom_covector = torch.zeros((batch.num_atoms, 3), device=u.device, dtype=compute_dtype)
        atom_covector.index_add_(
            0, source, scalar.to(compute_dtype)[:, None] * envelope[:, None] * direction
        )
        if reference_vector is not None:
            atom_covector = atom_covector + self.reference_output(hidden).to(u.dtype) * reference_vector
        graph_sum = torch.zeros((batch.batch_size, 3), device=u.device, dtype=compute_dtype)
        graph_sum.index_add_(0, atom_to_structure, atom_covector)
        counts = (batch.atom_ptr[1:] - batch.atom_ptr[:-1]).to(compute_dtype)
        atom_covector = atom_covector - (graph_sum / counts[:, None])[atom_to_structure]
        native_covector = batch.pullback(atom_covector).to(u.dtype)
        if self.score_parameterization == SIGMA_SCALED_SCORE_V1:
            negative_score = native_covector / sigma[batch.parameter_to_structure]
        else:
            negative_score = native_covector
        if not bool(torch.isfinite(negative_score).all().item()):
            raise FloatingPointError("score network produced a non-finite covector")
        return negative_score

    @staticmethod
    def _edge_values(hidden, source, target, edge_features, layer):
        return layer(torch.cat((hidden[source], hidden[target], edge_features.to(hidden)), dim=-1))


__all__ = ["HardConditionScoreNetwork", "lattice_invariants"]
