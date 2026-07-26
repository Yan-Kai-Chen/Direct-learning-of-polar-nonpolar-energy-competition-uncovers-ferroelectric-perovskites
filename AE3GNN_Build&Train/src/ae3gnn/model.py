from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn

from .batching import batch_graphs, mean_pool


@dataclass(frozen=True)
class ModelConfig:
    dim: int = 128
    n_layers: int = 2
    rbf_dim: int = 48
    dropout: float = 0.2
    z_max: int = 118
    angle_hidden: int = 160
    r_max: float = 6.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ModelConfig":
        aliases = {
            "DIM": "dim",
            "N_LAYERS": "n_layers",
            "RBF_DIM": "rbf_dim",
            "DROPOUT": "dropout",
            "Z_MAX": "z_max",
            "ANGLE_HID": "angle_hidden",
            "R_MAX": "r_max",
        }
        normalized = {aliases.get(key, key): value for key, value in values.items()}
        fields = cls.__dataclass_fields__
        return cls(**{key: value for key, value in normalized.items() if key in fields})


class GaussianRBF(nn.Module):
    def __init__(self, r_max: float, num_rbfs: int) -> None:
        super().__init__()
        self.r_max = float(r_max)
        self.register_buffer("centers", torch.linspace(0.0, r_max, num_rbfs))
        self.register_buffer(
            "widths",
            torch.full((num_rbfs,), r_max / num_rbfs),
        )

    def forward(self, radius: torch.Tensor) -> torch.Tensor:
        radius = radius.clamp(min=0.0, max=self.r_max)
        expanded = radius.unsqueeze(-1)
        radial = torch.exp(
            -((expanded - self.centers) ** 2) / (self.widths**2 + 1e-12)
        )
        envelope = 0.5 * (torch.cos(math.pi * radius / self.r_max) + 1.0)
        return radial * envelope.unsqueeze(-1)


class AnglePaiNNBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        rbf_dim: int,
        *,
        hidden: int = 160,
        angle_hidden: int = 160,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.edge_mlp = nn.Sequential(
            nn.Linear(rbf_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 4 * dim),
        )
        self.angle_mlp = nn.Sequential(
            nn.Linear(2 * rbf_dim + 1, angle_hidden),
            nn.SiLU(),
            nn.Linear(angle_hidden, dim),
        )
        self.scalar_mlp = nn.Sequential(
            nn.LayerNorm(3 * dim),
            nn.Linear(3 * dim, 2 * dim),
            nn.SiLU(),
            nn.Linear(2 * dim, dim),
        )
        self.vector_gate = nn.Sequential(
            nn.LayerNorm(3 * dim),
            nn.Linear(3 * dim, dim),
            nn.Sigmoid(),
        )

    def forward(
        self,
        scalar: torch.Tensor,
        vector: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vec: torch.Tensor,
        edge_dist: torch.Tensor,
        rbf: torch.Tensor,
        triplet_center: torch.Tensor,
        triplet_e1: torch.Tensor,
        triplet_e2: torch.Tensor,
        triplet_cos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        node_count, feature_dim = scalar.shape
        edge_count = edge_index.shape[1]
        if edge_count == 0:
            return scalar, vector

        center, neighbor = edge_index
        unit = edge_vec / (edge_dist.unsqueeze(-1) + 1e-12)
        w_s, w_sv, w_v, w_vs = self.edge_mlp(rbf).chunk(4, dim=-1)
        scalar_neighbor = scalar[neighbor]
        vector_neighbor = vector[neighbor]
        projection = (vector_neighbor * unit.unsqueeze(1)).sum(dim=-1)

        scalar_edge = w_s * scalar_neighbor + w_sv * projection
        vector_edge = w_v.unsqueeze(-1) * vector_neighbor + (
            w_vs.unsqueeze(-1)
            * scalar_neighbor.unsqueeze(-1)
            * unit.unsqueeze(1)
        )

        scalar_message = torch.zeros(
            (node_count, feature_dim),
            device=scalar.device,
            dtype=scalar.dtype,
        )
        vector_message = torch.zeros(
            (node_count, feature_dim, 3),
            device=vector.device,
            dtype=vector.dtype,
        )
        scalar_message.index_add_(0, center, scalar_edge)
        flat = vector_message.view(node_count, feature_dim * 3)
        flat.index_add_(0, center, vector_edge.view(edge_count, feature_dim * 3))
        vector_message = flat.view(node_count, feature_dim, 3)

        if triplet_center.numel():
            rbf1 = rbf[triplet_e1]
            rbf2 = rbf[triplet_e2]
            angle_input = torch.cat(
                [
                    rbf1 + rbf2,
                    (rbf1 - rbf2).abs(),
                    triplet_cos.unsqueeze(-1),
                ],
                dim=-1,
            )
            angle_nodes = torch.zeros_like(scalar_message)
            angle_nodes.index_add_(0, triplet_center, self.angle_mlp(angle_input))
            scalar_message = scalar_message + angle_nodes

        vector_norm = torch.sqrt((vector_message**2).sum(dim=-1) + self.eps)
        context = torch.cat([scalar, scalar_message, vector_norm], dim=-1)
        return (
            scalar + self.scalar_mlp(context),
            vector + self.vector_gate(context).unsqueeze(-1) * vector_message,
        )


class AnglePaiNNEncoder(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.dim = config.dim
        self.embedding = nn.Embedding(config.z_max + 1, config.dim)
        self.rbf = GaussianRBF(config.r_max, config.rbf_dim)
        self.layers = nn.ModuleList(
            [
                AnglePaiNNBlock(
                    config.dim,
                    config.rbf_dim,
                    angle_hidden=config.angle_hidden,
                )
                for _ in range(config.n_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(config.dim)

    def forward(
        self,
        z: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vec: torch.Tensor,
        edge_dist: torch.Tensor,
        triplet_center: torch.Tensor,
        triplet_e1: torch.Tensor,
        triplet_e2: torch.Tensor,
        triplet_cos: torch.Tensor,
    ) -> torch.Tensor:
        z = z.clamp(min=0, max=self.embedding.num_embeddings - 1)
        scalar = self.embedding(z)
        vector = torch.zeros(
            (scalar.size(0), self.dim, 3),
            device=scalar.device,
            dtype=scalar.dtype,
        )
        radial = self.rbf(edge_dist)
        for layer in self.layers:
            scalar, vector = layer(
                scalar,
                vector,
                edge_index,
                edge_vec,
                edge_dist,
                radial,
                triplet_center,
                triplet_e1,
                triplet_e2,
                triplet_cos,
            )
        return self.output_norm(scalar)


class PairEnergyModel(nn.Module):
    def __init__(self, config: ModelConfig = ModelConfig()) -> None:
        super().__init__()
        self.config = config
        self.encoder = AnglePaiNNEncoder(config)
        pair_dim = 4 * config.dim
        self.pair_mlp = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, 2 * config.dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(2 * config.dim, config.dim),
            nn.SiLU(),
        )
        self.head = nn.Linear(config.dim, 1)

    def _encode(
        self,
        graphs: list[dict[str, Any]],
        device: torch.device,
    ) -> torch.Tensor:
        values = batch_graphs(graphs)
        z, batch, edge_index, edge_vec, edge_dist = values[:5]
        triplet_center, triplet_e1, triplet_e2, triplet_cos = values[5:9]
        graph_count = values[9]
        scalar = self.encoder(
            z.to(device),
            edge_index.to(device),
            edge_vec.to(device),
            edge_dist.to(device),
            triplet_center.to(device),
            triplet_e1.to(device),
            triplet_e2.to(device),
            triplet_cos.to(device),
        )
        return mean_pool(scalar, batch.to(device), graph_count)

    def forward(
        self,
        polar_graphs: list[dict[str, Any]],
        nonpolar_graphs: list[dict[str, Any]],
    ) -> torch.Tensor:
        if len(polar_graphs) != len(nonpolar_graphs):
            raise ValueError("Polar and nonpolar batches must have equal length")
        device = next(self.parameters()).device
        polar = self._encode(polar_graphs, device)
        nonpolar = self._encode(nonpolar_graphs, device)
        delta = polar - nonpolar
        pair = torch.cat([polar, nonpolar, delta, delta.abs()], dim=-1)
        return self.head(self.pair_mlp(pair)).squeeze(-1)
