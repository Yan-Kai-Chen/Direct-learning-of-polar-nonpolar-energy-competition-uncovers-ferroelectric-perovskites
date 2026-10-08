from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def batch_graphs(
    graphs: list[dict[str, Any]],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
]:
    if not graphs:
        raise ValueError("Cannot batch an empty graph list")
    z_all: list[torch.Tensor] = []
    batch_all: list[torch.Tensor] = []
    edge_index_all: list[torch.Tensor] = []
    edge_vec_all: list[torch.Tensor] = []
    edge_dist_all: list[torch.Tensor] = []
    center_all: list[torch.Tensor] = []
    e1_all: list[torch.Tensor] = []
    e2_all: list[torch.Tensor] = []
    cosine_all: list[torch.Tensor] = []
    node_offset = 0
    edge_offset = 0
    for graph_id, graph in enumerate(graphs):
        z = torch.from_numpy(graph["z"]).long()
        edge_index = torch.from_numpy(graph["edge_index"]).long()
        edge_vec = torch.from_numpy(graph["edge_vec"]).float()
        edge_dist = torch.from_numpy(graph["edge_dist"]).float()
        center = torch.from_numpy(
            graph.get("triplet_center", np.zeros(0, np.int64))
        ).long()
        e1 = torch.from_numpy(
            graph.get("triplet_e1", np.zeros(0, np.int64))
        ).long()
        e2 = torch.from_numpy(
            graph.get("triplet_e2", np.zeros(0, np.int64))
        ).long()
        cosine = torch.from_numpy(
            graph.get("triplet_cos", np.zeros(0, np.float32))
        ).float()
        node_count = int(z.shape[0])
        edge_count = int(edge_index.shape[1])
        if edge_count:
            edge_index = edge_index + node_offset
        if center.numel():
            center = center + node_offset
            e1 = e1 + edge_offset
            e2 = e2 + edge_offset
        z_all.append(z)
        batch_all.append(
            torch.full((node_count,), graph_id, dtype=torch.long)
        )
        edge_index_all.append(edge_index)
        edge_vec_all.append(edge_vec)
        edge_dist_all.append(edge_dist)
        center_all.append(center)
        e1_all.append(e1)
        e2_all.append(e2)
        cosine_all.append(cosine)
        node_offset += node_count
        edge_offset += edge_count

    total_edges = sum(item.shape[1] for item in edge_index_all)
    if total_edges:
        edge_index = torch.cat(edge_index_all, dim=1)
        edge_vec = torch.cat(edge_vec_all)
        edge_dist = torch.cat(edge_dist_all)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_vec = torch.zeros((0, 3), dtype=torch.float32)
        edge_dist = torch.zeros((0,), dtype=torch.float32)
    return (
        torch.cat(z_all),
        torch.cat(batch_all),
        edge_index,
        edge_vec,
        edge_dist,
        torch.cat(center_all),
        torch.cat(e1_all),
        torch.cat(e2_all),
        torch.cat(cosine_all),
        len(graphs),
    )


def mean_pool(
    values: torch.Tensor,
    batch: torch.Tensor,
    graph_count: int,
) -> torch.Tensor:
    pooled = torch.zeros(
        (graph_count, values.size(-1)),
        device=values.device,
        dtype=values.dtype,
    )
    counts = torch.zeros(
        graph_count,
        device=values.device,
        dtype=values.dtype,
    )
    pooled.index_add_(0, batch, values)
    counts.index_add_(0, batch, torch.ones_like(batch, dtype=values.dtype))
    return pooled / counts.clamp(min=1).unsqueeze(-1)


@dataclass(frozen=True)
class ModelConfig:
    dim: int = 128
    n_layers: int = 2
    rbf_dim: int = 48
    dropout: float = 0.2
    z_max: int = 118
    angle_hidden: int = 160
    r_max: float = 6.0
    output_dim: int = 1


class GaussianRBF(nn.Module):
    def __init__(self, r_max: float, count: int) -> None:
        super().__init__()
        self.r_max = float(r_max)
        self.register_buffer("centers", torch.linspace(0.0, r_max, count))
        self.register_buffer("widths", torch.full((count,), r_max / count))

    def forward(self, radius: torch.Tensor) -> torch.Tensor:
        radius = radius.clamp(0.0, self.r_max)
        expanded = radius.unsqueeze(-1)
        rbf = torch.exp(
            -((expanded - self.centers) ** 2) / (self.widths**2 + 1e-12)
        )
        envelope = 0.5 * (torch.cos(math.pi * radius / self.r_max) + 1.0)
        return rbf * envelope.unsqueeze(-1)


class PaiNNBlockAngle(nn.Module):
    def __init__(
        self,
        dim: int,
        rbf_dim: int,
        angle_hidden: int,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.edge_mlp = nn.Sequential(
            nn.Linear(rbf_dim, 160),
            nn.SiLU(),
            nn.Linear(160, 160),
            nn.SiLU(),
            nn.Linear(160, 4 * dim),
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
        node_count, dim = scalar.shape
        edge_count = edge_index.shape[1]
        if edge_count == 0:
            return scalar, vector
        center, neighbor = edge_index
        unit = edge_vec / (edge_dist.unsqueeze(-1) + 1e-12)
        ws, wsv, wv, wvs = self.edge_mlp(rbf).chunk(4, dim=-1)
        scalar_neighbor = scalar[neighbor]
        vector_neighbor = vector[neighbor]
        projection = (vector_neighbor * unit.unsqueeze(1)).sum(dim=-1)
        scalar_edge = ws * scalar_neighbor + wsv * projection
        vector_edge = wv.unsqueeze(-1) * vector_neighbor + (
            wvs.unsqueeze(-1)
            * scalar_neighbor.unsqueeze(-1)
            * unit.unsqueeze(1)
        )
        scalar_message = torch.zeros_like(scalar)
        scalar_message.index_add_(0, center, scalar_edge)
        vector_message = torch.zeros_like(vector)
        vector_flat = vector_message.view(node_count, dim * 3)
        vector_flat.index_add_(0, center, vector_edge.view(edge_count, dim * 3))
        vector_message = vector_flat.view(node_count, dim, 3)
        if triplet_center.numel():
            rbf1, rbf2 = rbf[triplet_e1], rbf[triplet_e2]
            angle_input = torch.cat(
                [
                    rbf1 + rbf2,
                    (rbf1 - rbf2).abs(),
                    triplet_cos.unsqueeze(-1),
                ],
                dim=-1,
            )
            angle_message = self.angle_mlp(angle_input).to(
                dtype=scalar_message.dtype
            )
            angle_nodes = torch.zeros_like(scalar)
            angle_nodes.index_add_(0, triplet_center, angle_message)
            scalar_message = scalar_message + angle_nodes
        vector_norm = torch.sqrt(
            (vector_message**2).sum(dim=-1) + self.eps
        )
        context = torch.cat([scalar, scalar_message, vector_norm], dim=-1)
        return (
            scalar + self.scalar_mlp(context),
            vector + self.vector_gate(context).unsqueeze(-1) * vector_message,
        )


class PaiNNEncoderAngle(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.dim = config.dim
        self.rbf = GaussianRBF(config.r_max, config.rbf_dim)
        self.embedding = nn.Embedding(config.z_max + 1, config.dim)
        self.layers = nn.ModuleList(
            [
                PaiNNBlockAngle(
                    config.dim,
                    config.rbf_dim,
                    config.angle_hidden,
                )
                for _ in range(config.n_layers)
            ]
        )
        self.norm = nn.LayerNorm(config.dim)

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
        scalar = self.embedding(
            z.clamp(0, self.embedding.num_embeddings - 1)
        )
        vector = torch.zeros(
            (scalar.shape[0], self.dim, 3),
            dtype=scalar.dtype,
            device=scalar.device,
        )
        rbf = self.rbf(edge_dist)
        for layer in self.layers:
            scalar, vector = layer(
                scalar,
                vector,
                edge_index,
                edge_vec,
                edge_dist,
                rbf,
                triplet_center,
                triplet_e1,
                triplet_e2,
                triplet_cos,
            )
        return self.norm(scalar)


@dataclass(frozen=True)
class ParentGraphConfig:
    operation_count: int = 18
    template_feature_dim: int = 35
    dim: int = 128
    n_layers: int = 2
    rbf_dim: int = 48
    dropout: float = 0.2
    z_max: int = 118
    angle_hidden: int = 160
    r_max: float = 6.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ParentGraphEncoder(nn.Module):
    def __init__(self, config: ParentGraphConfig) -> None:
        super().__init__()
        self.config = config
        encoder_config = ModelConfig(
            dim=config.dim,
            n_layers=config.n_layers,
            rbf_dim=config.rbf_dim,
            dropout=config.dropout,
            z_max=config.z_max,
            angle_hidden=config.angle_hidden,
            r_max=config.r_max,
            output_dim=1,
        )
        self.encoder = PaiNNEncoderAngle(encoder_config)
        self.template_mlp = nn.Sequential(
            nn.LayerNorm(config.template_feature_dim),
            nn.Linear(config.template_feature_dim, config.dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim, config.dim),
        )
        self.operation_embedding = nn.Embedding(
            config.operation_count,
            config.dim,
        )
        self.query_mlp = nn.Sequential(
            nn.LayerNorm(3 * config.dim),
            nn.Linear(3 * config.dim, 2 * config.dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(2 * config.dim, config.dim),
            nn.SiLU(),
        )
        self.quantile_head = nn.Linear(config.dim, 3)

    def encode_graphs(self, graphs: list[dict[str, Any]]) -> torch.Tensor:
        device = next(self.parameters()).device
        (
            z,
            batch,
            edge_index,
            edge_vec,
            edge_dist,
            center,
            e1,
            e2,
            cosine,
            graph_count,
        ) = batch_graphs(graphs)
        scalar = self.encoder(
            z.to(device),
            edge_index.to(device),
            edge_vec.to(device),
            edge_dist.to(device),
            center.to(device),
            e1.to(device),
            e2.to(device),
            cosine.to(device),
        )
        return mean_pool(scalar, batch.to(device), graph_count)


@dataclass(frozen=True)
class OperationGraphConfig(ParentGraphConfig):
    rank_descriptor_dim: int = 196
    numeric_descriptor_dim: int = 128

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class OperationGraphNetwork(ParentGraphEncoder):
    """PolarGen graph branch with ranking, quantile, and direction heads."""

    def __init__(self, config: OperationGraphConfig) -> None:
        super().__init__(config)
        self.config = config
        del self.query_mlp
        self.rank_descriptor_mlp = nn.Sequential(
            nn.LayerNorm(config.rank_descriptor_dim),
            nn.Linear(config.rank_descriptor_dim, config.dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim, config.dim),
        )
        self.numeric_descriptor_mlp = nn.Sequential(
            nn.LayerNorm(config.numeric_descriptor_dim),
            nn.Linear(config.numeric_descriptor_dim, config.dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim, config.dim),
        )
        self.rank_query_mlp = nn.Sequential(
            nn.LayerNorm(4 * config.dim),
            nn.Linear(4 * config.dim, 2 * config.dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(2 * config.dim, config.dim),
            nn.SiLU(),
        )
        self.numeric_query_mlp = nn.Sequential(
            nn.LayerNorm(4 * config.dim),
            nn.Linear(4 * config.dim, 2 * config.dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(2 * config.dim, config.dim),
            nn.SiLU(),
        )
        self.rank_head = nn.Linear(config.dim, 1)
        self.quantile_head = nn.Linear(config.dim, 3)
        self.direction_head = nn.Linear(config.dim, 1)

    def forward(
        self,
        graphs: list[dict[str, Any]],
        template_features: torch.Tensor,
        rank_descriptor_features: torch.Tensor,
        numeric_descriptor_features: torch.Tensor,
        operation_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        graph_hidden = self.encode_graphs(graphs)
        template_hidden = self.template_mlp(template_features)
        rank_descriptor_hidden = self.rank_descriptor_mlp(
            rank_descriptor_features
        )
        numeric_descriptor_hidden = self.numeric_descriptor_mlp(
            numeric_descriptor_features
        )
        operation_hidden = self.operation_embedding(operation_indices)
        graph_expanded = graph_hidden[:, None, :].expand(
            -1, operation_indices.shape[1], -1
        )
        template_expanded = template_hidden[:, None, :].expand_as(
            graph_expanded
        )
        rank_descriptor_expanded = rank_descriptor_hidden[:, None, :].expand_as(
            graph_expanded
        )
        numeric_descriptor_expanded = numeric_descriptor_hidden[
            :, None, :
        ].expand_as(graph_expanded)
        rank_hidden = self.rank_query_mlp(
            torch.cat(
                [
                    graph_expanded,
                    template_expanded,
                    rank_descriptor_expanded,
                    operation_hidden,
                ],
                dim=-1,
            )
        )
        numeric_hidden = self.numeric_query_mlp(
            torch.cat(
                [
                    graph_expanded,
                    template_expanded,
                    numeric_descriptor_expanded,
                    operation_hidden,
                ],
                dim=-1,
            )
        )
        rank_logits = self.rank_head(rank_hidden).squeeze(-1)
        raw = self.quantile_head(numeric_hidden)
        median = raw[..., 0]
        lower = median - F.softplus(raw[..., 1])
        upper = median + F.softplus(raw[..., 2])
        quantiles = torch.stack([lower, median, upper], dim=-1)
        direction_logits = self.direction_head(numeric_hidden).squeeze(-1)
        return rank_logits, quantiles, direction_logits
