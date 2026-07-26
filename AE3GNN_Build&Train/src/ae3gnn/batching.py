from __future__ import annotations

from typing import Any

import numpy as np
import torch


Graph = dict[str, Any]


def batch_graphs(
    graphs: list[Graph],
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
    edge_indices: list[torch.Tensor] = []
    edge_vectors: list[torch.Tensor] = []
    edge_distances: list[torch.Tensor] = []
    triplet_centers: list[torch.Tensor] = []
    triplet_e1: list[torch.Tensor] = []
    triplet_e2: list[torch.Tensor] = []
    triplet_cosines: list[torch.Tensor] = []
    node_offset = 0
    edge_offset = 0

    for graph_id, graph in enumerate(graphs):
        z = torch.as_tensor(graph["z"], dtype=torch.long)
        edge_index = torch.as_tensor(graph["edge_index"], dtype=torch.long)
        edge_vec = torch.as_tensor(graph["edge_vec"], dtype=torch.float32)
        edge_dist = torch.as_tensor(graph["edge_dist"], dtype=torch.float32)
        node_count = int(z.shape[0])
        edge_count = int(edge_index.shape[1])

        center = torch.as_tensor(
            graph.get("triplet_center", np.zeros(0, dtype=np.int64)),
            dtype=torch.long,
        )
        e1 = torch.as_tensor(
            graph.get("triplet_e1", np.zeros(0, dtype=np.int64)),
            dtype=torch.long,
        )
        e2 = torch.as_tensor(
            graph.get("triplet_e2", np.zeros(0, dtype=np.int64)),
            dtype=torch.long,
        )
        cosine = torch.as_tensor(
            graph.get("triplet_cos", np.zeros(0, dtype=np.float32)),
            dtype=torch.float32,
        )
        if edge_count:
            edge_index = edge_index + node_offset
        if center.numel():
            center = center + node_offset
            e1 = e1 + edge_offset
            e2 = e2 + edge_offset

        z_all.append(z)
        batch_all.append(torch.full((node_count,), graph_id, dtype=torch.long))
        edge_indices.append(edge_index)
        edge_vectors.append(edge_vec)
        edge_distances.append(edge_dist)
        triplet_centers.append(center)
        triplet_e1.append(e1)
        triplet_e2.append(e2)
        triplet_cosines.append(cosine)
        node_offset += node_count
        edge_offset += edge_count

    total_edges = sum(int(value.shape[1]) for value in edge_indices)
    if total_edges:
        edge_index_batch = torch.cat(edge_indices, dim=1)
        edge_vec_batch = torch.cat(edge_vectors)
        edge_dist_batch = torch.cat(edge_distances)
    else:
        edge_index_batch = torch.zeros((2, 0), dtype=torch.long)
        edge_vec_batch = torch.zeros((0, 3), dtype=torch.float32)
        edge_dist_batch = torch.zeros((0,), dtype=torch.float32)

    return (
        torch.cat(z_all),
        torch.cat(batch_all),
        edge_index_batch,
        edge_vec_batch,
        edge_dist_batch,
        torch.cat(triplet_centers),
        torch.cat(triplet_e1),
        torch.cat(triplet_e2),
        torch.cat(triplet_cosines),
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
    return pooled / counts.clamp(min=1.0).unsqueeze(-1)
