# -*- coding: utf-8 -*-
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GINEConv, global_mean_pool


class RBF(nn.Module):
    def __init__(self, num_centers: int = 32, r_min: float = 0.0, r_max: float = 6.0, gamma: float | None = None):
        super().__init__()
        centers = torch.linspace(r_min, r_max, num_centers)
        self.register_buffer("centers", centers)
        if gamma is None:
            gamma = 10.0 / (r_max - r_min + 1e-6) ** 2
        self.gamma = float(gamma)

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        # d: (E,1) or (E,)
        if d.dim() == 2:
            d = d[:, 0]
        diff = d[:, None] - self.centers[None, :]
        return torch.exp(-self.gamma * diff * diff)


class GNNEncoder(nn.Module):
    def __init__(
        self,
        hidden: int = 128,
        layers: int = 4,
        out_dim: int = 128,
        dropout: float = 0.1,
        z_max: int = 100,
        cutoff: float = 6.0,
    ):
        super().__init__()
        self.cutoff = float(cutoff)
        self.dropout = float(dropout)

        self.emb = nn.Embedding(z_max + 1, hidden)
        self.rbf = RBF(num_centers=32, r_min=0.0, r_max=self.cutoff)

        self.edge_mlp = nn.Sequential(
            nn.Linear(32, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

        self.convs = nn.ModuleList()
        for _ in range(layers):
            nn_msg = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden),
            )
            self.convs.append(GINEConv(nn_msg, edge_dim=hidden))

        self.proj = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, data: Data) -> torch.Tensor:
        x = self.emb(data.z)
        e = self.edge_mlp(self.rbf(data.edge_attr))

        for conv in self.convs:
            x = conv(x, data.edge_index, e)
            x = F.silu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        if hasattr(data, "batch") and data.batch is not None:
            g = global_mean_pool(x, data.batch)
        else:
            g = x.mean(dim=0, keepdim=True)

        z = self.proj(g)
        z = F.normalize(z, dim=-1)
        return z