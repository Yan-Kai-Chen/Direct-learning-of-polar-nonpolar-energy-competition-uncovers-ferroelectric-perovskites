"""Small language-prior heads used above a frozen decoder representation."""

from __future__ import annotations

import torch
import torch.nn as nn

from .registry import CORE_OPERATION_IDS


class LanguageOperationHead(nn.Module):
    """Map one decoder representation to 18 structural-operation scores."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.output = nn.Linear(hidden_size, len(CORE_OPERATION_IDS))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.output(self.norm(hidden.float()))


class SelectiveLanguageGraphHead(nn.Module):
    """Use language context to selectively refine graph operation ranking."""

    def __init__(
        self,
        language_hidden_size: int,
        graph_hidden_size: int,
        projected_size: int = 64,
        operation_size: int = 16,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.language_norm = nn.LayerNorm(language_hidden_size)
        self.language_projection = nn.Sequential(
            nn.Linear(language_hidden_size, projected_size),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.language_prior = nn.Linear(
            projected_size, len(CORE_OPERATION_IDS)
        )
        self.graph_norm = nn.LayerNorm(graph_hidden_size)
        self.operation_embedding = nn.Embedding(
            len(CORE_OPERATION_IDS), operation_size
        )
        interaction_size = (
            graph_hidden_size + projected_size + operation_size + 2
        )
        self.interaction = nn.Sequential(
            nn.Linear(interaction_size, 96),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(96, 2),
        )

    @staticmethod
    def _row_z(values: torch.Tensor) -> torch.Tensor:
        centered = values - values.mean(dim=1, keepdim=True)
        scale = centered.std(dim=1, keepdim=True, unbiased=False)
        return centered / scale.clamp_min(1.0e-6)

    def forward(
        self,
        language_hidden: torch.Tensor,
        graph_hidden: torch.Tensor,
        graph_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        language_context = self.language_projection(
            self.language_norm(language_hidden)
        )
        raw_prior = self.language_prior(language_context)
        prior = self._row_z(raw_prior)
        graph = self._row_z(graph_logits)
        operation_indices = torch.arange(
            len(CORE_OPERATION_IDS), device=language_hidden.device
        )[None, :].expand(len(language_hidden), -1)
        operation = self.operation_embedding(operation_indices)
        expanded_context = language_context[:, None, :].expand(
            -1, len(CORE_OPERATION_IDS), -1
        )
        interaction_input = torch.cat(
            [
                self.graph_norm(graph_hidden),
                expanded_context,
                operation,
                graph.unsqueeze(-1),
                prior.unsqueeze(-1),
            ],
            dim=-1,
        )
        interaction = self.interaction(interaction_input)
        correction = interaction[..., 0]
        gate = torch.sigmoid(interaction[..., 1])
        fused = graph + gate * (prior + correction)
        return fused, raw_prior, gate


__all__ = ["LanguageOperationHead", "SelectiveLanguageGraphHead"]
