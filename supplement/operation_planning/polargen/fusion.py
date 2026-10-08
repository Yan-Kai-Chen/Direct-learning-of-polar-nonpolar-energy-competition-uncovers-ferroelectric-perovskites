"""Frozen language-graph operation ranking used by PolarGen."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def row_standardize(values: np.ndarray) -> np.ndarray:
    """Standardize every sample across the operation dimension."""

    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("operation scores must have shape [samples, operations]")
    centered = array - array.mean(axis=1, keepdims=True)
    scale = centered.std(axis=1, keepdims=True)
    return centered / np.maximum(scale, 1.0e-6)


def fuse_operation_scores(
    branches: Mapping[str, np.ndarray],
    weights: Mapping[str, float],
) -> np.ndarray:
    """Combine standardized language and graph branch scores.

    Every branch must use the same sample order and structural-operation order.
    The result changes ranking only; numerical operation outputs remain owned by
    the primary graph branch.
    """

    if not branches:
        raise ValueError("at least one ranking branch is required")
    unknown = sorted(set(weights) - set(branches))
    if unknown:
        raise ValueError(f"weights reference unknown branches: {unknown}")
    missing = sorted(set(branches) - set(weights))
    if missing:
        raise ValueError(f"missing branch weights: {missing}")
    shapes = {name: np.asarray(value).shape for name, value in branches.items()}
    if len(set(shapes.values())) != 1:
        raise ValueError(f"branch score shapes differ: {shapes}")
    first_shape = next(iter(shapes.values()))
    if len(first_shape) != 2:
        raise ValueError("branch scores must have shape [samples, operations]")
    fused = np.zeros(first_shape, dtype=np.float32)
    for name, values in branches.items():
        fused += float(weights[name]) * row_standardize(values)
    return fused


def select_top_operations(
    scores: np.ndarray,
    operation_ids: Sequence[str],
    top_k: int = 3,
) -> list[list[str]]:
    """Return ranked operation identifiers for every sample."""

    array = np.asarray(scores)
    if array.ndim != 2:
        raise ValueError("scores must have shape [samples, operations]")
    if array.shape[1] != len(operation_ids):
        raise ValueError("operation_ids do not match the score width")
    if not 1 <= int(top_k) <= len(operation_ids):
        raise ValueError("top_k must be within the operation vocabulary")
    order = np.argsort(-array, axis=1, kind="stable")[:, : int(top_k)]
    names = tuple(str(value) for value in operation_ids)
    return [[names[int(index)] for index in row] for row in order]


def preserve_graph_numerics(
    fused_scores: np.ndarray,
    graph_operations: Sequence[Mapping[str, object]],
    top_k: int = 3,
) -> list[dict[str, object]]:
    """Reorder graph predictions without changing graph-derived numerics."""

    scores = np.asarray(fused_scores, dtype=np.float32)
    if scores.ndim == 2:
        if scores.shape[0] != 1:
            raise ValueError("single-record graph outputs require one score row")
        scores = scores[0]
    if scores.ndim != 1 or len(scores) != len(graph_operations):
        raise ValueError("fused score width does not match graph operations")
    order = np.argsort(-scores, kind="stable")[: int(top_k)]
    output: list[dict[str, object]] = []
    for rank, index in enumerate(order, start=1):
        row = dict(graph_operations[int(index)])
        row["rank"] = rank
        row["fused_rank_score"] = float(scores[int(index)])
        output.append(row)
    return output


__all__ = [
    "fuse_operation_scores",
    "preserve_graph_numerics",
    "row_standardize",
    "select_top_operations",
]
