"""Ranking metrics reported for the PolarGen operation selector."""

from __future__ import annotations

import numpy as np


def ranking_metrics(
    relevance: np.ndarray,
    scores: np.ndarray,
    top_k: int = 3,
) -> dict[str, float | int]:
    """Compute Top-1, Top-k hit, strict recall, exact set, and NDCG."""

    truth = np.asarray(relevance, dtype=np.float32)
    prediction = np.asarray(scores, dtype=np.float32)
    if truth.shape != prediction.shape or truth.ndim != 2:
        raise ValueError("relevance and scores must share a 2D shape")
    if not 1 <= int(top_k) <= truth.shape[1]:
        raise ValueError("top_k is outside the operation vocabulary")
    count = len(truth)
    predicted_order = np.argsort(-prediction, axis=1, kind="stable")
    truth_order = np.argsort(-truth, axis=1, kind="stable")
    predicted_top = predicted_order[:, :top_k]
    truth_top = truth_order[:, :top_k]
    truth_first = truth_order[:, 0]
    top1 = predicted_order[:, 0] == truth_first
    top_hit = np.any(predicted_top == truth_first[:, None], axis=1)
    overlap = np.asarray(
        [
            len(set(predicted_top[row]) & set(truth_top[row]))
            for row in range(count)
        ],
        dtype=np.float32,
    )
    exact = np.asarray(
        [
            set(predicted_top[row]) == set(truth_top[row])
            for row in range(count)
        ],
        dtype=bool,
    )
    gains = np.take_along_axis(truth, predicted_top, axis=1)
    discounts = 1.0 / np.log2(np.arange(2, top_k + 2, dtype=np.float32))
    dcg = ((2.0**gains - 1.0) * discounts[None, :]).sum(axis=1)
    ideal_gains = np.take_along_axis(truth, truth_top, axis=1)
    ideal = ((2.0**ideal_gains - 1.0) * discounts[None, :]).sum(axis=1)
    ndcg = np.divide(dcg, ideal, out=np.zeros_like(dcg), where=ideal > 0)
    return {
        "rows": int(count),
        "top1": float(top1.mean()),
        "top3_hit": float(top_hit.mean()),
        "strict_top3_recall": float((overlap / float(top_k)).mean()),
        "exact_top3": float(exact.mean()),
        "ndcg_at_3": float(ndcg.mean()),
    }


__all__ = ["ranking_metrics"]
