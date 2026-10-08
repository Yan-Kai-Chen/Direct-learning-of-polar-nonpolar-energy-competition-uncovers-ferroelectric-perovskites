from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import pandas as pd

from .protocol import NONPOLAR_COLUMN, POLAR_COLUMN, normalize_structure_id


@dataclass(frozen=True)
class RankingAudit:
    n_total_queries: int
    n_ranked_queries: int
    n_evaluable_queries: int
    coverage_gt_in_pool: float
    mrr: float
    hit_at: dict[int, float]
    duplicate_predictions: int
    non_contiguous_rank_queries: int

    def to_dict(self) -> dict[str, object]:
        output = asdict(self)
        output["hit_at"] = {
            str(key): value for key, value in self.hit_at.items()
        }
        return output


def evaluate_topk_predictions(
    test_pairs: pd.DataFrame,
    ranked_predictions: pd.DataFrame,
    *,
    ks: Iterable[int] = (1, 3, 5, 10),
    rank_column: str = "Rank",
    prediction_column: str = "Pred_NPolar_mpid",
    coverage_column: str = "Has_GT_in_pool",
) -> RankingAudit:
    required = {
        POLAR_COLUMN,
        rank_column,
        prediction_column,
        coverage_column,
    }
    missing = sorted(required - set(ranked_predictions.columns))
    if missing:
        raise KeyError(f"Ranked predictions are missing columns: {missing}")

    truth: dict[str, set[str]] = {}
    for polar, nonpolar in test_pairs[
        [POLAR_COLUMN, NONPOLAR_COLUMN]
    ].itertuples(index=False, name=None):
        truth.setdefault(normalize_structure_id(polar), set()).add(
            normalize_structure_id(nonpolar)
        )

    ranking = ranked_predictions.copy()
    ranking[POLAR_COLUMN] = ranking[POLAR_COLUMN].map(normalize_structure_id)
    ranking[prediction_column] = ranking[prediction_column].map(
        normalize_structure_id
    )
    ranking[rank_column] = pd.to_numeric(
        ranking[rank_column],
        errors="raise",
    ).astype(int)
    ranking[coverage_column] = pd.to_numeric(
        ranking[coverage_column],
        errors="raise",
    ).astype(int)
    ranking = ranking.sort_values(
        [POLAR_COLUMN, rank_column],
        kind="mergesort",
    )
    requested_ks = tuple(sorted({int(value) for value in ks}))
    if not requested_ks or requested_ks[0] <= 0:
        raise ValueError("ks must contain positive integers")

    hit_counts = {key: 0 for key in requested_ks}
    reciprocal_rank_sum = 0.0
    evaluable = 0
    duplicate_predictions = 0
    non_contiguous = 0
    groups = list(ranking.groupby(POLAR_COLUMN, sort=False))
    for polar, group in groups:
        ranks = group[rank_column].tolist()
        if ranks != list(range(1, len(ranks) + 1)):
            non_contiguous += 1
        predictions = group[prediction_column].tolist()
        duplicate_predictions += len(predictions) - len(set(predictions))
        if not bool(group[coverage_column].max()):
            continue
        evaluable += 1
        ground_truth = truth.get(polar, set())
        first_hit = next(
            (
                index
                for index, candidate in enumerate(predictions, start=1)
                if candidate in ground_truth
            ),
            None,
        )
        if first_hit is not None:
            reciprocal_rank_sum += 1.0 / first_hit
        for key in requested_ks:
            if any(candidate in ground_truth for candidate in predictions[:key]):
                hit_counts[key] += 1

    denominator = max(evaluable, 1)
    return RankingAudit(
        n_total_queries=int(len(truth)),
        n_ranked_queries=int(len(groups)),
        n_evaluable_queries=int(evaluable),
        coverage_gt_in_pool=float(evaluable / max(len(truth), 1)),
        mrr=float(reciprocal_rank_sum / denominator),
        hit_at={
            key: float(hit_counts[key] / denominator)
            for key in requested_ks
        },
        duplicate_predictions=int(duplicate_predictions),
        non_contiguous_rank_queries=int(non_contiguous),
    )
