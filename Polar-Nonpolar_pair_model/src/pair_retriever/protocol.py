from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


POLAR_COLUMN = "Polar_mpid"
NONPOLAR_COLUMN = "NPolar_mpid"


def normalize_structure_id(value: object) -> str:
    text = os.path.basename(str(value).strip())
    lowered = text.lower()
    for suffix in (".cif.gz", ".cif"):
        if lowered.endswith(suffix):
            text = text[: -len(suffix)]
            break
    return text.strip().lower()


def load_pair_table(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str)
    missing = [
        column
        for column in (POLAR_COLUMN, NONPOLAR_COLUMN)
        if column not in frame.columns
    ]
    if missing:
        raise KeyError(f"Pair table is missing columns: {missing}")
    result = frame[[POLAR_COLUMN, NONPOLAR_COLUMN]].copy()
    result[POLAR_COLUMN] = result[POLAR_COLUMN].map(normalize_structure_id)
    result[NONPOLAR_COLUMN] = result[NONPOLAR_COLUMN].map(
        normalize_structure_id
    )
    empty = (
        result[POLAR_COLUMN].isin({"", "nan"})
        | result[NONPOLAR_COLUMN].isin({"", "nan"})
    )
    if empty.any():
        raise ValueError(f"Pair table contains {int(empty.sum())} empty IDs")
    return result


@dataclass(frozen=True)
class PairSplitAudit:
    n_pairs_train: int
    n_pairs_test: int
    n_unique_pairs_train: int
    n_unique_pairs_test: int
    n_polars_train: int
    n_polars_test: int
    n_nonpolars_train: int
    n_nonpolars_test: int
    polar_overlap_train_test: int
    nonpolar_overlap_train_test: int
    exact_pair_overlap_train_test: int
    duplicate_rows_train: int
    duplicate_rows_test: int
    polar_holdout_passed: bool

    def to_dict(self) -> dict[str, int | bool]:
        return asdict(self)


def _pair_set(frame: pd.DataFrame) -> set[tuple[str, str]]:
    return set(
        frame[[POLAR_COLUMN, NONPOLAR_COLUMN]].itertuples(
            index=False,
            name=None,
        )
    )


def audit_polar_holdout(
    train_pairs: pd.DataFrame,
    test_pairs: pd.DataFrame,
) -> PairSplitAudit:
    train_polars = set(train_pairs[POLAR_COLUMN])
    test_polars = set(test_pairs[POLAR_COLUMN])
    train_nonpolars = set(train_pairs[NONPOLAR_COLUMN])
    test_nonpolars = set(test_pairs[NONPOLAR_COLUMN])
    pair_overlap = _pair_set(train_pairs) & _pair_set(test_pairs)
    polar_overlap = train_polars & test_polars
    return PairSplitAudit(
        n_pairs_train=int(len(train_pairs)),
        n_pairs_test=int(len(test_pairs)),
        n_unique_pairs_train=int(len(_pair_set(train_pairs))),
        n_unique_pairs_test=int(len(_pair_set(test_pairs))),
        n_polars_train=int(len(train_polars)),
        n_polars_test=int(len(test_polars)),
        n_nonpolars_train=int(len(train_nonpolars)),
        n_nonpolars_test=int(len(test_nonpolars)),
        polar_overlap_train_test=int(len(polar_overlap)),
        nonpolar_overlap_train_test=int(
            len(train_nonpolars & test_nonpolars)
        ),
        exact_pair_overlap_train_test=int(len(pair_overlap)),
        duplicate_rows_train=int(train_pairs.duplicated().sum()),
        duplicate_rows_test=int(test_pairs.duplicated().sum()),
        polar_holdout_passed=not polar_overlap and not pair_overlap,
    )


@dataclass(frozen=True)
class SymmetryBiasConfig:
    weight: float = 0.15
    operations_normalizer: float = 192.0
    clamp_delta: float = 1.0
    symprec: float = 0.01


def apply_symmetry_bias(
    base_scores: Iterable[float],
    *,
    polar_operation_count: int | float | None,
    candidate_operation_counts: Iterable[int | float | None],
    config: SymmetryBiasConfig = SymmetryBiasConfig(),
) -> np.ndarray:
    scores = np.asarray(list(base_scores), dtype=np.float32)
    candidate_counts = list(candidate_operation_counts)
    if len(candidate_counts) != len(scores):
        raise ValueError(
            "candidate_operation_counts and base_scores must have equal length"
        )
    if polar_operation_count is None:
        return scores.copy()
    if config.operations_normalizer <= 0 or config.clamp_delta < 0:
        raise ValueError("Symmetry normalization and clamp must be valid")
    nonpolar = np.asarray(
        [0.0 if value is None else float(value) for value in candidate_counts],
        dtype=np.float32,
    )
    delta = (
        nonpolar - np.float32(float(polar_operation_count))
    ) / np.float32(config.operations_normalizer)
    delta = np.clip(delta, -config.clamp_delta, config.clamp_delta)
    return (
        scores + np.float32(config.weight) * delta
    ).astype(np.float32)
