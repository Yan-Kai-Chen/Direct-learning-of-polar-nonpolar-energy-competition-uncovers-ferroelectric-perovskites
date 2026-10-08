from __future__ import annotations

import hashlib
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


POLAR_COLUMN = "Polar_mpid"
NONPOLAR_COLUMN = "NPolar_mpid"
TARGET_COLUMN = "Energy_diff_meV"
SPLIT_COLUMN = "split"
REQUIRED_COLUMNS = {
    POLAR_COLUMN,
    NONPOLAR_COLUMN,
    TARGET_COLUMN,
    SPLIT_COLUMN,
}
EXCLUDED_FEATURE_COLUMNS = {
    "row_idx",
    POLAR_COLUMN,
    NONPOLAR_COLUMN,
    TARGET_COLUMN,
    SPLIT_COLUMN,
    "TARGET",
    "Target",
    "pair_key",
    "polar_id",
    "npolar_id",
}


def normalize_structure_id(value: object) -> str:
    text = os.path.basename(str(value).strip())
    lowered = text.lower()
    for suffix in (".cif.gz", ".cif"):
        if lowered.endswith(suffix):
            text = text[: -len(suffix)]
            break
    return text.strip()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_indices(
    frame: pd.DataFrame,
    split_column: str = SPLIT_COLUMN,
) -> dict[str, np.ndarray]:
    if split_column not in frame.columns:
        raise KeyError(f"Missing split column: {split_column}")
    labels = frame[split_column].astype(str).str.strip().str.lower()
    unexpected = sorted(set(labels) - {"train", "val", "test"})
    if unexpected:
        raise ValueError(f"Unexpected split labels: {unexpected}")
    result = {
        label: np.flatnonzero(labels.to_numpy() == label).astype(np.int64)
        for label in ("train", "val", "test")
    }
    if any(len(values) == 0 for values in result.values()):
        raise ValueError("train, val, and test must all be non-empty")
    joined = np.concatenate(list(result.values()))
    if len(joined) != len(frame) or len(np.unique(joined)) != len(frame):
        raise ValueError("Split labels do not partition the full table")
    return result


def derive_binary_label(
    energy_mev: pd.Series | np.ndarray,
    threshold_mev: float = 70.0,
) -> np.ndarray:
    values = np.asarray(energy_mev, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Energy target contains non-finite values")
    return np.where(np.abs(values) < threshold_mev, 1, -1).astype(np.int8)


@dataclass(frozen=True)
class TrainingTableAudit:
    path: str
    sha256: str
    rows: int
    columns: int
    split_counts: dict[str, int]
    unique_pairs: int
    duplicate_pairs: int
    unique_structures: int
    polar_overlap_train_val: int
    polar_overlap_train_test: int
    polar_overlap_val_test: int
    finite_target: bool
    row_idx_present: bool
    row_idx_unique: bool
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def audit_training_table(
    path: str | Path,
) -> tuple[pd.DataFrame, TrainingTableAudit]:
    data_path = Path(path)
    if not data_path.is_file():
        raise FileNotFoundError(f"Training table not found: {data_path}")
    frame = pd.read_csv(data_path, low_memory=False)
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise KeyError(f"Training table is missing columns: {missing}")

    indices = split_indices(frame)
    ids = frame[[POLAR_COLUMN, NONPOLAR_COLUMN]].copy()
    ids[POLAR_COLUMN] = ids[POLAR_COLUMN].map(normalize_structure_id)
    ids[NONPOLAR_COLUMN] = ids[NONPOLAR_COLUMN].map(normalize_structure_id)
    if ids.eq("").any().any() or ids.apply(
        lambda column: column.str.lower().eq("nan")
    ).any().any():
        raise ValueError("Training table contains empty structure identifiers")

    target = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce").to_numpy()
    finite_target = bool(np.isfinite(target).all())
    polar_sets = {
        name: set(ids.iloc[index][POLAR_COLUMN])
        for name, index in indices.items()
    }
    duplicate_pairs = int(ids.duplicated().sum())
    row_idx_present = "row_idx" in frame.columns
    row_idx_unique = False
    if row_idx_present:
        row_idx = pd.to_numeric(frame["row_idx"], errors="coerce")
        row_idx_unique = bool(
            row_idx.notna().all() and row_idx.nunique() == len(frame)
        )
    overlap_train_val = len(polar_sets["train"] & polar_sets["val"])
    overlap_train_test = len(polar_sets["train"] & polar_sets["test"])
    overlap_val_test = len(polar_sets["val"] & polar_sets["test"])
    passed = bool(
        finite_target
        and duplicate_pairs == 0
        and overlap_train_val == 0
        and overlap_train_test == 0
        and overlap_val_test == 0
    )
    audit = TrainingTableAudit(
        path=str(data_path),
        sha256=file_sha256(data_path),
        rows=int(len(frame)),
        columns=int(len(frame.columns)),
        split_counts={name: int(len(value)) for name, value in indices.items()},
        unique_pairs=int(len(ids.drop_duplicates())),
        duplicate_pairs=duplicate_pairs,
        unique_structures=int(
            len(set(ids[POLAR_COLUMN]) | set(ids[NONPOLAR_COLUMN]))
        ),
        polar_overlap_train_val=int(overlap_train_val),
        polar_overlap_train_test=int(overlap_train_test),
        polar_overlap_val_test=int(overlap_val_test),
        finite_target=finite_target,
        row_idx_present=row_idx_present,
        row_idx_unique=row_idx_unique,
        passed=passed,
    )
    return frame, audit


def select_numeric_features(frame: pd.DataFrame) -> pd.DataFrame:
    candidate = frame.drop(
        columns=[
            column
            for column in EXCLUDED_FEATURE_COLUMNS
            if column in frame.columns
        ],
        errors="ignore",
    )
    names = [
        column
        for column in candidate.columns
        if pd.api.types.is_numeric_dtype(candidate[column])
    ]
    return candidate.loc[:, names].copy()


def train_fitted_imputation(
    numeric: pd.DataFrame,
    train_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    medians = numeric.iloc[train_index].median(numeric_only=True).to_numpy(
        dtype=np.float32
    )
    medians = np.where(np.isfinite(medians), medians, 0.0)
    values = numeric.to_numpy(dtype=np.float32, copy=True)
    values = np.where(np.isfinite(values), values, medians[None, :])
    return values.astype(np.float32), medians.astype(np.float32)
