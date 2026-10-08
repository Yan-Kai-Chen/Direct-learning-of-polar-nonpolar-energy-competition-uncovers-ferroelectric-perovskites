from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd


def _safe_ratio(left: pd.Series, right: pd.Series) -> pd.Series:
    denominator = pd.to_numeric(right, errors="coerce")
    numerator = pd.to_numeric(left, errors="coerce")
    return numerator / denominator.replace(0.0, np.nan)


OPERATIONS: dict[
    str,
    Callable[[pd.Series, pd.Series], pd.Series],
] = {
    "difference": lambda left, right: pd.to_numeric(
        left, errors="coerce"
    )
    - pd.to_numeric(right, errors="coerce"),
    "absolute_difference": lambda left, right: (
        pd.to_numeric(left, errors="coerce")
        - pd.to_numeric(right, errors="coerce")
    ).abs(),
    "sum": lambda left, right: pd.to_numeric(left, errors="coerce")
    + pd.to_numeric(right, errors="coerce"),
    "product": lambda left, right: pd.to_numeric(left, errors="coerce")
    * pd.to_numeric(right, errors="coerce"),
    "ratio": _safe_ratio,
}


def run_derived_features_stage(
    frame: pd.DataFrame,
    rules: dict[str, Any],
) -> pd.DataFrame:
    """Apply auditable binary feature operations declared in a rule mapping."""

    output = frame.copy()
    operations = rules.get("operations", [])
    if not isinstance(operations, list):
        raise TypeError("DERIVED_RULES['operations'] must be a list")
    for index, spec in enumerate(operations):
        if not isinstance(spec, dict):
            raise TypeError(f"Derived operation {index} must be a mapping")
        required_keys = {"output", "left", "right", "operation"}
        missing_keys = sorted(required_keys - set(spec))
        if missing_keys:
            raise KeyError(
                f"Derived operation {index} is missing keys: {missing_keys}"
            )
        output_name = str(spec["output"])
        left_name = str(spec["left"])
        right_name = str(spec["right"])
        operation_name = str(spec["operation"])
        required = bool(spec.get("required", True))
        missing_columns = [
            name for name in (left_name, right_name) if name not in output
        ]
        if missing_columns:
            if required:
                raise KeyError(
                    f"Derived operation {output_name!r} is missing columns: "
                    f"{missing_columns}"
                )
            continue
        try:
            operation = OPERATIONS[operation_name]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported derived operation: {operation_name}"
            ) from exc
        output[output_name] = operation(
            output[left_name],
            output[right_name],
        )
    return output
