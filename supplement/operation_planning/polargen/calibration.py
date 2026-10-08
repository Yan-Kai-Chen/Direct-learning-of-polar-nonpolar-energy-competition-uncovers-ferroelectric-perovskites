"""Validation-fitted calibration records for PolarGen operation outputs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class OperationCalibration:
    operation_id: str
    status: str = "direction_only"
    interval_scale: float = 1.0
    interval_offset: float = 0.0
    confidence_slope: float = 1.0
    confidence_intercept: float = 0.0
    calibration_id: str | None = None

    @classmethod
    def from_dict(
        cls,
        operation_id: str,
        raw: Mapping[str, Any] | None,
    ) -> "OperationCalibration":
        if raw is None:
            return cls(operation_id=str(operation_id).upper())
        status = str(raw.get("status", "direction_only")).lower()
        if status not in {"direction_only", "numeric_calibrated"}:
            raise ValueError(
                f"{operation_id}: calibration status must be direction_only "
                "or numeric_calibrated"
            )
        values = cls(
            operation_id=str(operation_id).upper(),
            status=status,
            interval_scale=float(raw.get("interval_scale", 1.0)),
            interval_offset=float(raw.get("interval_offset", 0.0)),
            confidence_slope=float(raw.get("confidence_slope", 1.0)),
            confidence_intercept=float(raw.get("confidence_intercept", 0.0)),
            calibration_id=(
                None
                if raw.get("calibration_id") is None
                else str(raw["calibration_id"])
            ),
        )
        for name in (
            "interval_scale",
            "interval_offset",
            "confidence_slope",
            "confidence_intercept",
        ):
            if not math.isfinite(getattr(values, name)):
                raise ValueError(f"{operation_id}: non-finite calibration {name}")
        if values.interval_scale == 0.0:
            raise ValueError(f"{operation_id}: interval_scale cannot be zero")
        return values

    def calibrate_confidence(self, probability_positive: float) -> float:
        raw_direction_confidence = max(
            float(probability_positive),
            1.0 - float(probability_positive),
        )
        clipped = min(max(raw_direction_confidence, 1.0e-6), 1.0 - 1.0e-6)
        logit = math.log(clipped / (1.0 - clipped))
        calibrated_logit = (
            self.confidence_slope * logit + self.confidence_intercept
        )
        if calibrated_logit >= 0.0:
            calibrated = 1.0 / (1.0 + math.exp(-calibrated_logit))
        else:
            exp_value = math.exp(calibrated_logit)
            calibrated = exp_value / (1.0 + exp_value)
        return min(max(float(calibrated), 0.0), 1.0)

    def calibrate_interval(
        self,
        interval: list[float] | tuple[float, float] | None,
    ) -> list[float] | None:
        if self.status != "numeric_calibrated":
            return None
        if interval is None or len(interval) != 2:
            raise ValueError(
                f"{self.operation_id}: numeric calibration requires a "
                "two-value planner interval"
            )
        values = [
            self.interval_scale * float(value) + self.interval_offset
            for value in interval
        ]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(
                f"{self.operation_id}: calibrated interval is non-finite"
            )
        return [min(values), max(values)]


def load_operation_calibrations(
    raw: Mapping[str, Any] | None,
) -> dict[str, OperationCalibration]:
    if raw is None:
        return {}
    operations = raw.get("operations", raw)
    if not isinstance(operations, Mapping):
        raise ValueError("calibration manifest operations must be an object")
    return {
        str(operation_id).upper(): OperationCalibration.from_dict(
            str(operation_id).upper(),
            value,
        )
        for operation_id, value in operations.items()
    }


__all__ = ["OperationCalibration", "load_operation_calibrations"]
