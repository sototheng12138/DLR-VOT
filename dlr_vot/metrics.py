"""Event metrics used by validation-selected operating thresholding."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


def _binary_values(values: Sequence[object], *, name: str) -> tuple[int, ...]:
    parsed: list[int] = []
    for index, raw in enumerate(values):
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name}[{index}] must be 0 or 1") from exc
        if not math.isfinite(value) or value not in (0.0, 1.0):
            raise ValueError(f"{name}[{index}] must be 0 or 1")
        parsed.append(int(value))
    return tuple(parsed)


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


@dataclass(frozen=True)
class EventMetrics:
    """Confusion counts and metrics for active and zero states."""

    tp: int
    fp: int
    tn: int
    fn: int
    fpr: float
    fnr: float
    precision: float
    recall: float
    f1_active: float
    f1_zero: float

    @property
    def samples(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    def to_dict(self) -> dict[str, int | float]:
        return {
            "samples": self.samples,
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "fpr": self.fpr,
            "fnr": self.fnr,
            "precision": self.precision,
            "recall": self.recall,
            "f1_active": self.f1_active,
            "f1_zero": self.f1_zero,
        }


def event_metrics(y_true: Sequence[object], y_pred: Sequence[object]) -> EventMetrics:
    """Compute binary metrics with active records treated as the positive class."""

    truth = _binary_values(y_true, name="y_true")
    prediction = _binary_values(y_pred, name="y_pred")
    if len(truth) != len(prediction):
        raise ValueError("y_true and y_pred must have the same length")
    if not truth:
        raise ValueError("at least one sample is required")

    tp = sum(actual == 1 and predicted == 1 for actual, predicted in zip(truth, prediction))
    fp = sum(actual == 0 and predicted == 1 for actual, predicted in zip(truth, prediction))
    tn = sum(actual == 0 and predicted == 0 for actual, predicted in zip(truth, prediction))
    fn = sum(actual == 1 and predicted == 0 for actual, predicted in zip(truth, prediction))

    return EventMetrics(
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        fpr=_ratio(fp, fp + tn),
        fnr=_ratio(fn, fn + tp),
        precision=_ratio(tp, tp + fp),
        recall=_ratio(tp, tp + fn),
        f1_active=_ratio(2 * tp, 2 * tp + fp + fn),
        f1_zero=_ratio(2 * tn, 2 * tn + fp + fn),
    )
