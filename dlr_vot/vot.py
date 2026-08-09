"""Validation-selected operating thresholding."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Sequence

from .metrics import EventMetrics, event_metrics


def _scores(values: Sequence[object], *, name: str) -> tuple[float, ...]:
    parsed: list[float] = []
    for index, raw in enumerate(values):
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name}[{index}] must be numeric") from exc
        if not math.isfinite(value):
            raise ValueError(f"{name}[{index}] must be finite")
        parsed.append(value)
    if not parsed:
        raise ValueError(f"{name} must contain at least one value")
    return tuple(parsed)


def apply_threshold(scores: Sequence[object], threshold: float) -> tuple[int, ...]:
    """Return one where a score is strictly greater than the threshold."""

    parsed_scores = _scores(scores, name="scores")
    tau = float(threshold)
    if not math.isfinite(tau):
        raise ValueError("threshold must be finite")
    return tuple(int(score > tau) for score in parsed_scores)


@dataclass(frozen=True)
class VOTResult:
    """Threshold selected from validation data and its validation metrics."""

    threshold: float
    fpr_budget: float
    validation_metrics: EventMetrics
    candidates_evaluated: int
    feasible_candidates: int

    def to_dict(self) -> dict[str, object]:
        return {
            "threshold": self.threshold,
            "decision_rule": "active if score > threshold",
            "fpr_budget": self.fpr_budget,
            "objective": "maximize validation f1_active subject to the fpr budget",
            "tie_break": ["lower fpr", "lower fnr", "larger threshold"],
            "validation_metrics": self.validation_metrics.to_dict(),
            "candidates_evaluated": self.candidates_evaluated,
            "feasible_candidates": self.feasible_candidates,
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False, indent=indent)


def _candidate_key(metrics: EventMetrics, threshold: float) -> tuple[float, ...]:
    """Order feasible candidates according to the declared VOT policy."""

    return (
        metrics.f1_active,
        -metrics.fpr,
        -metrics.fnr,
        threshold,
    )


def _default_thresholds(scores: tuple[float, ...]) -> tuple[float, ...]:
    unique_scores = sorted(set(scores))
    below_minimum = math.nextafter(unique_scores[0], -math.inf)
    thresholds = unique_scores.copy()
    if math.isfinite(below_minimum):
        thresholds.insert(0, below_minimum)
    return tuple(thresholds)


def select_vot_threshold(
    validation_scores: Sequence[object],
    validation_labels: Sequence[object],
    *,
    fpr_budget: float,
    candidate_thresholds: Sequence[object] | None = None,
) -> VOTResult:
    """Select a threshold using validation scores and labels only.

    Feasible thresholds satisfy ``FPR <= fpr_budget``. Among them, the
    selector maximizes active-state F1. Ties are resolved by lower FPR, lower
    FNR, and then the larger threshold.
    """

    scores = _scores(validation_scores, name="validation_scores")
    if len(scores) != len(validation_labels):
        raise ValueError("validation_scores and validation_labels must have the same length")
    budget = float(fpr_budget)
    if not math.isfinite(budget) or not 0.0 <= budget <= 1.0:
        raise ValueError("fpr_budget must be between 0 and 1")

    if candidate_thresholds is None:
        thresholds = _default_thresholds(scores)
    else:
        thresholds = tuple(sorted(set(_scores(candidate_thresholds, name="candidate_thresholds"))))

    best: tuple[EventMetrics, float] | None = None
    feasible_count = 0
    for threshold in thresholds:
        prediction = tuple(int(score > threshold) for score in scores)
        metrics = event_metrics(validation_labels, prediction)
        if metrics.fpr > budget:
            continue
        feasible_count += 1
        if best is None or _candidate_key(metrics, threshold) > _candidate_key(best[0], best[1]):
            best = (metrics, threshold)

    if best is None:
        raise ValueError("no candidate threshold satisfies the FPR budget")

    return VOTResult(
        threshold=best[1],
        fpr_budget=budget,
        validation_metrics=best[0],
        candidates_evaluated=len(thresholds),
        feasible_candidates=feasible_count,
    )
