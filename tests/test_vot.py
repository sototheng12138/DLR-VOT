from __future__ import annotations

import inspect
import json

import pytest

from dlr_vot.metrics import EventMetrics
from dlr_vot.vot import (
    _candidate_key,
    apply_threshold,
    select_vot_threshold,
)


def _metrics(*, f1: float, fpr: float, fnr: float) -> EventMetrics:
    return EventMetrics(
        tp=0,
        fp=0,
        tn=0,
        fn=0,
        fpr=fpr,
        fnr=fnr,
        precision=0.0,
        recall=0.0,
        f1_active=f1,
        f1_zero=0.0,
    )


def test_threshold_rule_is_strict() -> None:
    assert apply_threshold([0.4, 0.5, 0.6], 0.5) == (0, 0, 1)


def test_vot_maximizes_f1_within_fpr_budget() -> None:
    scores = [0.90, 0.80, 0.65, 0.60, 0.20, 0.10]
    labels = [1, 1, 0, 1, 0, 0]

    result = select_vot_threshold(scores, labels, fpr_budget=0.0)

    assert result.threshold == pytest.approx(0.65)
    assert result.validation_metrics.fpr == 0.0
    assert result.validation_metrics.tp == 2
    assert result.validation_metrics.fn == 1
    assert result.validation_metrics.f1_active == pytest.approx(0.8)


def test_tie_break_prefers_lower_fpr() -> None:
    scores = [0.9, 0.5, 0.6, 0.55]
    labels = [1, 1, 0, 0]

    result = select_vot_threshold(
        scores,
        labels,
        fpr_budget=1.0,
        candidate_thresholds=[0.4, 0.7],
    )

    assert result.threshold == pytest.approx(0.7)
    assert result.validation_metrics.f1_active == pytest.approx(2 / 3)
    assert result.validation_metrics.fpr == 0.0


def test_tie_break_order_includes_fnr_and_larger_threshold() -> None:
    same_f1_lower_fnr = _candidate_key(_metrics(f1=0.7, fpr=0.1, fnr=0.2), 0.4)
    same_f1_higher_fnr = _candidate_key(_metrics(f1=0.7, fpr=0.1, fnr=0.3), 0.9)
    assert same_f1_lower_fnr > same_f1_higher_fnr

    same_metrics_lower_tau = _candidate_key(_metrics(f1=0.7, fpr=0.1, fnr=0.2), 0.4)
    same_metrics_higher_tau = _candidate_key(_metrics(f1=0.7, fpr=0.1, fnr=0.2), 0.5)
    assert same_metrics_higher_tau > same_metrics_lower_tau


def test_equivalent_decisions_choose_larger_threshold() -> None:
    result = select_vot_threshold(
        [0.9, 0.4],
        [1, 0],
        fpr_budget=0.0,
        candidate_thresholds=[0.5, 0.8],
    )
    assert result.threshold == pytest.approx(0.8)


def test_result_is_strict_json() -> None:
    result = select_vot_threshold([0.8, 0.2], [1, 0], fpr_budget=0.0)
    payload = json.loads(result.to_json())
    assert payload["threshold"] == result.threshold
    assert payload["decision_rule"] == "active if score > threshold"


def test_selector_api_has_no_test_labels() -> None:
    parameters = inspect.signature(select_vot_threshold).parameters
    assert tuple(parameters) == (
        "validation_scores",
        "validation_labels",
        "fpr_budget",
        "candidate_thresholds",
    )
