from __future__ import annotations

import pytest

from dlr_vot.metrics import event_metrics


def test_event_metrics_match_confusion_counts() -> None:
    metrics = event_metrics(
        [1, 1, 1, 0, 0, 0],
        [1, 0, 1, 1, 0, 0],
    )

    assert (metrics.tp, metrics.fp, metrics.tn, metrics.fn) == (2, 1, 2, 1)
    assert metrics.fpr == pytest.approx(1 / 3)
    assert metrics.fnr == pytest.approx(1 / 3)
    assert metrics.precision == pytest.approx(2 / 3)
    assert metrics.recall == pytest.approx(2 / 3)
    assert metrics.f1_active == pytest.approx(2 / 3)
    assert metrics.f1_zero == pytest.approx(2 / 3)


def test_zero_denominators_are_reported_as_zero() -> None:
    all_active = event_metrics([1, 1], [0, 0])
    assert all_active.fpr == 0.0
    assert all_active.precision == 0.0
    assert all_active.f1_active == 0.0

    all_zero = event_metrics([0, 0], [0, 0])
    assert all_zero.fnr == 0.0
    assert all_zero.recall == 0.0
    assert all_zero.f1_zero == 1.0


def test_event_metrics_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="same length"):
        event_metrics([1, 0], [1])
    with pytest.raises(ValueError, match="must be 0 or 1"):
        event_metrics([1, 2], [1, 0])
    with pytest.raises(ValueError, match="at least one sample"):
        event_metrics([], [])
