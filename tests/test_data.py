from __future__ import annotations

import csv
from pathlib import Path

import pytest

from dlr_vot.data import DataValidationError, chronological_split, validate_completed_daily_csv


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["date", "a", "b"])
        writer.writeheader()
        writer.writerows(rows)


def test_completed_daily_csv_returns_summary(tmp_path: Path) -> None:
    path = tmp_path / "daily.csv"
    _write_csv(
        path,
        [
            {"date": "2024-01-01", "a": 0, "b": 2.5},
            {"date": "2024-01-02", "a": 3, "b": 0},
            {"date": "2024-01-03", "a": 0, "b": 0},
        ],
    )

    summary = validate_completed_daily_csv(path)

    assert summary.rows == 3
    assert summary.start_date == "2024-01-01"
    assert summary.end_date == "2024-01-03"
    assert summary.value_columns == ("a", "b")
    assert summary.zero_fraction == pytest.approx({"a": 2 / 3, "b": 2 / 3})
    assert summary.to_dict()["rows"] == 3


def test_executed_slash_date_format_is_accepted(tmp_path: Path) -> None:
    path = tmp_path / "daily.csv"
    _write_csv(
        path,
        [
            {"date": "2024/1/1", "a": 0, "b": 1},
            {"date": "2024/1/2", "a": 2, "b": 0},
        ],
    )

    summary = validate_completed_daily_csv(path)
    assert summary.start_date == "2024-01-01"
    assert summary.end_date == "2024-01-02"


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (
            [
                {"date": "2024-01-01", "a": 0, "b": 1},
                {"date": "2024-01-03", "a": 2, "b": 0},
            ],
            "missing date 2024-01-02",
        ),
        (
            [
                {"date": "2024-01-01", "a": 0, "b": 1},
                {"date": "2024-01-01", "a": 2, "b": 0},
            ],
            "dates must be unique",
        ),
        (
            [{"date": "2024-01-01", "a": -1, "b": 0}],
            "must be nonnegative",
        ),
        (
            [{"date": "2024-01-01", "a": "not-a-number", "b": 0}],
            "must be numeric",
        ),
    ],
)
def test_invalid_daily_csv_is_rejected(
    tmp_path: Path, rows: list[dict[str, object]], message: str
) -> None:
    path = tmp_path / "invalid.csv"
    _write_csv(path, rows)

    with pytest.raises(DataValidationError, match=message):
        validate_completed_daily_csv(path)


def test_selected_value_columns_must_exist(tmp_path: Path) -> None:
    path = tmp_path / "daily.csv"
    _write_csv(path, [{"date": "2024-01-01", "a": 0, "b": 1}])

    with pytest.raises(DataValidationError, match="value column not found"):
        validate_completed_daily_csv(path, value_columns=["a", "missing"])


def test_date_column_must_be_first(tmp_path: Path) -> None:
    path = tmp_path / "daily.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["a", "date", "b"])
        writer.writeheader()
        writer.writerow({"a": 0, "date": "2024-01-01", "b": 1})

    with pytest.raises(DataValidationError, match="date column must be the first"):
        validate_completed_daily_csv(path)


def test_paper_chronological_split_and_origin_counts() -> None:
    split = chronological_split(1096, context_length=96, forecast_horizon=48)

    assert (split.train_days, split.validation_days, split.test_days) == (767, 110, 219)
    assert (split.train_origins, split.validation_origins, split.test_origins) == (624, 63, 172)
