"""Validation for the completed daily input table."""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Sequence


class DataValidationError(ValueError):
    """Raised when a daily input table does not follow the public schema."""


@dataclass(frozen=True)
class DatasetSummary:
    """Non-identifying summary returned after schema validation."""

    path: str
    date_column: str
    value_columns: tuple[str, ...]
    rows: int
    start_date: str
    end_date: str
    zero_fraction: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "date_column": self.date_column,
            "value_columns": list(self.value_columns),
            "rows": self.rows,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "zero_fraction": dict(self.zero_fraction),
        }


@dataclass(frozen=True)
class ChronologicalSplit:
    """Day and forecast-origin counts for the paper split."""

    train_days: int
    validation_days: int
    test_days: int
    train_origins: int
    validation_origins: int
    test_origins: int
    context_length: int
    forecast_horizon: int

    def to_dict(self) -> dict[str, int]:
        return {
            "train_days": self.train_days,
            "validation_days": self.validation_days,
            "test_days": self.test_days,
            "train_origins": self.train_origins,
            "validation_origins": self.validation_origins,
            "test_origins": self.test_origins,
            "context_length": self.context_length,
            "forecast_horizon": self.forecast_horizon,
        }


def chronological_split(
    rows: int, *, context_length: int = 96, forecast_horizon: int = 48
) -> ChronologicalSplit:
    """Return the 7:1:2 split and rolling-origin counts used by the loaders."""

    if rows <= 0:
        raise ValueError("rows must be positive")
    if context_length <= 0 or forecast_horizon <= 0:
        raise ValueError("context_length and forecast_horizon must be positive")
    train_days = int(rows * 0.7)
    test_days = int(rows * 0.2)
    validation_days = rows - train_days - test_days
    train_origins = train_days - context_length - forecast_horizon + 1
    validation_origins = validation_days - forecast_horizon + 1
    test_origins = test_days - forecast_horizon + 1
    if min(train_origins, validation_origins, test_origins) <= 0:
        raise ValueError("one or more splits are too short for the requested windows")
    return ChronologicalSplit(
        train_days=train_days,
        validation_days=validation_days,
        test_days=test_days,
        train_origins=train_origins,
        validation_origins=validation_origins,
        test_origins=test_origins,
        context_length=context_length,
        forecast_horizon=forecast_horizon,
    )


def _parse_date(raw: str, *, row_number: int, column: str) -> date:
    formats = []
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw):
        formats.append("%Y-%m-%d")
    if re.fullmatch(r"\d{4}/\d{1,2}/\d{1,2}", raw):
        formats.append("%Y/%m/%d")
    for date_format in formats:
        try:
            return datetime.strptime(raw, date_format).date()
        except ValueError:
            pass
    raise DataValidationError(f"row {row_number}: {column!r} must use YYYY-MM-DD or YYYY/M/D")


def _parse_value(raw: str, *, row_number: int, column: str) -> float:
    if raw == "":
        raise DataValidationError(f"row {row_number}: {column!r} is empty")
    try:
        value = float(raw)
    except ValueError as exc:
        raise DataValidationError(f"row {row_number}: {column!r} must be numeric") from exc
    if not math.isfinite(value):
        raise DataValidationError(f"row {row_number}: {column!r} must be finite")
    if value < 0:
        raise DataValidationError(f"row {row_number}: {column!r} must be nonnegative")
    return value


def validate_completed_daily_csv(
    path: str | Path,
    *,
    date_column: str | None = None,
    value_columns: Sequence[str] | None = None,
) -> DatasetSummary:
    """Validate a complete, chronological daily table.

    The date column must be first and contain one supported calendar date per row with no gaps.
    Selected value columns must be finite, numeric, and nonnegative. When
    ``value_columns`` is omitted, every column other than the date is checked.
    """

    csv_path = Path(path)
    if not csv_path.is_file():
        raise DataValidationError(f"input file does not exist: {csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise DataValidationError("the CSV header is missing")
        if any(name is None or name == "" for name in fieldnames):
            raise DataValidationError("the CSV header contains an empty column name")
        if len(fieldnames) != len(set(fieldnames)):
            raise DataValidationError("the CSV header contains duplicate column names")
        if date_column is None:
            date_column = "date" if "date" in fieldnames else "日期"
        if date_column not in fieldnames:
            raise DataValidationError("date column not found; expected 'date' or '日期'")
        if fieldnames[0] != date_column:
            raise DataValidationError("the date column must be the first column")

        if value_columns is None:
            checked_columns = tuple(name for name in fieldnames if name != date_column)
        else:
            checked_columns = tuple(value_columns)
            if len(checked_columns) != len(set(checked_columns)):
                raise DataValidationError("value_columns contains duplicates")
            missing = [name for name in checked_columns if name not in fieldnames]
            if missing:
                raise DataValidationError(
                    "value column not found: " + ", ".join(repr(name) for name in missing)
                )
        if not checked_columns:
            raise DataValidationError("at least one value column is required")
        if date_column in checked_columns:
            raise DataValidationError("the date column cannot also be a value column")

        first_date: date | None = None
        previous_date: date | None = None
        zero_counts = {name: 0 for name in checked_columns}
        row_count = 0

        for row_number, row in enumerate(reader, start=2):
            if None in row:
                raise DataValidationError(f"row {row_number}: too many fields")
            if any(value is None for value in row.values()):
                raise DataValidationError(f"row {row_number}: too few fields")

            current_date = _parse_date(row[date_column], row_number=row_number, column=date_column)
            if previous_date is not None:
                expected = previous_date + timedelta(days=1)
                if current_date != expected:
                    if current_date <= previous_date:
                        reason = "dates must be unique and strictly increasing"
                    else:
                        reason = f"missing date {expected.isoformat()}"
                    raise DataValidationError(f"row {row_number}: {reason}")

            for column in checked_columns:
                value = _parse_value(row[column], row_number=row_number, column=column)
                if value == 0.0:
                    zero_counts[column] += 1

            if first_date is None:
                first_date = current_date
            previous_date = current_date
            row_count += 1

    if row_count == 0 or first_date is None or previous_date is None:
        raise DataValidationError("the CSV contains no data rows")

    return DatasetSummary(
        path=str(csv_path),
        date_column=date_column,
        value_columns=checked_columns,
        rows=row_count,
        start_date=first_date.isoformat(),
        end_date=previous_date.isoformat(),
        zero_fraction={column: zero_counts[column] / row_count for column in checked_columns},
    )
