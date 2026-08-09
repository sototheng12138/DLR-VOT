#!/usr/bin/env python3
"""Validate a completed daily CSV before model fitting."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from dlr_vot import (  # noqa: E402
    DataValidationError,
    chronological_split,
    validate_completed_daily_csv,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--date-column")
    parser.add_argument(
        "--value-columns",
        nargs="+",
        help="columns to check; by default all columns except the date are checked",
    )
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument(
        "--expected-channels",
        type=int,
        default=4,
        help="required number of value columns; the paper model uses four",
    )
    parser.add_argument("--expected-sha256")
    parser.add_argument("--context-length", type=int, default=96)
    parser.add_argument("--forecast-horizon", type=int, default=48)
    args = parser.parse_args()

    try:
        summary = validate_completed_daily_csv(
            args.input,
            date_column=args.date_column,
            value_columns=args.value_columns,
        )
        if args.expected_rows is not None and summary.rows != args.expected_rows:
            raise DataValidationError(
                f"row count differs: expected {args.expected_rows}, found {summary.rows}"
            )
        if args.expected_channels <= 0:
            raise DataValidationError("expected channel count must be positive")
        if len(summary.value_columns) != args.expected_channels:
            raise DataValidationError(
                "channel count differs: "
                f"expected {args.expected_channels}, found {len(summary.value_columns)}"
            )
        if args.expected_sha256 is not None:
            digest = hashlib.sha256(args.input.read_bytes()).hexdigest()
            if digest.lower() != args.expected_sha256.lower():
                raise DataValidationError(
                    f"SHA-256 differs: expected {args.expected_sha256.lower()}, found {digest}"
                )
    except DataValidationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    payload = summary.to_dict()
    try:
        payload["chronological_split"] = chronological_split(
            summary.rows,
            context_length=args.context_length,
            forecast_horizon=args.forecast_horizon,
        ).to_dict()
    except ValueError as exc:
        payload["chronological_split"] = {"available": False, "reason": str(exc)}
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
