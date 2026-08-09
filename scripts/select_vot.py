#!/usr/bin/env python3
"""Select and record a VOT threshold from a validation CSV."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from dlr_vot import select_vot_threshold  # noqa: E402


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select a threshold from validation event scores.")
    parser.add_argument("input", type=Path, help="CSV containing scores and labels")
    parser.add_argument("--fpr-budget", type=float, required=True)
    parser.add_argument("--score-column", default="score")
    parser.add_argument("--label-column", default="label")
    parser.add_argument(
        "--output",
        type=Path,
        help="JSON output path; omit to print the result",
    )
    return parser.parse_args()


def _read_validation_rows(
    path: Path, *, score_column: str, label_column: str
) -> tuple[list[str], list[str]]:
    if not path.is_file():
        raise ValueError(f"input file does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("the input CSV header is missing")
        missing = [name for name in (score_column, label_column) if name not in reader.fieldnames]
        if missing:
            raise ValueError("column not found: " + ", ".join(repr(name) for name in missing))
        scores: list[str] = []
        labels: list[str] = []
        for row_number, row in enumerate(reader, start=2):
            score = row.get(score_column)
            label = row.get(label_column)
            if score is None or score == "":
                raise ValueError(f"row {row_number}: score is empty")
            if label is None or label == "":
                raise ValueError(f"row {row_number}: label is empty")
            scores.append(score)
            labels.append(label)
    if not scores:
        raise ValueError("the input CSV contains no validation rows")
    return scores, labels


def main() -> int:
    args = _arguments()
    try:
        scores, labels = _read_validation_rows(
            args.input,
            score_column=args.score_column,
            label_column=args.label_column,
        )
        result = select_vot_threshold(
            scores,
            labels,
            fpr_budget=args.fpr_budget,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    rendered = json.dumps(result.to_dict(), ensure_ascii=False, allow_nan=False, indent=2)
    if args.output is None:
        print(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
