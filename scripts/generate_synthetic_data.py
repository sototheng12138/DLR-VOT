#!/usr/bin/env python3
"""Generate a small synthetic daily table and validation event scores."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from datetime import date, timedelta
from pathlib import Path

CHANNELS = (
    ("commodity_1", 0.34, 90.0),
    ("commodity_2", 0.25, 65.0),
    ("commodity_3", 0.18, 45.0),
    ("commodity_4", 0.10, 30.0),
)


def generate_synthetic_data(
    daily_output: Path,
    event_output: Path,
    *,
    start: date,
    days: int,
    seed: int,
) -> dict[str, object]:
    """Write deterministic examples that contain no source data."""

    if days <= 0:
        raise ValueError("days must be positive")
    rng = random.Random(seed)
    daily_output.parent.mkdir(parents=True, exist_ok=True)
    event_output.parent.mkdir(parents=True, exist_ok=True)

    daily_rows: list[dict[str, str]] = []
    event_rows: list[dict[str, str]] = []
    for offset in range(days):
        current = start + timedelta(days=offset)
        weekly_factor = 1.0 + 0.12 * math.sin(2.0 * math.pi * offset / 7.0)
        daily_row = {"date": current.isoformat()}
        for channel, active_probability, scale in CHANNELS:
            active = rng.random() < active_probability
            if active:
                magnitude = rng.lognormvariate(math.log(scale), 0.35) * weekly_factor
                value = round(max(magnitude, 0.001), 3)
            else:
                value = 0.0
            daily_row[channel] = f"{value:.3f}"

            score_center = 0.72 if active else 0.22
            score = min(1.0, max(0.0, rng.gauss(score_center, 0.18)))
            event_rows.append(
                {
                    "date": current.isoformat(),
                    "commodity": channel,
                    "score": f"{score:.8f}",
                    "label": str(int(active)),
                }
            )
        daily_rows.append(daily_row)

    with daily_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["date", *(channel[0] for channel in CHANNELS)])
        writer.writeheader()
        writer.writerows(daily_rows)

    with event_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["date", "commodity", "score", "label"])
        writer.writeheader()
        writer.writerows(event_rows)

    return {
        "daily_output": str(daily_output),
        "event_output": str(event_output),
        "start_date": start.isoformat(),
        "days": days,
        "seed": seed,
    }


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily-output", type=Path, required=True)
    parser.add_argument("--event-output", type=Path, required=True)
    parser.add_argument("--start-date", default="2023-01-01")
    parser.add_argument("--days", type=int, default=1096)
    parser.add_argument("--seed", type=int, default=2025)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    try:
        start = date.fromisoformat(args.start_date)
        summary = generate_synthetic_data(
            args.daily_output,
            args.event_output,
            start=start,
            days=args.days,
            seed=args.seed,
        )
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from exc
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
