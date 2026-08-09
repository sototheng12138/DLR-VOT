from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

from dlr_vot import validate_completed_daily_csv

ROOT = Path(__file__).resolve().parents[1]


def test_synthetic_generator_and_selector_cli(tmp_path: Path) -> None:
    daily_path = tmp_path / "daily.csv"
    event_path = tmp_path / "events.csv"
    result_path = tmp_path / "threshold.json"

    generated = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "generate_synthetic_data.py"),
            "--daily-output",
            str(daily_path),
            "--event-output",
            str(event_path),
            "--days",
            "12",
            "--seed",
            "9",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert generated.returncode == 0, generated.stderr
    assert validate_completed_daily_csv(daily_path).rows == 12

    selected = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "select_vot.py"),
            str(event_path),
            "--fpr-budget",
            "0.10",
            "--output",
            str(result_path),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert selected.returncode == 0, selected.stderr
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["validation_metrics"]["fpr"] <= 0.10

    with event_path.open("r", encoding="utf-8", newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 12 * 4
