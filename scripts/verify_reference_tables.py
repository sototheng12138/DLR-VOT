#!/usr/bin/env python3
"""Check the released aggregate tables and principal locked values."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "results" / "tables"
EXPECTED = {
    "Table1_literature_positioning.csv",
    "Table2_TimeLLM_DLRVOT_comparison.csv",
    "Table3-19_captions.csv",
    "Table3_calendar_zero_and_chronological_drift.csv",
    "Table4_locked_implementation_and_hyperparameters.csv",
    "Table5_threshold_free_event_ranking.csv",
    "Table6_validation_only_probability_calibration.csv",
    "Table7_validation_only_threshold_selector_comparison.csv",
    "Table8_locked_selector_cost_scenarios.csv",
    "Table9_dlr_structural_controls_and_component_ablation.csv",
    "Table10_event_score_fusion_sensitivity.csv",
    "Table11_full_four_channel_final_magnitude_comparison.csv",
    "Table12_full_pooled_native_state_comparison.csv",
    "Table13_fair_state_aware_operating_comparison.csv",
    "Table14_fixed_policy_break_even_comparison.csv",
    "Table15_paired_block_bootstrap_vs_compact.csv",
    "Table16_llm_capacity_complete_system_compute_boundary.csv",
    "Table17_gold_concentrate_alignment_and_sparsity.csv",
    "Table18_gold_concentrate_severe_sparsity_comparison.csv",
    "Table19_gold_auxiliary_training_regime_sensitivity.csv",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows(name: str) -> list[dict[str, str]]:
    with (TABLE_DIR / name).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def find_row(table: list[dict[str, str]], column: str, value: str) -> dict[str, str]:
    for row in table:
        if row.get(column) == value:
            return row
    raise AssertionError(f"row {column}={value!r} was not found")


def check_manifest() -> None:
    manifest = TABLE_DIR / "SHA256SUMS"
    if not manifest.exists():
        raise AssertionError("results/tables/SHA256SUMS is missing")
    recorded: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        recorded[name.lstrip("*")] = digest
    if set(recorded) != EXPECTED:
        raise AssertionError("the checksum manifest does not match the expected table set")
    for name, digest in recorded.items():
        actual = sha256(TABLE_DIR / name)
        if actual != digest:
            raise AssertionError(f"checksum mismatch: {name}")


def check_locked_values() -> None:
    selector = find_row(
        rows("Table7_validation_only_threshold_selector_comparison.csv"),
        "Selector",
        "VOT (ρ=0.11)",
    )
    assert selector["τ"] == "0.6734"
    assert selector["Test FPR"] == "0.1520"
    assert selector["Test FNR"] == "0.5066"
    assert selector["Test F1_active"] == "0.5052"
    assert selector["Test F1_zero"] == "0.8415"
    assert selector["FP"] == "3772"
    assert selector["FN"] == "4156"

    magnitude = find_row(
        rows("Table11_full_four_channel_final_magnitude_comparison.csv"),
        "Method",
        "Proposed",
    )
    assert magnitude["Avg."] == "0.6433 / 1.0063"

    state = find_row(
        rows("Table12_full_pooled_native_state_comparison.csv"),
        "Method",
        "Proposed",
    )
    assert state == {
        "Method": "Proposed",
        "FPR": "0.1520",
        "FNR": "0.5066",
        "F1_active": "0.5052",
        "F1_zero": "0.8415",
        "FP": "3772",
        "FN": "4156",
    }

    compute = find_row(
        rows("Table16_llm_capacity_complete_system_compute_boundary.csv"),
        "System",
        "Proposed-7B",
    )
    assert compute["Avg. MAE/std"] == "0.6433"
    assert compute["Avg. RMSE/std"] == "1.0063"
    assert compute["Latency (ms)"] == "143.51 ± 0.87"


def main() -> None:
    found = {path.name for path in TABLE_DIR.glob("*.csv")}
    if found != EXPECTED:
        missing = sorted(EXPECTED - found)
        extra = sorted(found - EXPECTED)
        raise AssertionError(f"table set differs; missing={missing}, extra={extra}")
    check_manifest()
    check_locked_values()
    print(f"Reference tables verified: {len(EXPECTED)} files")


if __name__ == "__main__":
    main()
