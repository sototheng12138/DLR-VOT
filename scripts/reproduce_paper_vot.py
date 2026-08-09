#!/usr/bin/env python3
"""Compare VOT threshold selectors without using test labels for selection.

The score construction, magnitude candidate, chronological split and threshold
candidate grid are held fixed.  Five operating rules are compared:

1. predeclared fixed threshold 0.5;
2. validation maximum active-class F1;
3. validation maximum Youden's J;
4. validation maximum balanced accuracy;
5. validation maximum active-class F1 subject to FPR <= 0.11.

The script deliberately has two phases.  Phase 1 uses training/validation data
only and persists ``locked_thresholds.json``.  Phase 2 reloads that lock file,
then constructs the test score and applies every frozen threshold once.  No
test score or test label is passed to a selector.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

from paper_vot_utils import (
    average_precision,
    build_features,
    candidate_thresholds,
    compute_metrics,
    load_ci_diag,
    load_diag,
    load_iron_csv,
    load_npy,
    logit,
    make_split,
    percentile_from_reference,
    sigmoid,
    train_stds,
    validate_split_alignment,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_DIR = (
    ROOT
    / "checkpoints/long_term_forecast_Iron_96_48_TimeLLM_custom_ftM_sl96_ll48_pl48_dm32_nh8_el2_dl1_df128_fc3_ebtimeF_"
    "Iron_Ore_Transport_Exp_0-iron_stage2_linear_direct_nt1000_dk32_state_event_v2_window_aux"
)
DEFAULT_MAG_DIR = (
    ROOT
    / "checkpoints/long_term_forecast_Iron_96_48_TimeLLM_custom_ftM_sl96_ll48_pl48_dm32_nh8_el2_dl1_df128_fc3_ebtimeF_"
    "Iron_Ore_Transport_Exp_0-iron_stage2_multivariate"
)
DEFAULT_OUT_DIR = ROOT / "diagnostics/vot_selector_comparison"

SCORE_NAME = (
    "logit_blend[base_dg_prob]+0.1*"
    "[student_distill_chronos_percentile_channel_trainfit_b1]"
)
MAGNITUDE_NAME = "woci_mlp"
SELECTOR_ORDER = [
    "fixed_0.5",
    "validation_max_f1",
    "validation_youden_j",
    "validation_balanced_accuracy",
    "validation_max_f1_fpr_le_0.11",
]
TIE_BREAK = (
    "maximize the named validation objective; if tied, minimize validation "
    "FPR; then minimize validation FNR; then choose the larger threshold"
)
LOCKED_TAU = 0.6734481160689028
LOCKED_EXPECTED = {
    "MAE_std": 0.6433283055110045,
    "RMSE_std": 1.0063065111217289,
    "FPR": 0.15196809153539342,
    "FNR": 0.5066439107643546,
    "F1_active": 0.5051803769816502,
    "F1_zero": 0.8415224083476592,
    "FP": 3772,
    "FN": 4156,
    "TP": 4047,
    "TN": 21049,
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_div(num: int | float, den: int | float) -> float:
    return float(num) / float(den) if den else 0.0


def f1(tp: int, fp: int, fn: int) -> float:
    return safe_div(2 * tp, 2 * tp + fp + fn)


def event_metrics(
    score: np.ndarray,
    true: np.ndarray,
    tau: float,
    true_active_threshold: float,
) -> dict[str, Any]:
    """Metrics of the VOT event decision itself: score > tau."""
    actual = np.asarray(true) > true_active_threshold
    pred = np.asarray(score) > tau
    tp = int(np.sum(actual & pred))
    fp = int(np.sum(~actual & pred))
    tn = int(np.sum(~actual & ~pred))
    fn = int(np.sum(actual & ~pred))
    tpr = safe_div(tp, tp + fn)
    tnr = safe_div(tn, tn + fp)
    fpr = safe_div(fp, fp + tn)
    fnr = safe_div(fn, fn + tp)
    return {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "FPR": fpr,
        "FNR": fnr,
        "TPR": tpr,
        "TNR": tnr,
        "precision_active": safe_div(tp, tp + fp),
        "recall_active": tpr,
        "F1_active": f1(tp, fp, fn),
        "F1_zero": f1(tn, fn, fp),
        "youden_j": tpr - fpr,
        "balanced_accuracy": 0.5 * (tpr + tnr),
        "open_count": int(np.sum(pred)),
    }


def final_forecast_metrics(
    score: np.ndarray,
    true: np.ndarray,
    magnitude: np.ndarray,
    tau: float,
    stds: np.ndarray,
    active_threshold: float,
    true_active_threshold: float,
    postprocess_threshold: float,
) -> dict[str, Any]:
    """Metrics after retain-or-zero fusion, kept separate from event metrics."""
    gate = np.asarray(score) > tau
    pred = np.asarray(magnitude) * gate
    if postprocess_threshold > 0:
        pred = np.where(np.abs(pred) < postprocess_threshold, 0.0, pred)
    return compute_metrics(
        np.asarray(true),
        pred,
        gate,
        stds,
        active_threshold,
        true_active_threshold,
    )


def fit_teacher_students(
    teacher_train: np.ndarray,
    x_train: list[np.ndarray],
) -> list[Any]:
    """Fit the same four per-channel Ridge students as the locked Proposed score."""
    models: list[Any] = []
    for c in range(teacher_train.shape[-1]):
        target = teacher_train[:, :, c].reshape(-1)
        if float(np.max(target) - np.min(target)) < 1e-9:
            models.append(("constant", float(np.mean(target))))
            continue
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            Ridge(alpha=1.0),
        )
        model.fit(x_train[c], target)
        models.append(model)
    return models


def predict_teacher_students(
    models: list[Any],
    x_split: list[np.ndarray],
    shape: tuple[int, int, int],
) -> np.ndarray:
    n_win, pred_len, n_ch = shape
    out = np.zeros(shape, dtype=np.float64)
    for c, model in enumerate(models):
        if isinstance(model, tuple) and model[0] == "constant":
            pred = np.full(x_split[c].shape[0], float(model[1]))
        else:
            pred = model.predict(x_split[c])
        out[:, :, c] = np.clip(pred, 0.0, 1.0).reshape(n_win, pred_len)
    return out


def blend_locked_score(base_prob: np.ndarray, student: np.ndarray) -> np.ndarray:
    return sigmoid(0.9 * logit(base_prob) + 0.1 * logit(student))


def choose_row(
    rows: list[dict[str, Any]],
    objective: str,
    fpr_budget: float | None = None,
) -> tuple[dict[str, Any], bool]:
    eligible = rows
    feasible = True
    if fpr_budget is not None:
        eligible = [r for r in rows if float(r["FPR"]) <= fpr_budget + 1e-12]
        feasible = bool(eligible)
        if not eligible:
            eligible = rows
    return max(
        eligible,
        key=lambda r: (
            float(r[objective]),
            -float(r["FPR"]),
            -float(r["FNR"]),
            float(r["tau"]),
        ),
    ), feasible


def prefixed(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{k}": v for k, v in metrics.items()}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def source_record(path: Path, stage: str) -> dict[str, Any]:
    return {
        "stage_first_accessed": stage,
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def git_state() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True
            ).strip()
        )
        return {"commit": commit, "worktree_dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "worktree_dirty": None}


def fmt(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def write_markdown(
    path: Path,
    rows: list[dict[str, Any]],
    lock_sha: str,
    val_nonpositive_rate: float,
    test_nonpositive_rate: float,
) -> None:
    lines = [
        "# VOT validation-selector comparison",
        "",
        "The underlying Proposed score and candidate magnitude are fixed. Thresholds are "
        "selected without test labels, persisted, reloaded, and then applied once to test.",
        "",
        f"- score: `{SCORE_NAME}`",
        f"- magnitude candidate: `{MAGNITUDE_NAME}`",
        "- event decision: `score > tau`",
        "- true active definition: recorded value `> 1` in the source units",
        "- common searched threshold grid: 90 validation-score quantiles plus fixed edge values",
        f"- deterministic tie-break: {TIE_BREAK}",
        f"- locked-threshold artifact SHA-256: `{lock_sha}`",
        f"- locked Proposed threshold: `{LOCKED_TAU}`",
        "",
        "## Event-decision results",
        "",
        "| Selector | tau | Val FPR | Val FNR | Val F1 active | Val Youden J | Val balanced acc. | Test FPR | Test FNR | Test F1 active | Test F1 zero |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['selector_label']} | {fmt(r['tau'], 6)} "
            f"| {fmt(r['val_event_FPR'])} | {fmt(r['val_event_FNR'])} "
            f"| {fmt(r['val_event_F1_active'])} | {fmt(r['val_event_youden_j'])} "
            f"| {fmt(r['val_event_balanced_accuracy'])} | {fmt(r['test_event_FPR'])} "
            f"| {fmt(r['test_event_FNR'])} | {fmt(r['test_event_F1_active'])} "
            f"| {fmt(r['test_event_F1_zero'])} |"
        )
    lines.extend(
        [
            "",
            "## Final retain-or-zero forecast results",
            "",
            "| Selector | MAE/std | RMSE/std | Output FPR | Output FNR | Output F1 active | Output F1 zero | FP | FN |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for r in rows:
        lines.append(
            f"| {r['selector_label']} | {fmt(r['test_forecast_MAE_std'])} "
            f"| {fmt(r['test_forecast_RMSE_std'])} | {fmt(r['test_forecast_FPR'])} "
            f"| {fmt(r['test_forecast_FNR'])} | {fmt(r['test_forecast_F1_active'])} "
            f"| {fmt(r['test_forecast_F1_zero'])} | {fmt(r['test_forecast_FP'])} "
            f"| {fmt(r['test_forecast_FN'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "- Youden's J and balanced accuracy are affine-equivalent for a binary threshold "
            "decision (`balanced accuracy = (Youden J + 1) / 2`). They must select the "
            "same operating point under the same grid and tie-break; they are not two "
            "independent gains.",
            "- Selector objectives use the event decision itself. Final forecast state metrics "
            "are reported separately because a gate can open while the retained candidate "
            "magnitude is non-positive.",
            f"- Non-positive candidate-magnitude share: validation `{val_nonpositive_rate:.6f}`, "
            f"test `{test_nonpositive_rate:.6f}`. This chronological shift explains why event "
            "and final-output state metrics should not be conflated.",
            "- The fixed 0.5 rule is predeclared rather than selected on validation. Its "
            "validation metrics are shown only for comparison.",
            "- The default event diagnostics are the locked `state_event_v2_window_aux` "
            "artifacts used for the reported result.",
            "- No selector receives test scores, test labels, test errors, or test confusion "
            "counts. The neural forecasting/event models are not retrained. The already-defined "
            "four training-only Ridge score students are deterministically reconstructed and "
            "persisted before test construction. No threshold is revised after test reporting.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root_path", default="./dataset/")
    parser.add_argument("--data_path", default="2023_2025_Iron_data.csv")
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--pred_len", type=int, default=48)
    parser.add_argument("--true_active_threshold", type=float, default=1.0)
    parser.add_argument("--active_threshold", type=float, default=0.0)
    parser.add_argument("--fixed_tau", type=float, default=0.5)
    parser.add_argument("--fpr_budget", type=float, default=0.11)
    parser.add_argument("--n_quantiles", type=int, default=90)
    parser.add_argument(
        "--base_val_diag",
        default=str(DEFAULT_BASE_DIR / "diagnostics_dg_val_noZT_full.npz"),
    )
    parser.add_argument(
        "--base_test_diag",
        default=str(DEFAULT_BASE_DIR / "diagnostics_dg_test_noZT_full.npz"),
    )
    parser.add_argument(
        "--magnitude_val_diag",
        default=str(DEFAULT_MAG_DIR / "diagnostics_ci_val_full.npz"),
    )
    parser.add_argument(
        "--magnitude_test_diag",
        default=str(DEFAULT_MAG_DIR / "diagnostics_ci_test_full.npz"),
    )
    parser.add_argument(
        "--train_chronos_pred",
        default="checkpoints/pred_chronos2_small_unclipped_train_orig.npy",
    )
    parser.add_argument(
        "--train_chronos_true",
        default="checkpoints/true_chronos2_small_unclipped_train_orig.npy",
    )
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    if args.n_quantiles != 90:
        raise ValueError(
            "The locked Proposed comparison requires n_quantiles=90 to preserve "
            "the pre-existing VOT candidate-grid protocol."
        )

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = (Path(args.root_path) / args.data_path).resolve()
    base_val_path = Path(args.base_val_diag).resolve()
    base_test_path = Path(args.base_test_diag).resolve()
    mag_val_path = Path(args.magnitude_val_diag).resolve()
    mag_test_path = Path(args.magnitude_test_diag).resolve()
    chronos_train_pred_path = Path(args.train_chronos_pred).resolve()
    chronos_train_true_path = Path(args.train_chronos_true).resolve()

    # ------------------------------------------------------------------
    # PHASE 1: fit fixed training-only student and select using validation.
    # No test split object, test diagnostic or test feature matrix exists here.
    # ------------------------------------------------------------------
    df, value_cols = load_iron_csv(args.root_path, args.data_path)
    train = make_split(df, value_cols, "train", args.seq_len, args.pred_len)
    val = make_split(df, value_cols, "val", args.seq_len, args.pred_len)

    base_val = load_diag(str(base_val_path))
    mag_val = load_ci_diag(mag_val_path)
    validate_split_alignment(
        "val_base", val.y_future, base_val["true"], args.true_active_threshold
    )
    validate_split_alignment(
        "val_magnitude", val.y_future, mag_val["true"], args.true_active_threshold
    )
    if list(base_val["col_names"]) != list(mag_val["col_names"]):
        raise RuntimeError("Validation channel names differ between score and magnitude")

    stds = train_stds(dataset_path, base_val["col_names"])
    chronos_train_pred = load_npy(
        str(chronos_train_pred_path), train.y_future.shape, "train Chronos pred"
    )
    chronos_train_true = load_npy(
        str(chronos_train_true_path), train.y_future.shape, "train Chronos true"
    )
    validate_split_alignment(
        "train_chronos",
        train.y_future,
        chronos_train_true,
        args.true_active_threshold,
    )
    x_train, _ = build_features(train, args.true_active_threshold)
    x_val, _ = build_features(val, args.true_active_threshold)
    chronos_train_z = chronos_train_pred / stds.reshape((1, 1, -1))
    teacher_train = percentile_from_reference(
        chronos_train_z, chronos_train_z, per_channel=True
    )
    student_models = fit_teacher_students(teacher_train, x_train)
    student_val = predict_teacher_students(student_models, x_val, val.y_future.shape)
    score_val = blend_locked_score(base_val["dg_prob"], student_val)

    threshold_rows: list[dict[str, Any]] = []
    for tau in candidate_thresholds(score_val, n_quantiles=args.n_quantiles):
        metrics = event_metrics(
            score_val, mag_val["true"], float(tau), args.true_active_threshold
        )
        threshold_rows.append({"tau": float(tau), **metrics})

    fixed_val = event_metrics(
        score_val, mag_val["true"], args.fixed_tau, args.true_active_threshold
    )
    max_f1, _ = choose_row(threshold_rows, "F1_active")
    youden, _ = choose_row(threshold_rows, "youden_j")
    balanced, _ = choose_row(threshold_rows, "balanced_accuracy")
    constrained, constrained_feasible = choose_row(
        threshold_rows, "F1_active", args.fpr_budget
    )

    selector_specs = [
        {
            "selector": "fixed_0.5",
            "selector_label": "Fixed 0.5",
            "selection_split": "predeclared",
            "objective": "none",
            "constraint": "none",
            "validation_feasible": "not_applicable",
            "tau": float(args.fixed_tau),
            "val": fixed_val,
        },
        {
            "selector": "validation_max_f1",
            "selector_label": "Validation max-F1",
            "selection_split": "validation",
            "objective": "max_F1_active",
            "constraint": "none",
            "validation_feasible": "yes",
            "tau": float(max_f1["tau"]),
            "val": {k: v for k, v in max_f1.items() if k != "tau"},
        },
        {
            "selector": "validation_youden_j",
            "selector_label": "Validation Youden J",
            "selection_split": "validation",
            "objective": "max_youden_j",
            "constraint": "none",
            "validation_feasible": "yes",
            "tau": float(youden["tau"]),
            "val": {k: v for k, v in youden.items() if k != "tau"},
        },
        {
            "selector": "validation_balanced_accuracy",
            "selector_label": "Validation balanced accuracy",
            "selection_split": "validation",
            "objective": "max_balanced_accuracy",
            "constraint": "none",
            "validation_feasible": "yes",
            "tau": float(balanced["tau"]),
            "val": {k: v for k, v in balanced.items() if k != "tau"},
        },
        {
            "selector": "validation_max_f1_fpr_le_0.11",
            "selector_label": "Proposed VOT (max-F1, FPR<=0.11)",
            "selection_split": "validation",
            "objective": "max_F1_active",
            "constraint": f"FPR<={args.fpr_budget:g}",
            "validation_feasible": "yes" if constrained_feasible else "no",
            "tau": float(constrained["tau"]),
            "val": {k: v for k, v in constrained.items() if k != "tau"},
        },
    ]
    if not constrained_feasible:
        raise RuntimeError("No validation threshold satisfies the declared FPR budget")

    validation_rows: list[dict[str, Any]] = []
    for spec in selector_specs:
        val_forecast = final_forecast_metrics(
            score_val,
            mag_val["true"],
            mag_val["magnitude"],
            float(spec["tau"]),
            stds,
            args.active_threshold,
            args.true_active_threshold,
            float(mag_val.get("post", 0.0)),
        )
        validation_rows.append(
            {
                "selector": spec["selector"],
                "selector_label": spec["selector_label"],
                "selection_split": spec["selection_split"],
                "objective": spec["objective"],
                "constraint": spec["constraint"],
                "validation_feasible": spec["validation_feasible"],
                "tau": spec["tau"],
                **prefixed("val_event", spec["val"]),
                **prefixed("val_forecast", val_forecast),
                "val_AP": average_precision(
                    score_val, mag_val["true"], args.true_active_threshold
                ),
            }
        )
    write_csv(out_dir / "validation_selection.csv", validation_rows)
    write_csv(out_dir / "validation_threshold_candidates.csv", threshold_rows)

    model_path = out_dir / "ridge_students.joblib"
    joblib.dump(student_models, model_path)
    lock = {
        "schema_version": 1,
        "created_utc_before_test_evaluation": datetime.now(timezone.utc).isoformat(),
        "score": SCORE_NAME,
        "magnitude": MAGNITUDE_NAME,
        "gate_comparator": "score > tau",
        "true_active_rule": f"raw_y > {args.true_active_threshold:g}",
        "candidate_grid": {
            "source": "validation_score_only",
            "n_quantiles": args.n_quantiles,
            "quantile_range": [0.0, 1.0],
            "extras": [
                "validation_min - 1e-9",
                "validation_max + 1e-9",
                0.0,
                0.5,
                1.0,
            ],
            "candidate_count": len(threshold_rows),
        },
        "tie_break": TIE_BREAK,
        "ridge_students": {
            "path": str(model_path),
            "sha256": sha256_file(model_path),
        },
        "validation_phase_inputs": {
            str(dataset_path): sha256_file(dataset_path),
            str(base_val_path): sha256_file(base_val_path),
            str(mag_val_path): sha256_file(mag_val_path),
            str(chronos_train_pred_path): sha256_file(chronos_train_pred_path),
            str(chronos_train_true_path): sha256_file(chronos_train_true_path),
        },
        "selectors": [
            {
                k: spec[k]
                for k in (
                    "selector",
                    "selector_label",
                    "selection_split",
                    "objective",
                    "constraint",
                    "validation_feasible",
                    "tau",
                )
            }
            for spec in selector_specs
        ],
        "test_data_used_for_selection": False,
        "test_score_used_for_selection": False,
        "test_metric_used_for_selection": False,
    }
    lock_path = out_dir / "locked_thresholds.json"
    lock_path.write_text(
        json.dumps(lock, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lock_sha_before_test = sha256_file(lock_path)

    # ------------------------------------------------------------------
    # PHASE 2: reload the persisted lock; construct test score; apply once.
    # ------------------------------------------------------------------
    locked = json.loads(lock_path.read_text(encoding="utf-8"))
    if sha256_file(lock_path) != lock_sha_before_test:
        raise RuntimeError("Threshold lock changed before test application")
    if locked.get("test_data_used_for_selection") is not False:
        raise RuntimeError("Lock does not certify validation-only selection")
    if sha256_file(model_path) != locked["ridge_students"]["sha256"]:
        raise RuntimeError("Locked Ridge score students changed before test")
    locked_specs = locked["selectors"]
    if [x["selector"] for x in locked_specs] != SELECTOR_ORDER:
        raise RuntimeError("Unexpected selector order in lock file")

    test = make_split(df, value_cols, "test", args.seq_len, args.pred_len)
    base_test = load_diag(str(base_test_path))
    mag_test = load_ci_diag(mag_test_path)
    validate_split_alignment(
        "test_base", test.y_future, base_test["true"], args.true_active_threshold
    )
    validate_split_alignment(
        "test_magnitude", test.y_future, mag_test["true"], args.true_active_threshold
    )
    if list(base_test["col_names"]) != list(mag_test["col_names"]):
        raise RuntimeError("Test channel names differ between score and magnitude")
    x_test, _ = build_features(test, args.true_active_threshold)
    locked_models = joblib.load(model_path)
    student_test = predict_teacher_students(
        locked_models, x_test, test.y_future.shape
    )
    score_test = blend_locked_score(base_test["dg_prob"], student_test)

    final_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    validation_by_selector = {r["selector"]: r for r in validation_rows}
    for spec in locked_specs:
        selector = spec["selector"]
        tau = float(spec["tau"])
        test_event = event_metrics(
            score_test, mag_test["true"], tau, args.true_active_threshold
        )
        test_forecast = final_forecast_metrics(
            score_test,
            mag_test["true"],
            mag_test["magnitude"],
            tau,
            stds,
            args.active_threshold,
            args.true_active_threshold,
            float(mag_test.get("post", 0.0)),
        )
        test_row = {
            "selector": selector,
            "selector_label": spec["selector_label"],
            "tau": tau,
            **prefixed("test_event", test_event),
            **prefixed("test_forecast", test_forecast),
            "test_AP": average_precision(
                score_test, mag_test["true"], args.true_active_threshold
            ),
        }
        test_rows.append(test_row)
        final_rows.append({**validation_by_selector[selector], **test_row})

    uses_default_artifacts = (
        base_val_path
        == (DEFAULT_BASE_DIR / "diagnostics_dg_val_noZT_full.npz").resolve()
        and base_test_path
        == (DEFAULT_BASE_DIR / "diagnostics_dg_test_noZT_full.npz").resolve()
    )
    locked_result_check: dict[str, Any] = {
        "uses_default_artifacts": uses_default_artifacts,
        "locked_tau": LOCKED_TAU,
    }
    if uses_default_artifacts:
        proposed = next(
            r
            for r in final_rows
            if r["selector"] == "validation_max_f1_fpr_le_0.11"
        )
        if abs(float(proposed["tau"]) - LOCKED_TAU) > 5e-12:
            raise RuntimeError(
                "The validation selector did not recover the expected locked threshold"
            )
        for key, expected in LOCKED_EXPECTED.items():
            actual = proposed[f"test_forecast_{key}"]
            tolerance = 0.0 if isinstance(expected, int) else 5e-12
            if abs(float(actual) - float(expected)) > tolerance:
                raise RuntimeError(
                    f"Locked result mismatch for {key}: {actual} != {expected}"
                )
        locked_result_check.update(
            {
                "expected_metrics_reproduced": True,
                "expected_test_metrics": LOCKED_EXPECTED,
            }
        )

    if sha256_file(lock_path) != lock_sha_before_test:
        raise RuntimeError("Threshold lock changed during test application")
    write_csv(out_dir / "test_once_results.csv", test_rows)
    write_csv(out_dir / "vot_selector_comparison.csv", final_rows)

    val_nonpositive_rate = float(np.mean(mag_val["magnitude"] <= 0.0))
    test_nonpositive_rate = float(np.mean(mag_test["magnitude"] <= 0.0))
    write_markdown(
        out_dir / "README.md",
        final_rows,
        lock_sha_before_test,
        val_nonpositive_rate,
        test_nonpositive_rate,
    )

    sources = [
        source_record(dataset_path, "validation_selection"),
        source_record(base_val_path, "validation_selection"),
        source_record(mag_val_path, "validation_selection"),
        source_record(chronos_train_pred_path, "validation_selection"),
        source_record(chronos_train_true_path, "validation_selection"),
        source_record(base_test_path, "post_lock_test_once"),
        source_record(mag_test_path, "post_lock_test_once"),
    ]
    provenance = {
        "schema_version": 1,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "script": source_record(Path(__file__).resolve(), "code"),
        "git": git_state(),
        "protocol": {
            "chronological_split": True,
            "train_used_for_student_fit_only": True,
            "validation_used_for_all_nonfixed_threshold_selection": True,
            "fixed_0.5_predeclared": True,
            "thresholds_persisted_before_test_construction": True,
            "locked_thresholds_reloaded_before_test": True,
            "test_applied_once_per_locked_policy": True,
            "test_data_used_for_selection": False,
            "test_score_used_for_selection": False,
            "test_metric_used_for_selection": False,
            "neural_forecaster_or_event_model_retraining": False,
            "training_only_ridge_score_student_reconstructed": True,
            "post_test_threshold_revision": False,
        },
        "parameters": {
            "seq_len": args.seq_len,
            "pred_len": args.pred_len,
            "score": SCORE_NAME,
            "magnitude": MAGNITUDE_NAME,
            "student": "per-channel Ridge(alpha=1.0) fit on training split",
            "blend": "sigmoid(0.9*logit(base_dg_prob)+0.1*logit(student))",
            "fixed_tau": args.fixed_tau,
            "fpr_budget": args.fpr_budget,
            "true_active_threshold_source_units": args.true_active_threshold,
            "active_output_threshold": args.active_threshold,
            "candidate_quantiles": args.n_quantiles,
            "tie_break": TIE_BREAK,
        },
        "array_summary": {
            "validation_shape": list(score_val.shape),
            "test_shape": list(score_test.shape),
            "validation_points": int(score_val.size),
            "test_points": int(score_test.size),
            "validation_active_points": int(
                np.sum(mag_val["true"] > args.true_active_threshold)
            ),
            "test_active_points": int(
                np.sum(mag_test["true"] > args.true_active_threshold)
            ),
            "validation_candidate_magnitude_nonpositive_rate": val_nonpositive_rate,
            "test_candidate_magnitude_nonpositive_rate": test_nonpositive_rate,
        },
        "locked_result_check": locked_result_check,
        "artifacts": {
            "locked_thresholds": {
                "path": str(lock_path),
                "sha256_before_and_after_test": lock_sha_before_test,
            },
            "ridge_students": {
                "path": str(model_path),
                "sha256": sha256_file(model_path),
            },
            "validation_selection": str(out_dir / "validation_selection.csv"),
            "validation_candidates": str(
                out_dir / "validation_threshold_candidates.csv"
            ),
            "test_once_results": str(out_dir / "test_once_results.csv"),
            "comparison": str(out_dir / "vot_selector_comparison.csv"),
            "summary": str(out_dir / "README.md"),
        },
        "sources": sources,
        "scientific_note": (
            "Youden J and balanced accuracy are affine-equivalent for binary "
            "classification and therefore select the same row under a common "
            "candidate grid and deterministic tie-break."
        ),
    }
    (out_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"saved {out_dir / 'vot_selector_comparison.csv'}")
    print(f"saved {out_dir / 'README.md'}")
    print(f"saved {out_dir / 'provenance.json'}")
    print(f"locked thresholds sha256={lock_sha_before_test}")
    for row in final_rows:
        print(
            row["selector"],
            f"tau={float(row['tau']):.9f}",
            f"val_FPR={float(row['val_event_FPR']):.6f}",
            f"test_FPR={float(row['test_event_FPR']):.6f}",
            f"test_F1a={float(row['test_event_F1_active']):.6f}",
            f"test_MAEstd={float(row['test_forecast_MAE_std']):.6f}",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
