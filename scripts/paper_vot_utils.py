"""Utilities required by the locked DLR–VOT reconstruction."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def f1_score(tp: int, fp: int, fn: int) -> float:
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def train_stds(dataset_path: Path, col_names: list[str]) -> np.ndarray:
    frame = pd.read_csv(dataset_path)
    train_rows = int(len(frame) * 0.7)
    missing = [column for column in col_names if column not in frame.columns]
    if missing:
        raise ValueError(f"dataset is missing columns from diagnostics: {missing}")
    values = frame.loc[: train_rows - 1, col_names].to_numpy(dtype=np.float64)
    return np.std(values, axis=0, ddof=0)


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -60.0, 60.0)))


def logit(probabilities: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    probabilities = np.clip(probabilities, eps, 1.0 - eps)
    return np.log(probabilities / (1.0 - probabilities))


def average_precision(
    score: np.ndarray,
    true: np.ndarray,
    true_active_threshold: float,
) -> float:
    labels = (true.reshape(-1) > true_active_threshold).astype(np.int64)
    flat_score = score.reshape(-1)
    if np.sum(labels) <= 0:
        return 0.0
    order = np.argsort(-flat_score, kind="mergesort")
    sorted_labels = labels[order]
    cumulative_true = np.cumsum(sorted_labels)
    ranks = np.arange(1, sorted_labels.size + 1)
    precision = cumulative_true / ranks
    return float(np.sum(precision[sorted_labels == 1]) / np.sum(sorted_labels))


def compute_metrics(
    true: np.ndarray,
    pred: np.ndarray,
    gate_open: np.ndarray,
    stds: np.ndarray,
    active_threshold: float,
    true_active_threshold: float,
) -> dict[str, Any]:
    true_active = true > true_active_threshold
    true_zero = ~true_active
    pred_active = pred > active_threshold
    pred_zero = ~pred_active

    tp = int(np.sum(true_active & pred_active))
    fp = int(np.sum(true_zero & pred_active))
    tn = int(np.sum(true_zero & pred_zero))
    fn = int(np.sum(true_active & pred_zero))
    abs_error = np.abs(pred - true)
    mae_per_channel = np.mean(abs_error, axis=(0, 1))
    rmse_per_channel = np.sqrt(np.mean((pred - true) ** 2, axis=(0, 1)))
    return {
        "MAE": float(np.mean(abs_error)),
        "RMSE": float(np.sqrt(np.mean((pred - true) ** 2))),
        "MAE_std": float(np.mean(mae_per_channel / stds)),
        "RMSE_std": float(np.mean(rmse_per_channel / stds)),
        "FPR": safe_div(fp, fp + tn),
        "FNR": safe_div(fn, fn + tp),
        "F1_active": f1_score(tp, fp, fn),
        "F1_zero": f1_score(tn, fn, fp),
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "FN_gate": int(np.sum(true_active & ~gate_open)),
        "open_count": int(np.sum(gate_open)),
    }


def percentile_from_reference(
    reference: np.ndarray,
    values: np.ndarray,
    per_channel: bool,
) -> np.ndarray:
    """Map values to empirical CDF scores fitted on the reference only."""

    reference = np.asarray(reference, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    output = np.zeros_like(values, dtype=np.float64)
    if per_channel:
        for channel in range(reference.shape[-1]):
            sorted_reference = np.sort(reference[..., channel].reshape(-1), kind="mergesort")
            ranks = np.searchsorted(
                sorted_reference,
                values[..., channel].reshape(-1),
                side="right",
            )
            output[..., channel] = (ranks / max(sorted_reference.size, 1)).reshape(
                values.shape[:-1]
            )
        return np.clip(output, 0.0, 1.0)

    sorted_reference = np.sort(reference.reshape(-1), kind="mergesort")
    ranks = np.searchsorted(sorted_reference, values.reshape(-1), side="right")
    return np.clip(
        (ranks / max(sorted_reference.size, 1)).reshape(values.shape),
        0.0,
        1.0,
    )


def load_diag(path: str) -> dict[str, Any]:
    diagnostics = np.load(path, allow_pickle=False)
    return {
        "true": np.asarray(diagnostics["true_y"], dtype=np.float64),
        "magnitude": np.asarray(diagnostics["raw_magnitude_pred"], dtype=np.float64),
        "dg_prob": np.asarray(diagnostics["gate_prob"], dtype=np.float64),
        "post": float(
            np.asarray(
                diagnostics.get("zero_threshold_effective", np.asarray(0.0))
            ).reshape(-1)[0]
        ),
        "col_names": [str(value) for value in np.asarray(diagnostics["col_names"]).tolist()],
    }


def load_ci_diag(path: Path) -> dict[str, Any]:
    return load_diag(str(path))


def load_npy(path: str, expected_shape: tuple[int, ...], name: str) -> np.ndarray:
    array_path = Path(path)
    if not array_path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")
    array = np.load(array_path).astype(np.float64)
    if array.shape != expected_shape:
        raise ValueError(f"{name} shape {array.shape} != expected {expected_shape}")
    return array


def validate_split_alignment(
    split_name: str,
    data_true: np.ndarray,
    diagnostics_true: np.ndarray,
    true_active_threshold: float,
) -> None:
    if data_true.shape != diagnostics_true.shape:
        raise ValueError(
            f"{split_name} shape mismatch data={data_true.shape}, diagnostics={diagnostics_true.shape}"
        )
    max_difference = float(np.max(np.abs(data_true - diagnostics_true)))
    label_mismatch = int(
        np.sum(
            (data_true > true_active_threshold)
            != (diagnostics_true > true_active_threshold)
        )
    )
    if max_difference > 1.0 or label_mismatch:
        raise ValueError(
            f"{split_name} labels do not align: "
            f"max_difference={max_difference}, label_mismatch={label_mismatch}"
        )


@dataclass
class SplitArrays:
    name: str
    x_hist: np.ndarray
    y_future: np.ndarray


def load_iron_csv(root_path: str, data_path: str) -> tuple[pd.DataFrame, list[str]]:
    frame = pd.read_csv(os.path.join(root_path, data_path))
    if "日期" in frame.columns:
        frame = frame.rename(columns={"日期": "date"})
    if "date" not in frame.columns:
        raise ValueError("CSV must contain 'date' or '日期'")
    value_columns = [column for column in frame.columns if column != "date"]
    if not value_columns:
        raise ValueError("no value columns found")
    return frame, value_columns


def split_borders(rows: int, context_length: int) -> dict[str, tuple[int, int]]:
    train_rows = int(rows * 0.7)
    test_rows = int(rows * 0.2)
    validation_rows = rows - train_rows - test_rows
    return {
        "train": (0, train_rows),
        "val": (train_rows - context_length, train_rows + validation_rows),
        "test": (rows - test_rows - context_length, rows),
    }


def make_split(
    frame: pd.DataFrame,
    value_columns: list[str],
    split: str,
    context_length: int,
    forecast_horizon: int,
) -> SplitArrays:
    start, end = split_borders(len(frame), context_length)[split]
    values = frame[value_columns].to_numpy(dtype=np.float64)[start:end]
    origins = len(values) - context_length - forecast_horizon + 1
    if origins <= 0:
        raise ValueError(f"{split} split is too short: len={len(values)}")
    history = np.stack(
        [values[index : index + context_length] for index in range(origins)],
        axis=0,
    )
    future = np.stack(
        [
            values[
                index + context_length : index + context_length + forecast_horizon
            ]
            for index in range(origins)
        ],
        axis=0,
    )
    return SplitArrays(split, x_hist=history, y_future=future)


def _safe_log1p(values: np.ndarray) -> np.ndarray:
    return np.log1p(np.maximum(np.asarray(values, dtype=np.float64), 0.0))


def _days_since_last_active(history: np.ndarray, threshold: float) -> float:
    active_indices = np.where(history > threshold)[0]
    if active_indices.size == 0:
        return float(len(history) + 1)
    return float(len(history) - 1 - int(active_indices[-1]))


def _tail_zero_run(history: np.ndarray, threshold: float) -> float:
    run = 0
    for active in (history > threshold)[::-1]:
        if active:
            break
        run += 1
    return float(run)


def window_base_features(history: np.ndarray, threshold: float) -> list[float]:
    features: list[float] = []
    history_log = _safe_log1p(history)
    for window in (7, 14, 30, 60, 96):
        values = history[-min(window, len(history)) :]
        values_log = _safe_log1p(values)
        active = values > threshold
        positive = values[values > threshold]
        features.extend(
            [
                float(np.mean(values_log)),
                float(np.std(values_log)),
                float(np.max(values_log)),
                float(np.min(values_log)),
                float(np.sum(active)),
                float(np.mean(active)),
                float(np.mean(_safe_log1p(positive))) if positive.size else 0.0,
                float(np.max(_safe_log1p(positive))) if positive.size else 0.0,
            ]
        )
    last_7 = np.mean(history_log[-7:])
    previous_7 = np.mean(history_log[-14:-7]) if len(history_log) >= 14 else last_7
    last_30 = np.mean(history_log[-30:])
    previous_30 = np.mean(history_log[-60:-30]) if len(history_log) >= 60 else last_30
    features.extend(
        [
            float(history_log[-1]),
            float(history_log[-1] - history_log[-2]) if len(history_log) >= 2 else 0.0,
            float(last_7 - previous_7),
            float(last_30 - previous_30),
            _days_since_last_active(history, threshold),
            _tail_zero_run(history, threshold),
        ]
    )
    return features


def build_features(
    split: SplitArrays,
    true_active_threshold: float,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Build the four training-only Ridge feature matrices and labels."""

    batch_size, _, channels = split.x_hist.shape
    forecast_horizon = split.y_future.shape[1]
    features_by_channel: list[list[list[float]]] = [[] for _ in range(channels)]
    labels_by_channel: list[list[int]] = [[] for _ in range(channels)]
    horizon_denominator = max(forecast_horizon - 1, 1)
    for sample in range(batch_size):
        for channel in range(channels):
            history = split.x_hist[sample, :, channel]
            base = window_base_features(history, true_active_threshold)
            aligned = history[-forecast_horizon:]
            aligned_log = _safe_log1p(aligned)
            aligned_active = aligned > true_active_threshold
            last_active = history[-1] > true_active_threshold
            for horizon in range(forecast_horizon):
                normalized_horizon = horizon / horizon_denominator
                features_by_channel[channel].append(
                    base
                    + [
                        normalized_horizon,
                        np.sin(2.0 * np.pi * normalized_horizon),
                        np.cos(2.0 * np.pi * normalized_horizon),
                        float(aligned_log[horizon]),
                        float(aligned_active[horizon]),
                        float(last_active),
                        float(horizon),
                    ]
                )
                labels_by_channel[channel].append(
                    int(
                        split.y_future[sample, horizon, channel]
                        > true_active_threshold
                    )
                )
    return (
        [np.asarray(values, dtype=np.float64) for values in features_by_channel],
        [np.asarray(values, dtype=np.int64) for values in labels_by_channel],
    )


def candidate_thresholds(score: np.ndarray, n_quantiles: int = 90) -> np.ndarray:
    flat = np.asarray(score, dtype=np.float64).reshape(-1)
    quantiles = np.unique(np.quantile(flat, np.linspace(0.0, 1.0, n_quantiles)))
    fixed = np.array(
        [flat.min() - 1e-9, flat.max() + 1e-9, 0.0, 0.5, 1.0],
        dtype=np.float64,
    )
    return np.unique(np.concatenate([quantiles, fixed]))
