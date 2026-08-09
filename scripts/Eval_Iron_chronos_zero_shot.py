import argparse
import os
import sys
from typing import List, Tuple

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
LOCAL_CHRONOS_SRC = os.path.join(PROJECT_ROOT, "chronos-forecasting", "src")
if os.path.isdir(LOCAL_CHRONOS_SRC) and LOCAL_CHRONOS_SRC not in sys.path:
    sys.path.insert(0, LOCAL_CHRONOS_SRC)


def _load_col_names(root_path: str, data_path: str, n_channels: int) -> List[str]:
    """
    Match the evaluation scripts: use csv header (skip first date column).
    Fallback to generic channel names if header parsing fails.
    """
    try:
        import pandas as pd

        df = pd.read_csv(os.path.join(root_path, data_path), nrows=0)
        cols = list(df.columns[1:])
        if len(cols) != n_channels:
            cols = [f"channel_{i + 1}" for i in range(n_channels)]
        return cols
    except Exception:
        return [f"channel_{i + 1}" for i in range(n_channels)]


def _compute_mae_rmse(pred: np.ndarray, true: np.ndarray) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    pred/true: (tot_len, pred_len, n_channels)
    """
    mae_all = float(np.mean(np.abs(pred - true)))
    rmse_all = float(np.sqrt(np.mean((pred - true) ** 2)))
    mae_per = np.mean(np.abs(pred - true), axis=(0, 1))
    rmse_per = np.sqrt(np.mean((pred - true) ** 2, axis=(0, 1)))
    return mae_all, rmse_all, mae_per, rmse_per


def main() -> None:
    ap = argparse.ArgumentParser(description="Chronos zero-shot baseline evaluation (Iron dataset).")
    ap.add_argument("--root_path", type=str, default="./dataset", help="dataset root directory")
    ap.add_argument("--data_path", type=str, default="2023_2025_Iron_data.csv", help="csv file under root_path")
    ap.add_argument("--seq_len", type=int, default=96)
    ap.add_argument("--label_len", type=int, default=48)
    ap.add_argument("--pred_len", type=int, default=48)
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="dataset split to evaluate")
    ap.add_argument("--features", type=str, default="M", choices=["S", "M", "MS"])
    ap.add_argument("--percent", type=int, default=100, help="train-only percent (affects scaler fit window)")

    ap.add_argument("--model_id", type=str, default="autogluon/chronos-2-small", help="Chronos-2 HF model id")
    ap.add_argument("--batch_size", type=int, default=8, help="inference batch size")
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--device_map", type=str, default="cuda", help="device_map for from_pretrained, e.g. cuda/cpu")
    ap.add_argument("--torch_dtype", type=str, default="float32", help="float32/bfloat16/float16")
    ap.add_argument("--dry_run", action="store_true", help="skip model download/prediction; only validate dataset size")
    ap.add_argument("--out_dir", type=str, default="", help="optional output dir for eval_result.txt")
    ap.add_argument("--save_plot", action="store_true", help="save True vs Pred plot for the first test window")
    ap.add_argument("--plot_space", type=str, default="scaled", choices=["scaled", "original"], help="plot in standardized or original space")
    ap.add_argument("--save_pred_true", action="store_true", help="save prediction and target arrays for downstream analysis")
    ap.add_argument("--pred_tag", type=str, default="chronos2_small_zero_shot", help="suffix tag for saved pred/true files")
    ap.add_argument("--local_files_only", action="store_true", help="force using local cached model files (offline)")
    ap.add_argument(
        "--save_pred_true_space",
        type=str,
        default="both",
        choices=["scaled", "original", "both"],
        help="which space to save for pred/true plots",
    )
    args = ap.parse_args()

    # Dataset: reuse the exact Iron split/scaler logic from your project.
    # IMPORTANT: timeenc=1 to avoid a known pandas apply bug in timeenc=0 path.
    from data_provider_pretrain.data_loader import Dataset_Custom_Iron
    from torch.utils.data import DataLoader

    ds = Dataset_Custom_Iron(
        root_path=args.root_path,
        flag=args.split,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target="铁矿石",
        scale=True,
        timeenc=1,
        freq="d",
        percent=args.percent,
        pretrain=True,
        return_multivariate=False,
        data_path=args.data_path,
    )

    # ds.__len__ equals (tot_len * enc_in) when return_multivariate=False.
    n_samples_total = len(ds)
    tot_len = ds.tot_len
    enc_in = ds.enc_in
    expected = tot_len * enc_in
    print(f"[Chronos eval] split={args.split}, tot_len={tot_len}, enc_in={enc_in}, samples_total={n_samples_total}, expected={expected}")
    if n_samples_total != expected:
        raise RuntimeError("Dataset sample count mismatch: expected tot_len*enc_in.")

    if args.dry_run:
        col_names = _load_col_names(args.root_path, args.data_path, enc_in)
        print("[Chronos eval] dry_run done. No truncation/subsampling applied.")
        print("[Chronos eval] Expected output shape after aggregation: "
              f"({tot_len}, {args.pred_len}, {enc_in})")
        print("[Chronos eval] Channels:", ", ".join(col_names))
        return

    # Chronos model
    import torch
    from chronos import Chronos2Pipeline

    if args.local_files_only:
        # Avoid any network calls during model/config resolution.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    torch_dtype = getattr(torch, args.torch_dtype, torch.float32)
    pipeline = Chronos2Pipeline.from_pretrained(
        args.model_id,
        device_map=args.device_map,
        torch_dtype=torch_dtype,
        local_files_only=args.local_files_only,
    )

    # Log context/prediction length to be explicit about any internal truncation.
    try:
        model_ctx = pipeline.model_context_length
        print(f"[Chronos eval] model_context_length={model_ctx} (input seq_len={args.seq_len})")
        if model_ctx is not None and args.seq_len > int(model_ctx):
            print("[Chronos eval] NOTE: Chronos may truncate history to model_context_length.")
    except Exception:
        pass

    try:
        model_pred = pipeline.model_prediction_length
        print(f"[Chronos eval] model_prediction_length={model_pred} (eval pred_len={args.pred_len})")
        if model_pred is not None and args.pred_len > int(model_pred):
            print("[Chronos eval] NOTE: prediction_length may be generated in multiple internal steps.")
    except Exception:
        pass

    # Inference
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,  # ensure no test truncation
    )

    all_pred = []
    all_true = []

    with torch.no_grad():
        for batch in loader:
            # return_multivariate=False => (seq_x, seq_y, seq_x_mark, seq_y_mark, feat_id)
            if len(batch) == 5:
                batch_x, batch_y, _, _, _ = batch
            else:
                raise RuntimeError("Unexpected batch format from Dataset_Custom_Iron.")

            # batch_x: (B, seq_len, 1), batch_y: (B, label_len+pred_len, 1)
            context = batch_x[..., 0].to(torch.float32)  # (B, seq_len)
            true_future = batch_y[:, -args.pred_len :, 0].to(torch.float32)  # (B, pred_len)

            # Chronos-2 is deterministic here. The point forecast is the median
            # returned as `mean` from predict_quantiles.
            #
            # Input format: for Chronos-2, using a list of 1D tensors is the most explicit for
            # univariate forecasting (each element is one time series).
            context_list = [context[i].contiguous() for i in range(context.shape[0])]
            _, mean = pipeline.predict_quantiles(
                inputs=context_list,
                prediction_length=args.pred_len,
                quantile_levels=[0.5],
                limit_prediction_length=False,
            )  # mean: list[Tensor], each tensor has shape (n_variates, pred_len)

            # Convert mean list -> (B, pred_len)
            if isinstance(mean, list):
                if len(mean) != context.shape[0]:
                    # Fallback for unexpected return shape.
                    pred_future = mean[0].squeeze(0).unsqueeze(0).repeat(context.shape[0], 1).to(torch.float32)
                else:
                    pred_future = torch.stack([m.squeeze(0) for m in mean], dim=0).to(torch.float32)
            else:
                pred_future = mean.to(torch.float32)

            all_pred.append(pred_future.cpu().numpy()[..., None])  # (B, pred_len, 1)
            all_true.append(true_future.cpu().numpy()[..., None])  # (B, pred_len, 1)

    pred = np.concatenate(all_pred, axis=0)
    true = np.concatenate(all_true, axis=0)

    # Aggregate across channels to match your project's evaluation layout.
    # Current layout: pred/true are (tot_len*enc_in, pred_len, 1) ordered by (channel, window).
    n_full = tot_len * enc_in
    if enc_in > 1 and pred.shape[-1] == 1:
        pred = pred[:n_full].reshape(enc_in, tot_len, args.pred_len, 1).transpose(1, 2, 0, 3).squeeze(-1)
        true = true[:n_full].reshape(enc_in, tot_len, args.pred_len, 1).transpose(1, 2, 0, 3).squeeze(-1)

    # pred/true: (tot_len, pred_len, enc_in)
    if pred.shape != true.shape or pred.ndim != 3:
        raise RuntimeError(f"Unexpected aggregated shapes: pred={pred.shape}, true={true.shape}")

    col_names = _load_col_names(args.root_path, args.data_path, enc_in)

    # 1) Standardized space metrics (main comparison)
    mae_all_scaled, rmse_all_scaled, mae_per_scaled, rmse_per_scaled = _compute_mae_rmse(pred, true)
    print("-" * 60)
    print("[Chronos eval] 【不逆变换，标准化空间】")
    print(f"{args.split} 整体  MAE = {mae_all_scaled:.6f}  RMSE = {rmse_all_scaled:.6f}")
    print("各指标:")
    for i in range(enc_in):
        print(f"  {col_names[i]}  MAE = {mae_per_scaled[i]:.6f}  RMSE = {rmse_per_scaled[i]:.6f}")

    # 2) Inverse transform to original scale (optional, for completeness)
    pred_flat = pred.reshape(-1, enc_in)
    true_flat = true.reshape(-1, enc_in)
    pred_inv = ds.inverse_transform(pred_flat).reshape(pred.shape)
    true_inv = ds.inverse_transform(true_flat).reshape(true.shape)
    mae_all, rmse_all, mae_per, rmse_per = _compute_mae_rmse(pred_inv, true_inv)

    print("-" * 60)
    print("[Chronos eval] 【逆变换到原始量纲】")
    print(f"{args.split} 整体  MAE = {mae_all:.6f}  RMSE = {rmse_all:.6f}")
    print("各指标:")
    for i in range(enc_in):
        print(f"  {col_names[i]}  MAE = {mae_per[i]:.6f}  RMSE = {rmse_per[i]:.6f}")

    # 3) Save (optional)
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        out_path = os.path.join(args.out_dir, "eval_result_chronos_zero_shot.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("[Chronos eval] Chronos zero-shot baseline\n")
            f.write(f"model_id={args.model_id}\n")
            f.write(f"split={args.split}\n")
            f.write(f"seq_len={args.seq_len}, pred_len={args.pred_len}, tot_len={tot_len}, enc_in={enc_in}\n")
            f.write("\n[Unscaled / standardized space]\n")
            f.write(f"MAE={mae_all_scaled:.6f}, RMSE={rmse_all_scaled:.6f}\n")
            for i in range(enc_in):
                f.write(f"{col_names[i]}: MAE={mae_per_scaled[i]:.6f}, RMSE={rmse_per_scaled[i]:.6f}\n")
            f.write("\n[Inverse transformed / original space]\n")
            f.write(f"MAE={mae_all:.6f}, RMSE={rmse_all:.6f}\n")
            for i in range(enc_in):
                f.write(f"{col_names[i]}: MAE={mae_per[i]:.6f}, RMSE={rmse_per[i]:.6f}\n")
        print("Saved:", out_path)

    # Optional plot: use raw predictions (no aux gating / no zero-threshold / no clipping).
    # This intentionally avoids the `0.5`-based aux gate and any "spike" post-processing you use in Time-LLM eval.
    if args.save_plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plot_pred = pred_inv if args.plot_space == "original" else pred
        plot_true = true_inv if args.plot_space == "original" else true

        first_pred = plot_pred[0]  # (pred_len, enc_in)
        first_true = plot_true[0]
        pred_len = int(args.pred_len)

        n_plot = min(4, enc_in)
        fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=120, sharex=True)
        axes = axes.flatten()

        t = np.arange(pred_len)
        for c in range(n_plot):
            ax = axes[c]
            ax.plot(t, first_true[:, c], color="#1f77b4", linewidth=2.0, linestyle="-", label="True", alpha=0.95)
            ax.plot(t, first_pred[:, c], color="#d62728", linewidth=1.8, linestyle="--", label="Chronos", alpha=0.9)
            title = col_names[c] if c < len(col_names) else f"Ch{c}"
            ax.set_title(title, fontsize=12, fontweight="medium")
            ax.grid(True, alpha=0.25)
            ax.set_xlabel("Forecast step")
            y_label = "Value (original)" if args.plot_space == "original" else "Scaled value (std space)"
            ax.set_ylabel(y_label)

        for k in range(n_plot, 4):
            axes[k].axis("off")

        space_tag = "orig" if args.plot_space == "original" else "scaled"
        out_base_dir = args.out_dir if args.out_dir else os.path.dirname(__file__)
        os.makedirs(out_base_dir, exist_ok=True)
        plot_png = os.path.join(out_base_dir, f"chronos_zero_shot_pred_true_window0_{space_tag}.png")
        plot_pdf = os.path.join(out_base_dir, f"chronos_zero_shot_pred_true_window0_{space_tag}.pdf")

        fig.legend(loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.01))
        fig.tight_layout()
        fig.savefig(plot_png, bbox_inches="tight", facecolor="white")
        fig.savefig(plot_pdf, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print("Saved plot:", plot_png, "and", plot_pdf)

    # Save pred/true for the existing "full test" plotting script.
    # This also keeps the "no 0.5clip / no spike clip" requirement, because pred/true are the raw Chronos outputs.
    if args.save_pred_true:
        if not args.out_dir:
            # default to checkpoints (same as your other eval outputs)
            args.out_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
            args.out_dir = os.path.abspath(args.out_dir)

        os.makedirs(args.out_dir, exist_ok=True)
        tag_suffix = f"_{args.pred_tag}" if args.pred_tag else ""

        def _save_one(space: str, pred_arr: np.ndarray, true_arr: np.ndarray) -> None:
            if space == "scaled":
                space_tag = "scaled"
                suffix = tag_suffix
            else:
                space_tag = "original"
                suffix = f"{tag_suffix}_orig" if tag_suffix else "_orig"

            pred_path = os.path.join(args.out_dir, f"pred{suffix}.npy")
            true_path = os.path.join(args.out_dir, f"true{suffix}.npy")
            np.save(pred_path, pred_arr)
            np.save(true_path, true_arr)
            print(f"Saved pred/true ({space_tag}):")
            print("  pred:", pred_path)
            print("  true:", true_path)

        # Preserve channel names with the saved plotting arrays.
        col_names = _load_col_names(args.root_path, args.data_path, enc_in)
        pred_true_npz = os.path.join(args.out_dir, f"pred_true_4channels_data{tag_suffix}.npz")
        np.savez(
            pred_true_npz,
            pred_first=pred[0],
            true_first=true[0],
            pred_len=np.array(args.pred_len),
            col_names=np.array(col_names[:enc_in], dtype=object),
        )

        if args.save_pred_true_space in ("scaled", "both"):
            _save_one("scaled", pred, true)
        if args.save_pred_true_space in ("original", "both"):
            _save_one("original", pred_inv, true_inv)

        print("pred_true_4channels_data.npz:", pred_true_npz)


if __name__ == "__main__":
    main()
