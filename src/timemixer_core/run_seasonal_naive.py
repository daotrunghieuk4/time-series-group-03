"""
Seasonal Naive baseline aligned with the TimeMixer repo/paper setting for ECL.

This script intentionally keeps the same data protocol as Dataset_Custom in
`data_provider/data_loader.py`:
  - chronological split: train/val/test = 70%/10%/20%
  - StandardScaler is fitted on train only
  - test windows start with the lookback immediately before the test target range
  - no extra IQR clipping / outlier processing is applied

Note: Seasonal Naive is an additional baseline for reference. It is not one of
the main baselines reported in the TimeMixer paper.
"""

import argparse
import io
import os
import sys
import time
import warnings
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


# ============================================================
# 1. DATA PROTOCOL: MATCH TimeMixer Dataset_Custom
# ============================================================

def load_and_scale_like_timemixer(
    root_path: str,
    data_path: str,
    target: str = "OT",
    verbose: bool = True,
) -> Tuple[np.ndarray, StandardScaler, int, int, int]:
    """Load and scale data using the same split/scaling logic as Dataset_Custom."""
    csv_path = os.path.join(root_path, data_path)
    df_raw = pd.read_csv(csv_path)

    if "date" not in df_raw.columns:
        raise ValueError("Input csv must contain a 'date' column.")
    if target not in df_raw.columns:
        raise ValueError(f"Target column '{target}' not found in {data_path}.")

    # Same column ordering as Dataset_Custom: ['date'] + other features + [target]
    cols = list(df_raw.columns)
    cols.remove(target)
    cols.remove("date")
    df_raw = df_raw[["date"] + cols + [target]]

    df_data = df_raw[df_raw.columns[1:]]
    n = len(df_raw)
    num_train = int(n * 0.7)
    num_test = int(n * 0.2)
    num_val = n - num_train - num_test

    missing = int(df_data.isna().sum().sum())
    if verbose:
        print("=" * 70)
        print("SEASONAL NAIVE BASELINE - TimeMixer-aligned ECL protocol")
        print("=" * 70)
        print(f"Data shape          : {df_raw.shape}")
        print(f"Number of variables : {df_data.shape[1]}")
        print(f"Missing values      : {missing}")
        print("Split protocol      : chronological 70% / 10% / 20%")
        print(f"Train | Val | Test  : {num_train} | {num_val} | {num_test}")
        print("Scaler              : fit on train only, transform all")
        print("Extra clipping      : disabled, to keep data close to repo/paper")
        print("=" * 70)

    if missing > 0:
        raise ValueError(
            "Missing values were found. The official ECL file normally has no missing values. "
            "Please clean the data explicitly before running this paper-aligned baseline."
        )

    scaler = StandardScaler()
    data_values = df_data.values.astype(np.float32)
    scaler.fit(data_values[:num_train])
    data_scaled = scaler.transform(data_values).astype(np.float32)

    return data_scaled, scaler, num_train, num_val, num_test


# ============================================================
# 2. SEASONAL NAIVE MODEL
# ============================================================

def seasonal_naive_predict(
    data_scaled: np.ndarray,
    seq_len: int,
    pred_len: int,
    seasonal_period: int,
    test_target_start: int,
    test_end: int,
    test_step: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Seasonal Naive test prediction using the same indexing as Dataset_Custom(flag='test').

    The first input window starts at test_target_start - seq_len, so the first
    forecast target is exactly test_target_start.
    """
    if test_step < 1:
        raise ValueError("test_step must be >= 1")
    if seasonal_period < 1:
        raise ValueError("seasonal_period must be >= 1")

    n_features = data_scaled.shape[1]
    test_input_start = test_target_start - seq_len
    n_test_windows = test_end - test_target_start - pred_len + 1
    if n_test_windows <= 0:
        raise ValueError(
            f"Not enough test data for pred_len={pred_len}. Test size={test_end - test_target_start}."
        )

    test_indices = list(range(0, n_test_windows, test_step))
    preds = np.zeros((len(test_indices), pred_len, n_features), dtype=np.float32)
    trues = np.zeros_like(preds)

    for out_i, i in enumerate(test_indices):
        s_begin = test_input_start + i
        s_end = s_begin + seq_len
        r_end = s_end + pred_len

        input_window = data_scaled[s_begin:s_end]
        true_window = data_scaled[s_end:r_end]
        trues[out_i] = true_window

        for h in range(pred_len):
            # Daily naive with seq_len=96 can use h modulo 24 from the last day.
            lag_idx = seq_len - seasonal_period + (h % seasonal_period)
            if 0 <= lag_idx < seq_len:
                preds[out_i, h] = input_window[lag_idx]
            else:
                # Fallback for seasonal_period > seq_len, e.g. 168 with lookback 96.
                preds[out_i, h] = input_window[-1]

    return preds, trues


def compute_metrics(preds: np.ndarray, trues: np.ndarray) -> Tuple[float, float, float]:
    mae = float(np.mean(np.abs(preds - trues)))
    mse = float(np.mean((preds - trues) ** 2))
    rmse = float(np.sqrt(mse))
    return mse, mae, rmse


# ============================================================
# 3. MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Seasonal Naive baseline aligned with TimeMixer ECL protocol")
    parser.add_argument("--root_path", type=str, default="./dataset/electricity/")
    parser.add_argument("--data_path", type=str, default="electricity.csv")
    parser.add_argument("--target", type=str, default="OT")
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--pred_lens", type=int, nargs="+", default=[96, 192, 336, 720])
    parser.add_argument("--seasonal_period", type=int, default=24,
                        help="Seasonal period. For ECL hourly data, 24=daily. With seq_len=96, weekly 168 is not available.")
    parser.add_argument("--test_step", type=int, default=1,
                        help="Evaluate every k-th test window. Use 1 for the full test set.")
    parser.add_argument("--save_csv", type=str, default="./results/seasonal_naive_results.csv")
    args = parser.parse_args()

    data, scaler, n_train, n_val, n_test = load_and_scale_like_timemixer(
        args.root_path, args.data_path, target=args.target, verbose=True
    )

    n = len(data)
    test_target_start = n - n_test

    print(f"Seasonal period     : {args.seasonal_period}")
    print(f"Test target start   : {test_target_start}")
    print(f"Test step           : every {args.test_step} window(s)")

    print("\n" + "=" * 70)
    print("EVALUATION ON TEST SET")
    print("=" * 70)
    print(f"{'Pred_len':<12} {'MSE':<12} {'MAE':<12} {'RMSE':<12} {'Time':<10}")
    print("-" * 60)

    results = []
    for pred_len in args.pred_lens:
        t0 = time.time()
        preds, trues = seasonal_naive_predict(
            data,
            seq_len=args.seq_len,
            pred_len=pred_len,
            seasonal_period=args.seasonal_period,
            test_target_start=test_target_start,
            test_end=n,
            test_step=args.test_step,
        )
        mse, mae, rmse = compute_metrics(preds, trues)
        elapsed = time.time() - t0
        print(f"{pred_len:<12} {mse:<12.6f} {mae:<12.6f} {rmse:<12.6f} {elapsed:<10.1f}s")
        results.append({"pred_len": pred_len, "mse": mse, "mae": mae, "rmse": rmse, "time_sec": elapsed})

    if args.save_csv:
        os.makedirs(os.path.dirname(args.save_csv), exist_ok=True)
        pd.DataFrame(results).to_csv(args.save_csv, index=False)
        print(f"\nSaved results to: {args.save_csv}")

    print("Done!")
    return results


if __name__ == "__main__":
    main()
