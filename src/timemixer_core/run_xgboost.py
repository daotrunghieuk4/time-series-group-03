"""
XGBoost baseline aligned with the TimeMixer repo/paper setting for ECL.

This script intentionally keeps the same data protocol as Dataset_Custom in
`data_provider/data_loader.py`:
  - chronological split: train/val/test = 70%/10%/20%
  - StandardScaler is fitted on train only
  - test windows start with the lookback immediately before the test target range
  - no extra IQR clipping / outlier processing is applied

Note: XGBoost is an additional baseline for reference. It is not one of the
main baselines reported in the TimeMixer paper.
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
from xgboost import XGBRegressor

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
) -> Tuple[np.ndarray, pd.Series, StandardScaler, int, int, int]:
    """
    Load ECL/custom data using the same split/scaling logic as Dataset_Custom.

    Returns:
        data_scaled: np.ndarray, shape [time, channels]
        dates: pd.Series of datetime64
        scaler: fitted StandardScaler
        num_train, num_val, num_test: split sizes
    """
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

    dates = pd.to_datetime(df_raw["date"])
    df_data = df_raw[df_raw.columns[1:]]

    n = len(df_raw)
    num_train = int(n * 0.7)
    num_test = int(n * 0.2)
    num_val = n - num_train - num_test

    missing = int(df_data.isna().sum().sum())
    if verbose:
        print("=" * 70)
        print("XGBOOST BASELINE - TimeMixer-aligned ECL protocol")
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

    return data_scaled, dates, scaler, num_train, num_val, num_test


# ============================================================
# 2. FEATURE ENGINEERING FOR THE EXTRA XGBoost BASELINE
# ============================================================

def make_calendar_features(dates: pd.Series) -> np.ndarray:
    """Small calendar feature set for the XGBoost baseline."""
    return np.stack(
        [
            dates.dt.hour.to_numpy(dtype=np.float32) / 23.0 - 0.5,
            dates.dt.dayofweek.to_numpy(dtype=np.float32) / 6.0 - 0.5,
            (dates.dt.month.to_numpy(dtype=np.float32) - 1.0) / 11.0 - 0.5,
            (dates.dt.dayofweek.to_numpy() >= 5).astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)


def create_features_for_channel(
    data_1d: np.ndarray,
    time_feats: np.ndarray,
    seq_len: int,
    idx: int,
    channel_id: int,
    n_channels: int,
) -> np.ndarray:
    """Create one feature vector from one channel and one lookback window."""
    window = data_1d[idx: idx + seq_len]
    if len(window) != seq_len:
        raise ValueError(f"Invalid window length at idx={idx}: got {len(window)}, expected {seq_len}")

    lag_24 = window[-24] if seq_len >= 24 else window[0]
    lag_48 = window[-48] if seq_len >= 48 else window[0]
    lag_96 = window[-96] if seq_len >= 96 else window[0]

    feats = [
        window[-1],
        lag_24,
        lag_48,
        lag_96,
        float(np.mean(window)),
        float(np.std(window)),
        float(np.min(window)),
        float(np.max(window)),
        float(window[-1] - lag_24),
        float(window[-1] - lag_48),
        float(channel_id / max(n_channels - 1, 1)),
    ]

    # Calendar features at the first forecast timestamp.
    tf_idx = min(idx + seq_len, len(time_feats) - 1)
    feats.extend(time_feats[tf_idx].tolist())

    return np.array(feats, dtype=np.float32)


def build_horizon_dataset(
    data_scaled: np.ndarray,
    time_feats: np.ndarray,
    seq_len: int,
    pred_len: int,
    start_idx: int,
    end_idx: int,
    n_channels_sample: int,
    n_windows_train: int,
    n_horizons_train: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a single-output XGBoost dataset.

    Each row represents one tuple: (window, channel, horizon) -> target value.
    Training windows are constrained to the training range to match the repo's
    train/validation/test separation.
    """
    n_features = data_scaled.shape[1]
    n_windows = end_idx - start_idx - seq_len - pred_len + 1
    if n_windows <= 0:
        raise ValueError(
            f"Not enough training data for seq_len={seq_len}, pred_len={pred_len}. "
            f"Available range: start={start_idx}, end={end_idx}."
        )

    if n_channels_sample <= 0 or n_channels_sample >= n_features:
        channels = np.arange(n_features)
    else:
        channels = np.sort(rng.choice(n_features, n_channels_sample, replace=False))

    if n_windows_train <= 0 or n_windows_train >= n_windows:
        window_indices = np.arange(n_windows)
    else:
        window_indices = np.sort(rng.choice(n_windows, n_windows_train, replace=False))

    if n_horizons_train <= 0 or n_horizons_train >= pred_len:
        horizons = np.arange(pred_len)
    else:
        horizons = np.sort(rng.choice(pred_len, n_horizons_train, replace=False))

    X_list = []
    y_list = []

    for i in window_indices:
        abs_idx = start_idx + int(i)
        for c in channels:
            c = int(c)
            base_feats = create_features_for_channel(
                data_scaled[:, c], time_feats, seq_len, abs_idx, c, n_features
            )
            for h in horizons:
                h = int(h)
                h_norm = np.array([h / max(pred_len - 1, 1)], dtype=np.float32)
                X_list.append(np.concatenate([base_feats, h_norm]))
                y_list.append(data_scaled[abs_idx + seq_len + h, c])

    return np.asarray(X_list, dtype=np.float32), np.asarray(y_list, dtype=np.float32)


# ============================================================
# 3. PREDICTION & EVALUATION
# ============================================================

def predict_timemixer_test_windows(
    model: XGBRegressor,
    data_scaled: np.ndarray,
    time_feats: np.ndarray,
    seq_len: int,
    pred_len: int,
    test_target_start: int,
    test_end: int,
    test_step: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict test windows using the same indexing as Dataset_Custom(flag='test').

    Dataset_Custom test range is:
        border1 = len(data) - num_test - seq_len
        border2 = len(data)
    Therefore the first target point is exactly len(data) - num_test.
    """
    if test_step < 1:
        raise ValueError("test_step must be >= 1")

    n_features = data_scaled.shape[1]
    test_input_start = test_target_start - seq_len
    n_test_windows = test_end - test_target_start - pred_len + 1
    if n_test_windows <= 0:
        raise ValueError(
            f"Not enough test data for pred_len={pred_len}. Test size={test_end - test_target_start}."
        )

    test_indices = list(range(0, n_test_windows, test_step))
    preds_list = []
    trues_list = []

    for idx_i, i in enumerate(test_indices, start=1):
        abs_idx = test_input_start + i
        true_start = abs_idx + seq_len
        sample_trues = data_scaled[true_start: true_start + pred_len, :]

        # Compute base features once per channel, then reuse for all horizons.
        base_feats_by_channel = [
            create_features_for_channel(data_scaled[:, c], time_feats, seq_len, abs_idx, c, n_features)
            for c in range(n_features)
        ]

        X_batch = []
        for h in range(pred_len):
            h_norm = h / max(pred_len - 1, 1)
            for c in range(n_features):
                X_batch.append(np.concatenate([base_feats_by_channel[c], [h_norm]]))

        X_batch = np.asarray(X_batch, dtype=np.float32)
        y_pred = model.predict(X_batch).reshape(pred_len, n_features)

        preds_list.append(y_pred)
        trues_list.append(sample_trues)

        if idx_i % 50 == 0 or idx_i == len(test_indices):
            print(f"   Predicted {idx_i}/{len(test_indices)} test windows...")

    return np.asarray(preds_list, dtype=np.float32), np.asarray(trues_list, dtype=np.float32)


def compute_metrics(preds: np.ndarray, trues: np.ndarray) -> Tuple[float, float, float]:
    mae = float(np.mean(np.abs(preds - trues)))
    mse = float(np.mean((preds - trues) ** 2))
    rmse = float(np.sqrt(mse))
    return mse, mae, rmse


# ============================================================
# 4. MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="XGBoost baseline aligned with TimeMixer ECL protocol")
    parser.add_argument("--root_path", type=str, default="./dataset/electricity/")
    parser.add_argument("--data_path", type=str, default="electricity.csv")
    parser.add_argument("--target", type=str, default="OT")
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--pred_lens", type=int, nargs="+", default=[96, 192, 336, 720])
    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--max_depth", type=int, default=6)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--n_channels_train", type=int, default=50,
                        help="Number of channels sampled for training. Use <=0 for all channels.")
    parser.add_argument("--n_windows_train", type=int, default=1000,
                        help="Number of train windows sampled. Use <=0 for all windows.")
    parser.add_argument("--n_horizons_train", type=int, default=30,
                        help="Number of horizons sampled for training. Use <=0 for all horizons.")
    parser.add_argument("--test_step", type=int, default=10,
                        help="Evaluate every k-th test window. Use 1 for the full test set.")
    parser.add_argument("--train_with_val", action="store_true",
                        help="Optional: train XGBoost with train+val. Default uses train only, like TimeMixer training.")
    parser.add_argument("--use_gpu_xgb", action="store_true",
                        help="Use XGBoost CUDA device if your installed xgboost supports it.")
    parser.add_argument("--random_seed", type=int, default=2021)
    parser.add_argument("--save_csv", type=str, default="./results/xgboost_baseline_results.csv")
    args = parser.parse_args()

    rng = np.random.default_rng(args.random_seed)

    data, dates, scaler, n_train, n_val, n_test = load_and_scale_like_timemixer(
        args.root_path, args.data_path, target=args.target, verbose=True
    )
    time_feats = make_calendar_features(dates)

    n = len(data)
    test_target_start = n - n_test
    train_end = n_train + n_val if args.train_with_val else n_train

    print(f"XGBoost params      : n_estimators={args.n_estimators}, max_depth={args.max_depth}, lr={args.learning_rate}")
    print(f"Training range      : [0, {train_end}) {'(train+val)' if args.train_with_val else '(train only)'}")
    print(f"Train subsample     : {args.n_channels_train} channels, {args.n_windows_train} windows, {args.n_horizons_train} horizons")
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
        print(f"\n--- pred_len={pred_len} ---")
        print("   Building training data...")
        X_train, y_train = build_horizon_dataset(
            data, time_feats, args.seq_len, pred_len,
            start_idx=0,
            end_idx=train_end,
            n_channels_sample=args.n_channels_train,
            n_windows_train=args.n_windows_train,
            n_horizons_train=min(args.n_horizons_train, pred_len) if args.n_horizons_train > 0 else args.n_horizons_train,
            rng=rng,
        )
        print(f"   Training data: X={X_train.shape}, y={y_train.shape}")

        xgb_params = dict(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            tree_method="hist",
            n_jobs=-1,
            verbosity=0,
            random_state=args.random_seed,
        )
        if args.use_gpu_xgb:
            xgb_params["device"] = "cuda"

        print("   Training XGBoost...")
        model = XGBRegressor(**xgb_params)
        model.fit(X_train, y_train)
        print(f"   Training done ({time.time() - t0:.1f}s)")

        print("   Predicting on TimeMixer-aligned test windows...")
        preds, trues = predict_timemixer_test_windows(
            model, data, time_feats, args.seq_len, pred_len,
            test_target_start=test_target_start,
            test_end=n,
            test_step=args.test_step,
        )

        mse, mae, rmse = compute_metrics(preds, trues)
        elapsed = time.time() - t0
        print(f"   pred_len={pred_len}: MSE={mse:.6f}, MAE={mae:.6f}, RMSE={rmse:.6f}, Time={elapsed:.1f}s")
        print(f"{pred_len:<12} {mse:<12.6f} {mae:<12.6f} {rmse:<12.6f} {elapsed:<10.1f}s")
        results.append({"pred_len": pred_len, "mse": mse, "mae": mae, "rmse": rmse, "time_sec": elapsed})

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Pred_len':<12} {'MSE':<12} {'MAE':<12} {'RMSE':<12}")
    print("-" * 48)
    for r in results:
        print(f"{r['pred_len']:<12} {r['mse']:<12.6f} {r['mae']:<12.6f} {r['rmse']:<12.6f}")

    if args.save_csv:
        os.makedirs(os.path.dirname(args.save_csv), exist_ok=True)
        pd.DataFrame(results).to_csv(args.save_csv, index=False)
        print(f"\nSaved results to: {args.save_csv}")

    print("Done!")
    return results


if __name__ == "__main__":
    main()
