"""Weather-data cleaning, feature engineering, and scaling utilities."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


DATE_CANDIDATES = ("date", "datetime", "date time", "timestamp", "time")
SPARSE_COLUMNS = {"rain", "raining", "swdr", "par", "max. par"}


def detect_datetime_column(columns):
    """Return the most likely datetime column."""
    lower_map = {str(col).strip().lower(): col for col in columns}
    for candidate in DATE_CANDIDATES:
        if candidate in lower_map:
            return lower_map[candidate]
    return columns[0]


def load_raw_weather(path, target_col="rain"):
    """Read weather data, parse time, and convert measurements to numeric."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Weather data not found: {path}")

    frame = pd.read_csv(path)
    frame.columns = [str(col).strip() for col in frame.columns]
    date_col = detect_datetime_column(frame.columns)

    parsed = pd.to_datetime(frame[date_col], errors="coerce")
    if parsed.notna().mean() < 0.80:
        parsed = pd.to_datetime(
            frame[date_col],
            errors="coerce",
            dayfirst=True,
        )

    frame[date_col] = parsed
    frame = (
        frame.dropna(subset=[date_col])
        .sort_values(date_col)
        .drop_duplicates(subset=[date_col], keep="last")
    )

    for col in frame.columns:
        if col != date_col:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

    frame = frame.replace([np.inf, -np.inf], np.nan)
    empty_cols = [
        col
        for col in frame.columns
        if col != date_col and frame[col].notna().sum() == 0
    ]
    frame = frame.drop(columns=empty_cols)

    matches = [
        col for col in frame.columns if col.lower() == target_col.lower()
    ]
    if not matches:
        raise KeyError(
            f"Target '{target_col}' not found. Columns: {list(frame.columns)}"
        )

    return frame, date_col, matches[0]


def resample_weather(frame, date_col, frequency="10min"):
    """Convert the series to a regular time grid."""
    return frame.set_index(date_col).resample(frequency).asfreq()


def split_by_time(frame, train_ratio=0.70, val_ratio=0.15):
    """Split a time series without shuffling."""
    n_samples = len(frame)
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))

    return (
        frame.iloc[:train_end].copy(),
        frame.iloc[train_end:val_end].copy(),
        frame.iloc[val_end:].copy(),
    )


def fill_missing_values(frame):
    """Interpolate and fill remaining missing values inside one split."""
    return (
        frame.copy()
        .interpolate(method="time", limit_direction="both")
        .ffill()
        .bfill()
    )


def fit_iqr_bounds(train_df, sparse_columns=None):
    """Learn outlier clipping bounds from the training split."""
    sparse_columns = {
        col.lower() for col in (sparse_columns or SPARSE_COLUMNS)
    }
    bounds = {}

    for col in train_df.columns:
        if col.lower() in sparse_columns:
            continue

        q1, q3 = train_df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        if np.isfinite(iqr) and iqr > 0:
            bounds[col] = (
                float(q1 - 1.5 * iqr),
                float(q3 + 1.5 * iqr),
            )

    return bounds


def clip_outliers(frame, bounds):
    """Clip values using bounds learned from the training split."""
    frame = frame.copy()
    for col, (lower, upper) in bounds.items():
        frame[col] = frame[col].clip(lower, upper)
    return frame


def add_time_features(frame):
    """Add calendar and Fourier features from the DatetimeIndex."""
    frame = frame.copy()
    index = frame.index
    minute = index.hour * 60 + index.minute
    weekday = index.dayofweek
    day_of_year = index.dayofyear

    frame["hour"] = index.hour
    frame["day_of_week"] = weekday
    frame["month"] = index.month
    frame["is_weekend"] = (weekday >= 5).astype(int)

    frame["sin_day"] = np.sin(2 * np.pi * minute / 1440.0)
    frame["cos_day"] = np.cos(2 * np.pi * minute / 1440.0)
    frame["sin_week"] = np.sin(2 * np.pi * weekday / 7.0)
    frame["cos_week"] = np.cos(2 * np.pi * weekday / 7.0)
    frame["sin_year"] = np.sin(
        2 * np.pi * day_of_year / 365.25
    )
    frame["cos_year"] = np.cos(
        2 * np.pi * day_of_year / 365.25
    )

    return frame


def fit_scaler(train_df, feature_cols=None):
    """Fit StandardScaler using training data only."""
    feature_cols = feature_cols or list(train_df.columns)
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols])
    return scaler


def make_model_frame(
    frame,
    scaler,
    feature_cols,
    target_col="rain",
    output_target_col="rain_target_mm",
):
    """Scale model features while retaining the target in millimetres."""
    scaled = pd.DataFrame(
        scaler.transform(frame[feature_cols]),
        index=frame.index,
        columns=feature_cols,
    )
    scaled[output_target_col] = frame[target_col].to_numpy()
    scaled.index.name = "date"
    return scaled.reset_index()


def preprocess_weather(
    input_path,
    target_col="rain",
    frequency="10min",
    train_ratio=0.70,
    val_ratio=0.15,
):
    """Run the complete preprocessing pipeline without saving files."""
    raw_df, date_col, target_col = load_raw_weather(
        input_path,
        target_col,
    )
    regular_df = resample_weather(raw_df, date_col, frequency)
    train_df, val_df, test_df = split_by_time(
        regular_df,
        train_ratio,
        val_ratio,
    )

    train_df = fill_missing_values(train_df)
    val_df = fill_missing_values(val_df)
    test_df = fill_missing_values(test_df)

    bounds = fit_iqr_bounds(train_df)
    train_df = add_time_features(clip_outliers(train_df, bounds))
    val_df = add_time_features(clip_outliers(val_df, bounds))
    test_df = add_time_features(clip_outliers(test_df, bounds))

    feature_cols = list(train_df.columns)
    scaler = fit_scaler(train_df, feature_cols)

    return {
        "train": make_model_frame(
            train_df,
            scaler,
            feature_cols,
            target_col,
        ),
        "validation": make_model_frame(
            val_df,
            scaler,
            feature_cols,
            target_col,
        ),
        "test": make_model_frame(
            test_df,
            scaler,
            feature_cols,
            target_col,
        ),
        "clean_splits": (train_df, val_df, test_df),
        "scaler": scaler,
        "feature_cols": feature_cols,
        "target_col": target_col,
        "iqr_bounds": bounds,
    }
