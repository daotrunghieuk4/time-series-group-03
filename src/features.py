"""Feature engineering helpers for time-series forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_time_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Add calendar variables from a datetime column."""
    if date_col not in df.columns:
        raise KeyError(f"{date_col!r} is not in dataframe columns.")
    out = df.copy()
    dt = pd.to_datetime(out[date_col])
    out["hour"] = dt.dt.hour
    out["day_of_week"] = dt.dt.dayofweek
    out["month"] = dt.dt.month
    out["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
    return out


def add_fourier_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Add cyclic Fourier features for daily and weekly seasonality."""
    out = add_time_features(df, date_col=date_col)
    out["sin_day"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["cos_day"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["sin_week"] = np.sin(2 * np.pi * out["day_of_week"] / 7)
    out["cos_week"] = np.cos(2 * np.pi * out["day_of_week"] / 7)
    return out
