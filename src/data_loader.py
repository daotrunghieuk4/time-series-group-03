"""Data loading utilities for the TimeMixer Electricity experiment.

The mandatory repository structure keeps the original data at:
    data/raw/electricity.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def project_root() -> Path:
    """Return repository root inferred from this file location."""
    return Path(__file__).resolve().parents[1]


def load_electricity_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load the Electricity dataset.

    Parameters
    ----------
    path:
        Optional path to CSV. If omitted, ``data/raw/electricity.csv`` is used.

    Returns
    -------
    pandas.DataFrame
        DataFrame with a parsed ``date`` column when present.
    """
    if path is None:
        path = project_root() / "data" / "raw" / "electricity.csv"
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find {path}. Put electricity.csv into data/raw/ before running."
        )
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
    return df


def split_by_time(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.10,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a time series dataframe into train/validation/test without shuffling."""
    if not 0 < train_ratio < 1 or not 0 <= val_ratio < 1:
        raise ValueError("train_ratio and val_ratio must be valid fractions.")
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return (
        df.iloc[:train_end].copy(),
        df.iloc[train_end:val_end].copy(),
        df.iloc[val_end:].copy(),
    )


def numeric_values(df: pd.DataFrame) -> np.ndarray:
    """Return numeric channels only, excluding the date column."""
    return df.drop(columns=["date"], errors="ignore").select_dtypes(include=[np.number]).to_numpy(dtype="float32")


def make_sliding_windows(
    values: np.ndarray,
    seq_len: int = 96,
    pred_len: int = 96,
    stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create supervised sliding-window samples.

    X shape: (n_windows, seq_len, n_channels)
    y shape: (n_windows, pred_len, n_channels)
    """
    if values.ndim != 2:
        raise ValueError("values must have shape (time, channels).")
    if seq_len <= 0 or pred_len <= 0 or stride <= 0:
        raise ValueError("seq_len, pred_len and stride must be positive.")

    xs, ys = [], []
    last_start = len(values) - seq_len - pred_len + 1
    for start in range(0, max(0, last_start), stride):
        mid = start + seq_len
        end = mid + pred_len
        xs.append(values[start:mid])
        ys.append(values[mid:end])
    if not xs:
        return np.empty((0, seq_len, values.shape[1])), np.empty((0, pred_len, values.shape[1]))
    return np.stack(xs), np.stack(ys)
