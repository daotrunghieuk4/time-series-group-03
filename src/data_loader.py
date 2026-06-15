"""Utilities for loading processed weather data and creating forecast samples."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def load_processed_data(data_dir="../data/processed"):
    """Load the train, validation, and test CSV files."""
    data_dir = Path(data_dir)
    paths = {
        "train": data_dir / "final_train.csv",
        "validation": data_dir / "final_val.csv",
        "test": data_dir / "final_test.csv",
    }

    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing processed data files: " + ", ".join(missing)
        )

    return (
        pd.read_csv(paths["train"], parse_dates=["date"]),
        pd.read_csv(paths["validation"], parse_dates=["date"]),
        pd.read_csv(paths["test"], parse_dates=["date"]),
    )


def combine_processed_data(
    train_df,
    val_df,
    test_df,
    target_col="rain_target_mm",
):
    """Combine chronological splits and return model arrays and boundaries."""
    feature_cols = [
        col
        for col in train_df.columns
        if col not in {"date", target_col}
    ]

    combined_df = pd.concat(
        [train_df, val_df, test_df],
        ignore_index=True,
    )

    x_all = combined_df[feature_cols].to_numpy(dtype=np.float32)
    y_all = combined_df[target_col].to_numpy(dtype=np.float32)
    timestamps = combined_df["date"].to_numpy()

    train_end = len(train_df)
    val_end = train_end + len(val_df)

    return {
        "dataframe": combined_df,
        "features": x_all,
        "targets": y_all,
        "timestamps": timestamps,
        "feature_cols": feature_cols,
        "train_end": train_end,
        "val_end": val_end,
    }


def build_split(
    x_all,
    y_all,
    target_start,
    target_end,
    seq_len=144,
    horizon=1,
):
    """Create aligned tabular inputs, sequence starts, targets, and positions."""
    first_target = max(seq_len, target_start)
    target_positions = np.arange(
        first_target,
        target_end,
        dtype=np.int64,
    )

    starts = target_positions - horizon + 1
    targets = y_all[target_positions].astype(np.float32)
    tabular_inputs = x_all[target_positions - horizon]

    return starts, tabular_inputs, targets, target_positions


def create_forecast_splits(data, seq_len=144, horizon=1):
    """Build train, validation, and test samples from combined data."""
    x_all = data["features"]
    y_all = data["targets"]
    train_end = data["train_end"]
    val_end = data["val_end"]

    return {
        "train": build_split(
            x_all,
            y_all,
            0,
            train_end,
            seq_len,
            horizon,
        ),
        "validation": build_split(
            x_all,
            y_all,
            train_end,
            val_end,
            seq_len,
            horizon,
        ),
        "test": build_split(
            x_all,
            y_all,
            val_end,
            len(y_all),
            seq_len,
            horizon,
        ),
    }


class WindowDataset(Dataset):
    """PyTorch dataset returning a look-back window and rainfall target."""

    def __init__(self, values, starts, targets, seq_len=144):
        self.values = values
        self.starts = starts
        self.targets = targets
        self.seq_len = seq_len

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        start = int(self.starts[index])
        window = self.values[start - self.seq_len : start]
        target_mm = max(float(self.targets[index]), 0.0)

        return (
            torch.from_numpy(window),
            torch.tensor(
                [np.log1p(target_mm)],
                dtype=torch.float32,
            ),
            torch.tensor([target_mm], dtype=torch.float32),
        )
