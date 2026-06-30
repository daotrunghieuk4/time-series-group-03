"""Evaluation helpers for forecasting results."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def mse(y_true, y_pred) -> float:
    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def evaluate_predictions(y_true, y_pred) -> dict[str, float]:
    """Return MSE, MAE and RMSE."""
    return {"mse": mse(y_true, y_pred), "mae": mae(y_true, y_pred), "rmse": rmse(y_true, y_pred)}


def load_metrics(path: str | Path | None = None) -> pd.DataFrame:
    """Load saved experiment metrics."""
    if path is None:
        path = Path(__file__).resolve().parents[1] / "results" / "metrics.csv"
    return pd.read_csv(path)


if __name__ == "__main__":
    print(load_metrics())
