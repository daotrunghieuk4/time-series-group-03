"""Metrics, result tables, and plots for rainfall forecasting."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
)


def apply_rain_threshold(values, threshold=0.1):
    """Set very small rainfall predictions to zero."""
    result = np.asarray(values, dtype=float).copy()
    result[result < threshold] = 0.0
    return result


def calculate_smape(y_true, y_pred):
    """Calculate symmetric mean absolute percentage error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0

    ratio = np.divide(
        np.abs(y_true - y_pred),
        denominator,
        out=np.zeros_like(denominator, dtype=float),
        where=denominator > 1e-10,
    )
    return float(100 * ratio.mean())


def evaluate_model(name, y_true, y_pred, rain_threshold=0.1):
    """Calculate regression metrics and rain-event classification metrics."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mse = mean_squared_error(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true >= rain_threshold,
        y_pred >= rain_threshold,
        average="binary",
        zero_division=0,
    )

    return {
        "Model": name,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mse),
        "sMAPE (%)": calculate_smape(y_true, y_pred),
        "MSE": mse,
        "R2": r2_score(y_true, y_pred),
        "Rain precision": precision,
        "Rain recall": recall,
        "Rain F1": f1,
    }


def build_metrics_table(
    y_true,
    predictions,
    rain_threshold=0.1,
    decimals=6,
):
    """Build a comparison table from a model-name to prediction mapping."""
    rows = [
        evaluate_model(
            model_name,
            y_true,
            y_pred,
            rain_threshold,
        )
        for model_name, y_pred in predictions.items()
    ]

    result = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    numeric_cols = result.select_dtypes(include=np.number).columns
    result[numeric_cols] = result[numeric_cols].round(decimals)
    return result


def plot_predictions(
    timestamps,
    y_true,
    predictions,
    output_path=None,
    start=0,
    end=350,
):
    """Plot actual rainfall and predictions over one test interval."""
    timestamps = pd.to_datetime(timestamps)
    y_true = np.asarray(y_true)
    end = min(end, len(y_true))

    colors = ["#7A5195", "#EF5675", "#2A9D8F", "#FFA600"]

    plt.figure(figsize=(18, 6))
    plt.plot(
        timestamps[start:end],
        y_true[start:end],
        label="Actual",
        color="#172B4D",
        linewidth=2.2,
        zorder=5,
    )

    for color, (name, values) in zip(colors, predictions.items()):
        plt.plot(
            timestamps[start:end],
            np.asarray(values)[start:end],
            label=name,
            color=color,
            linewidth=1.4,
            alpha=0.9,
        )

    plt.xlabel("Forecast time")
    plt.ylabel("Rainfall (mm)")
    plt.title("Actual and predicted rainfall")
    plt.legend()
    plt.gcf().autofmt_xdate()
    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches="tight")

    return plt.gcf()


def plot_metric_comparison(metrics_df, output_path=None):
    """Plot MAE, RMSE, and sMAPE for all models."""
    metrics_df = metrics_df.set_index("Model")
    colors = ["#7A5195", "#EF5675", "#2A9D8F", "#FFA600"]
    metrics = ["MAE", "RMSE", "sMAPE (%)"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for axis, metric in zip(axes, metrics):
        axis.bar(
            metrics_df.index,
            metrics_df[metric],
            color=colors[: len(metrics_df)],
        )
        axis.set_title(metric)
        axis.tick_params(axis="x", rotation=20)

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches="tight")

    return fig
