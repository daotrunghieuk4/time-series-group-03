"""Model utilities for the repository.

This file keeps the mandatory structure simple while the original TimeMixer
implementation from the submitted ZIP is preserved in ``src/timemixer_core``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np


class SeasonalNaiveForecaster:
    """Seasonal naive baseline: repeat the same seasonal phase from the past."""

    def __init__(self, season_length: int = 24):
        if season_length <= 0:
            raise ValueError("season_length must be positive.")
        self.season_length = season_length

    def predict(self, history: np.ndarray, pred_len: int) -> np.ndarray:
        """Forecast by repeating the last seasonal block."""
        if history.ndim != 2:
            raise ValueError("history must have shape (time, channels).")
        if len(history) < self.season_length:
            raise ValueError("history is shorter than season_length.")
        block = history[-self.season_length:]
        reps = int(np.ceil(pred_len / self.season_length))
        return np.tile(block, (reps, 1))[:pred_len]


def timemixer_core_dir() -> Path:
    """Return the directory containing the original TimeMixer code."""
    return Path(__file__).resolve().parent / "timemixer_core"


def build_timemixer_command(pred_len: int = 96, seq_len: int = 96) -> list[str]:
    """Build a command that runs the original TimeMixer implementation.

    The command points to ``data/raw/electricity.csv`` to match the mandatory
    repository layout.
    """
    root = Path(__file__).resolve().parents[1]
    core = timemixer_core_dir()
    return [
        sys.executable,
        str(core / "run.py"),
        "--task_name", "long_term_forecast",
        "--is_training", "1",
        "--root_path", str(root / "data" / "raw") + "/",
        "--data_path", "electricity.csv",
        "--model_id", f"ECL_{seq_len}_{pred_len}",
        "--model", "TimeMixer",
        "--data", "custom",
        "--features", "M",
        "--seq_len", str(seq_len),
        "--label_len", "0",
        "--pred_len", str(pred_len),
        "--e_layers", "3",
        "--d_layers", "1",
        "--factor", "3",
        "--enc_in", "321",
        "--dec_in", "321",
        "--c_out", "321",
        "--des", "Exp",
        "--itr", "1",
        "--d_model", "16",
        "--d_ff", "32",
        "--batch_size", "32",
        "--learning_rate", "0.01",
        "--train_epochs", "20",
        "--patience", "10",
        "--down_sampling_layers", "3",
        "--down_sampling_method", "avg",
        "--down_sampling_window", "2",
    ]


def run_timemixer(pred_len: int = 96, seq_len: int = 96) -> subprocess.CompletedProcess:
    """Run TimeMixer as a subprocess from the preserved original code."""
    command = build_timemixer_command(pred_len=pred_len, seq_len=seq_len)
    return subprocess.run(command, cwd=timemixer_core_dir(), check=True)
