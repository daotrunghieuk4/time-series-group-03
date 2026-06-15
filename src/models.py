"""Model definitions for rainfall forecasting."""

from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

import torch.nn as nn


def create_linear_regression():
    """Create the baseline linear-regression model."""
    return LinearRegression()


def create_xgboost(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
):
    """Create the XGBoost rainfall-regression model."""
    return XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=random_state,
        n_jobs=-1,
    )


class GRUSingleOutput(nn.Module):
    """GRU that maps a multivariate input window to one rainfall value."""

    def __init__(
        self,
        input_size,
        hidden_size=128,
        num_layers=2,
        dropout=0.25,
        head_size=64,
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, head_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_size, 1),
        )

    def forward(self, x):
        _, hidden = self.gru(x)
        return self.head(self.norm(hidden[-1]))


def count_trainable_parameters(model):
    """Return the number of trainable model parameters."""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
