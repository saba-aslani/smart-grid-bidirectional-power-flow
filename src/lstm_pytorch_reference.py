"""
REFERENCE IMPLEMENTATION -- NOT EXECUTED IN THIS SANDBOX.

This sandbox has no PyTorch/TensorFlow installed and no internet access to
install one, so this file could not be run or validated here. The trained
and validated forecaster actually shipped with this project is the
feed-forward MLPRegressor in forecasting.py.

This script is a drop-in true-LSTM replacement: same synthetic-series
generator and lag-window framing, but using a real recurrent network.
Run it locally (`pip install torch`) to compare against the MLP baseline,
e.g. on a real measured dataset once you have one.
"""

import numpy as np
import torch
import torch.nn as nn
from forecasting import generate_synthetic_series, make_lag_features


class LSTMForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, 1)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


def train_lstm(series, n_lags=24, test_days=10, epochs=60, lr=1e-3, hidden_size=32):
    X, y = make_lag_features(series, n_lags)
    n_test = test_days * 24
    X_train, X_test = X[:-n_test], X[-n_test:]
    y_train, y_test = y[:-n_test], y[-n_test:]

    Xtr = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
    ytr = torch.tensor(y_train, dtype=torch.float32)
    Xte = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
    yte = torch.tensor(y_test, dtype=torch.float32)

    model = LSTMForecaster(hidden_size=hidden_size)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        pred = model(Xtr)
        loss = loss_fn(pred, ytr)
        loss.backward()
        opt.step()
        if epoch % 10 == 0:
            print(f"epoch {epoch:3d}  train MSE={loss.item():.5f}")

    model.eval()
    with torch.no_grad():
        pred_test = model(Xte)
        test_rmse = torch.sqrt(loss_fn(pred_test, yte)).item()
    print(f"Test RMSE (LSTM): {test_rmse:.4f}")
    return model, test_rmse


if __name__ == "__main__":
    df = generate_synthetic_series(n_days=90)
    for col in ["load", "irradiance", "wind"]:
        print(f"\n=== {col} ===")
        train_lstm(df[col].values)
