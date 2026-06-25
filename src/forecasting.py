"""
Short-term forecasting module for load / wind / solar, feeding day-ahead
inputs into the probabilistic bidirectional power-flow pipeline.

IMPORTANT, stated honestly: a recurrent network (LSTM/GRU) is the natural
choice for this task, but this sandbox has no PyTorch/TensorFlow available
and no internet access to install one. What is implemented and actually
trained/validated here is a feed-forward neural network (MLPRegressor)
over a sliding lag-window of past observations -- a standard, legitimate
autoregressive NN forecaster, but NOT a true recurrent LSTM. A ready-to-run
PyTorch LSTM version is provided separately (lstm_pytorch_reference.py) for
the user to run in an environment with PyTorch installed; it is NOT
executed or validated in this sandbox.

No real measured dataset was available offline, so synthetic but
physically-motivated hourly time series are generated (diurnal solar bell
curve with cloud-cover noise, autocorrelated wind speed, double-peak load
profile with weekday/weekend pattern). This is clearly a methodology
placeholder -- a real deployment / publication would replace this with a
measured dataset (e.g. an open utility/ISO dataset).
"""

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error


def generate_synthetic_series(n_days=90, seed=7):
    rng = np.random.default_rng(seed)
    hours = np.arange(n_days * 24)
    hod = hours % 24
    dow = (hours // 24) % 7
    is_weekend = (dow >= 5).astype(float)

    # --- Load: double-peak (morning + evening), lower on weekends ---
    morning = 0.35 * np.exp(-0.5 * ((hod - 8) / 2.0) ** 2)
    evening = 0.55 * np.exp(-0.5 * ((hod - 19) / 2.5) ** 2)
    base = 0.45
    load = base + morning + evening - 0.15 * is_weekend
    load += rng.normal(0, 0.03, size=len(hours))
    load = np.clip(load, 0.15, None)

    # --- Solar irradiance: daylight bell curve x day-level cloud factor ---
    daylight = np.clip(np.sin(np.pi * (hod - 6) / 12.0), 0, None)  # 0 at night, peak at 12:00
    cloud_factor_daily = np.clip(rng.normal(0.8, 0.25, size=n_days), 0.1, 1.15)
    cloud_factor = np.repeat(cloud_factor_daily, 24)
    irradiance = daylight * cloud_factor
    irradiance += rng.normal(0, 0.02, size=len(hours)) * daylight  # passing-cloud noise
    irradiance = np.clip(irradiance, 0, None)

    # --- Wind speed: AR(1) process (synoptic-scale persistence) ---
    wind = np.zeros(len(hours))
    wind[0] = 8.0
    phi = 0.92
    sigma_eps = 1.3
    mean_wind = 8.0
    for t in range(1, len(hours)):
        wind[t] = mean_wind + phi * (wind[t - 1] - mean_wind) + rng.normal(0, sigma_eps)
    wind = np.clip(wind, 0, 25)

    df = pd.DataFrame({"hour": hours, "load": load, "irradiance": irradiance, "wind": wind})
    return df


def make_lag_features(series, n_lags=24):
    X, y = [], []
    for t in range(n_lags, len(series)):
        X.append(series[t - n_lags:t])
        y.append(series[t])
    return np.array(X), np.array(y)


def train_forecaster(df, target_col, n_lags=24, test_days=10):
    series = df[target_col].values
    X, y = make_lag_features(series, n_lags)
    n_test = test_days * 24
    X_train, X_test = X[:-n_test], X[-n_test:]
    y_train, y_test = y[:-n_test], y[-n_test:]

    scaler = StandardScaler().fit(X_train)
    Xtr, Xte = scaler.transform(X_train), scaler.transform(X_test)

    model = MLPRegressor(hidden_layer_sizes=(48, 24), activation='tanh',
                          max_iter=3000, random_state=0, early_stopping=True)
    model.fit(Xtr, y_train)
    pred = model.predict(Xte)
    pred = np.clip(pred, 0, None)

    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mae = mean_absolute_error(y_test, pred)
    nrmse = rmse / (series.max() - series.min())

    # naive persistence baseline (yesterday-same-hour) for comparison
    persistence_pred = series[-n_test - 24:-24]
    rmse_naive = np.sqrt(mean_squared_error(y_test, persistence_pred))

    return {
        "model": model, "scaler": scaler, "y_test": y_test, "pred": pred,
        "rmse": rmse, "mae": mae, "nrmse": nrmse, "rmse_naive_persistence": rmse_naive,
    }


if __name__ == "__main__":
    df = generate_synthetic_series(n_days=90)
    print(f"Generated {len(df)} hourly samples ({len(df)//24} days) of synthetic "
          f"load/wind/irradiance data.\n")

    for col in ["load", "irradiance", "wind"]:
        r = train_forecaster(df, col)
        skill = (1 - r["rmse"] / r["rmse_naive_persistence"]) * 100
        print(f"[{col:>10}] MLP forecaster RMSE={r['rmse']:.4f}  NRMSE={r['nrmse']:.2%}  "
              f"| naive persistence RMSE={r['rmse_naive_persistence']:.4f}  "
              f"| skill vs. persistence: {skill:+.1f}%")
