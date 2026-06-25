"""
ML surrogate model for the probabilistic power flow.
Learns the mapping (wind_speed, irradiance) -> (bus voltage magnitudes,
branch active-power flows) directly from Monte-Carlo training data,
bypassing the iterative BFS power-flow solve at prediction time.

This mirrors the "model-based deep learning probabilistic power flow"
approach used in recent literature (e.g. Yang et al., IEEE Trans. Smart
Grid 2020) -- here implemented with scikit-learn (MLP) since no GPU deep
learning framework is available in this sandbox, and benchmarked against
linear regression and random forest baselines.
"""

import time
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

from network import N_BUS, BRANCH_DATA, load_array_kw_kvar, backward_forward_sweep
from monte_carlo import run_mcs, WIND_BUS, PV_BUS


def build_dataset(mcs):
    X = np.column_stack([mcs["wind_speed"], mcs["irradiance"]])
    Y_v = mcs["V_mag"][:, 1:]                       # 32 bus voltages (bus 2..33) + bus1 (constant)
    Y_p = mcs["branch_P_pu"][:, 2:]                  # 32 branch flows
    Y = np.column_stack([Y_v, Y_p])
    return X, Y


def train_and_compare(n_samples=4000, seed=42):
    mcs = run_mcs(n_samples=n_samples, seed=seed)
    X, Y = build_dataset(mcs)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=seed)

    xscaler = StandardScaler().fit(X_train)
    Xtr, Xte = xscaler.transform(X_train), xscaler.transform(X_test)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=12, random_state=seed, n_jobs=-1),
        "MLP (neural net)": MLPRegressor(hidden_layer_sizes=(64, 64), activation='relu',
                                          max_iter=2000, random_state=seed, early_stopping=True),
    }

    results = {}
    for name, model in models.items():
        t0 = time.perf_counter()
        model.fit(Xtr, Y_train)
        train_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        pred = model.predict(Xte)
        pred_time = time.perf_counter() - t0

        rmse_v = np.sqrt(mean_squared_error(Y_test[:, :N_BUS - 1], pred[:, :N_BUS - 1]))
        mae_v = mean_absolute_error(Y_test[:, :N_BUS - 1], pred[:, :N_BUS - 1])
        results[name] = {
            "model": model, "train_time_s": train_time, "pred_time_s": pred_time,
            "rmse_voltage_pu": rmse_v, "mae_voltage_pu": mae_v,
            "pred_time_per_sample_ms": pred_time / len(X_test) * 1000,
        }

    # --- benchmark true BFS solve time per sample for comparison ---
    P_load, Q_load = load_array_kw_kvar()
    t0 = time.perf_counter()
    n_bench = 300
    for i in range(n_bench):
        der = np.zeros(N_BUS + 1)
        der[WIND_BUS] = mcs["p_wind_kw"][i]
        der[PV_BUS] = mcs["p_pv_kw"][i]
        backward_forward_sweep(P_load, Q_load, der_p_kw=der)
    bfs_time_per_sample_ms = (time.perf_counter() - t0) / n_bench * 1000

    return results, bfs_time_per_sample_ms, xscaler, mcs


if __name__ == "__main__":
    results, bfs_ms, scaler, mcs = train_and_compare(n_samples=4000)

    print(f"{'Model':<20} {'RMSE (pu)':>12} {'MAE (pu)':>12} {'Train(s)':>10} {'Predict(ms/sample)':>20}")
    for name, r in results.items():
        print(f"{name:<20} {r['rmse_voltage_pu']:>12.6f} {r['mae_voltage_pu']:>12.6f} "
              f"{r['train_time_s']:>10.3f} {r['pred_time_per_sample_ms']:>20.5f}")

    most_accurate = min(results, key=lambda k: results[k]['rmse_voltage_pu'])
    fastest = min(results, key=lambda k: results[k]['pred_time_per_sample_ms'])
    print(f"\nFull BFS power-flow solve: {bfs_ms:.4f} ms/sample")
    print(f"Most accurate surrogate: {most_accurate} "
          f"(RMSE={results[most_accurate]['rmse_voltage_pu']:.6f} pu, "
          f"{bfs_ms / results[most_accurate]['pred_time_per_sample_ms']:,.0f}x faster than BFS)")
    print(f"Fastest surrogate: {fastest} "
          f"(RMSE={results[fastest]['rmse_voltage_pu']:.6f} pu, "
          f"{bfs_ms / results[fastest]['pred_time_per_sample_ms']:,.0f}x faster than BFS)")
