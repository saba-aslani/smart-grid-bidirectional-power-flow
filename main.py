"""
Run the full Smart-Grid Bidirectional Power Flow pipeline end-to-end:
  1. Validate the base-case power flow against published benchmarks
  2. Monte-Carlo probabilistic power flow (LHS) with wind+PV DER
  3. Bidirectional (reverse) power-flow analysis per branch
  4. ML surrogate model benchmark (accuracy vs. speed)
  5. RL DER-curtailment agent (training + evaluation)
  6. Short-term load/wind/PV forecasting benchmark
  7. Save all figures to results/

Usage:  python main.py
"""

import sys
import time

sys.path.insert(0, "src")

from network import load_array_kw_kvar, backward_forward_sweep
import numpy as np


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main():
    t_start = time.time()

    section("1. BASE-CASE VALIDATION (IEEE 33-bus, no DER)")
    P, Q = load_array_kw_kvar()
    res = backward_forward_sweep(P, Q)
    vmag = np.abs(res["V"][1:])
    print(f"Converged in {res['iterations']} iterations")
    print(f"Min voltage: {vmag.min():.4f} pu at bus {int(np.argmin(vmag)) + 1}  "
          f"(reference: 0.9131 pu @ bus 18, Baran & Wu 1989)")
    print(f"Total loss: {res['loss_kw']:.2f} kW  (reference: ~202.7 kW)")

    section("2. MONTE-CARLO PROBABILISTIC POWER FLOW + BIDIRECTIONAL FLOW ANALYSIS")
    from monte_carlo import run_mcs, summarize_bidirectional
    mcs = run_mcs(n_samples=3000)
    print(f"{mcs['n_samples']} scenarios, all converged: {mcs['converged'].all()}")
    print(f"Voltage range: {mcs['V_mag'][:,1:].min():.4f} - {mcs['V_mag'][:,1:].max():.4f} pu")
    lines = summarize_bidirectional(mcs)
    n_reverse_branches = sum(1 for f, t, p in lines if p > 0.01)
    print(f"Branches with >1% probability of reverse flow: {n_reverse_branches} / {len(lines)}")
    print(f"Substation feeder-head reverse-flow probability: "
          f"{mcs['reverse_flags'][:,2].mean():.2%}")

    section("3. ML SURROGATE MODEL (accuracy vs. speed benchmark)")
    from surrogate import train_and_compare
    results, bfs_ms, scaler, _ = train_and_compare(n_samples=4000)
    for name, r in results.items():
        print(f"  {name:<20} RMSE={r['rmse_voltage_pu']:.6f} pu   "
              f"{r['pred_time_per_sample_ms']*1000:.2f} us/sample")
    print(f"  Full BFS solve: {bfs_ms*1000:.2f} us/sample")

    section("4. RL DER-CURTAILMENT AGENT")
    from rl_control import train_q_learning, evaluate_policy
    Q_table, env, hist = train_q_learning(n_episodes=20000)
    rl, base, n = evaluate_policy(Q_table, env, n_eval=3000)
    print(f"Overvoltage rate:  no-control={base['over']/n:.1%}  "
          f"RL={rl['over']/n:.1%}")
    print(f"Renewable energy curtailed by RL policy: "
          f"{rl['curtailed_energy_kwh']/rl['total_energy_kwh']:.1%}")

    section("5. SHORT-TERM FORECASTING (load / wind / irradiance)")
    from forecasting import generate_synthetic_series, train_forecaster
    df = generate_synthetic_series(n_days=90)
    for col in ["load", "irradiance", "wind"]:
        r = train_forecaster(df, col)
        skill = (1 - r["rmse"] / r["rmse_naive_persistence"]) * 100
        print(f"  {col:<12} RMSE={r['rmse']:.4f}  skill vs persistence: {skill:+.1f}%")

    section("6. GENERATING FIGURES")
    import plots
    plots.plot_voltage_profile(mcs)
    plots.plot_reverse_flow_prob(mcs)
    plots.plot_network_diagram(mcs)
    plots.plot_surrogate_accuracy()
    plots.plot_rl_training()
    plots.plot_forecast_example()
    print("Saved 6 figures to results/")

    print(f"\nTotal runtime: {time.time() - t_start:.1f} s")


if __name__ == "__main__":
    main()
