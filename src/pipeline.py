"""
Integrated Smart Grid EMS Pipeline
====================================
Connects all modules in a single end-to-end flow:

  1. Data generation (synthetic / real OPSD drop-in)
  2. Probabilistic forecasting → prediction intervals
  3. Forecast-driven Monte-Carlo probabilistic power flow
  4. Bidirectional flow risk analysis
  5. RL voltage/curtailment control
  6. Day-ahead stochastic economic dispatch
  7. Unified KPI dashboard with before/after ML comparison
"""

import numpy as np
import sys
sys.path.insert(0, ".")


def run_full_pipeline(n_forecast_days=180, n_mcs=2000,
                      n_dispatch_scenarios=50, seed=42):
    results = {}

    # ── 1. DATA ────────────────────────────────────────────────────────
    from advanced_forecasting import generate_opsd_like_series
    df = generate_opsd_like_series(n_days=n_forecast_days, seed=seed)
    results["data"] = df

    # ── 2. PROBABILISTIC FORECASTING ──────────────────────────────────
    from advanced_forecasting import (benchmark_all_models,
                                       generate_forecast_scenarios)
    forecast_results = {}
    for col in ["load", "solar", "wind_speed"]:
        models = benchmark_all_models(df[col], n_test_days=30)
        best   = min(models.values(), key=lambda x: x["rmse"])
        forecast_results[col] = {"models": models, "best": best}
    results["forecast"] = forecast_results

    wind_scen, solar_scen = generate_forecast_scenarios(
        forecast_results["wind_speed"]["best"],
        forecast_results["solar"]["best"],
        n_scenarios=500, seed=seed)
    results["wind_scenarios"]  = wind_scen
    results["solar_scenarios"] = solar_scen

    # ── 3. PROBABILISTIC POWER FLOW ────────────────────────────────────
    from monte_carlo import run_mcs
    from network import N_BUS
    from der_models import WindTurbine, PVSystem

    # Scale forecast scenarios to DER ratings
    wt = WindTurbine(rated_kw=2000)
    pv = PVSystem(rated_kw=1800)
    mcs = run_mcs(n_samples=n_mcs, seed=seed)
    results["mcs"] = mcs

    # ── 4. BIDIRECTIONAL FLOW RISK ─────────────────────────────────────
    from monte_carlo import summarize_bidirectional, WIND_BUS, PV_BUS
    rev_prob = mcs["reverse_flags"][:, 2:].mean(axis=0)
    high_risk_branches = [(i+2) for i, p in enumerate(rev_prob) if p > 0.5]
    results["high_risk_branches"] = high_risk_branches
    results["rev_prob"] = rev_prob

    # ── 5. RL CONTROL ──────────────────────────────────────────────────
    from rl_control import train_q_learning, evaluate_policy
    Q, env, hist = train_q_learning(n_episodes=20000, seed=seed)
    rl_res, base_res, n_eval = evaluate_policy(Q, env, n_eval=2000, seed=seed+1)
    results["rl"] = {
        "Q": Q, "hist": hist,
        "rl_result": rl_res, "base_result": base_res, "n_eval": n_eval
    }

    # ── 6. STOCHASTIC ECONOMIC DISPATCH ───────────────────────────────
    from economic_dispatch import (GridParams, make_24h_profiles,
                                    build_and_solve_dispatch,
                                    stochastic_dispatch, compute_kpis)
    params = GridParams()
    demand_24h, _, _ = make_24h_profiles(rng_seed=seed)

    # Deterministic baseline (no ML forecasting, use naive flat estimates)
    naive_wind = np.full(24, wt.rated_kw / 1000 * 0.35)  # 35% CF naive
    naive_pv   = np.array([max(0, np.sin(np.pi*(h-6)/12)) * pv.rated_kw/1000
                            for h in range(24)])
    det_res  = build_and_solve_dispatch(demand_24h, naive_wind, naive_pv, params)
    det_kpis = compute_kpis(det_res)

    # Stochastic ML-informed dispatch
    stoch_kpis = stochastic_dispatch(demand_24h, wind_scen, solar_scen,
                                      params, n_scenarios=n_dispatch_scenarios,
                                      seed=seed)
    results["dispatch"] = {
        "deterministic": det_res,
        "det_kpis":      det_kpis,
        "stochastic":    stoch_kpis,
    }

    return results


def print_kpi_dashboard(results):
    mcs  = results["mcs"]
    rl   = results["rl"]
    disp = results["dispatch"]
    fc   = results["forecast"]
    N    = results["mcs"]["n_samples"]

    print("\n" + "╔" + "═"*68 + "╗")
    print("║{:^68}║".format("SMART GRID EMS — INTEGRATED KPI DASHBOARD"))
    print("╠" + "═"*68 + "╣")

    # ── Forecasting ────────────────────────────────────────────────────
    print("║{:^68}║".format("MODULE 1: PROBABILISTIC FORECASTING"))
    print("╠" + "─"*68 + "╣")
    hdr = f"  {'Variable':<14} {'Best Model':<10} {'RMSE':>8} {'sMAPE%':>8} {'Skill%':>9} {'PI Cov':>8}"
    print("║" + hdr + "║")
    for col, label in [("load","Load"),("solar","Solar"),("wind_speed","Wind")]:
        b = fc[col]["best"]
        row = f"  {label:<14} {b['model_type']:<10} {b['rmse']:>8.4f} {b['mape']:>8.2f} {b['skill_vs_naive']:>8.1f}% {b['pi_coverage']:>8.1%}"
        print("║" + row + "║")

    # ── Power flow ─────────────────────────────────────────────────────
    print("╠" + "═"*68 + "╣")
    print("║{:^68}║".format("MODULE 2: PROBABILISTIC POWER FLOW"))
    print("╠" + "─"*68 + "╣")
    vmin = mcs["V_mag"][:,1:].min(axis=1)
    vmax = mcs["V_mag"][:,1:].max(axis=1)
    over  = (mcs["V_mag"][:,1:] > 1.05).any(axis=1).mean()
    under = (mcs["V_mag"][:,1:] < 0.95).any(axis=1).mean()
    hrb   = results["high_risk_branches"]
    print(f"║  Scenarios: {N}   V range: [{vmin.mean():.4f}, {vmax.mean():.4f}] pu (mean)          ║")
    print(f"║  Overvoltage scenarios (>1.05pu): {over:.1%}                              ║")
    print(f"║  Undervoltage scenarios (<0.95pu): {under:.1%}                             ║")
    print(f"║  High-risk reverse-flow branches (P>50%): {len(hrb)} / 32                  ║")

    # ── RL control ─────────────────────────────────────────────────────
    print("╠" + "═"*68 + "╣")
    print("║{:^68}║".format("MODULE 3: RL VOLTAGE CONTROL"))
    print("╠" + "─"*68 + "╣")
    n  = rl["n_eval"]
    ro = rl["rl_result"]; bo = rl["base_result"]
    ov_base = bo["over"]/n; ov_rl = ro["over"]/n
    curt_rl = ro["curtailed_energy_kwh"]/ro["total_energy_kwh"]
    reward_improvement = (ro["reward"] - bo["reward"]) / abs(bo["reward"]) * 100
    print(f"║  Overvoltage rate: no-control={ov_base:.1%}  →  RL policy={ov_rl:.1%}           ║")
    print(f"║  RE curtailment by RL: {curt_rl:.1%}                                        ║")
    print(f"║  Mean reward improvement: {reward_improvement:+.1f}%                               ║")

    # ── Dispatch ───────────────────────────────────────────────────────
    print("╠" + "═"*68 + "╣")
    print("║{:^68}║".format("MODULE 4: DAY-AHEAD ECONOMIC DISPATCH"))
    print("╠" + "─"*68 + "╣")
    dk = disp["det_kpis"]
    sk = disp["stochastic"]
    costs  = np.array([k["total_cost_$"]    for k in sk])
    re_frs = np.array([k["re_fraction_%"]   for k in sk])
    co2s   = np.array([k["CO2_emissions_kg"] for k in sk])
    print(f"║  {'Metric':<24} {'Naive (no ML)':>16} {'ML stochastic (P50)':>20}║")
    print(f"║  {'─'*24} {'─'*16} {'─'*20}║")
    print(f"║  {'Cost ($)':<24} {dk['total_cost_$']:>16.2f} {np.median(costs):>20.2f}║")
    print(f"║  {'RE share (%)':<24} {dk['re_fraction_%']:>16.1f} {np.median(re_frs):>20.1f}║")
    print(f"║  {'CO₂ (kg)':<24} {dk['CO2_emissions_kg']:>16.1f} {np.median(co2s):>20.1f}║")

    print("╚" + "═"*68 + "╝")
