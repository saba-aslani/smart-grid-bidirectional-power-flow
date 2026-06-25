"""
Day-Ahead Economic Dispatch Optimization
=========================================
Optimizes the 24-hour generation schedule for a smart distribution grid
with mixed conventional + renewable + storage resources.

Problem formulation (Linear Program):
  Variables per hour t (24 hours):
    p_conv[t]   : conventional generator output (MW)
    p_wind[t]   : wind power dispatched (≤ forecast) (MW)
    p_pv[t]     : PV power dispatched   (≤ forecast) (MW)
    p_stor_c[t] : storage charging power (MW)
    p_stor_d[t] : storage discharging power (MW)
    s[t]        : storage state-of-charge (MWh)
    curt_w[t]   : wind curtailment (MW)
    curt_s[t]   : solar curtailment (MW)

  Objective: minimize  Σ_t [cost_conv(t) + cost_emission(t)
                              + penalty_curtailment(t)]

  Constraints:
    - Power balance: p_conv + p_wind + p_pv + p_stor_d
                     = demand + p_stor_c  (every hour)
    - Ramp-rate limits: |p_conv[t] - p_conv[t-1]| ≤ ramp_max
    - Conventional capacity: P_conv_min ≤ p_conv ≤ P_conv_max
    - Storage SoC dynamics: s[t] = s[t-1] + η_c*p_stor_c - p_stor_d/η_d
    - Storage SoC limits: SoC_min ≤ s[t] ≤ SoC_max
    - Storage power limits: 0 ≤ p_stor_c, p_stor_d ≤ P_stor_max
    - Curtailment: curt_w = p_wind_avail - p_wind, curt_s = p_pv_avail - p_pv
    - Reserve requirement: p_conv ≥ demand * reserve_fraction

Solver: scipy.optimize.linprog  (simplex / HiGHS)
"""

import numpy as np
from scipy.optimize import linprog
from dataclasses import dataclass


# ─────────────────────────────────────────────
# 1.  SYSTEM PARAMETERS
# ─────────────────────────────────────────────
@dataclass
class GridParams:
    # Conventional generator
    p_conv_max:    float = 4.0    # MW
    p_conv_min:    float = 0.5    # MW  (minimum stable generation)
    ramp_max:      float = 1.2    # MW/h
    cost_conv:     float = 85.0   # $/MWh   (fuel + O&M)
    cost_emission: float = 22.0   # $/MWh   (carbon price equivalent)

    # Renewables
    p_wind_max:    float = 2.0    # MW rated
    p_pv_max:      float = 1.8    # MW rated
    cost_wind:     float = 0.0    # $/MWh  (zero marginal cost)
    cost_pv:       float = 0.0    # $/MWh
    penalty_curt:  float = 40.0   # $/MWh  (curtailment penalty)

    # Battery storage
    p_stor_max:    float = 0.8    # MW charge/discharge
    soc_max:       float = 3.2    # MWh
    soc_min:       float = 0.32   # MWh  (10% min SoC)
    soc_init:      float = 1.6    # MWh
    eta_c:         float = 0.95   # charging efficiency
    eta_d:         float = 0.95   # discharging efficiency

    # Grid constraints
    reserve_frac:  float = 0.10   # 10% spinning reserve on conventional


def make_24h_profiles(rng_seed=0):
    """Generate representative 24h demand + renewable availability profiles."""
    rng = np.random.default_rng(rng_seed)
    t = np.arange(24)
    # Load profile (MW) — residential + commercial feeder
    morning = 1.8 * np.exp(-0.5 * ((t - 8)  / 1.5) ** 2)
    evening = 2.6 * np.exp(-0.5 * ((t - 19) / 2.0) ** 2)
    base    = 1.0
    demand  = base + morning + evening + rng.normal(0, 0.08, 24)
    demand  = np.clip(demand, 0.8, 3.8)

    # Wind availability (MW)
    wind_mean = 1.2 + 0.5 * np.sin(2 * np.pi * t / 24 + np.pi)
    wind_avail = (wind_mean + rng.normal(0, 0.25, 24)).clip(0, 2.0)

    # Solar availability (MW) — daytime bell
    sun = np.clip(np.sin(np.pi * (t - 6) / 12), 0, None)
    pv_avail = (1.5 * sun + rng.normal(0, 0.05, 24) * sun).clip(0, 1.8)

    return demand, wind_avail, pv_avail


# ─────────────────────────────────────────────
# 2.  LP FORMULATION AND SOLVE
# ─────────────────────────────────────────────
def build_and_solve_dispatch(demand, wind_avail, pv_avail,
                              params=None, verbose=False):
    """
    Formulate and solve the 24h day-ahead economic dispatch LP.

    Variable vector x layout (8 variables × T hours = 8T total):
      [p_conv(0..T-1), p_wind(0..T-1), p_pv(0..T-1),
       p_stor_c(0..T-1), p_stor_d(0..T-1), soc(0..T-1),
       curt_w(0..T-1), curt_s(0..T-1)]
    """
    if params is None:
        params = GridParams()

    T = 24
    assert len(demand) == T

    # index offsets
    I_conv   = slice(0*T, 1*T)
    I_wind   = slice(1*T, 2*T)
    I_pv     = slice(2*T, 3*T)
    I_storc  = slice(3*T, 4*T)
    I_stord  = slice(4*T, 5*T)
    I_soc    = slice(5*T, 6*T)
    I_curtw  = slice(6*T, 7*T)
    I_curts  = slice(7*T, 8*T)
    N = 8 * T

    # ── Objective ──────────────────────────────────────────────────────
    c = np.zeros(N)
    c[I_conv]  = params.cost_conv + params.cost_emission  # $/MWh
    c[I_curtw] = params.penalty_curt
    c[I_curts] = params.penalty_curt
    # storage has small cost to prevent unnecessary cycling
    c[I_storc] = 0.5
    c[I_stord] = 0.5

    # ── Bounds ─────────────────────────────────────────────────────────
    bounds = []
    for t in range(T):
        bounds.append((params.p_conv_min, params.p_conv_max))  # p_conv
    for t in range(T):
        bounds.append((0, wind_avail[t]))   # p_wind  ≤ available
    for t in range(T):
        bounds.append((0, pv_avail[t]))     # p_pv    ≤ available
    for t in range(T):
        bounds.append((0, params.p_stor_max))  # p_stor_c
    for t in range(T):
        bounds.append((0, params.p_stor_max))  # p_stor_d
    for t in range(T):
        bounds.append((params.soc_min, params.soc_max))  # soc
    for t in range(T):
        bounds.append((0, wind_avail[t]))  # curt_w  ≤ wind_avail
    for t in range(T):
        bounds.append((0, pv_avail[t]))    # curt_s  ≤ pv_avail

    # ── Equality constraints ────────────────────────────────────────────
    A_eq_rows, b_eq = [], []

    # Power balance: p_conv + p_wind + p_pv + p_stor_d - p_stor_c = demand
    for t in range(T):
        row = np.zeros(N)
        row[0*T + t] =  1   # p_conv
        row[1*T + t] =  1   # p_wind
        row[2*T + t] =  1   # p_pv
        row[3*T + t] = -1   # p_stor_c  (charging consumes power)
        row[4*T + t] =  1   # p_stor_d
        A_eq_rows.append(row)
        b_eq.append(demand[t])

    # SoC dynamics: soc[t] = soc[t-1] + eta_c*stor_c[t] - stor_d[t]/eta_d
    for t in range(T):
        row = np.zeros(N)
        row[5*T + t] = 1                      # soc[t]
        row[3*T + t] = -params.eta_c          # +charging
        row[4*T + t] =  1.0/params.eta_d      # -discharging
        if t > 0:
            row[5*T + (t-1)] = -1             # -soc[t-1]
            A_eq_rows.append(row)
            b_eq.append(0.0)
        else:
            row_init = row.copy()
            row_init[5*T + t] = 1
            # soc[0] - eta_c*storc[0] + stord[0]/eta_d = soc_init
            A_eq_rows.append(row_init)
            b_eq.append(params.soc_init)

    # Curtailment accounting: p_wind + curt_w = wind_avail
    for t in range(T):
        row = np.zeros(N)
        row[1*T + t] = 1
        row[6*T + t] = 1
        A_eq_rows.append(row)
        b_eq.append(wind_avail[t])

    # Curtailment accounting: p_pv + curt_s = pv_avail
    for t in range(T):
        row = np.zeros(N)
        row[2*T + t] = 1
        row[7*T + t] = 1
        A_eq_rows.append(row)
        b_eq.append(pv_avail[t])

    A_eq = np.array(A_eq_rows)
    b_eq = np.array(b_eq)

    # ── Inequality constraints ──────────────────────────────────────────
    A_ub_rows, b_ub = [], []

    # Ramp-rate up:   p_conv[t] - p_conv[t-1] ≤ ramp_max
    # Ramp-rate down: p_conv[t-1] - p_conv[t] ≤ ramp_max
    for t in range(1, T):
        r_up = np.zeros(N)
        r_up[0*T + t]     =  1
        r_up[0*T + (t-1)] = -1
        A_ub_rows.append(r_up);  b_ub.append(params.ramp_max)

        r_dn = np.zeros(N)
        r_dn[0*T + (t-1)] =  1
        r_dn[0*T + t]     = -1
        A_ub_rows.append(r_dn);  b_ub.append(params.ramp_max)

    # Reserve: p_conv[t] ≥ reserve_frac * demand[t]
    # => -p_conv[t] ≤ -reserve_frac * demand[t]
    for t in range(T):
        row = np.zeros(N)
        row[0*T + t] = -1
        A_ub_rows.append(row)
        b_ub.append(-params.reserve_frac * demand[t])

    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub)

    # ── Solve ───────────────────────────────────────────────────────────
    res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                  A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")

    if not res.success:
        raise RuntimeError(f"LP did not solve: {res.message}")

    x = res.x
    return {
        "status":     res.message,
        "total_cost": float(res.fun),
        "p_conv":     x[I_conv],
        "p_wind":     x[I_wind],
        "p_pv":       x[I_pv],
        "p_stor_c":   x[I_storc],
        "p_stor_d":   x[I_stord],
        "soc":        x[I_soc],
        "curt_wind":  x[I_curtw],
        "curt_solar": x[I_curts],
        "demand":     demand,
        "wind_avail": wind_avail,
        "pv_avail":   pv_avail,
        "params":     params,
    }


# ─────────────────────────────────────────────
# 3.  KPI CALCULATOR
# ─────────────────────────────────────────────
def compute_kpis(result):
    p = result["params"]
    conv_energy   = result["p_conv"].sum()
    re_dispatched = result["p_wind"].sum() + result["p_pv"].sum()
    curt_total    = result["curt_wind"].sum() + result["curt_solar"].sum()
    re_available  = result["wind_avail"].sum() + result["pv_avail"].sum()
    re_fraction   = re_dispatched / result["demand"].sum() * 100
    curt_fraction = curt_total / re_available * 100 if re_available > 0 else 0
    co2_kg        = conv_energy * 0.45 * 1000   # 0.45 tCO2/MWh → kg/h*h

    return {
        "total_cost_$":        result["total_cost"],
        "conv_energy_MWh":     conv_energy,
        "re_dispatched_MWh":   re_dispatched,
        "re_fraction_%":       re_fraction,
        "curtailment_MWh":     curt_total,
        "curtailment_%":       curt_fraction,
        "CO2_emissions_kg":    co2_kg,
        "storage_cycles":      float(result["p_stor_c"].sum()),
    }


# ─────────────────────────────────────────────
# 4.  STOCHASTIC DISPATCH (with forecast scenarios)
# ─────────────────────────────────────────────
def stochastic_dispatch(demand_24h, wind_scenarios, pv_scenarios,
                         params=None, n_scenarios=50, seed=1):
    """
    Run dispatch for N forecast scenarios and return distribution of KPIs.
    This connects the forecast module → economic dispatch in one call.
    """
    if params is None:
        params = GridParams()

    rng = np.random.default_rng(seed)
    # pick a 24-hour window from the scenario array
    T_start = rng.integers(0, max(1, wind_scenarios.shape[0] - 24))
    w_24h = wind_scenarios[T_start:T_start+24, :]
    s_24h = pv_scenarios[T_start:T_start+24, :]

    idx = rng.choice(w_24h.shape[1], size=n_scenarios, replace=False)
    kpi_list = []
    for i in idx:
        w = np.clip(w_24h[:, i] / 25 * params.p_wind_max, 0, params.p_wind_max)
        s = np.clip(s_24h[:, i]      * params.p_pv_max,   0, params.p_pv_max)
        try:
            res = build_and_solve_dispatch(demand_24h, w, s, params)
            kpi_list.append(compute_kpis(res))
        except RuntimeError:
            pass

    return kpi_list


if __name__ == "__main__":
    print("=" * 65)
    print("DAY-AHEAD ECONOMIC DISPATCH — DETERMINISTIC BASELINE")
    print("=" * 65)
    demand, wind_avail, pv_avail = make_24h_profiles(rng_seed=42)
    res = build_and_solve_dispatch(demand, wind_avail, pv_avail)
    kpi = compute_kpis(res)

    print(f"  Status         : {res['status']}")
    print(f"  Total cost     : ${kpi['total_cost_$']:.2f}")
    print(f"  Conventional   : {kpi['conv_energy_MWh']:.2f} MWh")
    print(f"  RE dispatched  : {kpi['re_dispatched_MWh']:.2f} MWh  "
          f"({kpi['re_fraction_%']:.1f}% of demand)")
    print(f"  Curtailment    : {kpi['curtailment_MWh']:.3f} MWh "
          f"({kpi['curtailment_%']:.1f}%)")
    print(f"  CO₂ emissions  : {kpi['CO2_emissions_kg']:.1f} kg")

    print("\n" + "=" * 65)
    print("STOCHASTIC DISPATCH — KPI DISTRIBUTION (50 forecast scenarios)")
    print("=" * 65)
    from advanced_forecasting import (generate_opsd_like_series,
                                       benchmark_all_models,
                                       generate_forecast_scenarios)
    df = generate_opsd_like_series(n_days=180)
    best = {c: min(benchmark_all_models(df[c]).values(),
                   key=lambda x: x["rmse"])
            for c in ["wind_speed", "solar"]}
    w_sc, s_sc = generate_forecast_scenarios(best["wind_speed"],
                                              best["solar"], n_scenarios=200)
    kpi_list = stochastic_dispatch(demand, w_sc, s_sc, n_scenarios=50)

    costs = np.array([k["total_cost_$"]      for k in kpi_list])
    re_fr = np.array([k["re_fraction_%"]      for k in kpi_list])
    curt  = np.array([k["curtailment_%"]      for k in kpi_list])
    co2   = np.array([k["CO2_emissions_kg"]   for k in kpi_list])

    for name, arr in [("Cost ($)",     costs),
                      ("RE share (%)", re_fr),
                      ("Curtail (%)",  curt),
                      ("CO₂ (kg)",     co2)]:
        print(f"  {name:<15}  P10={np.percentile(arr,10):>8.2f}  "
              f"P50={np.percentile(arr,50):>8.2f}  "
              f"P90={np.percentile(arr,90):>8.2f}")
