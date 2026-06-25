"""
Monte-Carlo Probabilistic Power Flow (MCS-PPF) for the IEEE 33-bus system
with two DER sites (wind @ bus 18, PV @ bus 33), plus bidirectional
(reverse) power-flow detection on every branch.

Variance reduction: Latin Hypercube Sampling (LHS) on the two uncertain
inputs (wind speed, solar irradiance) instead of crude random sampling.
"""

import numpy as np
from scipy.stats import qmc

from network import (N_BUS, BRANCH_DATA, load_array_kw_kvar,
                      backward_forward_sweep, PARENT)
from der_models import WindTurbine, PVSystem

WIND_BUS = 18
PV_BUS = 33


def run_mcs(n_samples=3000, wind_rated_kw=2000.0, pv_rated_kw=1800.0, seed=42):
    rng = np.random.default_rng(seed)
    wt = WindTurbine(rated_kw=wind_rated_kw)
    pv = PVSystem(rated_kw=pv_rated_kw)

    # --- Latin Hypercube Sampling over [0,1]^2, mapped to each marginal's inverse CDF ---
    sampler = qmc.LatinHypercube(d=2, seed=seed)
    u = sampler.random(n=n_samples)  # uniform [0,1)^2

    from scipy.stats import weibull_min, beta as beta_dist
    wind_speed = weibull_min.ppf(u[:, 0], c=wt.k, scale=wt.c)
    irradiance = beta_dist.ppf(u[:, 1], pv.alpha, pv.beta) * pv.g_max

    p_wind = wt.power_kw(wind_speed)
    p_pv = pv.power_kw(irradiance)

    P_load, Q_load = load_array_kw_kvar()

    n_branches = len(BRANCH_DATA)
    V_mag = np.zeros((n_samples, N_BUS + 1))
    branch_P = np.zeros((n_samples, N_BUS + 1))  # indexed by 'to' bus (2..33)
    losses = np.zeros(n_samples)
    reverse_flags = np.zeros((n_samples, N_BUS + 1), dtype=bool)
    converged = np.zeros(n_samples, dtype=bool)

    for s in range(n_samples):
        der = np.zeros(N_BUS + 1)
        der[WIND_BUS] = p_wind[s]
        der[PV_BUS] = p_pv[s]
        res = backward_forward_sweep(P_load, Q_load, der_p_kw=der)
        converged[s] = res["converged"]
        V_mag[s, :] = np.abs(res["V"])
        losses[s] = res["loss_kw"]
        for b in range(2, N_BUS + 1):
            sflow = res["Sline"][b]
            p_flow = sflow.real  # pu, sending-end (parent -> b) active power
            branch_P[s, b] = p_flow
            # "reverse" = power flowing from child toward parent (i.e. up the
            # feeder, opposite of the normal substation-to-load direction)
            reverse_flags[s, b] = p_flow < 0

    return {
        "n_samples": n_samples,
        "p_wind_kw": p_wind, "p_pv_kw": p_pv,
        "wind_speed": wind_speed, "irradiance": irradiance,
        "V_mag": V_mag, "branch_P_pu": branch_P, "losses_kw": losses,
        "reverse_flags": reverse_flags, "converged": converged,
        "wind_rated_kw": wind_rated_kw, "pv_rated_kw": pv_rated_kw,
    }


def summarize_bidirectional(mcs):
    """Per-branch probability of reverse power flow + substation-level summary."""
    n = mcs["n_samples"]
    prob_reverse = mcs["reverse_flags"][:, 2:].mean(axis=0)  # per branch (bus 2..33)
    lines = []
    for idx, (f, t, r, x) in enumerate(BRANCH_DATA):
        lines.append((f, t, prob_reverse[idx]))
    sub_reverse_prob = mcs["reverse_flags"][:, PARENT_OF(2)] if False else None
    return lines


def PARENT_OF(b):
    return PARENT[b]


if __name__ == "__main__":
    mcs = run_mcs(n_samples=3000)
    print(f"All {mcs['n_samples']} scenarios converged: {mcs['converged'].all()}")
    print(f"Wind power: mean={mcs['p_wind_kw'].mean():.1f} kW "
          f"(rated {mcs['wind_rated_kw']} kW)")
    print(f"PV power:   mean={mcs['p_pv_kw'].mean():.1f} kW "
          f"(rated {mcs['pv_rated_kw']} kW)")
    print(f"Voltage range across all buses/scenarios: "
          f"{mcs['V_mag'][:,1:].min():.4f} - {mcs['V_mag'][:,1:].max():.4f} pu")
    print(f"Loss range: {mcs['losses_kw'].min():.1f} - {mcs['losses_kw'].max():.1f} kW "
          f"(base case no-DER loss = 202.68 kW)")

    lines = summarize_bidirectional(mcs)
    reverse_lines = [(f, t, p) for f, t, p in lines if p > 0.001]
    print(f"\nBranches with non-zero probability of reverse power flow:")
    for f, t, p in sorted(reverse_lines, key=lambda x: -x[2]):
        print(f"  Branch {f}->{t}: P(reverse) = {p:.1%}")

    # substation (branch 1->2) reverse flow probability
    sub_prob = mcs["reverse_flags"][:, 2].mean()
    print(f"\nSubstation feeder-head (1->2) reverse-flow probability: {sub_prob:.2%}")
