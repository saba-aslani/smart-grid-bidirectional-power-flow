"""
Comprehensive Dashboard Plots for the Smart Grid EMS Pipeline
"""
import sys
sys.path.insert(0, ".")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import pandas as pd

OUT = "/home/claude/sgbpf/results"


def plot_forecast_intervals(results):
    """4-panel: actual vs. forecast + PI for load/solar/wind + scenario fan."""
    fc = results["forecast"]
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=False)

    for ax, (col, label, unit) in zip(axes, [
        ("load",       "Load",        "(normalised MW)"),
        ("solar",      "Solar",       "(irradiance, kW/m²)"),
        ("wind_speed", "Wind Speed",  "(m/s)"),
    ]):
        b   = fc[col]["best"]
        y_t = b["y_test"].values
        hrs = np.arange(len(y_t))
        ax.fill_between(hrs, b["lower_10"], b["upper_90"],
                        alpha=0.25, color="#2980b9", label="80% PI")
        # 25-75 PI
        if 0.25 in b["preds"] and 0.75 in b["preds"]:
            ax.fill_between(hrs, b["preds"][0.25], b["preds"][0.75],
                            alpha=0.35, color="#2980b9", label="50% PI")
        ax.plot(hrs, y_t,      color="#2c3e50", lw=1.4, label="Actual")
        ax.plot(hrs, b["y_hat"], color="#e74c3c", lw=1.1,
                linestyle="--", label="Forecast (median)")
        ax.set_ylabel(f"{label} {unit}", fontsize=9)
        ax.set_title(f"{label} — {b['model_type'].upper()} | "
                     f"RMSE={b['rmse']:.4f}  Skill={b['skill_vs_naive']:+.1f}%  "
                     f"PI coverage={b['pi_coverage']:.1%}", fontsize=9)
        ax.legend(loc="upper right", fontsize=7, ncol=4)
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Hour (held-out test period, 30 days)")
    fig.suptitle("Probabilistic Forecasting — Load / Solar / Wind  "
                 "(80% and 50% prediction intervals)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(f"{OUT}/forecast_intervals.png", dpi=150)
    plt.close(fig)


def plot_dispatch_schedule(results):
    """24h stacked generation schedule + SoC + curtailment."""
    res = results["dispatch"]["deterministic"]
    t   = np.arange(24)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9),
                              gridspec_kw={"height_ratios": [3, 1.2, 1]})

    # ── Generation stack ──
    ax = axes[0]
    ax.stackplot(t,
                 res["p_conv"], res["p_wind"], res["p_pv"], res["p_stor_d"],
                 labels=["Conventional", "Wind", "PV", "Storage discharge"],
                 colors=["#7f8c8d", "#3498db", "#f1c40f", "#2ecc71"],
                 alpha=0.85)
    ax.plot(t, res["demand"], "k--", lw=2.0, label="Demand")
    ax.plot(t, res["wind_avail"], "b:", lw=1.2, alpha=0.6, label="Wind available")
    ax.plot(t, res["pv_avail"],   "y:", lw=1.2, alpha=0.6, label="PV available")
    ax.set_ylabel("Power (MW)")
    ax.set_title("Day-Ahead Economic Dispatch — Optimal Generation Schedule",
                 fontweight="bold")
    ax.legend(loc="upper left", fontsize=8, ncol=3)
    ax.grid(alpha=0.3)

    # ── Storage SoC ──
    ax2 = axes[1]
    ax2.fill_between(t, res["soc"], alpha=0.5, color="#9b59b6")
    ax2.plot(t, res["soc"], color="#9b59b6", lw=1.5)
    ax2.axhline(res["params"].soc_max, color="r", linestyle=":", lw=1)
    ax2.axhline(res["params"].soc_min, color="r", linestyle=":", lw=1)
    ax2.set_ylabel("Storage SoC (MWh)")
    ax2.set_ylim(0, res["params"].soc_max * 1.15)
    ax2.grid(alpha=0.3)

    # ── Curtailment ──
    ax3 = axes[2]
    ax3.bar(t - 0.2, res["curt_wind"],  0.4,
            color="#3498db", alpha=0.7, label="Wind curtailed")
    ax3.bar(t + 0.2, res["curt_solar"], 0.4,
            color="#f1c40f", alpha=0.7, label="Solar curtailed")
    ax3.set_ylabel("Curtailment (MW)")
    ax3.set_xlabel("Hour of day")
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)

    for ax in axes:
        ax.set_xlim(-0.5, 23.5)
        ax.set_xticks(range(24))

    fig.tight_layout()
    fig.savefig(f"{OUT}/dispatch_schedule.png", dpi=150)
    plt.close(fig)


def plot_dispatch_stochastic(results):
    """KPI distributions from stochastic dispatch scenarios."""
    sk = results["dispatch"]["stochastic"]
    dk = results["dispatch"]["det_kpis"]
    if not sk:
        return

    costs  = np.array([k["total_cost_$"]      for k in sk])
    re_frs = np.array([k["re_fraction_%"]      for k in sk])
    co2s   = np.array([k["CO2_emissions_kg"]   for k in sk])
    curt   = np.array([k["curtailment_%"]       for k in sk])

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    specs = [
        (axes[0,0], costs,  f"Total Cost ($)\n(naive={dk['total_cost_$']:.0f})",     "#e74c3c"),
        (axes[0,1], re_frs, f"RE Share (%)\n(naive={dk['re_fraction_%']:.1f}%)",     "#27ae60"),
        (axes[1,0], co2s,   f"CO₂ Emissions (kg)\n(naive={dk['CO2_emissions_kg']:.0f})", "#95a5a6"),
        (axes[1,1], curt,   f"Curtailment (%)\n(naive={dk['curtailment_%']:.1f}%)",  "#f39c12"),
    ]
    for ax, data, title, color in specs:
        ax.hist(data, bins=20, color=color, alpha=0.75, edgecolor="white")
        ax.axvline(np.median(data), color="navy",  lw=2, linestyle="-",
                   label=f"P50={np.median(data):.1f}")
        ax.axvline(np.percentile(data, 10), color="navy", lw=1.2,
                   linestyle=":", label=f"P10={np.percentile(data,10):.1f}")
        ax.axvline(np.percentile(data, 90), color="navy", lw=1.2,
                   linestyle="--", label=f"P90={np.percentile(data,90):.1f}")
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    fig.suptitle("Stochastic Dispatch KPI Distributions — "
                 "ML Forecast Scenarios vs. Naive Baseline",
                 fontweight="bold", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f"{OUT}/dispatch_stochastic.png", dpi=150)
    plt.close(fig)


def plot_full_kpi_summary(results):
    """Single-page 6-panel summary of the entire pipeline."""
    fig = plt.figure(figsize=(16, 11))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38)

    mcs  = results["mcs"]
    rl   = results["rl"]
    disp = results["dispatch"]
    fc   = results["forecast"]
    N    = mcs["n_samples"]
    from network import N_BUS, BRANCH_DATA

    # ── 1. Voltage profile ──
    ax1 = fig.add_subplot(gs[0, :2])
    buses = np.arange(1, N_BUS+1)
    vmed = np.median(mcs["V_mag"][:,1:], axis=0)
    vp10 = np.percentile(mcs["V_mag"][:,1:], 10, axis=0)
    vp90 = np.percentile(mcs["V_mag"][:,1:], 90, axis=0)
    ax1.fill_between(buses, vp10, vp90, alpha=0.3, color="#3498db", label="10–90% range")
    ax1.plot(buses, vmed, color="#2980b9", lw=1.5, label="Median")
    ax1.axhline(1.05, color="r", linestyle="--", lw=1, label="Limits")
    ax1.axhline(0.95, color="r", linestyle="--", lw=1)
    ax1.set_xlabel("Bus"); ax1.set_ylabel("Voltage (pu)")
    ax1.set_title("Probabilistic Voltage Profile (IEEE 33-bus)", fontsize=9, fontweight="bold")
    ax1.legend(fontsize=7); ax1.grid(alpha=0.3)

    # ── 2. Bidirectional flow probability ──
    ax2 = fig.add_subplot(gs[0, 2])
    rev_p = mcs["reverse_flags"][:, 2:].mean(axis=0)
    order = np.argsort(rev_p)[-15:]
    labels_short = [f"{BRANCH_DATA[i][0]}→{BRANCH_DATA[i][1]}" for i in order]
    colors_b = ["#c0392b" if p > 0.5 else "#e67e22" for p in rev_p[order]]
    ax2.barh(range(15), rev_p[order]*100, color=colors_b)
    ax2.set_yticks(range(15)); ax2.set_yticklabels(labels_short, fontsize=7)
    ax2.set_xlabel("P(reverse flow) %")
    ax2.set_title("Top 15 Reverse-Flow\nRisk Branches", fontsize=9, fontweight="bold")
    ax2.grid(alpha=0.3, axis="x")

    # ── 3. RL training curve ──
    ax3 = fig.add_subplot(gs[1, 0])
    w = 300
    smooth = pd.Series(rl["hist"]).rolling(w).mean()
    ax3.plot(smooth, color="#8e44ad", lw=1.2)
    ax3.set_xlabel("Episode"); ax3.set_ylabel("Reward (smoothed)")
    ax3.set_title("RL Agent Training Curve", fontsize=9, fontweight="bold")
    ax3.grid(alpha=0.3)

    # ── 4. RL comparison bar ──
    ax4 = fig.add_subplot(gs[1, 1])
    n_eval = rl["n_eval"]
    ov_base = rl["base_result"]["over"] / n_eval * 100
    ov_rl   = rl["rl_result"]["over"]  / n_eval * 100
    curt_rl = rl["rl_result"]["curtailed_energy_kwh"] / rl["rl_result"]["total_energy_kwh"] * 100
    x = np.array([0, 1.5])
    ax4.bar(x[0]-0.2, ov_base, 0.35, color="#c0392b", label="No control")
    ax4.bar(x[0]+0.2, ov_rl,   0.35, color="#27ae60", label="RL policy")
    ax4.bar(x[1],     curt_rl, 0.35, color="#f39c12")
    ax4.set_xticks([0, 1.5])
    ax4.set_xticklabels(["Overvoltage\nrate (%)", "Curtailment\n(%)"], fontsize=8)
    ax4.legend(fontsize=7); ax4.set_title("RL Control Impact", fontsize=9, fontweight="bold")
    ax4.grid(alpha=0.3, axis="y")

    # ── 5. Dispatch schedule (mini) ──
    ax5 = fig.add_subplot(gs[1, 2])
    res = disp["deterministic"]
    t   = np.arange(24)
    ax5.stackplot(t, res["p_conv"], res["p_wind"], res["p_pv"],
                  colors=["#7f8c8d","#3498db","#f1c40f"], alpha=0.85,
                  labels=["Conv","Wind","PV"])
    ax5.plot(t, res["demand"], "k--", lw=1.5, label="Demand")
    ax5.set_xlabel("Hour"); ax5.set_ylabel("MW")
    ax5.set_title("24h Dispatch Schedule", fontsize=9, fontweight="bold")
    ax5.legend(fontsize=6, ncol=2); ax5.grid(alpha=0.3)

    # ── 6. Forecast quality ──
    ax6 = fig.add_subplot(gs[2, :])
    model_names = ["linear", "gbm", "rf", "mlp"]
    x_pos = np.arange(len(model_names))
    w_bar = 0.22
    colors_m = ["#3498db","#e74c3c","#2ecc71","#f39c12"]
    for ci, (col, lbl) in enumerate([("load","Load"),("solar","Solar"),
                                      ("wind_speed","Wind")]):
        skills = [fc[col]["models"][m]["skill_vs_naive"] for m in model_names]
        ax6.bar(x_pos + ci*w_bar, skills, w_bar,
                color=colors_m[ci], alpha=0.8, label=lbl)
    ax6.set_xticks(x_pos + w_bar)
    ax6.set_xticklabels([m.upper() for m in model_names])
    ax6.set_ylabel("Forecast skill vs. naive persistence (%)")
    ax6.set_title("Forecasting Model Comparison — Skill Score across All Variables",
                  fontsize=9, fontweight="bold")
    ax6.legend(fontsize=8); ax6.grid(alpha=0.3, axis="y")
    ax6.axhline(0, color="black", lw=0.8)

    fig.suptitle("Smart Grid Bidirectional Power Flow — Integrated EMS Dashboard\n"
                 "IEEE 33-Bus Distribution System | ML + RL + Optimization Pipeline",
                 fontsize=12, fontweight="bold", y=0.99)
    fig.savefig(f"{OUT}/full_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved full_dashboard.png")


if __name__ == "__main__":
    from pipeline import run_full_pipeline
    print("Running full pipeline (this takes ~2 min)...")
    results = run_full_pipeline(n_mcs=1500, n_dispatch_scenarios=40)
    plot_forecast_intervals(results)
    plot_dispatch_schedule(results)
    plot_dispatch_stochastic(results)
    plot_full_kpi_summary(results)
    print(f"All plots saved to {OUT}/")
