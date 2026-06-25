import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from network import N_BUS, BRANCH_DATA, TOPO_ORDER, PARENT
from monte_carlo import run_mcs, WIND_BUS, PV_BUS

OUT = "results"


def plot_voltage_profile(mcs):
    fig, ax = plt.subplots(figsize=(11, 5))
    buses = np.arange(1, N_BUS + 1)
    data = [mcs["V_mag"][:, b] for b in buses]
    bp = ax.boxplot(data, positions=buses, widths=0.6, showfliers=False,
                     patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('#7fb3d5')
        patch.set_alpha(0.7)
    ax.axhline(1.05, color='red', linestyle='--', linewidth=1, label='Upper limit (1.05 pu)')
    ax.axhline(0.95, color='orange', linestyle='--', linewidth=1, label='Lower limit (0.95 pu)')
    ax.axvline(WIND_BUS, color='green', linestyle=':', alpha=0.6)
    ax.axvline(PV_BUS, color='goldenrod', linestyle=':', alpha=0.6)
    ax.text(WIND_BUS, 1.07, 'Wind', ha='center', color='green', fontsize=9)
    ax.text(PV_BUS, 1.07, 'PV', ha='center', color='goldenrod', fontsize=9)
    ax.set_xlabel("Bus number")
    ax.set_ylabel("Voltage magnitude (pu)")
    ax.set_title(f"IEEE 33-Bus Voltage Profile under Probabilistic DER "
                 f"(n={mcs['n_samples']} Monte-Carlo scenarios)")
    ax.set_xticks(buses)
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/voltage_profile.png", dpi=150)
    plt.close(fig)


def plot_reverse_flow_prob(mcs):
    prob = mcs["reverse_flags"][:, 2:].mean(axis=0)
    labels = [f"{f}->{t}" for f, t, r, x in BRANCH_DATA]
    order = np.argsort(prob)[::-1]
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['#c0392b' if p > 0.5 else '#e67e22' if p > 0 else '#95a5a6' for p in prob[order]]
    ax.barh(np.array(labels)[order], prob[order] * 100, color=colors)
    ax.set_xlabel("Probability of reverse (bidirectional) power flow (%)")
    ax.set_title("Bidirectional Power Flow Probability per Branch\n"
                 "(probability that active power flows from child toward substation)")
    ax.grid(alpha=0.3, axis='x')
    fig.tight_layout()
    fig.savefig(f"{OUT}/reverse_flow_probability.png", dpi=150)
    plt.close(fig)


def plot_network_diagram(mcs):
    prob = {t: mcs["reverse_flags"][:, t].mean() for f, t, r, x in BRANCH_DATA}
    G = nx.DiGraph()
    for f, t, r, x in BRANCH_DATA:
        G.add_edge(f, t, weight=prob[t])
    pos = nx.kamada_kawai_layout(G)
    fig, ax = plt.subplots(figsize=(11, 9))
    edge_colors = [G[u][v]['weight'] for u, v in G.edges()]
    node_colors = ['#27ae60' if n == WIND_BUS else '#f39c12' if n == PV_BUS
                   else '#3498db' if n == 1 else '#bdc3c7' for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=420, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
    edges = nx.draw_networkx_edges(G, pos, edge_color=edge_colors, edge_cmap=plt.cm.Reds,
                                    width=2.5, ax=ax, arrows=True, arrowsize=12)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.04)
    cbar.set_label("P(reverse power flow)")
    ax.set_title("IEEE 33-Bus Feeder Topology — Bidirectional Flow Risk Map\n"
                 "(blue = substation, green = wind site, orange = PV site)")
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(f"{OUT}/network_diagram.png", dpi=150)
    plt.close(fig)


def plot_surrogate_accuracy():
    from surrogate import train_and_compare
    from sklearn.model_selection import train_test_split
    results, bfs_ms, scaler, mcs = train_and_compare(n_samples=4000)

    X = np.column_stack([mcs["wind_speed"], mcs["irradiance"]])
    Yv = mcs["V_mag"][:, 1:N_BUS]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Yv, test_size=0.25, random_state=42)

    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4.3))
    for ax, (name, r) in zip(axes, results.items()):
        pred = r["model"].predict(scaler.transform(X_test))[:, :Y_test.shape[1]]
        ax.scatter(Y_test.flatten(), pred.flatten(), s=2, alpha=0.15, color='#2980b9')
        lims = [Y_test.min() - 0.005, Y_test.max() + 0.005]
        ax.plot(lims, lims, 'r--', linewidth=1)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel("True voltage (pu)")
        ax.set_ylabel("Predicted voltage (pu)")
        ax.set_title(f"{name}\nRMSE={r['rmse_voltage_pu']:.5f} pu, "
                     f"{r['pred_time_per_sample_ms']*1000:.2f} \u03bcs/sample")
        ax.grid(alpha=0.3)
    fig.suptitle("ML Surrogate Power-Flow Model: Predicted vs. True Bus Voltages (held-out test set)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/surrogate_accuracy.png", dpi=150)
    plt.close(fig)


def plot_rl_training():
    from rl_control import train_q_learning, evaluate_policy, CURTAIL_ACTIONS
    Q, env, hist = train_q_learning(n_episodes=20000)
    rl, base, n = evaluate_policy(Q, env, n_eval=3000)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    window = 250
    smoothed = pd.Series(hist).rolling(window).mean()
    axes[0].plot(smoothed, color='#2980b9')
    axes[0].set_xlabel("Training episode")
    axes[0].set_ylabel(f"Reward ({window}-episode moving average)")
    axes[0].set_title("Q-Learning Training Curve (DER Curtailment Agent)")
    axes[0].grid(alpha=0.3)

    metrics = ["Overvoltage\nrate (>1.05pu)", "Renewable energy\ncurtailed"]
    base_vals = [base["over"] / n * 100, 0]
    rl_vals = [rl["over"] / n * 100, rl["curtailed_energy_kwh"] / rl["total_energy_kwh"] * 100]
    x = np.arange(len(metrics))
    w = 0.35
    axes[1].bar(x - w / 2, base_vals, w, label="No control", color='#c0392b')
    axes[1].bar(x + w / 2, rl_vals, w, label="RL curtailment policy", color='#27ae60')
    axes[1].set_xticks(x); axes[1].set_xticklabels(metrics)
    axes[1].set_ylabel("%")
    axes[1].set_title("RL Policy vs. No Control (held-out evaluation)")
    axes[1].legend()
    axes[1].grid(alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(f"{OUT}/rl_training.png", dpi=150)
    plt.close(fig)
    return rl, base, n


def plot_forecast_example():
    from forecasting import generate_synthetic_series, train_forecaster
    df = generate_synthetic_series(n_days=90)
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for ax, col in zip(axes, ["load", "irradiance", "wind"]):
        r = train_forecaster(df, col)
        hrs = np.arange(len(r["y_test"]))
        ax.plot(hrs, r["y_test"], label="actual", color='#2c3e50', linewidth=1.3)
        ax.plot(hrs, r["pred"], label="NN forecast", color='#e74c3c', linewidth=1.1, linestyle='--')
        ax.set_ylabel(col)
        ax.set_title(f"{col}: RMSE={r['rmse']:.4f} (vs. naive persistence "
                     f"RMSE={r['rmse_naive_persistence']:.4f})")
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Hour (held-out test period)")
    fig.suptitle("Short-Term Forecasting: NN (lag-window) vs. Actual, Held-Out Test Period")
    fig.tight_layout()
    fig.savefig(f"{OUT}/forecasting.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    mcs = run_mcs(n_samples=3000)
    plot_voltage_profile(mcs)
    plot_reverse_flow_prob(mcs)
    plot_network_diagram(mcs)
    plot_surrogate_accuracy()
    plot_rl_training()
    plot_forecast_example()
    print("Saved all plots to", OUT)



