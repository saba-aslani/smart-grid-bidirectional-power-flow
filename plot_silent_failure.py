"""
plot_silent_failure.py
======================
Generates BOTH the figures and the exact markdown tables for the
Robustness / partial-observability section, from a SINGLE run so the
numbers in the README and in the figures always match.

Outputs:
  results/silent_failure_shift.png
  results/partial_obs_fo_vs_po.png
  results/robustness_tables.md     <- copy these tables straight into the README

Requires the evaluate_policy fix in rl_control.py.
Run from the repo root:
    python plot_silent_failure.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, "src")
from rl_control import train_q_learning, evaluate_policy, CurtailmentEnv
import experiment_partial_observability as po

OUT = "results"
os.makedirs(OUT, exist_ok=True)

RED, GREEN, ORANGE = "#c0392b", "#27ae60", "#e67e22"
FACTORS = [0.5, 1.0, 1.5, 2.0]

# Full canonical budget. Lower these only if a run is too slow on your machine,
# but then RE-RUN this whole script so figures and tables stay in sync.
SHIFT_TRAIN, SHIFT_EVAL = 5, 5
PO_SEEDS, PO_EPISODES = 3, 40000

_md = []  # collected markdown, written out at the end


def figure_shift():
    base = {lf: [] for lf in FACTORS}
    rl = {lf: [] for lf in FACTORS}
    for seed in range(SHIFT_TRAIN):
        Q, env, _ = train_q_learning(n_episodes=20000, seed=seed)
        for lf in FACTORS:
            et = CurtailmentEnv(wind_rated_kw=env.wind_rated, pv_rated_kw=env.pv_rated, seed=999)
            et.P_load = et.P_load * lf
            et.Q_load = et.Q_load * lf
            for ev in range(SHIFT_EVAL):
                r, b, n = evaluate_policy(Q, et, n_eval=1000, seed=ev)
                base[lf].append(b["over"] / n * 100)
                rl[lf].append(r["over"] / n * 100)

    bm = [np.mean(base[lf]) for lf in FACTORS]
    bs = [np.std(base[lf]) for lf in FACTORS]
    rm = [np.mean(rl[lf]) for lf in FACTORS]
    rs = [np.std(rl[lf]) for lf in FACTORS]

    # ---- figure ----
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.errorbar(FACTORS, bm, yerr=bs, marker="o", lw=2, capsize=4, color=RED, label="No control")
    ax.errorbar(FACTORS, rm, yerr=rs, marker="s", lw=2, capsize=4, color=GREEN, label="RL policy")
    ax.axvspan(0.42, 0.62, color=RED, alpha=0.07)
    ax.annotate("silent failure:\nRL \u2248 no control", xy=(0.5, rm[0]), xytext=(0.72, 26),
                fontsize=10, color=RED, arrowprops=dict(arrowstyle="->", color=RED))
    ax.annotate("trained here\n(works)", xy=(1.0, rm[1]), xytext=(1.15, 9),
                fontsize=10, color=GREEN, arrowprops=dict(arrowstyle="->", color=GREEN))
    ax.set_xlabel("Load factor (load is NOT in the agent's state)")
    ax.set_ylabel("Overvoltage rate  (>1.05 pu)  [%]")
    ax.set_title("Silent Failure Under Load Shift")
    ax.set_xticks(FACTORS); ax.set_ylim(-2, 40); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(f"{OUT}/silent_failure_shift.png", dpi=150); plt.close(fig)

    # ---- matching markdown table ----
    names = {0.5: "x0.5 (low)", 1.0: "x1.0 (nominal)", 1.5: "x1.5 (high)", 2.0: "x2.0 (very high)"}
    _md.append("### Experiment 1 - Distribution shift\n")
    _md.append("| Load factor      | No control      | RL policy       |")
    _md.append("| ---------------- | --------------- | --------------- |")
    for i, lf in enumerate(FACTORS):
        _md.append(f"| {names[lf]:<16} | {bm[i]:.1f}% \u00b1 {bs[i]:.1f}    | {rm[i]:.1f}% \u00b1 {rs[i]:.1f}    |")
    _md.append("")
    print("saved results/silent_failure_shift.png")


def figure_fo_vs_po():
    groups = [0.5, 1.0]
    res = {(c, lf): [] for c in ("base", "PO", "FO") for lf in groups}
    for seed in range(PO_SEEDS):
        Qpo = po.train(False, PO_EPISODES, seed)
        Qfo = po.train(True, PO_EPISODES, seed)
        for ev in range(PO_SEEDS):
            for lf in groups:
                b, rpo = po.evaluate(Qpo, False, lf, ev)
                _, rfo = po.evaluate(Qfo, True, lf, ev)
                res[("base", lf)].append(b * 100)
                res[("PO", lf)].append(rpo * 100)
                res[("FO", lf)].append(rfo * 100)

    def mean(c, lf):
        return np.mean(res[(c, lf)])
    def std(c, lf):
        return np.std(res[(c, lf)])

    # ---- figure ----
    x = np.arange(len(groups)); w = 0.26
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for i, (c, color, lab) in enumerate([
        ("base", RED, "No control"),
        ("PO", ORANGE, "PO  (blind to load)"),
        ("FO", GREEN, "FO  (observes load)"),
    ]):
        ax.bar(x + (i - 1) * w, [mean(c, lf) for lf in groups], w,
               yerr=[std(c, lf) for lf in groups], capsize=3, color=color, label=lab)
    ax.set_xticks(x); ax.set_xticklabels([f"load x{lf}" for lf in groups])
    ax.set_ylabel("Overvoltage rate  (>1.05 pu)  [%]")
    ax.set_title("Observing Load Prevents Silent Failure\n(same training distribution; only the observation differs)")
    ax.grid(alpha=0.3, axis="y"); ax.legend()
    fig.tight_layout(); fig.savefig(f"{OUT}/partial_obs_fo_vs_po.png", dpi=150); plt.close(fig)

    # ---- matching markdown table ----
    _md.append("### Experiment 2 - Is it the missing observation?\n")
    _md.append("| Load factor    | No control | PO (blind to load) | FO (observes load) |")
    _md.append("| -------------- | ---------- | ------------------ | ------------------ |")
    for lf in groups:
        _md.append(f"| x{lf}           | {mean('base',lf):.1f}%      | {mean('PO',lf):.1f}%              | {mean('FO',lf):.1f}%              |")
    _md.append("")
    print("saved results/partial_obs_fo_vs_po.png")


if __name__ == "__main__":
    figure_shift()
    figure_fo_vs_po()
    with open(f"{OUT}/robustness_tables.md", "w") as f:
        f.write("\n".join(_md))
    print("\n================  COPY THE TABLES BELOW INTO THE README  ================\n")
    print("\n".join(_md))
