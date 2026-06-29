"""
Reinforcement learning (tabular Q-learning) for real-time DER curtailment
control, used to keep bus voltages within [0.95, 1.05] pu when high
wind+PV output causes reverse power flow and local voltage rise.

State   : discretized (pre-curtailment wind power, pre-curtailment PV power)
Action  : joint curtailment fraction applied to both DER outputs
Reward  : -(voltage-violation penalty) - (curtailment / lost-energy cost)

No RL framework (gymnasium/stable-baselines) is available in this sandbox,
so the environment and Q-learning update are implemented directly in numpy.
For a 2-D discretized state / 1-D discrete action space this is a standard,
fully legitimate choice (tabular Q-learning), not a simplification of the
method itself.
"""

import numpy as np

from network import N_BUS, load_array_kw_kvar, backward_forward_sweep
from der_models import WindTurbine, PVSystem
from monte_carlo import WIND_BUS, PV_BUS

V_MIN, V_MAX = 0.95, 1.05
CURTAIL_ACTIONS = np.array([0.0, 0.15, 0.30, 0.45, 0.60, 0.80])
N_BINS = 8  # per dimension state discretization


class CurtailmentEnv:
    def __init__(self, wind_rated_kw=2000.0, pv_rated_kw=1800.0, seed=0):
        self.wt = WindTurbine(rated_kw=wind_rated_kw)
        self.pv = PVSystem(rated_kw=pv_rated_kw)
        self.rng = np.random.default_rng(seed)
        self.P_load, self.Q_load = load_array_kw_kvar()
        self.wind_rated = wind_rated_kw
        self.pv_rated = pv_rated_kw

    def sample_scenario(self):
        v = self.wt.sample_wind_speed(1, self.rng)[0]
        g = self.pv.sample_irradiance(1, self.rng)[0]
        p_wind = self.wt.power_kw(np.array([v]))[0]
        p_pv = self.pv.power_kw(np.array([g]))[0]
        return p_wind, p_pv

    def state_bin(self, p_wind, p_pv):
        bw = min(int(p_wind / self.wind_rated * N_BINS), N_BINS - 1)
        bp = min(int(p_pv / self.pv_rated * N_BINS), N_BINS - 1)
        return bw * N_BINS + bp

    def step(self, p_wind, p_pv, action_idx):
        c = CURTAIL_ACTIONS[action_idx]
        der = np.zeros(N_BUS + 1)
        der[WIND_BUS] = p_wind * (1 - c)
        der[PV_BUS] = p_pv * (1 - c)
        res = backward_forward_sweep(self.P_load, self.Q_load, der_p_kw=der)
        vmag = np.abs(res["V"][1:])

        over = np.clip(vmag - V_MAX, 0, None)
        under = np.clip(V_MIN - vmag, 0, None)
        # Overvoltage is the phenomenon curtailment can actually fix (it is
        # caused by DER reverse export); undervoltage is a chronic baseline
        # characteristic of this feeder (present even with zero DER -- see
        # network.py base-case Vmin=0.9131pu) that curtailment cannot
        # improve, so it is tracked but only lightly weighted in the reward.
        over_penalty = float(np.sum(over ** 2)) * 15000.0
        under_penalty = float(np.sum(under ** 2)) * 200.0
        curtailment_cost = c * (p_wind + p_pv) / (self.wind_rated + self.pv_rated) * 3.0

        reward = -(over_penalty + under_penalty + curtailment_cost)
        n_over = int(np.sum(over > 0))
        n_under = int(np.sum(under > 0))
        return reward, vmag, n_over, n_under, c


def train_q_learning(n_episodes=20000, alpha=0.2, gamma=0.0, eps_start=1.0,
                      eps_end=0.02, seed=0):
    # gamma=0: single-step (contextual bandit) episodes -- each scenario is iid
    env = CurtailmentEnv(seed=seed)
    n_states = N_BINS * N_BINS
    n_actions = len(CURTAIL_ACTIONS)
    Q = np.zeros((n_states, n_actions))
    rng = np.random.default_rng(seed)

    reward_history = []
    for ep in range(n_episodes):
        eps = eps_start + (eps_end - eps_start) * min(ep / (n_episodes * 0.8), 1.0)
        p_wind, p_pv = env.sample_scenario()
        s = env.state_bin(p_wind, p_pv)

        if rng.random() < eps:
            a = rng.integers(n_actions)
        else:
            a = int(np.argmax(Q[s]))

        r, vmag, n_over, n_under, c = env.step(p_wind, p_pv, a)
        Q[s, a] += alpha * (r - Q[s, a])  # gamma=0 -> no bootstrapped next-state term
        reward_history.append(r)

    return Q, env, np.array(reward_history)


def evaluate_policy(Q, env, n_eval=3000, seed=123):
    rng_eval = np.random.default_rng(seed)
    env_eval = CurtailmentEnv(wind_rated_kw=env.wind_rated, pv_rated_kw=env.pv_rated, seed=seed)
    env_eval.P_load = env.P_load.copy()   
    env_eval.Q_load = env.Q_load.copy()
    
    results_rl = {"over": 0, "under": 0, "curtailed_energy_kwh": 0.0, "total_energy_kwh": 0.0, "reward": 0.0}
    results_base = {"over": 0, "under": 0, "curtailed_energy_kwh": 0.0, "total_energy_kwh": 0.0, "reward": 0.0}

    for _ in range(n_eval):
        p_wind, p_pv = env_eval.sample_scenario()
        s = env_eval.state_bin(p_wind, p_pv)
        a_rl = int(np.argmax(Q[s]))
        a_base = 0  # baseline: no curtailment at all

        r, vmag, n_over, n_under, c = env_eval.step(p_wind, p_pv, a_rl)
        results_rl["over"] += (n_over > 0)
        results_rl["under"] += (n_under > 0)
        results_rl["curtailed_energy_kwh"] += c * (p_wind + p_pv)
        results_rl["total_energy_kwh"] += (p_wind + p_pv)
        results_rl["reward"] += r

        r0, vmag0, n_over0, n_under0, c0 = env_eval.step(p_wind, p_pv, a_base)
        results_base["over"] += (n_over0 > 0)
        results_base["under"] += (n_under0 > 0)
        results_base["total_energy_kwh"] += (p_wind + p_pv)
        results_base["reward"] += r0

    return results_rl, results_base, n_eval


if __name__ == "__main__":
    Q, env, hist = train_q_learning(n_episodes=20000)
    print(f"Training complete. Mean reward (last 2000 eps): {hist[-2000:].mean():.3f} "
          f"(first 2000 eps: {hist[:2000].mean():.3f})")

    rl, base, n = evaluate_policy(Q, env, n_eval=3000)
    print(f"\nEvaluation over {n} unseen scenarios:")
    print(f"{'Metric':<35}{'No control (baseline)':>24}{'RL curtailment policy':>24}")
    print(f"{'Overvoltage rate (>1.05pu)':<35}{base['over']/n:>23.1%}{rl['over']/n:>24.1%}")
    print(f"{'Undervoltage rate (<0.95pu)*':<35}{base['under']/n:>23.1%}{rl['under']/n:>24.1%}")
    print(f"{'Renewable energy curtailed':<35}{'0.0%':>24}"
          f"{rl['curtailed_energy_kwh']/rl['total_energy_kwh']:>23.1%}")
    print(f"{'Mean reward':<35}{base['reward']/n:>24.2f}{rl['reward']/n:>24.2f}")
    print("\n* Undervoltage is a chronic base-feeder characteristic (present even "
          "at zero DER output) that DER curtailment cannot resolve -- shown for "
          "completeness, not a target of this controller.")
