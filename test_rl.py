import sys
sys.path.insert(0, 'src')
from rl_control import train_q_learning, evaluate_policy, CurtailmentEnv
import numpy as np

print("Training agent...")
Q, env, hist = train_q_learning(n_episodes=20000)

print("\nTesting on different load levels:")
print(f"{'Load Factor':<15} {'No Control':>12} {'RL Policy':>12}")
print("-" * 42)

for load_factor in [0.5, 1.0, 1.5, 2.0]:
    env_test = CurtailmentEnv(seed=999)
    env_test.P_load = env_test.P_load * load_factor
    env_test.Q_load = env_test.Q_load * load_factor
    rl, base, n = evaluate_policy(Q, env_test, n_eval=1000, seed=42)
    print(f"x{load_factor:<14} {base['over']/n:>11.1%} {rl['over']/n:>12.1%}")