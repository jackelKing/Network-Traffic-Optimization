"""
OSPF Baseline — simulates shortest-path routing with fixed bandwidth.
Runs the same NS3 environment but with a fixed greedy action
instead of PPO, giving us a fair comparison.
"""
import os
import sys
import yaml
import numpy as np
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.traffic_env import TrafficEnv


class OSPFBaseline:
    def __init__(self, num_nodes, max_nodes=20):
        self.num_nodes = num_nodes
        self.max_nodes = max_nodes

    def predict(self, obs):
        action = []
        for i in range(self.max_nodes):
            next_hop = min(i + 1, self.num_nodes - 1)
            bw_level = 4
            action.append(float(next_hop))
            action.append(float(bw_level))
        return np.array(action, dtype=np.float32)


class RandomBaseline:
    def __init__(self, action_size):
        self.action_size = action_size

    def predict(self, obs):
        return np.random.uniform(0, 4, size=self.action_size).astype(np.float32)


def run_baseline(env, agent, phase, n_episodes=3, agent_name="OSPF"):
    rewards, delays, utils, losses = [], [], [], []

    for ep in range(n_episodes):
        obs, _    = env.reset()
        done      = False
        truncated = False
        ep_reward = 0.0
        ep_d, ep_u, ep_l = [], [], []

        while not (done or truncated):
            action = agent.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            ep_d.append(float(np.mean(obs[2::3])))
            ep_u.append(float(np.mean(obs[1::3])))
            ep_l.append(float(np.mean(obs[0::3])))
            ep_reward += reward

        rewards.append(ep_reward)
        delays.append(np.mean(ep_d))
        utils.append(np.mean(ep_u))
        losses.append(np.mean(ep_l))
        print(f"    {agent_name} ep {ep+1}/{n_episodes} "
              f"| reward={ep_reward:.4f} "
              f"| delay={np.mean(ep_d):.4f} "
              f"| util={np.mean(ep_u):.4f}")

    return {
        "agent":       agent_name,
        "phase":       phase["name"],
        "num_nodes":   phase["num_nodes"],
        "topo":        phase["topo_type"],
        "rewards":     rewards,
        "delays":      delays,
        "utils":       utils,
        "losses":      losses,
        "mean_reward": float(np.mean(rewards)),
        "std_reward":  float(np.std(rewards)),
        "mean_delay":  float(np.mean(delays)),
        "mean_util":   float(np.mean(utils)),
        "mean_loss":   float(np.mean(losses)),
    }


def main():
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    print("=" * 60)
    print("  Baseline: OSPF + Random on all phases")
    print("=" * 60)

    phases      = config["curriculum"]["phases"]
    all_results = []

    for phase in phases:
        print(f"\n--- Phase: {phase['name']} "
              f"| nodes={phase['num_nodes']} ---")

        env = TrafficEnv(config)
        env.set_phase(phase)

        print("  Running OSPF...")
        ospf   = OSPFBaseline(phase["num_nodes"])
        result = run_baseline(env, ospf, phase, 3, "OSPF")
        all_results.append(result)

        print("  Running Random...")
        rnd    = RandomBaseline(action_size=40)
        result = run_baseline(env, rnd, phase, 3, "Random")
        all_results.append(result)

        env.close()

    os.makedirs("results", exist_ok=True)
    import json
    with open("results/baseline_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  {'Phase':<15} {'Agent':<10} {'Reward':>8} {'Delay':>8} {'Util':>8}")
    print(f"  {'-'*55}")
    for r in all_results:
        print(f"  {r['phase']:<15} {r['agent']:<10} "
              f"{r['mean_reward']:>8.4f} "
              f"{r['mean_delay']:>8.4f} "
              f"{r['mean_util']:>8.4f}")
    print("=" * 60)
    print("Saved: results/baseline_results.json")


if __name__ == "__main__":
    main()
