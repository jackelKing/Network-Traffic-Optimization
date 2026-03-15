import os
import sys
import yaml
import numpy as np
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.traffic_env import TrafficEnv
from agent.ppo_agent import PPOAgent


class OSPFBaseline:
    """Sequential next-hop, max bandwidth always."""
    def __init__(self, num_nodes, max_nodes=20):
        self.num_nodes = num_nodes
        self.max_nodes = max_nodes

    def predict(self, obs):
        action = []
        for i in range(self.max_nodes):
            next_hop = min(i + 1, self.num_nodes - 1)
            action.append(float(next_hop))
            action.append(4.0)
        return np.array(action, dtype=np.float32)


class RandomBaseline:
    """Completely random actions — lower bound."""
    def __init__(self, max_nodes=20):
        self.max_nodes = max_nodes

    def predict(self, obs):
        return np.random.uniform(0, 4, size=self.max_nodes * 2).astype(np.float32)


class WorstCaseBaseline:
    """Always routes to node 0, min bandwidth — worst case."""
    def __init__(self, max_nodes=20):
        self.max_nodes = max_nodes

    def predict(self, obs):
        action = []
        for i in range(self.max_nodes):
            action.append(0.0)
            action.append(0.0)
        return np.array(action, dtype=np.float32)


def evaluate_agent(env, agent, phase, n_episodes=5):
    """Run agent and collect per-step raw metrics."""
    all_rewards  = []
    all_delays   = []
    all_utils    = []
    all_losses   = []
    all_steps    = []

    for ep in range(n_episodes):
        obs, _    = env.reset()
        done      = False
        truncated = False
        ep_reward = 0.0
        ep_d, ep_u, ep_l = [], [], []
        steps = 0

        while not (done or truncated):
            action = agent.predict(obs)
            obs, reward, done, truncated, info = env.step(action)

            # Raw obs stats
            delays = obs[2::3]
            utils  = obs[1::3]
            queues = obs[0::3]

            ep_d.append(float(np.mean(delays)))
            ep_u.append(float(np.mean(utils)))
            ep_l.append(float(np.mean(queues)))
            ep_reward += reward
            steps += 1

        all_rewards.append(ep_reward)
        all_delays.append(np.mean(ep_d) if ep_d else 0.0)
        all_utils.append(np.mean(ep_u)  if ep_u else 0.0)
        all_losses.append(np.mean(ep_l) if ep_l else 0.0)
        all_steps.append(steps)

    return {
        "mean_reward": float(np.mean(all_rewards)),
        "std_reward":  float(np.std(all_rewards)),
        "mean_delay":  float(np.mean(all_delays)),
        "mean_util":   float(np.mean(all_utils)),
        "mean_loss":   float(np.mean(all_losses)),
        "mean_steps":  float(np.mean(all_steps)),
        "all_rewards": all_rewards,
    }


def main():
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    phases  = config["curriculum"]["phases"]
    results = {p["name"]: {} for p in phases}

    print("=" * 65)
    print("  Full Baseline Comparison: PPO vs OSPF vs Random vs Worst")
    print("=" * 65)

    for phase in phases:
        print(f"\n--- {phase['name']} | nodes={phase['num_nodes']} | topo={phase['topo_type']} ---")

        # PPO
        print("  [PPO]")
        env   = TrafficEnv(config)
        agent = PPOAgent(env, config)
        for ckpt in [
            f"results/ppo_model_{phase['name']}",
            "results/ppo_model_final",
            "results/ppo_model_random-16",
        ]:
            if os.path.exists(ckpt + ".zip"):
                agent.load(ckpt)
                break
        env.set_phase(phase)
        res = evaluate_agent(env, agent, phase, n_episodes=5)
        results[phase["name"]]["PPO"] = res
        env.close()
        print(f"    reward={res['mean_reward']:.4f}±{res['std_reward']:.4f} "
              f"delay={res['mean_delay']:.4f} util={res['mean_util']:.4f}")

        # OSPF
        print("  [OSPF]")
        env  = TrafficEnv(config)
        env.set_phase(phase)
        ospf = OSPFBaseline(phase["num_nodes"])
        res  = evaluate_agent(env, ospf, phase, n_episodes=5)
        results[phase["name"]]["OSPF"] = res
        env.close()
        print(f"    reward={res['mean_reward']:.4f}±{res['std_reward']:.4f} "
              f"delay={res['mean_delay']:.4f} util={res['mean_util']:.4f}")

        # Random
        print("  [Random]")
        env = TrafficEnv(config)
        env.set_phase(phase)
        rnd = RandomBaseline()
        res = evaluate_agent(env, rnd, phase, n_episodes=5)
        results[phase["name"]]["Random"] = res
        env.close()
        print(f"    reward={res['mean_reward']:.4f}±{res['std_reward']:.4f} "
              f"delay={res['mean_delay']:.4f} util={res['mean_util']:.4f}")

        # Worst case
        print("  [Worst]")
        env  = TrafficEnv(config)
        env.set_phase(phase)
        wst  = WorstCaseBaseline()
        res  = evaluate_agent(env, wst, phase, n_episodes=5)
        results[phase["name"]]["Worst"] = res
        env.close()
        print(f"    reward={res['mean_reward']:.4f}±{res['std_reward']:.4f} "
              f"delay={res['mean_delay']:.4f} util={res['mean_util']:.4f}")

    # Save
    import json
    os.makedirs("results", exist_ok=True)
    with open("results/full_comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print table
    print("\n" + "=" * 75)
    print("  FINAL RESULTS TABLE")
    print("=" * 75)
    print(f"  {'Phase':<14} {'PPO':>10} {'OSPF':>10} {'Random':>10} {'Worst':>10} {'PPO>OSPF':>10}")
    print(f"  {'-'*70}")
    for phase in phases:
        p   = phase["name"]
        ppo = results[p].get("PPO",   {}).get("mean_reward", 0)
        osp = results[p].get("OSPF",  {}).get("mean_reward", 0)
        rnd = results[p].get("Random",{}).get("mean_reward", 0)
        wst = results[p].get("Worst", {}).get("mean_reward", 0)
        imp = ((ppo - osp) / max(abs(osp), 0.001)) * 100
        print(f"  {p:<14} {ppo:>10.4f} {osp:>10.4f} {rnd:>10.4f} {wst:>10.4f} {imp:>9.1f}%")
    print("=" * 75)

    # Plot
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    phase_names = [p["name"] for p in phases]
    agents      = ["PPO", "OSPF", "Random", "Worst"]
    colors      = {"PPO": "#2ECC71", "OSPF": "#4C9BE8",
                   "Random": "#E67E22", "Worst": "#E74C3C"}

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("PPO vs OSPF vs Random vs Worst Case — Full Comparison",
                 fontsize=14, fontweight="bold")
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # Reward comparison
    ax1 = fig.add_subplot(gs[0, 0])
    x   = np.arange(len(phase_names))
    w   = 0.2
    for i, agent in enumerate(agents):
        vals = [results[p].get(agent, {}).get("mean_reward", 0) for p in phase_names]
        stds = [results[p].get(agent, {}).get("std_reward",  0) for p in phase_names]
        ax1.bar(x + i*w - 1.5*w, vals, w, yerr=stds,
                label=agent, color=colors[agent], alpha=0.85, capsize=3)
    ax1.set_title("Mean Episode Reward")
    ax1.set_xticks(x)
    ax1.set_xticklabels(phase_names, rotation=15, fontsize=8)
    ax1.set_ylabel("Reward")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3, axis="y")

    # Reward trend lines
    ax2 = fig.add_subplot(gs[0, 1])
    markers = {"PPO": "s", "OSPF": "o", "Random": "^", "Worst": "x"}
    for agent in agents:
        vals = [results[p].get(agent, {}).get("mean_reward", 0) for p in phase_names]
        stds = [results[p].get(agent, {}).get("std_reward",  0) for p in phase_names]
        ax2.plot(phase_names, vals, marker=markers[agent],
                 color=colors[agent], linewidth=2, label=agent, markersize=8)
        ax2.fill_between(phase_names,
                         [v-s for v,s in zip(vals,stds)],
                         [v+s for v,s in zip(vals,stds)],
                         alpha=0.1, color=colors[agent])
    ax2.set_title("Reward Trend (increasing complexity)")
    ax2.set_ylabel("Mean Reward")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=15, fontsize=8)

    # PPO improvement heatmap
    ax3 = fig.add_subplot(gs[1, 0])
    baselines  = ["OSPF", "Random", "Worst"]
    hmap_data  = []
    for p in phase_names:
        ppo_r = results[p].get("PPO", {}).get("mean_reward", 0)
        row   = []
        for b in baselines:
            b_r = results[p].get(b, {}).get("mean_reward", 0)
            imp = ((ppo_r - b_r) / max(abs(b_r), 0.001)) * 100
            row.append(imp)
        hmap_data.append(row)
    hmap = np.array(hmap_data)
    im   = ax3.imshow(hmap, cmap="RdYlGn", aspect="auto", vmin=-50, vmax=50)
    ax3.set_xticks(range(len(baselines)))
    ax3.set_xticklabels(baselines, fontsize=9)
    ax3.set_yticks(range(len(phase_names)))
    ax3.set_yticklabels(phase_names, fontsize=8)
    for i in range(len(phase_names)):
        for j in range(len(baselines)):
            ax3.text(j, i, f"{hmap[i,j]:.1f}%",
                     ha="center", va="center", fontsize=8, fontweight="bold")
    plt.colorbar(im, ax=ax3)
    ax3.set_title("PPO Improvement over Baselines (%)")

    # Delay comparison
    ax4 = fig.add_subplot(gs[1, 1])
    for agent in ["PPO", "OSPF", "Random"]:
        vals = [results[p].get(agent, {}).get("mean_delay", 0) for p in phase_names]
        ax4.plot(phase_names, vals, marker=markers[agent],
                 color=colors[agent], linewidth=2, label=agent, markersize=8)
    ax4.set_title("Average Delay (lower = better)")
    ax4.set_ylabel("Normalised Delay")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=15, fontsize=8)

    os.makedirs("results/plots", exist_ok=True)
    plt.savefig("results/plots/ppo_vs_baseline.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\nSaved: results/plots/ppo_vs_baseline.png")


if __name__ == "__main__":
    main()
