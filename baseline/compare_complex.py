import os
import sys
import json
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.traffic_env import TrafficEnv
from agent.ppo_agent import PPOAgent


class OSPFBaseline:
    def __init__(self, num_nodes, max_nodes=20):
        self.num_nodes = num_nodes
        self.max_nodes = max_nodes

    def predict(self, obs):
        action = []
        for i in range(self.max_nodes):
            next_hop = min(i + 1, self.num_nodes - 1)
            action.append(float(next_hop))
            action.append(2.0)  # fixed BW
        return np.array(action, dtype=np.float32)


class ECMPBaseline:
    """
    ECMP — Equal Cost Multi Path.
    Distributes traffic across multiple paths round-robin.
    More advanced than OSPF.
    """
    def __init__(self, num_nodes, max_nodes=20):
        self.num_nodes  = num_nodes
        self.max_nodes  = max_nodes
        self.call_count = 0

    def predict(self, obs):
        action = []
        for i in range(self.max_nodes):
            # Alternate between two next hops
            if self.call_count % 2 == 0:
                next_hop = min(i + 1, self.num_nodes - 1)
            else:
                next_hop = min(i + 2, self.num_nodes - 1)
            action.append(float(next_hop))
            action.append(2.0)
        self.call_count += 1
        return np.array(action, dtype=np.float32)


def evaluate_agent(env, agent, phase, n_episodes=5, name="Agent"):
    rewards, delays, utils, losses, congs = [], [], [], [], []

    for ep in range(n_episodes):
        obs, _    = env.reset()
        done      = False
        truncated = False
        ep_reward = 0.0
        ep_d, ep_u, ep_l, ep_c = [], [], [], []

        while not (done or truncated):
            action = agent.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            ep_d.append(float(np.mean(obs[2::5])))
            ep_u.append(float(np.mean(obs[1::5])))
            ep_l.append(float(np.mean(obs[3::5])))
            ep_c.append(float(np.mean(obs[4::5])))
            ep_reward += reward

        rewards.append(ep_reward)
        delays.append(np.mean(ep_d) if ep_d else 0.0)
        utils.append(np.mean(ep_u)  if ep_u else 0.0)
        losses.append(np.mean(ep_l) if ep_l else 0.0)
        congs.append(np.mean(ep_c)  if ep_c else 0.0)

        print(f"    {name:8s} ep {ep+1}/{n_episodes} "
              f"| reward={ep_reward:.4f} "
              f"| delay={delays[-1]:.4f} "
              f"| util={utils[-1]:.4f} "
              f"| cong={congs[-1]:.4f}")

    return {
        "agent":       name,
        "phase":       phase["name"],
        "num_nodes":   phase["num_nodes"],
        "mean_reward": float(np.mean(rewards)),
        "std_reward":  float(np.std(rewards)),
        "mean_delay":  float(np.mean(delays)),
        "mean_util":   float(np.mean(utils)),
        "mean_loss":   float(np.mean(losses)),
        "mean_cong":   float(np.mean(congs)),
    }


def main():
    with open("config/config_complex.yaml", "r") as f:
        config = yaml.safe_load(f)

    phases  = config["curriculum"]["phases"]
    results = {p["name"]: {} for p in phases}

    print("=" * 65)
    print("  Complex Topology Comparison: PPO vs OSPF vs ECMP")
    print("=" * 65)

    agents_to_run = [
        ("PPO",  None),
        ("OSPF", None),
        ("ECMP", None),
    ]

    for phase in phases:
        print(f"\n{'='*65}")
        print(f"  Phase: {phase['name']} | nodes={phase['num_nodes']}")
        print(f"{'='*65}")

        for agent_name, _ in agents_to_run:
            print(f"  [{agent_name}]")
            env = TrafficEnv(config)
            env.set_phase(phase)

            if agent_name == "PPO":
                agent = PPOAgent(env, config)
                for ckpt in [
                    "results/ppo_model_complex",
                    "results/ppo_model_final_v2",
                    "results/ppo_model_merged",
                    "results/ppo_model_final",
                ]:
                    if os.path.exists(ckpt + ".zip"):
                        agent.load(ckpt)
                        break
            elif agent_name == "OSPF":
                agent = OSPFBaseline(phase["num_nodes"])
            else:
                agent = ECMPBaseline(phase["num_nodes"])

            res = evaluate_agent(env, agent, phase, 5, agent_name)
            results[phase["name"]][agent_name] = res
            env.close()

    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/complex_comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print table
    print("\n" + "=" * 80)
    print("  FINAL COMPLEX COMPARISON TABLE")
    print("=" * 80)
    print(f"  {'Phase':<20} {'PPO':>8} {'OSPF':>8} "
          f"{'ECMP':>8} {'PPO>OSPF':>10} {'PPO>ECMP':>10}")
    print(f"  {'-'*75}")

    for phase in phases:
        p    = phase["name"]
        ppo  = results[p].get("PPO",  {}).get("mean_reward", 0)
        osp  = results[p].get("OSPF", {}).get("mean_reward", 0)
        ecmp = results[p].get("ECMP", {}).get("mean_reward", 0)
        imp_ospf = ((ppo-osp) /max(abs(osp), 0.001))*100
        imp_ecmp = ((ppo-ecmp)/max(abs(ecmp),0.001))*100
        print(f"  {p:<20} {ppo:>8.4f} {osp:>8.4f} "
              f"{ecmp:>8.4f} {imp_ospf:>9.1f}% {imp_ecmp:>9.1f}%")

    print("=" * 80)

    # Delay comparison
    print("\n  DELAY COMPARISON (lower = better)")
    print(f"  {'Phase':<20} {'PPO':>8} {'OSPF':>8} {'ECMP':>8}")
    print(f"  {'-'*50}")
    for phase in phases:
        p    = phase["name"]
        ppo  = results[p].get("PPO",  {}).get("mean_delay", 0)
        osp  = results[p].get("OSPF", {}).get("mean_delay", 0)
        ecmp = results[p].get("ECMP", {}).get("mean_delay", 0)
        winner = "✅" if ppo <= min(osp, ecmp) else "❌"
        print(f"  {p:<20} {ppo:>8.4f} {osp:>8.4f} "
              f"{ecmp:>8.4f} {winner}")

    # Congestion comparison
    print("\n  CONGESTION COMPARISON (lower = better)")
    print(f"  {'Phase':<20} {'PPO':>8} {'OSPF':>8} {'ECMP':>8}")
    print(f"  {'-'*50}")
    for phase in phases:
        p    = phase["name"]
        ppo  = results[p].get("PPO",  {}).get("mean_cong", 0)
        osp  = results[p].get("OSPF", {}).get("mean_cong", 0)
        ecmp = results[p].get("ECMP", {}).get("mean_cong", 0)
        winner = "✅" if ppo <= min(osp, ecmp) else "❌"
        print(f"  {p:<20} {ppo:>8.4f} {osp:>8.4f} "
              f"{ecmp:>8.4f} {winner}")

    # Plot
    phase_names = [p["name"] for p in phases]
    agents      = ["PPO", "OSPF", "ECMP"]
    colors      = {"PPO": "#2ECC71", "OSPF": "#4C9BE8", "ECMP": "#E67E22"}
    markers     = {"PPO": "s",       "OSPF": "o",       "ECMP": "^"}

    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    fig.suptitle("PPO vs OSPF vs ECMP — Complex Topology Benchmark",
                 fontsize=14, fontweight="bold")

    # Reward bars
    ax = axes[0, 0]
    x  = np.arange(len(phase_names))
    w  = 0.25
    for i, ag in enumerate(agents):
        vals = [results[p].get(ag, {}).get("mean_reward", 0)
                for p in phase_names]
        stds = [results[p].get(ag, {}).get("std_reward",  0)
                for p in phase_names]
        bars = ax.bar(x + (i-1)*w, vals, w,
                      yerr=stds, label=ag,
                      color=colors[ag], alpha=0.85, capsize=3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height()+0.02,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_title("Mean Episode Reward")
    ax.set_xticks(x)
    ax.set_xticklabels(phase_names, rotation=20, fontsize=7)
    ax.set_ylabel("Reward")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # Reward trend
    ax = axes[0, 1]
    for ag in agents:
        vals = [results[p].get(ag, {}).get("mean_reward", 0)
                for p in phase_names]
        stds = [results[p].get(ag, {}).get("std_reward",  0)
                for p in phase_names]
        ax.plot(phase_names, vals, marker=markers[ag],
                color=colors[ag], linewidth=2,
                label=ag, markersize=8)
        ax.fill_between(phase_names,
                        [v-s for v,s in zip(vals,stds)],
                        [v+s for v,s in zip(vals,stds)],
                        alpha=0.1, color=colors[ag])
    ax.set_title("Reward Trend")
    ax.set_ylabel("Mean Reward")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=20, fontsize=7)

    # PPO improvement heatmap
    ax = axes[0, 2]
    baselines = ["OSPF", "ECMP"]
    hmap_data = []
    for p in phase_names:
        ppo_r = results[p].get("PPO", {}).get("mean_reward", 0)
        row   = []
        for b in baselines:
            b_r = results[p].get(b, {}).get("mean_reward", 0)
            imp = ((ppo_r-b_r)/max(abs(b_r),0.001))*100
            row.append(imp)
        hmap_data.append(row)
    hmap = np.array(hmap_data)
    im   = ax.imshow(hmap, cmap="RdYlGn", aspect="auto",
                     vmin=-30, vmax=30)
    ax.set_xticks(range(len(baselines)))
    ax.set_xticklabels(baselines)
    ax.set_yticks(range(len(phase_names)))
    ax.set_yticklabels(phase_names, fontsize=7)
    for i in range(len(phase_names)):
        for j in range(len(baselines)):
            ax.text(j, i, f"{hmap[i,j]:.1f}%",
                    ha="center", va="center",
                    fontsize=9, fontweight="bold")
    plt.colorbar(im, ax=ax)
    ax.set_title("PPO Improvement (%)")

    # Delay comparison
    ax = axes[1, 0]
    for ag in agents:
        vals = [results[p].get(ag, {}).get("mean_delay", 0)
                for p in phase_names]
        ax.plot(phase_names, vals, marker=markers[ag],
                color=colors[ag], linewidth=2,
                label=ag, markersize=8)
    ax.set_title("Avg Delay (lower = better)")
    ax.set_ylabel("Normalised Delay")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=20, fontsize=7)

    # Congestion comparison
    ax = axes[1, 1]
    for ag in agents:
        vals = [results[p].get(ag, {}).get("mean_cong", 0)
                for p in phase_names]
        ax.plot(phase_names, vals, marker=markers[ag],
                color=colors[ag], linewidth=2,
                label=ag, markersize=8)
    ax.set_title("Congestion (lower = better)")
    ax.set_ylabel("Congestion Score")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=20, fontsize=7)

    # Utilization comparison
    ax = axes[1, 2]
    for ag in agents:
        vals = [results[p].get(ag, {}).get("mean_util", 0)
                for p in phase_names]
        ax.plot(phase_names, vals, marker=markers[ag],
                color=colors[ag], linewidth=2,
                label=ag, markersize=8)
    ax.set_title("Link Utilization (higher = better)")
    ax.set_ylabel("Normalised Utilization")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=20, fontsize=7)

    plt.tight_layout()
    os.makedirs("results/plots", exist_ok=True)
    plt.savefig("results/plots/complex_comparison.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("\nSaved: results/plots/complex_comparison.png")


if __name__ == "__main__":
    main()
