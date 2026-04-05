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
    """
    Strong OSPF: sequential next-hop + MAX bandwidth (100Mbps).
    No congestion awareness, no load balancing, no dynamic rerouting.
    PPO must beat this through intelligence, not just higher BW.
    """
    def __init__(self, num_nodes, max_nodes=20):
        self.num_nodes = num_nodes
        self.max_nodes = max_nodes

    def predict(self, obs):
        action = []
        for i in range(self.max_nodes):
            next_hop = min(i + 1, self.num_nodes - 1)
            action.append(float(next_hop))
            action.append(4.0)   # 100Mbps max — strongest fixed OSPF
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


def evaluate_agent(env, agent, phase, n_episodes=10):
    """Run agent and collect per-step raw metrics.
    Obs layout: [queue(0), util(1), delay(2), loss(3), congestion(4)] x N
    Stride = 5 (5 metrics per node).
    """
    all_rewards  = []
    all_delays   = []
    all_utils    = []
    all_losses   = []
    all_congs    = []
    all_steps    = []

    for ep in range(n_episodes):
        obs, _    = env.reset()
        done      = False
        truncated = False
        ep_reward = 0.0
        ep_d, ep_u, ep_l, ep_c = [], [], [], []
        steps = 0

        while not (done or truncated):
            action = agent.predict(obs)
            obs, reward, done, truncated, info = env.step(action)

            # CORRECT stride=5 for 5-metric obs
            ep_d.append(float(np.mean(obs[2::5])))   # delay
            ep_u.append(float(np.mean(obs[1::5])))   # util
            ep_l.append(float(np.mean(obs[3::5])))   # loss
            ep_c.append(float(np.mean(obs[4::5])))   # congestion
            ep_reward += reward
            steps += 1

        all_rewards.append(ep_reward)
        all_delays.append(np.mean(ep_d) if ep_d else 0.0)
        all_utils.append(np.mean(ep_u)  if ep_u else 0.0)
        all_losses.append(np.mean(ep_l) if ep_l else 0.0)
        all_congs.append(np.mean(ep_c)  if ep_c else 0.0)
        all_steps.append(steps)

    return {
        "mean_reward": float(np.mean(all_rewards)),
        "std_reward":  float(np.std(all_rewards)),
        "mean_delay":  float(np.mean(all_delays)),
        "mean_util":   float(np.mean(all_utils)),
        "mean_loss":   float(np.mean(all_losses)),
        "mean_cong":   float(np.mean(all_congs)),
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
        # Auto-find best checkpoint: prefer phase-specific, fall back to latest
        import glob
        ckpt_found = False
        candidates = [
            f"results/ppo_model_{phase['name']}",
            "results/ppo_model_final",
        ] + sorted(glob.glob("results/ppo_model_*.zip"), reverse=True)
        for ckpt in candidates:
            ckpt_clean = ckpt.replace(".zip", "")
            if os.path.exists(ckpt_clean + ".zip"):
                agent.load(ckpt_clean)
                ckpt_found = True
                break
        if not ckpt_found:
            print(f"    ⚠️  No checkpoint found for {phase['name']} — skipping PPO")
            env.close()
            continue
        env.set_phase(phase)
        res = evaluate_agent(env, agent, phase, n_episodes=10)
        results[phase["name"]]["PPO"] = res
        env.close()
        print(f"    reward={res['mean_reward']:.4f}±{res['std_reward']:.4f} "
              f"delay={res['mean_delay']:.4f} util={res['mean_util']:.4f} "
              f"cong={res['mean_cong']:.4f}")

        # OSPF
        print("  [OSPF]")
        env  = TrafficEnv(config)
        env.set_phase(phase)
        ospf = OSPFBaseline(phase["num_nodes"], topo_type=phase["topo_type"])
        res  = evaluate_agent(env, ospf, phase, n_episodes=10)
        results[phase["name"]]["OSPF"] = res
        env.close()
        print(f"    reward={res['mean_reward']:.4f}±{res['std_reward']:.4f} "
              f"delay={res['mean_delay']:.4f} util={res['mean_util']:.4f} "
              f"cong={res['mean_cong']:.4f}")

        # Random
        print("  [Random]")
        env = TrafficEnv(config)
        env.set_phase(phase)
        rnd = RandomBaseline()
        res = evaluate_agent(env, rnd, phase, n_episodes=10)
        results[phase["name"]]["Random"] = res
        env.close()
        print(f"    reward={res['mean_reward']:.4f}±{res['std_reward']:.4f} "
              f"delay={res['mean_delay']:.4f} util={res['mean_util']:.4f} "
              f"cong={res['mean_cong']:.4f}")

        # Worst case
        print("  [Worst]")
        env  = TrafficEnv(config)
        env.set_phase(phase)
        wst  = WorstCaseBaseline()
        res  = evaluate_agent(env, wst, phase, n_episodes=10)
        results[phase["name"]]["Worst"] = res
        env.close()
        print(f"    reward={res['mean_reward']:.4f}±{res['std_reward']:.4f} "
              f"delay={res['mean_delay']:.4f} util={res['mean_util']:.4f} "
              f"cong={res['mean_cong']:.4f}")

    import json
    os.makedirs("results", exist_ok=True)
    with open("results/full_comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 75)
    print("  FINAL RESULTS TABLE")
    print("=" * 75)
    print(f"  {'Phase':<14} {'PPO':>10} {'OSPF':>10} {'Random':>10} {'Worst':>10} {'PPO>OSPF':>10}")
    print(f"  {'-'*70}")
    for phase in phases:
        p   = phase["name"]
        ppo = results[p].get("PPO",    {}).get("mean_reward", 0)
        osp = results[p].get("OSPF",   {}).get("mean_reward", 0)
        rnd = results[p].get("Random", {}).get("mean_reward", 0)
        wst = results[p].get("Worst",  {}).get("mean_reward", 0)
        imp = ((ppo - osp) / max(abs(osp), 0.001)) * 100
        flag = "✅" if ppo > osp else "❌"
        print(f"  {p:<14} {ppo:>10.4f} {osp:>10.4f} {rnd:>10.4f} {wst:>10.4f} {imp:>9.1f}% {flag}")
    print("=" * 75)

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

    ax2 = fig.add_subplot(gs[0, 1])
    markers = {"PPO": "s", "OSPF": "o", "Random": "^", "Worst": "x"}
    for agent in agents:
        vals = [results[p].get(agent, {}).get("mean_reward", 0) for p in phase_names]
        stds = [results[p].get(agent, {}).get("std_reward",  0) for p in phase_names]
        ax2.plot(phase_names, vals, marker=markers[agent],
                 color=colors[agent], linewidth=2, label=agent, markersize=8)
        ax2.fill_between(phase_names,
                         [v-s for v, s in zip(vals, stds)],
                         [v+s for v, s in zip(vals, stds)],
                         alpha=0.1, color=colors[agent])
    ax2.set_title("Reward Trend (increasing complexity)")
    ax2.set_ylabel("Mean Reward")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=15, fontsize=8)

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

    # ── Extended 3x3 dashboard ────────────────────────────────
    fig2, axes = plt.subplots(3, 3, figsize=(22, 16))
    fig2.suptitle("PPO vs OSPF — Full Network Performance Dashboard",
                  fontsize=15, fontweight="bold", y=0.98)
    agents  = ["PPO", "OSPF", "Random"]
    colors  = {"PPO": "#2ECC71", "OSPF": "#4C9BE8", "Random": "#E67E22"}
    markers = {"PPO": "s", "OSPF": "o", "Random": "^"}

    def plot_metric(ax, metric_key, title, ylabel, lower_better=False):
        for agent in agents:
            vals = [results[p].get(agent, {}).get(metric_key, 0) for p in phase_names]
            stds = [results[p].get(agent, {}).get("std_reward", 0) for p in phase_names]
            ax.plot(phase_names, vals, marker=markers[agent],
                    color=colors[agent], linewidth=2, label=agent, markersize=7)
        if lower_better:
            ax.invert_yaxis()
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, fontsize=7)

    # Row 0: core metrics
    plot_metric(axes[0,0], "mean_reward",  "Episode Reward (higher=better)",    "Reward")
    plot_metric(axes[0,1], "mean_delay",   "Avg Latency (lower=better)",         "Norm. Delay",  lower_better=True)
    plot_metric(axes[0,2], "mean_util",    "Link Utilisation (higher=better)",   "Norm. Util")

    # Row 1: reliability metrics
    plot_metric(axes[1,0], "mean_loss",    "Packet Loss Rate (lower=better)",    "Norm. Loss",   lower_better=True)
    plot_metric(axes[1,1], "mean_cong",    "Congestion Level (lower=better)",    "Norm. Cong",   lower_better=True)

    # Row 1,2: PPO % improvement over OSPF per metric
    ax_imp = axes[1,2]
    metrics_imp = ["mean_reward","mean_util","mean_delay","mean_loss","mean_cong"]
    metric_labels = ["Reward","Util","Delay↓","Loss↓","Cong↓"]
    x_imp = np.arange(len(metrics_imp))
    width = 0.07
    for pi, p in enumerate(phase_names):
        imps = []
        for m in metrics_imp:
            ppo_v  = results[p].get("PPO",  {}).get(m, 0)
            ospf_v = results[p].get("OSPF", {}).get(m, 0)
            imp    = ((ppo_v - ospf_v) / max(abs(ospf_v), 0.001)) * 100
            imps.append(imp)
        ax_imp.bar(x_imp + pi*width - (len(phase_names)/2)*width,
                   imps, width, label=p, alpha=0.8)
    ax_imp.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax_imp.set_xticks(x_imp)
    ax_imp.set_xticklabels(metric_labels, fontsize=8)
    ax_imp.set_title("PPO % Improvement over OSPF per Metric", fontsize=10, fontweight="bold")
    ax_imp.set_ylabel("% Improvement")
    ax_imp.legend(fontsize=5, ncol=2)
    ax_imp.grid(True, alpha=0.3, axis="y")

    # Row 2: per-topology breakdown (linear / grid / random)
    topo_groups = {
        "linear": [p for p in phase_names if "linear" in p],
        "grid":   [p for p in phase_names if "grid"   in p],
        "random": [p for p in phase_names if "random" in p],
    }
    topo_colors = {"linear": "#9B59B6", "grid": "#1ABC9C", "random": "#E67E22"}
    ax_topo = axes[2,0]
    for topo, topo_phases in topo_groups.items():
        if not topo_phases:
            continue
        for agent in ["PPO", "OSPF"]:
            vals = [results[p].get(agent, {}).get("mean_reward", 0) for p in topo_phases]
            ls   = "-" if agent == "PPO" else "--"
            ax_topo.plot(topo_phases, vals, linestyle=ls,
                         color=topo_colors[topo], linewidth=2,
                         label=f"{topo}-{agent}", marker="o", markersize=6)
    ax_topo.set_title("Reward by Topology Type: PPO vs OSPF", fontsize=10, fontweight="bold")
    ax_topo.set_ylabel("Mean Reward")
    ax_topo.legend(fontsize=6, ncol=2)
    ax_topo.grid(True, alpha=0.3)
    plt.setp(ax_topo.xaxis.get_majorticklabels(), rotation=30, fontsize=7)

    # Radar / spider chart: PPO vs OSPF average across all phases
    ax_radar = axes[2,1]
    radar_metrics = ["mean_reward","mean_util","mean_delay","mean_loss","mean_cong"]
    radar_labels  = ["Reward","Util","Delay↓","Loss↓","Cong↓"]
    ppo_avg  = [np.mean([results[p].get("PPO",  {}).get(m, 0) for p in phase_names]) for m in radar_metrics]
    ospf_avg = [np.mean([results[p].get("OSPF", {}).get(m, 0) for p in phase_names]) for m in radar_metrics]
    x_r = np.arange(len(radar_metrics))
    ax_radar.bar(x_r - 0.18, ppo_avg,  0.35, label="PPO",  color="#2ECC71", alpha=0.85)
    ax_radar.bar(x_r + 0.18, ospf_avg, 0.35, label="OSPF", color="#4C9BE8", alpha=0.85)
    ax_radar.set_xticks(x_r)
    ax_radar.set_xticklabels(radar_labels, fontsize=9)
    ax_radar.set_title("PPO vs OSPF — Average Across All Phases", fontsize=10, fontweight="bold")
    ax_radar.legend(fontsize=9)
    ax_radar.grid(True, alpha=0.3, axis="y")

    # Win-rate summary table
    ax_tbl = axes[2,2]
    ax_tbl.axis("off")
    wins = {"reward":0,"util":0,"delay":0,"loss":0,"cong":0}
    totl = len(phase_names)
    for p in phase_names:
        ppo  = results[p].get("PPO",  {})
        ospf = results[p].get("OSPF", {})
        if ppo.get("mean_reward",0) > ospf.get("mean_reward",0): wins["reward"] += 1
        if ppo.get("mean_util",0)   > ospf.get("mean_util",0):   wins["util"]   += 1
        if ppo.get("mean_delay",0)  < ospf.get("mean_delay",0):  wins["delay"]  += 1
        if ppo.get("mean_loss",0)   < ospf.get("mean_loss",0):   wins["loss"]   += 1
        if ppo.get("mean_cong",0)   < ospf.get("mean_cong",0):   wins["cong"]   += 1
    tbl_data = [[m.capitalize(), f"{w}/{totl}", "✅" if w > totl//2 else "❌"]
                for m, w in wins.items()]
    tbl = ax_tbl.table(
        cellText=tbl_data,
        colLabels=["Metric","PPO Wins","Result"],
        loc="center", cellLoc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.4, 2.2)
    ax_tbl.set_title("PPO Win Rate vs OSPF", fontsize=10, fontweight="bold", pad=20)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig("results/plots/ppo_vs_ospf_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: results/plots/ppo_vs_ospf_dashboard.png")


if __name__ == "__main__":
    main()


# ── Failure-resilience benchmark ─────────────────────────────
def evaluate_under_failure(env, agent, phase, n_episodes=3, name="Agent"):
    """
    Injects a synthetic link failure mid-episode by zeroing out
    one node's obs slice and measuring how quickly the agent recovers.
    OSPF would require full SPF recomputation; PPO adapts on-policy.
    """
    rewards, recovery_steps = [], []

    for ep in range(n_episodes):
        obs, _    = env.reset()
        done      = False
        truncated = False
        ep_reward = 0.0
        step      = 0
        failure_injected = False
        recovery_step    = None
        pre_fail_util    = None

        while not (done or truncated):
            # Inject failure at step 10: zero out node 1's obs
            if step == 10 and not failure_injected:
                obs[5:10] = 0.0   # node-1 goes dark (stride=5, node 1 starts at idx 5)
                failure_injected = True
                pre_fail_util    = float(np.mean(obs[1::5]))

            action = agent.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward

            # Detect recovery: util bounces back above 50% of pre-failure level
            if failure_injected and recovery_step is None and pre_fail_util:
                cur_util = float(np.mean(obs[1::5]))
                if cur_util >= pre_fail_util * 0.5:
                    recovery_step = step - 10

            step += 1

        rewards.append(ep_reward)
        recovery_steps.append(recovery_step if recovery_step else step - 10)

    return {
        "agent":          name,
        "phase":          phase["name"],
        "failure_reward": float(np.mean(rewards)),
        "recovery_steps": float(np.mean(recovery_steps)),
    }


def print_failure_benchmark(results_fail):
    print("\n" + "=" * 65)
    print("  FAILURE RESILIENCE BENCHMARK")
    print(f"  {'Phase':<14} {'Agent':<10} {'Reward':>10} {'Recovery Steps':>16}")
    print(f"  {'-'*55}")
    for r in results_fail:
        print(f"  {r['phase']:<14} {r['agent']:<10} "
              f"{r['failure_reward']:>10.4f} {r['recovery_steps']:>16.1f}")
    print("=" * 65)
    print("Lower recovery_steps → faster adaptation after link failure.")
    print("PPO should outperform OSPF here due to on-policy re-routing.\n")
