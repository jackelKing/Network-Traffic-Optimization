"""
Merges two trained PPO models by averaging their weights.
Run after both main training and grid3x3 training are complete.
"""
import os
import torch
import warnings
import yaml
warnings.filterwarnings("ignore")

from stable_baselines3 import PPO
from env.traffic_env import TrafficEnv


def merge_models(model_a_path, model_b_path,
                 output_path, alpha=0.5):
    """
    Merge two PPO models by weighted averaging.
    alpha=0.5 means equal weight to both models.
    alpha=0.7 means 70% model_a, 30% model_b.
    """
    print("=" * 55)
    print("  PPO Model Merger")
    print("=" * 55)
    print(f"  Model A:  {model_a_path} (weight={alpha:.1f})")
    print(f"  Model B:  {model_b_path} (weight={1-alpha:.1f})")
    print(f"  Output:   {output_path}")
    print("=" * 55)

    # Load both models
    print("\nLoading models...")
    model_a = PPO.load(model_a_path)
    model_b = PPO.load(model_b_path)

    # Get state dicts
    params_a = model_a.policy.state_dict()
    params_b = model_b.policy.state_dict()

    # Check they have same architecture
    if params_a.keys() != params_b.keys():
        print("ERROR: Models have different architectures — cannot merge")
        return False

    # Average weights
    print("Averaging weights...")
    merged_params = {}
    for key in params_a.keys():
        if params_a[key].shape != params_b[key].shape:
            print(f"  Skipping {key} — shape mismatch")
            merged_params[key] = params_a[key]
            continue
        merged_params[key] = (alpha * params_a[key]
                              + (1 - alpha) * params_b[key])
        print(f"  Merged: {key} {tuple(params_a[key].shape)}")

    # Load merged weights into model_a
    model_a.policy.load_state_dict(merged_params)

    # Save merged model
    os.makedirs("results", exist_ok=True)
    model_a.save(output_path)
    print(f"\nMerged model saved → {output_path}")
    return True


def evaluate_merged(output_path, config, phases, n_episodes=3):
    """Quick evaluation of merged model across all phases."""
    import numpy as np
    from agent.ppo_agent import PPOAgent

    print("\n" + "=" * 55)
    print("  Evaluating Merged Model")
    print("=" * 55)

    results = []
    for phase in phases:
        env   = TrafficEnv(config)
        agent = PPOAgent(env, config)
        agent.load(output_path)
        env.set_phase(phase)

        ep_rewards = []
        for ep in range(n_episodes):
            obs, _    = env.reset()
            done      = False
            truncated = False
            ep_reward = 0.0
            while not (done or truncated):
                action = agent.predict(obs)
                obs, reward, done, truncated, _ = env.step(action)
                ep_reward += reward
            ep_rewards.append(ep_reward)

        mean_r = np.mean(ep_rewards)
        results.append((phase["name"], mean_r))
        print(f"  {phase['name']:<15} | mean reward={mean_r:.4f}")
        env.close()

    print("=" * 55)
    return results


def main():
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Paths — adjust if needed
    model_a = "results/ppo_model_final"       # main overnight training
    model_b = "results/ppo_model_grid3x3"     # grid3x3 training

    # Check both exist
    if not os.path.exists(model_a + ".zip"):
        print(f"Model A not found: {model_a}.zip")
        print("Wait for main training to finish first")
        return

    if not os.path.exists(model_b + ".zip"):
        print(f"Model B not found: {model_b}.zip")
        print("Wait for grid3x3 training to finish first")
        return

    # Merge with equal weights
    success = merge_models(
        model_a, model_b,
        output_path="results/ppo_model_merged",
        alpha=0.5
    )

    if success:
        phases = config["curriculum"]["phases"]
        evaluate_merged("results/ppo_model_merged", config, phases)
        print("\nNow run: python3 baseline/compare_ppo_vs_baseline.py")
        print("Update checkpoint path to: results/ppo_model_merged")


if __name__ == "__main__":
    main()
