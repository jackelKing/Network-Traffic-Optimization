import yaml
import warnings
import os
warnings.filterwarnings("ignore")

from env.traffic_env import TrafficEnv
from agent.ppo_agent import PPOAgent


def main():
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Deep copy config and use different port
    import copy
    grid_config = copy.deepcopy(config)
    grid_config["simulation"]["port"] = 5556  # different port

    grid_phase = {
        "name":      "grid-3x3",
        "num_nodes": 9,
        "topo_type": "grid",
        "sim_time":  10.0,
        "timesteps": 150000
    }

    print("=" * 55)
    print("  PPO Training — 3x3 Grid Mesh (port 5556)")
    print("=" * 55)
    print(f"  Nodes:     9 (3x3 grid)")
    print(f"  Port:      5556 (separate from main training)")
    print(f"  Timesteps: {grid_phase['timesteps']:,}")
    print("=" * 55)

    env = TrafficEnv(grid_config)
    env.set_phase(grid_phase)
    obs, _ = env.reset()

    print(f"  Obs space:    {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    print("=" * 55)

    agent = PPOAgent(env, grid_config)

    # Load best available checkpoint from main training
    for ckpt in [
        "results/ppo_model_grid-9",
        "results/ppo_model_linear-6",
        "results/ppo_model_linear-4",
    ]:
        if os.path.exists(ckpt + ".zip"):
            agent.load(ckpt)
            print(f"  Loaded weights from: {ckpt}")
            break

    agent.train_phase(grid_phase, reset_timesteps=True)
    agent.save("results/ppo_model_grid3x3")

    env.close()
    print("\nDone. Model saved to results/ppo_model_grid3x3")


if __name__ == "__main__":
    main()
