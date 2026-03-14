import yaml
import warnings
warnings.filterwarnings("ignore")

from env.traffic_env import TrafficEnv
from agent.ppo_agent import PPOAgent


def main():
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    print("=" * 50)
    print("  PPO Network Traffic Optimization")
    print("=" * 50)
    print(f"  Nodes:     {config['simulation']['num_nodes']}")
    print(f"  Topology:  {config['topology']['type']}")
    print(f"  Sim time:  {config['simulation']['sim_time']}s")
    print(f"  Timesteps: {config['ppo']['total_timesteps']}")
    print("=" * 50)

    env   = TrafficEnv(config)
    obs, _ = env.reset()
    print(f"  Obs space:    {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    print("=" * 50)

    agent = PPOAgent(env, config)
    agent.train(config["ppo"]["total_timesteps"])
    agent.save("results/ppo_model")

    env.close()
    print("\nDone. Run evaluate.py to test the trained model.")


if __name__ == "__main__":
    main()
