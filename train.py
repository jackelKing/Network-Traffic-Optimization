import yaml
import warnings
warnings.filterwarnings("ignore")

from env.traffic_env import TrafficEnv
from agent.ppo_agent import PPOAgent


def main():
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    print("=" * 55)
    print("  PPO Network Traffic Optimization")
    print("  Curriculum Learning Mode")
    print("=" * 55)

    env = TrafficEnv(config)

    # Init with first phase so spaces are correct
    first_phase = config["curriculum"]["phases"][0]
    env.set_phase(first_phase)
    obs, _ = env.reset()

    print(f"  Obs space:    {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    print("=" * 55)

    agent  = PPOAgent(env, config)
    phases = config["curriculum"]["phases"]

    agent.train_curriculum(phases)
    agent.save("results/ppo_model_final")

    env.close()
    print("\nDone. Run python3 evaluate.py to test.")


if __name__ == "__main__":
    main()
