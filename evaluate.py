import yaml
import warnings
import numpy as np
warnings.filterwarnings("ignore")

from env.traffic_env import TrafficEnv
from agent.ppo_agent import PPOAgent


def main():
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    print("=" * 55)
    print("  PPO Evaluation — All Phases")
    print("=" * 55)

    env    = TrafficEnv(config)
    agent  = PPOAgent(env, config)
    agent.load("results/ppo_model_final")

    phases = config["curriculum"]["phases"]

    for phase in phases:
        env.set_phase(phase)
        ep_rewards = []

        for ep in range(3):
            obs, _    = env.reset()
            done      = False
            truncated = False
            ep_reward = 0.0
            steps     = 0

            while not (done or truncated):
                action                      = agent.predict(obs)
                obs, reward, done, truncated, _ = env.step(action)
                ep_reward += reward
                steps     += 1

            ep_rewards.append(ep_reward)

        print(f"  {phase['name']:15s} | "
              f"mean={np.mean(ep_rewards):.4f} | "
              f"best={np.max(ep_rewards):.4f} | "
              f"steps={steps}")

    print("=" * 55)
    env.close()


if __name__ == "__main__":
    main()
