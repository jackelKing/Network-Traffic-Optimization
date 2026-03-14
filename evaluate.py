import yaml
import warnings
import numpy as np
warnings.filterwarnings("ignore")

from env.traffic_env import TrafficEnv
from agent.ppo_agent import PPOAgent


def main():
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    print("=" * 50)
    print("  PPO Evaluation")
    print("=" * 50)

    env = TrafficEnv(config)
    agent = PPOAgent(env, config)
    agent.load("results/ppo_model")

    n_episodes = 5
    all_rewards = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done      = False
        truncated = False
        ep_reward = 0.0
        steps     = 0

        while not (done or truncated):
            action = agent.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward
            steps     += 1

        all_rewards.append(ep_reward)
        print(f"  Episode {ep+1} | steps={steps} | reward={ep_reward:.4f}")

    print("=" * 50)
    print(f"  Mean reward: {np.mean(all_rewards):.4f}")
    print(f"  Std  reward: {np.std(all_rewards):.4f}")
    print("=" * 50)

    env.close()


if __name__ == "__main__":
    main()
