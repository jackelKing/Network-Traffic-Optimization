import yaml
from env.traffic_env import TrafficEnv
from agent.ppo_agent import PPOAgent

def main():
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    env = TrafficEnv(config)
    agent = PPOAgent(env, config)
    agent.load("results/ppo_model")

    obs, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _ = agent.model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        done = done or truncated

    print(f"Total reward: {total_reward}")

if __name__ == "__main__":
    main()
