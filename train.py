import yaml
from env.traffic_env import TrafficEnv
from agent.ppo_agent import PPOAgent

def main():
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    env = TrafficEnv(config)
    agent = PPOAgent(env, config)
    agent.train(config["ppo"]["total_timesteps"])
    agent.save("results/ppo_model")
    print("Training complete. Model saved.")

if __name__ == "__main__":
    main()
