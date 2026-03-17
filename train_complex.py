import yaml
import warnings
import os
import copy
warnings.filterwarnings("ignore")

from env.traffic_env import TrafficEnv
from agent.ppo_agent import PPOAgent


def main():
    with open("config/config_complex.yaml", "r") as f:
        config = yaml.safe_load(f)

    print("=" * 60)
    print("  PPO COMPLEX TRAINING — BEAT OSPF MODE")
    print("=" * 60)
    print("  Topologies: random-12, random-16, random-20")
    print("  Traffic:    Heavy load (2-6ms intervals)")
    print("  Bandwidth:  5Mbps (bottleneck forcing)")
    print("  Goal:       Beat OSPF on all complex topologies")
    total = sum(p["timesteps"] for p in config["curriculum"]["phases"])
    print(f"  Timesteps:  {total:,}")
    print("=" * 60)

    env = TrafficEnv(config)
    phases = config["curriculum"]["phases"]

    # Start with first phase
    env.set_phase(phases[0])
    obs, _ = env.reset()

    print(f"  Obs space:    {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    print("=" * 60)

    agent = PPOAgent(env, config)

    # Load merged model as starting point — don't waste previous training
    for ckpt in [
        "results/ppo_model_merged",
        "results/ppo_model_final",
        "results/ppo_model_random-16",
    ]:
        if os.path.exists(ckpt + ".zip"):
            agent.load(ckpt)
            print(f"  Starting from: {ckpt}")
            break

    print("=" * 60)

    # Train all complex phases
    agent.train_curriculum(phases)
    agent.save("results/ppo_model_complex")

    # Also save as new merged — overwrite with better model
    agent.save("results/ppo_model_final_v2")

    env.close()
    print("\nDone. Run python3 baseline/compare_complex.py")


if __name__ == "__main__":
    main()
