import yaml
import warnings
import os
warnings.filterwarnings("ignore")

from env.traffic_env import TrafficEnv
from agent.ppo_agent import PPOAgent


def main():
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    print("=" * 55)
    print("  PPO Curriculum - Resuming from Phase 4 (random-12)")
    print("=" * 55)

    env = TrafficEnv(config)

    phases      = config["curriculum"]["phases"]
    resume_from = 3  # index 3 = random-12

    first_phase = phases[resume_from]
    env.set_phase(first_phase)
    obs, _ = env.reset()

    print(f"  Obs space:    {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    print("=" * 55)

    agent = PPOAgent(env, config)

    # Load best checkpoint
    for checkpoint in [
        "results/ppo_model_grid-9",
        "results/ppo_model_linear-6",
        "results/ppo_model_linear-4",
    ]:
        if os.path.exists(checkpoint + ".zip"):
            agent.load(checkpoint)
            print(f"  Resumed from: {checkpoint}")
            break

    remaining = phases[resume_from:]
    print(f"  Remaining: {[p['name'] for p in remaining]}")
    print("=" * 55)

    agent.train_curriculum(remaining)
    agent.save("results/ppo_model_final")

    env.close()
    print("\nDone. Run python3 evaluate.py to test.")


if __name__ == "__main__":
    main()
