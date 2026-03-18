import yaml
import warnings
import os
import glob
warnings.filterwarnings("ignore")

from env.traffic_env import TrafficEnv
from agent.ppo_agent import PPOAgent


def get_latest_checkpoint(phase_name):
    pattern = f"results/checkpoints/{phase_name}/ppo_traffic_*_steps.zip"
    files   = glob.glob(pattern)
    if not files:
        return None
    files.sort(key=lambda x: int(x.split("_")[-2]))
    return files[-1].replace(".zip", "")


def main():
    with open("config/config_complex.yaml", "r") as f:
        config = yaml.safe_load(f)

    phases = config["curriculum"]["phases"]

    print("=" * 60)
    print("  PPO Complex Training — RESUME")
    print("=" * 60)

    env = TrafficEnv(config)
    env.set_phase(phases[0])
    obs, _ = env.reset()

    agent = PPOAgent(env, config)

    # Find best checkpoint to resume from
    ckpt = get_latest_checkpoint("random-12-heavy")
    if ckpt and os.path.exists(ckpt + ".zip"):
        agent.load(ckpt)
        print(f"  Resumed from checkpoint: {ckpt}")
    else:
        # Fall back to saved phase models
        for fallback in [
            "results/ppo_model_merged",
            "results/ppo_model_final",
        ]:
            if os.path.exists(fallback + ".zip"):
                agent.load(fallback)
                print(f"  Resumed from: {fallback}")
                break

    print("=" * 60)
    agent.train_curriculum(phases)
    agent.save("results/ppo_model_complex")
    agent.save("results/ppo_model_final_v2")

    env.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
