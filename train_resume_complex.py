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
    latest = files[-1].replace(".zip", "")
    steps  = int(files[-1].split("_")[-2])
    return latest, steps


def main():
    with open("config/config_complex.yaml", "r") as f:
        config = yaml.safe_load(f)

    phases = config["curriculum"]["phases"]

    print("=" * 60)
    print("  PPO Complex Training — SMART RESUME")
    print("=" * 60)

    # Find which phase to resume from
    resume_phase_idx = 0
    resume_ckpt      = None

    for i, phase in enumerate(phases):
        # Check if phase fully completed
        final_model = f"results/ppo_model_{phase['name']}.zip"
        if os.path.exists(final_model):
            print(f"  Phase {phase['name']}: COMPLETE ✓")
            resume_phase_idx = i + 1
            continue

        # Check for partial checkpoint
        result = get_latest_checkpoint(phase["name"])
        if result:
            ckpt, steps = result
            print(f"  Phase {phase['name']}: PARTIAL ({steps} steps done)")
            resume_ckpt      = ckpt
            resume_phase_idx = i
        else:
            print(f"  Phase {phase['name']}: NOT STARTED")
        break

    if resume_phase_idx >= len(phases):
        print("  All phases complete!")
        return

    remaining = phases[resume_phase_idx:]
    print(f"\n  Resuming from phase: {remaining[0]['name']}")
    print("=" * 60)

    env = TrafficEnv(config)
    env.set_phase(remaining[0])
    obs, _ = env.reset()

    print(f"  Obs space:    {env.observation_space}")
    print(f"  Action space: {env.action_space}")

    agent = PPOAgent(env, config)

    # Load best checkpoint
    load_priority = []
    if resume_ckpt:
        load_priority.append(resume_ckpt)
    load_priority += [
        "results/ppo_model_merged",
        "results/ppo_model_final",
    ]

    for ckpt in load_priority:
        if os.path.exists(ckpt + ".zip"):
            agent.load(ckpt)
            print(f"  Loaded: {ckpt}")
            break

    print("=" * 60)

    agent.train_curriculum(remaining)
    agent.save("results/ppo_model_complex")
    agent.save("results/ppo_model_final_v2")

    env.close()
    print("\nDone. Run: python3 baseline/compare_complex.py")


if __name__ == "__main__":
    main()
