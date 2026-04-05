import yaml
import warnings
warnings.filterwarnings("ignore")

from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from env.traffic_env import TrafficEnv
from agent.ppo_agent import PPOAgent

N_ENVS = 3
BASE_PORT = 5555

def make_env(config, port_offset):
    def _init():
        cfg = {k: v for k, v in config.items()}
        cfg["simulation"] = {**config["simulation"], "port_override": BASE_PORT + port_offset}
        env = TrafficEnv(cfg)
        first_phase = config["curriculum"]["phases"][0]
        env.set_phase(first_phase)
        return env
    return _init

def main():
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    print("=" * 55)
    print("  PPO Network Traffic Optimization")
    print(f"  Parallel Envs: {N_ENVS}  (ports {BASE_PORT}-{BASE_PORT+N_ENVS-1})")
    print("=" * 55)

    import time
    env_fns = [make_env(config, i) for i in range(N_ENVS)]
    time.sleep(1)  # stagger startup
    vec_env = SubprocVecEnv(env_fns, start_method="fork")
    vec_env = VecMonitor(vec_env)

    print(f"  Obs space:    {vec_env.observation_space}")
    print(f"  Action space: {vec_env.action_space}")
    print("=" * 55)

    agent = PPOAgent(vec_env, config)
    phases = config["curriculum"]["phases"]

    print("\n" + "=" * 55)
    print("  CURRICULUM TRAINING")
    print("=" * 55)
    total = sum(p["timesteps"] for p in phases)
    print(f"  Phases:          {len(phases)}")
    print(f"  Total timesteps: {total:,}  (effective: {total*N_ENVS:,})")
    print("=" * 55)

    for i, phase in enumerate(phases):
        print(f"\n[{i+1}/{len(phases)}] {phase['name']}"
              f" | nodes={phase['num_nodes']}"
              f" | topo={phase['topo_type']}"
              f" | steps={phase['timesteps']:,}")

        # Push phase to all envs
        vec_env.env_method("set_phase", phase)
        agent.train_phase(phase, reset_timesteps=(i == 0))
        print(f"  Phase {phase['name']} complete.")

    agent.save("results/ppo_model_final")
    vec_env.close()
    print("\nDone.")

if __name__ == "__main__":
    main()
