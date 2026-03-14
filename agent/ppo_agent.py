import os
import numpy as np
from tqdm import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn


class TrafficNet(BaseFeaturesExtractor):
    """256 → 256 → 128 with LayerNorm + ReLU."""
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        input_dim = int(np.prod(observation_space.shape))
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )

    def forward(self, obs):
        return self.net(obs)


def linear_schedule(initial_lr, final_lr=1e-5):
    def schedule(progress):
        return final_lr + progress * (initial_lr - final_lr)
    return schedule


class TqdmCallback(BaseCallback):
    def __init__(self, total_timesteps, phase_name="", verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.phase_name      = phase_name
        self.pbar            = None
        self.episode_reward  = 0.0
        self.episode_count   = 0
        self.best_reward     = -np.inf

    def _on_training_start(self):
        self.pbar = tqdm(
            total=self.total_timesteps,
            desc=f"Phase [{self.phase_name}]",
            unit="step",
            dynamic_ncols=True,
            colour="green"
        )

    def _on_step(self) -> bool:
        self.pbar.update(1)
        self.episode_reward += self.locals["rewards"][0]

        if self.locals["dones"][0]:
            self.episode_count += 1
            if self.episode_reward > self.best_reward:
                self.best_reward = self.episode_reward
            self.pbar.set_postfix({
                "ep":     self.episode_count,
                "reward": f"{self.episode_reward:.3f}",
                "best":   f"{self.best_reward:.3f}",
            })
            self.logger.record("train/episode_reward", self.episode_reward)
            self.logger.record("train/episode",        self.episode_count)
            self.episode_reward = 0.0
        return True

    def _on_training_end(self):
        self.pbar.close()


class PPOAgent:
    def __init__(self, env, config):
        self.env    = env
        self.config = config
        ppo_cfg     = config["ppo"]

        os.makedirs("results/checkpoints", exist_ok=True)
        os.makedirs("logs",                exist_ok=True)

        policy_kwargs = dict(
            features_extractor_class  = TrafficNet,
            features_extractor_kwargs = dict(features_dim=128),
            net_arch                  = dict(pi=[128, 64], vf=[128, 64])
        )

        self.model = PPO(
            policy          = ppo_cfg["policy"],
            env             = env,
            learning_rate   = linear_schedule(ppo_cfg["learning_rate"]),
            n_steps         = ppo_cfg["n_steps"],
            batch_size      = ppo_cfg["batch_size"],
            n_epochs        = ppo_cfg["n_epochs"],
            gamma           = ppo_cfg["gamma"],
            gae_lambda      = ppo_cfg.get("gae_lambda", 0.95),
            clip_range      = ppo_cfg.get("clip_range", 0.2),
            ent_coef        = ppo_cfg.get("ent_coef", 0.05),
            vf_coef         = ppo_cfg.get("vf_coef", 0.5),
            max_grad_norm   = ppo_cfg.get("max_grad_norm", 0.5),
            policy_kwargs   = policy_kwargs,
            verbose         = 0,
            tensorboard_log = "logs/"
        )

    def train_phase(self, phase, reset_timesteps=False):
        """Train on a single curriculum phase."""
        timesteps  = phase["timesteps"]
        phase_name = phase["name"]

        self.env.set_phase(phase)

        tqdm_cb = TqdmCallback(
            total_timesteps=timesteps,
            phase_name=phase_name,
            verbose=1
        )
        checkpoint_cb = CheckpointCallback(
            save_freq   = 10000,
            save_path   = f"results/checkpoints/{phase_name}/",
            name_prefix = "ppo_traffic",
            verbose     = 0
        )

        self.model.learn(
            total_timesteps     = timesteps,
            callback            = [tqdm_cb, checkpoint_cb],
            tb_log_name         = f"PPO_{phase_name}",
            reset_num_timesteps = reset_timesteps
        )

        # Save after each phase
        self.save(f"results/ppo_model_{phase_name}")

    def train_curriculum(self, phases):
        """Run all curriculum phases sequentially."""
        print("\n" + "=" * 55)
        print("  CURRICULUM TRAINING")
        print("=" * 55)
        total = sum(p["timesteps"] for p in phases)
        print(f"  Phases:          {len(phases)}")
        print(f"  Total timesteps: {total:,}")
        print("=" * 55)

        for i, phase in enumerate(phases):
            print(f"\n[{i+1}/{len(phases)}] {phase['name']}"
                  f" | nodes={phase['num_nodes']}"
                  f" | topo={phase['topo_type']}"
                  f" | steps={phase['timesteps']:,}")

            self.train_phase(
                phase,
                reset_timesteps=(i == 0)  # only reset on first phase
            )
            print(f"  Phase {phase['name']} complete.")

        print("\n" + "=" * 55)
        print("  CURRICULUM COMPLETE")
        print("=" * 55)

    def save(self, path="results/ppo_model"):
        self.model.save(path)
        print(f"  Saved → {path}")

    def load(self, path="results/ppo_model"):
        self.model = PPO.load(path, env=self.env)
        print(f"  Loaded ← {path}")

    def predict(self, obs):
        action, _ = self.model.predict(obs, deterministic=True)
        return action
