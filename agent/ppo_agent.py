import os
import numpy as np
from tqdm import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback


class TqdmCallback(BaseCallback):
    def __init__(self, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.pbar            = None
        self.episode_reward  = 0.0
        self.episode_count   = 0
        self.best_reward     = -np.inf

    def _on_training_start(self):
        self.pbar = tqdm(
            total=self.total_timesteps,
            desc="Training",
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
                "ep":       self.episode_count,
                "reward":   f"{self.episode_reward:.3f}",
                "best":     f"{self.best_reward:.3f}",
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

        self.model = PPO(
            policy          = ppo_cfg["policy"],
            env             = env,
            learning_rate   = ppo_cfg["learning_rate"],
            n_steps         = ppo_cfg["n_steps"],
            batch_size      = ppo_cfg["batch_size"],
            n_epochs        = ppo_cfg["n_epochs"],
            gamma           = ppo_cfg["gamma"],
            gae_lambda      = ppo_cfg.get("gae_lambda", 0.95),
            clip_range      = ppo_cfg.get("clip_range", 0.2),
            ent_coef        = ppo_cfg.get("ent_coef", 0.01),
            verbose         = 0,
            tensorboard_log = "logs/"
        )

    def train(self, timesteps):
        tqdm_cb = TqdmCallback(total_timesteps=timesteps, verbose=1)

        checkpoint_cb = CheckpointCallback(
            save_freq   = 10000,
            save_path   = "results/checkpoints/",
            name_prefix = "ppo_traffic",
            verbose     = 0
        )

        print(f"\nStarting PPO training for {timesteps} timesteps...")
        self.model.learn(
            total_timesteps     = timesteps,
            callback            = [tqdm_cb, checkpoint_cb],
            tb_log_name         = "PPO_traffic",
            reset_num_timesteps = True
        )
        print("\nTraining complete.")

    def save(self, path="results/ppo_model"):
        self.model.save(path)
        print(f"Model saved → {path}")

    def load(self, path="results/ppo_model"):
        self.model = PPO.load(path, env=self.env)
        print(f"Model loaded ← {path}")

    def predict(self, obs):
        action, _ = self.model.predict(obs, deterministic=True)
        return action
