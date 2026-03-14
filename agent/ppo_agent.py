import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback, CheckpointCallback, EvalCallback
)
from stable_baselines3.common.monitor import Monitor
import numpy as np


class TrainingMetricsCallback(BaseCallback):
    """
    Logs reward, delay, throughput, loss to TensorBoard every episode.
    """
    def __init__(self, log_interval=1, verbose=0):
        super().__init__(verbose)
        self.log_interval  = log_interval
        self.episode_rewards = []
        self.current_reward  = 0.0

    def _on_step(self) -> bool:
        self.current_reward += self.locals["rewards"][0]
        if self.locals["dones"][0]:
            self.episode_rewards.append(self.current_reward)
            ep = len(self.episode_rewards)
            self.logger.record("train/episode_reward", self.current_reward)
            self.logger.record("train/episode",        ep)
            if self.verbose > 0:
                print(f"  Episode {ep:4d} | reward={self.current_reward:.4f}")
            self.current_reward = 0.0
        return True


class PPOAgent:
    def __init__(self, env, config):
        self.env    = env
        self.config = config
        ppo_cfg     = config["ppo"]

        os.makedirs("results", exist_ok=True)
        os.makedirs("logs",    exist_ok=True)

        self.model = PPO(
            policy            = ppo_cfg["policy"],
            env               = env,
            learning_rate     = ppo_cfg["learning_rate"],
            n_steps           = ppo_cfg["n_steps"],
            batch_size        = ppo_cfg["batch_size"],
            n_epochs          = ppo_cfg["n_epochs"],
            gamma             = ppo_cfg["gamma"],
            gae_lambda        = ppo_cfg.get("gae_lambda", 0.95),
            clip_range        = ppo_cfg.get("clip_range", 0.2),
            ent_coef          = ppo_cfg.get("ent_coef", 0.01),
            verbose           = 1,
            tensorboard_log   = "logs/"
        )

    def train(self, timesteps):
        metrics_cb = TrainingMetricsCallback(verbose=1)

        checkpoint_cb = CheckpointCallback(
            save_freq       = 10000,
            save_path       = "results/checkpoints/",
            name_prefix     = "ppo_traffic",
            verbose         = 1
        )

        print(f"\nStarting PPO training for {timesteps} timesteps...")
        self.model.learn(
            total_timesteps  = timesteps,
            callback         = [metrics_cb, checkpoint_cb],
            tb_log_name      = "PPO_traffic",
            reset_num_timesteps = True
        )
        print("Training complete.")

    def save(self, path="results/ppo_model"):
        self.model.save(path)
        print(f"Model saved to {path}")

    def load(self, path="results/ppo_model"):
        self.model = PPO.load(path, env=self.env)
        print(f"Model loaded from {path}")

    def predict(self, obs):
        action, _ = self.model.predict(obs, deterministic=True)
        return action
