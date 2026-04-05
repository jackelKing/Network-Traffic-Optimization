import os
import numpy as np
from tqdm import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn


class NodeAttention(nn.Module):
    def __init__(self, node_dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(node_dim, num_heads,
                                          batch_first=True, dropout=0.0)
        self.norm = nn.LayerNorm(node_dim)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        return self.norm(x + attn_out)


class TrafficNet(BaseFeaturesExtractor):
    MAX_NODES   = 20
    OBS_PER_NODE = 8   # 5 core + 3 history

    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        input_dim = int(np.prod(observation_space.shape))
        node_dim  = 64

        self.node_embed = nn.Sequential(
            nn.Linear(self.OBS_PER_NODE, node_dim),
            nn.LayerNorm(node_dim),
            nn.ReLU()
        )
        self.attn1 = NodeAttention(node_dim, num_heads=4)
        self.attn2 = NodeAttention(node_dim, num_heads=4)

        self.global_proj = nn.Sequential(
            nn.Linear(node_dim * self.MAX_NODES, 512),
            nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, features_dim),
            nn.LayerNorm(features_dim), nn.ReLU(),
        )
        self.fallback = nn.Sequential(
            nn.Linear(input_dim, 512), nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, features_dim), nn.LayerNorm(features_dim), nn.ReLU(),
        )

    def forward(self, obs):
        B        = obs.shape[0]
        expected = self.MAX_NODES * self.OBS_PER_NODE
        if obs.shape[1] == expected:
            x = obs.view(B, self.MAX_NODES, self.OBS_PER_NODE)
            x = self.node_embed(x)
            x = self.attn1(x)
            x = self.attn2(x)
            x = x.reshape(B, -1)
            return self.global_proj(x)
        else:
            return self.fallback(obs)


def cosine_schedule(initial_lr, final_lr=3e-5):   # decay to 10% of initial
    """Cosine annealing — never decays all the way to zero."""
    import math
    def schedule(progress_remaining):
        # progress_remaining: 1.0 → 0.0
        cos_val = 0.5 * (1 + math.cos(math.pi * (1 - progress_remaining)))
        return final_lr + (initial_lr - final_lr) * cos_val
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
                "ep":   self.episode_count,
                "ep_r": f"{self.episode_reward:.2f}",
                "best": f"{self.best_reward:.2f}",
            })
            self.logger.record("train/episode_reward", self.episode_reward)
            self.logger.record("train/episode",        self.episode_count)
            self.episode_reward = 0.0
        return True

    def _on_training_end(self):
        self.pbar.close()


class EnhancedMetricsCallback(BaseCallback):
    def __init__(self, phase_name="", convergence_threshold=0.5, verbose=0):
        super().__init__(verbose)
        self.phase_name            = phase_name
        self.convergence_threshold = convergence_threshold
        self._ep_delays            = []
        self._ep_losses            = []
        self._ep_rewards           = []
        self._convergence_step     = None
        self._best_5ep_avg         = -float("inf")
        self._reward_window        = []

    def _on_step(self) -> bool:
        obs    = self.locals.get("new_obs", None)
        reward = self.locals["rewards"][0]
        done   = self.locals["dones"][0]

        if obs is not None:
            o = obs[0] if hasattr(obs, "__len__") and len(obs.shape) > 1 else obs
            # Slice into CORE only (first 100 elements, stride 5)
            self._ep_delays.append(float(np.mean(o[2:100:5])))
            self._ep_losses.append(float(np.mean(o[3:100:5])))

        self._ep_rewards.append(float(reward))

        if done:
            ep_reward = float(np.sum(self._ep_rewards))
            ep_delay  = float(np.mean(self._ep_delays)) if self._ep_delays else 0.0
            ep_loss   = float(np.mean(self._ep_losses)) if self._ep_losses else 0.0

            self.logger.record("metrics/ep_delay",       ep_delay)
            self.logger.record("metrics/ep_packet_loss", ep_loss)
            self.logger.record("metrics/ep_reward",      ep_reward)

            self._reward_window.append(ep_reward)
            if len(self._reward_window) > 5:
                self._reward_window.pop(0)

            avg5 = float(np.mean(self._reward_window))
            if len(self._reward_window) == 5 and avg5 > self._best_5ep_avg:
                self._best_5ep_avg = avg5
                self.logger.record("metrics/best_5ep_avg", avg5)

            if (self._convergence_step is None
                    and avg5 > self.convergence_threshold
                    and len(self._reward_window) == 5):
                self._convergence_step = self.num_timesteps
                self.logger.record("metrics/convergence_step",
                                   self._convergence_step)
                print(f"  ⚡ Convergence at step {self._convergence_step:,}"
                      f" (avg5={avg5:.4f})")

            self._ep_delays  = []
            self._ep_losses  = []
            self._ep_rewards = []

        return True


class PPOAgent:
    def __init__(self, env, config):
        self.env    = env
        self.config = config
        ppo_cfg     = config["ppo"]

        os.makedirs("results/checkpoints", exist_ok=True)
        os.makedirs("logs",                exist_ok=True)

        policy_kwargs = dict(
            features_extractor_class  = TrafficNet,
            features_extractor_kwargs = dict(features_dim=256),
            net_arch                  = dict(pi=[256, 256, 128], vf=[256, 256, 128])
        )

        self.model = PPO(
            policy              = ppo_cfg["policy"],
            env                 = env,
            learning_rate       = cosine_schedule(ppo_cfg["learning_rate"]),
            n_steps             = ppo_cfg["n_steps"],
            batch_size          = ppo_cfg["batch_size"],
            n_epochs            = ppo_cfg["n_epochs"],
            gamma               = ppo_cfg["gamma"],
            gae_lambda          = ppo_cfg.get("gae_lambda", 0.90),
            clip_range          = ppo_cfg.get("clip_range", 0.2),
            clip_range_vf       = ppo_cfg.get("clip_range_vf", 0.2),
            ent_coef            = ppo_cfg.get("ent_coef", 0.02),
            vf_coef             = ppo_cfg.get("vf_coef", 0.5),
            max_grad_norm       = ppo_cfg.get("max_grad_norm", 0.5),
            normalize_advantage = True,
            policy_kwargs       = policy_kwargs,
            verbose             = 0,
            tensorboard_log     = "logs/"
        )

    def train_phase(self, phase, reset_timesteps=False):
        timesteps  = phase["timesteps"]
        phase_name = phase["name"]

        # phase set externally via env_method for VecEnv

        tqdm_cb = TqdmCallback(
            total_timesteps=timesteps,
            phase_name=phase_name,
            verbose=1
        )
        metrics_cb = EnhancedMetricsCallback(
            phase_name=phase_name,
            convergence_threshold=0.5
        )
        checkpoint_cb = CheckpointCallback(
            save_freq   = max(timesteps // 10, 10000),
            save_path   = f"results/checkpoints/{phase_name}/",
            name_prefix = "ppo_traffic",
            verbose     = 0
        )

        self.model.learn(
            total_timesteps     = timesteps,
            callback            = [tqdm_cb, metrics_cb, checkpoint_cb],
            tb_log_name         = f"PPO_{phase_name}",
            reset_num_timesteps = reset_timesteps
        )

        self.save(f"results/ppo_model_{phase_name}")

    def train_curriculum(self, phases):
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

            self.train_phase(phase, reset_timesteps=(i == 0))
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


import torch.nn.functional as F

def action_to_soft_routes(action, num_nodes, temperature=1.0):
    routes = {}
    action = np.array(action, dtype=np.float32)
    for i in range(num_nodes):
        nh_raw     = action[i * 2]
        candidates = list(set([
            max(0, int(nh_raw) - 1),
            int(nh_raw) % num_nodes,
            min(num_nodes - 1, int(nh_raw) + 1),
        ]))
        candidates = [c for c in candidates if c != i] or [int(nh_raw) % num_nodes]
        dists      = np.array([abs(c - nh_raw) for c in candidates], dtype=np.float32)
        logits     = -dists / max(temperature, 1e-3)
        probs      = np.exp(logits - logits.max())
        probs     /= probs.sum()
        routes[i]  = {c: float(p) for c, p in zip(candidates, probs)}
    return routes


def select_next_hop(routes, node_id, deterministic=True):
    hops  = list(routes[node_id].keys())
    probs = np.array(list(routes[node_id].values()), dtype=np.float32)
    if deterministic or len(hops) == 1:
        return hops[int(np.argmax(probs))]
    return np.random.choice(hops, p=probs)
