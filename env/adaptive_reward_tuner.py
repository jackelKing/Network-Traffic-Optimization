"""
Adaptive Reward Tuner — keeps PPO competitive under distribution shift.

Every `update_interval` episodes it:
1. Measures rolling average of delay, loss, util, congestion.
2. Computes normalised 'distance to target KPI' for each metric.
3. Shifts the corresponding reward weight proportionally.
4. Re-normalises weights to sum to 1.

Plug in via TrafficEnv by calling:
    tuner = AdaptiveRewardTuner(env.reward_cfg)
    env.reward_cfg = tuner.step(metrics_dict)  # call after each episode
"""

import numpy as np
from collections import deque


class AdaptiveRewardTuner:
    # Target KPI values (normalised 0-1)
    TARGETS = {
        "delay":      0.05,   # want avg normalised delay < 5%
        "loss":       0.02,   # want avg packet loss < 2%
        "util":       0.60,   # want avg utilisation > 60%
        "congestion": 0.10,   # want avg congestion < 10%
    }
    WEIGHT_KEYS = {
        "delay":      "delay_weight",
        "loss":       "loss_weight",
        "util":       "throughput_weight",
        "congestion": "congestion_weight",
    }
    ALPHA = 0.05    # step size for weight update
    CLIP  = (0.05, 0.60)   # weight bounds

    def __init__(self, reward_cfg, update_interval=10, window=20):
        self.cfg             = dict(reward_cfg)
        self.update_interval = update_interval
        self._ep_count       = 0
        self._history        = {k: deque(maxlen=window)
                                 for k in self.TARGETS}

    def record(self, metrics):
        """metrics: dict with keys delay, loss, util, congestion (floats)"""
        for k in self.TARGETS:
            if k in metrics:
                self._history[k].append(float(metrics[k]))

    def step(self):
        """
        Called once per episode. Every update_interval episodes,
        adjust weights and return updated reward_cfg dict.
        """
        self._ep_count += 1
        if self._ep_count % self.update_interval != 0:
            return self.cfg

        gradients = {}
        for metric, target in self.TARGETS.items():
            hist = self._history[metric]
            if not hist:
                continue
            avg = float(np.mean(hist))
            # How far are we from target, signed
            if metric == "util":
                gap = target - avg          # positive → we're below target → need more weight
            else:
                gap = avg - target          # positive → we're above target → need more weight
            gradients[metric] = float(np.clip(gap, -0.5, 0.5))

        for metric, grad in gradients.items():
            key  = self.WEIGHT_KEYS[metric]
            cur  = float(self.cfg.get(key, 0.1))
            new  = cur + self.ALPHA * grad
            self.cfg[key] = float(np.clip(new, *self.CLIP))

        # Re-normalise
        total = sum(self.cfg[v] for v in self.WEIGHT_KEYS.values())
        for v in self.WEIGHT_KEYS.values():
            self.cfg[v] = float(self.cfg[v]) / total

        return self.cfg

    def summary(self):
        return {v: round(self.cfg[v], 4) for v in self.WEIGHT_KEYS.values()}
