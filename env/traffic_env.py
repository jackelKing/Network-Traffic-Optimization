import numpy as np
import subprocess
import time
import os
import gymnasium as gym
from gymnasium import spaces
from ns3gym import ns3env


class TrafficEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config):
        super().__init__()
        self.config        = config
        self.sim_cfg       = config["simulation"]
        self.topo          = config["topology"]
        self.reward_cfg    = config["reward"]

        self.port          = self.sim_cfg["port"]
        self.sim_time      = self.sim_cfg["sim_time"]
        self.num_nodes     = self.sim_cfg["num_nodes"]
        self.topo_type     = self.topo["type"]
        self.data_rate     = self.topo["data_rate"]
        self.delay         = self.topo["delay"]
        self.step_interval = self.sim_cfg.get("step_interval", 0.5)

        self.ns3_path      = os.path.expanduser("~/ns-3-dev")
        self.ns3_process   = None
        self._ns3env       = None

        obs_size = self.num_nodes * 3
        act_size = self.num_nodes * 2

        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(obs_size,),
            dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=0.0,
            high=float(max(self.num_nodes, 5) - 1),
            shape=(act_size,),
            dtype=np.float32
        )

        # Reward shaping state
        self._prev_reward    = None
        self._episode        = 0
        self._total_steps    = 0
        self._ep_rewards     = []
        self._best_ep_reward = -np.inf

    # ── NS3 process management ────────────────────────────────
    def _start_ns3(self):
        if self.ns3_process is not None:
            self._stop_ns3()

        cmd = (
            f"{self.ns3_path}/ns3 run "
            f"\"scratch/sim"
            f" --numNodes={self.num_nodes}"
            f" --topoType={self.topo_type}"
            f" --simTime={self.sim_time}"
            f" --port={self.port}"
            f" --dataRate={self.data_rate}"
            f" --delay={self.delay}"
            f" --stepInterval={self.step_interval}\""
        )
        self.ns3_process = subprocess.Popen(
            cmd, shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=self.ns3_path
        )
        time.sleep(2.0)

    def _stop_ns3(self):
        if self._ns3env is not None:
            try:
                self._ns3env.close()
            except Exception:
                pass
            self._ns3env = None
        if self.ns3_process is not None:
            self.ns3_process.terminate()
            try:
                self.ns3_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.ns3_process.kill()
            self.ns3_process = None

    # ── Reset ─────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._stop_ns3()
        self._start_ns3()

        self._ns3env = ns3env.Ns3Env(
            port=self.port,
            startSim=False,
            debug=False
        )

        obs             = self._ns3env.reset()
        obs             = np.clip(np.array(obs, dtype=np.float32), 0.0, 1.0)
        self._prev_reward = None
        self._ep_step     = 0
        self._ep_reward   = 0.0
        self._episode    += 1
        return obs, {}

    # ── Step ──────────────────────────────────────────────────
    def step(self, action):
        action_int = [int(round(float(a))) for a in action]
        obs, raw_reward, done, info = self._ns3env.step(action_int)

        obs = np.clip(np.array(obs, dtype=np.float32), 0.0, 1.0)

        # ── Reward shaping ────────────────────────────────────
        shaped_reward = self._shape_reward(obs, raw_reward, done)

        self._total_steps += 1
        self._ep_step     += 1
        self._ep_reward   += shaped_reward

        if done:
            self._ep_rewards.append(self._ep_reward)
            if self._ep_reward > self._best_ep_reward:
                self._best_ep_reward = self._ep_reward

        if isinstance(info, str):
            info = {"ns3_info": info}
        elif info is None:
            info = {}

        return obs, float(shaped_reward), bool(done), False, info

    def _shape_reward(self, obs, raw_reward, done):
        cfg = self.reward_cfg

        # Extract per-node stats from obs
        # obs layout: [queue0, util0, delay0, queue1, util1, delay1, ...]
        delays     = obs[2::3]   # every 3rd starting at index 2
        utils      = obs[1::3]   # every 3rd starting at index 1
        queues     = obs[0::3]   # every 3rd starting at index 0

        avg_delay  = float(np.mean(delays))
        avg_util   = float(np.mean(utils))
        avg_queue  = float(np.mean(queues))

        # Base reward components
        delay_pen  = - cfg["delay_weight"]      * avg_delay
        tput_bonus =   cfg["throughput_weight"] * avg_util
        loss_pen   = - cfg["loss_weight"]        * avg_queue

        base_reward = delay_pen + tput_bonus + loss_pen

        # Progress bonus: reward improvement over previous step
        progress = 0.0
        if self._prev_reward is not None and cfg.get("shaping", False):
            progress = cfg.get("progress_bonus", 0.1) * (
                base_reward - self._prev_reward
            )

        self._prev_reward = base_reward

        # Utilization bonus: reward high link utilization
        util_bonus = 0.05 * avg_util if avg_util > 0.5 else 0.0

        # Low queue bonus: reward keeping queues short
        queue_bonus = 0.05 * (1.0 - avg_queue)

        shaped = base_reward + progress + util_bonus + queue_bonus
        return shaped

    def close(self):
        self._stop_ns3()
