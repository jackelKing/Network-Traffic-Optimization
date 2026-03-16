import numpy as np
import subprocess
import time
import os
import gymnasium as gym
from gymnasium import spaces
from ns3gym import ns3env


class Mesh3x3Env(gym.Env):
    """3x3 Mesh Network Environment for PPO Training"""
    metadata = {"render_modes": []}

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.port = config["simulation"]["port"]
        self.sim_time = config["simulation"]["sim_time"]
        
        # Find ns-3-dev path
        self.ns3_path = os.path.expanduser("~/ns-3-dev")
        self.ns3_process = None
        self._ns3env = None

        # Fixed 3x3 mesh: 9 nodes
        self.num_nodes = 9
        
        # Observation: 5 metrics per node (queue, util, delay, loss, congestion)
        obs_size = self.num_nodes * 5
        # Action: routing + bandwidth per node
        act_size = self.num_nodes * 2

        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(obs_size,),
            dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=0.0,
            high=float(self.num_nodes - 1),
            shape=(act_size,),
            dtype=np.float32
        )

        self._prev_reward = None
        self._episode = 0
        self._total_steps = 0

        print(f"Mesh3x3Env initialized:")
        print(f"  Obs space: {self.observation_space.shape}")
        print(f"  Act space: {self.action_space.shape}")

    def _start_ns3(self):
        if self.ns3_process is not None:
            self._stop_ns3()

        cmd = (
            f"{self.ns3_path}/ns3 run "
            f"\"mesh-3x3-trainer/scratch/mesh_3x3_sim"
            f" --simTime={self.sim_time}"
            f" --port={self.port}"
            f" --dataRate={self.config['topology']['data_rate']}"
            f" --delay={self.config['topology']['delay']}"
            f" --stepInterval={self.config['simulation']['step_interval']}\""
        )
        
        print(f"Starting NS-3: {cmd}")
        
        self.ns3_process = subprocess.Popen(
            cmd, shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=self.ns3_path
        )
        time.sleep(3.0)  # Give NS-3 time to start

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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._stop_ns3()
        self._start_ns3()

        self._ns3env = ns3env.Ns3Env(
            port=self.port,
            startSim=False,
            debug=False
        )

        raw_obs = self._ns3env.reset()
        obs = self._process_obs(raw_obs)
        
        self._prev_reward = None
        self._ep_step = 0
        self._ep_reward = 0.0
        self._episode += 1
        
        return obs, {}

    def step(self, action):
        # Convert actions to integers
        action_int = [int(round(float(a))) % self.num_nodes 
                      for a in action]

        obs, raw_reward, done, info = self._ns3env.step(action_int)
        
        obs = self._process_obs(obs)
        shaped_reward = self._shape_reward(obs, done)

        self._total_steps += 1
        self._ep_step += 1
        self._ep_reward += shaped_reward

        if isinstance(info, str):
            info = {"ns3_info": info}
        elif info is None:
            info = {}

        return obs, float(shaped_reward), bool(done), False, info

    def _process_obs(self, raw_obs):
        """Process and normalize observation"""
        obs = np.array(raw_obs, dtype=np.float32)
        expected_size = self.num_nodes * 5
        
        if len(obs) < expected_size:
            obs = np.concatenate([
                obs, 
                np.zeros(expected_size - len(obs), dtype=np.float32)
            ])
        else:
            obs = obs[:expected_size]
        
        return np.clip(obs, 0.0, 1.0)

    def _shape_reward(self, obs, done):
        """Reward shaping for mesh topology"""
        cfg = self.config["reward"]

        # Extract metrics (5 per node)
        queues = obs[0::5]
        utils = obs[1::5]
        delays = obs[2::5]
        losses = obs[3::5]
        congestion = obs[4::5]

        avg_queue = float(np.mean(queues))
        avg_util = float(np.mean(utils))
        avg_delay = float(np.mean(delays))
        avg_loss = float(np.mean(losses))
        avg_cong = float(np.mean(congestion))

        # Base reward
        delay_pen = -cfg["delay_weight"] * avg_delay
        tput_bonus = cfg["throughput_weight"] * avg_util
        loss_pen = -cfg["loss_weight"] * avg_loss
        cong_pen = -cfg.get("congestion_weight", 0.15) * avg_cong
        base = delay_pen + tput_bonus + loss_pen + cong_pen

        # Progress bonus
        progress = 0.0
        if self._prev_reward is not None:
            progress = cfg.get("progress_bonus", 0.1) * (base - self._prev_reward)
        self._prev_reward = base

        # Mesh-specific: Path diversity bonus
        # Reward balanced utilization across mesh links
        if len(utils) > 1:
            util_std = float(np.std(utils))
            path_diversity = 0.15 * (1.0 - util_std)
        else:
            path_diversity = 0.0

        # Congestion avoidance bonus
        cong_avoid = 0.0
        if avg_cong < 0.1 and avg_util > 0.5:
            cong_avoid = 0.3
        elif avg_cong < 0.3 and avg_util > 0.3:
            cong_avoid = 0.15

        # High throughput bonus
        tput_bonus2 = 0.1 * avg_util if avg_util > 0.6 else 0.0

        total_reward = (base + progress + cong_avoid + 
                       path_diversity + tput_bonus2)

        return total_reward

    def close(self):
        self._stop_ns3()
