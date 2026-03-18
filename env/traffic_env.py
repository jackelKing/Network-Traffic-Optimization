import numpy as np
import subprocess
import time
import os
import gymnasium as gym
from gymnasium import spaces
from ns3gym import ns3env


class TrafficEnv(gym.Env):
    metadata = {"render_modes": []}

    # Max nodes we'll ever use — fixes observation space size
    MAX_NODES = 20

    def __init__(self, config, phase=None):
        super().__init__()
        self.config     = config
        self.topo       = config["topology"]
        self.reward_cfg = config["reward"]
        self.port       = config["simulation"]["port"]

        # Phase can be overridden externally by curriculum
        self._phase     = phase or {
            "num_nodes": 4,
            "topo_type": "linear",
            "sim_time":  8.0,
            "name":      "default"
        }

        self.ns3_path     = os.path.expanduser("~/ns-3-dev")
        self.ns3_process  = None
        self._ns3env      = None

        # Fixed obs/action size based on MAX_NODES
        # so the same model works across all phases
        obs_size = self.MAX_NODES * 3
        act_size = self.MAX_NODES * 2

        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(obs_size,),
            dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=0.0,
            high=float(self.MAX_NODES - 1),
            shape=(act_size,),
            dtype=np.float32
        )

        self._prev_reward = None
        self._episode     = 0
        self._total_steps = 0

    def set_phase(self, phase):
        """Called by curriculum trainer to switch topology."""
        self._phase = phase
        print(f"\n  Switching to phase: {phase['name']}"
              f" | nodes={phase['num_nodes']}"
              f" | topo={phase['topo_type']}"
              f" | simtime={phase['sim_time']}s")

    def _start_ns3(self):
        if self.ns3_process is not None:
            self._stop_ns3()

        p = self._phase
        cmd = (
            f"{self.ns3_path}/ns3 run "
            f"\"scratch/sim"
            f" --numNodes={p['num_nodes']}"
            f" --topoType={p['topo_type']}"
            f" --simTime={p['sim_time']}"
            f" --port={self.port}"
            f" --dataRate={self.topo['data_rate']}"
            f" --delay={self.topo['delay']}"
            f" --stepInterval={self.config['simulation']['step_interval']}\""
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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._stop_ns3()
        self._start_ns3()

        self._ns3env = ns3env.Ns3Env(
            port=self.port,
            startSim=False,
            debug=False
        )

        raw_obs       = self._ns3env.reset()
        obs           = self._pad_obs(raw_obs)
        self._prev_reward = None
        self._ep_step     = 0
        self._ep_reward   = 0.0
        self._episode    += 1
        return obs, {}

    def step(self, action):
        # Only send actions for active nodes
        n = self._phase["num_nodes"]
        action_int = [int(round(float(a))) % n
                      for a in action[:n * 2]]

        obs, raw_reward, done, info = self._ns3env.step(action_int)

        obs           = self._pad_obs(obs)
        shaped_reward = self._shape_reward(obs, done)

        self._total_steps += 1
        self._ep_step     += 1
        self._ep_reward   += shaped_reward

        if isinstance(info, str):
            info = {"ns3_info": info}
        elif info is None:
            info = {}

        return obs, float(shaped_reward), bool(done), False, info

    def _pad_obs(self, raw_obs):
        """Pad observation to MAX_NODES*3 so shape is always fixed."""
        obs    = np.array(raw_obs, dtype=np.float32)
        target = self.MAX_NODES * 3
        if len(obs) < target:
            obs = np.concatenate([obs, np.zeros(target - len(obs),
                                                dtype=np.float32)])
        else:
            obs = obs[:target]
        return np.clip(obs, 0.0, 1.0)

    def _shape_reward(self, obs, done):
        cfg = self.reward_cfg

        # Obs layout: [queue, util, delay, loss, congestion] x numNodes
        queues     = obs[0::5]
        utils      = obs[1::5]
        delays     = obs[2::5]
        losses     = obs[3::5]
        congestion = obs[4::5]

        avg_queue  = float(np.mean(queues))
        avg_util   = float(np.mean(utils))
        avg_delay  = float(np.mean(delays))
        avg_loss   = float(np.mean(losses))
        avg_cong   = float(np.mean(congestion))

        # Base reward
        delay_pen  = - cfg["delay_weight"]                * avg_delay
        tput_bonus =   cfg["throughput_weight"]           * avg_util
        loss_pen   = - cfg["loss_weight"]                  * avg_loss
        cong_pen   = - cfg.get("congestion_weight", 0.15) * avg_cong
        base       = delay_pen + tput_bonus + loss_pen + cong_pen

        # CRITICAL FIX: Zero throughput penalty
        # Prevents PPO from reward hacking by doing nothing
        zero_tput_penalty = 0.0
        if avg_util < 0.01:
            zero_tput_penalty = -2.0  # severe: must route traffic
        elif avg_util < 0.05:
            zero_tput_penalty = -0.5  # moderate penalty

        # Progress bonus
        progress = 0.0
        if self._prev_reward is not None and cfg.get("shaping", False):
            progress = cfg.get("progress_bonus", 0.1) * (
                base - self._prev_reward)
        self._prev_reward = base

        # Congestion avoidance bonus ONLY when traffic is flowing
        cong_avoid = 0.0
        if avg_util > 0.1 and avg_cong < 0.1:
            cong_avoid = 0.3
        elif avg_util > 0.05 and avg_cong < 0.3:
            cong_avoid = 0.1

        # Load balance bonus only when traffic is flowing
        if len(utils) > 1 and avg_util > 0.05:
            balance = 0.1 * (1.0 - (float(np.max(utils))
                                   - float(np.min(utils))))
        else:
            balance = 0.0

        # Throughput bonus
        tput_bonus2 = 0.1 * avg_util if avg_util > 0.1 else 0.0

        return (base + zero_tput_penalty + progress
                + cong_avoid + balance + tput_bonus2)

    def close(self):
        self._stop_ns3()
