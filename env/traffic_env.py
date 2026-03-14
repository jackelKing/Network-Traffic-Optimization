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
        self.config    = config
        self.sim_cfg   = config["simulation"]
        self.topo      = config["topology"]

        self.port          = self.sim_cfg["port"]
        self.sim_time      = self.sim_cfg["sim_time"]
        self.num_nodes     = self.sim_cfg["num_nodes"]
        self.topo_type     = self.topo["type"]
        self.data_rate     = self.topo["data_rate"]
        self.delay         = self.topo["delay"]
        self.step_interval = self.sim_cfg.get("step_interval", 0.5)

        self.ns3_path    = os.path.expanduser("~/ns-3-dev")
        self.ns3_process = None
        self._ns3env     = None

        obs_size = self.num_nodes * 3
        act_size = self.num_nodes * 2

        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(obs_size,),
            dtype=np.float32
        )
        # Use float32 for action space so SB3 PPO works natively
        # We cast to int before sending to NS3
        self.action_space = spaces.Box(
            low=0.0,
            high=float(max(self.num_nodes, 5) - 1),
            shape=(act_size,),
            dtype=np.float32
        )

        self._episode     = 0
        self._total_steps = 0

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
            cmd,
            shell=True,
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

        obs = self._ns3env.reset()
        obs = np.clip(np.array(obs, dtype=np.float32), 0.0, 1.0)
        self._episode += 1
        return obs, {}

    def step(self, action):
        # Cast to list of Python ints — ns3gym requires native int
        action_int = [int(round(float(a))) for a in action]

        obs, reward, done, info = self._ns3env.step(action_int)

        obs = np.clip(np.array(obs, dtype=np.float32), 0.0, 1.0)
        self._total_steps += 1

        if info is None:
            info = {}

        return obs, float(reward), bool(done), False, info

    def close(self):
        self._stop_ns3()
