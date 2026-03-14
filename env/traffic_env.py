import numpy as np
import subprocess
import time
import os
from ns3gym import ns3env

class TrafficEnv:
    """
    Wraps ns3gym's Ns3Env to provide a clean interface for PPO.
    Handles NS3 process launch, reset, step, and cleanup.
    """

    def __init__(self, config):
        self.config  = config
        self.sim_cfg = config["simulation"]
        self.topo    = config["topology"]

        self.port        = self.sim_cfg["port"]
        self.sim_time    = self.sim_cfg["sim_time"]
        self.num_nodes   = self.sim_cfg["num_nodes"]
        self.topo_type   = self.topo["type"]
        self.data_rate   = self.topo["data_rate"]
        self.delay       = self.topo["delay"]
        self.step_interval = self.sim_cfg.get("step_interval", 0.5)

        self.ns3_path    = os.path.expanduser("~/ns-3-dev")
        self.ns3_process = None
        self.env         = None

        self._episode     = 0
        self._total_steps = 0

    # ── Launch NS3 process ────────────────────────────────────
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
        time.sleep(1.5)  # give NS3 time to start and open ZMQ socket

    # ── Stop NS3 process ──────────────────────────────────────
    def _stop_ns3(self):
        if self.ns3_process is not None:
            self.ns3_process.terminate()
            try:
                self.ns3_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.ns3_process.kill()
            self.ns3_process = None
        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass
            self.env = None

    # ── Reset ─────────────────────────────────────────────────
    def reset(self):
        self._stop_ns3()
        self._start_ns3()

        self.env = ns3env.Ns3Env(
            port=self.port,
            startSim=False,
            debug=False
        )

        self.observation_space = self.env.observation_space
        self.action_space      = self.env.action_space

        obs = self.env.reset()
        self._episode += 1
        return np.array(obs, dtype=np.float32), {}

    # ── Step ──────────────────────────────────────────────────
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._total_steps += 1
        obs = np.array(obs, dtype=np.float32)

        truncated = False
        if info is None:
            info = {}

        return obs, float(reward), bool(done), truncated, info

    # ── Close ─────────────────────────────────────────────────
    def close(self):
        self._stop_ns3()

    # ── Properties for SB3 compatibility ─────────────────────
    def __getattr__(self, name):
        if name in ("observation_space", "action_space"):
            raise AttributeError(f"{name} not set yet — call reset() first")
        raise AttributeError(name)
