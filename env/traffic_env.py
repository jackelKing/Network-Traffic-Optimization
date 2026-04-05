import numpy as np
import subprocess
import time
import os
import gymnasium as gym
from gymnasium import spaces
from ns3gym import ns3env


class TrafficEnv(gym.Env):
    metadata = {"render_modes": []}

    MAX_NODES = 20

    def __init__(self, config, phase=None):
        super().__init__()
        self.config     = config
        self.topo       = config["topology"]
        self.reward_cfg = config["reward"]
        self.port = config["simulation"].get("port_override", config["simulation"]["port"])

        self._phase = phase or {
            "num_nodes": 4,
            "topo_type": "linear",
            "sim_time":  30.0,
            "name":      "default"
        }

        self.ns3_path    = os.path.expanduser("~/ns-3-dev")
        self.ns3_process = None
        self._ns3env     = None

        self.HIST_LEN = 3
        self.OBS_CORE = 5   # [queue, util, delay, loss, cong]
        obs_size = self.MAX_NODES * (self.OBS_CORE + self.HIST_LEN)
        act_size = self.MAX_NODES * 2

        self._delay_hist = []

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
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
        for _ in range(20):
            time.sleep(0.5)
            if self.ns3_process.poll() is not None:
                break
            try:
                import socket
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(0.3)
                result = s.connect_ex(('localhost', self.port))
                s.close()
                if result == 0:
                    break
            except Exception:
                pass
        time.sleep(0.3)

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
        time.sleep(0.2)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._stop_ns3()
        self._start_ns3()

        self._ns3env = ns3env.Ns3Env(
            port=self.port,
            startSim=False,
            debug=False
        )

        raw_obs          = self._ns3env.reset()
        obs              = self._pad_obs(raw_obs)
        self._prev_reward = None
        self._ep_step    = 0
        self._delay_hist = []
        self._ep_reward  = 0.0
        self._episode   += 1
        return obs, {}

    def step(self, action):
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
        """Pad/trim raw ns3 obs to MAX_NODES*OBS_CORE, then append history."""
        try:
            obs = np.array(raw_obs, dtype=np.float32).flatten()
            if obs.size == 0:
                raise ValueError
        except Exception:
            obs = np.zeros(self.MAX_NODES * self.OBS_CORE, dtype=np.float32)
        core_target = self.MAX_NODES * self.OBS_CORE
        if obs.size < core_target:
            obs = np.concatenate([obs, np.zeros(core_target - len(obs), dtype=np.float32)])
        else:
            obs = obs[:core_target]
        obs = np.clip(obs, 0.0, 1.0)
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=0.0)

        # Latency history — index into CORE obs only (stride=OBS_CORE, offset=2)
        delays_now = obs[2::self.OBS_CORE][:self.MAX_NODES]
        self._delay_hist.append(delays_now.copy())
        if len(self._delay_hist) > self.HIST_LEN:
            self._delay_hist.pop(0)

        hist_cols = []
        for t in range(self.HIST_LEN):
            if t < len(self._delay_hist):
                hist_cols.append(self._delay_hist[-(t + 1)])
            else:
                hist_cols.append(np.zeros(self.MAX_NODES, dtype=np.float32))

        hist_vec = np.concatenate(hist_cols, axis=0).astype(np.float32)
        return np.concatenate([obs, hist_vec], axis=0)

    def _shape_reward(self, obs, done):
        """
        Clean, stable reward for PPO to beat OSPF.

        Key insight from obs inspection:
          - nodes 0-4 in linear-12 always have loss=1.0 (no active flow)
          - util is typically 0.001-0.01 for idle, 0.1-0.6 for active nodes
          - real signal comes from active nodes only

        OSPF baseline: fixed next-hop, no BW adaptation.
        PPO wins by: load balancing + congestion avoidance + multi-path.
        Reward must incentivise exactly those behaviours.
        """
        cfg  = self.reward_cfg
        n    = self._phase["num_nodes"]
        # Only use obs from actual nodes in this phase, not padded zeros
        core = obs[:n * self.OBS_CORE]

        utils      = core[1::self.OBS_CORE][:n]
        delays     = core[2::self.OBS_CORE][:n]
        losses     = core[3::self.OBS_CORE][:n]
        congestion = core[4::self.OBS_CORE][:n]

        # Active node mask — loss=1.0 on util<0.02 means no flow, not real loss
        active = (utils > 0.02).astype(np.float32)
        n_act  = max(float(active.sum()), 1.0)

        avg_util  = float(np.mean(utils))
        avg_delay = float(np.dot(active, delays)     / n_act)
        avg_loss  = float(np.dot(active, losses)     / n_act)
        avg_cong  = float(np.dot(active, congestion) / n_act)

        # ── Core: throughput dominates, penalties are secondary ──
        reward = (
              cfg["throughput_weight"] * avg_util        # maximise traffic flow
            - cfg["delay_weight"]      * avg_delay       # minimise latency
            - cfg["loss_weight"]       * avg_loss        # minimise drops
            - cfg.get("congestion_weight", 0.10) * avg_cong  # avoid hotspots
        )

        # ── Load balance bonus — PPO's PRIMARY edge over OSPF ────
        # OSPF sends all traffic on shortest path → one link saturated
        # PPO learns to spread → all links used → higher total throughput
        if n_act >= 2:
            active_utils = utils[active > 0]
            if len(active_utils) >= 2:
                # Reward even distribution: std=0 → max bonus, std=0.5 → zero
                std_u   = float(np.std(active_utils))
                balance = max(0.0, 0.40 * (1.0 - std_u * 2.0))
                reward += balance

        # ── Congestion avoidance bonus — beats OSPF under heavy load ─
        # OSPF is blind to queue state; PPO observes and re-routes
        if avg_util > 0.05:
            if avg_cong < 0.05:
                reward += 0.20   # excellent: traffic flowing, no congestion
            elif avg_cong < 0.15:
                reward += 0.10   # good
            elif avg_cong < 0.30:
                reward += 0.03   # acceptable

        # ── Throughput excellence — extra push for high utilisation ──
        if avg_util > 0.40:
            reward += 0.20 * avg_util
        elif avg_util > 0.15:
            reward += 0.10 * avg_util

        # ── BW adaptation bonus — reward PPO for using high bandwidth ─
        # PPO action[i*2+1] controls bw_level (0-4 = 1/5/10/50/100Mbps)
        # We infer BW usage from util: high util on active nodes means
        # PPO is pushing more traffic through — reward it
        if avg_util > 0.30 and avg_cong < 0.20:
            # High util + low cong = smart BW allocation (PPO's edge)
            reward += 0.15

        # ── Multi-path bonus — reward routing diversity ───────────────
        # Count how many distinct next-hops are used across active nodes
        # OSPF always uses i→i+1; PPO can use diverse paths
        # We approximate via util variance: higher variance across nodes
        # means more diverse routing (some nodes carry more, some less)
        # but STD penalised above — so this rewards MODERATE spread
        if n_act >= 3:
            active_utils = utils[active > 0]
            if len(active_utils) >= 3:
                util_range = float(np.max(active_utils) - np.min(active_utils))
                # Sweet spot: some spread (0.1-0.4) = multi-path routing
                if 0.05 < util_range < 0.40:
                    reward += 0.10 * (1.0 - abs(util_range - 0.20) / 0.20)

        # ── Multi-path diversity bonus ────────────────────────────────
        # OSPF always routes i→i+1 (one fixed path)
        # PPO can spread traffic across multiple next-hops
        if n_act >= 3:
            active_utils = utils[active > 0]
            if len(active_utils) >= 3:
                util_range = float(np.max(active_utils) - np.min(active_utils))
                if 0.05 < util_range < 0.40:
                    reward += 0.10 * (1.0 - abs(util_range - 0.20) / 0.20)

        # shaping: false in config — disabled intentionally
        return float(reward)

    def close(self):
        self._stop_ns3()
