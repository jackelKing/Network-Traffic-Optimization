"""
Traffic prediction module — short-term delay forecasting.
Uses a 2-layer LSTM trained online on the agent's own experience.
Predicted delay is prepended to obs, giving PPO look-ahead ability.

Usage:
    predictor = TrafficPredictor(input_dim=20, hidden=64, horizon=1)
    pred_delay = predictor.predict(delay_history_tensor)   # (1, horizon, 20)
    predictor.update(delay_history_tensor, next_delays)    # online learning
"""

import torch
import torch.nn as nn
import numpy as np
from collections import deque


class TrafficPredictor(nn.Module):
    def __init__(self, input_dim=20, hidden=64, horizon=1, lr=1e-3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden    = hidden
        self.horizon   = horizon

        self.lstm = nn.LSTM(input_dim, hidden, num_layers=2,
                            batch_first=True, dropout=0.1)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, input_dim * horizon),
        )

        self.optim  = torch.optim.Adam(self.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self._buffer = deque(maxlen=512)   # (seq, target) pairs
        self._device = torch.device("cpu")

    def forward(self, x):
        """x: (B, seq_len, input_dim) → (B, horizon, input_dim)"""
        out, _ = self.lstm(x)
        pred   = self.head(out[:, -1, :])
        return pred.view(-1, self.horizon, self.input_dim)

    @torch.no_grad()
    def predict(self, delay_history):
        """
        delay_history: numpy (seq_len, num_nodes) or tensor
        Returns numpy (horizon, num_nodes) — predicted next delays
        """
        if isinstance(delay_history, np.ndarray):
            delay_history = torch.tensor(delay_history, dtype=torch.float32)
        x = delay_history.unsqueeze(0).to(self._device)  # (1, seq, N)
        return self.forward(x).squeeze(0).cpu().numpy()   # (horizon, N)

    def store(self, seq_np, target_np):
        """Store (seq, target) pair for online training."""
        self._buffer.append((
            torch.tensor(seq_np,    dtype=torch.float32),
            torch.tensor(target_np, dtype=torch.float32),
        ))

    def update(self, batch_size=32):
        """One gradient step on a random mini-batch from the buffer."""
        if len(self._buffer) < batch_size:
            return None

        idxs   = np.random.choice(len(self._buffer), batch_size, replace=False)
        seqs   = torch.stack([self._buffer[i][0] for i in idxs])    # (B, seq, N)
        targets= torch.stack([self._buffer[i][1] for i in idxs])    # (B, N)
        targets= targets.unsqueeze(1)                                 # (B, 1, N)

        self.optim.zero_grad()
        preds = self.forward(seqs)
        loss  = self.loss_fn(preds, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optim.step()
        return float(loss.item())


class PredictiveObsWrapper:
    """
    Wraps TrafficEnv to append predicted next-step delays to obs,
    giving the PPO agent proactive (not reactive) routing signals.
    """
    def __init__(self, env, seq_len=3, num_nodes=20):
        self.env        = env
        self.seq_len    = seq_len
        self.num_nodes  = num_nodes
        self.predictor  = TrafficPredictor(input_dim=num_nodes)
        self._hist      = deque(maxlen=seq_len)
        self._prev_delays = None

        # Extend obs space to include predicted delays
        import gymnasium as gym
        import numpy as np
        base_shape = env.observation_space.shape[0]
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0,
            shape=(base_shape + num_nodes,),
            dtype=np.float32
        )
        self.action_space = env.action_space

    def reset(self, **kwargs):
        obs, info     = self.env.reset(**kwargs)
        self._hist    = deque(maxlen=self.seq_len)
        delays        = obs[2::5][:self.num_nodes]
        self._hist.append(delays.copy())
        return self._augment(obs), info

    def step(self, action):
        obs, reward, done, trunc, info = self.env.step(action)
        delays = obs[2::5][:self.num_nodes]

        # Online train: store (hist, current_delays) pair
        if len(self._hist) == self.seq_len and self._prev_delays is not None:
            seq = np.array(list(self._hist), dtype=np.float32)
            self.predictor.store(seq, delays.copy())
            self.predictor.update()

        self._prev_delays = delays.copy()
        self._hist.append(delays.copy())

        return self._augment(obs), reward, done, trunc, info

    def _augment(self, obs):
        import numpy as np
        if len(self._hist) == self.seq_len:
            seq = np.array(list(self._hist), dtype=np.float32)
            pred = self.predictor.predict(seq)[0]    # (num_nodes,)
        else:
            pred = np.zeros(self.num_nodes, dtype=np.float32)
        return np.concatenate([obs, pred], axis=0).astype(np.float32)

    def close(self):
        return self.env.close()

    def set_phase(self, phase):
        return self.env.set_phase(phase)
