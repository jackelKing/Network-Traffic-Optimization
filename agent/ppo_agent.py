# PPO Agent wrapper
# Full implementation coming in next step
from stable_baselines3 import PPO

class PPOAgent:
    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log="../logs/"
        )

    def train(self, timesteps):
        self.model.learn(total_timesteps=timesteps)

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = PPO.load(path, env=self.env)
