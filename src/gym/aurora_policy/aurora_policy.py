import numpy as np
from stable_baselines import PPO1


class AuroraPolicy:
    def __init__(self, model_file_location: str):
        self.model = PPO1.load(model_file_location, None)
        self.reset()

    def reset(self):
        pass

    def predict(self, observation: np.array, reward: float):
        return self.model.predict(observation)
