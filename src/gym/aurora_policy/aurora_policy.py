import numpy as np
from stable_baselines import PPO1
from src.common import sender_obs, config
from src.gym.simulate_network.constants import *


class AuroraPolicy:
    def __init__(self, model_file_location: str, env):
        self.model = PPO1.load(model_file_location, env)
        self.reset()

    def reset(self):
        self.action = 40
        pass

    def change_action_delta(self, delta: float):
        delta *= config.DELTA_SCALE
        # print("Applying delta %f" % delta)
        if delta >= 0.0:
            self.change_action(self.action * (1.0 + delta))
        else:
            self.change_action(self.action / (1.0 - delta))

    def change_action(self, new_action: float):
        self.action = new_action
        # print("Attempt to set new rate to %f (min %f, max %f)" % (new_rate, MIN_RATE, MAX_RATE))
        if self.action > MAX_RATE:
            self.action = MAX_RATE
        if self.action < MIN_RATE:
            self.action = MIN_RATE

    def predict(self, observation: np.array, reward: float):
        self.change_action_delta(self.model.predict(observation)[0][0])

        return self.action
