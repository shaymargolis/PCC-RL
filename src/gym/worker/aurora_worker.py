from stable_baselines import PPO1

from src.common import config
from src.gym.worker.worker import Worker


class AuroraWorker(Worker):
    def __init__(self, model_file_location: str, env, action_limits):
        super().__init__(env, action_limits)
        self.model = PPO1.load(model_file_location, env)
        self.reset()

    def reset(self):
        self.action = self.action_limits[0]

    def change_action_delta(self, delta: float):
        delta *= config.DELTA_SCALE
        # print("Applying delta %f" % delta)
        if delta >= 0.0:
            self.set_action(self.action * (1.0 + delta))
        else:
            self.set_action(self.action / (1.0 - delta))

    def step(self, ds) -> float:
        obs, reward = ds.data

        self.change_action_delta(self.model.predict(obs)[0][0])
        yield self.action
        yield True

        obs, reward = ds.data

        self.change_action_delta(self.model.predict(obs)[0][0])
        yield self.action
        yield True
