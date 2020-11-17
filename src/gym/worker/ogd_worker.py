import random
import numpy as np
from src.gym.worker.worker import Worker


rng = random.SystemRandom()


class OGDWorker(Worker):
    def __init__(self, env, actions_limits: tuple, C: float, L: float):
        super().__init__(env, actions_limits)

        self.D = actions_limits[1] - actions_limits[0]
        self.C = C
        self.L = L
        self.mu = None
        self.delta = None
        self.T = None

        self.reset()

    def reset(self):
        self.reset_learning_rate()
        self.set_action(0)

    def reset_learning_rate(self):
        self.mu = self.D
        self.delta = 1
        self.T = 0

    def update_gradient_ascent_speed(self):
        if self.T < 100:
            self.T += 1

        # self.delta = np.sqrt( self.D * self.C / ((self.D + 2) * self.L * self.T ** (1/4) ))
        self.delta = 0.01
        self.mu = (self.D * self.delta) / (self.C * self.T ** (1/2))

    def get_direction_randomly(self):
        choice = rng.choice([1, -1])
        return choice

    def step(self, ds) -> float:
        direction = self.get_direction_randomly()

        yield self.action*(1 + direction * self.delta)
        yield True

        obs, reward = ds.data
        gradient = direction * reward / (self.delta*self.action)

        self.set_action(self.action + self.mu * gradient)

        self.update_gradient_ascent_speed()
