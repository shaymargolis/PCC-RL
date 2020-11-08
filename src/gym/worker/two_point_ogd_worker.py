import random
import numpy as np
from src.gym.worker.worker import Worker


rng = random.SystemRandom()


class TwoPointOGDWorker(Worker):
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
        self.set_action(40)

    def reset_learning_rate(self):
        self.T = 1
        self.update_gradient_ascent_speed()

    def update_gradient_ascent_speed(self):
        if self.T < 50:
            self.T += 1

        self.T += 1
        self.delta = self.D / (self.T ** (1/2))
        self.mu = self.D / (self.L * self.T ** (1/2))

    def get_direction_randomly(self):
        choice = rng.choice([1, -1])
        return choice

    def step(self, ds) -> float:
        direction = self.get_direction_randomly()

        yield self.action + direction * self.delta
        yield True

        _, reward0 = ds.data

        yield self.action - direction * self.delta
        yield True

        _, reward1 = ds.data

        gradient = direction * (reward0 - reward1) / (2*self.delta)

        self.set_action(self.action + self.mu * gradient)

        self.update_gradient_ascent_speed()
