import random
import numpy as np
from src.gym.worker.worker import Worker


rng = random.SystemRandom()


class TwoPointOGDWorker(Worker):
    def __init__(self, env, actions_limits: tuple, C: float, L: float, lr=10000, lower_lr=False, delta=0.01, sender_id=0):
        super().__init__(env, actions_limits)

        self.sender_id = sender_id

        self.D = actions_limits[1] - actions_limits[0]
        self.C = C
        self.L = L
        self.mu = None
        self.delta = delta
        self.T = None

        self.lr = lr
        self.lower_lr = lower_lr

        self.reset()

    def reset(self):
        self.reset_learning_rate()
        self.set_action(40)

    def reset_learning_rate(self):
        self.T = self.lr
        self.update_gradient_ascent_speed()

    def update_gradient_ascent_speed(self):
        if self.lower_lr:
            self.T += 1

        # self.delta = 0.01
        self.mu = self.D / (self.L * self.T ** (1/2))

    def get_direction_randomly(self):
        choice = rng.choice([1, -1])
        return choice

    def step(self, ds) -> float:
        _, reward0 = ds.data

        direction = self.get_direction_randomly()

        yield self.action*(1 + direction*self.delta)
        yield True

        _, reward1 = ds.data

        gradient = direction * (reward1 - reward0) / (self.action * self.delta)

        self.set_action(self.action + self.mu * gradient)
        self.update_gradient_ascent_speed()

        yield self.action
        yield True

    # def step(self, ds) -> float:
    #     direction = self.get_direction_randomly()
    #
    #     yield self.action*(1 + direction*self.delta)
    #     yield True
    #
    #     _, reward0 = ds.data
    #
    #     yield self.action*(1 - direction*self.delta)
    #     yield True
    #
    #     _, reward1 = ds.data
    #
    #     gradient = direction * (reward0 - reward1) / (2*self.action*self.delta)
    #
    #     self.set_action(self.action + self.mu * gradient)
    #     self.update_gradient_ascent_speed()
