import random
import numpy as np

sys_random = random.SystemRandom()

class NoRegretAgent:
    def __init__(self, actions_limits: tuple, index=0):
        self.index = index
        self.action_limits = actions_limits
        self.D = actions_limits[1] - actions_limits[0]
        self.mu = None
        self.delta = None
        self.T = None
        self.action = None
        self.last_direction_choice = 0
        self.grad = list()

        self.reset()

    def reset(self):
        # self.mu = self.D
        self.mu = self.D
        self.delta = 1
        self.T = 0
        self.action = 0

    def update_gradient_ascent_speed(self):
        self.T += 1
        # self.mu = self.D / (self.T ** (3/4))
        self.mu = self.D / (self.T ** (3 / 4))
        self.delta = 1 / (self.T ** (1/4))

    def get_direction_randomly(self):
        choice = sys_random.choice([1, -1])
        self.last_direction_choice = choice
        return choice

    def project_action(self, action):
        if action >= self.action_limits[1]:
            return self.action_limits[1]

        if action <= self.action_limits[0]:
            return self.action_limits[0]

        return action

    def predict(self, observation: float):
        #  Update action according to observation
        gradient = self.last_direction_choice * observation / self.delta
        #if gradient > 5:
        #    gradient = 5
        #if gradient < -5:
        #    gradient = -5

        self.action = self.project_action(self.action - self.mu * gradient)
        self.update_gradient_ascent_speed()

        #  Generate next random choice
        direction = self.get_direction_randomly()
        return self.action + direction * self.delta
