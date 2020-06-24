import random
import numpy as np

rng = random.SystemRandom()


class NoRegretAgent:
    def __init__(self, actions_limits: tuple, C: float, L: float):
        self.action_limits = actions_limits
        self.D = actions_limits[1] - actions_limits[0]
        self.C = C
        self.L = L
        self.mu = None
        self.delta = None
        self.T = None
        self.action = None
        self.last_direction_choice = 0

        self.reset()

    def reset(self):
        self.reset_learning_rate()
        self.action = 0

    def reset_learning_rate(self):
        self.mu = self.D
        self.delta = 1
        self.T = 0

    def faster_learning_rate(self):
        self.T = 500
        self.update_gradient_ascent_speed()

    def update_gradient_ascent_speed(self):
        self.T += 1
        self.delta = np.sqrt( self.D * self.C / ((self.D + 2) * self.L * self.T ** (1/4) ))
        self.mu = (self.D * self.delta) / (self.C * self.T ** (1/2))

    def get_direction_randomly(self):
        choice = rng.choice([1, -1])
        self.last_direction_choice = choice
        return choice

    def project_action(self, action):
        if action >= self.action_limits[1]:
            return self.action_limits[1]

        if action <= self.action_limits[0]:
            return self.action_limits[0]

        return action

    def predict(self, observation: np.array, reward: float):
        #  Update action according to observation
        gradient = self.last_direction_choice * reward / self.delta
        self.action = self.project_action(self.action + self.mu * gradient)

        self.update_gradient_ascent_speed()

        #  Generate next random choice
        direction = self.get_direction_randomly()
        return self.action + direction * self.delta
