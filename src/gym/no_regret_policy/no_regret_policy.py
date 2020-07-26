import numpy as np

from src.gym.no_regret_policy.agent import Agent
from src.gym.no_regret_policy.gradient_calculating_agent import GradientCalculatingAgent


class NoRegretAgent(Agent):
    def __init__(self, gradient_calculating: GradientCalculatingAgent):
        super().__init__()

        self.gradient_calculating = gradient_calculating

    def predict(self, observation: np.array, reward: float):
        #  Update action according to observation
        gradient = self.gradient_calculating.get_gradient_estimate(reward)
        self.gradient_calculating.action = self.gradient_calculating.project_action(self.gradient_calculating.action + self.gradient_calculating.mu * gradient)

        return self.gradient_calculating.predict(observation, reward)
