import math
import numpy as np
from src.gym.aurora_policy.aurora_policy import AuroraPolicy
from src.gym.no_regret_policy.agent import Agent
from src.gym.no_regret_policy.gradient_calculating_agent import GradientCalculatingAgent
from src.gym.no_regret_policy.no_regret_policy import NoRegretAgent

class NoRegretAdvisor:
    def __init__(self, gradient_calculating_agent: GradientCalculatingAgent):
        self.gradient_calculating = gradient_calculating_agent
        self.action = None

    def predict(self, observation: np.array, reward: float):
        #  Update action according to observation
        gradient = self.gradient_calculating.get_gradient_estimate(reward)
        return self.gradient_calculating.project_action(self.gradient_calculating.action + self.gradient_calculating.mu * gradient)

class NoRegretCombiningConnectPolicy(Agent):
    MIN_PROBA_THRESH = 0.001

    def __init__(self, aurora_policy: AuroraPolicy, gradient_calculating_agent: GradientCalculatingAgent):
        super().__init__()

        self.gradient_calculating_agent = gradient_calculating_agent

        no_regret_advisor = NoRegretAdvisor(gradient_calculating_agent)

        self.agents: [Agent] = [
            aurora_policy,
            no_regret_advisor
        ]

        self.weights: np.array = None
        self.mu = None
        self.delta = None
        self.T = None
        self.selected_index = None
        self.actions = None

        # Number of possible actions -
        # OGD or RL = 2
        self.N = 2

        self.reset()

    def reset(self):
        self.actions = [0, 0]
        self.selected_index = 0
        self.weights = np.array([0, 0], dtype=np.int64)
        self.mu = 0
        self.delta = 0
        self.T = 0

    def update_parameters(self):
        self.T += 1
        # General equation:
        # self.mu = math.sqrt((2 * np.log(self.N)) / (self.T * self.N))
        # In our case (N=2)
        # self.mu = math.sqrt((2 * np.log(2)) / (self.T * 2))
        # transforms to:
        self.mu = math.sqrt(np.log(2) / self.T)

    def calculate_proba(self):
        proba = np.exp(self.mu * self.weights) / np.sum(np.exp(self.mu * self.weights))

        proba *= (1 - NoRegretCombiningConnectPolicy.MIN_PROBA_THRESH)
        proba += 0.5 * NoRegretCombiningConnectPolicy.MIN_PROBA_THRESH

        return [0, 1]
        return proba

    def update_weights(self, chosen_index: int, proba: float, reward: float):
        self.weights[chosen_index] += reward / 1000 / proba

    def predict(self, observation: np.array, reward: float):
        #  Step 0 - Update both agents
        for agent in self.agents:
            agent.action = self.gradient_calculating_agent.action

        #  Step 1 - Calculate next move and reward the chosen one
        self.actions = [agent.predict(observation, reward) for agent in self.agents]

        if self.T != 0:
            self.update_weights(self.selected_index, self.calculate_proba()[self.selected_index], reward)

        self.update_parameters()

        #  Step 2 - Choose next agent
        self.selected_index = np.random.choice(self.N, 1, p=self.calculate_proba())[0]
        self.gradient_calculating_agent.action = self.actions[self.selected_index]

        return self.gradient_calculating_agent.predict(observation, reward)
