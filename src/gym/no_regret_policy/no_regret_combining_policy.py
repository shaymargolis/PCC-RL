import math
import numpy as np
from src.gym.aurora_policy.aurora_policy import AuroraPolicy
from src.gym.no_regret_policy.agent import Agent
from src.gym.no_regret_policy.no_regret_policy import NoRegretAgent


class NoRegretCombiningPolicy(Agent):
    MIN_PROBA_THRESH = 0.001

    def __init__(self, aurora_policy: AuroraPolicy, no_regret_policy: NoRegretAgent):
        super().__init__()

        self.agents: [Agent] = [
            aurora_policy,
            no_regret_policy
        ]

        self.weights: np.array = None
        self.mu = None
        self.T = None
        self.selected_index = None
        self.actions = None

        # Number of possible actions -
        # OGD or RL = 2
        self.N = 2

        self.reset()

    def reset(self):
        self.actions = None
        self.selected_index = 0
        self.weights = np.array([0, 0], dtype=np.int64)
        self.mu = 0
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

        proba *= (1 - NoRegretCombiningPolicy.MIN_PROBA_THRESH)
        proba += 0.5 * NoRegretCombiningPolicy.MIN_PROBA_THRESH

        # return [1,0]
        return proba

    def update_weights(self, chosen_index: int, proba: float, reward: float):
        self.weights[chosen_index] += reward / 1000 / proba

    def predict(self, observation: np.array, reward: float):
        #  Step 0 - Initialize actions
        if self.T == 0:
            self.actions = [agent.predict(observation, reward) for agent in self.agents]

        #  Step 1 - Reward the chosen one
        if self.T != 0:
            self.actions[self.selected_index] = self.agents[self.selected_index].predict(observation, reward)
            self.update_weights(self.selected_index, self.calculate_proba()[self.selected_index], reward)

        self.update_parameters()

        #  Step 2 - Choose next agent
        self.selected_index = np.random.choice(self.N, 1, p=self.calculate_proba())[0]

        return self.actions[self.selected_index]
