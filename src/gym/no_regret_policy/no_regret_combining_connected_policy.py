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


# @see https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
def exp_normalize(x):
    b = x.max()
    y = np.exp(x - b)
    return y / y.sum()



class NoRegretCombiningConnectPolicy(Agent):
    MIN_PROBA_THRESH = 0.1

    def __init__(self, aurora_policy: AuroraPolicy, gradient_calculating_agent: GradientCalculatingAgent):
        super().__init__()

        self.gradient_calculating_agent = gradient_calculating_agent

        no_regret_advisor = NoRegretAdvisor(gradient_calculating_agent)

        self.agents: [Agent] = [
            aurora_policy,
            no_regret_advisor
        ]

        self.weights: np.array = None
        self.e = None
        self.delta = None
        self.T = None
        self.selected_index = None
        self.actions = None
        self.proba = None

        # Number of possible actions -
        # OGD or RL = 2
        self.N = 2

        self.reset()

    def reset(self):
        self.proba = [0.5, 0.5]
        self.actions = [0, 0]
        self.selected_index = 0
        self.weights = np.array([0, 0], dtype=np.int64)
        self.e = 0.5
        self.delta = 0
        self.T = 0

    def update_parameters(self):
        if self.T < 100:
            self.T += 1
        # self.T += 1

        # General equation:
        # self.e = min(1/N, sqrt(log(N)/N*T)
        # In our case (N=2)
        # self.e = min(0.5, sqrt(log(2)/(2*T))
        # transforms to:
        self.e = min(0.5, math.sqrt(np.log(2) / (2*self.T)))

    def calculate_proba(self):
        proba = exp_normalize(-1 * self.e * self.weights)

        self.update_parameters()

        proba *= (1 - 2*NoRegretCombiningConnectPolicy.MIN_PROBA_THRESH)
        proba += NoRegretCombiningConnectPolicy.MIN_PROBA_THRESH

        self.proba = proba

    def get_proba(self):
        # return [0, 1]
        return self.proba

    def update_weights(self, chosen_index: int, proba: float, reward: float):
        self.weights[chosen_index] += reward / 100 / proba

    def predict(self, observation: np.array, reward: float):
        #  Step 0 - Update both agents
        for agent in self.agents:
            agent.action = self.gradient_calculating_agent.action

        #  Step 1 - Calculate next move and reward the chosen one
        self.actions = [agent.predict(observation, reward) for agent in self.agents]

        if self.T != 0:
            self.update_weights(self.selected_index, self.get_proba()[self.selected_index], reward)
            self.calculate_proba()
        else:
            self.update_parameters()

        #  Step 2 - Choose next agent
        self.selected_index = np.random.choice(self.N, 1, p=self.get_proba())[0]
        # self.selected_index = 1
        self.gradient_calculating_agent.action = self.actions[self.selected_index]

        return self.gradient_calculating_agent.predict(observation, reward)
