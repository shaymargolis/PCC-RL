import math
import numpy as np
from src.gym.worker.worker import Worker


# @see https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
def exp_normalize(x):
    b = x.max()
    y = np.exp(x - b)
    return y / y.sum()


class CombiningWorker(Worker):
    MIN_PROBA_THRESH = 0.01

    def __init__(self, action_limits, env, workers: list):
        super().__init__(env, action_limits)

        self.workers = workers
        self.weights: np.array = None
        self.e = None
        self.delta = None
        self.T = None
        self.proba = None

        self.N = 2

        self.reset()

    def reset(self):
        self.proba = [0.5, 0.5]
        self.weights = np.array([0, 0], dtype=np.int64)
        self.e = 0.5
        self.delta = 0
        self.T = 0
        self.T = 5000

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

        proba *= (1 - 2*CombiningWorker.MIN_PROBA_THRESH)
        proba += CombiningWorker.MIN_PROBA_THRESH

        self.proba = proba

    def get_proba(self):
        # return [0, 1]
        return self.proba

    def update_weights(self, chosen_index: int, proba: float, reward: float):
        self.weights[chosen_index] += reward / proba

    def step(self, ds) -> float:
        ind = np.random.choice(self.N, 1, p=self.get_proba())[0]

        #  Run the selected worker
        gen = self.workers[ind].step(ds)
        reward = 0
        reward_steps = 0

        while True:
            try:
                action = next(gen)
            except StopIteration as ex:
                break

            yield action
            yield True

            reward += ds.data[1]
            reward_steps += 1

            #  Continue to next action
            next(gen)

        reward = reward / reward_steps if reward_steps >= 1 else 0

        self.update_weights(ind, self.get_proba()[ind], reward)
        self.calculate_proba()

        new_action = self.workers[ind].get_action()
        for worker in self.workers:
            worker.set_action(new_action)