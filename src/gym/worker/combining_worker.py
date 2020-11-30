import math
import numpy as np

from src.gym.worker.reward_calculator.reward_calculator import RewardCalculator, AverageRewardCalculator
from src.gym.worker.worker import Worker


# @see https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
def exp_normalize(x):
    b = x.max()
    y = np.exp(x - b)
    return y / y.sum()


class CombiningWorker(Worker):
    def __init__(self, action_limits, env, workers: list, min_proba_thresh=0.1, lr=500, lower_lr=False, sender_idx=0, debug=False, const_proba=None, reward_calculator: RewardCalculator=None):
        super().__init__(env, action_limits)

        self.workers = workers
        self.weights: np.array = None
        self.e = None
        self.delta = None
        self.T = None
        self.proba = None
        self.min_proba_thresh = min_proba_thresh
        self.lr = lr
        self.lower_lr = lower_lr

        self.sender_idx = sender_idx
        self.debug = debug
        self.const_proba = const_proba
        self.reward_calculator = reward_calculator if reward_calculator else AverageRewardCalculator()

        self.N = 2

        self.reset()

    def reset(self):
        self.proba = [0.5, 0.5]
        self.weights = np.array([0, 0], dtype=np.int64)
        self.e = 0.5
        self.delta = 0
        self.T = self.lr

    def update_parameters(self):
        if self.lower_lr:
            self.T += 1

        # General equation:
        # self.e = min(1/N, sqrt(log(N)/N*T)
        # In our case (N=2)
        # self.e = min(0.5, sqrt(log(2)/(2*T))
        # transforms to:
        self.e = min(0.5, math.sqrt(np.log(2) / (2*self.T)))

    def calculate_proba(self):
        proba = exp_normalize(self.e * self.weights)

        self.update_parameters()

        proba *= (1 - 2*self.min_proba_thresh)
        proba += self.min_proba_thresh

        self.proba = proba

    def get_proba(self):
        # return [Aurora, OGD]
        # return [1, 0]
        if self.const_proba:
            return self.const_proba

        return self.proba

    def update_weights(self, chosen_index: int, proba: float, reward: float):
        self.weights[chosen_index] += reward / proba

    def set_action(self, new_action):
        [worker.set_action(new_action) for worker in self.workers]
        
        super().set_action(new_action)

    def _debugPrint(self, *args, **kwargs):
        if self.debug:
            print(*args, **kwargs)

    def step(self, ds) -> float:
        ind = np.random.choice(self.N, 1, p=self.get_proba())[0]

        #  Run the selected worker
        reward = yield from self.reward_calculator(ds, self.workers[ind])

        self._debugPrint("[Combining %d] worker %d rewarded %.2f [%.2f, %.2f]" % (self.sender_idx, ind, reward, self.weights[0], self.weights[1]))

        self.update_weights(ind, self.get_proba()[ind], reward)
        self.calculate_proba()

        new_action = self.workers[ind].get_action()
        for worker in self.workers:
            worker.set_action(new_action)
