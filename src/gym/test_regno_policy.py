import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
pparentdir = os.path.dirname(parentdir)
sys.path.insert(0,pparentdir)

import numpy as np
from src.gym.no_regret_policy.no_regret_policy import NoRegretAgent
from src.gym.no_regret_policy.gradient_calculating_agent import GradientCalculatingAgent

from tqdm import tqdm

import matplotlib.pyplot as plt

def convex_func(x):
    # return x
    return - np.sqrt((x - 20) **2 + 5)

def convex_func2(x):
    # return x
    return - np.sqrt((x - 40) **2 + 5)

agent = NoRegretAgent(GradientCalculatingAgent(actions_limits=(-60, 60), C=100, L=1))

pbar = tqdm(total=100)
x = 8

xs = list()
i_list = list()

for i in range(200000):
    if i < 50000:
        x = agent.predict(None, convex_func(x))
    if i > 50000:
        x = agent.predict(None, convex_func2(x))

    xs.append(x)
    i_list.append(i)

    if i % 100 == 0:
        # print(x)
        pbar.update(1)

plt.figure()
plt.plot(i_list, xs, "r.")
plt.show()