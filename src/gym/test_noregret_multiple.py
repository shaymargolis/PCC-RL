# Copyright 2019 Nathan Jay and Noga Rotman
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import random

import gym
import os
import sys
import inspect
import numpy as np


currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
pparentdir = os.path.dirname(parentdir)
sys.path.insert(0,pparentdir)

from src.gym.simulate_network.link import Link
from src.gym.simulate_network.network import Network
from src.gym.simulate_network.sender import Sender

from src.gym.simulate_network.simulated_network_env import SimulatedNetworkEnv

import src.gym.simulate_network.single_sender_network
from src.common.simple_arg_parse import arg_or_default
from src.gym.no_regret_policy.no_regret_policy import NoRegretAgent
from src.gym.no_regret_policy.gradient_calculating_agent import GradientCalculatingAgent

history_len = 10
features = "sent latency inflation," + "latency ratio," + "send ratio"

def get_network(senders: [Sender], bw: int):
    #  Create two random identical links
    link1 = Link.generate_random_link()
    link1.bw = bw

    links = [link1]

    #  Init the SimulatedNetwork using the parameters
    return Network(senders, links)

bws = [200]

senders = [
    Sender(
        random.uniform(0.3, 1.5) * bws[0],
        None, 0, features.split(","),
        history_len=history_len
    ),
    Sender(
        random.uniform(0.3, 1.5) * bws[0],
        None, 0, features.split(","),
        history_len=history_len
    )
]


import matplotlib.pyplot as plt

networks = [get_network(senders, bw) for bw in bws]

env = SimulatedNetworkEnv(senders, networks, history_len=history_len, features=features)
model = NoRegretAgent(GradientCalculatingAgent(actions_limits=(40, 300), C=11 * 300, L=8))
model2 = NoRegretAgent(GradientCalculatingAgent(actions_limits=(40, 300), C=11 * 300, L=8))
model3 = NoRegretAgent(GradientCalculatingAgent(actions_limits=(40, 300), C=11 * 300, L=8))

#time_data = [float(event["Time"]) for event in data["Events"][1:]]
#rew_data = [float(event["Reward"]) for event in data["Events"][1:]]
#optimal_data = [float(event["Optimal"]) for event in data["Optimal"][1:]]
#send_data = [float(event["Send Rate"]) for event in data["Events"][1:]]

fig, axes = plt.subplots(2, figsize=(10, 12))
senders_axis = axes[0]
sender_ewma_axis = axes[1]

def plot_axis(axis, events_arr):
    colors = [('r', 'g'), ('b', 'm'), ('k', 'y')]

    times = []
    optim = []

    for i in range(len(events_arr)):
        events = events_arr[i]

        times = [event["Time"] for event in events[-501:]]
        optim = [8*event["Optimal"] for event in events[-501:]]
        send = [event["Send Rate"] for event in events[-500:]]
        throu = [event["Throughput"] for event in events[-500:]]

        axis.plot(times[:500], send, colors[i][0] + "-", label="[%d] Sent" % (i+1))
        # axis.plot(times[:500], throu, colors[i][1] + "x", label="[%d] Throughput" % (i+1))

    axis.plot(times, optim, "b--", label="Optimal")
    axis.plot(times, np.array(optim)/2, "r--", label="Optimal/2")

def plot_ewma(axis, event_arr):
    colors = ["r", "b", "g", "p"]
    i = 0
    for events in event_arr:
        times = [event["Time"] for event in events[-500:]]
        ewma = [event["EWMA"] for event in events[-500:]]

        axis.plot(times, ewma, colors[i] + "-", label="Sender" + str(i))

        i += 1


obs = env.reset()
rewards = [0, 0, 0]
for i in range(1600 * 410):
    #env.senders[0].set_rate(200)
    action = model.predict(obs[0], rewards[0])
    action2 = model2.predict(obs[1], rewards[1])
    # action3 = model3.predict(rewards[2])

    # print("[Step %d] actions are" % i, action, action2)

    env.senders[0].set_rate(action)
    env.senders[1].set_rate(action2)
    # env.senders[2].set_rate(action3)

    obs, rewards, dones, info = env.step([0, 0])

    # print("[Step %d] rewards are" % i, rewards)

    if i > 0 and i % 400 == 0:
        obs = env.reset()
        plot_axis(senders_axis, [x["Events"] for x in info])
        plot_ewma(sender_ewma_axis, [x["Events"] for x in info])

        if i == 400:
            senders_axis.legend()

        plt.draw()
        plt.pause(0.1)

    if i > 0 and i % 10000 == 0:
        obs = env.reset(True)


    env.render()
