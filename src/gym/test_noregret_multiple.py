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

history_len = 10
features = "sent latency inflation," + "latency ratio," + "send ratio"

def get_network(senders: [Sender], bw: int):
    #  Create two random identical links
    link1 = Link.generate_random_link()
    link1.bw = bw
    link2 = Link(bw, link1.delay, link1.queue_delay, link1.loss_rate)

    links = [link1, link2]

    #  Init the SimulatedNetwork using the parameters
    return Network(senders, links)

bws = [200, 300, 200, 400, 100, 300, 600]

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
model = NoRegretAgent(actions_limits=(40, 1000))
model2 = NoRegretAgent(actions_limits=(40, 1000))

#time_data = [float(event["Time"]) for event in data["Events"][1:]]
#rew_data = [float(event["Reward"]) for event in data["Events"][1:]]
#optimal_data = [float(event["Optimal"]) for event in data["Optimal"][1:]]
#send_data = [float(event["Send Rate"]) for event in data["Events"][1:]]

fig, axes = plt.subplots(3, figsize=(10, 12))
sender1_axis = axes[0]
sender2_axis = axes[1]
sender_ewma_axis = axes[2]

def plot_axis(axis, events):
    times = [event["Time"] for event in events[-501:]]
    send = [event["Send Rate"] for event in info[0]["Events"][-500:]]
    throu = [event["Throughput"] for event in events[-500:]]
    optim = [8*event["Optimal"] for event in events[-501:]]
    axis.plot(times[:500], send, "g-", label="Sent")
    axis.plot(times[:500], throu, "r-", label="Throughput")
    axis.plot(times, optim, "b--", label="Optimal")

def plot_ewma(axis, event_arr):
    colors = ["r", "b", "g", "p"]
    i = 0
    for events in event_arr:
        times = [event["Time"] for event in events[-500:]]
        ewma = [event["EWMA"] for event in events[-500:]]

        axis.plot(times, ewma, colors[i] + "-", label="Sender" + str(i))

        i += 1


obs = env.reset()
rewards = [0, 0]
for i in range(1600 * 410):
    #env.senders[0].set_rate(200)
    action = model.predict(rewards[0])
    action2 = model2.predict(rewards[1])

    # print("[Step %d] actions are" % i, action, action2)

    env.senders[0].set_rate(action)
    env.senders[1].set_rate(action2)

    obs, rewards, dones, info = env.step([0, 0])

    # print("[Step %d] rewards are" % i, rewards)

    if i > 0 and i % 400 == 0:
        obs = env.reset()
        plot_axis(sender1_axis, info[0]["Events"])
        plot_axis(sender2_axis, info[1]["Events"])
        plot_ewma(sender_ewma_axis, [info[0]["Events"], info[1]["Events"]])
        plt.draw()
        plt.pause(0.1)

    if i > 0 and i % 10000 == 0:
        obs = env.reset(True)


    env.render()
