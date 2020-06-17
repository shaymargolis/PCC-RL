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
from tqdm import tqdm
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
    links = [link1]

    #  Init the SimulatedNetwork using the parameters
    return Network(senders, links)

bws = [200]

senders = [
    Sender(
        random.uniform(0.3, 1.5) * bws[0],
        None, 0, features.split(","),
        history_len=history_len
    )
]

import matplotlib.pyplot as plt

networks = [get_network(senders, bw) for bw in bws]

env = SimulatedNetworkEnv(senders, networks, history_len=history_len, features=features)
model = NoRegretAgent(actions_limits=(40, 300), C=300, L=8)

#time_data = [float(event["Time"]) for event in data["Events"][1:]]
#rew_data = [float(event["Reward"]) for event in data["Events"][1:]]
#optimal_data = [float(event["Optimal"]) for event in data["Optimal"][1:]]
#send_data = [float(event["Send Rate"]) for event in data["Events"][1:]]


# TIMES = 51000
TIMES = 7000

pbar = tqdm(total=TIMES / 100)

obs = env.reset()
rewards = [0, 0]
for i in range(TIMES):
    #env.senders[0].set_rate(200)
    action = model.predict(rewards[0])
    env.senders[0].set_rate(action)

    #action = model2.predict(rewards[0])
    #env.senders[1].set_rate(action)
    #env.senders[0].set_rate(int(250-i/2000*250))
    #env.senders[0].set_rate(i / 5000 * 220)
    # print("Sending rate %d Reward %f" % (env.senders[0].rate, rewards[0]))
    obs, rewards, dones, info = env.step([0])

    # if i > 0 and i % 400 == 0:
    #     event = info[0]["Events"][-1]
    #     obs = env.reset()
    #     times = [event["Time"] for event in info[0]["Events"][-501:]]
    #     send = [event["Send Rate"] for event in info[0]["Events"][-500:]]
    #     throu = [event["Throughput"] for event in info[0]["Events"][-500:]]
    #     optim = [8*event["Optimal"] for event in info[0]["Events"][-501:]]
    #     plt.plot(times[:500], throu, "g.", label="Throughput")
    #     plt.plot(times[:500], send, "r.", label="Send rate")
    #     plt.plot(times, optim, "b--", label="Optimal")
    #     plt.draw()
    #     plt.pause(0.01)
    #
    #     for sender in env.senders:
    #         sender.reset_events()

    if i > 0 and i % 10000 == 0:
        obs = env.reset(True)


    env.render()

    if i % 100 == 0:
        pbar.update(1)


fig, ax = plt.subplots(nrows=3, ncols=2)

times = [event["Time"] for event in info[0]["Events"]]
send = [event["Send Rate"] for event in info[0]["Events"]]
throu = [event["Throughput"] for event in info[0]["Events"]]
optim = [8*event["Optimal"] for event in info[0]["Events"]]
latency = [event["Latency Gradient"] for event in info[0]["Events"]]
lat = [event["Latency"] for event in info[0]["Events"]]
loss = [event["Loss Rate"] for event in info[0]["Events"]]
reward = [event["Reward"] for event in info[0]["Events"]]
ax[0][0].title.set_text("Sending rate")
ax[0][0].plot(times, throu, "g.", label="Throughput")
ax[0][0].plot(times, send, "r.", label="Send rate")
ax[0][0].plot(times, optim, "b--", label="Optimal")
ax[0][0].legend()
ax[0][0].grid()

ax[0][1].title.set_text("Latency Gradient")
ax[0][1].plot(times, latency, "r.", label="Latency Gradient")
ax[0][1].legend()
ax[0][1].grid()

ax[1][0].title.set_text("Reward")
ax[1][0].plot(times, reward, "b.", label="Reward")
ax[1][0].legend()
ax[1][0].grid()

ax[1][1].title.set_text("Loss")
ax[1][1].plot(times, loss, "g.", label="Loss")
ax[1][1].legend()
ax[1][1].grid()

ax[2][0].title.set_text("Latency")
ax[2][0].plot(times, lat, "b.", label="Latency")
ax[2][0].plot(times, np.ones(len(times)) * networks[0].links[0].delay, "r--", label="Link latency")
ax[2][0].legend()
ax[2][0].grid()

plt.show()
