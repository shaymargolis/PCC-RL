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
from src.gym.worker.aurora_worker import AuroraWorker
from src.gym.worker.ogd_worker import OGDWorker
from src.gym.worker.two_point_ogd_worker import TwoPointOGDWorker
from src.gym.worker.combining_worker import CombiningWorker
from src.gym.worker.worker_runner import WorkerRunner

from src.gym.simulate_network.link import Link
from src.gym.simulate_network.network import Network
from src.gym.simulate_network.sender import Sender

from src.gym.simulate_network.single_sender_network import SingleSenderNetwork

from src.common.simple_arg_parse import arg_or_default
from src.gym.no_regret_policy.no_regret_policy import NoRegretAgent
from src.gym.aurora_policy.aurora_policy import AuroraPolicy
from src.gym.no_regret_policy.no_regret_combining_connected_policy import NoRegretCombiningConnectPolicy
from src.gym.no_regret_policy.gradient_calculating_agent import GradientCalculatingAgent

bws = [100, 240] # [100, 240] # [200, 300, 200, 300]
index = 0

def get_network():
    global index

    while True:
        link1 = Link.generate_link(bws[index], 0.2, 6, 0)
        links = [link1]

        yield links
        index = 1 - index


import matplotlib.pyplot as plt

env = SingleSenderNetwork(get_network())
# model = AuroraWorker("./rand_model_12", env, (40, 300))
model = CombiningWorker(
    (40, 300),
    env,
    [
        AuroraWorker("./rand_model_12", env, (40, 300)),
        TwoPointOGDWorker(env, (40, 300), C=11 * 300, L=20)
    ]
)

#time_data = [float(event["Time"]) for event in data["Events"][1:]]
#rew_data = [float(event["Reward"]) for event in data["Events"][1:]]
#optimal_data = [float(event["Optimal"]) for event in data["Optimal"][1:]]
#send_data = [float(event["Send Rate"]) for event in data["Events"][1:]]


TIMES = 5000

pbar = tqdm(total=TIMES / 100)

significance = []

obs = env.reset()
reward = 0

wr = WorkerRunner([model], [obs], [reward])

for i in range(TIMES):
    action = wr.start_step()

    #env.senders[0].set_rate(40)
    env.senders[0].set_rate(action[0])

    #action = model2.predict(obs[0], rewards[0])
    #env.senders[1].set_rate(action)
    #env.senders[0].set_rate(int(250-i/2000*250))
    #env.senders[0].set_rate(i / 5000 * 220)
    #env.senders[0].set_rate(210)
    # print("Sending rate %d Reward %f" % (env.senders[0].rate, rewards[0]))
    obs, reward, dones, info = env.step([0])

    wr.finish_step([obs], [reward])

    significance.append(model.get_proba()[:])

    # if i > 0 and i % 400 == 0:
    #     event = info["Events"][-1]
    #     obs = env.reset()
    #     times = [event["Time"] for event in info["Events"][-501:]]
    #     send = [event["Send Rate"] for event in info["Events"][-500:]]
    #     throu = [event["Throughput"] for event in info["Events"][-500:]]
    #     optim = [8*event["Optimal"] for event in info["Events"][-501:]]
    #     plt.plot(times[:500], throu, "g.", label="Throughput")
    #     plt.plot(times[:500], send, "r.", label="Send rate")
    #     plt.plot(times, optim, "b--", label="Optimal")
    #     plt.draw()
    #     plt.pause(0.01)
    #
    #     for sender in env.senders:
    #         sender.reset_events()

    #if i > 0 and i % 5500 == 0:
    #    model.faster_learning_rate()

    if i > 0 and i % 400 == 0:
        obs = env.reset(True)


    env.render()

    if i % 100 == 0:
        pbar.update(1)


fig, ax = plt.subplots(nrows=3, ncols=2)

times = [event["Time"] for event in info["Events"]]
send = [event["Send Rate"] for event in info["Events"]]
throu = [event["Throughput"] for event in info["Events"]]
optim = [8*event["Optimal"] for event in info["Events"]]
latency = [event["Latency Gradient"] for event in info["Events"]]
lat = [event["Latency"] for event in info["Events"]]
loss = [event["Loss Rate"] for event in info["Events"]]
reward = [event["Reward"] for event in info["Events"]]
ax[0][0].title.set_text("Sending rate")
ax[0][0].plot(times, throu, "g.", label="Throughput")
ax[0][0].plot(times, send, "r-", label="Send rate")
ax[0][0].plot(times, optim, "b--", label="Optimal")
ax[0][0].legend()
ax[0][0].grid()

ax[0][1].title.set_text("Reward")
ax[0][1].plot(times, reward, "b.", label="Reward")
ax[0][1].legend()
ax[0][1].grid()

ax[1][0].title.set_text("Significance")
ax[1][0].plot(times, list(map(lambda x: x[0], significance)), "b-", label="Aurora Sig")
ax[1][0].plot(times, list(map(lambda x: x[1], significance)), "g-", label="OGD Sig")
ax[1][0].legend()
ax[1][0].grid()

ax[1][1].title.set_text("Loss")
ax[1][1].plot(times, loss, "g.", label="Loss")
ax[1][1].legend()
ax[1][1].grid()

ax[2][0].title.set_text("Latency")
ax[2][0].plot(times, lat, "b.", label="Latency")
ax[2][0].plot(times, np.ones(len(times)) * env.net.links[0].delay, "r--", label="Link latency")
ax[2][0].legend()
ax[2][0].grid()

plt.show()
