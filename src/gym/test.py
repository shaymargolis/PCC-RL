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
import tensorflow as tf

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.policies import FeedForwardPolicy
from stable_baselines import PPO1
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
pparentdir = os.path.dirname(parentdir)
sys.path.insert(0,pparentdir)

from src.gym.simulate_network.link import Link
from src.gym.simulate_network.network import Network
from src.gym.simulate_network.sender import Sender

from src.gym.simulate_network.simulated_network_env import SimulatedNetworkEnv

from src.gym.simulate_network.single_sender_network import SingleSenderNetwork
from src.common.simple_arg_parse import arg_or_default
from src.gym.no_regret_policy.simple_mlp_policy import SimpleMlpPolicy

history_len = 10
features = "sent latency inflation," + "latency ratio," + "send ratio"


def get_network(senders: [Sender], bw: int):
    #  Create two random identical links
    link1 = Link.generate_random_link()
    link1.bw = bw
    links = [link1]

    #  Init the SimulatedNetwork using the parameters
    return Network(senders, links)

bws = [200, 200, 300, 200, 100, 300, 200]

senders = [
    Sender(
        random.uniform(0.3, 1.5) * bws[0],
        None, 0, features.split(","),
        history_len=history_len
    )
]

import matplotlib.pyplot as plt

networks = [get_network(senders, bw) for bw in bws]
gamma = arg_or_default("--gamma", default=0.99)
output = arg_or_default("--output", default=".")

# env = SimulatedNetworkEnv(senders, networks, history_len=history_len, features=features)
env = SingleSenderNetwork()
model = PPO1.load("./pcc_model_23", env)

#time_data = [float(event["Time"]) for event in data["Events"][1:]]
#rew_data = [float(event["Reward"]) for event in data["Events"][1:]]
#optimal_data = [float(event["Optimal"]) for event in data["Optimal"][1:]]
#send_data = [float(event["Send Rate"]) for event in data["Events"][1:]]
from tqdm import tqdm

pbar = tqdm(total=5000)

plt.figure()
plt.legend()

obs = env.reset(False)
for i in range(5000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)

    inf = info["Events"]

    event = inf[-1]

    if i > 0 and i % 400 == 0:
        times = [event["Time"] for event in inf[-501:]]
        throu = [event["Throughput"] for event in inf[-500:]]
        optim = [8*event["Optimal"] for event in inf[-501:]]
        plt.plot(times[:500], throu, "r-", label="Throughput")
        plt.plot(times, optim, "b--", label="Optimal")
        plt.draw()
        plt.pause(0.01)

    if i > 0 and i % 2500 == 0:
        obs = env.reset(True)
        pbar.update(2500)


    env.render()

plt.show()

##
#   Save the model to the location specified below.
##
default_export_dir = "/tmp/pcc_saved_models/model_A/"
export_dir = arg_or_default("--model-dir", default=default_export_dir)

model.save(export_dir)
