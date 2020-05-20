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

import src.gym.simulate_network.single_sender_network
from src.common.simple_arg_parse import arg_or_default
from src.gym.no_regret_policy.simple_mlp_policy import SimpleMlpPolicy

history_len = 10
features = "sent latency inflation," + "latency ratio," + "send ratio"

def get_network(senders: [Sender], bw: int):
    #  Create two random identical links
    link1 = Link.generate_random_link()
    link2 = Link(link1.bw, link1.delay, link1.queue_delay, link1.loss_rate)

    links = [link1, link2]

    #  Init the SimulatedNetwork using the parameters
    return Network(senders, links)

bws = [20, 30, 20, 10, 30, 20]

senders = [
    Sender(
        random.uniform(0.3, 1.5) * bws[0],
        None, 0, features.split(","),
        history_len=history_len
    )
]

import matplotlib.pyplot as plt

networks = [get_network(senders, bw) for bw in bws]

model = PPO1.load("./pcc_model_5")
env = SimulatedNetworkEnv(senders, networks, history_len=history_len, features=features)

#time_data = [float(event["Time"]) for event in data["Events"][1:]]
#rew_data = [float(event["Reward"]) for event in data["Events"][1:]]
#optimal_data = [float(event["Optimal"]) for event in data["Optimal"][1:]]
#send_data = [float(event["Send Rate"]) for event in data["Events"][1:]]

plt.figure()
plt.legend()

obs = env.reset()
for i in range(1600 * 410):
    print("AAA", env.run_dur)
    action, _states = model.predict(obs[0])
    obs, rewards, dones, info = env.step(action)

    event = info[0]["Events"][-1]

    plt.plot(event["Time"], event["Throughput"], "r.", label="Throughput")
    plt.plot(event["Time"], event["Optimal"], "b--", label="Optimal")
    plt.draw()

    print("BBB", event["Optimal"])

    if i % 100 == 0:
        plt.pause(0.01)

    env.render()

##
#   Save the model to the location specified below.
##
default_export_dir = "/tmp/pcc_saved_models/model_A/"
export_dir = arg_or_default("--model-dir", default=default_export_dir)

model.save(export_dir)
