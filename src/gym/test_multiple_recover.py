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

bws = [200, 200, 300, 200, 100, 300, 200] # [200, 300, 200, 300]
index = 0

def get_network():
    global index

    while True:
        link1 = Link.generate_link(bws[index], 0.2, 6, 0)
        links = [link1]

        yield links
        index += 1
        if index >= len(bws):
            index = 0

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
    ),
    Sender(
        random.uniform(0.3, 1.5) * bws[0],
        None, 0, features.split(","),
        history_len=history_len
    )
]

import matplotlib.pyplot as plt

env = SimulatedNetworkEnv(senders, get_network(), history_len=history_len, features=features)
model = PPO1.load("./pcc_model_23", env)
model2 = PPO1.load("./pcc_model_23", env)
model3 = PPO1.load("./pcc_model_23", env)

#time_data = [float(event["Time"]) for event in data["Events"][1:]]
#rew_data = [float(event["Reward"]) for event in data["Events"][1:]]
#optimal_data = [float(event["Optimal"]) for event in data["Optimal"][1:]]
#send_data = [float(event["Send Rate"]) for event in data["Events"][1:]]

plt.figure()

fig, axes = plt.subplots(3, figsize=(10, 12))
sender1_axis = axes[0]
sender2_axis = axes[1]
sender3_axis = axes[2]


def plot_axis(axis, events):
    times = [event["Time"] for event in events[-501:]]
    throu = [event["Throughput"] for event in events[-500:]]
    optim = [8*event["Optimal"] for event in events[-501:]]
    axis.plot(times[:500], throu, "r-", label="Throughput")
    axis.plot(times, optim, "b--", label="Optimal")

obs = env.reset()
for i in range(1600 * 410):
    action, _states = model.predict(obs[0])
    action2, _states = model2.predict(obs[1])
    action3, _states = model3.predict(obs[2])
    obs, rewards, dones, info = env.step(action + action2 + action3)

    if i > 0 and i % 400 == 0:
        obs = env.reset()
        
        plot_axis(sender1_axis, info[0]["Events"])
        plot_axis(sender2_axis, info[1]["Events"])
        plot_axis(sender3_axis, info[2]["Events"])
        plt.draw()
        plt.pause(0.01)

    if i > 7500:
        env.senders = [env.senders[0]]
        obs = env.reset(True)
        break
        
    if i > 0 and i % 2500 == 0:
        obs = env.reset(True)

    env.render()

for i in range(1600 * 410):
    action, _states = model.predict(obs[0])
    obs, rewards, dones, info = env.step(action)

    if i > 0 and i % 400 == 0:
        obs = env.reset()
        
        plot_axis(sender1_axis, info[0]["Events"])
        plt.draw()
        plt.pause(0.01)

    if i > 0 and i % 2500 == 0:
        obs = env.reset(True)

    env.render()

##
#   Save the model to the location specified below.
##
default_export_dir = "/tmp/pcc_saved_models/model_A/"
export_dir = arg_or_default("--model-dir", default=default_export_dir)

model.save(export_dir)
