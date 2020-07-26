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
import re

from stable_baselines import PPO1
import os
import sys
import inspect


currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
pparentdir = os.path.dirname(parentdir)
sys.path.insert(0,pparentdir)

from src.gym.simulate_network.link import Link
from src.gym.simulate_network.single_sender_network import SingleSenderNetwork

from src.common.simple_arg_parse import arg_or_default
from src.gym.no_regret_policy.simple_mlp_policy import SimpleMlpPolicy

training_sess = None

#env = gym.make('CartPole-v0')

gamma = arg_or_default("--gamma", default=0.99)
output = arg_or_default("--output", default=".")

history_len = 10
features = "sent latency inflation," + "latency ratio," + "send ratio"

bws = [100, 240]
index = 0

def get_network():
    global index

    while True:
        link1 = Link.generate_link(bws[index], 0.2, 6, 0)
        links = [link1]

        yield links
        index = 1 - index

env = SingleSenderNetwork(get_network(), output=output)

print("gamma = %f" % gamma)
model = PPO1(SimpleMlpPolicy, env, verbose=1, schedule='constant', timesteps_per_actorbatch=8192, optim_batchsize=2048, gamma=gamma)

start_point = 0
for file in os.listdir(output):
    x = re.findall("^pcc_model_(\d+).zip$", file)

    if len(x) == 0:
        continue

    start_point = max(start_point, int(x[0]))

if start_point != 0:
    model = PPO1.load(output + "/pcc_model_%d.zip" % (start_point), env)

for i in range(start_point+1, 30):
    # model.learn(total_timesteps=(1600 * 410))
    model.learn(total_timesteps=(1600 * 410))
    model.save(output + "/pcc_model_%d.zip" % i)

##
#   Save the model to the location specified below.
##
model.save(output + "/saved_model.zip")
