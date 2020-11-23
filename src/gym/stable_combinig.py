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

from src.gym.simulate_network.single_sender_combining_network import SingleSenderCombiningNetwork
from src.gym.simulate_network.reward.aurora_reward import AuroraReward
from src.common.simple_arg_parse import arg_or_default
from src.gym.no_regret_policy.simple_mlp_policy import SimpleMlpPolicy

training_sess = None

#env = gym.make('CartPole-v0')

gamma = arg_or_default("--gamma", default=0.99)
output = arg_or_default("--output", default=".")

env = SingleSenderCombiningNetwork({
    'lr': 5000
}, reward=AuroraReward(), output=output)

print("gamma = %f" % gamma)
model = PPO1(SimpleMlpPolicy, env, verbose=1, schedule='constant', timesteps_per_actorbatch=8192, optim_batchsize=2048, gamma=gamma)

for i in range(0, 20):
    model.learn(total_timesteps=(1600 * 410))
    model.save(output + "/pcc_model_%d.zip" % i)

    [sender.reset_event_record() for sender in env.net.senders]

##
#   Save the model to the location specified below.
##
model.save(output + "/saved_model.zip")
