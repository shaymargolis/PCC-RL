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
import src.gym.simulate_network.simulated_network_env
import tensorflow as tf

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.policies import FeedForwardPolicy
from stable_baselines import PPO1
import os
import sys
import inspect
from src.common.simple_arg_parse import arg_or_default
from src.gym.no_regret_policy.simple_mlp_policy import SimpleMlpPolicy

training_sess = None

env = gym.make('PccNs-v1')
#env = gym.make('CartPole-v0')

gamma = arg_or_default("--gamma", default=0.99)
print("gamma = %f" % gamma)
model = PPO1(SimpleMlpPolicy, env, verbose=1, schedule='constant', timesteps_per_actorbatch=8192, optim_batchsize=2048, gamma=gamma)

for i in range(0, 6):
    model.learn(total_timesteps=(1600 * 410))
    model.save("./pcc_model_%d.zip" % i)

##
#   Save the model to the location specified below.
##
default_export_dir = "/tmp/pcc_saved_models/model_A/"
export_dir = arg_or_default("--model-dir", default=default_export_dir)

model.save(export_dir)
