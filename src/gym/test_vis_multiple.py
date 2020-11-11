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
import os
import sys
import inspect
import random

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
pparentdir = os.path.dirname(parentdir)
sys.path.insert(0,pparentdir)

from src.gym.network_creator import get_env

from src.gym.worker.aurora_worker import AuroraWorker
from src.gym.worker.ogd_worker import OGDWorker
from src.gym.worker.two_point_ogd_worker import TwoPointOGDWorker
from src.gym.worker.combining_worker import CombiningWorker

from src.gym.visualizer.multiple_sender_visualizer import MultipleSenderVisualizer

bws = [240]

env = get_env(bws, 2)

print("ENV", env)

model = CombiningWorker(
    (40, 300),
    env,
    [
        AuroraWorker("./rand_model_12", env, (40, 300)),
        TwoPointOGDWorker(env, (40, 300), C=11 * 300, L=20)
    ]
)

model2 = CombiningWorker(
    (40, 300),
    env,
    [
        AuroraWorker("./rand_model_12", env, (40, 300)),
        TwoPointOGDWorker(env, (40, 300), C=11 * 300, L=20)
    ]
)

model.workers[1].set_action(random.uniform(40, 300))
model2.workers[1].set_action(random.uniform(40, 300))

TIMES = 5000

vis = MultipleSenderVisualizer(env, [model, model2])
fig = vis.steps(TIMES, 5000, 100)

import matplotlib.pyplot as plt
plt.show()