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
import os
import sys
import inspect
from tqdm import tqdm


currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
pparentdir = os.path.dirname(parentdir)
sys.path.insert(0,pparentdir)
from src.gym.worker.aurora_worker import AuroraWorker
from src.gym.worker.ogd_worker import OGDWorker
from src.gym.worker.two_point_ogd_worker import TwoPointOGDWorker
from src.gym.worker.combining_worker import CombiningWorker

from src.gym.simulate_network.link import Link
from src.gym.simulate_network.sender import Sender
from src.gym.simulate_network.simulated_network_env import SimulatedNetworkEnv
from src.gym.visualizer.single_sender_visualizer import BatchVisualizer

history_len = 10
features = "sent latency inflation," + "latency ratio," + "send ratio"
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

senders = [
    Sender(
        random.uniform(0.3, 1.5) * bws[0],
        None, 0, features.split(","),
        history_len=history_len
    ),
]


env = SimulatedNetworkEnv(senders, get_network(), history_len=history_len, features=features)

model = CombiningWorker(
    (40, 300),
    env,
    [
        AuroraWorker("./rand_model_12", env, (40, 300)),
        TwoPointOGDWorker(env, (40, 300), C=11 * 300, L=20)
    ]
)


TIMES = 5000

vis = BatchVisualizer(env, [model])
vis.steps(TIMES, 300, 100)