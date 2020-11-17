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
import random
import sys
import inspect

import matplotlib.pyplot as plt

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
pparentdir = os.path.dirname(parentdir)
sys.path.insert(0,pparentdir)

from src.gym.network_creator import get_env
from src.gym.parameter_extractor import extract_parameters
from src.gym.parameter_readme import create_readmefile

from src.gym.worker.aurora_worker import AuroraWorker
from src.gym.worker.ogd_worker import OGDWorker
from src.gym.worker.two_point_ogd_worker import TwoPointOGDWorker
from src.gym.worker.combining_worker import CombiningWorker

from src.gym.visualizer.single_sender_visualizer import SingleSenderVisualizer

NUMBER_OF_EPOCHES = 10
TIMES = 5000
bws = [100, 240]

params = extract_parameters()

comb_kwargs = params["comb_kwargs"]
two_point_kwargs = params["two_point_kwargs"]
OUTPUT = params["output"]
offset = params["offset"]

create_readmefile(comb_kwargs, two_point_kwargs, OUTPUT)


# Race cond bug
import matplotlib
matplotlib.use('Agg')


for i in range(NUMBER_OF_EPOCHES):
    env = get_env(bws, 1, params["reward_type"])

    model = CombiningWorker(
        (40, 300),
        env,
        [
            AuroraWorker("./rand_model_12", env, (40, 300)),
            TwoPointOGDWorker(env, (40, 300), C=11 * 300, L=20, **two_point_kwargs)
        ],
        **comb_kwargs
    )

    start1 = random.uniform(40, 300)
    model.set_action(start1)


    vis = SingleSenderVisualizer(env, [model], 0)
    vis.steps(TIMES, 300, 100)
    fig = vis.parse_data()

    fig.suptitle('COMB=(%d, %r, %.2f), TWOP=(%d, %r, %.2f), \n REW=%s, START=%.0f' % (
        comb_kwargs["lr"],
        comb_kwargs["lower_lr"],
        comb_kwargs["min_proba_thresh"],
        two_point_kwargs["lr"],
        two_point_kwargs["lower_lr"],
        two_point_kwargs["delta"],
        params["reward_type"],
        start1
    ), fontsize=16)
    fig.savefig(OUTPUT + "/%d.png" % (i + offset))
    plt.show()
    vis._save_data(OUTPUT + "/%d.json" % (i + offset))
