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
import matplotlib.pyplot as plt


currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
pparentdir = os.path.dirname(parentdir)
sys.path.insert(0,pparentdir)

from src.gym.parameter_readme import create_readmefile
from src.gym.parameter_extractor import extract_parameters

from src.common.simple_arg_parse import arg_or_default

from src.gym.network_creator import get_env

from src.gym.worker.aurora_worker import AuroraWorker
from src.gym.worker.ogd_worker import OGDWorker
from src.gym.worker.two_point_ogd_worker import TwoPointOGDWorker
from src.gym.worker.combining_worker import CombiningWorker

from src.gym.visualizer.multiple_sender_visualizer import MultipleSenderVisualizer
from src.gym.visualizer.multiple_sender_stats_visualizer import MultipleSenderStatsVisualizer

NUMBER_OF_EPOCHES = 1
TIMES = 15000
bws = [600]

params = extract_parameters()

comb_kwargs = params["comb_kwargs"]
two_point_kwargs = params["two_point_kwargs"]
OUTPUT = params["output"]
offset = params["offset"]

#  Fix race cond bug
if params["concurrent"] == 1:
    import matplotlib
    matplotlib.use('Agg')


create_readmefile(params)

for i in range(NUMBER_OF_EPOCHES):
    env = get_env(bws, 3, "loss")

    print("ENV", env)

    model = CombiningWorker(
        (100, 600),
        env,
        [
            AuroraWorker("./rand_model_12", env, (100, 600)),
            TwoPointOGDWorker(env, (100, 600), C=11 * 600, L=20, **two_point_kwargs)
        ],
        **comb_kwargs
    )

    model2 = CombiningWorker(
        (100, 600),
        env,
        [
            AuroraWorker("./rand_model_12", env, (100, 600)),
            TwoPointOGDWorker(env, (100, 600), C=11 * 600, L=20, **two_point_kwargs)
        ],
        **comb_kwargs
    )


    model3 = CombiningWorker(
        (100, 600),
        env,
        [
            AuroraWorker("./rand_model_12", env, (100, 600)),
            TwoPointOGDWorker(env, (100, 600), C=11 * 300, L=20, **two_point_kwargs)
        ],
        **comb_kwargs
    )

    start1 = random.uniform(100, 600)
    start2 = random.uniform(100, 600)
    start3 = random.uniform(100, 600)

    model.set_action(start1)
    model2.set_action(start2)
    model3.set_action(start3)

    vis = MultipleSenderStatsVisualizer(env, [model, model2, model3])
    vis.steps(TIMES, TIMES, 100)

    fig = vis.parse_data()

    fig.suptitle('COMB=(%d, %r, %.2f), TWOP=(%d, %r, %.2f),\n START=(%.0f, %.0f)' % (comb_lr, comb_lower_lr, comb_min_proba, twop_lr, twop_lower_lr, twop_delta, start1, start2), fontsize=16)
    fig.savefig(OUTPUT + "/%d.png" % i)
    plt.show()
    vis._save_data(OUTPUT + "/%d.json" % i)
