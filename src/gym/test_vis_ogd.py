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
bws = [240]

OUTPUT = arg_or_default("--output", default=None)

comb_lr = arg_or_default("--comb_lr", default=200)
comb_lower_lr = arg_or_default("--comb_lower_lr", default=0) == 1
comb_min_proba = arg_or_default("--comb_min_proba", default=0.1)

twop_lr = arg_or_default("--twop_lr", default=5000)
twop_lower_lr = arg_or_default("--twop_lower_lr", default=0) == 1
twop_delta = arg_or_default("--twop_delta", default=0.01)

comb_kwargs = {
    'lr': comb_lr,
    'lower_lr': comb_lower_lr,
    'min_proba_thresh': comb_min_proba
}

two_point_kwargs = {
    'lr': twop_lr,
    'lower_lr': twop_lower_lr,
    'delta': twop_delta
}

#  Fix race cond bug
# import matplotlib
# matplotlib.use('Agg')


create_readmefile(params)

for i in range(NUMBER_OF_EPOCHES):
    env = get_env(bws, 2, "loss")

    print("ENV", env)

    model = CombiningWorker(
        (40, 300),
        env,
        [
            AuroraWorker("./rand_model_12", env, (40, 300)),
            OGDWorker(env, (40, 300), C=11 * 300, L=20)
        ],
        **comb_kwargs
    )

    model2 = CombiningWorker(
        (40, 300),
        env,
        [
            AuroraWorker("./rand_model_12", env, (40, 300)),
            OGDWorker(env, (40, 300), C=11 * 300, L=20)
        ],
        **comb_kwargs
    )

    start1 = random.uniform(40, 300)
    start2 = random.uniform(40, 300)

    model.set_action(start1)
    model2.set_action(start2)

    vis = MultipleSenderStatsVisualizer(env, [model, model2])
    vis.steps(TIMES, TIMES, 100)

    fig = vis.parse_data()

    fig.suptitle('COMB=(%d, %r, %.2f), TWOP=(%d, %r, %.2f),\n START=(%.0f, %.0f)' % (comb_lr, comb_lower_lr, comb_min_proba, twop_lr, twop_lower_lr, twop_delta, start1, start2), fontsize=16)
    fig.savefig(OUTPUT + "/%d.png" % i)
    plt.show()
    vis._save_data(OUTPUT + "/%d.json" % i)
