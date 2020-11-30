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

from src.gym.network_creator import get_env, get_ogd_worker, get_agent_reward_calculator

from src.gym.worker.aurora_worker import AuroraWorker
from src.gym.worker.ogd_worker import OGDWorker
from src.gym.worker.two_point_ogd_worker import TwoPointOGDWorker
from src.gym.worker.combining_worker import CombiningWorker

from src.gym.visualizer.multiple_sender_visualizer import MultipleSenderVisualizer
from src.gym.visualizer.multiple_sender_stats_visualizer import MultipleSenderStatsVisualizer

NUMBER_OF_EPOCHES = 10
TIMES = 15000
bws = [400]

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
    env = get_env(bws, 2, params["reward_type"])

    print("ENV", env)

    model = CombiningWorker(
        (80, 400),
        env,
        [
            AuroraWorker("./" + params["aurora_agent"], env, (80, 400)),
            get_ogd_worker(params["ogd_worker"], env, (80, 400), C=11 * 400, L=20, sender_id=0, **two_point_kwargs)
        ],
        reward_calculator=get_agent_reward_calculator(params["agent_reward"]),
        **comb_kwargs
    )

    model2 = CombiningWorker(
        (80, 400),
        env,
        [
            AuroraWorker("./" + params["aurora_agent"], env, (80, 400)),
            get_ogd_worker(params["ogd_worker"], env, (80, 400), C=11 * 400, L=20, sender_id=1, **two_point_kwargs)
        ],
        reward_calculator=get_agent_reward_calculator(params["agent_reward"]),
        **comb_kwargs
    )

    start1 = random.uniform(40, 300)
    start2 = random.uniform(40, 300)

    model.set_action(start1)
    model2.set_action(start2)

    vis = MultipleSenderStatsVisualizer(env, [model, model2])
    vis.steps(TIMES, TIMES, 100)

    fig = vis.parse_data()

    fig.suptitle('COMB=(%d, %r, %.2f), TWOP=(%d, %r, %.2f),\n REW=%s START=(%.0f, %.0f)' % (
        comb_kwargs["lr"],
        comb_kwargs["lower_lr"],
        comb_kwargs["min_proba_thresh"],
        two_point_kwargs["lr"],
        two_point_kwargs["lower_lr"],
        two_point_kwargs["delta"],
        params["reward_type"],
        start1, start2
    ), fontsize=16)

    plt.show()

    fig.savefig(OUTPUT + "/%d.png" % (i + offset))
    vis._save_data(OUTPUT + "/%d.json" % (i + offset))
