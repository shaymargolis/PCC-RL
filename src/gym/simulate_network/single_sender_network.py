import copy
import random
from typing import Generator, List

import numpy as np

from src.common.simple_arg_parse import arg_or_default
from src.gym.simulate_network.link import Link
from src.gym.simulate_network.network import Network
from src.gym.simulate_network.reward.reward import Reward
from src.gym.simulate_network.reward.vivace_loss_reward import VivaceLossReward
from src.gym.simulate_network.sender import Sender
from src.gym.simulate_network.simulated_network_env import SimulatedNetworkEnv


class SingleSenderNetwork(SimulatedNetworkEnv):
    def __init__(self,
                 network_generator: Generator[List[Link], None, None],
                 output=".",
                 history_len=arg_or_default("--history-len", default=10),
                 features=arg_or_default("--input-features",
                                         default="sent latency inflation,"
                                                 + "latency ratio,"
                                                 + "send ratio"),
                 reward: Reward = VivaceLossReward()):
        self.last_bw = None

        self.history_len = history_len
        self.features = features

        senders = [
            Sender(
                50,
                None, 0, features.split(","),
                history_len=history_len,
                reward=reward
            )
        ]

        super().__init__(senders, network_generator, output=output, history_len=history_len, features=features)

        # self.senders[0].set_rate(random.uniform(0.3, 1.5) * self.last_bw)

    def step(self, action):
        obs_n, reward_n, done_n, info_n = super().step(action)
        return obs_n[0], reward_n[0], done_n[0], info_n[0]

    def reset(self, next_network: bool = True):
        return super().reset(next_network)[0]
