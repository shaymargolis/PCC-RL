import copy
import random
import numpy as np

from gym.envs.registration import register

from src.common.simple_arg_parse import arg_or_default
from src.gym.simulate_network.link import Link
from src.gym.simulate_network.network import Network
from src.gym.simulate_network.sender import Sender
from src.gym.simulate_network.simulated_network_env import SimulatedNetworkEnv


class SingleSenderNetwork(SimulatedNetworkEnv):
    def __init__(self,
                 history_len=arg_or_default("--history-len", default=10),
                 features=arg_or_default("--input-features",
                                         default="sent latency inflation,"
                                                 + "latency ratio,"
                                                 + "send ratio")):
        """
        Generate two links, with sender that has a random
        uniformly disturbed Sending rate between 0.3*bw and 1.5*bw.
        """

        #  Create two random identical links
        link1 = Link.generate_random_link(self)
        link2 = copy.copy(link1)

        links = [link1, link2]

        bw = link1.bw

        #  Create the sender
        senders = [
            Sender(
                random.uniform(0.3, 1.5) * bw,
                links, 0, self.features,
                history_len=history_len
            )
        ]

        #  Init the SimulatedNetwork using the parameters
        network = Network(links, senders)

        super().__init__(network, senders, history_len=history_len, features=features)

    def step(self, action):
        obs_n, reward_n, done_n, info_n = super().step([action])

        return obs_n[0], reward_n[0], done_n[0], info_n[0]

register(id='PccNs-v1', entry_point='simulate_network:SingleSenderNetwork')