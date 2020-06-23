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
                 output=".",
                 history_len=arg_or_default("--history-len", default=10),
                 features=arg_or_default("--input-features",
                                         default="sent latency inflation,"
                                                 + "latency ratio,"
                                                 + "send ratio")):
        """
        Generate two links, with sender that has a random
        uniformly disturbed Sending rate between 0.3*bw and 1.5*bw.
        """ 

        self.history_len = history_len
        self.features = features.split(",")
        self.senders = self.get_senders()
        network, rate = self.generate_random_network()
        self.senders[0].set_rate(random.uniform(0.3, 1.5) * rate)

        print("First network", network)

        next_net, _ = self.generate_random_network()
        print(network.links[0].bw, next_net.links[0].bw)

        super().__init__(self.senders, [network], output=output, history_len=history_len, features=features)

    def step(self, action):
        obs_n, reward_n, done_n, info_n = super().step(action)
        return obs_n[0], reward_n[0], done_n[0], info_n[0]

    def generate_random_network(self):
        #  Create two random identical links
        link1 = Link.generate_random_link()
        links = [link1]

        bw = link1.bw
        rate = random.uniform(0.3, 1.5) * bw

        #  Init the SimulatedNetwork using the parameters
        return Network([], links), rate

    def get_senders(self):
        return [
            Sender(
                50,
                None, 0, self.features,
                history_len=self.history_len
            )
        ]

    def use_next_network(self):
        super().use_next_network()

        # # self.senders[0].set_rate(float(rate))

    def reset(self, next_network: bool = False):
        if next_network:
            net, rate = self.generate_random_network()
            self.networks = [net]
            self.next_network_id = 0

        return super().reset(next_network)[0]

register(id='PccNs-v1', entry_point='src.gym.simulate_network.single_sender_network:SingleSenderNetwork')
