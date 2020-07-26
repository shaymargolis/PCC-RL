from typing import Generator, List

from src.gym.simulate_network.link import Link
from src.gym.simulate_network.simulated_network_env import SimulatedNetworkEnv
from src.gym.simulate_network.single_sender_network import SingleSenderNetwork

from gym.envs.registration import register


class SingleSenderRandomNetwork(SingleSenderNetwork):
    @staticmethod
    def generate_next_network():
        while True:
            #  Create two random identical links
            link1 = Link.generate_random_link()
            links = [link1]

            #  Init the SimulatedNetwork using the parameters
            yield links

    def __init__(self):
        super().__init__(self.generate_next_network())

register(id='PccNs-v1', entry_point='src.gym.simulate_network.single_sender_network:SingleSenderNetwork')
