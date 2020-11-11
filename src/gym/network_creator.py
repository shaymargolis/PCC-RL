import random

from src.gym.simulate_network.link import Link
from src.gym.simulate_network.sender import Sender
from src.gym.simulate_network.simulated_network_env import SimulatedNetworkEnv

history_len = 10
features = "sent latency inflation," + "latency ratio," + "send ratio"


class NetworkGenerator:
    def __init__(self, bws):
        self.index = 0
        self.bws = bws

    def get_network(self):
        while True:
            link1 = Link.generate_link(self.bws[self.index], 0.2, 6, 0)
            links = [link1]

            yield links

            self.index += 1

            if self.index == len(self.bws):
                self.index = 0


def get_env(bws, sender_count):
    senders = []

    for i in range(sender_count):
        senders.append(
            Sender(
                random.uniform(0.3, 1.5) * bws[0],
                None, 0, features.split(","),
                history_len=history_len
            )
        )

    generator = NetworkGenerator(bws)

    return SimulatedNetworkEnv(senders, generator.get_network(), history_len=history_len, features=features)
