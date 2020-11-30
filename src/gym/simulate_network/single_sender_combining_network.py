from typing import Generator, List
import random
import numpy as np
from src.gym.simulate_network.link import Link
from src.gym.simulate_network.simulated_network_env import SimulatedNetworkEnv
from src.gym.simulate_network.single_sender_network import SingleSenderNetwork

from gym.envs.registration import register

from src.gym.worker.two_point_ogd_worker import TwoPointOGDWorker
from src.gym.worker.worker_runner import DS


class SingleSenderCombiningNetwork(SingleSenderNetwork):
    @staticmethod
    def generate_next_network():
        while True:
            #  Create two random identical links
            link1 = Link.generate_random_link()
            links = [link1]

            #  Init the SimulatedNetwork using the parameters
            yield links

    def get_next_ogd_proba(self):
        return random.uniform(0, 0.3)

    def __init__(self, two_point_kwargs, reward, output='.'):
        super().__init__(self.generate_next_network(), reward=reward, output=output)

        self.ogd = TwoPointOGDWorker(self, (80, 400), C=11 * 400, L=20, sender_id=0, **two_point_kwargs)
        self.ogd.set_action(random.uniform(40, 400))
        self.ogd_proba = self.get_next_ogd_proba()
        self.last_reward = 0

    def reset(self, next_network: bool = True):
        if next_network:
            self.ogd.set_action(random.uniform(40, 400))
            self.ogd_proba = self.get_next_ogd_proba()

        return super().reset(next_network)

    def ogd_steps(self):
        ds = DS(0, self.last_reward)
        gen = self.ogd.step(ds)

        while True:
            try:
                action = next(gen)
            except StopIteration as ex:
                break

            self.senders[0].set_rate(action)
            obs, reward, done, info = super().step([0])
            self.last_reward = reward
            ds.data = (0, reward)

            next(gen)

        return obs, reward, done, info

    def step(self, action):
        ind = np.random.choice(2, 1, p=[self.ogd_proba, 1-self.ogd_proba])[0]

        if ind == 0:
            #  Run the OGD
            return self.ogd_steps()

        obs, reward, done, info = super().step(action)
        self.last_reward = reward

        self.ogd.set_action(self.senders[0].rate)

        return obs, reward, done, info


register(id='PccNs-v2', entry_point='src.gym.simulate_network.single_sender_combining_network:SingleSenderCombiningNetwork')
