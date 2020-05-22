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
        self.created_network = None
        network = self.generate_random_network()
        self.created_network = network
        
        super().__init__(network.senders, [network], output=output, history_len=history_len, features=features)

    def step(self, action):
        obs_n, reward_n, done_n, info_n = super().step(action)
				
        return obs_n[0], reward_n[0], done_n[0], info_n[0]
	
    def generate_random_network(self):
        #  Create two random identical links
        link1 = Link.generate_random_link()
        link2 = Link(link1.bw, link1.delay, link1.queue_delay, link1.loss_rate)

        links = [link1, link2]

        bw = link1.bw

        #  Create the sender
        senders = [
            
        ]
        
        if self.created_network is not None:
            senders = self.created_network.senders
            senders[0].path = links
            senders[0].rate = random.uniform(0.3, 1.5) * bw
            senders[0].reset()
        else:
            senders = [Sender(
                random.uniform(0.3, 1.5) * bw,
                links, 0, self.features,
                history_len=self.history_len
            )]

        #  Init the SimulatedNetwork using the parameters
        return Network(senders, links)
    
    def use_next_network(self):
        self.net = self.generate_random_network()

        lat = np.max([link.delay for link in self.net.links])
        self.run_dur = 3 * lat

    
    def reset(self):
        return super().reset(True)[0]

register(id='PccNs-v1', entry_point='src.gym.simulate_network.single_sender_network:SingleSenderNetwork')