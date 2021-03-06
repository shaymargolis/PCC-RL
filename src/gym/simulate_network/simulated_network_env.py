from typing import Generator, List

import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.registration import register
import numpy as np
import random
import copy
import json

from src.common import sender_obs
from src.common.simple_arg_parse import arg_or_default

from src.gym.simulate_network.network import Network
from src.gym.simulate_network.link import Link
from src.gym.simulate_network.sender import Sender
from src.gym.simulate_network.constants import *


class SimulatedNetworkEnv(gym.Env):
    def __init__(self,
                 senders: [Sender],
                 network_generator: Generator[List[Link], None, None],
                 output=".",
                 history_len=arg_or_default("--history-len", default=10),
                 features=arg_or_default("--input-features",
                                         default="sent latency inflation,"
                                                 + "latency ratio,"
                                                 + "send ratio")):
        self.viewer = None
        self.rand = None
        self.history_len = history_len
        print("History length: %d" % history_len)
        self.features = features.split(",")
        print("Features: %s" % str(self.features))
		
        self.output = output
        self.network_generator: Generator[List[Link], None, None] = network_generator
        self.senders: [Sender] = senders
        self.net: Network = None
        self.init_default_network()
        self.use_next_network()
		
        self.run_dur = None

        self.run_period = 0.1
        self.steps_taken = 0
        self.total_steps_taken = 0
        self.max_steps = MAX_STEPS
        self.debug_thpt_changes = False
        self.last_thpt = None
        self.last_rate = None

        if USE_CWND:
            self.action_space = spaces.Box(np.array([-1e12, -1e12]), np.array([1e12, 1e12]), dtype=np.float32)
        else:
            self.action_space = spaces.Box(np.array([-1e12]), np.array([1e12]), dtype=np.float32)

        self.observation_space = None
        use_only_scale_free = True
        single_obs_min_vec = sender_obs.get_min_obs_vector(self.features)
        single_obs_max_vec = sender_obs.get_max_obs_vector(self.features)
        self.observation_space = spaces.Box(np.tile(single_obs_min_vec, self.history_len),
                                            np.tile(single_obs_max_vec, self.history_len),
                                            dtype=np.float32)

        self.episodes_run = -1

    def seed(self, seed=None):
        self.rand, seed = seeding.np_random(seed)
        return [seed]

    def _get_all_sender_obs(self):
        def get_single_sender_obs(sender):
            sender_obs = sender.get_obs()
            sender_obs = np.array(sender_obs).reshape(-1,)
            return sender_obs

        sender_obs = [get_single_sender_obs(sen) for sen in self.senders]

        return sender_obs

    def step(self, actions: list):
        for i in range(len(actions)):  # len(actions)):
            # print("Updating rate for sender %d" % i)
            action = actions[i]
            self.senders[i].apply_rate_delta(action)
            if USE_CWND:
                self.senders[i].apply_cwnd_delta(action)

        #for sender in self.senders:
        #    print("Sender %d" % sender.id, "at speed", sender.rate)
                
        # print("Running for %fs" % self.run_dur)
        self.net.run_for_dur(self.run_dur)
		
        for sender in self.senders:
            sender.record_run()
            #print("OBS", self._get_all_sender_obs())
            #print("SENT", sender.sent)
            #print("RECV", sender.acked)

		
        should_stop = False

        obs_n = self._get_all_sender_obs()
        done_n = [(self.steps_taken >= self.max_steps or should_stop)] * len(self.senders)

        reward_n = [sender.get_reward()[0] for sender in self.senders]

        self.steps_taken += 1
        self.total_steps_taken += 1

        self.run_dur = np.min([sender.add_event(self.total_steps_taken, self.run_dur) for sender in self.senders])

        info_n = [sender.event_record for sender in self.senders]

        return obs_n, reward_n, done_n, info_n

    def print_debug(self):
        print("---Sender Debug---")
        for sender in self.senders:
            sender.print_debug()

            print("*Link Debug*")
            for link in sender.path:
                link.print_debug()

    def create_new_links_and_senders(self):
        lat = np.max([link.delay for link in self.net.links])
        self.run_dur = 5 * lat

    def init_default_network(self):
        self.net = Network(self.senders, [Link.generate_random_link()])

        for sender in self.senders:
            sender.path = self.net.links
            sender.register_network(self.net)

        self.net.reset()

    def use_next_network(self):
        new_links: List[Link] = next(self.network_generator)

        for i in range(len(new_links)):
            new_link = new_links[i]
            print("Updating parameters", new_link.bw)
            self.net.links[i].update_parameters(new_link.bw,
                                                new_link.delay,
                                                new_link.queue_size,
                                                new_link.loss_rate)

        lat = np.max([link.delay for link in self.net.links])
        
        self.run_dur = 3 * lat

        print("Using next network", self.net)

    def reset(self, use_next_network=False):
        self.steps_taken = 0
        if use_next_network:
            self.use_next_network()
            #  Run the network for a while
            for i in range(3):
                self.net.run_for_dur(self.run_dur)
        else:
            self.create_new_links_and_senders()

        [sender.reset() for sender in self.senders]

        self.episodes_run += 1
        if self.episodes_run > 0 and self.episodes_run % 100 == 0:
            self.dump_events_to_file(self.output + "/pcc_env_log_run_%d" % self.episodes_run)

        return self._get_all_sender_obs()

    def render(self, mode='human'):
        pass

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def dump_events_to_file(self, filename_start):
        for sender in self.senders:
            sender.dump_events_to_file(filename_start)
