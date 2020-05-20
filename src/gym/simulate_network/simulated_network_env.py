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
                 networks: [Network],
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
		
        self.networks: [Network] = networks
        self.senders: [Sender] = senders
        self.net: Network = None
        self.next_network_id = 0
        self.create_new_links_and_senders()
		
        self.run_dur = None

        self.run_period = 0.1
        self.steps_taken = 0
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
        # print(self.run_dur)
        for i in range(len(actions)):  # len(actions)):
            # print("Updating rate for sender %d" % i)
            action = actions[i]
            self.senders[i].apply_rate_delta(action)
            if USE_CWND:
                self.senders[i].apply_cwnd_delta(action)

        # print("Running for %fs" % self.run_dur)
        self.net.run_for_dur(self.run_dur)
		
        for sender in self.senders:
            sender.record_run()

		
        should_stop = False

        obs_n = self._get_all_sender_obs()
        done_n = [(self.steps_taken >= self.max_steps or should_stop)] * len(self.senders)
        info_n = [{}] * len(self.senders)

        reward_n = [sender.get_reward() for sender in self.senders]

        self.steps_taken += 1

        self.run_dur = np.min([sender.add_event(self.steps_taken, self.run_dur) for sender in self.senders])

        return obs_n, reward_n, done_n, info_n

    def print_debug(self):
        print("---Sender Debug---")
        for sender in self.senders:
            sender.print_debug()

            print("*Link Debug*")
            for link in sender.path:
                link.print_debug()

    def create_new_links_and_senders(self):
        self.net = self.networks[self.next_network_id]
        self.net.reset()

        self.next_network_id += 1
        if self.next_network_id >= len(self.networks):
            self.next_network_id = 0

        lat = np.max([link.delay for link in self.net.links])
        self.run_dur = 3 * lat

    def reset(self):
        self.steps_taken = 0
        self.create_new_links_and_senders()

        self.episodes_run += 1
        if self.episodes_run > 0 and self.episodes_run % 100 == 0:
            self.dump_events_to_file("pcc_env_log_run_%d" % self.episodes_run)

        self.net.run_for_dur(self.run_dur)
        self.net.run_for_dur(self.run_dur)
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


# register(id='PccNs-v1', entry_point='simulate_network:simulated_network_env:SimulatedNetworkEnv')