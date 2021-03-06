import json
import numpy as np
from src.common import sender_obs, config
from src.gym.simulate_network.constants import *
from scipy.optimize import curve_fit

from src.gym.simulate_network.reward.reward import Reward
from src.gym.simulate_network.reward.vivace_loss_reward import VivaceLossReward

BIG_NUMBER = 500

class Sender:
    def __init__(self, rate, path, dest, features, cwnd=25, history_len=10, reward: Reward = VivaceLossReward()):
        """
        :param rate: Rate in MBps
        :param path:
        :param dest:
        :param features:
        :param cwnd:
        :param history_len:
        """
        self.id = Sender._get_next_id()
        self.starting_rate = rate
        self.rate = rate
        self.sent = 0
        self.acked = 0
        self.lost = 0
        self.bytes_in_flight = 0
        self.min_latency = None
        self.rtt_samples = list()
        self.event_samples = list()
        self.sample_time = []
        self.net = None
        self.path = path
        self.dest = dest
        self.history_len = history_len
        self.features = features
        self.history = sender_obs.SenderHistory(self.history_len,
                                                self.features, self.id)

        self.event_record\
            = {"Events": []}
        self.reward_sum = 0
        self.reward_ewma = 0
        self.last_latency = [0]
        self.reward = reward
        self.real_time = 0

        self.cwnd = cwnd

    _next_id = 1

    def reset_event_record(self):
        self.event_record \
            = {"Events": []}

    @staticmethod
    def _get_next_id():
        result = Sender._next_id
        Sender._next_id += 1
        return result

    def apply_rate_delta(self, delta):    
        delta *= config.DELTA_SCALE
        # print("Applying delta %f" % delta)
        if delta >= 0.0:
            self.set_rate(self.rate * (1.0 + delta))
        else:
            self.set_rate(self.rate / (1.0 - delta))

    def apply_cwnd_delta(self, delta):
        delta *= config.DELTA_SCALE
        # print("Applying delta %f" % delta)
        if delta >= 0.0:
            self.set_cwnd(self.cwnd * (1.0 + delta))
        else:
            self.set_cwnd(self.cwnd / (1.0 - delta))

    def can_send_packet(self):
        if USE_CWND:
            return int(self.bytes_in_flight) / BYTES_PER_PACKET < self.cwnd
        else:
            return True

    def register_network(self, net):
        self.net = net

    def on_packet_sent(self):
        self.sent += 1
        self.bytes_in_flight += BYTES_PER_PACKET

    def on_packet_acked(self, rtt, event_time):
        if self.acked < BIG_NUMBER:
            self.rtt_samples.append(rtt)
            self.event_samples.append(event_time)

        self.acked += 1
        # self.rtt_samples.append(rtt)
        # self.event_samples.append(event_time)
        if (self.min_latency is None) or (rtt < self.min_latency):
            self.min_latency = rtt
        self.bytes_in_flight -= BYTES_PER_PACKET

    def on_packet_lost(self):
        self.lost += 1
        self.bytes_in_flight -= BYTES_PER_PACKET

    def set_rate(self, new_rate):
        self.rate = new_rate
        # print("Attempt to set new rate to %f (min %f, max %f)" % (new_rate, MIN_RATE, MAX_RATE))
        if self.rate > MAX_RATE:
            self.rate = MAX_RATE
        if self.rate < MIN_RATE:
            self.rate = MIN_RATE

    def set_cwnd(self, new_cwnd):
        self.cwnd = int(new_cwnd)
        # print("Attempt to set new rate to %f (min %f, max %f)" % (new_rate, MIN_RATE, MAX_RATE))
        if self.cwnd > MAX_CWND:
            self.cwnd = MAX_CWND
        if self.cwnd < MIN_CWND:
            self.cwnd = MIN_CWND

    def record_run(self):
        smi = self.get_run_data()
        self.history.step(smi)

    def get_obs(self):
        return self.history.as_array()

    def get_run_data(self):
        obs_end_time = self.net.get_cur_time()
		
        # obs_dur = obs_end_time - self.obs_start_time
        # print("Got %d acks in %f seconds" % (self.acked, obs_dur))
        # print("Sent %d packets in %f seconds" % (self.sent, obs_dur))
        # print("self.rate = %f" % self.rate)
        # self.print_debug()
        # print("Start: %f" % self.obs_start_time)
        # print("End: %f" % obs_end_time)

        return sender_obs.SenderMonitorInterval(
            self.id,
            bytes_sent=self.sent * BYTES_PER_PACKET,
            bytes_acked=self.acked * BYTES_PER_PACKET,
            bytes_lost=self.lost * BYTES_PER_PACKET,
            send_start=self.obs_start_time,
            send_end=obs_end_time,
            recv_start=self.obs_start_time,
            recv_end=obs_end_time,
            rtt_samples=self.rtt_samples,
            event_samples=self.event_samples,
            packet_size=BYTES_PER_PACKET
        )

    def reset_obs(self):
        self.sent = 0
        self.acked = 0
        self.lost = 0
        self.rtt_samples = list()
        self.event_samples = list()
        self.obs_start_time = self.net.get_cur_time()

    def print_debug(self):
        print("[Sender %d]: " % self.id)
        print("Obs: %s" % str(self.get_obs()))
        print("Rate: %f" % self.rate)
        print("Sent: %d" % self.sent)
        print("Acked: %d" % self.acked)
        print("Lost: %d" % self.lost)
        print("Min Latency: %s" % str(self.min_latency))

    def reset(self):
        # print("Resetting sender!")
        # self.rate = self.starting_rate
        self.bytes_in_flight = 0
        self.min_latency = None
        self.reset_obs()
        self.history = sender_obs.SenderHistory(self.history_len,
                                                self.features, self.id)

        self.reward_ewma *= 0.99
        self.reward_ewma += 0.01 * self.reward_sum
        print("[Sender %d] Reward: %0.2f, Ewma Reward: %0.2f" % (self.id, self.reward_sum, self.reward_ewma))
        self.reward_sum = 0.0

    def dump_events_to_file(self, filename):
        shallow_record = {"Events": self.event_record["Events"][-500:]}
    
        with open(filename + "_sender_%d.json" % self.id, 'w') as f:
            json.dump(shallow_record, f, indent=4)

    def add_event(self, steps_taken, run_dur):
        """
        Adds step event and returns the new run_dur
        :param steps_taken: The number of steps taken until
        this point.
        :param run_dur: The original run_dur
        :return: New run dur
        """

        self.real_time += run_dur

        reward, latency_grad = self.get_reward()

        sender_mi = self.get_run_data()

        event = {}
        event["Name"] = "Step"
        event["EWMA"] = self.reward_ewma
        event["Time"] = steps_taken
        event["RealTime"] = self.real_time
        event["Reward"] = reward
        event["Optimal"] = BYTES_PER_PACKET * np.min([link.bw for link in self.path])
        # event["Target Rate"] = sender_mi.target_rate
        event["Send Rate"] = sender_mi.get("send rate")
        event["Throughput"] = sender_mi.get("recv rate")
        event["Latency"] = sender_mi.get("avg latency")
        event["Loss Rate"] = sender_mi.get("loss ratio")
        event["Latency Gradient"] = latency_grad
        event["Latency Inflation"] = sender_mi.get("sent latency inflation")
        event["Latency Ratio"] = sender_mi.get("latency ratio")
        event["Send Ratio"] = sender_mi.get("send ratio")
        # event["Cwnd"] = sender_mi.cwnd
        # event["Cwnd Used"] = sender_mi.cwnd_used
        self.event_record["Events"].append(event)
        if event["Latency"] > 0.0:
            run_dur = 1.5 * sender_mi.get("avg latency")
        # print("Sender obs: %s" % sender_obs)

        self.reward_sum += reward
        return run_dur

    def get_reward(self):
        sender_mi = self.get_run_data()
        return self.reward.get_reward(sender_mi, self.rate)

    def __le__(self, other):
        return self.id <= other.id

    def __lt__(self, other):
        return self.id < other.id
