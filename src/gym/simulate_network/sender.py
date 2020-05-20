import json
import numpy as np
from src.common import sender_obs, config
from src.gym.simulate_network.constants import *


class Sender:
    def __init__(self, rate, path, dest, features, cwnd=25, history_len=10):
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
        self.rtt_samples = []
        self.sample_time = []
        self.net = None
        self.path = path
        self.dest = dest
        self.history_len = history_len
        self.features = features
        self.history = sender_obs.SenderHistory(self.history_len,
                                                self.features, self.id)

        self.event_record = {"Events": []}
        self.reward_sum = 0
        self.reward_ewma = 0
        
        self.cwnd = cwnd

    _next_id = 1

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

    def on_packet_acked(self, rtt):
        self.acked += 1
        self.rtt_samples.append(rtt)
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
            packet_size=BYTES_PER_PACKET
        )

    def reset_obs(self):
        self.sent = 0
        self.acked = 0
        self.lost = 0
        self.rtt_samples = []
        self.obs_start_time = self.net.get_cur_time()

    def print_debug(self):
        print("Sender:")
        print("Obs: %s" % str(self.get_obs()))
        print("Rate: %f" % self.rate)
        print("Sent: %d" % self.sent)
        print("Acked: %d" % self.acked)
        print("Lost: %d" % self.lost)
        print("Min Latency: %s" % str(self.min_latency))

    def reset(self):
        # print("Resetting sender!")
        self.rate = self.starting_rate
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
        with open(filename + "_sender_%d.json" % self.id, 'w') as f:
            json.dump(self.event_record, f, indent=4)

    def add_event(self, steps_taken, run_dur):
        """
        Adds step event and returns the new run_dur
        :param steps_taken: The number of steps taken until
        this point.
        :param run_dur: The original run_dur
        :return: New run dur
        """

        reward = self.get_reward()

        sender_mi = self.get_run_data()

        event = {}
        event["Name"] = "Step"
        event["Time"] = steps_taken
        event["Reward"] = reward
        # event["Optimal"] = BYTES_PER_PACKET * np.min([link.bw for link in self.path])
        # event["Target Rate"] = sender_mi.target_rate
        event["Send Rate"] = sender_mi.get("send rate")
        event["Throughput"] = sender_mi.get("recv rate")
        event["Latency"] = sender_mi.get("avg latency")
        event["Loss Rate"] = sender_mi.get("loss ratio")
        event["Latency Inflation"] = sender_mi.get("sent latency inflation")
        event["Latency Ratio"] = sender_mi.get("latency ratio")
        event["Send Ratio"] = sender_mi.get("send ratio")
        # event["Cwnd"] = sender_mi.cwnd
        # event["Cwnd Used"] = sender_mi.cwnd_used
        self.event_record["Events"].append(event)
        if event["Latency"] > 0.0:
            run_dur = 0.5 * sender_mi.get("avg latency")
        # print("Sender obs: %s" % sender_obs)

        self.reward_sum += reward
        return run_dur

    def get_reward(self):
        sender_mi = self.get_run_data()
        throughput = sender_mi.get("recv rate")
        latency = sender_mi.get("avg latency")
        loss = sender_mi.get("loss ratio")
        bw_cutoff = self.path[0].bw * 0.8
        lat_cutoff = 2.0 * self.path[0].delay * 1.5
        loss_cutoff = 2.0 * self.path[0].loss_rate * 1.5
        # print("thpt %f, bw %f" % (throughput, bw_cutoff))
        # reward = 0 if (loss > 0.1 or throughput < bw_cutoff or latency > lat_cutoff or loss > loss_cutoff) else 1 #

        # Super high throughput
        # reward = REWARD_SCALE * (20.0 * throughput / RATE_OBS_SCALE - 1e3 * latency / LAT_OBS_SCALE - 2e3 * loss)

        # Very high thpt
        reward = (10.0 * throughput / (8 * BYTES_PER_PACKET) - 1e3 * latency - 2e3 * loss)

        # High thpt
        # reward = REWARD_SCALE * (5.0 * throughput / RATE_OBS_SCALE - 1e3 * latency / LAT_OBS_SCALE - 2e3 * loss)

        # Low latency
        # reward = REWARD_SCALE * (2.0 * throughput / RATE_OBS_SCALE - 1e3 * latency / LAT_OBS_SCALE - 2e3 * loss)
        # if reward > 857:
        # print("Reward = %f, thpt = %f, lat = %f, loss = %f" % (reward, throughput, latency, loss))

        # reward = (throughput / RATE_OBS_SCALE) * np.exp(-1 * (LATENCY_PENALTY * latency / LAT_OBS_SCALE + LOSS_PENALTY * loss))
        return reward * REWARD_SCALE

