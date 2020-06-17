import random
import numpy as np


class Link:
    @staticmethod
    def get_bw_limits():
        return (100, 500)

    @staticmethod
    def get_lat_limits():
        return (0.08, 0.1)

    @staticmethod
    def get_queue_limits():
        return (5, 5)

    @staticmethod
    def get_loss_limits():
        return (0.0, 0.0)

    @staticmethod
    def generate_random_link():
        min_bw, max_bw = Link.get_bw_limits()
        min_lat, max_lat = Link.get_lat_limits()
        min_queue, max_queue = Link.get_queue_limits()
        min_loss, max_loss = Link.get_loss_limits()

        bw = random.uniform(min_bw, max_bw)
        lat = random.uniform(min_lat, max_lat)
        queue = 1 + int(np.exp(random.uniform(min_queue, max_queue)))
        loss = random.uniform(min_loss, max_loss)

        return Link(bw, lat, queue, loss)

    def __init__(self, bandwidth, delay, queue_size, loss_rate):
        self.bw = float(bandwidth)
        self.delay = delay
        self.loss_rate = loss_rate
        self.queue_delay = 0.0
        self.queue_delay_update_time = 0.0
        self.queue_size = queue_size
        self.max_queue_delay = queue_size / self.bw

    def update_parameters(self, bandwidth, delay, queue_size, loss_rate):
        self.bw = float(bandwidth)
        self.delay = delay
        self.queue_size = queue_size
        self.queue_delay = 0.0
        self.queue_delay_update_time = 0.0
        self.loss_rate = loss_rate
        self.max_queue_delay = queue_size / self.bw

    def get_cur_queue_delay(self, event_time):
        # print("GET_QUEUE queue_delay", self.queue_delay)
        # print("GET_QUEUE update_time", self.queue_delay_update_time)
        return max(0.0, self.queue_delay - (event_time - self.queue_delay_update_time))

    def get_cur_latency(self, event_time):
        return self.delay + self.get_cur_queue_delay(event_time)

    def packet_enters_link(self, event_time):
        if (random.random() < self.loss_rate):
            return False
        self.queue_delay = self.get_cur_queue_delay(event_time)
        self.queue_delay_update_time = event_time
        extra_delay = 1.0 / self.bw
        #print("Extra delay: %f, Current delay: %f, Max delay: %f" % (extra_delay, self.queue_delay, self.max_queue_delay))
        if extra_delay + self.queue_delay > self.max_queue_delay:
            #print("\tDrop!")
            return False
        self.queue_delay += extra_delay
        #print("\tNew delay = %f" % self.queue_delay)
        return True

    def print_debug(self):
        print("Link:")
        print("Bandwidth: %f" % self.bw)
        print("Delay: %f" % self.delay)
        print("Queue Delay: %f" % self.queue_delay)
        print("Max Queue Delay: %f" % self.max_queue_delay)
        print("One Packet Queue Delay: %f" % (1.0 / self.bw))

    def reset(self):
        self.queue_delay = 0.0
        self.queue_delay_update_time = 0.0
