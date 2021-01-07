print("Beginning module import!")
import socket

# Rates should be in mbps
MAX_RATE = 1000.0
MIN_RATE = 0.25
STARTING_RATE = 2.0

DELTA_SCALE = 0.025# * 1e-12
#DELTA_SCALE = 0.1

RATE_OBS_SCALE = 0.001
LAT_OBS_SCALE = 0.1


class PccShimDriver():
    
    flow_lookup = {}
    
    def __init__(self, flow_id):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect(("127.0.0.1", 9787))
        
        self.replay_rate = False
        self.rate = STARTING_RATE
        self.last_rate_delta = 0
        PccShimDriver.flow_lookup[flow_id] = self
    
    def get_rate(self):
        if self.replay_rate:
            return self.apply_rate_delta(self.last_rate_delta)
        self.replay_rate = True
        
        action = self.sock.recv(1024).decode()
        
        if action == "RESET":
            self.reset()
            self.last_rate_delta = 0
        else:
            self.last_rate_delta = float(action)
        
        return self.apply_rate_limits(self.apply_rate_delta(self.last_rate_delta))
    
    def apply_rate_delta(self, action):
        delta = action * DELTA_SCALE
        #print("Applying delta %f" % delta)
        if delta >= 0.0:
            return (self.rate * (1.0 + delta))
        else:
            return (self.rate / (1.0 - delta))
    
    def apply_rate_limits(self, new_rate):
        #print("Attempt to set new rate to %f (min %f, max %f)" % (new_rate, MIN_RATE, MAX_RATE))
        if new_rate > MAX_RATE:
            new_rate = MAX_RATE
        if new_rate < MIN_RATE:
            new_rate = MIN_RATE
        
        return new_rate

    def reset(self):
        self.rate = self.apply_rate_limits(STARTING_RATE)
        pass # Nothing to reset in the shim driver.

    def give_sample(self, flow_id, bytes_sent, bytes_acked, bytes_lost,
                    send_start_time, send_end_time, recv_start_time,
                    recv_end_time, rtt_samples, packet_size, utility):
        if not self.replay_rate:
            print("Detected repeat sample! Ignoring.")
            return
        
        self.rate = self.apply_rate_limits(self.apply_rate_delta(self.last_rate_delta))
        
        self.sock.send(("%d;%d;%d;%d;%f;%f;%f;%f;%s;%d;%f\n" % (
            flow_id,
            bytes_sent,
            bytes_acked,
            bytes_lost,
            send_start_time,
            send_end_time,
            recv_start_time,
            recv_end_time,
            rtt_samples,
            packet_size,
            utility)).encode())
        self.replay_rate = False

    def get_by_flow_id(flow_id):
        return PccShimDriver.flow_lookup[flow_id]

def give_sample(flow_id, bytes_sent, bytes_acked, bytes_lost,
                send_start_time, send_end_time, recv_start_time,
                recv_end_time, rtt_samples, packet_size, utility):
    driver = PccShimDriver.get_by_flow_id(flow_id)
    driver.give_sample(flow_id,
        bytes_sent,
        bytes_acked,
        bytes_lost,
        send_start_time,
        send_end_time,
        recv_start_time,
        recv_end_time,
        rtt_samples,
        packet_size,
        utility)
    
def reset(flow_id):
    driver = PccShimDriver.get_by_flow_id(flow_id)
    driver.reset()

def get_rate(flow_id):
    driver = PccShimDriver.get_by_flow_id(flow_id)
    return driver.get_rate() * 1e6

def set_rate(flow_id, new_rate):
    driver = PccShimDriver.get_by_flow_id(flow_id)
    driver.set_rate(new_rate / 1e6)

def init(flow_id):
    driver = PccShimDriver(flow_id)
