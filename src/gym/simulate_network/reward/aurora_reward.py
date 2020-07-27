from src.common.sender_obs import SenderMonitorInterval
from src.gym.simulate_network.constants import REWARD_SCALE, BYTES_PER_PACKET
from src.gym.simulate_network.reward.reward import Reward


class AuroraReward(Reward):
    def get_reward(self, sender_mi: SenderMonitorInterval, rate: float):
        throughput = sender_mi.get("recv rate")
        latency = sender_mi.get("avg latency")
        loss = sender_mi.get("loss ratio")

        #  AURORA THROUGHPUT
        reward = (10.0 * throughput / (8 * BYTES_PER_PACKET) - 1e3 * latency - 2e3 * loss)
        return reward * REWARD_SCALE, latency
