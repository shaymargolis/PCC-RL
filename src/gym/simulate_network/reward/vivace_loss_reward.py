from src.common.sender_obs import SenderMonitorInterval
from src.gym.simulate_network.constants import BYTES_PER_PACKET
from src.gym.simulate_network.reward.reward import Reward


class VivaceLossReward(Reward):
    def get_reward(self, sender_mi: SenderMonitorInterval, rate: float):
        grad_latency = sender_mi.get("grad latency")
        loss = sender_mi.get("loss ratio")
        # throughput = sender_mi.get("recv rate")

        # VIVACE TRHOUGHPUT
        # x = 10 * throughput / 100 / (8 * BYTES_PER_PACKET)
        # x = sent / (8 * BYTES_PER_PACKET)
        x = rate

        latency = grad_latency

        # reward = (x - x * (900 * latency + 11 * loss))
        reward = (x - x * (11 * loss))

        if not isinstance(reward, float):
            print("NOOOOO")

        return reward, latency
