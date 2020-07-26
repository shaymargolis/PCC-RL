from src.gym.simulate_network.simulated_network_env import SimulatedNetworkEnv


class SingleSenderRepeatedNetwork(SimulatedNetworkEnv):
    def step(self, actions: list):
        obs_n, reward_n, done_n, info_n = super().step(actions)

        return obs_n[0], reward_n[0], done_n[0], info_n[0]

    def reset(self, use_next_network=True):
        return super().reset(use_next_network)[0]
