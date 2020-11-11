from tqdm import tqdm
from src.gym.simulate_network.network import Network
from src.gym.worker.worker import Worker
from src.gym.worker.worker_runner import WorkerRunner


class Visualizer:
    def __init__(self, env, workers: [Worker]):
        self.env = env
        self.workers = workers
        self.worker_runner = self.create_worker_runner()

    def create_worker_runner(self) -> WorkerRunner:
        obs = self.env.reset()
        reward = [0] * len(self.workers)

        return WorkerRunner(self.workers, obs, reward)

    def render_step(self, obs, reward, dones, info):
        pass

    def render_data(self, obs, reward, dones, info):
        pass

    def finish_step(self):
        pass

    def steps(self, num_steps: int, reset_interval: int, data_interval: int):
        for i in tqdm(range(num_steps)):
            action = self.worker_runner.start_step()

            for j in range(len(self.workers)):
                self.env.senders[j].set_rate(action[j])

            obs, reward, dones, info = self.env.step([0] * len(self.workers))

            self.render_step(obs, reward, dones, info)

            if i > 0 and i % data_interval == 0:
                self.render_data(obs, reward, dones, info)

                [sender.reset_event_record() for sender in self.env.net.senders]

            if i > 0 and i % reset_interval == 0:
                obs = self.env.reset(True)
                reward = [0] * len(self.workers)

            self.worker_runner.finish_step(obs, reward)

        return self.finish_step()
