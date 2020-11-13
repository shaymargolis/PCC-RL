from tqdm import tqdm
from src.gym.simulate_network.network import Network
import numpy as np
import json
from src.gym.worker.worker import Worker
from src.gym.worker.worker_runner import WorkerRunner


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)


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
    
    def _save_data(self, path):
        f = open(path, "w")
        json.dump(self.data, f, cls=NumpyEncoder)
        f.close()

    def _load_data(self, path):
        f = open(path, "r")
        self.data = json.load(f)
        f.close()

    def steps(self, num_steps: int, reset_interval: int, data_interval: int):
        for i in tqdm(range(num_steps)):
            action = self.worker_runner.start_step()

            for j in range(len(self.workers)):
                self.env.senders[j].set_rate(action[j])

            obs, reward, dones, info = self.env.step([0] * len(self.workers))

            self.render_step(obs, reward, dones, info)

            if i > 0 and i % data_interval == 0:
                obs = self.env.reset()
                self.render_data(obs, reward, dones, info)

                [sender.reset_event_record() for sender in self.env.net.senders]

            if i > 0 and i % reset_interval == 0:
                obs = self.env.reset(True)
                reward = [0] * len(self.workers)

            self.worker_runner.finish_step(obs, reward)

    def parse_data(self):
        pass