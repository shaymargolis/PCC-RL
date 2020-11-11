import numpy as np
import matplotlib.pyplot as plt
from src.gym.visualizer.visualizer import Visualizer
from src.gym.worker.combining_worker import CombiningWorker


class SingleSenderVisualizer(Visualizer):
    def __init__(self, env, workers, sender_ind):
        super().__init__(env, workers)

        self.sender_ind = sender_ind

        self.data = []
        for i in range(len(self.workers)):
            self.data.append({
                "times": [],
                "send": [],
                "throu": [],
                "optim": [],
                "latency": [],
                "lat": [],
                "loss": [],
                "reward": [],

                "significance": []
            })

    def render_step(self, obs, reward, dones, info):
        #  Calculate SIG
        i = self.sender_ind

        if isinstance(self.workers[i], CombiningWorker):
            self.data[i]["significance"] += [self.workers[i].get_proba()[:]]

    def render_data(self, obs, reward, dones, info):
        i = self.sender_ind
        info = info[i]
        data = self.data[i]

        data["times"] += [event["Time"] for event in info["Events"]]
        data["send"] += [event["Send Rate"] for event in info["Events"]]
        data["throu"] += [event["Throughput"] for event in info["Events"]]
        data["optim"] += [8 * event["Optimal"] for event in info["Events"]]
        data["latency"] += [event["Latency Gradient"] for event in info["Events"]]
        data["lat"] += [event["Latency"] for event in info["Events"]]
        data["loss"] += [event["Loss Rate"] for event in info["Events"]]
        data["reward"] += [event["Reward"] for event in info["Events"]]

    def finish_step(self):
        i = self.sender_ind
        data = self.data[i]

        fig, ax = plt.subplots(nrows=3, ncols=2)

        ax[0][0].title.set_text("Sending rate")
        ax[0][0].plot(data["times"], data["throu"], "g.", label="Throughput")
        ax[0][0].plot(data["times"], data["send"], "r-", label="Send rate")
        ax[0][0].plot(data["times"], data["optim"], "b--", label="Optimal")
        ax[0][0].legend()
        ax[0][0].grid()

        ax[0][1].title.set_text("Reward")
        ax[0][1].plot(data["times"], data["reward"], "b.", label="Reward")
        ax[0][1].legend()
        ax[0][1].grid()

        if len(data["significance"]) > 0:
            sig = data["significance"][:len(data["times"])]
            ax[1][0].title.set_text("Significance")
            ax[1][0].plot(data["times"], list(map(lambda x: x[0], sig)), "b-", label="Aurora Sig")
            ax[1][0].plot(data["times"], list(map(lambda x: x[1], sig)), "g-", label="OGD Sig")
            ax[1][0].legend()
            ax[1][0].grid()

        ax[1][1].title.set_text("Loss")
        ax[1][1].plot(data["times"], data["loss"], "g.", label="Loss")
        ax[1][1].legend()
        ax[1][1].grid()

        ax[2][0].title.set_text("Latency")
        ax[2][0].plot(data["times"], data["lat"], "b.", label="Latency")
        ax[2][0].plot(data["times"], np.ones(len(data["times"])) * self.env.net.links[0].delay, "r--", label="Link latency")
        ax[2][0].legend()
        ax[2][0].grid()

        return fig
