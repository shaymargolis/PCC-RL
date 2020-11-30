import numpy as np
import matplotlib.pyplot as plt
import json
from src.gym.visualizer.visualizer import Visualizer
from src.gym.worker.combining_worker import CombiningWorker


class MultipleSenderStatsVisualizer(Visualizer):
    def __init__(self, env, workers):
        super().__init__(env, workers)

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
                "ewma": [],

                "significance": []
            })

    def _save_data(self, path):
        def round_by_decimals(dec):
            def round_floats(f):
                return np.around(f, decimals=dec)

            return round_floats

        two = round_by_decimals(2)

        for i in range(len(self.workers)):
            sig = self.data[i]["significance"]

            for j in range(len(sig)):
                sig[j] = [
                    two(sig[j][0]),
                    two(sig[j][1])
                ]

            self.data[i] = {
                'times': self.data[i]["times"],
                "send": list(map(two, self.data[i]["send"])),
                "throu": list(map(two, self.data[i]["throu"])),
                "ewma": list(map(two, self.data[i]["ewma"])),
                "reward": list(map(two, self.data[i]["reward"])),
                "significance": self.data[i]["significance"]
            }

        super()._save_data(path)

    def render_step(self, obs, reward, dones, info):
        #  Calculate SIG
        for i in range(len(self.workers)):
            if isinstance(self.workers[i], CombiningWorker):
                self.data[i]["significance"] += [self.workers[i].get_proba()[:]]

    def render_data(self, obs, reward, dones, info_arr):
        for i in range(len(self.workers)):
            info = info_arr[i]
            data = self.data[i]

            data["times"] += [event["RealTime"] for event in info["Events"]]
            data["send"] += [event["Send Rate"] for event in info["Events"]]
            data["throu"] += [event["Throughput"] for event in info["Events"]]
            data["optim"] += [8 * event["Optimal"] for event in info["Events"]]
            data["latency"] += [event["Latency Gradient"] for event in info["Events"]]
            data["lat"] += [event["Latency"] for event in info["Events"]]
            data["loss"] += [event["Loss Rate"] for event in info["Events"]]
            data["reward"] += [event["Reward"] for event in info["Events"]]
            data["ewma"] += [event["EWMA"] for event in info["Events"]]

    def parse_data(self):
        fig, axes = plt.subplots(nrows=len(self.workers), ncols=3, figsize=(10, 12))

        senders_axis = axes[0][0]
        sender_ewma_axis = axes[1][0]
        sender1_sig_axis = axes[0][1]
        sender2_sig_axis = axes[1][1]

        diff_axis = axes[0][2]
        abs_diff_axis = axes[1][2]

        senders_axis.title.set_text("Sending Rate")
        sender_ewma_axis.title.set_text("Reward")
        sender1_sig_axis.title.set_text("Sender 1 Sig")
        sender2_sig_axis.title.set_text("Sender 2 Sig")
        diff_axis.title.set_text("Diff")
        abs_diff_axis.title.set_text("Abs diff Axis")

        self._plot_axis(senders_axis)
        self._plot_ewma(sender_ewma_axis)
        self._plot_sender_sig(sender1_sig_axis, 0)
        self._plot_sender_sig(sender2_sig_axis, 1)

        diff, abs_diff = self._calculate_fairness()

        diff_axis.plot(self.data[0]["times"], diff, "r.")
        abs_diff_axis.plot(self.data[0]["times"], abs_diff, "b.")

        if len(self.workers) == 3:
            sender3_sig_axis = axes[2][1]
            sender3_sig_axis.title.set_text("Sender 3 Sig")
            self._plot_sender_sig(sender3_sig_axis, 2)

        return fig

    def _calculate_fairness(self):
        send1 = np.array(self.data[0]["send"])
        send2 = np.array(self.data[1]["send"])

        diff = send1 - send2
        abs_diff = np.abs(diff)

        return diff, abs_diff

    def _plot_axis(self, axis):
        colors = [('r', 'g'), ('b', 'm'), ('k', 'y')]

        if len(self.workers) == 0:
            return

        for i in range(len(self.workers)):
            data = self.data[i]

            axis.plot(data["times"], data["send"], colors[i][0] + "-", label="[%d] Sent" % (i + 1))

        wc = len(self.workers)

        axis.plot(data["times"], data["optim"], "b--", label="Optimal")
        axis.plot(data["times"], np.array(data["optim"]) / wc, "r--", label="Optimal/%d" % wc)

        axis.legend()

    def _plot_ewma(self, axis):
        colors = ["r", "b", "g", "p"]

        for i in range(len(self.workers)):
            data = self.data[i]

            axis.plot(data["times"], data["ewma"], colors[i] + "-", label="Sender" + str(i))

        axis.legend()

    def _plot_sender_sig(self, axis, i):
        sig = self.data[i]["significance"][:len(self.data[i]["times"])]

        axis.plot(self.data[i]["times"], list(map(lambda x: x[0], sig)), "b-", label="Aurora Sig")
        axis.plot(self.data[i]["times"], list(map(lambda x: x[1], sig)), "g-", label="OGD Sig")

        axis.legend()
