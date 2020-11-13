import os
import numpy as np
import json
import pandas as pd
import tqdm

from src.gym.visualizer.multiple_sender_stats_visualizer import MultipleSenderStatsVisualizer

INPUT_DIR = "/cs/labs/schapiram/shaymar/parameter_tests/env_multiple_with_data"

data = []


class FakeEnv:
    def reset(self, bool=False):
        return None


def analyze_file(file):
    vis = MultipleSenderStatsVisualizer(FakeEnv(), [])
    vis._load_data(file)

    ewma1 = np.array(vis.data[0]["ewma"])
    ewma2 = np.array(vis.data[1]["ewma"])
    
    diffEwma = ewma1 - ewma2
    absDiffEwma = np.abs(diffEwma)

    rate1 = np.array(vis.data[0]["send"])
    rate2 = np.array(vis.data[1]["send"])
    
    diffRate = rate1 - rate2
    absDiffRate = np.abs(diffRate)

    sig1 = np.array(vis.data[0]["significance"])[:, 1]
    sig2 = np.array(vis.data[1]["significance"])[:, 1]

    avgSig1 = np.mean(sig1)
    avgSig2 = np.mean(sig2)
    
    return [
        np.sum(diffEwma),
        np.sum(absDiffEwma),
        np.sum(diffRate),
        np.sum(absDiffRate),
        avgSig1,
        avgSig2
    ]


def analyze_dir(dir_path):
    params = dir_path.split("-")

    # $combLr-$combLowerLr-$combMinProba--$twopLr-$twopLowerLr-$twopDelta
    dir_params = [
        params[0], # combLr
        params[1], # combLowerLr
        params[2], # combMinProba
        params[4], # twopLr
        params[5], # twopLowerLr
        params[6], # twopDelta
    ]
    
    for file_name in os.listdir(INPUT_DIR + "/" + dir_name):
        if file_name.endswith(".json"):
            res = []
            try:
                res = analyze_file(INPUT_DIR + "/" + dir_path + "/" + file_name)
            except:
                print("\t[x] Error while analyzing %s" % (INPUT_DIR + "/" + dir_path + "/" + file_name))
                continue

            arr = dir_params[:]
            arr += res

            data.append(arr)


for dir_name in tqdm.tqdm(os.listdir(INPUT_DIR)):
    analyze_dir(dir_name)

result = pd.DataFrame(data, columns=["combLr", "combLowerLr", "combMinProba", "twopLr", "twopLowerLr", "twopDelta",
                                     "diffEwma", "absDiffEwma", "diffRate", "absDiffRate", "sig1", "sig2"])
result.to_csv("/cs/labs/schapiram/shaymar/out2.csv")

print(result)

