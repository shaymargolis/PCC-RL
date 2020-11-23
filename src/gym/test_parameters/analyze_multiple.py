import os
import numpy as np
import json
import pandas as pd
import tqdm

from src.gym.visualizer.multiple_sender_stats_visualizer import MultipleSenderStatsVisualizer

INPUT_DIR = "/cs/labs/schapiram/shaymar/parameter_tests/pls_sign"

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

    ewma1Final = ewma1[-1]
    ewma2Final = ewma2[-1]

    rate1 = np.array(vis.data[0]["send"])
    rate2 = np.array(vis.data[1]["send"])
    
    diffRate = rate1 - rate2
    absDiffRate = np.abs(diffRate)

    sig1 = np.array(vis.data[0]["significance"])[:, 1]
    sig2 = np.array(vis.data[1]["significance"])[:, 1]

    avgSig1 = np.mean(sig1)
    avgSig2 = np.mean(sig2)

    sig1Final = np.mean(sig1[-500:])
    sig2Final = np.mean(sig2[-500:])
    
    return [
        np.sum(diffEwma),
        np.sum(absDiffEwma),
        ewma1Final,
        ewma2Final,
        np.sum(diffRate),
        np.sum(absDiffRate),
        avgSig1,
        avgSig2,
        sig1Final,
        sig2Final,
        file
    ]


def analyze_dir_with_params(dir_path, dir_params):
    for file_name in os.listdir(INPUT_DIR + "/" + dir_path):
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

    analyze_dir_with_params(dir_path, dir_params)

# for dir_name in tqdm.tqdm(os.listdir(INPUT_DIR)):
#     analyze_dir(dir_name)

dir_params = [
    3000, # combLr
    0, # combLowerLr
    0.1, # combMinProba
    5000, # twopLr
    0, # twopLowerLr
    0.02, # twopDelta
]

FILE_NAME = "multiple_3000_0.1_5000_0.02"

analyze_dir_with_params(FILE_NAME, dir_params)

result = pd.DataFrame(data, columns=["combLr", "combLowerLr", "combMinProba", "twopLr", "twopLowerLr", "twopDelta",
                                     "diffEwma", "absDiffEwma", "ewma1Final", "ewma2Final", "diffRate", "absDiffRate", "sig1", "sig2", "sig1F", "sig2F", "file_name"])
result.to_csv("/cs/labs/schapiram/shaymar/out-fixed-%s.csv" % FILE_NAME)

print(result)

