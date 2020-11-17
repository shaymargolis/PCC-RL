import os
import numpy as np
import json
import pandas as pd
import tqdm

from src.gym.visualizer.single_sender_visualizer import SingleSenderVisualizer

INPUT_DIR = "/cs/labs/schapiram/shaymar/parameter_tests"

data = []


class FakeEnv:
    def reset(self, bool=False):
        return None


def analyze_file(file):
    vis = SingleSenderVisualizer(FakeEnv(), [], 0)
    vis._load_data(file)

    send = np.array(vis.data[0]["send"])
    optim = np.array(vis.data[0]["optim"])
    sig = np.array(vis.data[0]["significance"])[:, 0]

    diffRate = optim - send
    absDiffRate = np.abs(diffRate)

    avgSig = np.mean(sig)
    sigFinal = np.mean(sig[-500:])

    return [
        np.sum(diffRate), # DiffRate
        np.sum(absDiffRate), # absDiffRate
        avgSig, # avgSig
        sigFinal, # sigFinal
        file # filename
    ]


def analyze_dir_with_params(dir_path, dir_params):
    for file_name in os.listdir(INPUT_DIR + "/" + dir_path):
        if file_name.endswith(".json"):
            res = []
            try:
                res = analyze_file(INPUT_DIR + "/" + dir_path + "/" + file_name)
            except Exception as e:
                # print(e)
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
    500, # combLr
    0, # combLowerLr
    0.1, # combMinProba
    10000, # twopLr
    0, # twopLowerLr
    0.01, # twopDelta
]


analyze_dir_with_params("comb_500_0.1_1000_0.01", dir_params)

result = pd.DataFrame(data, columns=["combLr", "combLowerLr", "combMinProba", "twopLr", "twopLowerLr", "twopDelta",
                                     "diffRate", "absDiffRate", "avgSig", "sigFinal", "file_name"])
result.to_csv("/cs/labs/schapiram/shaymar/out_comb0.csv")

print(result)

