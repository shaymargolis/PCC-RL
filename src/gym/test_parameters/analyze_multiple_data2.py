import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.stats import kde
import numpy as np

result = pd.read_csv("/cs/labs/schapiram/shaymar/out-fixed-reg_aurora.csv")

# result.columns = ["idx", "combLr", "combLowerLr", "combMinProba", "twopLr", "twopLowerLr", "twopDelta",
#                                      "diffEwma", "absDiffEwma", "diffRate", "absDiffRate", "sig1", "sig2", "file_name"]

result["totalPredF"] = result["sig1F"] + result["sig2F"]
result["totalPred"] = result["sig1"] + result["sig2"]
result["diffRate"] = result["diffRate"].abs()
result["totalDiff"] = result["absDiffRate"] + result["diffRate"]

result["totalEwma"] = result["ewma1Final"] + result["ewma2Final"]

result = result.sort_values('absDiffRate')
result.reset_index(drop=True, inplace=True)

def showImportanceInSorted(sorted, column, chunk=20):
    x = []
    data = []

    mean = []

    for index, row in sorted.iterrows():
        mean.append(row[column])

        if index > 0 and index % chunk == 0:
            x.append(index)
            data.append(np.mean(mean))
            mean = []
            continue

    plt.figure()
    plt.title(column)
    plt.plot(x, data, "b.", label=column)

    yabs_max = 1.1*np.max(np.abs(np.array(data)))
    plt.ylim([0, yabs_max])

    plt.legend()
    plt.show()

showImportanceInSorted(result, "combLr")
showImportanceInSorted(result, "twopLr")
showImportanceInSorted(result, "combMinProba")
showImportanceInSorted(result, "twopDelta")

result = result.groupby(['combLr', 'combLowerLr', 'combMinProba', 'twopLr', 'twopLowerLr', 'twopDelta'], as_index=False).mean()


result.plot(x='combLr', y='absDiffRate', style='o')
plt.show()

result.plot(x='combLr', y='diffRate', style='o')
plt.show()
result.plot(x="combLowerLr", y="diffRate", style="o")
plt.show()
result.plot(x="twopLr", y="diffRate", style="o")
plt.show()