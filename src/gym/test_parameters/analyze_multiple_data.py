import pandas as pd
import matplotlib.pyplot as plt

result = pd.read_csv("/cs/labs/schapiram/shaymar/out2.csv")

# result.columns = ["idx", "combLr", "combLowerLr", "combMinProba", "twopLr", "twopLowerLr", "twopDelta",
#                                      "diffEwma", "absDiffEwma", "diffRate", "absDiffRate", "sig1", "sig2"]

result["totalPred"] = result["sig1"] + result["sig2"]
result["diffRate"] = result["diffRate"].abs()
result["totalDiff"] = result["absDiffRate"] + result["diffRate"]

print(result["absDiffRate"].plot.kde())

#  NOT USING LOWER_LR
print("absDiffRate")
print(result.sort_values('absDiffRate').to_string())

print("totalPred")
# print(result.sort_values('totalPred').to_string())

result = result.groupby(['combLr', 'combLowerLr', 'combMinProba', 'twopLr', 'twopLowerLr', 'twopDelta'], as_index=False).mean()

result.plot.scatter(
    x='combLr',
    y='combLowerLr',
    c='totalDiff',
    colormap='viridis'
)
plt.title("1")
plt.show()

result.plot.scatter(
    x='twopLr',
    y='twopLowerLr',
    c='totalDiff',
    colormap='viridis'
)
plt.title("2")
plt.show()

# result.plot(x='combLr', y='absDiffRate', style='o')
# plt.show()
#
# result.plot(x='combLr', y='diffRate', style='o')
# plt.show()
# result.plot(x="combLowerLr", y="diffRate", style="o")
# plt.show()
# result.plot(x="twopLr", y="diffRate", style="o")
# plt.show()