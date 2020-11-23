import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.stats import kde
import numpy as np

result = pd.read_csv("/cs/labs/schapiram/shaymar/out-fixed-comb_3000_0.1_5000_0.02.csv")
# result = pd.read_csv("/cs/labs/schapiram/shaymar/out-comb_shuffled_500_0.1_10000_0.01.csv")

# result.columns = ["idx", "combLr", "combLowerLr", "combMinProba", "twopLr", "twopLowerLr", "twopDelta",
#                                      "diffRate", "absDiffRate", "avgSig", "sigFinal", "file_name"]

print(result["avgSig"])

result["avgSig"].plot.kde()
plt.show()

for i in range(1000):
    pass

# exit(1)

result.plot.scatter(x='avgSig', y='diffRate')
plt.show()

#  NOT USING LOWER_LR
print("diffRate")
absDiffRate = result.sort_values('diffRate', ascending=False)

i = 0

for index, row in absDiffRate.iterrows():
    print('%.2E (%.2f)' % (row["diffRate"], row["avgSig"]))
    if 0 <= i:
        f = row['file_name'][:-5] + '.png'
        image = mpimg.imread(f)
        plt.title('%.2E (%.2f)' % (row["diffRate"], row["avgSig"]))
        plt.imshow(image)
        plt.show()
    i += 1
    if i == 2500:
        break

for i in range(100):
    pass

exit(1)
print(absDiffRate.to_string())


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