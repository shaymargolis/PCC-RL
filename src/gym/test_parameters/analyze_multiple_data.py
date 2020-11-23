import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.stats import kde
import numpy as np

result = pd.read_csv("/cs/labs/schapiram/shaymar/out-fixed-multiple_3000_0.1_5000_0.02.csv")

# result.columns = ["idx", "combLr", "combLowerLr", "combMinProba", "twopLr", "twopLowerLr", "twopDelta",
#                                      "diffEwma", "absDiffEwma", "diffRate", "absDiffRate", "sig1", "sig2", "file_name"]

result["totalPredF"] = result["sig1F"] + result["sig2F"]
result["totalPred"] = result["sig1"] + result["sig2"]
result["diffRate"] = result["diffRate"].abs()
result["totalDiff"] = result["absDiffRate"] + result["diffRate"]

result["totalEwma"] = result["ewma1Final"] + result["ewma2Final"]

result.plot.scatter(x='totalPred', y='totalEwma')
plt.show()

result["diffRate"].plot.kde()
plt.show()

for i in range(1000):
    pass

def plot_2d_kde(x, y, xlabel="x", ylabel="y", title="2d kde"):
    nbins = 300
    k = kde.gaussian_kde([x, y])
    xi, yi = np.mgrid[x.min():x.max():nbins * 1j, y.min():y.max():nbins * 1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))

    #  Calculate prob total
    grid = zi.reshape(xi.shape)
    print(grid)
    print(grid.shape)
    print(np.sum(grid[150:,150:]))
    print(np.sum(zi))

    # Make the plot
    plt.pcolormesh(xi, yi, zi.reshape(xi.shape))
    plt.colorbar()
    plt.xlabel(xlabel),
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

plot_2d_kde(result["sig1"], result["sig2"],
            xlabel='Sender 1 Sig',
            ylabel='Sender 2 Sig',
            title='Probability Density of OGD Significance (1 = Most significant)')
plt.show()

#  NOT USING LOWER_LR
print("diffRate")
absDiffRate = result.sort_values('absDiffRate', ascending=True)

i = 0

for index, row in absDiffRate.iterrows():
    if row["absDiffRate"] <= 2*10**10:
        continue

    print('%.2E' % row["absDiffRate"])
    if 0 <= i:
        f = row['file_name'][:-5] + '.png'
        image = mpimg.imread(f)
        plt.title('%.2E' % row["absDiffRate"])
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