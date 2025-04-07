import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("resources/new_data/data.csv")

df["target"].plot.hist(bins=30)
plt.title("Histogram of Ejection Fraction", fontsize="large")
plt.xlabel("Ejection Fraction")
plt.savefig("resources/hist.png", dpi=300)
