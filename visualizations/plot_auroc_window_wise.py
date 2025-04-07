import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("resources/wandb_export_window_wise_performance.csv")
df = df.sort_values(by=["threshold"], ascending=True)

thresholds = df["threshold"].unique()
print(thresholds)

aurocs_60 = {
    35: 0.8596,
    40: 0.8335,
    45: 0.8375,
    50: 0.8166,
}

f1s_60 = {
    35: 0.6296,
    40: 0.6582,
    45: 0.7041,
    50: 0.6686,
}

for w in [35, 40, 45, 50]:
    tdf = df[df["threshold"] == w]
    tdf = tdf.sort_values(by=["ef_match_window"], ascending=True)

    # thresholds = tdf["threshold"].values
    windows = tdf["ef_match_window"].values
    aurocs = tdf["test_f1_(best_f1)"].values

    windows = np.append(windows, 60)
    # aurocs = np.append(aurocs, aurocs_60[w])
    aurocs = np.append(aurocs, f1s_60[w])

    plt.plot(windows, aurocs, label=f"Threshold = {w}", marker="|")
# plt.plot(thresholds, [], label=60)
plt.legend()
plt.title(
    "ResNet Model Performance (F1) for Different EF-ECG Matching Windows",
    fontsize="large",
)
plt.xlabel("EF-ECG Matching Window Sizes")
plt.ylabel("F1-Score")
plt.savefig("resources/window_wise_f1.png", dpi=300, bbox_inches="tight")
