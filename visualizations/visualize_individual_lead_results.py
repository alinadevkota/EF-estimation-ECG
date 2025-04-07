import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.figure(figsize=(12, 8))

df = pd.read_csv("resources/wandb_export_best_aug5.csv")

df_12ch = df[df["num_channels"] == 12]
df_12ch = df_12ch.sort_values(by=["threshold"], ascending=True)


df = df[df["num_channels"] == 1]
df = df.sort_values(by=["threshold"], ascending=True)

leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

markers = [".", "v", "1", "s", "p", "*", "P", "X", "H", "<", "8", "|"]
colors = [
    "black",
    "blue",
    "orange",
    "yellow",
    "green",
    "red",
    "purple",
    "brown",
    "pink",
    "gray",
    "cyan",
    "olive",
]

for channel in range(12):
    ch_df = df[df["channel"] == channel]
    thresholds = ch_df["threshold"]
    accuracies = ch_df["test_accuracy_(best_f1)"]
    f1_scores = ch_df["test_f1_(best_f1)"] * 100

    plt.plot(
        thresholds,
        accuracies,
        marker=markers[channel],
        color=colors[channel],
        label=f"Accuracy {leads[channel]}",
    )

thresholds = np.array([35, 40, 45, 50])

accuracies = df_12ch["test_accuracy_(best_f1)"].values
f1_scores = df_12ch["test_f1_(best_f1)"].values * 100


plt.plot(
    thresholds, accuracies, marker="o", color="darkolivegreen", label="Accuracy All"
)

plt.xlabel("Thresholds")
plt.ylabel("Scores")
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.title("Model Performance for Individual Leads")
plt.savefig("resources/vis_per_channel_aug5.png", bbox_inches="tight", dpi=300)
