import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.figure(figsize=(12, 8))

df = pd.read_csv("resources/two_lead_results.csv")

df = df.sort_values(by=["threshold"], ascending=True)

leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

markers = [".", "v", "1", "s", "p", "*", "P", "X", "H", "<", "8", "|"]
colors = [
    "black",
    "blue",
    "orange",
    "green",
    "red",
    "purple",
    "brown",
    "pink",
    "gray",
    "cyan",
    "olive",
    "yellow",
]


def get_leads(channels):
    channels = channels.split(",")
    channels = ",".join([leads[int(x)] for x in channels])
    return channels


all_channels = df["channel"].unique()
for idx, channel in enumerate(all_channels):
    ch_df = df[df["channel"] == channel]
    thresholds = ch_df["threshold"]
    accuracies = ch_df["test_accuracy_(best_f1)"]
    f1_scores = ch_df["test_f1_(best_f1)"] * 100

    print(channel)

    plt.plot(
        thresholds,
        accuracies,
        marker=markers[idx],
        color=colors[idx],
        label=f"Accuracy {get_leads(channel)}",
    )
    # for x, y, text in zip(thresholds, accuracies, accuracies):
    #     plt.text(x, y + 0.5, str(text))

    plt.plot(
        thresholds,
        f1_scores,
        "--",
        marker=markers[idx],
        color=colors[idx],
        label=f"F1-Score {get_leads(channel)}",
    )
    # for x, y, text in zip(thresholds, f1_scores, f1_scores):
    #     plt.text(x, y, str(text))

thresholds = np.array([35, 40, 45, 50])

accuracies = np.array([85.588, 83.817, 81.276, 77.34])
f1_scores = np.array([71.02, 71.88, 73.62, 73.7])

two_lead_accuracuies = np.array([78.36, 82.12, 79.45, 76.84])
two_lead_f1_scores = np.array([64.77, 69.22, 71.91, 71.96])

plt.plot(
    thresholds, accuracies, marker="o", color="darkolivegreen", label="Accuracy All"
)
plt.plot(
    thresholds,
    f1_scores,
    "--",
    marker="o",
    color="darkolivegreen",
    label="F1-Score All",
)

plt.xlabel("Thresholds")
plt.ylabel("Scores")
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.title("Model Performance for Individual Leads")
plt.savefig("resources/vis_two_channel.png", bbox_inches="tight", dpi=300)
