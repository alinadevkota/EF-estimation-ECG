import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.figure(figsize=(12, 8))

df = pd.read_csv("resources/wandb_export_best_aug5.csv")

df_12ch = df[df["num_channels"] == 12]
df_12ch = df_12ch.sort_values(by=["threshold"], ascending=True)


df = pd.read_csv("resources/wandb_export_aug5_two_channels.csv")
# df = df[df["num_channels"]==1]
df = df.sort_values(by=["threshold"], ascending=True)

leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def map_leads(channel_str):
    channels = channel_str.split(",")
    channels = [leads[int(x)] for x in channels]
    channels = ",".join(channels)
    return channels


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

# for i, channel in enumerate(range(12)):
for i, channel in enumerate(df["channel"].unique()):
    print(channel)
    ch_df = df[df["channel"] == channel]
    thresholds = ch_df["threshold"]
    aurocs = ch_df["test_auroc_(best_f1)"]
    # f1_scores = ch_df["test_f1_(best_f1)"] * 100

    plt.plot(
        thresholds,
        aurocs,
        marker=markers[i],
        color=colors[i],
        label=f"{map_leads(channel)}",
        # label=f"{leads[channel]}",
    )

thresholds = np.array([35, 40, 45, 50])

aurocs = df_12ch["test_auroc_(best_f1)"].values

plt.plot(
    thresholds, aurocs, "--", marker="o", color="darkolivegreen", label="All Channels"
)

plt.xticks(thresholds)
# plt.xlim((30,55))

plt.xlabel("Thresholds")
plt.ylabel("Scores")
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.title("Model Performance (AUROC) for Two Leads")
plt.savefig("resources/vis_two_channel_aug5_auroc.png", bbox_inches="tight", dpi=300)
