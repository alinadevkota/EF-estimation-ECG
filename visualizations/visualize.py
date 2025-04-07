import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# plt.figure(figsize=(12, 8))

bar_width = 0.8

thresholds = np.array([35, 40, 45, 50])

df = pd.read_csv("resources/wandb_export_best_aug5.csv")

df_12ch = df[df["num_channels"] == 12]
df_12ch = df_12ch.sort_values(by=["threshold"], ascending=True)

accuracies = df_12ch["test_accuracy_(best_f1)"].values
f1_scores = df_12ch["test_f1_(best_f1)"].values * 100
aurocs = df_12ch["test_auroc_(best_f1)"].values * 100


train_sizes = [(2780, 22154), (4003, 20931), (5338, 19596), (6869, 18065)]
val_sizes = [(732, 5502), (1020, 5214), (1390, 4844), (1772, 4462)]
test_sizes = [(877, 6915), (1233, 6559), (1715, 6077), (2175, 5617)]


train_zeros = np.array([x[0] for x in train_sizes])
train_ones = np.array([x[1] for x in train_sizes])

val_zeros = np.array([x[0] for x in val_sizes])
val_ones = np.array([x[1] for x in val_sizes])

test_zeros = np.array([x[0] for x in test_sizes])
test_ones = np.array([x[1] for x in test_sizes])

# print(len(train_ones), len(val_ones))

# plot bars in stack manner
plt.bar(thresholds - bar_width, train_zeros, color="#ffaa00", label="Train low EF")
plt.bar(
    thresholds - bar_width,
    train_ones,
    bottom=train_zeros,
    color="#e85d04",
    label="Train high EF",
)
plt.bar(thresholds, val_zeros, color="#9ef01a", label="Val low EF")
plt.bar(thresholds, val_ones, bottom=val_zeros, color="#38b000", label="Val high EF")
plt.bar(thresholds + bar_width, test_zeros, color="#07C8F9", label="Test low EF")
plt.bar(
    thresholds + bar_width,
    test_ones,
    bottom=test_zeros,
    color="#0D41E1",
    label="Test high EF",
)


plt.xlabel("Thresholds")
plt.ylabel("Sizes")
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.title("Label Distribution for Different Thresholds", fontsize="large")
plt.savefig("resources/vis_small_data.png", dpi=300, bbox_inches="tight")
