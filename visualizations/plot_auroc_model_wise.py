import matplotlib.pyplot as plt
import numpy as np

# import pandas as pd

num_channels = 12
channel = 0
threshold = 35

colors = ["red", "blue", "orange", "green", "darkorchid"]

models = ["transformer", "svm", "random_forest", "mlp"]

model_names = ["Transformer", "SVM", "RandomForest", "MLP"]

obj_dir = f"resources/res_numch{num_channels}_ch-1_th{threshold}.npy"
obj = np.load(obj_dir, allow_pickle=True).item()
tpr = obj["tpr"]
fpr = obj.get("fpr")
roc_auc = obj.get("roc_auc")
plt.plot(
    fpr, tpr, color=colors[0], lw=2, label=f"ResNet (AUROC={np.round(roc_auc, 3)})"
)

for c, model in enumerate(models):
    if model == "mlp" or model == "transformer":
        channel = -1
    else:
        channel = 0
    obj_dir = (
        f"resources/res_model{model}_numch{num_channels}_ch{channel}_th{threshold}.npy"
    )
    obj = np.load(obj_dir, allow_pickle=True).item()
    tpr = obj["tpr"]
    fpr = obj.get("fpr")
    roc_auc = obj.get("roc_auc")
    # print(obj['acc'], obj['f1'], roc_auc)

    plt.plot(
        fpr,
        tpr,
        color=colors[c + 1],
        lw=2,
        label=f"{model_names[c]} (AUROC={np.round(roc_auc, 3)})",
    )

    # plt.plot([0, 1], [0, 1],'r--')
plt.plot([0, 1], [0, 1], "k--", lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate", fontsize="large")
plt.ylabel("True Positive Rate", fontsize="large")
plt.legend(loc="lower right", fontsize="medium")
plt.tick_params(labelsize="large")
plt.title(
    f"Receiver Operating Characteristic (ROC) Curves\nfor Various AI Models (at threshold = {threshold})",
    fontsize="large",
)
plt.savefig(f"resources/AUROC_threshold{threshold}", dpi=300)
# plt.clf()
