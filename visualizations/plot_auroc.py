import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("resources/wandb_export_best_aug5.csv")
gdf = df.groupby(["num_channels", "channel"])

colors = ["red", "blue", "orange", "green"]

for keys, idf in gdf:
    print(keys)
    c = 0
    for idx, row in idf.iterrows():
        obj_dir = f"resources/res_numch{keys[0]}_ch{keys[1]}_th{row.threshold}.npy"
        obj = np.load(obj_dir, allow_pickle=True).item()
        tpr = obj["tpr"]
        fpr = obj.get("fpr")
        roc_auc = obj.get("roc_auc")
        print(idx)
        plt.plot(
            fpr,
            tpr,
            color=colors[c],
            lw=2,
            label=f"AUROC (Threshold {row.threshold}) = {np.round(roc_auc, 4)}",
        )

        c = c + 1

    # plt.plot([0, 1], [0, 1],'r--')
    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize="large")
    plt.ylabel("True Positive Rate", fontsize="large")
    plt.legend(loc="lower right", fontsize="small")
    plt.tick_params(labelsize="large")
    plt.title("ROC Curves", fontsize="large")
    plt.savefig(f"resources/AUROC_numch{keys[0]}_ch{keys[1]}", dpi=300)
    plt.clf()
