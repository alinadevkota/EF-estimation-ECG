import argparse
import os
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from captum.attr import ShapleyValues
from tqdm import tqdm

from main import CustomTensorDataset, get_model, get_train_test_data, test


def plot_ig2_tp_fp_fn_tn(svs, y_scores, targets, cmap=plt.cm.Blues):
    """
    Population-level interpretation of IG values aggregated as TP, FP, FN, TN.

    Parameters:
    - svs: 2D array of IG values (num_classes x num_samples)
    - y_scores: 2D array of predicted probabilities/logits (num_samples x num_classes)
    - targets: 1D array of ground truth labels (num_samples)
    - cmap: Colormap for the heatmap
    """
    leads = np.array(
        ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    )
    n = y_scores.shape[0]

    # Initialize lists for TP, FP, FN, and TN
    results = {"TP": [], "FP": [], "FN": [], "TN": []}
    print(svs.shape)

    # Compute the predicted labels
    predicted_labels = np.argmax(y_scores, axis=1)

    # Classify each sample into TP, FP, FN, TN
    for i in tqdm(range(n)):
        if predicted_labels[i] == 1 and targets[i] == 1:
            results["TP"].append(svs[1, i])  # True Positive
        elif predicted_labels[i] == 1 and targets[i] == 0:
            results["FP"].append(svs[1, i])  # False Positive
        elif predicted_labels[i] == 0 and targets[i] == 1:
            results["FN"].append(svs[0, i])  # False Negative
        elif predicted_labels[i] == 0 and targets[i] == 0:
            results["TN"].append(svs[0, i])  # True Negative

    ys = []
    categories = ["TP", "FP", "FN", "TN"]

    for category in categories:
        result = np.array(results[category])
        if result.size == 0:
            ys.append(np.zeros(len(leads)))  # Handle empty cases gracefully
            continue
        y = []
        for i, _ in enumerate(leads):
            y.append(result[:, i].sum())  # Aggregate IG values for each lead
        y = np.array(y) / np.sum(y)  # Normalize
        ys.append(y)
        plt.plot(leads, y, label=category)  # Add category label to the plot

    ys = np.array(ys)

    # Create a heatmap
    fig, axs = plt.subplots()
    im = axs.imshow(ys, cmap=cmap)
    axs.figure.colorbar(im, ax=axs)

    fmt = ".2f"
    xlabels = leads
    ylabels = categories  # TP, FP, FN, TN
    axs.set_xticks(np.arange(len(xlabels)))
    axs.set_yticks(np.arange(len(ylabels)))
    axs.set_xticklabels(xlabels)
    axs.set_yticklabels(ylabels)

    # Annotate heatmap cells
    thresh = ys.max() / 2
    for i in range(ys.shape[0]):
        for j in range(ys.shape[1]):
            axs.text(
                j,
                i,
                format(ys[i, j], fmt),
                ha="center",
                va="center",
                color="white" if ys[i, j] > thresh else "black",
            )

    np.set_printoptions(precision=2)
    fig.tight_layout()
    # plt.legend(loc='upper right')  # Add a legend for the categories
    plt.savefig("shapely_captum_tp_fp_fn_tn.png")
    plt.clf()


def plot_ig2(svs, y_scores, cmap=plt.cm.Blues):
    print(svs.shape, y_scores.shape)
    print(svs[0].shape)
    print(y_scores[0][0])
    print(y_scores[0].shape)
    # population-level interpretation
    leads = np.array(
        ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    )
    n = y_scores.shape[0]
    results = [[], []]
    print(svs.shape)
    for i in tqdm(range(n)):
        label = np.argmax(y_scores[i])
        # label = 0 if y_scores[i][0] < 0.5 else 1
        results[label].append(svs[label, i])
    ys = []
    for label in range(2):
        result = np.array(results[label])
        y = []
        for i, _ in enumerate(leads):
            y.append(result[:, i].sum())
        y = np.array(y) / np.sum(y)
        ys.append(y)
        plt.plot(leads, y)
    ys.append(np.array(ys).mean(axis=0))
    ys = np.array(ys)
    fig, axs = plt.subplots()
    im = axs.imshow(ys, cmap=cmap)
    axs.figure.colorbar(im, ax=axs)
    fmt = ".2f"
    xlabels = leads
    ylabels = [0, 1] + ["AVG"]
    axs.set_xticks(np.arange(len(xlabels)))
    axs.set_yticks(np.arange(len(ylabels)))
    axs.set_xticklabels(xlabels)
    axs.set_yticklabels(ylabels)
    thresh = ys.max() / 2
    for i in range(ys.shape[0]):
        for j in range(ys.shape[1]):
            axs.text(
                j,
                i,
                format(ys[i, j], fmt),
                ha="center",
                va="center",
                color="white" if ys[i, j] > thresh else "black",
            )
    np.set_printoptions(precision=2)
    fig.tight_layout()
    plt.savefig("shaply_captum.png")


def ig_explanation(model, test_dataset, device):
    """
    Use IG to explain the feature importance of a 12-lead ECG dataset.

    Args:
        model: The trained PyTorch model.
        test_dataset: PyTorch dataset for testing.
        device: The device (CPU, GPU) where the model is loaded.
        num_samples: Number of samples to explain (for visualization).

    Returns:
        None
    """
    # Select a subset of test data for IG analysis
    test_x, test_y = zip(*[test_dataset[i] for i in range(len(test_dataset))])
    test_x = torch.stack(test_x).numpy()

    ig = ShapleyValues(model)

    # Get IG values for the selected data
    # ig_values = explainer.ig_values(torch.tensor(test_x, dtype=torch.float32).to(device), check_additivity=False)
    batch_size = 64  # Adjust this based on your GPU memory
    ig_values = {0: [], 1: []}  # Dictionary to store IG values for both classes
    y_scores = []

    # Process data in batches
    for i in range(0, len(test_x), batch_size):
        batch = torch.tensor(test_x[i : i + batch_size], dtype=torch.float32).to(device)
        # label = torch.tensor(test_y[i : i + batch_size], dtype=torch.int64).to(device)

        # Initialize lists for IG values per class
        batch_ig_values_class_0 = []
        batch_ig_values_class_1 = []

        # Get IG values for the current batch for both classes
        for target_class in range(2):  # Loop over both classes (0 and 1)
            # Compute IG for the current class
            batch_ig_values = ig.attribute(
                batch, target=torch.tensor([target_class] * batch.size(0)).to(device)
            )

            # Append the IG values for the current class
            if target_class == 0:
                batch_ig_values_class_0.append(batch_ig_values.detach().cpu().numpy())
            else:
                batch_ig_values_class_1.append(batch_ig_values.detach().cpu().numpy())

        # Append IG values for the current batch
        ig_values[0].append(np.concatenate(batch_ig_values_class_0, axis=0))
        ig_values[1].append(np.concatenate(batch_ig_values_class_1, axis=0))

        # Get model output (logits) for the batch
        out = model(batch)
        y_scores.append(out.detach().cpu().numpy())

    # Concatenate results from all batches for both classes
    ig_values_class_0 = np.concatenate(
        ig_values[0], axis=0
    )  # Shape: [num_samples, num_leads, num_time_steps]
    ig_values_class_1 = np.concatenate(
        ig_values[1], axis=0
    )  # Shape: [num_samples, num_leads, num_time_steps]
    y_scores = np.concatenate(y_scores, axis=0)  # Shape: [num_samples, num_classes]

    # Stack the IG values for both classes along the first axis
    ig_values = np.stack(
        [ig_values_class_0, ig_values_class_1], axis=0
    )  # Shape: [2, num_samples, num_leads, num_time_steps]

    # Return IG values in the desired shape (num_classes, num_samples, num_leads, num_time_steps)
    return ig_values, y_scores, test_y


def main():
    # Training settings+
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=32,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--no-mps",
        action="store_true",
        default=False,
        help="disables macOS GPU training",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="resnet1d",
        help="model type (default: resnet1d)",
    )
    parser.add_argument(
        "--num-channels",
        type=int,
        default=12,
        help="number of channels of EKG to use for input (possible values: 1, 2, 12)",
    )
    parser.add_argument(
        "--channel",
        default=-1,
        help="Individual lead or comma separated leads inside quotes",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=35,
        help="Binary classification threshold for ejection fraction. eg: 35, 40, 45",
    )
    parser.add_argument(
        "--num-blocks",
        type=int,
        default=10,
        help="number of blocks in resnet",
    )
    parser.add_argument(
        "--kernel-size",
        type=int,
        default=16,
        help="kernel size in resnet",
    )
    parser.add_argument(
        "--base-filters",
        type=int,
        default=32,
        help="kernel size in resnet",
    )
    parser.add_argument(
        "--regression",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="new_corrected",
        help="dataset to use (default: new_corrected)",
    )
    parser.add_argument(
        "--model_dir_path",
        type=str,
        default="",
        help="path to saved model",
    )

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if use_cuda:
        torch.cuda.set_device(2)
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model_dir_path = args.model_dir_path
    print(model_dir_path)
    model_path = os.path.join(model_dir_path, "best_f1/model.pt")
    # experiment_path = "/".join(model_dir_path.split("/")[:-1])
    experiment_path = model_dir_path

    _, _, _, _, test_x, test_y = get_train_test_data(
        args.num_channels, args.regression, args.dataset, args.threshold, args.channel
    )

    print(np.unique(test_y, return_counts=True))

    test_x = test_x[:, :2048, :]
    print(test_x.shape, test_y.shape)

    test_x = test_x.reshape(-1, 2048 * args.num_channels)

    scaler_x = pickle.load(open(os.path.join(experiment_path, "scaler_x.pkl"), "rb"))
    test_x = scaler_x.transform(test_x)

    if args.regression:
        scaler_y = pickle.load(
            open(os.path.join(experiment_path, "scaler_y.pkl"), "rb")
        )
        test_y = scaler_y.transform(test_y.reshape(-1, 1))
    else:
        scaler_y = None

    test_x = test_x.reshape(-1, 2048, args.num_channels)

    test_x = test_x.transpose(0, 2, 1)

    test_kwargs = {"batch_size": args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {"num_workers": 4, "pin_memory": True, "shuffle": True}
        test_kwargs.update(cuda_kwargs)

    if args.regression:
        test_dataset = CustomTensorDataset(
            tensors=(
                torch.Tensor(test_x),
                torch.Tensor(test_y),
            ),
            transform=None,
        )
    else:
        test_dataset = CustomTensorDataset(
            tensors=(
                torch.Tensor(test_x),
                torch.Tensor(test_y).type(torch.LongTensor),
            ),
            transform=None,
        )

    # release memory
    test_x = None

    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    model = get_model(
        args.model_type,
        args.num_channels,
        args.regression,
        args.num_blocks,
        args.kernel_size,
        args.base_filters,
    ).to(device)
    test_model = model

    if not args.regression:
        test_model.load_state_dict(torch.load(model_path))
        test_loss, acc, f1, auroc, roc_auc, fpr, tpr = test(
            test_model,
            device,
            test_loader,
            eval_set="test",
            regression=args.regression,
            scaler_y=scaler_y,
            visualize=False,
            visualization_path=model_dir_path,
        )

    ig_values, y_scores, targets = ig_explanation(test_model, test_dataset, device)
    plot_ig2(ig_values, y_scores, cmap=plt.cm.Blues)
    plot_ig2_tp_fp_fn_tn(ig_values, y_scores, targets, cmap=plt.cm.Blues)


if __name__ == "__main__":
    main()
    # python -m interpretability.test_interpretability --num-channels 12 --channel -1 --threshold 35 --model_dir_path saved_models/nch-12-ch-1-th50-all-data-2024-07-30_12-56-28 --model-type resnet1d # noqa: E501
