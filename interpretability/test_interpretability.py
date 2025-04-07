import argparse
import copy
import os
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from captum.attr import IntegratedGradients, visualization

from main import CustomTensorDataset, get_model, get_train_test_data, test


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
        torch.cuda.set_device(0)
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

    # ImageClassifier takes a single input tensor of images Nx3x32x32,
    # and returns an Nx10 tensor of class probabilities.
    net = test_model
    ig = IntegratedGradients(net)

    for idx in range(len(test_dataset)):
        ekg, label = test_dataset[idx]
        ekg = ekg[None, :, :]

        _ekg = copy.copy(ekg)
        ekg = ekg.to(device)
        label = label.reshape(1, 1)
        label = label.to(device)
        print(ekg.shape, label)
        # Computes integrated gradients for class 3.
        attribution = ig.attribute(ekg, target=label)

        # print(type(attribution[0]))
        # print(attribution[0].shape if hasattr(attribution[0], 'shape') else "No shape")
        # print(type(ekg[0]))
        # print(ekg[0].shape if hasattr(ekg[0], 'shape') else "No shape")

        attribution = attribution.permute(0, 2, 1)
        ekg = ekg.permute(0, 2, 1)
        print(attribution.shape, ekg.shape)

        fig, ax = visualization.visualize_timeseries_attr(
            attribution[0].cpu(),
            ekg[0].cpu(),
            method="overlay_individual",
            alpha_overlay=1.0,
            cmap="PuRd",
            fig_size=(10, 10),
            channel_labels=[
                "I",
                "II",
                "III",
                "aVR",
                "aVL",
                "aVF",
                "V1",
                "V2",
                "V3",
                "V4",
                "V5",
                "V6",
            ],
        )

        pred = net(_ekg.to(device))
        label_val = label.item()
        pred_val = torch.argmax(pred).item()
        if pred_val == 0 and label_val == 0:
            title = "TP"  # True Positive
        elif pred_val == 0 and label_val == 1:
            title = "FP"  # False Positive
        elif pred_val == 1 and label_val == 0:
            title = "FN"  # False Negative
        elif pred_val == 1 and label_val == 1:
            title = "TN"  # True Negative
        fig.suptitle(
            f"Patient-level Interpretation using Integrated Gradients for {title}",
            fontsize="large",
            y=0.91,
        )
        
        plt.savefig(
            f"interpretability/results/interpretability_ig_{idx}_label{label_val}_pred{pred_val}.png",
            dpi=300,
            bbox_inches="tight",
        )

        if idx > 9:
            break


if __name__ == "__main__":
    main()
    # python -m interpretability.test_interpretability --num-channels 12 --channel -1 --threshold 35 --model_dir_path saved_models/nch-12-ch-1-th50-all-data-2024-07-30_12-56-28 --model-type resnet1d # noqa: E501

    # python -m interpretability.test_interpretability --num-channels 12 --channel -1 --threshold 35 --model_dir_path saved_models/nch-12-ch-1-th35-all-data-2024-07-30_09-45-02 --model-type resnet1d # noqa: E501

    # python -m interpretability.test_interpretability --num-channels 12 --channel -1 --threshold 40 --model_dir_path saved_models/nch-12-ch-1-th40-all-data-2024-07-30_10-48-47 --model-type resnet1d # noqa: E501

    # python -m interpretability.test_interpretability --num-channels 12 --channel -1 --threshold 45 --model_dir_path saved_models/nch-12-ch-1-th45-all-data-2024-07-30_11-52-32 --model-type resnet1d # noqa: E501

    # python -m interpretability.test_interpretability --num-channels 12 --channel -1 --threshold 50 --model_dir_path saved_models/nch-12-ch-1-th50-all-data-2024-07-30_12-56-28 --model-type resnet1d # noqa: E501
