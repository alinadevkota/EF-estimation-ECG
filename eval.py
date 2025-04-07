from __future__ import print_function

import argparse
import os
import pickle
import random

# import matplotlib.pyplot as plt
import numpy as np
import torch

from main import CustomTensorDataset, get_model, get_train_test_data, test


def main():
    # Training settings
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
            visualize=True,
            visualization_path=model_dir_path,
        )

    obj = {"tpr": tpr, "fpr": fpr, "roc_auc": roc_auc}
    np.save(
        f"resources/res_model{args.model_type}_numch{args.num_channels}_ch{args.channel}_th{args.threshold}.npy",
        obj,
    )


if __name__ == "__main__":
    main()
