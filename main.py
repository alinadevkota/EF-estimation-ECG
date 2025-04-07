from __future__ import print_function

import argparse
import datetime
import os
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import scikitplot as skplt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn import preprocessing
from sklearn.metrics import (  # ConfusionMatrixDisplay,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    roc_auc_score,
    roc_curve,
)

import wandb
from data_utils import CustomTensorDataset, augmentation
from ecgformer import ECGformer
from generate_dataset import get_train_test_data
from models import Net, ResNet1D

os.environ["WANDB_MODE"] = "offline"

wandb.login()


def train(
    args, model, device, train_loader, optimizer, epoch, regression, weight=(1, 1)
):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        if not regression:
            # loss = F.nll_loss(output, target, weight=torch.Tensor(weight).to(device))
            loss = nn.CrossEntropyLoss(weight=torch.Tensor(weight).to(device))(
                output, target
            )
        else:
            loss = F.mse_loss(output, target.float())
        # print(loss)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
            if args.dry_run:
                break
    # print(model.first_block_conv.conv.weight)


def test(
    model,
    device,
    test_loader,
    eval_set,
    regression,
    scaler_y=None,
    visualize=None,
    visualization_path="./",
):
    model.eval()
    test_loss = 0
    correct = 0

    if not regression:
        with torch.no_grad():
            all_targets = []
            all_preds = []
            all_probs = []
            all_probs_all_dim = []
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(
                    output,
                    target,
                    reduction="sum",
                ).item()  # sum up batch loss
                pred = output.argmax(
                    dim=1, keepdim=True
                )  # get the index of the max log-probability
                all_probs.extend(output[:, 1].tolist())
                all_probs_all_dim.extend(output.tolist())
                all_targets.extend(target.tolist())
                all_preds.extend(pred.tolist())
                correct += pred.eq(target.view_as(pred)).sum().item()
            cm = confusion_matrix(all_targets, all_preds)
            print(cm)
            if visualize:
                skplt.metrics.plot_roc(all_targets, all_probs_all_dim, plot_micro=False)
                plt.savefig(os.path.join(visualization_path, "AUROC.png"), dpi=300)
            print(classification_report(all_targets, all_preds))
            f1 = f1_score(all_targets, all_preds, average="macro")
            auroc = roc_auc_score(all_targets, all_probs, multi_class="ovr")
            fpr, tpr, th = roc_curve(all_targets, all_probs)
            roc_auc = auc(fpr, tpr)

        test_loss /= len(test_loader.dataset)

        acc = 100.0 * correct / len(test_loader.dataset)

        print(
            "\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n, F1-Score: {}\n, AUROC: {}".format(
                eval_set, test_loss, correct, len(test_loader.dataset), acc, f1, auroc
            )
        )
        return test_loss, acc, f1, auroc, roc_auc, fpr, tpr
    else:
        with torch.no_grad():
            all_targets = []
            all_preds = []
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.mse_loss(
                    output,
                    target,
                    reduction="sum",
                ).item()
                pred = output
                all_targets.extend(target.tolist())
                all_preds.extend(pred.tolist())

        test_loss /= len(test_loader.dataset)
        all_targets = scaler_y.inverse_transform(np.array(all_targets).reshape(-1, 1))
        all_preds = scaler_y.inverse_transform(np.array(all_preds).reshape(-1, 1))
        mae = mean_absolute_error(all_targets, all_preds)

        print(
            "\n{} set: Average loss: {:.4f}, MAE: {:.0f}\n".format(
                eval_set,
                test_loss,
                mae,
            )
        )
        return test_loss, mae, None, None, None, None, None


def get_model(model_type, num_channels, regression, n_block, kernel_size, base_filters):
    print("----------------------------model hyperparams------------------------")
    print(
        f"num blocks: {n_block}, kernel size: {kernel_size}, base filters: {base_filters}"
    )
    print(num_channels)
    if model_type == "resnet1d":
        return ResNet1D(
            in_channels=num_channels,
            base_filters=base_filters,
            kernel_size=kernel_size,
            stride=2,
            groups=32,
            n_block=n_block,
            n_classes=2,
            regression=regression,
        )
    elif model_type == "mlp":
        return Net(in_channels=num_channels, regression=regression)

    elif model_type == "transformer":
        return ECGformer(
            num_layers=4,
            signal_length=256,
            num_classes=2,
            input_channels=num_channels,
            embed_size=64,
            num_heads=4,
            expansion=2,
        )


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        metavar="LR",
        help="learning rate (default: 1.0)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        metavar="M",
        help="Learning rate step gamma (default: 0.7)",
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
        "--dry-run",
        action="store_true",
        default=False,
        help="quickly check a single pass",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=50,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=True,
        help="For Saving the current Model",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="resnet1d",
        help="model type (default: resnet1d)",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="tag name for experiment",
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
        help="Binary classification threshold for ejection fraction. eg: 35, 40, 45, 50",
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
        "--undersample",
        action="store_true",
        default=False,
        help="Whether to undersample higher occurring classes",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="new_corrected",
        help="dataset to use (default: new_corrected)",
    )
    parser.add_argument(
        "--ef-match-window",
        type=int,
        default=60,
        help="Maximum Date Difference for creating EF-ECG pair in dataset",
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

    config = vars(args)

    model_dir = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_dir_path = os.path.join(
        "saved_models",
        f"rebuttal_nch-{args.num_channels}-ch{args.channel}-th{args.threshold}-{args.tag}-{model_dir}",
    )
    os.makedirs(model_dir_path)

    if args.regression:
        best_mae_path = os.path.join(model_dir_path, "best_mae")
        os.makedirs(best_mae_path)
    else:
        best_f1_path = os.path.join(model_dir_path, "best_f1")
        best_acc_path = os.path.join(model_dir_path, "best_acc")
        os.makedirs(best_f1_path)
        os.makedirs(best_acc_path)

    config["model_dir"] = model_dir_path
    run = wandb.init(  # noqa:
        # Set the project where this run will be logged
        project="EKG_cleaned_data",
        # Track hyperparameters and run metadata
        config=config,
    )

    train_x, train_y, val_x, val_y, test_x, test_y = get_train_test_data(
        args.num_channels,
        args.regression,
        args.dataset,
        args.threshold,
        args.channel,
        args.ef_match_window,
    )
    print(
        train_x.shape,
        train_y.shape,
        val_x.shape,
        val_y.shape,
        test_x.shape,
        test_y.shape,
    )
    print(len(train_y), np.unique(train_y, return_counts=True))
    print(len(val_y), np.unique(val_y, return_counts=True))
    print(len(test_y), np.unique(test_y, return_counts=True))

    # train_y[train_y == 2] = 1
    # test_y[test_y == 2] = 1

    if not args.regression:
        class1_indices = np.where(train_y == 1)[0]
        class0_indices = np.where(train_y == 0)[0]

        class1_class0_ratio = len(class1_indices) / len(class0_indices)
    else:
        class1_class0_ratio = None

    # undersampling
    if not args.regression and args.undersample:
        indices_to_remove2 = random.sample(
            list(class1_indices), len(class1_indices) - len(class0_indices)
        )
        train_x = np.delete(train_x, (indices_to_remove2), axis=0)
        train_y = np.delete(train_y, (indices_to_remove2), axis=0)

    val_x = val_x[:, :2048, :]
    test_x = test_x[:, :2048, :]

    train_x, train_y = augmentation(train_x, train_y)
    # print(train_x.shape, train_y.shape)

    train_x = train_x.reshape(-1, 2048 * args.num_channels)
    test_x = test_x.reshape(-1, 2048 * args.num_channels)
    val_x = val_x.reshape(-1, 2048 * args.num_channels)

    scaler_x = preprocessing.StandardScaler().fit(train_x)
    pickle.dump(scaler_x, open(os.path.join(model_dir_path, "scaler_x.pkl"), "wb"))
    train_x = scaler_x.transform(train_x)
    val_x = scaler_x.transform(val_x)
    test_x = scaler_x.transform(test_x)

    if args.regression:
        scaler_y = preprocessing.StandardScaler().fit(train_y.reshape(-1, 1))
        pickle.dump(scaler_y, open(os.path.join(model_dir_path, "scaler_y.pkl"), "wb"))
        train_y = scaler_y.transform(train_y.reshape(-1, 1))
        val_y = scaler_y.transform(val_y.reshape(-1, 1))
        test_y = scaler_y.transform(test_y.reshape(-1, 1))
    else:
        scaler_y = None

    train_x = train_x.reshape(-1, 2048, args.num_channels)
    val_x = val_x.reshape(-1, 2048, args.num_channels)
    test_x = test_x.reshape(-1, 2048, args.num_channels)

    if args.model_type == "transformer":
        factor = 8
        train_x = train_x[:, ::factor, :]
        val_x = val_x[:, ::factor, :]
        test_x = test_x[:, ::factor, :]

    train_x = train_x.transpose(0, 2, 1)
    val_x = val_x.transpose(0, 2, 1)
    test_x = test_x.transpose(0, 2, 1)

    train_kwargs = {"batch_size": args.batch_size}
    val_kwargs = {"batch_size": args.test_batch_size}
    test_kwargs = {"batch_size": args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {"num_workers": 4, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        val_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    if args.regression:
        train_dataset = CustomTensorDataset(
            tensors=(
                torch.Tensor(train_x),
                torch.Tensor(train_y),
            ),
            transform=None,
        )
        val_dataset = CustomTensorDataset(
            tensors=(
                torch.Tensor(val_x),
                torch.Tensor(val_y),
            ),
            transform=None,
        )
        test_dataset = CustomTensorDataset(
            tensors=(
                torch.Tensor(test_x),
                torch.Tensor(test_y),
            ),
            transform=None,
        )
    else:
        train_dataset = CustomTensorDataset(
            tensors=(
                torch.Tensor(train_x),
                torch.Tensor(train_y).type(torch.LongTensor),
            ),
            transform=None,
        )
        val_dataset = CustomTensorDataset(
            tensors=(
                torch.Tensor(val_x),
                torch.Tensor(val_y).type(torch.LongTensor),
            ),
            transform=None,
        )
        test_dataset = CustomTensorDataset(
            tensors=(
                torch.Tensor(test_x),
                torch.Tensor(test_y).type(torch.LongTensor),
            ),
            transform=None,
        )

    # release memory
    train_x = None
    val_x = None
    test_x = None

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, **val_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)  # noqa: F841

    model = get_model(
        args.model_type,
        args.num_channels,
        args.regression,
        args.num_blocks,
        args.kernel_size,
        args.base_filters,
    ).to(device)
    test_model = model

    optimizer = optim.Adam(model.parameters(), args.lr)

    best_f1 = 0
    best_acc = 0
    best_f1_epoch = -1
    best_acc_epoch = -1
    for epoch in range(1, args.epochs + 1):
        train(
            args,
            model,
            device,
            train_loader,
            optimizer,
            epoch,
            regression=args.regression,
            weight=(class1_class0_ratio, 1),
        )
        train_loss, train_acc, train_f1, train_auroc, _, _, _ = test(
            model,
            device,
            train_loader,
            eval_set="Train",
            regression=args.regression,
            scaler_y=scaler_y,
        )
        val_loss, val_acc, val_f1, val_auroc, _, _, _ = test(
            model,
            device,
            val_loader,
            eval_set="val",
            regression=args.regression,
            scaler_y=scaler_y,
        )

        # scheduler.step()
        if not args.regression:
            if val_acc > best_acc:
                torch.save(model.state_dict(), os.path.join(best_acc_path, "model.pt"))
                best_acc_epoch = epoch
                best_acc = val_acc
            if val_f1 > best_f1:
                torch.save(model.state_dict(), os.path.join(best_f1_path, "model.pt"))
                best_f1_epoch = epoch
                best_f1 = val_f1
            wandb.log(
                {
                    "train_accuracy": train_acc,
                    "train_loss": train_loss,
                    "val_accuracy": val_acc,
                    "val_loss": val_loss,
                    "train f1-score (macro)": train_f1,
                    "val f1-score (macro)": val_f1,
                    "train_auroc": train_auroc,
                    "val_auroc": val_auroc,
                }
            )
        else:
            val_mae = val_acc
            train_mae = train_acc
            if epoch == 1:
                best_mae_epoch = 1
                best_mae = val_mae
                torch.save(model.state_dict(), os.path.join(best_mae_path, "model.pt"))
            else:
                if val_mae < best_mae:
                    best_mae = val_mae
                    best_mae_epoch = epoch
                    torch.save(
                        model.state_dict(), os.path.join(best_mae_path, "model.pt")
                    )
            wandb.log(
                {
                    "train_mae": train_mae,
                    "train_loss": train_loss,
                    "val_mae": val_mae,
                    "val_loss": val_loss,
                }
            )

    # log metrics and save best model for classification in test set
    if not args.regression:
        wandb.log(
            {
                "best_val_acc": best_acc,
                "best_val_acc_epoch": best_acc_epoch,
                "best_val_f1": best_f1,
                "best_val_f1_epoch": best_f1_epoch,
            }
        )

        test_model.load_state_dict(torch.load(os.path.join(best_acc_path, "model.pt")))
        test_loss, test_acc, test_f1, test_auroc, _, _, _ = test(
            test_model,
            device,
            test_loader,
            eval_set="test",
            regression=args.regression,
            scaler_y=scaler_y,
            visualize=True,
            visualization_path=best_acc_path,
        )
        wandb.log(
            {
                "test_accuracy_(best_acc)": test_acc,
                "test_loss_(best_acc)": test_loss,
                "test_f1_(best_acc)": test_f1,
                "test_auroc_(best_acc)": test_auroc,
            }
        )

        test_model.load_state_dict(torch.load(os.path.join(best_f1_path, "model.pt")))
        test_loss, test_acc, test_f1, test_auroc, _, _, _ = test(
            test_model,
            device,
            test_loader,
            eval_set="test",
            regression=args.regression,
            scaler_y=scaler_y,
            visualize=True,
            visualization_path=best_f1_path,
        )
        wandb.log(
            {
                "test_accuracy_(best_f1)": test_acc,
                "test_loss_(best_f1)": test_loss,
                "test_f1_(best_f1)": test_f1,
                "test_auroc_(best_f1)": test_auroc,
            }
        )
    else:
        wandb.log(
            {
                "best_val_mae": best_mae,
                "best_val_mae_epoch": best_mae_epoch,
            }
        )

        test_model.load_state_dict(torch.load(os.path.join(best_mae_path, "model.pt")))
        test_loss, test_mae, _, _, _, _, _ = test(
            test_model,
            device,
            test_loader,
            eval_set="test",
            regression=args.regression,
            scaler_y=scaler_y,
            visualize=True,
            visualization_path=best_mae_path,
        )
        wandb.log(
            {
                "test_mae": test_mae,
                "test_loss": test_loss,
            }
        )

    # if args.save_model:
    #     torch.save(model.state_dict(), f"./saved_models/{args.model_type}.pt")


if __name__ == "__main__":
    main()
