from __future__ import print_function

import argparse
import os
import random

import numpy as np
from cuml.ensemble import RandomForestClassifier as cuRF
from cuml.svm import SVR
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_curve,
)

from data_utils import augmentation
from generate_dataset import get_train_test_data



def get_model(model_type):
    if model_type == "random_forest":
        return cuRF(n_estimators=15, random_state=0, split_criterion="gini")
    elif model_type == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=1000, learning_rate=0.1, max_depth=20, random_state=0
        )
    elif model_type == "svm":
        return SVR()


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
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
        default=0,
        help="Individual lead or comma separated leads inside quotes",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=35,
        help="Binary classification threshold for ejection fraction. eg: 35, 40, 45, 50",
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

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    train_x, train_y, val_x, val_y, test_x, test_y = get_train_test_data(
        args.num_channels, args.regression, args.dataset, args.threshold, args.channel
    )
    print(
        train_x.shape,
        train_y.shape,
        val_x.shape,
        val_y.shape,
        test_x.shape,
        test_y.shape,
    )

    if not args.regression:
        class1_indices = np.where(train_y == 1)[0]
        class0_indices = np.where(train_y == 0)[0]

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

    train_x = train_x.reshape(-1, 2048 * args.num_channels)
    test_x = test_x.reshape(-1, 2048 * args.num_channels)
    val_x = val_x.reshape(-1, 2048 * args.num_channels)

    scaler_x = preprocessing.StandardScaler().fit(train_x)
    train_x = scaler_x.transform(train_x)
    val_x = scaler_x.transform(val_x)
    test_x = scaler_x.transform(test_x)

    if args.regression:
        scaler_y = preprocessing.StandardScaler().fit(train_y.reshape(-1, 1))
        train_y = scaler_y.transform(train_y.reshape(-1, 1))
        val_y = scaler_y.transform(val_y.reshape(-1, 1))
        test_y = scaler_y.transform(test_y.reshape(-1, 1))
    else:
        scaler_y = None

    print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)
    print("training model")
    model = get_model(args.model_type)

    model.fit(train_x, train_y)
    # if args.model_type == "svm":
    #     model = model.best_estimator_

    pred_y = model.predict(train_x)
    fpr, tpr, th = roc_curve(train_y, pred_y)
    roc_auc = auc(fpr, tpr)
    if args.model_type == "svm":
        pred_y = [0 if x < 0.5 else 1 for x in pred_y]

    cm = confusion_matrix(train_y, pred_y)
    print(cm)
    cm_normed = cm / cm.sum(axis=1)[:, np.newaxis]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_normed)
    disp.plot(cmap="Blues").figure_.savefig(
        os.path.join("resources", f"confusion_matrix_{args.model_type}.png")
    )
    print(classification_report(train_y, pred_y, digits=4))
    f1 = f1_score(train_y, pred_y, average="macro")

    acc = accuracy_score(train_y, pred_y) * 100

    print(
        "\n{} set: \n Accuracy on Train Set: {:.0f}%\n, F1-Score: {}".format(
            "Train", acc, f1
        )
    )

    pred_y = model.predict(test_x)
    if args.model_type == "svm":
        prob_y = pred_y
    else:
        prob_y = model.predict_proba(test_x)[:, 1]
    fpr, tpr, th = roc_curve(test_y, prob_y)
    roc_auc = auc(fpr, tpr)
    if args.model_type == "svm":
        pred_y = [0 if x < 0.5 else 1 for x in pred_y]

    cm = confusion_matrix(test_y, pred_y)
    print(cm)
    cm_normed = cm / cm.sum(axis=1)[:, np.newaxis]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_normed)
    disp.plot(cmap="Blues").figure_.savefig(
        os.path.join("resources", f"confusion_matrix_{args.model_type}.png")
    )
    print(classification_report(test_y, pred_y, digits=4))
    f1 = f1_score(test_y, pred_y, average="macro")

    acc = accuracy_score(test_y, pred_y) * 100

    print("\n{} set: \n Accuracy: {:.0f}%\n, F1-Score: {}".format("Test", acc, f1))
    obj = {"tpr": tpr, "fpr": fpr, "roc_auc": roc_auc, "acc": acc, "f1": f1}
    np.save(
        f"resources/res_model{args.model_type}_numch{args.num_channels}_ch{args.channel}_th{args.threshold}_seed{args.seed}.npy",  # noqa: E501
        obj,
    )


if __name__ == "__main__":
    main()
