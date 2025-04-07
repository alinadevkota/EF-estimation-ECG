import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from data_utils import get_channel_indices


def create_classes(value, threshold=35):
    if 0 < value and value <= threshold:
        return 0
    else:
        return 1



def generate_for_corrected(ef_match_window):
    train_root = "./data/xml_csv_all/"

    data_dir = "./resources/new_data/"

    dtrain_path = os.path.join(data_dir, f"train_data_w{ef_match_window}.npy")
    ltrain_path = os.path.join(
        data_dir, f"train_label_regression_w{ef_match_window}.npy"
    )
    dval_path = os.path.join(data_dir, f"val_data_w{ef_match_window}.npy")
    lval_path = os.path.join(data_dir, f"val_label_regression_w{ef_match_window}.npy")
    dtest_path = os.path.join(data_dir, f"test_data_w{ef_match_window}.npy")
    ltest_path = os.path.join(data_dir, f"test_label_regression_w{ef_match_window}.npy")

    df = pd.read_csv("resources/new_data/data.csv")
    df = df[df["target"].notna()]
    df = df.reset_index(drop=True)

    train_df = df.reset_index(drop=True)

    train_labels = []

    print("------------------------LOADING TRAIN DATA------------------------")
    train_data = []
    train_date_diff = []
    for i, csv_path in enumerate(train_df["FILENAME"].values):
        df = pd.read_csv(os.path.join(train_root, csv_path.replace(".xml", ".csv")))
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
        df = df.drop(" ", axis=1)
        df_numpy = df.to_numpy()

        if df_numpy.shape != (2500, 12):
            if df_numpy.shape[1] != 12:
                continue
            df_numpy = df_numpy[:2500, :12]
        label = train_df.loc[i]["target"]
        if label > 100:
            continue
        train_data.append(df_numpy)
        train_labels.append(label)
        train_date_diff.append(train_df.loc[i]["Date_Difference"])

    train_labels = np.array(train_labels)
    train_data = np.array(train_data)

    (
        train_data,
        test_data,
        train_labels,
        test_labels,
        train_date_diff,
        test_date_diff,
    ) = train_test_split(
        train_data, train_labels, train_date_diff, test_size=0.2, random_state=42
    )
    train_data, val_data, train_labels, val_labels, train_date_diff, val_date_diff = (
        train_test_split(
            train_data, train_labels, train_date_diff, test_size=0.2, random_state=42
        )
    )

    filtered_train_data = []
    filtered_train_labels = []
    for data, label, date_diff in zip(train_data, train_labels, train_date_diff):
        if date_diff <= ef_match_window:
            filtered_train_data.append(data)
            filtered_train_labels.append(label)
    train_data, train_labels = np.array(filtered_train_data), np.array(
        filtered_train_labels
    )

    
    # save data
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    np.save(dtrain_path, train_data)
    np.save(ltrain_path, train_labels)
    np.save(dval_path, val_data)
    np.save(lval_path, val_labels)
    np.save(dtest_path, test_data)
    np.save(ltest_path, test_labels)

    return train_data, train_labels, val_data, val_labels, test_data, test_labels


def _get_train_test_data(
    num_channels, regression, threshold, channel_indices, ef_match_window
):
    data_dir = "./resources/new_data/"
    dtrain_path = os.path.join(data_dir, f"train_data_w{ef_match_window}.npy")
    ltrain_path = os.path.join(
        data_dir, f"train_label_regression_w{ef_match_window}.npy"
    )
    dval_path = os.path.join(data_dir, f"val_data_w{ef_match_window}.npy")
    lval_path = os.path.join(data_dir, f"val_label_regression_w{ef_match_window}.npy")
    dtest_path = os.path.join(data_dir, f"test_data_w{ef_match_window}.npy")
    ltest_path = os.path.join(data_dir, f"test_label_regression_w{ef_match_window}.npy")

    data_path_lists = [
        dtrain_path,
        ltrain_path,
        dval_path,
        lval_path,
        dtest_path,
        ltest_path,
    ]

    if not all(os.path.isfile(fpath) for fpath in data_path_lists):
        train_data, train_labels, val_data, val_labels, test_data, test_labels = (
            generate_for_corrected(ef_match_window)
        )
    else:
        train_data = np.load(dtrain_path)
        train_labels = np.load(ltrain_path)
        val_data = np.load(dval_path)
        val_labels = np.load(lval_path)
        test_data = np.load(dtest_path)
        test_labels = np.load(ltest_path)

    if not regression:
        train_labels = np.array([create_classes(x, threshold) for x in train_labels])
        val_labels = np.array([create_classes(x, threshold) for x in val_labels])
        test_labels = np.array([create_classes(x, threshold) for x in test_labels])

    if num_channels == 12:
        return (
            train_data,
            train_labels,
            val_data,
            val_labels,
            test_data,
            test_labels,
        )
    elif num_channels == 1:
        return (
            np.expand_dims(train_data[:, :, channel_indices], axis=-1),
            train_labels,
            np.expand_dims(val_data[:, :, channel_indices], axis=-1),
            val_labels,
            np.expand_dims(test_data[:, :, channel_indices], axis=-1),
            test_labels,
        )
    elif num_channels >= 2 and num_channels < 12:
        return (
            train_data[:, :, channel_indices],
            train_labels,
            val_data[:, :, channel_indices],
            val_labels,
            test_data[:, :, channel_indices],
            test_labels,
        )


def get_train_test_data(
    num_channels, regression, dataset, threshold, channel=None, ef_match_window=60
):
    channel_indices = get_channel_indices(num_channels, channel)

    return _get_train_test_data(
        num_channels, regression, threshold, channel_indices, ef_match_window
    )

