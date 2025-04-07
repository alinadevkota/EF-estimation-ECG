import numpy as np
from torch.utils.data import Dataset


def get_channel_indices(num_channels, channel):
    if num_channels == 1:
        return map_channel_idx(str(channel))
    elif num_channels == 12:
        channels = [map_channel_idx(x) for x in range(12)]
        return channels
    elif num_channels >= 2:
        channels = channel.split(",")
        channels = [map_channel_idx(x) for x in channels]
        return channels


def map_channel_idx(channel):
    if channel == "0" or "I":
        return 0
    elif channel == "1" or "II":
        return 1
    elif channel == "2" or "III":
        return 2
    elif channel == "3" or "aVR":
        return 3
    elif channel == "4" or "aVL":
        return 4
    elif channel == "5" or "aVF":
        return 5
    elif channel == "6" or "V1":
        return 6
    elif channel == "7" or "V2":
        return 7
    elif channel == "8" or "V3":
        return 8
    elif channel == "9" or "V4":
        return 9
    elif channel == "10" or "V5":
        return 10
    elif channel == "11" or "V6":
        return 11


def mean_std(loader):
    images, lebels = next(iter(loader))
    # shape of images = [b,c,l]
    mean, std = images.mean([0, 2]), images.std([0, 2])
    return mean, std


def augmentation(x, y):
    max_start_idx = x.shape[1] - 2048 - 1

    noise1 = np.random.normal(0, 10, x.shape)
    aug_x1 = x + noise1
    aug_y1 = y
    start_indices1 = np.random.randint(max_start_idx, size=(x.shape[0]))
    aug_x1 = np.concatenate(
        [
            np.expand_dims(
                aug_x1[
                    i, start_indices1[i].item() : start_indices1[i].item() + 2048, :
                ],
                axis=0,
            )
            for i in range(aug_x1.shape[0])
        ],
        axis=0,
    )

    noise2 = np.random.normal(0, 15, x.shape)
    aug_x2 = x + noise2
    aug_y2 = y
    start_indices2 = np.random.randint(max_start_idx, size=(x.shape[0]))
    aug_x2 = np.concatenate(
        [
            np.expand_dims(
                aug_x2[
                    i, start_indices2[i].item() : start_indices2[i].item() + 2048, :
                ],
                axis=0,
            )
            for i in range(aug_x2.shape[0])
        ],
        axis=0,
    )

    start_indices0 = np.random.randint(max_start_idx, size=(x.shape[0]))
    x = np.concatenate(
        [
            np.expand_dims(
                x[i, start_indices0[i].item() : start_indices0[i].item() + 2048, :],
                axis=0,
            )
            for i in range(x.shape[0])
        ],
        axis=0,
    )

    return np.concatenate([x, aug_x1, aug_x2], axis=0), np.concatenate(
        [y, aug_y1, aug_y2], axis=0
    )


class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms."""

    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        print(tensors[0].shape)
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)
