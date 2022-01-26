

import os

import numpy as np
import torch
from torch.utils.data import Subset
from torchvision import transforms, datasets
import torchvision.transforms.functional as tf

from guided_diffusion.dataset_utils import TransformedDataset

BASE_PATH = os.path.expandvars("$WORKSPACE/pmneila/cityscapes/")
# BASE_PATH = os.path.expandvars("/home/lukas/datasets/cityscapes/")
# see datasets.Cityscapes:     classes = [ ...
# https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
CATEGORYMAP = np.array([
    0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1,
    2, 2, 2, 2, 2, 2,
    3, 3, 3, 3,
    4, 4,
    5,
    6, 6,
    7, 7, 7, 7, 7, 7, 7, 7, 7
])
CLASSMAP = np.array([
    0, 0, 0, 0, 0, 0, 0,
    1, 2, 0, 0,
    3, 4, 5, 0, 0, 0,
    6, 0, 7, 8,
    9, 10,
    11,
    12, 13,
    14, 15, 16, 0, 0, 17, 18, 19, 0
])
LABEL_MAP = CLASSMAP
NUM_CLASSES = len(np.unique(LABEL_MAP))


def labels_to_categories(arr: np.ndarray) -> np.ndarray:
    return LABEL_MAP[arr]


def one_hot_encoding(arr: np.ndarray) -> np.ndarray:
    res = np.zeros(arr.shape + (NUM_CLASSES,), dtype=np.float32)
    h, w = np.ix_(np.arange(arr.shape[0]), np.arange(arr.shape[1]))
    res[h, w, arr] = 1.0
    return res


def resize(arr: torch.Tensor) -> torch.Tensor:
    return arr[..., ::4, ::4]


def to_tensor(arr: np.ndarray) -> torch.Tensor:
    if arr.ndim == 2:
        arr = arr[:, :, None]

    return torch.from_numpy(arr.transpose((2, 0, 1))).contiguous()


def training_transform(image, labels):

    labels = labels_to_categories(labels)
    labels = one_hot_encoding(labels)

    image = tf.to_tensor(image)
    labels = tf.to_tensor(labels)

    image = resize(image)
    labels = resize(labels)

    i, j, th, tw = transforms.RandomCrop.get_params(labels, (256, 256))
    image = tf.crop(image, i, j, th, tw)
    labels = tf.crop(labels, i, j, th, tw)

    if torch.rand(1) < 0.5:
        image = tf.hflip(image)
        labels = tf.hflip(labels)

    return image, labels


def training_dataset():
    dataset = datasets.Cityscapes(root=BASE_PATH, split="train", mode="fine", target_type="semantic")
    dataset = TransformedDataset(dataset, training_transform)
    return dataset


def validation_transform(image, labels):

    labels = labels_to_categories(labels)
    labels = one_hot_encoding(labels)

    image = tf.to_tensor(image)
    labels = tf.to_tensor(labels)

    image = resize(image)
    labels = resize(labels)

    image = tf.center_crop(image, (256, 256))
    labels = tf.center_crop(labels, (256, 256))

    return image, labels


def validation_dataset():
    dataset = datasets.Cityscapes(root=BASE_PATH, split="val", mode="fine", target_type="semantic")
    dataset = TransformedDataset(dataset, validation_transform)
    return dataset
