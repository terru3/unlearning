import os
import random

import numpy as np
import torch
import torchvision
from torchvision import transforms

# from constants import *

## CIFAR-10 only used for ViT at the moment
from vit_constants import *

transform_fn = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        # precomputed CIFAR100 mean and std
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2434, 0.2616)
        ),
    ]
)


## for reverting normalization for visualization purposes
invTrans = transforms.Normalize(
    mean=[-0.4914 / 0.2470, -0.4822 / 0.2434, -0.4465 / 0.2616],
    std=[1 / 0.2470, 1 / 0.2434, 1 / 0.2616],
)

train_data = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform_fn
)
val_data = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform_fn
)

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=BATCH_SIZE, shuffle=False
)
# no shuffle for ease of reproducibility when debugging
val_loader = torch.utils.data.DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

class_labels = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]
