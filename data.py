"""
data.py — Fashion-MNIST and CIFAR-10 dataloaders

Images are normalized to [-1, 1], which is standard for diffusion training.
The model learns to add and remove noise in this range.
"""

import warnings

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Per-dataset metadata used by train.py, sample.py, and analyze.py.
DATASET_INFO = {
    "fashionmnist": {"in_channels": 1, "image_size": 28},
    "cifar10":      {"in_channels": 3, "image_size": 32},
}

SUPPORTED_DATASETS = list(DATASET_INFO.keys())


def get_dataloaders(
    batch_size: int = 128,
    num_workers: int = 2,
    data_root: str = "./data",
    val: bool = False,
    dataset: str = "fashionmnist",
) -> DataLoader | tuple[DataLoader, DataLoader]:
    """
    Returns a DataLoader for the specified dataset (training split).
    If val=True, also returns a DataLoader for the test split.

    Supported datasets: "fashionmnist", "cifar10"

    Normalization: pixels are scaled to [-1, 1] via mean=0.5, std=0.5
    (applied per-channel; CIFAR-10 uses three-channel normalization).

    Args:
        batch_size:  samples per batch
        num_workers: parallel workers for data loading (set to 0 on Windows
                     if you encounter multiprocessing errors)
        data_root:   directory to download the dataset into
        val:         if True, also return a test-split loader
        dataset:     one of "fashionmnist" or "cifar10"

    Returns:
        train_loader              (if val=False)
        train_loader, val_loader  (if val=True)
    """
    if dataset == "fashionmnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,)),
        ])
        cls = datasets.FashionMNIST
    elif dataset == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])
        cls = datasets.CIFAR10
    else:
        raise ValueError(
            f"Unknown dataset '{dataset}'. Choose from: {SUPPORTED_DATASETS}"
        )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="torchvision")
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="numpy")
        train_dataset = cls(root=data_root, train=True, download=True, transform=transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,   # keeps every batch the same size
    )

    if val:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning, module="torchvision")
            warnings.filterwarnings("ignore", category=DeprecationWarning, module="numpy")
            val_dataset = cls(root=data_root, train=False, download=True, transform=transform)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        return train_loader, val_loader

    return train_loader
