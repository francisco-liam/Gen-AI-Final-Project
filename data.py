"""
data.py — Fashion-MNIST dataloaders

Images are normalized to [-1, 1], which is standard for diffusion training.
The model learns to add and remove noise in this range.
"""

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_dataloaders(
    batch_size: int = 128,
    num_workers: int = 2,
    data_root: str = "./data",
    val: bool = False,
) -> DataLoader | tuple[DataLoader, DataLoader]:
    """
    Returns a DataLoader for Fashion-MNIST training data.
    If val=True, also returns a DataLoader for the test split.

    Normalization: pixels are scaled to [-1, 1] via mean=0.5, std=0.5.
    This keeps values in a symmetric range around zero, which works well
    with the Gaussian noise added during diffusion.

    Args:
        batch_size:  samples per batch
        num_workers: parallel workers for data loading (set to 0 on Windows
                     if you encounter multiprocessing errors)
        data_root:   directory to download FashionMNIST into
        val:         if True, also return a test-split loader

    Returns:
        train_loader          (if val=False)
        train_loader, val_loader  (if val=True)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),                        # [0, 1]
        transforms.Normalize(mean=(0.5,), std=(0.5,)) # -> [-1, 1]
    ])

    train_dataset = datasets.FashionMNIST(
        root=data_root, train=True, download=True, transform=transform
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,   # keeps every batch the same size
    )

    if val:
        val_dataset = datasets.FashionMNIST(
            root=data_root, train=False, download=True, transform=transform
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        return train_loader, val_loader

    return train_loader
