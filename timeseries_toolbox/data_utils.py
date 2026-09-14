"""
Utility functions for train/val/test datasets.
"""

from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset, Subset


def prepare_data(
    data_dir: Path,
    should_normalize: bool = True,
    train_proportion: float = 0.9,
    batch_size: int = 32,
) -> tuple[DataLoader, DataLoader | None, dict[str, Tensor]]:
    """
    Prepare the data for training and validation.

    Args:
        data_dir: Directory to the dataset. This should contain a 'data.pt'
            file as well as a 'masks.pt' file. The former should be a tensor
            of shape (num_sequences, seq_length, num_features) and the latter
            should be a tensor of shape (num_sequences, seq_length) with 1s
            for real values and 0s for missing values.
        should_normalize: Whether to normalize the data.
        train_proportion: The fraction of the dataset to use for training.
    """
    dataset = load_data(data_dir)
    if train_proportion < 1.0:
        train_dataset, val_dataset = make_data_splits(
            dataset,
            train_proportion=train_proportion,
        )
    else:
        train_dataset = dataset
        val_dataset = None
    stats = {}
    if should_normalize:
        train_dataset, stats = normalize_data(train_dataset)
        if val_dataset is not None:
            val_dataset, _ = normalize_data(val_dataset, stats=stats)
    train_data = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
    )
    val_data = None if val_dataset is None else DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
    )
    return train_data, val_data, stats


def load_data(
    data_dir: Path,
) -> TensorDataset:
    """
    Given a path to a dataset directory, load in a dataset we can use.

    Args:
        data_dir: Directory to the dataset. This should contain a 'data.pt'
            file as well as a 'masks.pt' file. The former should be a tensor
            of shape (num_sequences, seq_length, num_features) and the latter
            should be a tensor of shape (num_sequences, seq_length) with 1s
            for real values and 0s for missing values.

    Returns: TensorDataset containing x (inputs), y (labels), and masks. The
        y values are the deltas between the current time point and next point.
    """
    data = torch.load(data_dir / "data.pt")
    masks = torch.load(data_dir / "masks.pt")
    x = data[:, :-1, :]
    y = data[:, 1:, :] - data[:, :-1, :]
    masks = masks[:, :-1]
    dataset = TensorDataset(x, y, masks)
    return dataset


def make_data_splits(
    dataset: TensorDataset,
    train_proportion: float = 0.9,
) -> tuple[Subset, Subset]:
    """
    Given a dataset, split it into training, validation, and test sets.

    Args:
        dataset: The dataset to split.
        train_proportion: The fraction of the dataset to use for training.

    Returns:
        A tuple containing the training, validation, and test data loaders.
    """
    num_samples = len(dataset)
    train_size = int(num_samples * train_proportion)
    val_size = num_samples - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size],
    )
    return train_dataset, val_dataset


def normalize_data(
    data: TensorDataset | Subset,
    stats: dict[str, Tensor] | None = None,
) -> tuple[TensorDataset | Subset, dict[str, Tensor]]:
    """
    Normlalize a dataset.

    Args:
        data: The dataset to noramlize.
        stats: The statistics to use for normalization. If None, the
            statistics will be computed from the data.

    Returns:
        A tuple containing the normalized dataset and the statistics used for
        normalization.
    """
    x_data, y_data, masks = [
        data.dataset[data.indices][i]
        for i in range(3)
    ]
    data_dim = x_data.shape[-1]
    if stats is None:
        stats = {
            "x_mean": x_data.view(-1, data_dim).mean(dim=0),
            "x_std": x_data.view(-1, data_dim).std(dim=0),
            "y_mean": y_data.view(-1, data_dim).mean(dim=0),
            "y_std": y_data.view(-1, data_dim).std(dim=0),
        }
    # Normalize the data
    normed_dataset = TensorDataset(
        (x_data - stats["x_mean"]) / stats["x_std"],
        (y_data - stats["y_mean"]) / stats["y_std"],
        masks,
    )
    return normed_dataset, stats
