from collections import defaultdict
import logging

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from timeseries_toolbox.models.base_timeseries import BaseTimeSeriesModel


logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more details
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def train_model(
    model: BaseTimeSeriesModel,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    epochs: int,
    save_path: str,
    validate_every: int = 10,
) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    """
    Train a model.

    Args:
        model: The model to train.
        train_loader: The training data loader.
        val_loader: The validation data loader.
        epochs: The number of epochs to train for.
        save_path: The path to save the model.
        validate_every: The number of epochs between validation checks.

    Returns: Statistics for training and validation.
    """
    optimizer = model.optimizer
    train_stats, val_stats = [], []
    for ep in range(epochs):
        ep_stats = defaultdict(float)
        for batch in train_loader:
            optimizer.zero_grad()
            net_in, labels, masks = batch
            loss, stats = model.train_loss(net_in, labels, masks)
            loss.backward()
            optimizer.step()
            for k, v in stats.items():
                ep_stats[k] += v / len(train_loader)
        logger.info("=" * 50)
        logger.info(f"Epoch {ep}")
        logger.info("Train Statistics")
        for k, v in ep_stats.items():
            logger.info(f"\t{k}: {v:.4f}")
        ep_stats["epoch"] = ep
        train_stats.append(ep_stats)
        # Validation step can be added here if needed
        if ep % validate_every == 0 and val_loader:
            ep_stats = defaultdict(float)
            with torch.no_grad():
                for batch in val_loader:
                    net_in, labels, masks = batch
                    loss, stats = model.train_loss(net_in, labels, masks)
                    for k, v in stats.items():
                        ep_stats[k] += v / len(val_loader)
            logger.info("Validation Statistics")
            for k, v in ep_stats.items():
                logger.info(f"\t{k}: {v:.4f}")
            ep_stats["epoch"] = ep
            val_stats.append(ep_stats)
        logger.info("=" * 50)
    # Save off the model.
    torch.save(model.state_dict(), save_path)
    return train_stats, val_stats


def plot_training(
    train_stats: list[dict[str, float]],
    val_stats: list[dict[str, float]],
    plot_key: str = "loss",
):
    """
    Plot the statistics over time.

    Args:
        train_stats: The training statistics.
        val_stats: The validation statistics.
        plot_key: The key to plot.
    """
    valid_train_stats = [s for s in train_stats if plot_key in s]
    valid_val_stats = [s for s in val_stats if plot_key in s]
    if len(valid_train_stats):
        train_x = [s["epoch"] for s in valid_train_stats]
        train_y = [s[plot_key] for s in valid_train_stats]
        plt.plot(train_x, train_y, label="train")
    if len(valid_val_stats):
        val_x = [s["epoch"] for s in valid_val_stats]
        val_y = [s[plot_key] for s in valid_val_stats]
        plt.plot(val_x, val_y, label="val")
    plt.show()
