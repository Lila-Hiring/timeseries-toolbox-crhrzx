"""
Simple MLP
"""
import torch.nn as nn


class MLP(nn.Module):
    """
    Simple MLP model for time series forecasting.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        network_depth: int,
    ):
        super(MLP, self).__init__()
        layers = []
        curr_dim = input_dim
        for _ in range(network_depth):
            layers.append(nn.Linear(curr_dim, hidden_dim))
            layers.append(nn.ReLU())
            curr_dim = hidden_dim
        layers.append(nn.Linear(curr_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the MLP model.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, input_dim).

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, output_dim).
        """
        return self.model(x)
