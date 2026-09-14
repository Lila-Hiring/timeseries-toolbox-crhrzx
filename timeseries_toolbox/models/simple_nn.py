
import logging

import torch
from torch import Tensor
from torch.optim import Optimizer

from timeseries_toolbox.models.base_timeseries import BaseTimeSeriesModel
from timeseries_toolbox.torch.modules.mlp import MLP


logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more details
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class SimpleNN(BaseTimeSeriesModel):

    def __init__(
        self,
        data_dim: int,
        hidden_dim: int,
        network_depth: int,
        in_mean: Tensor | None,
        in_std: Tensor | None,
        out_mean: Tensor | None,
        out_std: Tensor | None,
        learning_rate: float = 3e-4,
    ):
        super(SimpleNN, self).__init__(
            data_dim=data_dim,
            in_mean=in_mean,
            in_std=in_std,
            out_mean=out_mean,
            out_std=out_std,
        )
        self.learning_rate = learning_rate
        self.model = MLP(
            input_dim=data_dim,
            output_dim=data_dim,
            hidden_dim=hidden_dim,
            network_depth=network_depth,
        )

    def forward(self, x):
        return self.model(x)

    @property
    def optimizer(self) -> Optimizer:
        """
        The optimizer to use for training the model.
        """
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def train_loss(
        self,
        net_in: Tensor,
        labels: Tensor,
        masks: Tensor,
    ) -> tuple[Tensor, dict[str, float]]:
        """
        Compute the training loss for the model.

        Args:
            net_in (batch_size, seq_len, num_features): The sequence of time
                series features that are fed into the model and used for the
                next time step prediction.
            labels (batch_size, seq_len, num_features): The labels for the
                corresponding time series features.
            masks (batch_size, seq_len): The masks which are used to indicate
                which time steps are real.

        Returns:
            * loss (Tensor): The training loss.
            * metrics (dict): A dictionary containing the training metrics.
                These metrics should be somewhat cheap to compute.
        """
        self.model.train()
        net_in = net_in.view(-1, self.input_dim)
        labels = labels.view(-1, self.input_dim)
        masks = masks.view(-1, 1)
        preds = self.model(net_in)
        loss = ((preds - labels).square() * masks).sum() / masks.sum()
        return loss, {
            "loss": loss.item(),
            "mse": loss.item(),
        }

    def evaluation_metrics(
        self,
        net_in: Tensor,
        labels: Tensor,
        masks: Tensor,
    ) -> dict[str, float]:
        """
        Compute the evaluations loss for the model. These metrics can be
        expensive to compute. We want every metric indicative of performance
        to be returned from this method.

        Args:
            net_in (batch_size, seq_len, num_features): The sequence of time
                series features that are fed into the model and used for the
                next time step prediction.
            labels (batch_size, seq_len, num_features): The labels for the
                corresponding time series features.
            masks (batch_size, seq_len): The masks which are used to indicate
                which time steps are real.

        Returns: Evaluation metrics.
        """
        self.model.eval()
        net_in = net_in.view(-1, self.input_dim)
        labels = labels.view(-1, self.input_dim)
        masks = masks.view(-1, 1)
        with torch.no_grad():
            preds = self.model(net_in)
        mse = ((preds - labels).square() * masks).sum() / masks.sum()
        rsquared = 1 - (
            ((preds - labels).square() * masks).sum()
            / ((labels - labels.mean(dim=0)).square() * masks).sum()
        )
        return {
            "loss": mse.item(),
            "mse": mse.item(),
            "R2": rsquared.item(),
        }

    def forecast(
        self,
        observations: Tensor,
        n_steps: int,
        deterministic: bool = True,
    ) -> Tensor:
        """
        Do a forecast into the future.

        Args:
            observations (seq_len, num_features): The observations seen
                thusfar for the time series we would like to forecast. The
                observations here should be not yet be normalized.
            n_steps (int): The number of steps to forecast into the future.
            deterministic (bool): Whether to do a stochastic sample for the
                future forecast or not.

        Returns: Tensor of shape (seq_len + n_steps, num_features) containing
            the original observations plus the forecasted values.
        """
        if not deterministic:
            logger.warning(
                "Stochastic forecasting is not supported for this model. "
                "Using deterministic forecasting instead."
            )
        self.model.eval()
        for _ in range(n_steps):
            net_in = self.normalize_input(observations[-1].unsqueeze(0))
            with torch.no_grad():
                pred = self.model(net_in)
            # We always assume that the model is predicting the delta.
            nxt = (
                observations[-1].unsqueeze(0)
                + self.denormalize_output(pred)
            )
            observations = torch.cat((observations, nxt), dim=0)
        return observations
