
from abc import ABC, abstractmethod

from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer


class BaseTimeSeriesModel(Module, ABC):
    """
    Base class for time series models.
    """

    def __init__(
        self,
        data_dim: int,
        in_mean: Tensor | None,
        in_std: Tensor | None,
        out_mean: Tensor | None,
        out_std: Tensor | None,
    ):
        super(BaseTimeSeriesModel, self).__init__()
        self._data_dim = data_dim
        if in_mean is None:
            in_mean = Tensor([0.0] * self._data_dim)
        if in_std is None:
            in_std = Tensor([1.0] * self._data_dim)
        if out_mean is None:
            out_mean = Tensor([0.0] * self._data_dim)
        if out_std is None:
            out_std = Tensor([1.0] * self._data_dim)
        self.register_buffer("in_mean", in_mean)
        self.register_buffer("in_std", in_std)
        self.register_buffer("out_mean", out_mean)
        self.register_buffer("out_std", out_std)

    @property
    def input_dim(self) -> int:
        """
        The number of features in the input time series.
        """
        return self._data_dim

    @property
    def ouptut_dim(self) -> int:
        """
        The number of features in the input time series.
        """
        return self.input_dim

    @property
    @abstractmethod
    def optimizer(self) -> Optimizer:
        """
        The optimizer to use for training the model.
        """


    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
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
                thusfar for the time series we would like to forecast.
            n_steps (int): The number of steps to forecast into the future.
            deterministic (bool): Whether to do a stochastic sample for the
                future forecast or not.

        Returns: Tensor of shape (seq_len + n_steps, num_features) containing
            the original observations plus the forecasted values.
        """

    def normalize_input(self, x: Tensor) -> Tensor:
        """
        Normalize the input data.

        Args:
            x (Tensor): The input data to normalize.

        Returns: Normalized input data.
        """
        return (x - self.in_mean) / self.in_std

    def normalize_output(self, x: Tensor) -> Tensor:
        """
        Normalize the output data.
        Args:
            x (Tensor): The output data to normalize.
        Returns: Normalized output data.
        """
        return (x - self.out_mean) / self.out_std

    def denormalize_input(self, x: Tensor) -> Tensor:
        """
        Denormalize the input data.
        Args:
            x (Tensor): The input data to denormalize.
        Returns: Denormalized input data.
        """
        return x * self.in_std + self.in_mean

    def denormalize_output(self, x: Tensor) -> Tensor:
        """
        Denormalize the output data.
        Args:
            x (Tensor): The output data to denormalize.
        Returns: Denormalized output data.
        """
        return x * self.out_std + self.out_mean
