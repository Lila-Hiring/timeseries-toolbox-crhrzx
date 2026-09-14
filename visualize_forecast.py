import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from timeseries_toolbox.data_utils import prepare_data
from timeseries_toolbox.models.simple_nn import SimpleNN


###########################################################################
# Parse the arguments.
###########################################################################
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_path",
    type=str,
    required=True,
)
parser.add_argument(
    "--plot_path",
    type=str,
)
parser.add_argument(
    "--model",
    type=str,
    choices=["simple_nn"],
)
parser.add_argument(
    "--data",
    type=str,
    choices=["complex", "linear"],
)
parser.add_argument(
    "--samples",
    type=int,
    default=1,
)
parser.add_argument(
    "--plot_uq",
    action="store_true",
)
parser.add_argument(
    "--num_init_obs",
    type=int,
    default=10,
)
args = parser.parse_args()
if args.plot_uq:
    assert args.samples >= 30, "For UQ, at least 30 samples are required."


###########################################################################
# Load in the data.
###########################################################################
if args.data == "complex":
    train, _, _ = prepare_data(
        data_dir=Path("data/complex/train"),
        should_normalize=False,
        train_proportion=1.0,
    )
elif args.data == "linear":
    train, _, _ = prepare_data(
        data_dir=Path("data/linear/train"),
        should_normalize=False,
        train_proportion=1.0,
    )
else:
    raise ValueError(f"Unknown data type: {args.data}")
if args.model == "simple_nn":
    model = SimpleNN(
        data_dim=train.dataset[0][0].shape[-1],
        hidden_dim=64,
        network_depth=2,
        in_mean=None,
        in_std=None,
        out_mean=None,
        out_std=None,
    )
else:
    raise ValueError(f"Unknown model type: {args.model}")
model.load_state_dict(
    torch.load(args.model_path, map_location="cpu"),
)

###########################################################################
# Make predictions.
###########################################################################
preds = []
full_obs = train.dataset[0][0]
obs = full_obs[:args.num_init_obs]
num_steps = train.dataset[0][0].shape[0] - args.num_init_obs
for _ in tqdm(range(args.samples), desc="Generating samples..."):
    pred = model.forecast(
        observations=obs,
        n_steps=num_steps,
        deterministic=args.samples == 1,
    )
    preds.append(pred)
preds = torch.stack(preds)

###########################################################################
# Make a plot of the predictions.
###########################################################################
steps = np.arange(len(full_obs))
_, axs = plt.subplots(2, 2)
for i in range(full_obs.shape[1]):
    axs[i // 2, i % 2].plot(
        steps,
        full_obs[:, i],
        label="Real",
        color="black",
    )
    axs[i // 2, i % 2].plot(
        steps,
        preds.mean(dim=0)[:, i],
        label="Forecast",
        color="cornflowerblue",
        alpha=0.6,
    )
    axs[i // 2, i % 2].set_title(f"Feature {i}")
    axs[i // 2, i % 2].legend()
plt.show()
if args.plot_path:
    plt.savefig(f'{args.plot_path}/predictions.png', dpi=300, bbox_inches='tight')

