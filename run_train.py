
import argparse
from pathlib import Path

from timeseries_toolbox.data_utils import prepare_data
from timeseries_toolbox.models.simple_nn import SimpleNN
from timeseries_toolbox.train import train_model, plot_training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
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
        "--num_epochs",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--plot",
        action="store_true",
    )
    args = parser.parse_args()
    if args.data == "complex":
        train, val, normalization = prepare_data(
            data_dir=Path("data/complex/train"),
            should_normalize=True,
            train_proportion=0.9,
        )
    elif args.data == "linear":
        train, val, normalization = prepare_data(
            data_dir=Path("data/linear/train"),
            should_normalize=True,
            train_proportion=0.9,
        )
    else:
        raise ValueError(f"Unknown data type: {args.data}")
    if args.model == "simple_nn":
        model = SimpleNN(
            data_dim=train.dataset[0][0].shape[-1],
            hidden_dim=64,
            network_depth=2,
            in_mean=normalization["x_mean"],
            in_std=normalization["x_std"],
            out_mean=normalization["y_mean"],
            out_std=normalization["y_std"],
        )
    else:
        raise ValueError(f"Unknown model type: {args.model}")
    train_stats, val_stats = train_model(
        model=model,
        train_loader=train,
        val_loader=val,
        epochs=args.num_epochs,
        save_path=args.save_path,
        validate_every=10,
    )
    if args.plot:
        plot_training(
            train_stats=train_stats,
            val_stats=val_stats,
        )


if __name__ == "__main__":
    main()
