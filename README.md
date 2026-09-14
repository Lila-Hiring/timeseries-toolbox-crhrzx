# Time Series Toolbox

A python library for time series modeling. This library contains deep learning
models that can be used to make future forecasts.

## Installation

This library uses `uv` for package management. To install the library, run:

```bash
uv sync
```

## Usage

To train a model...

```bash
uv run run_train.py \
    --save_path <path_to_save> \
    --model simple_nn \
    --data linear \
    --plot
```

After the model has been trained, you can visualize a forecast with the model
with the following command

```bash
uv run visualize_forecast.py \
    --model_path <path_to_model> \
    --model simple_nn \
    --data linear
```
