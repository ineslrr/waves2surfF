# Waves2surfF

A small, portable PyTorch project for learning

```text
2-D ocean fields + metadata → surface u, v
```

The model is called **Waves2SurfNet**. It uses a four-level U-Net backbone
derived from GOFLOW, reduced to base width 8 and generalized to arbitrary named
input channels and metadata conditioning.

## Data

The baseline reader expects one NetCDF file with aligned variables. Inputs may
be:

- Time-varying fields shaped `[time, y, x]`, such as SSH or wind components.
- Static fields shaped `[y, x]`, such as bathymetry or latitude.
- Scalar time series shaped `[time]`, such as domain-mean temperature.
- Cyclic calendar metadata derived from the time coordinate.

Targets are normally two variables shaped `[time, y, x]`: surface `u` and `v`.
An optional land/ocean mask and all non-finite values are excluded from loss
and metrics.

Edit [configs/example.json](configs/example.json) with the real variable names,
training-only normalization statistics, time indices, grid spacing, and HPC
settings.

Calculate statistics using only the configured training indices:

```bash
python calculate_statistics.py \
  --config configs/example.json \
  --output normalization.json
```

Copy the resulting `input`, `metadata`, and `target` objects into the configuration's
`data.normalization` section.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

The code has no scheduler- or cluster-specific dependency. Launch it through
the local HPC scheduler as an ordinary Python command.

## Training

```bash
python train.py \
  --config configs/example.json \
  --output-dir runs/baseline
```

For an interactive, step-by-step introduction, open
[notebooks/tutorial_data_and_training.ipynb](notebooks/tutorial_data_and_training.ipynb).
It constructs datasets, splits, DataLoaders, the model, optimizer, and
train/validation loops explicitly so each piece can be inspected or modified.

The run directory contains:

```text
config.json     exact configuration
history.json    per-epoch training/validation metrics
best.pt         checkpoint selected by validation loss
last.pt         final epoch
```

Evaluate the selected checkpoint on the untouched test split:

```bash
python evaluate.py --checkpoint runs/baseline/best.pt --split test
```

Start with pointwise loss only. Once that baseline is stable, an optional
spectral or velocity-gradient term can be enabled in `loss_weights`.
For a GOFLOW-style second stage, set `training.initialize_from` to the first
run's `best.pt`, change the loss weights, and use a new output directory.

## Configuration notes

- Input normalization statistics include calendar channels in the same order
  as `input_variables`. Metadata statistics are ordered as
  `metadata_variables` followed by `calendar_features`.
- Statistics must be computed from training samples only.
- `year_sin`, `year_cos`, `day_sin`, and `day_cos` are supported calendar
  features.
- `dx` and `dy` are only used by gradient loss and must use the same physical
  units.
- Splits can be JSON index lists or paths to `.npy` arrays.
- Use time-separated train, validation, and test periods. Spatially held-out
  tests should be an additional experiment.

## Metadata conditioning experiments

Global scalar/vector metadata is listed separately under
`data.metadata_variables`; calendar features are appended to that vector.
`model.conditioning` selects one of four comparable methods:

| Value | Behavior |
|---|---|
| `none` | Ignore metadata; spatial-field baseline |
| `extra_channels` | Broadcast every normalized metadata value as an input channel |
| `broadcast` | Compress metadata with an MLP to `metadata_channels` (default 4), then broadcast |
| `film` | Use an MLP to generate residual feature-wise scales and offsets throughout the U-Net |

All four modes use the same configurable normalization. The recommended
`normalization: "group"` applies GroupNorm in every convolution block and is
stable for small per-device batches. `batch` and `none` are also available.

For controlled comparisons, keep the split, seed, base width, optimizer, and
loss fixed while changing only `model.conditioning`. The broadcast and FiLM
MLPs use `metadata_width` as their hidden width. FiLM generators are
zero-initialized and apply:

```text
features * (1 + gamma(metadata)) + beta(metadata)
```

Spatial fields such as wind maps remain in `input_variables`; only global
values such as mean wind, calendar encodings, or mean temperature belong in
`metadata_variables`.

## Source layout

```text
ocean_velocity/
  model.py       Waves2SurfNet architecture and U-Net building blocks
  data.py        worker-safe NetCDF dataset
  losses.py      masked, spectral, and derivative losses
  metrics.py     streaming regression metrics
  config.py      JSON configuration validation
  train.py       training and validation loop
  statistics.py  streaming training-set statistics
train.py         command-line entrypoint
calculate_statistics.py
evaluate.py
```

See [PROJECT_MEMORY.md](PROJECT_MEMORY.md) for the GOFLOW review, retained
design choices, experimental recommendations, and known follow-up work.
