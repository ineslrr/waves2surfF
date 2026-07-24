# Project memory

## Objective

Train a lightweight 2-D U-Net to map aligned snapshots of sea-surface height
and optional auxiliary fields (for example wind, temperature, bathymetry, and
position) to two surface-velocity components. Calendar and other scalar
metadata should also be supported.

## Findings from the original GOFLOW repository

The final GOFLOW model was the classical U-Net represented by
`lgt_unet16_1_3_0.2cs.pth`:

- Three SST-gradient frames in and `(u, v)` out.
- Base width 16, four down/up levels, skip concatenation, SiLU activations.
- Strided-convolution downsampling and interpolated upsampling.
- About 1.08 million parameters.
- Input batch normalization was enabled by the training factory.
- Training used AdamW at `1e-3` and cosine scheduling.
- Stage 0 used L1 loss. Stage 1 loaded Stage 0 and used
  `0.8 L1 + 0.2 spectral loss`.
- The spectral loss compared log 2-D kinetic-energy spectra after tapering.
- Checkpoints were selected using vorticity/strain R2 on what was called the
  test set.

Useful concepts retained here are the U-Net topology, direct two-component
output, masked pointwise loss, and optional spectral/derivative objectives.

GOFLOW-specific pieces intentionally removed include GOES inference,
satellite preprocessing, fixed spatial boxes, fixed SST-gradient
normalization, legacy architectures, NetCDF prediction writers, large images,
and the pretrained checkpoint.

## Current design decisions

- The public architecture name is `Waves2SurfNet`; it retains a U-Net backbone
  and keeps `UNet` as a compatibility alias for early notebooks.
- `base_channels=8` is the lightweight default.
- Group normalization replaces input/batch normalization, making training
  less dependent on per-device batch size.
- Every spatial physical input is a named channel. Static 2-D fields are
  broadcast automatically when needed.
- Global scalar/vector metadata is kept separate from spatial inputs.
  Calendar encodings are cyclic `sin/cos` metadata derived from NetCDF time.
- Four conditioning experiments share one GroupNorm U-Net: no conditioning,
  raw metadata channels, an MLP-compressed four-channel broadcast, and FiLM.
- FiLM generators are zero-initialized and use residual scales, preserving the
  unconditioned network at initialization.
- Input and target normalization statistics are explicit configuration,
  calculated on training data only, and stored with each checkpoint.
- Missing values and an optional ocean mask feed a correctly normalized
  masked loss.
- Train, validation, and test indices are explicit. Validation selects the
  checkpoint; test data should not be used during training.
- NetCDF handles open lazily per DataLoader worker.
- Checkpoints contain model state, configuration, epoch, and (for the best
  checkpoint) optimizer state and validation loss.
- A second-stage experiment can initialize model weights from an earlier
  `best.pt` using `training.initialize_from`; optimizer state is intentionally
  restarted for the new objective.

## Data contract

The baseline expects one NetCDF file containing aligned variables. Spatial
variables may be `[time, y, x]` or static `[y, x]`; scalar metadata may be
`[time]`. Targets are two `[time, y, x]` variables.

Each dataset sample is:

```text
x          float32 [C_in, H, W]
metadata   float32 [D]
y          float32 [2, H, W]
valid_mask bool    [1, H, W]
index      int64
```

Configuration defines variable names, calendar channels, normalization,
splits, grid spacing, model width, and training hyperparameters.

## Recommended experiment sequence

1. Establish an `Nbase=8` pointwise L1 baseline.
2. Compare with `Nbase=16` only if the small model underfits.
3. Evaluate component RMSE/MAE/R2, vector error, speed/direction, and spectra
   in physical units.
4. Add a small spectral weight (around 0.05--0.1) only after the baseline is
   stable.
5. Add derivative loss only when `dx` and `dy` are physically correct. The
   present implementation assumes constant Cartesian spacing.
6. Compare against geostrophic velocity inferred directly from SSH.
7. Test temporal generalization and, separately, held-out spatial regions.

## Known follow-up work

- Decide the actual NetCDF variable names, units, grid geometry, and split
  periods.
- Run `calculate_statistics.py` on the finalized training indices and copy its
  output into the experiment configuration.
- Add an inference/export command once the desired output format is known.
- Add distributed training only after the single-device pipeline is validated.
- For latitude/longitude or curvilinear grids, replace constant `dx/dy`
  derivatives with metric-aware operators.
- Spatial augmentation of vector fields must rotate/reflect vector components
  consistently; generic image flips are physically wrong.
