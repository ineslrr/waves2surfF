#!/usr/bin/env python3
"""Small command-line entrypoint for the full training pipeline.

Most logic lives in the importable ``ocean_velocity`` package so it can be
tested or reused from notebooks. This script only translates command-line
arguments into a validated configuration and an experiment output directory.

Typical usage::

    python train.py --config configs/example.json --output-dir runs/baseline
"""

import argparse

from ocean_velocity.config import load_config
from ocean_velocity.train import run_training


def main() -> None:
    """Parse user inputs, validate the config, and start training."""
    parser = argparse.ArgumentParser(
        description="Train a U-Net to predict 2-D surface velocity fields."
    )
    parser.add_argument("--config", required=True, help="Path to a JSON configuration")
    parser.add_argument("--output-dir", required=True, help="Experiment output directory")
    args = parser.parse_args()
    # Configuration validation occurs before data files or accelerators are
    # initialized, producing faster and clearer errors for common mistakes.
    run_training(load_config(args.config), args.output_dir)


if __name__ == "__main__":
    # This guard prevents training from starting when another script merely
    # imports this module.
    main()
