#!/usr/bin/env python3
"""Command-line entrypoint for model training."""

import argparse

from ocean_velocity.config import load_config
from ocean_velocity.train import run_training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to a JSON configuration")
    parser.add_argument("--output-dir", required=True, help="Experiment output directory")
    args = parser.parse_args()
    run_training(load_config(args.config), args.output_dir)


if __name__ == "__main__":
    main()
