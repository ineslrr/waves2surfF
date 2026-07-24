#!/usr/bin/env python3
"""Calculate training-only input and target normalization statistics."""

import argparse
import json

from ocean_velocity.config import load_config
from ocean_velocity.statistics import calculate_statistics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    print(
        json.dumps(
            calculate_statistics(
                load_config(args.config), args.output, args.batch_size
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
