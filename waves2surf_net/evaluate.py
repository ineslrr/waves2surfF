#!/usr/bin/env python3
"""Evaluate a saved checkpoint on a configured split."""

import argparse
import json

import torch
from torch.utils.data import DataLoader

from ocean_velocity.train import evaluate, make_dataset, make_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = make_model(config).to(device)
    model.load_state_dict(checkpoint["model"])
    dataset = make_dataset(config, args.split)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    print(json.dumps(evaluate(model, loader, device, config), indent=2))


if __name__ == "__main__":
    main()
