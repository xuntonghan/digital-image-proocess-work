from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove optimizer state from a training checkpoint"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source = Path(args.input)
    output = Path(args.output)
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    required = ("model", "model_config")
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise KeyError(f"Checkpoint is missing required fields: {missing}")

    keep = (
        "model",
        "model_config",
        "experiment_config",
        "global_step",
        "best_loss",
    )
    payload = {key: checkpoint[key] for key in keep if key in checkpoint}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output)
    print(f"Exported {source} -> {output} ({output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
